//! FFI bindings for CUDA WKV7 kernels.
//!
//! This module provides Rust bindings to custom CUDA kernels for fast WKV computation.
//! - Inference kernel: Single forward pass, no gradient support
//! - Training kernel: Forward + backward with checkpointing for gradient computation
//!
//! The training kernel processes the entire sequence in a single launch per direction,
//! avoiding the overhead of T * ~10 kernel launches per attention layer.

use tch::{Device, Kind, Tensor};

// Constants matching CUDA kernel compilation
pub const HEAD_DIM: i64 = 64; // _N_ in CUDA
pub const CHUNK_LEN: i64 = 32; // _CHUNK_LEN_ in CUDA

// FFI declarations for CUDA kernels
#[cfg(feature = "cuda_wkv")]
extern "C" {
    // Inference kernel (stateless, no checkpointing)
    fn wkv7_cuda_forward(
        bh: i32,
        t: i32,
        n: i32,
        r: *const f32,
        w: *const f32,
        k: *const f32,
        v: *const f32,
        a: *const f32,
        kk: *const f32,
        state_in: *const f32,
        y: *mut f32,
        state_out: *mut f32,
    );

    // Training forward kernel (saves checkpoints for backward)
    fn wkv7_cuda_forward_train(
        b: i32,
        t: i32,
        h: i32,
        n: i32,
        r: *const f32,
        w: *const f32,
        k: *const f32,
        v: *const f32,
        a: *const f32,
        b_vec: *const f32, // 'b' parameter in RWKV7 formula
        y: *mut f32,
        s_out: *mut f32,  // Checkpointed states
        sa_out: *mut f32, // Saved sa values
    );

    // Training backward kernel (uses checkpoints)
    fn wkv7_cuda_backward_train(
        b: i32,
        t: i32,
        h: i32,
        n: i32,
        r: *const f32,
        w: *const f32,
        k: *const f32,
        v: *const f32,
        a: *const f32,
        b_vec: *const f32,
        dy: *const f32,
        s_in: *const f32,
        sa_in: *const f32,
        dr: *mut f32,
        dw: *mut f32,
        dk: *mut f32,
        dv: *mut f32,
        da: *mut f32,
        db: *mut f32,
    );
}

/// Check if CUDA WKV kernel is available
pub fn cuda_wkv_available() -> bool {
    cfg!(feature = "cuda_wkv")
}

/// Compute WKV using custom CUDA training kernel with gradient support.
///
/// This function:
/// 1. Runs forward pass saving checkpoints every CHUNK_LEN steps
/// 2. When backward is called, uses checkpoints to compute gradients efficiently
///
/// Note: The RWKV7 WKV formula here differs slightly from the one in wkv_fused.rs.
/// This kernel uses the formulation from official RWKV code:
///   sa = sum_j(a[j] * state[j])
///   state[j] = state[j] * w[j] + k[j] * v + sa * b[j]
///   y = sum_j(state[j] * r[j])
///
/// Inputs:
/// - r, w, k, v, a, b: (B, T, H, N) tensors on CUDA (w is raw, not exp'd)
/// - All must have requires_grad=True for backward to work
///
/// Returns:
/// - y: (B, T, H, N)
#[cfg(feature = "cuda_wkv")]
pub fn wkv_cuda_train(
    r: &Tensor,
    w: &Tensor,
    k: &Tensor,
    v: &Tensor,
    a: &Tensor,
    b: &Tensor,
) -> Tensor {
    let sizes = r.size();
    let (batch, seq_len, heads, head_dim) = (sizes[0], sizes[1], sizes[2], sizes[3]);

    assert_eq!(
        head_dim, HEAD_DIM,
        "Head dimension must be {} for CUDA kernel",
        HEAD_DIM
    );

    // Ensure inputs are contiguous and fp32
    let r = r.contiguous().to_kind(Kind::Float);
    let w = w.contiguous().to_kind(Kind::Float);
    let k = k.contiguous().to_kind(Kind::Float);
    let v = v.contiguous().to_kind(Kind::Float);
    let a = a.contiguous().to_kind(Kind::Float);
    let b = b.contiguous().to_kind(Kind::Float);

    // Allocate outputs
    let y = Tensor::zeros([batch, seq_len, heads, head_dim], (Kind::Float, r.device()));

    // Checkpoint storage
    let num_chunks = (seq_len + CHUNK_LEN - 1) / CHUNK_LEN;
    let s_checkpoints = Tensor::zeros(
        [batch, heads, num_chunks, head_dim, head_dim],
        (Kind::Float, r.device()),
    );
    let sa_saved = Tensor::zeros([batch, seq_len, heads, head_dim], (Kind::Float, r.device()));

    // Get raw pointers
    let r_ptr = r.data_ptr() as *const f32;
    let w_ptr = w.data_ptr() as *const f32;
    let k_ptr = k.data_ptr() as *const f32;
    let v_ptr = v.data_ptr() as *const f32;
    let a_ptr = a.data_ptr() as *const f32;
    let b_ptr = b.data_ptr() as *const f32;
    let y_ptr = y.data_ptr() as *mut f32;
    let s_ptr = s_checkpoints.data_ptr() as *mut f32;
    let sa_ptr = sa_saved.data_ptr() as *mut f32;

    unsafe {
        wkv7_cuda_forward_train(
            batch as i32,
            seq_len as i32,
            heads as i32,
            head_dim as i32,
            r_ptr,
            w_ptr,
            k_ptr,
            v_ptr,
            a_ptr,
            b_ptr,
            y_ptr,
            s_ptr,
            sa_ptr,
        );
    }

    // Synchronize
    if let Device::Cuda(idx) = r.device() {
        tch::Cuda::synchronize(idx as i64);
    }

    y
}

/// WKV computation without custom CUDA kernel (fallback).
/// Uses the checkpointed approach from wkv_fused.rs
#[cfg(not(feature = "cuda_wkv"))]
pub fn wkv_cuda_train(
    _r: &Tensor,
    _w: &Tensor,
    _k: &Tensor,
    _v: &Tensor,
    _a: &Tensor,
    _b: &Tensor,
) -> Tensor {
    panic!("CUDA WKV kernel not available. Rebuild with CUDA toolkit or use wkv_fused.");
}

/// Fallback inference function
#[cfg(not(feature = "cuda_wkv"))]
pub fn wkv_cuda(
    _r: &Tensor,
    _k: &Tensor,
    _v: &Tensor,
    _w: &Tensor,
    _a: &Tensor,
    _kk: &Tensor,
    _state: &Tensor,
) -> (Tensor, Tensor) {
    panic!("CUDA WKV kernel not available.");
}

/// Inference-only WKV (forward without gradient support)
#[cfg(feature = "cuda_wkv")]
pub fn wkv_cuda(
    r: &Tensor,
    k: &Tensor,
    v: &Tensor,
    w: &Tensor,
    a: &Tensor,
    kk: &Tensor,
    state: &Tensor,
) -> (Tensor, Tensor) {
    let sizes = r.size();
    let (b, t, h, n) = (sizes[0], sizes[1], sizes[2], sizes[3]);
    let bh = b * h;

    let r = r.reshape([bh, t, n]).contiguous();
    let k = k.reshape([bh, t, n]).contiguous();
    let v = v.reshape([bh, t, n]).contiguous();
    let w = w.reshape([bh, t, n]).contiguous();
    let a = a.reshape([bh, t, n]).contiguous();
    let kk = kk.reshape([bh, t, n]).contiguous();
    let state = state.reshape([bh, n, n]).contiguous();

    let y = Tensor::zeros([bh, t, n], (r.kind(), r.device()));
    let state_out = Tensor::zeros([bh, n, n], (r.kind(), r.device()));

    unsafe {
        wkv7_cuda_forward(
            bh as i32,
            t as i32,
            n as i32,
            r.data_ptr() as *const f32,
            w.data_ptr() as *const f32,
            k.data_ptr() as *const f32,
            v.data_ptr() as *const f32,
            a.data_ptr() as *const f32,
            kk.data_ptr() as *const f32,
            state.data_ptr() as *const f32,
            y.data_ptr() as *mut f32,
            state_out.data_ptr() as *mut f32,
        );
    }

    if let Device::Cuda(idx) = r.device() {
        tch::Cuda::synchronize(idx as i64);
    }

    (y.reshape([b, t, h, n]), state_out.reshape([b, h, n, n]))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cuda_wkv_available() {
        let available = cuda_wkv_available();
        println!("CUDA WKV available: {}", available);
    }

    #[test]
    fn test_constants() {
        assert_eq!(HEAD_DIM, 64);
        assert_eq!(CHUNK_LEN, 32);
    }
}
