//! Hybrid CPU/GPU WKV computation.
//!
//! Strategy:
//! - All linear projections (matmuls) stay on GPU - these parallelize well
//! - The sequential WKV recurrence runs on CPU - faster for sequential ops
//! - Data transfer is minimized by batching transfers
//!
//! This approach works because:
//! 1. WKV is memory-bound, not compute-bound (small NxN state)
//! 2. CPU has better cache locality for sequential access patterns
//! 3. GPU kernel launch overhead dominates for small operations

use anyhow::Result;
use tch::{Device, Kind, Tensor};

/// Run WKV recurrence on CPU with efficient memory layout.
///
/// Input tensors should be on CPU, contiguous, and in (B, T, H, N) layout.
pub fn wkv_cpu(
    r: &Tensor,      // (B, T, H, N)
    k: &Tensor,
    v: &Tensor,
    w: &Tensor,      // decay factor (already exp(-exp(raw)))
    kk: &Tensor,     // normalized key for state
    a: &Tensor,
    state: &mut Tensor,  // (B, H, N, N)
) -> Result<Tensor> {
    debug_assert_eq!(r.device(), Device::Cpu);
    debug_assert_eq!(state.device(), Device::Cpu);
    
    let sizes = r.size();
    let b = sizes[0] as usize;
    let t = sizes[1] as usize;
    let h = sizes[2] as usize;
    let n = sizes[3] as usize;
    
    // Create output tensor
    let mut y = Tensor::zeros(&sizes, (Kind::Float, Device::Cpu));
    
    // Get mutable views into the data
    // We'll use raw pointer access for maximum speed
    let r_data = Vec::<f32>::try_from(r.flatten(0, -1))?;
    let k_data = Vec::<f32>::try_from(k.flatten(0, -1))?;
    let v_data = Vec::<f32>::try_from(v.flatten(0, -1))?;
    let w_data = Vec::<f32>::try_from(w.flatten(0, -1))?;
    let kk_data = Vec::<f32>::try_from(kk.flatten(0, -1))?;
    let a_data = Vec::<f32>::try_from(a.flatten(0, -1))?;
    
    let mut state_data = Vec::<f32>::try_from(state.flatten(0, -1))?;
    let mut y_data = vec![0.0f32; b * t * h * n];
    
    // Process each batch and head independently (parallelizable)
    for bi in 0..b {
        for hi in 0..h {
            // State for this batch/head: (N, N) stored row-major
            let state_offset = (bi * h + hi) * n * n;
            
            for ti in 0..t {
                // Input offset for this timestep
                let inp_offset = ((bi * t + ti) * h + hi) * n;
                
                // Load vectors for this timestep
                let r_t = &r_data[inp_offset..inp_offset + n];
                let k_t = &k_data[inp_offset..inp_offset + n];
                let v_t = &v_data[inp_offset..inp_offset + n];
                let w_t = &w_data[inp_offset..inp_offset + n];
                let kk_t = &kk_data[inp_offset..inp_offset + n];
                let a_t = &a_data[inp_offset..inp_offset + n];
                
                // Compute u = S @ kk (state @ kk for each row)
                let mut u = vec![0.0f32; n];
                for i in 0..n {
                    let mut sum = 0.0f32;
                    for j in 0..n {
                        sum += state_data[state_offset + i * n + j] * kk_t[j];
                    }
                    u[i] = sum;
                }
                
                // Update state: S = S * w - u @ (kk * a).T + v @ k.T
                // S[i,j] = S[i,j] * w[j] - u[i] * kk[j] * a[j] + v[i] * k[j]
                for i in 0..n {
                    for j in 0..n {
                        let idx = state_offset + i * n + j;
                        state_data[idx] = state_data[idx] * w_t[j] 
                                        - u[i] * kk_t[j] * a_t[j] 
                                        + v_t[i] * k_t[j];
                    }
                }
                
                // Compute y = S @ r
                for i in 0..n {
                    let mut sum = 0.0f32;
                    for j in 0..n {
                        sum += state_data[state_offset + i * n + j] * r_t[j];
                    }
                    y_data[inp_offset + i] = sum;
                }
            }
        }
    }
    
    // Write back results
    *state = Tensor::from_slice(&state_data)
        .reshape([b as i64, h as i64, n as i64, n as i64]);
    y = Tensor::from_slice(&y_data)
        .reshape([b as i64, t as i64, h as i64, n as i64]);
    
    Ok(y)
}

/// Parallel WKV using rayon for multi-threaded CPU execution.
#[cfg(feature = "rayon")]
pub fn wkv_cpu_parallel(
    r: &Tensor,
    k: &Tensor,
    v: &Tensor,
    w: &Tensor,
    kk: &Tensor,
    a: &Tensor,
    state: &mut Tensor,
) -> Result<Tensor> {
    use rayon::prelude::*;
    
    // Similar to wkv_cpu but parallelizes across batch/head dimensions
    todo!("Implement parallel version")
}

/// Hybrid forward pass: projections on GPU, WKV on CPU
pub struct HybridWKV {
    device: Device,
}

impl HybridWKV {
    pub fn new(device: Device) -> Self {
        Self { device }
    }
    
    /// Process WKV with hybrid CPU/GPU execution.
    /// 
    /// Inputs should be on GPU. They will be moved to CPU for WKV,
    /// then the result moved back to GPU.
    pub fn forward(
        &self,
        r: &Tensor,      // (B, T, H, N) on GPU
        k: &Tensor,
        v: &Tensor,
        w: &Tensor,
        kk: &Tensor,
        a: &Tensor,
        state: &mut Tensor,  // (B, H, N, N) on GPU
    ) -> Result<Tensor> {
        // Move to CPU (contiguous for efficient access)
        let r_cpu = r.to_device(Device::Cpu).contiguous();
        let k_cpu = k.to_device(Device::Cpu).contiguous();
        let v_cpu = v.to_device(Device::Cpu).contiguous();
        let w_cpu = w.to_device(Device::Cpu).contiguous();
        let kk_cpu = kk.to_device(Device::Cpu).contiguous();
        let a_cpu = a.to_device(Device::Cpu).contiguous();
        let mut state_cpu = state.to_device(Device::Cpu).contiguous();
        
        // Run WKV on CPU
        let y_cpu = wkv_cpu(&r_cpu, &k_cpu, &v_cpu, &w_cpu, &kk_cpu, &a_cpu, &mut state_cpu)?;
        
        // Move results back to GPU
        *state = state_cpu.to_device(self.device);
        Ok(y_cpu.to_device(self.device))
    }
}
