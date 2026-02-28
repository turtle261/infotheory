//! Portable SIMD kernels for RWKV7 operations.
//!
//! This module uses `wide` vectors so LLVM/rustc can select the best SIMD path
//! per target (x86_64, aarch64, wasm32-simd128) while preserving a scalar
//! fallback on targets without SIMD support.

#![allow(dead_code)]

use wide::f32x8;

const LANES: usize = 8;
const HEAD_DIM: usize = 64;
const HEAD_CHUNKS: usize = HEAD_DIM / LANES;

#[inline(always)]
unsafe fn load8(ptr: *const f32) -> f32x8 {
    ptr.cast::<f32x8>().read_unaligned()
}

#[inline(always)]
unsafe fn store8(ptr: *mut f32, v: f32x8) {
    ptr.cast::<f32x8>().write_unaligned(v)
}

#[inline(always)]
fn reduce_max(v: f32x8) -> f32 {
    let lanes = v.to_array();
    let mut m = f32::NEG_INFINITY;
    for &x in &lanes {
        m = m.max(x);
    }
    m
}

/// Dot product of two slices.
#[inline(always)]
pub unsafe fn dot_avx(a: *const f32, b: *const f32, len: usize) -> f32 {
    let mut sum = f32x8::ZERO;
    let mut i = 0;

    while i + LANES <= len {
        let av = load8(a.add(i));
        let bv = load8(b.add(i));
        sum += av * bv;
        i += LANES;
    }

    let mut out = sum.reduce_add();
    while i < len {
        out += *a.add(i) * *b.add(i);
        i += 1;
    }

    out
}

/// Matrix-vector multiply: y = A @ x where A is (rows, cols), x is (cols,), y is (rows,).
#[inline(always)]
pub unsafe fn gemv_avx(a: *const f32, x: *const f32, y: *mut f32, rows: usize, cols: usize) {
    let mut r = 0;

    while r + 4 <= rows {
        let row0 = a.add(r * cols);
        let row1 = a.add((r + 1) * cols);
        let row2 = a.add((r + 2) * cols);
        let row3 = a.add((r + 3) * cols);

        let mut sum0 = f32x8::ZERO;
        let mut sum1 = f32x8::ZERO;
        let mut sum2 = f32x8::ZERO;
        let mut sum3 = f32x8::ZERO;

        let mut c = 0;
        while c + LANES <= cols {
            let xv = load8(x.add(c));
            sum0 += load8(row0.add(c)) * xv;
            sum1 += load8(row1.add(c)) * xv;
            sum2 += load8(row2.add(c)) * xv;
            sum3 += load8(row3.add(c)) * xv;
            c += LANES;
        }

        let mut out0 = sum0.reduce_add();
        let mut out1 = sum1.reduce_add();
        let mut out2 = sum2.reduce_add();
        let mut out3 = sum3.reduce_add();

        while c < cols {
            let xv = *x.add(c);
            out0 += *row0.add(c) * xv;
            out1 += *row1.add(c) * xv;
            out2 += *row2.add(c) * xv;
            out3 += *row3.add(c) * xv;
            c += 1;
        }

        *y.add(r) = out0;
        *y.add(r + 1) = out1;
        *y.add(r + 2) = out2;
        *y.add(r + 3) = out3;

        r += 4;
    }

    while r < rows {
        *y.add(r) = dot_avx(a.add(r * cols), x, cols);
        r += 1;
    }
}

/// Matrix-vector multiply with transposed matrix: y = A^T @ x.
#[inline(always)]
pub unsafe fn gemv_t_avx(a: *const f32, x: *const f32, y: *mut f32, rows: usize, cols: usize) {
    let mut c = 0;
    while c + LANES <= cols {
        store8(y.add(c), f32x8::ZERO);
        c += LANES;
    }
    while c < cols {
        *y.add(c) = 0.0;
        c += 1;
    }

    for r in 0..rows {
        let x_r = f32x8::splat(*x.add(r));
        let row = a.add(r * cols);

        let mut c = 0;
        while c + LANES <= cols {
            let yv = load8(y.add(c));
            let av = load8(row.add(c));
            store8(y.add(c), yv + av * x_r);
            c += LANES;
        }

        while c < cols {
            *y.add(c) += *row.add(c) * *x.add(r);
            c += 1;
        }
    }
}

/// Element-wise multiply: y = a * b.
#[inline(always)]
pub unsafe fn mul_avx(a: *const f32, b: *const f32, y: *mut f32, len: usize) {
    let mut i = 0;
    while i + LANES <= len {
        store8(y.add(i), load8(a.add(i)) * load8(b.add(i)));
        i += LANES;
    }
    while i < len {
        *y.add(i) = *a.add(i) * *b.add(i);
        i += 1;
    }
}

/// Element-wise add: y = a + b.
#[inline(always)]
pub unsafe fn add_avx(a: *const f32, b: *const f32, y: *mut f32, len: usize) {
    let mut i = 0;
    while i + LANES <= len {
        store8(y.add(i), load8(a.add(i)) + load8(b.add(i)));
        i += LANES;
    }
    while i < len {
        *y.add(i) = *a.add(i) + *b.add(i);
        i += 1;
    }
}

/// Element-wise fused multiply-add: y = a * b + c.
#[inline(always)]
pub unsafe fn fma_avx(a: *const f32, b: *const f32, c: *const f32, y: *mut f32, len: usize) {
    let mut i = 0;
    while i + LANES <= len {
        let av = load8(a.add(i));
        let bv = load8(b.add(i));
        let cv = load8(c.add(i));
        store8(y.add(i), av.mul_add(bv, cv));
        i += LANES;
    }
    while i < len {
        *y.add(i) = *a.add(i) * *b.add(i) + *c.add(i);
        i += 1;
    }
}

/// Scaled add: y = y + scale * x.
#[inline(always)]
pub unsafe fn scaled_add_avx(y: *mut f32, x: *const f32, scale: f32, len: usize) {
    let scale_v = f32x8::splat(scale);
    let mut i = 0;
    while i + LANES <= len {
        let yv = load8(y.add(i));
        let xv = load8(x.add(i));
        store8(y.add(i), scale_v.mul_add(xv, yv));
        i += LANES;
    }
    while i < len {
        *y.add(i) += scale * *x.add(i);
        i += 1;
    }
}

/// Copy: dst = src.
#[inline(always)]
pub unsafe fn copy(src: *const f32, dst: *mut f32, len: usize) {
    std::ptr::copy_nonoverlapping(src, dst, len);
}

/// Token shift: out = x + mix * (prev - x).
#[inline(always)]
pub unsafe fn token_shift_avx(
    x: *const f32,
    prev: *const f32,
    mix: *const f32,
    out: *mut f32,
    len: usize,
) {
    let mut i = 0;
    while i + LANES <= len {
        let xv = load8(x.add(i));
        let pv = load8(prev.add(i));
        let mv = load8(mix.add(i));
        store8(out.add(i), mv.mul_add(pv - xv, xv));
        i += LANES;
    }
    while i < len {
        let xi = *x.add(i);
        *out.add(i) = xi + *mix.add(i) * (*prev.add(i) - xi);
        i += 1;
    }
}

/// Token shift for six projections sharing the same x/prev inputs.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
pub unsafe fn token_shift_multi6_avx(
    x: *const f32,
    prev: *const f32,
    mix0: *const f32,
    mix1: *const f32,
    mix2: *const f32,
    mix3: *const f32,
    mix4: *const f32,
    mix5: *const f32,
    out0: *mut f32,
    out1: *mut f32,
    out2: *mut f32,
    out3: *mut f32,
    out4: *mut f32,
    out5: *mut f32,
    len: usize,
) {
    let mut i = 0;
    while i + LANES <= len {
        let xv = load8(x.add(i));
        let diff = load8(prev.add(i)) - xv;

        store8(out0.add(i), load8(mix0.add(i)).mul_add(diff, xv));
        store8(out1.add(i), load8(mix1.add(i)).mul_add(diff, xv));
        store8(out2.add(i), load8(mix2.add(i)).mul_add(diff, xv));
        store8(out3.add(i), load8(mix3.add(i)).mul_add(diff, xv));
        store8(out4.add(i), load8(mix4.add(i)).mul_add(diff, xv));
        store8(out5.add(i), load8(mix5.add(i)).mul_add(diff, xv));

        i += LANES;
    }

    while i < len {
        let xi = *x.add(i);
        let d = *prev.add(i) - xi;

        *out0.add(i) = xi + *mix0.add(i) * d;
        *out1.add(i) = xi + *mix1.add(i) * d;
        *out2.add(i) = xi + *mix2.add(i) * d;
        *out3.add(i) = xi + *mix3.add(i) * d;
        *out4.add(i) = xi + *mix4.add(i) * d;
        *out5.add(i) = xi + *mix5.add(i) * d;

        i += 1;
    }
}

/// Layer normalization: y = (x - mean) / sqrt(var + eps) * weight + bias.
#[inline(always)]
pub unsafe fn layer_norm_avx(
    x: *const f32,
    weight: *const f32,
    bias: *const f32,
    y: *mut f32,
    len: usize,
    eps: f32,
) {
    let mut sum = f32x8::ZERO;
    let mut i = 0;

    while i + LANES <= len {
        sum += load8(x.add(i));
        i += LANES;
    }

    let mut mean = sum.reduce_add();
    while i < len {
        mean += *x.add(i);
        i += 1;
    }
    mean /= len as f32;

    let mean_v = f32x8::splat(mean);

    let mut var_sum = f32x8::ZERO;
    i = 0;
    while i + LANES <= len {
        let d = load8(x.add(i)) - mean_v;
        var_sum += d * d;
        i += LANES;
    }

    let mut var = var_sum.reduce_add();
    while i < len {
        let d = *x.add(i) - mean;
        var += d * d;
        i += 1;
    }
    var /= len as f32;

    let inv_std = 1.0 / (var + eps).sqrt();
    let inv_std_v = f32x8::splat(inv_std);

    i = 0;
    while i + LANES <= len {
        let xv = load8(x.add(i));
        let wv = load8(weight.add(i));
        let bv = load8(bias.add(i));
        let out = ((xv - mean_v) * inv_std_v).mul_add(wv, bv);
        store8(y.add(i), out);
        i += LANES;
    }

    while i < len {
        let normed = (*x.add(i) - mean) * inv_std;
        *y.add(i) = normed * *weight.add(i) + *bias.add(i);
        i += 1;
    }
}

/// Group normalization for RWKV7.
#[inline(always)]
pub unsafe fn group_norm_avx(
    x: *const f32,
    weight: *const f32,
    bias: *const f32,
    y: *mut f32,
    groups: usize,
    group_size: usize,
    eps: f32,
) {
    for g in 0..groups {
        let offset = g * group_size;
        let x_g = x.add(offset);
        let w_g = weight.add(offset);
        let b_g = bias.add(offset);
        let y_g = y.add(offset);

        let mut sum = f32x8::ZERO;
        let mut i = 0;

        while i + LANES <= group_size {
            sum += load8(x_g.add(i));
            i += LANES;
        }

        let mut mean = sum.reduce_add();
        while i < group_size {
            mean += *x_g.add(i);
            i += 1;
        }
        mean /= group_size as f32;

        let mean_v = f32x8::splat(mean);

        let mut var_sum = f32x8::ZERO;
        i = 0;
        while i + LANES <= group_size {
            let d = load8(x_g.add(i)) - mean_v;
            var_sum += d * d;
            i += LANES;
        }

        let mut var = var_sum.reduce_add();
        while i < group_size {
            let d = *x_g.add(i) - mean;
            var += d * d;
            i += 1;
        }
        var /= group_size as f32;

        let inv_std = 1.0 / (var + eps).sqrt();
        let inv_std_v = f32x8::splat(inv_std);

        i = 0;
        while i + LANES <= group_size {
            let xv = load8(x_g.add(i));
            let wv = load8(w_g.add(i));
            let bv = load8(b_g.add(i));
            let out = ((xv - mean_v) * inv_std_v).mul_add(wv, bv);
            store8(y_g.add(i), out);
            i += LANES;
        }

        while i < group_size {
            let normed = (*x_g.add(i) - mean) * inv_std;
            *y_g.add(i) = normed * *w_g.add(i) + *b_g.add(i);
            i += 1;
        }
    }
}

/// Sigmoid: y = 1 / (1 + exp(-x)).
#[inline(always)]
pub unsafe fn sigmoid_avx(x: *const f32, y: *mut f32, len: usize) {
    let ones = f32x8::ONE;

    let mut i = 0;
    while i + LANES <= len {
        let xv = load8(x.add(i));
        let out = ones / (ones + (-xv).exp());
        store8(y.add(i), out);
        i += LANES;
    }

    while i < len {
        let xv = *x.add(i);
        *y.add(i) = 1.0 / (1.0 + (-xv).exp());
        i += 1;
    }
}

/// Tanh: y = tanh(x).
#[inline(always)]
pub unsafe fn tanh_avx(x: *const f32, y: *mut f32, len: usize) {
    let ones = f32x8::ONE;
    let two = f32x8::splat(2.0);

    let mut i = 0;
    while i + LANES <= len {
        let xv = load8(x.add(i));
        let exp_neg_2x = (-xv * two).exp();
        let out = (ones - exp_neg_2x) / (ones + exp_neg_2x);
        store8(y.add(i), out);
        i += LANES;
    }

    while i < len {
        *y.add(i) = (*x.add(i)).tanh();
        i += 1;
    }
}

/// In-place transform: x = exp(-x * scale).
#[inline(always)]
pub unsafe fn exp_neg_scaled_inplace(x: *mut f32, scale: f32, len: usize) {
    let neg_scale = f32x8::splat(-scale);
    let mut i = 0;

    while i + LANES <= len {
        let xv = load8(x.add(i));
        store8(x.add(i), (xv * neg_scale).exp());
        i += LANES;
    }

    while i < len {
        let v = *x.add(i);
        *x.add(i) = (-v * scale).exp();
        i += 1;
    }
}

/// ReLU squared: y = max(0, x)^2.
#[inline(always)]
pub unsafe fn relu_squared_avx(x: *const f32, y: *mut f32, len: usize) {
    let zero = f32x8::ZERO;
    let mut i = 0;

    while i + LANES <= len {
        let xv = load8(x.add(i));
        let relu = xv.fast_max(zero);
        store8(y.add(i), relu * relu);
        i += LANES;
    }

    while i < len {
        let v = (*x.add(i)).max(0.0);
        *y.add(i) = v * v;
        i += 1;
    }
}

/// exp(x) element-wise.
#[inline(always)]
pub unsafe fn exp_scalar(x: *const f32, y: *mut f32, len: usize) {
    let mut i = 0;

    while i + LANES <= len {
        store8(y.add(i), load8(x.add(i)).exp());
        i += LANES;
    }

    while i < len {
        *y.add(i) = (*x.add(i)).exp();
        i += 1;
    }
}

/// L2 norm of a vector.
#[inline(always)]
pub unsafe fn l2_norm_avx(x: *const f32, len: usize) -> f32 {
    let mut sum = f32x8::ZERO;
    let mut i = 0;

    while i + LANES <= len {
        let xv = load8(x.add(i));
        sum += xv * xv;
        i += LANES;
    }

    let mut out = sum.reduce_add();
    while i < len {
        let xv = *x.add(i);
        out += xv * xv;
        i += 1;
    }

    out.sqrt()
}

/// Normalize vector to unit length (L2), with min norm threshold.
#[inline(always)]
pub unsafe fn l2_normalize_avx(x: *const f32, y: *mut f32, len: usize, min_norm: f32) {
    let norm = l2_norm_avx(x, len).max(min_norm);
    let inv = f32x8::splat(1.0 / norm);

    let mut i = 0;
    while i + LANES <= len {
        store8(y.add(i), load8(x.add(i)) * inv);
        i += LANES;
    }

    while i < len {
        *y.add(i) = *x.add(i) / norm;
        i += 1;
    }
}

/// RWKV7 state update kernel for single token, N=64 head dimension.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
pub unsafe fn rwkv7_wkv_update_avx(
    state: *mut f32,
    w: *const f32,
    k: *const f32,
    v: *const f32,
    kk: *const f32,
    a: *const f32,
    r: *const f32,
    y: *mut f32,
    num_heads: usize,
    head_dim: usize,
) {
    debug_assert_eq!(head_dim, HEAD_DIM);

    for h in 0..num_heads {
        let s_h = state.add(h * HEAD_DIM * HEAD_DIM);
        let w_h = w.add(h * HEAD_DIM);
        let k_h = k.add(h * HEAD_DIM);
        let v_h = v.add(h * HEAD_DIM);
        let kk_h = kk.add(h * HEAD_DIM);
        let a_h = a.add(h * HEAD_DIM);
        let r_h = r.add(h * HEAD_DIM);
        let y_h = y.add(h * HEAD_DIM);

        let w0 = load8(w_h.add(0));
        let w1 = load8(w_h.add(8));
        let w2 = load8(w_h.add(16));
        let w3 = load8(w_h.add(24));
        let w4 = load8(w_h.add(32));
        let w5 = load8(w_h.add(40));
        let w6 = load8(w_h.add(48));
        let w7 = load8(w_h.add(56));

        let k0 = load8(k_h.add(0));
        let k1 = load8(k_h.add(8));
        let k2 = load8(k_h.add(16));
        let k3 = load8(k_h.add(24));
        let k4 = load8(k_h.add(32));
        let k5 = load8(k_h.add(40));
        let k6 = load8(k_h.add(48));
        let k7 = load8(k_h.add(56));

        let kk0 = load8(kk_h.add(0));
        let kk1 = load8(kk_h.add(8));
        let kk2 = load8(kk_h.add(16));
        let kk3 = load8(kk_h.add(24));
        let kk4 = load8(kk_h.add(32));
        let kk5 = load8(kk_h.add(40));
        let kk6 = load8(kk_h.add(48));
        let kk7 = load8(kk_h.add(56));

        let a0 = load8(a_h.add(0));
        let a1 = load8(a_h.add(8));
        let a2 = load8(a_h.add(16));
        let a3 = load8(a_h.add(24));
        let a4 = load8(a_h.add(32));
        let a5 = load8(a_h.add(40));
        let a6 = load8(a_h.add(48));
        let a7 = load8(a_h.add(56));

        let kka0 = kk0 * a0;
        let kka1 = kk1 * a1;
        let kka2 = kk2 * a2;
        let kka3 = kk3 * a3;
        let kka4 = kk4 * a4;
        let kka5 = kk5 * a5;
        let kka6 = kk6 * a6;
        let kka7 = kk7 * a7;

        let r0 = load8(r_h.add(0));
        let r1 = load8(r_h.add(8));
        let r2 = load8(r_h.add(16));
        let r3 = load8(r_h.add(24));
        let r4 = load8(r_h.add(32));
        let r5 = load8(r_h.add(40));
        let r6 = load8(r_h.add(48));
        let r7 = load8(r_h.add(56));

        for i in 0..HEAD_DIM {
            let row = s_h.add(i * HEAD_DIM);
            let v_i = f32x8::splat(*v_h.add(i));

            // RWKV7 reference ordering:
            // 1) overlap t = dot(S_old_row, kk)
            // 2) decay/write S_new_row = S_old_row * w - t * (kk * a) + v_i * k
            let old0 = load8(row.add(0));
            let old1 = load8(row.add(8));
            let old2 = load8(row.add(16));
            let old3 = load8(row.add(24));
            let old4 = load8(row.add(32));
            let old5 = load8(row.add(40));
            let old6 = load8(row.add(48));
            let old7 = load8(row.add(56));

            let mut dot_acc = old0 * kk0;
            dot_acc = old1.mul_add(kk1, dot_acc);
            dot_acc = old2.mul_add(kk2, dot_acc);
            dot_acc = old3.mul_add(kk3, dot_acc);
            dot_acc = old4.mul_add(kk4, dot_acc);
            dot_acc = old5.mul_add(kk5, dot_acc);
            dot_acc = old6.mul_add(kk6, dot_acc);
            dot_acc = old7.mul_add(kk7, dot_acc);

            let t = f32x8::splat(dot_acc.reduce_add());

            let s0 = old0 * w0;
            let s1 = old1 * w1;
            let s2 = old2 * w2;
            let s3 = old3 * w3;
            let s4 = old4 * w4;
            let s5 = old5 * w5;
            let s6 = old6 * w6;
            let s7 = old7 * w7;

            let u0 = (v_i * k0) + (s0 - t * kka0);
            let u1 = (v_i * k1) + (s1 - t * kka1);
            let u2 = (v_i * k2) + (s2 - t * kka2);
            let u3 = (v_i * k3) + (s3 - t * kka3);
            let u4 = (v_i * k4) + (s4 - t * kka4);
            let u5 = (v_i * k5) + (s5 - t * kka5);
            let u6 = (v_i * k6) + (s6 - t * kka6);
            let u7 = (v_i * k7) + (s7 - t * kka7);

            store8(row.add(0), u0);
            store8(row.add(8), u1);
            store8(row.add(16), u2);
            store8(row.add(24), u3);
            store8(row.add(32), u4);
            store8(row.add(40), u5);
            store8(row.add(48), u6);
            store8(row.add(56), u7);

            let mut y_acc = u0 * r0;
            y_acc = u1.mul_add(r1, y_acc);
            y_acc = u2.mul_add(r2, y_acc);
            y_acc = u3.mul_add(r3, y_acc);
            y_acc = u4.mul_add(r4, y_acc);
            y_acc = u5.mul_add(r5, y_acc);
            y_acc = u6.mul_add(r6, y_acc);
            y_acc = u7.mul_add(r7, y_acc);

            *y_h.add(i) = y_acc.reduce_add();
        }
    }
}

/// Softmax: computes softmax(x) and stores in y.
/// Returns log(sum(exp(x - max))).
#[inline(always)]
pub unsafe fn softmax_avx(x: *const f32, y: *mut f32, len: usize) -> f32 {
    let mut i = 0;
    let mut max_v = f32x8::splat(f32::NEG_INFINITY);

    while i + LANES <= len {
        max_v = max_v.fast_max(load8(x.add(i)));
        i += LANES;
    }

    let mut max_val = reduce_max(max_v);
    while i < len {
        max_val = max_val.max(*x.add(i));
        i += 1;
    }

    let max_vec = f32x8::splat(max_val);

    let mut sum = 0.0f32;
    i = 0;
    while i + LANES <= len {
        let exp_v = (load8(x.add(i)) - max_vec).exp();
        store8(y.add(i), exp_v);
        sum += exp_v.reduce_add();
        i += LANES;
    }

    while i < len {
        let exp_val = (*x.add(i) - max_val).exp();
        *y.add(i) = exp_val;
        sum += exp_val;
        i += 1;
    }

    let inv_sum = f32x8::splat(1.0 / sum);
    i = 0;

    while i + LANES <= len {
        store8(y.add(i), load8(y.add(i)) * inv_sum);
        i += LANES;
    }

    while i < len {
        *y.add(i) /= sum;
        i += 1;
    }

    sum.ln() + max_val
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Lcg {
        state: u64,
    }

    impl Lcg {
        fn new(seed: u64) -> Self {
            Self { state: seed }
        }

        fn next_f32(&mut self) -> f32 {
            self.state = self
                .state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1);
            let v = (self.state >> 32) as u32;
            (v as f32) * (1.0 / (u32::MAX as f32))
        }

        fn centered(&mut self, scale: f32) -> f32 {
            (self.next_f32() - 0.5) * 2.0 * scale
        }
    }

    fn fill_centered(buf: &mut [f32], rng: &mut Lcg, scale: f32) {
        for v in buf {
            *v = rng.centered(scale);
        }
    }

    fn assert_close_slice(lhs: &[f32], rhs: &[f32], tol: f32) {
        assert_eq!(lhs.len(), rhs.len());
        for i in 0..lhs.len() {
            let d = (lhs[i] - rhs[i]).abs();
            assert!(
                d <= tol,
                "index={i} lhs={} rhs={} abs_diff={d} tol={tol}",
                lhs[i],
                rhs[i]
            );
        }
    }

    fn gemv_scalar(a: &[f32], x: &[f32], rows: usize, cols: usize) -> Vec<f32> {
        let mut y = vec![0.0; rows];
        for r in 0..rows {
            let mut s = 0.0;
            for c in 0..cols {
                s += a[r * cols + c] * x[c];
            }
            y[r] = s;
        }
        y
    }

    fn gemv_t_scalar(a: &[f32], x: &[f32], rows: usize, cols: usize) -> Vec<f32> {
        let mut y = vec![0.0; cols];
        for r in 0..rows {
            let xr = x[r];
            for c in 0..cols {
                y[c] += a[r * cols + c] * xr;
            }
        }
        y
    }

    fn layer_norm_scalar(x: &[f32], w: &[f32], b: &[f32], eps: f32) -> Vec<f32> {
        let n = x.len() as f32;
        let mean = x.iter().sum::<f32>() / n;
        let var = x.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / n;
        let inv_std = 1.0 / (var + eps).sqrt();
        let mut out = vec![0.0; x.len()];
        for i in 0..x.len() {
            out[i] = ((x[i] - mean) * inv_std) * w[i] + b[i];
        }
        out
    }

    fn group_norm_scalar(
        x: &[f32],
        w: &[f32],
        b: &[f32],
        groups: usize,
        group_size: usize,
        eps: f32,
    ) -> Vec<f32> {
        let mut out = vec![0.0; x.len()];
        for g in 0..groups {
            let start = g * group_size;
            let end = start + group_size;
            let xg = &x[start..end];
            let wg = &w[start..end];
            let bg = &b[start..end];
            let n = group_size as f32;
            let mean = xg.iter().sum::<f32>() / n;
            let var = xg.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / n;
            let inv_std = 1.0 / (var + eps).sqrt();
            for i in 0..group_size {
                out[start + i] = ((xg[i] - mean) * inv_std) * wg[i] + bg[i];
            }
        }
        out
    }

    fn token_shift_scalar(x: &[f32], prev: &[f32], mix: &[f32]) -> Vec<f32> {
        let mut out = vec![0.0; x.len()];
        for i in 0..x.len() {
            out[i] = x[i] + mix[i] * (prev[i] - x[i]);
        }
        out
    }

    fn rwkv_update_scalar(
        state: &mut [f32],
        w: &[f32],
        k: &[f32],
        v: &[f32],
        kk: &[f32],
        a: &[f32],
        r: &[f32],
        y: &mut [f32],
        num_heads: usize,
    ) {
        const N: usize = 64;
        for h in 0..num_heads {
            let state_base = h * N * N;
            let vec_base = h * N;
            for i in 0..N {
                let row_base = state_base + i * N;
                let mut dot = 0.0;
                for j in 0..N {
                    dot += state[row_base + j] * kk[vec_base + j];
                }

                let vi = v[vec_base + i];
                for j in 0..N {
                    state[row_base + j] = state[row_base + j] * w[vec_base + j]
                        - dot * (kk[vec_base + j] * a[vec_base + j])
                        + vi * k[vec_base + j];
                }

                let mut yi = 0.0;
                for j in 0..N {
                    yi += state[row_base + j] * r[vec_base + j];
                }
                y[vec_base + i] = yi;
            }
        }
    }

    #[test]
    fn gemv_and_norm_match_scalar_reference() {
        let mut rng = Lcg::new(0xBAD5EED);

        let rows = 11;
        let cols = 37;
        let mut a = vec![0.0; rows * cols];
        let mut x = vec![0.0; cols];
        let mut x_t = vec![0.0; rows];
        fill_centered(&mut a, &mut rng, 0.75);
        fill_centered(&mut x, &mut rng, 0.5);
        fill_centered(&mut x_t, &mut rng, 0.5);

        let mut y = vec![0.0; rows];
        unsafe { gemv_avx(a.as_ptr(), x.as_ptr(), y.as_mut_ptr(), rows, cols) };
        let y_ref = gemv_scalar(&a, &x, rows, cols);
        assert_close_slice(&y, &y_ref, 2.5e-5);

        let mut yt = vec![0.0; cols];
        unsafe { gemv_t_avx(a.as_ptr(), x_t.as_ptr(), yt.as_mut_ptr(), rows, cols) };
        let yt_ref = gemv_t_scalar(&a, &x_t, rows, cols);
        assert_close_slice(&yt, &yt_ref, 2.5e-5);

        let ln_len = 137;
        let mut ln_x = vec![0.0; ln_len];
        let mut ln_w = vec![0.0; ln_len];
        let mut ln_b = vec![0.0; ln_len];
        fill_centered(&mut ln_x, &mut rng, 0.85);
        fill_centered(&mut ln_w, &mut rng, 0.4);
        fill_centered(&mut ln_b, &mut rng, 0.3);

        let mut ln_out = vec![0.0; ln_len];
        unsafe {
            layer_norm_avx(
                ln_x.as_ptr(),
                ln_w.as_ptr(),
                ln_b.as_ptr(),
                ln_out.as_mut_ptr(),
                ln_len,
                1e-5,
            )
        };
        let ln_ref = layer_norm_scalar(&ln_x, &ln_w, &ln_b, 1e-5);
        assert_close_slice(&ln_out, &ln_ref, 2.5e-5);

        let groups = 5;
        let group_size = 26;
        let gn_len = groups * group_size;
        let mut gn_x = vec![0.0; gn_len];
        let mut gn_w = vec![0.0; gn_len];
        let mut gn_b = vec![0.0; gn_len];
        fill_centered(&mut gn_x, &mut rng, 0.7);
        fill_centered(&mut gn_w, &mut rng, 0.45);
        fill_centered(&mut gn_b, &mut rng, 0.2);

        let mut gn_out = vec![0.0; gn_len];
        unsafe {
            group_norm_avx(
                gn_x.as_ptr(),
                gn_w.as_ptr(),
                gn_b.as_ptr(),
                gn_out.as_mut_ptr(),
                groups,
                group_size,
                1e-5,
            )
        };
        let gn_ref = group_norm_scalar(&gn_x, &gn_w, &gn_b, groups, group_size, 1e-5);
        assert_close_slice(&gn_out, &gn_ref, 2.5e-5);
    }

    #[test]
    fn elementwise_and_softmax_match_scalar_reference() {
        let mut rng = Lcg::new(0xA11CE55);
        let len = 145;

        let mut a = vec![0.0; len];
        let mut b = vec![0.0; len];
        let mut c = vec![0.0; len];
        fill_centered(&mut a, &mut rng, 0.95);
        fill_centered(&mut b, &mut rng, 0.6);
        fill_centered(&mut c, &mut rng, 0.25);

        let mut mul = vec![0.0; len];
        let mut add = vec![0.0; len];
        let mut fma = vec![0.0; len];
        let mut sig = vec![0.0; len];
        let mut tanh = vec![0.0; len];
        let mut relu2 = vec![0.0; len];
        let mut exp = vec![0.0; len];
        let mut soft = vec![0.0; len];

        unsafe {
            mul_avx(a.as_ptr(), b.as_ptr(), mul.as_mut_ptr(), len);
            add_avx(a.as_ptr(), b.as_ptr(), add.as_mut_ptr(), len);
            fma_avx(a.as_ptr(), b.as_ptr(), c.as_ptr(), fma.as_mut_ptr(), len);
            sigmoid_avx(a.as_ptr(), sig.as_mut_ptr(), len);
            tanh_avx(a.as_ptr(), tanh.as_mut_ptr(), len);
            relu_squared_avx(a.as_ptr(), relu2.as_mut_ptr(), len);
            exp_scalar(a.as_ptr(), exp.as_mut_ptr(), len);
        }
        let lse = unsafe { softmax_avx(a.as_ptr(), soft.as_mut_ptr(), len) };

        let mut mul_ref = vec![0.0; len];
        let mut add_ref = vec![0.0; len];
        let mut fma_ref = vec![0.0; len];
        let mut sig_ref = vec![0.0; len];
        let mut tanh_ref = vec![0.0; len];
        let mut relu2_ref = vec![0.0; len];
        let mut exp_ref = vec![0.0; len];

        let mut max_val = f32::NEG_INFINITY;
        for &v in &a {
            max_val = max_val.max(v);
        }
        let mut sum_exp = 0.0;
        let mut soft_ref = vec![0.0; len];

        for i in 0..len {
            mul_ref[i] = a[i] * b[i];
            add_ref[i] = a[i] + b[i];
            fma_ref[i] = a[i] * b[i] + c[i];
            sig_ref[i] = 1.0 / (1.0 + (-a[i]).exp());
            tanh_ref[i] = a[i].tanh();
            let relu = a[i].max(0.0);
            relu2_ref[i] = relu * relu;
            exp_ref[i] = a[i].exp();

            let e = (a[i] - max_val).exp();
            soft_ref[i] = e;
            sum_exp += e;
        }
        for v in &mut soft_ref {
            *v /= sum_exp;
        }

        assert_close_slice(&mul, &mul_ref, 2.5e-5);
        assert_close_slice(&add, &add_ref, 2.5e-5);
        assert_close_slice(&fma, &fma_ref, 2.5e-5);
        assert_close_slice(&sig, &sig_ref, 2.5e-5);
        assert_close_slice(&tanh, &tanh_ref, 2.5e-5);
        assert_close_slice(&relu2, &relu2_ref, 2.5e-5);
        assert_close_slice(&exp, &exp_ref, 2.5e-5);
        assert_close_slice(&soft, &soft_ref, 2.5e-5);
        assert!((lse - (sum_exp.ln() + max_val)).abs() <= 2.5e-5);
    }

    #[test]
    fn token_shift_kernels_match_scalar_reference() {
        let mut rng = Lcg::new(0x71F7_5EED);
        let len = 131;

        let mut x = vec![0.0; len];
        let mut prev = vec![0.0; len];
        let mut mix0 = vec![0.0; len];
        let mut mix1 = vec![0.0; len];
        let mut mix2 = vec![0.0; len];
        let mut mix3 = vec![0.0; len];
        let mut mix4 = vec![0.0; len];
        let mut mix5 = vec![0.0; len];

        fill_centered(&mut x, &mut rng, 0.8);
        fill_centered(&mut prev, &mut rng, 0.8);
        fill_centered(&mut mix0, &mut rng, 1.0);
        fill_centered(&mut mix1, &mut rng, 1.0);
        fill_centered(&mut mix2, &mut rng, 1.0);
        fill_centered(&mut mix3, &mut rng, 1.0);
        fill_centered(&mut mix4, &mut rng, 1.0);
        fill_centered(&mut mix5, &mut rng, 1.0);

        let mut out_single = vec![0.0; len];
        unsafe {
            token_shift_avx(
                x.as_ptr(),
                prev.as_ptr(),
                mix0.as_ptr(),
                out_single.as_mut_ptr(),
                len,
            )
        };
        let single_ref = token_shift_scalar(&x, &prev, &mix0);
        assert_close_slice(&out_single, &single_ref, 2.5e-5);

        let mut out0 = vec![0.0; len];
        let mut out1 = vec![0.0; len];
        let mut out2 = vec![0.0; len];
        let mut out3 = vec![0.0; len];
        let mut out4 = vec![0.0; len];
        let mut out5 = vec![0.0; len];

        unsafe {
            token_shift_multi6_avx(
                x.as_ptr(),
                prev.as_ptr(),
                mix0.as_ptr(),
                mix1.as_ptr(),
                mix2.as_ptr(),
                mix3.as_ptr(),
                mix4.as_ptr(),
                mix5.as_ptr(),
                out0.as_mut_ptr(),
                out1.as_mut_ptr(),
                out2.as_mut_ptr(),
                out3.as_mut_ptr(),
                out4.as_mut_ptr(),
                out5.as_mut_ptr(),
                len,
            )
        };

        let ref0 = token_shift_scalar(&x, &prev, &mix0);
        let ref1 = token_shift_scalar(&x, &prev, &mix1);
        let ref2 = token_shift_scalar(&x, &prev, &mix2);
        let ref3 = token_shift_scalar(&x, &prev, &mix3);
        let ref4 = token_shift_scalar(&x, &prev, &mix4);
        let ref5 = token_shift_scalar(&x, &prev, &mix5);

        assert_close_slice(&out0, &ref0, 2.5e-5);
        assert_close_slice(&out1, &ref1, 2.5e-5);
        assert_close_slice(&out2, &ref2, 2.5e-5);
        assert_close_slice(&out3, &ref3, 2.5e-5);
        assert_close_slice(&out4, &ref4, 2.5e-5);
        assert_close_slice(&out5, &ref5, 2.5e-5);
    }

    #[test]
    fn rwkv_update_matches_scalar_reference() {
        let num_heads = 3;
        let mut rng = Lcg::new(0xFACEFEED);

        let mut state = vec![0.0; num_heads * HEAD_DIM * HEAD_DIM];
        let mut state_ref = vec![0.0; num_heads * HEAD_DIM * HEAD_DIM];
        let mut w = vec![0.0; num_heads * HEAD_DIM];
        let mut k = vec![0.0; num_heads * HEAD_DIM];
        let mut v = vec![0.0; num_heads * HEAD_DIM];
        let mut kk = vec![0.0; num_heads * HEAD_DIM];
        let mut a = vec![0.0; num_heads * HEAD_DIM];
        let mut r = vec![0.0; num_heads * HEAD_DIM];
        let mut y = vec![0.0; num_heads * HEAD_DIM];
        let mut y_ref = vec![0.0; num_heads * HEAD_DIM];

        fill_centered(&mut state, &mut rng, 0.4);
        state_ref.copy_from_slice(&state);
        fill_centered(&mut w, &mut rng, 0.2);
        fill_centered(&mut k, &mut rng, 0.3);
        fill_centered(&mut v, &mut rng, 0.3);
        fill_centered(&mut kk, &mut rng, 0.25);
        fill_centered(&mut a, &mut rng, 0.5);
        fill_centered(&mut r, &mut rng, 0.25);

        unsafe {
            rwkv7_wkv_update_avx(
                state.as_mut_ptr(),
                w.as_ptr(),
                k.as_ptr(),
                v.as_ptr(),
                kk.as_ptr(),
                a.as_ptr(),
                r.as_ptr(),
                y.as_mut_ptr(),
                num_heads,
                HEAD_DIM,
            )
        };

        rwkv_update_scalar(
            &mut state_ref,
            &w,
            &k,
            &v,
            &kk,
            &a,
            &r,
            &mut y_ref,
            num_heads,
        );

        assert_close_slice(&state, &state_ref, 5e-4);
        assert_close_slice(&y, &y_ref, 5e-4);
    }

    #[test]
    fn rwkv_update_uses_pre_decay_overlap_order() {
        let num_heads = 1;
        let mut state = vec![0.0f32; HEAD_DIM * HEAD_DIM];
        let mut w = vec![1.0f32; HEAD_DIM];
        let k = vec![0.0f32; HEAD_DIM];
        let v = vec![0.0f32; HEAD_DIM];
        let mut kk = vec![0.0f32; HEAD_DIM];
        let a = vec![1.0f32; HEAD_DIM];
        let r = vec![0.0f32; HEAD_DIM];
        let mut y = vec![0.0f32; HEAD_DIM];

        // Construct a row where pre-decay and post-decay overlap differ.
        // S_old row: [1, 2, 0, ...], w: [0, 1, ...], kk: [1, 1, ...]
        // Official overlap = dot(S_old, kk) = 3.
        state[0] = 1.0;
        state[1] = 2.0;
        w[0] = 0.0;
        w[1] = 1.0;
        kk[0] = 1.0;
        kk[1] = 1.0;

        unsafe {
            rwkv7_wkv_update_avx(
                state.as_mut_ptr(),
                w.as_ptr(),
                k.as_ptr(),
                v.as_ptr(),
                kk.as_ptr(),
                a.as_ptr(),
                r.as_ptr(),
                y.as_mut_ptr(),
                num_heads,
                HEAD_DIM,
            )
        };

        // Expected with official ordering:
        // S_new = S_old * w - dot(S_old, kk) * (kk * a)
        //       = [0,2] - 3 * [1,1] = [-3,-1]
        assert!((state[0] + 3.0).abs() <= 1e-6, "state[0]={}", state[0]);
        assert!((state[1] + 1.0).abs() <= 1e-6, "state[1]={}", state[1]);
    }

    #[test]
    fn deterministic_kernel_snapshot() {
        let num_heads = 2;
        let mut rng = Lcg::new(0xDEC0DED);

        let mut state = vec![0.0; num_heads * HEAD_DIM * HEAD_DIM];
        let mut w = vec![0.0; num_heads * HEAD_DIM];
        let mut k = vec![0.0; num_heads * HEAD_DIM];
        let mut v = vec![0.0; num_heads * HEAD_DIM];
        let mut kk = vec![0.0; num_heads * HEAD_DIM];
        let mut a = vec![0.0; num_heads * HEAD_DIM];
        let mut r = vec![0.0; num_heads * HEAD_DIM];
        let mut y = vec![0.0; num_heads * HEAD_DIM];
        fill_centered(&mut state, &mut rng, 0.4);
        fill_centered(&mut w, &mut rng, 0.2);
        fill_centered(&mut k, &mut rng, 0.3);
        fill_centered(&mut v, &mut rng, 0.25);
        fill_centered(&mut kk, &mut rng, 0.2);
        fill_centered(&mut a, &mut rng, 0.35);
        fill_centered(&mut r, &mut rng, 0.2);

        unsafe {
            rwkv7_wkv_update_avx(
                state.as_mut_ptr(),
                w.as_ptr(),
                k.as_ptr(),
                v.as_ptr(),
                kk.as_ptr(),
                a.as_ptr(),
                r.as_ptr(),
                y.as_mut_ptr(),
                num_heads,
                HEAD_DIM,
            )
        };

        let mut sm = vec![0.0; HEAD_DIM];
        let lse = unsafe { softmax_avx(y.as_ptr(), sm.as_mut_ptr(), HEAD_DIM) };

        let mut normed = vec![0.0; HEAD_DIM];
        unsafe { l2_normalize_avx(y.as_ptr(), normed.as_mut_ptr(), HEAD_DIM, 1e-6) };

        let checksum = |data: &[f32]| -> f64 {
            data.iter()
                .enumerate()
                .map(|(i, &v)| (i as f64 + 1.0) * (v as f64))
                .sum::<f64>()
        };

        let state_checksum = checksum(&state);
        let y_checksum = checksum(&y);
        let softmax_checksum = checksum(&sm);
        let normed_checksum = checksum(&normed);
        let lse_val = lse as f64;

        let expected_state_checksum = 9_361.599_056_353_56_f64;
        let expected_y_checksum = 24.356_927_025_131_88_f64;
        let expected_softmax_checksum = 32.418_806_117_028_f64;
        let expected_normed_checksum = -0.442_276_961_402_967_57_f64;
        let expected_lse = 4.161_582_469_940_185_5_f64;

        let tol = 2e-4_f64;
        assert!(
            (state_checksum - expected_state_checksum).abs() <= tol,
            "state_checksum={state_checksum}"
        );
        assert!(
            (y_checksum - expected_y_checksum).abs() <= tol,
            "y_checksum={y_checksum}"
        );
        assert!(
            (softmax_checksum - expected_softmax_checksum).abs() <= tol,
            "softmax_checksum={softmax_checksum}"
        );
        assert!(
            (normed_checksum - expected_normed_checksum).abs() <= tol,
            "normed_checksum={normed_checksum}"
        );
        assert!((lse_val - expected_lse).abs() <= tol, "lse={lse_val}");
    }
}
