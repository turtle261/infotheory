//! AVX2/FMA SIMD kernels for RWKV7 operations.
//!
//! All functions require x86_64 with AVX2 and FMA support.
//! No runtime feature detection - caller must ensure features are available.

#![allow(clippy::identity_op)]
#![allow(dead_code, unused_macros)]

use std::arch::asm;
use std::arch::x86_64::*;

/// Force inline and AVX2/FMA for all kernel functions
macro_rules! kernel_fn {
    ($vis:vis fn $name:ident $($tt:tt)*) => {
        #[inline(always)]
        #[target_feature(enable = "avx2,fma")]
        $vis unsafe fn $name $($tt)*
    };
}

#[inline(always)]
unsafe fn prefetch_t0(ptr: *const f32) {
    asm!("prefetcht0 [{0}]", in(reg) ptr, options(nostack, preserves_flags, readonly));
}

const EXP_MAX: f32 = 88.376_26;
const EXP_MIN: f32 = -88.376_26;
const LOG2EF: f32 = std::f32::consts::LOG2_E;
#[allow(clippy::excessive_precision)]
const LN2_HI: f32 = 0.693_359_4;
#[allow(clippy::excessive_precision)]
const LN2_LO: f32 = -2.121_944_4e-4;
#[allow(clippy::excessive_precision)]
const EXP_P0: f32 = 1.987_569_1e-4;
#[allow(clippy::excessive_precision)]
const EXP_P1: f32 = 1.398_199_9e-3;
#[allow(clippy::excessive_precision)]
const EXP_P2: f32 = 8.333_452e-3;
#[allow(clippy::excessive_precision)]
const EXP_P3: f32 = 4.166_579_6e-2;
#[allow(clippy::excessive_precision)]
const EXP_P4: f32 = 1.666_666_6e-1;
const EXP_P5: f32 = 5e-1;

/// Horizontal sum of 8 __m256 vectors, returning results packed in a single __m256.
#[inline(always)]
pub unsafe fn hsum_8x_avx(
    s0: __m256,
    s1: __m256,
    s2: __m256,
    s3: __m256,
    s4: __m256,
    s5: __m256,
    s6: __m256,
    s7: __m256,
) -> __m256 {
    // Stage 1: add pairs horizontally within each vector (8 -> 4 per vector)
    let a01 = _mm256_hadd_ps(s0, s1); // [a0+a1, a2+a3, b0+b1, b2+b3 | a4+a5, a6+a7, b4+b5, b6+b7]
    let a23 = _mm256_hadd_ps(s2, s3);
    let a45 = _mm256_hadd_ps(s4, s5);
    let a67 = _mm256_hadd_ps(s6, s7);

    // Stage 2: add pairs again (4 -> 2 per vector)
    let b0123 = _mm256_hadd_ps(a01, a23); // [sum0, sum1, sum2, sum3 | ...]
    let b4567 = _mm256_hadd_ps(a45, a67);

    // Stage 3: combine low and high 128-bit lanes
    let lo_0123 = _mm256_castps256_ps128(b0123);
    let hi_0123 = _mm256_extractf128_ps(b0123, 1);
    let lo_4567 = _mm256_castps256_ps128(b4567);
    let hi_4567 = _mm256_extractf128_ps(b4567, 1);

    let sum_lo = _mm_add_ps(lo_0123, hi_0123); // [sum0, sum1, sum2, sum3]
    let sum_hi = _mm_add_ps(lo_4567, hi_4567); // [sum4, sum5, sum6, sum7]

    // Combine into single __m256
    _mm256_set_m128(sum_hi, sum_lo)
}

/// Horizontal sum of 4 __m256 vectors, returning results in lower 4 floats of __m128.
#[inline(always)]
pub unsafe fn hsum_4x_avx(s0: __m256, s1: __m256, s2: __m256, s3: __m256) -> __m128 {
    // Stage 1: hadd pairs
    let a01 = _mm256_hadd_ps(s0, s1);
    let a23 = _mm256_hadd_ps(s2, s3);

    // Stage 2: hadd again
    let b = _mm256_hadd_ps(a01, a23);

    // Stage 3: add low and high 128-bit lanes
    let lo = _mm256_castps256_ps128(b);
    let hi = _mm256_extractf128_ps(b, 1);
    _mm_add_ps(lo, hi)
}

/// Horizontal sum of __m256 (8 floats -> 1 float).
#[inline(always)]
pub unsafe fn hsum_avx(v: __m256) -> f32 {
    let x128 = _mm_add_ps(_mm256_extractf128_ps(v, 1), _mm256_castps256_ps128(v));
    let x64 = _mm_add_ps(x128, _mm_movehl_ps(x128, x128));
    let x32 = _mm_add_ss(x64, _mm_shuffle_ps(x64, x64, 0x55));
    _mm_cvtss_f32(x32)
}

#[inline(always)]
unsafe fn exp256_ps(mut x: __m256) -> __m256 {
    let max = _mm256_set1_ps(EXP_MAX);
    let min = _mm256_set1_ps(EXP_MIN);
    x = _mm256_max_ps(_mm256_min_ps(x, max), min);

    let log2ef = _mm256_set1_ps(LOG2EF);
    let half = _mm256_set1_ps(0.5);
    let mut fx = _mm256_mul_ps(x, log2ef);
    fx = _mm256_add_ps(fx, half);

    let fx_floor = _mm256_floor_ps(fx);
    let tmp = fx_floor;

    let mut r = _mm256_fnmadd_ps(tmp, _mm256_set1_ps(LN2_HI), x);
    r = _mm256_fnmadd_ps(tmp, _mm256_set1_ps(LN2_LO), r);

    let mut y = _mm256_set1_ps(EXP_P0);
    y = _mm256_fmadd_ps(y, r, _mm256_set1_ps(EXP_P1));
    y = _mm256_fmadd_ps(y, r, _mm256_set1_ps(EXP_P2));
    y = _mm256_fmadd_ps(y, r, _mm256_set1_ps(EXP_P3));
    y = _mm256_fmadd_ps(y, r, _mm256_set1_ps(EXP_P4));
    y = _mm256_fmadd_ps(y, r, _mm256_set1_ps(EXP_P5));

    let r2 = _mm256_mul_ps(r, r);
    y = _mm256_mul_ps(y, r2);
    y = _mm256_add_ps(y, r);
    y = _mm256_add_ps(y, _mm256_set1_ps(1.0));

    let emm0 = _mm256_cvtps_epi32(fx_floor);
    let emm0 = _mm256_add_epi32(emm0, _mm256_set1_epi32(127));
    let emm0 = _mm256_slli_epi32(emm0, 23);
    let pow2n = _mm256_castsi256_ps(emm0);

    _mm256_mul_ps(y, pow2n)
}

/// Dot product of two aligned f32 slices (length must be multiple of 8).
#[inline(always)]
pub unsafe fn dot_avx(a: *const f32, b: *const f32, len: usize) -> f32 {
    debug_assert!(len % 8 == 0);

    let mut sum0 = _mm256_setzero_ps();
    let mut sum1 = _mm256_setzero_ps();
    let mut sum2 = _mm256_setzero_ps();
    let mut sum3 = _mm256_setzero_ps();

    let mut i = 0;
    // Unroll 4x for better pipelining
    while i + 32 <= len {
        let a0 = _mm256_load_ps(a.add(i));
        let b0 = _mm256_load_ps(b.add(i));
        sum0 = _mm256_fmadd_ps(a0, b0, sum0);

        let a1 = _mm256_load_ps(a.add(i + 8));
        let b1 = _mm256_load_ps(b.add(i + 8));
        sum1 = _mm256_fmadd_ps(a1, b1, sum1);

        let a2 = _mm256_load_ps(a.add(i + 16));
        let b2 = _mm256_load_ps(b.add(i + 16));
        sum2 = _mm256_fmadd_ps(a2, b2, sum2);

        let a3 = _mm256_load_ps(a.add(i + 24));
        let b3 = _mm256_load_ps(b.add(i + 24));
        sum3 = _mm256_fmadd_ps(a3, b3, sum3);

        i += 32;
    }

    // Handle remaining (up to 24 elements)
    while i + 8 <= len {
        let av = _mm256_load_ps(a.add(i));
        let bv = _mm256_load_ps(b.add(i));
        sum0 = _mm256_fmadd_ps(av, bv, sum0);
        i += 8;
    }

    // Combine accumulators
    sum0 = _mm256_add_ps(sum0, sum1);
    sum2 = _mm256_add_ps(sum2, sum3);
    sum0 = _mm256_add_ps(sum0, sum2);

    hsum_avx(sum0)
}

/// Matrix-vector multiply: y = A @ x where A is (rows, cols), x is (cols,), y is (rows,).
/// A is row-major, cols must be multiple of 8.
#[inline(always)]
pub unsafe fn gemv_avx(a: *const f32, x: *const f32, y: *mut f32, rows: usize, cols: usize) {
    debug_assert!(cols % 8 == 0);

    // Process 8 rows at a time to maximize FMA throughput and amortize hsum overhead
    let mut r = 0;
    while r + 8 <= rows {
        let row0 = a.add(r * cols);
        let row1 = a.add((r + 1) * cols);
        let row2 = a.add((r + 2) * cols);
        let row3 = a.add((r + 3) * cols);
        let row4 = a.add((r + 4) * cols);
        let row5 = a.add((r + 5) * cols);
        let row6 = a.add((r + 6) * cols);
        let row7 = a.add((r + 7) * cols);

        // Prefetch next batch of rows
        if r + 16 <= rows {
            prefetch_t0(a.add((r + 8) * cols));
            prefetch_t0(a.add((r + 9) * cols));
            prefetch_t0(a.add((r + 10) * cols));
            prefetch_t0(a.add((r + 11) * cols));
            prefetch_t0(a.add((r + 12) * cols));
            prefetch_t0(a.add((r + 13) * cols));
            prefetch_t0(a.add((r + 14) * cols));
            prefetch_t0(a.add((r + 15) * cols));
        }

        let mut sum0 = _mm256_setzero_ps();
        let mut sum1 = _mm256_setzero_ps();
        let mut sum2 = _mm256_setzero_ps();
        let mut sum3 = _mm256_setzero_ps();
        let mut sum4 = _mm256_setzero_ps();
        let mut sum5 = _mm256_setzero_ps();
        let mut sum6 = _mm256_setzero_ps();
        let mut sum7 = _mm256_setzero_ps();

        // Unroll inner loop 2x to hide FMA latency
        let mut c = 0;
        while c + 16 <= cols {
            let xv0 = _mm256_load_ps(x.add(c));
            let xv1 = _mm256_load_ps(x.add(c + 8));

            sum0 = _mm256_fmadd_ps(_mm256_load_ps(row0.add(c)), xv0, sum0);
            sum0 = _mm256_fmadd_ps(_mm256_load_ps(row0.add(c + 8)), xv1, sum0);
            sum1 = _mm256_fmadd_ps(_mm256_load_ps(row1.add(c)), xv0, sum1);
            sum1 = _mm256_fmadd_ps(_mm256_load_ps(row1.add(c + 8)), xv1, sum1);
            sum2 = _mm256_fmadd_ps(_mm256_load_ps(row2.add(c)), xv0, sum2);
            sum2 = _mm256_fmadd_ps(_mm256_load_ps(row2.add(c + 8)), xv1, sum2);
            sum3 = _mm256_fmadd_ps(_mm256_load_ps(row3.add(c)), xv0, sum3);
            sum3 = _mm256_fmadd_ps(_mm256_load_ps(row3.add(c + 8)), xv1, sum3);
            sum4 = _mm256_fmadd_ps(_mm256_load_ps(row4.add(c)), xv0, sum4);
            sum4 = _mm256_fmadd_ps(_mm256_load_ps(row4.add(c + 8)), xv1, sum4);
            sum5 = _mm256_fmadd_ps(_mm256_load_ps(row5.add(c)), xv0, sum5);
            sum5 = _mm256_fmadd_ps(_mm256_load_ps(row5.add(c + 8)), xv1, sum5);
            sum6 = _mm256_fmadd_ps(_mm256_load_ps(row6.add(c)), xv0, sum6);
            sum6 = _mm256_fmadd_ps(_mm256_load_ps(row6.add(c + 8)), xv1, sum6);
            sum7 = _mm256_fmadd_ps(_mm256_load_ps(row7.add(c)), xv0, sum7);
            sum7 = _mm256_fmadd_ps(_mm256_load_ps(row7.add(c + 8)), xv1, sum7);
            c += 16;
        }
        // Handle remaining 8 elements if cols not multiple of 16
        while c < cols {
            let xv = _mm256_load_ps(x.add(c));
            sum0 = _mm256_fmadd_ps(_mm256_load_ps(row0.add(c)), xv, sum0);
            sum1 = _mm256_fmadd_ps(_mm256_load_ps(row1.add(c)), xv, sum1);
            sum2 = _mm256_fmadd_ps(_mm256_load_ps(row2.add(c)), xv, sum2);
            sum3 = _mm256_fmadd_ps(_mm256_load_ps(row3.add(c)), xv, sum3);
            sum4 = _mm256_fmadd_ps(_mm256_load_ps(row4.add(c)), xv, sum4);
            sum5 = _mm256_fmadd_ps(_mm256_load_ps(row5.add(c)), xv, sum5);
            sum6 = _mm256_fmadd_ps(_mm256_load_ps(row6.add(c)), xv, sum6);
            sum7 = _mm256_fmadd_ps(_mm256_load_ps(row7.add(c)), xv, sum7);
            c += 8;
        }

        // Batch horizontal sums using SIMD hadd
        let sums = hsum_8x_avx(sum0, sum1, sum2, sum3, sum4, sum5, sum6, sum7);
        _mm256_storeu_ps(y.add(r), sums);
        r += 8;
    }

    // Handle remaining 4 rows
    while r + 4 <= rows {
        let row0 = a.add(r * cols);
        let row1 = a.add((r + 1) * cols);
        let row2 = a.add((r + 2) * cols);
        let row3 = a.add((r + 3) * cols);

        let mut sum0 = _mm256_setzero_ps();
        let mut sum1 = _mm256_setzero_ps();
        let mut sum2 = _mm256_setzero_ps();
        let mut sum3 = _mm256_setzero_ps();

        for c in (0..cols).step_by(8) {
            let xv = _mm256_load_ps(x.add(c));
            sum0 = _mm256_fmadd_ps(_mm256_load_ps(row0.add(c)), xv, sum0);
            sum1 = _mm256_fmadd_ps(_mm256_load_ps(row1.add(c)), xv, sum1);
            sum2 = _mm256_fmadd_ps(_mm256_load_ps(row2.add(c)), xv, sum2);
            sum3 = _mm256_fmadd_ps(_mm256_load_ps(row3.add(c)), xv, sum3);
        }

        *y.add(r) = hsum_avx(sum0);
        *y.add(r + 1) = hsum_avx(sum1);
        *y.add(r + 2) = hsum_avx(sum2);
        *y.add(r + 3) = hsum_avx(sum3);
        r += 4;
    }

    // Handle remaining rows
    while r < rows {
        *y.add(r) = dot_avx(a.add(r * cols), x, cols);
        r += 1;
    }
}

/// Matrix-vector multiply with transposed matrix: y = A^T @ x
/// A is (rows, cols) row-major, we compute A^T @ x = (cols, rows) @ (rows,) = (cols,)
#[inline(always)]
pub unsafe fn gemv_t_avx(
    a: *const f32, // (rows, cols) row-major
    x: *const f32, // (rows,)
    y: *mut f32,   // (cols,)
    rows: usize,
    cols: usize,
) {
    debug_assert!(cols % 8 == 0);

    // Zero output
    for c in (0..cols).step_by(8) {
        _mm256_store_ps(y.add(c), _mm256_setzero_ps());
    }

    // Accumulate A[r, :] * x[r] for each row
    for r in 0..rows {
        let row_ptr = a.add(r * cols);
        let x_r = _mm256_set1_ps(*x.add(r));

        for c in (0..cols).step_by(8) {
            let a_vec = _mm256_load_ps(row_ptr.add(c));
            let y_vec = _mm256_load_ps(y.add(c));
            let result = _mm256_fmadd_ps(a_vec, x_r, y_vec);
            _mm256_store_ps(y.add(c), result);
        }
    }
}

/// Element-wise multiply: y = a * b
#[inline(always)]
pub unsafe fn mul_avx(a: *const f32, b: *const f32, y: *mut f32, len: usize) {
    let mut i = 0;
    while i + 8 <= len {
        let av = _mm256_load_ps(a.add(i));
        let bv = _mm256_load_ps(b.add(i));
        _mm256_store_ps(y.add(i), _mm256_mul_ps(av, bv));
        i += 8;
    }
    // Scalar remainder
    while i < len {
        *y.add(i) = *a.add(i) * *b.add(i);
        i += 1;
    }
}

/// Element-wise add: y = a + b
#[inline(always)]
pub unsafe fn add_avx(a: *const f32, b: *const f32, y: *mut f32, len: usize) {
    let mut i = 0;
    while i + 8 <= len {
        let av = _mm256_load_ps(a.add(i));
        let bv = _mm256_load_ps(b.add(i));
        _mm256_store_ps(y.add(i), _mm256_add_ps(av, bv));
        i += 8;
    }
    while i < len {
        *y.add(i) = *a.add(i) + *b.add(i);
        i += 1;
    }
}

/// Element-wise fused multiply-add: y = a * b + c
#[inline(always)]
pub unsafe fn fma_avx(a: *const f32, b: *const f32, c: *const f32, y: *mut f32, len: usize) {
    let mut i = 0;
    while i + 8 <= len {
        let av = _mm256_load_ps(a.add(i));
        let bv = _mm256_load_ps(b.add(i));
        let cv = _mm256_load_ps(c.add(i));
        _mm256_store_ps(y.add(i), _mm256_fmadd_ps(av, bv, cv));
        i += 8;
    }
    while i < len {
        *y.add(i) = *a.add(i) * *b.add(i) + *c.add(i);
        i += 1;
    }
}

/// Scaled add: y = y + scale * x
#[inline(always)]
pub unsafe fn scaled_add_avx(y: *mut f32, x: *const f32, scale: f32, len: usize) {
    let scale_v = _mm256_set1_ps(scale);
    let mut i = 0;
    while i + 8 <= len {
        let yv = _mm256_load_ps(y.add(i));
        let xv = _mm256_load_ps(x.add(i));
        _mm256_store_ps(y.add(i), _mm256_fmadd_ps(xv, scale_v, yv));
        i += 8;
    }
    while i < len {
        *y.add(i) += scale * *x.add(i);
        i += 1;
    }
}

/// Copy: dst = src
#[inline(always)]
pub unsafe fn copy(src: *const f32, dst: *mut f32, len: usize) {
    std::ptr::copy_nonoverlapping(src, dst, len);
}

/// Token shift: out = x + mix * (prev - x)
/// Equivalent to: out = (1 - mix) * x + mix * prev = lerp(x, prev, mix)
#[inline(always)]
pub unsafe fn token_shift_avx(
    x: *const f32,
    prev: *const f32,
    mix: *const f32,
    out: *mut f32,
    len: usize,
) {
    let mut i = 0;
    while i + 8 <= len {
        let xv = _mm256_load_ps(x.add(i));
        let pv = _mm256_load_ps(prev.add(i));
        let mv = _mm256_load_ps(mix.add(i));

        // out = x + mix * (prev - x) = x + mix*prev - mix*x
        let diff = _mm256_sub_ps(pv, xv);
        let result = _mm256_fmadd_ps(mv, diff, xv);
        _mm256_store_ps(out.add(i), result);
        i += 8;
    }
    while i < len {
        let xi = *x.add(i);
        let pi = *prev.add(i);
        let mi = *mix.add(i);
        *out.add(i) = xi + mi * (pi - xi);
        i += 1;
    }
}

/// Token shift for six projections sharing the same x/prev inputs.
#[inline(always)]
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
    while i + 8 <= len {
        let xv = _mm256_load_ps(x.add(i));
        let pv = _mm256_load_ps(prev.add(i));
        let diff = _mm256_sub_ps(pv, xv);

        let m0 = _mm256_load_ps(mix0.add(i));
        let m1 = _mm256_load_ps(mix1.add(i));
        let m2 = _mm256_load_ps(mix2.add(i));
        let m3 = _mm256_load_ps(mix3.add(i));
        let m4 = _mm256_load_ps(mix4.add(i));
        let m5 = _mm256_load_ps(mix5.add(i));

        _mm256_store_ps(out0.add(i), _mm256_fmadd_ps(m0, diff, xv));
        _mm256_store_ps(out1.add(i), _mm256_fmadd_ps(m1, diff, xv));
        _mm256_store_ps(out2.add(i), _mm256_fmadd_ps(m2, diff, xv));
        _mm256_store_ps(out3.add(i), _mm256_fmadd_ps(m3, diff, xv));
        _mm256_store_ps(out4.add(i), _mm256_fmadd_ps(m4, diff, xv));
        _mm256_store_ps(out5.add(i), _mm256_fmadd_ps(m5, diff, xv));
        i += 8;
    }

    while i < len {
        let xi = *x.add(i);
        let pi = *prev.add(i);
        let d = pi - xi;
        *out0.add(i) = xi + *mix0.add(i) * d;
        *out1.add(i) = xi + *mix1.add(i) * d;
        *out2.add(i) = xi + *mix2.add(i) * d;
        *out3.add(i) = xi + *mix3.add(i) * d;
        *out4.add(i) = xi + *mix4.add(i) * d;
        *out5.add(i) = xi + *mix5.add(i) * d;
        i += 1;
    }
}

/// Layer normalization: y = (x - mean) / sqrt(var + eps) * weight + bias
/// Uses population variance (divide by N, not N-1).
#[inline(always)]
pub unsafe fn layer_norm_avx(
    x: *const f32,
    weight: *const f32,
    bias: *const f32,
    y: *mut f32,
    len: usize,
    eps: f32,
) {
    // Compute mean
    let mut sum = _mm256_setzero_ps();
    let mut i = 0;
    while i + 8 <= len {
        sum = _mm256_add_ps(sum, _mm256_load_ps(x.add(i)));
        i += 8;
    }
    let mut mean = hsum_avx(sum);
    // Handle remainder
    while i < len {
        mean += *x.add(i);
        i += 1;
    }
    mean /= len as f32;
    let mean_v = _mm256_set1_ps(mean);

    // Compute variance = mean((x - mean)^2)
    let mut var_sum = _mm256_setzero_ps();
    i = 0;
    while i + 8 <= len {
        let xv = _mm256_load_ps(x.add(i));
        let diff = _mm256_sub_ps(xv, mean_v);
        var_sum = _mm256_fmadd_ps(diff, diff, var_sum);
        i += 8;
    }
    let mut var = hsum_avx(var_sum);
    while i < len {
        let diff = *x.add(i) - mean;
        var += diff * diff;
        i += 1;
    }
    var /= len as f32;

    // Normalize: y = (x - mean) / sqrt(var + eps) * weight + bias
    let inv_std = 1.0 / (var + eps).sqrt();
    let inv_std_v = _mm256_set1_ps(inv_std);

    i = 0;
    while i + 8 <= len {
        let xv = _mm256_load_ps(x.add(i));
        let wv = _mm256_load_ps(weight.add(i));
        let bv = _mm256_load_ps(bias.add(i));

        let centered = _mm256_sub_ps(xv, mean_v);
        let normed = _mm256_mul_ps(centered, inv_std_v);
        let result = _mm256_fmadd_ps(normed, wv, bv);

        _mm256_store_ps(y.add(i), result);
        i += 8;
    }
    while i < len {
        let normed = (*x.add(i) - mean) * inv_std;
        *y.add(i) = normed * *weight.add(i) + *bias.add(i);
        i += 1;
    }
}

/// Group normalization for RWKV7 (groups = num_heads, elements per group = head_dim).
/// Input shape is (groups * group_size), normalize within each group.
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

        // Mean within group
        let mut sum = _mm256_setzero_ps();
        let mut i = 0;
        while i + 8 <= group_size {
            sum = _mm256_add_ps(sum, _mm256_load_ps(x_g.add(i)));
            i += 8;
        }
        let mut mean = hsum_avx(sum);
        while i < group_size {
            mean += *x_g.add(i);
            i += 1;
        }
        mean /= group_size as f32;
        let mean_v = _mm256_set1_ps(mean);

        // Variance within group
        let mut var_sum = _mm256_setzero_ps();
        i = 0;
        while i + 8 <= group_size {
            let xv = _mm256_load_ps(x_g.add(i));
            let diff = _mm256_sub_ps(xv, mean_v);
            var_sum = _mm256_fmadd_ps(diff, diff, var_sum);
            i += 8;
        }
        let mut var = hsum_avx(var_sum);
        while i < group_size {
            let diff = *x_g.add(i) - mean;
            var += diff * diff;
            i += 1;
        }
        var /= group_size as f32;

        let inv_std = 1.0 / (var + eps).sqrt();
        let inv_std_v = _mm256_set1_ps(inv_std);

        // Normalize
        i = 0;
        while i + 8 <= group_size {
            let xv = _mm256_load_ps(x_g.add(i));
            let wv = _mm256_load_ps(w_g.add(i));
            let bv = _mm256_load_ps(b_g.add(i));

            let centered = _mm256_sub_ps(xv, mean_v);
            let normed = _mm256_mul_ps(centered, inv_std_v);
            let result = _mm256_fmadd_ps(normed, wv, bv);

            _mm256_store_ps(y_g.add(i), result);
            i += 8;
        }
        while i < group_size {
            let normed = (*x_g.add(i) - mean) * inv_std;
            *y_g.add(i) = normed * *w_g.add(i) + *b_g.add(i);
            i += 1;
        }
    }
}

/// Sigmoid: y = 1 / (1 + exp(-x))
#[inline(always)]
pub unsafe fn sigmoid_avx(x: *const f32, y: *mut f32, len: usize) {
    let ones = _mm256_set1_ps(1.0);
    let twos = _mm256_set1_ps(2.0);
    let zeros = _mm256_setzero_ps();
    let mut i = 0;
    while i + 8 <= len {
        let xv = _mm256_load_ps(x.add(i));
        let neg = _mm256_sub_ps(zeros, xv);
        let exp_neg = exp256_ps(neg);
        let denom = _mm256_add_ps(ones, exp_neg);
        let mut recip = _mm256_rcp_ps(denom);
        recip = _mm256_mul_ps(recip, _mm256_sub_ps(twos, _mm256_mul_ps(denom, recip)));
        _mm256_store_ps(y.add(i), recip);
        i += 8;
    }
    while i < len {
        let v = *x.add(i);
        *y.add(i) = 1.0 / (1.0 + (-v).exp());
        i += 1;
    }
}

/// Tanh: y = tanh(x)
#[inline(always)]
pub unsafe fn tanh_avx(x: *const f32, y: *mut f32, len: usize) {
    let ones = _mm256_set1_ps(1.0);
    let twos = _mm256_set1_ps(2.0);
    let minus_two = _mm256_set1_ps(-2.0);
    let mut i = 0;
    while i + 8 <= len {
        let xv = _mm256_load_ps(x.add(i));
        let neg2x = _mm256_mul_ps(minus_two, xv);
        let exp_neg2x = exp256_ps(neg2x);
        let numer = _mm256_sub_ps(ones, exp_neg2x);
        let denom = _mm256_add_ps(ones, exp_neg2x);
        let mut recip = _mm256_rcp_ps(denom);
        recip = _mm256_mul_ps(recip, _mm256_sub_ps(twos, _mm256_mul_ps(denom, recip)));
        let tanh = _mm256_mul_ps(numer, recip);
        _mm256_store_ps(y.add(i), tanh);
        i += 8;
    }
    while i < len {
        *y.add(i) = (*x.add(i)).tanh();
        i += 1;
    }
}

/// In-place transform: x = exp(-x * scale)
#[inline(always)]
pub unsafe fn exp_neg_scaled_inplace(x: *mut f32, scale: f32, len: usize) {
    let scale_v = _mm256_set1_ps(-scale);
    let mut i = 0;
    while i + 8 <= len {
        let xv = _mm256_load_ps(x.add(i));
        let scaled = _mm256_mul_ps(xv, scale_v);
        let expv = exp256_ps(scaled);
        _mm256_store_ps(x.add(i), expv);
        i += 8;
    }
    while i < len {
        let ptr = x.add(i);
        let val = *ptr;
        *ptr = (-val * scale).exp();
        i += 1;
    }
}

/// ReLU squared: y = max(0, x)^2
#[inline(always)]
pub unsafe fn relu_squared_avx(x: *const f32, y: *mut f32, len: usize) {
    let zero = _mm256_setzero_ps();
    let mut i = 0;
    while i + 8 <= len {
        let xv = _mm256_load_ps(x.add(i));
        let relu = _mm256_max_ps(xv, zero);
        _mm256_store_ps(y.add(i), _mm256_mul_ps(relu, relu));
        i += 8;
    }
    while i < len {
        let v = (*x.add(i)).max(0.0);
        *y.add(i) = v * v;
        i += 1;
    }
}

/// exp(x) element-wise
#[inline]
pub unsafe fn exp_scalar(x: *const f32, y: *mut f32, len: usize) {
    for i in 0..len {
        *y.add(i) = (*x.add(i)).exp();
    }
}

/// L2 norm of a vector
#[inline(always)]
pub unsafe fn l2_norm_avx(x: *const f32, len: usize) -> f32 {
    let mut sum = _mm256_setzero_ps();
    let mut i = 0;
    while i + 8 <= len {
        let xv = _mm256_load_ps(x.add(i));
        sum = _mm256_fmadd_ps(xv, xv, sum);
        i += 8;
    }
    let mut result = hsum_avx(sum);
    while i < len {
        result += (*x.add(i)) * (*x.add(i));
        i += 1;
    }
    result.sqrt()
}

/// Normalize vector to unit length (L2), with min norm threshold.
#[inline(always)]
pub unsafe fn l2_normalize_avx(x: *const f32, y: *mut f32, len: usize, min_norm: f32) {
    let norm = l2_norm_avx(x, len).max(min_norm);
    let inv_norm = 1.0 / norm;
    let inv_norm_v = _mm256_set1_ps(inv_norm);

    let mut i = 0;
    while i + 8 <= len {
        let xv = _mm256_load_ps(x.add(i));
        _mm256_store_ps(y.add(i), _mm256_mul_ps(xv, inv_norm_v));
        i += 8;
    }
    while i < len {
        *y.add(i) = *x.add(i) * inv_norm;
        i += 1;
    }
}

/// RWKV7 state update kernel for single token, N=64 head dimension.
/// Implements: S = S * w.T - S @ kk * (kk*a).T + v * k.T; y = S @ r
///
/// This is the critical inner kernel - must be maximally optimized.
/// Processes 4 rows at a time for better instruction-level parallelism.
#[inline(always)]
pub unsafe fn rwkv7_wkv_update_avx(
    state: *mut f32, // (H, N, N) = H * 64 * 64 floats
    w: *const f32,   // (H, N) decay
    k: *const f32,   // (H, N) key (already scaled)
    v: *const f32,   // (H, N) value
    kk: *const f32,  // (H, N) normalized key
    a: *const f32,   // (H, N) gate for subtraction term
    r: *const f32,   // (H, N) receptance
    y: *mut f32,     // (H, N) output
    num_heads: usize,
    head_dim: usize, // Must be 64
) {
    debug_assert_eq!(head_dim, 64);
    const N: usize = 64;

    for h in 0..num_heads {
        let s_h = state.add(h * N * N);
        let w_h = w.add(h * N);
        let k_h = k.add(h * N);
        let v_h = v.add(h * N);
        let kk_h = kk.add(h * N);
        let a_h = a.add(h * N);
        let r_h = r.add(h * N);
        let y_h = y.add(h * N);

        // Preload w, kk, r, k, a vectors (8 AVX registers each = 64 floats)
        let w0 = _mm256_load_ps(w_h.add(0));
        let w1 = _mm256_load_ps(w_h.add(8));
        let w2 = _mm256_load_ps(w_h.add(16));
        let w3 = _mm256_load_ps(w_h.add(24));
        let w4 = _mm256_load_ps(w_h.add(32));
        let w5 = _mm256_load_ps(w_h.add(40));
        let w6 = _mm256_load_ps(w_h.add(48));
        let w7 = _mm256_load_ps(w_h.add(56));

        let kk0 = _mm256_load_ps(kk_h.add(0));
        let kk1 = _mm256_load_ps(kk_h.add(8));
        let kk2 = _mm256_load_ps(kk_h.add(16));
        let kk3 = _mm256_load_ps(kk_h.add(24));
        let kk4 = _mm256_load_ps(kk_h.add(32));
        let kk5 = _mm256_load_ps(kk_h.add(40));
        let kk6 = _mm256_load_ps(kk_h.add(48));
        let kk7 = _mm256_load_ps(kk_h.add(56));

        let r0 = _mm256_load_ps(r_h.add(0));
        let r1 = _mm256_load_ps(r_h.add(8));
        let r2 = _mm256_load_ps(r_h.add(16));
        let r3 = _mm256_load_ps(r_h.add(24));
        let r4 = _mm256_load_ps(r_h.add(32));
        let r5 = _mm256_load_ps(r_h.add(40));
        let r6 = _mm256_load_ps(r_h.add(48));
        let r7 = _mm256_load_ps(r_h.add(56));

        let k0 = _mm256_load_ps(k_h.add(0));
        let k1 = _mm256_load_ps(k_h.add(8));
        let k2 = _mm256_load_ps(k_h.add(16));
        let k3 = _mm256_load_ps(k_h.add(24));
        let k4 = _mm256_load_ps(k_h.add(32));
        let k5 = _mm256_load_ps(k_h.add(40));
        let k6 = _mm256_load_ps(k_h.add(48));
        let k7 = _mm256_load_ps(k_h.add(56));

        // Precompute kk*a
        let a0 = _mm256_load_ps(a_h.add(0));
        let a1 = _mm256_load_ps(a_h.add(8));
        let a2 = _mm256_load_ps(a_h.add(16));
        let a3 = _mm256_load_ps(a_h.add(24));
        let a4 = _mm256_load_ps(a_h.add(32));
        let a5 = _mm256_load_ps(a_h.add(40));
        let a6 = _mm256_load_ps(a_h.add(48));
        let a7 = _mm256_load_ps(a_h.add(56));

        let kka0 = _mm256_mul_ps(kk0, a0);
        let kka1 = _mm256_mul_ps(kk1, a1);
        let kka2 = _mm256_mul_ps(kk2, a2);
        let kka3 = _mm256_mul_ps(kk3, a3);
        let kka4 = _mm256_mul_ps(kk4, a4);
        let kka5 = _mm256_mul_ps(kk5, a5);
        let kka6 = _mm256_mul_ps(kk6, a6);
        let kka7 = _mm256_mul_ps(kk7, a7);

        // Process 4 rows at a time for maximum ILP
        let mut i = 0;
        while i + 4 <= N {
            let row0 = s_h.add(i * N);
            let row1 = s_h.add((i + 1) * N);
            let row2 = s_h.add((i + 2) * N);
            let row3 = s_h.add((i + 3) * N);

            // Prefetch next batch
            if i + 7 < N {
                prefetch_t0(s_h.add((i + 4) * N));
                prefetch_t0(s_h.add((i + 4) * N).add(32));
                prefetch_t0(s_h.add((i + 5) * N));
                prefetch_t0(s_h.add((i + 5) * N).add(32));
                prefetch_t0(s_h.add((i + 6) * N));
                prefetch_t0(s_h.add((i + 6) * N).add(32));
                prefetch_t0(s_h.add((i + 7) * N));
                prefetch_t0(s_h.add((i + 7) * N).add(32));
            }

            // Load all 4 rows' first half
            let s00 = _mm256_mul_ps(_mm256_load_ps(row0.add(0)), w0);
            let s01 = _mm256_mul_ps(_mm256_load_ps(row0.add(8)), w1);
            let s02 = _mm256_mul_ps(_mm256_load_ps(row0.add(16)), w2);
            let s03 = _mm256_mul_ps(_mm256_load_ps(row0.add(24)), w3);
            let s04 = _mm256_mul_ps(_mm256_load_ps(row0.add(32)), w4);
            let s05 = _mm256_mul_ps(_mm256_load_ps(row0.add(40)), w5);
            let s06 = _mm256_mul_ps(_mm256_load_ps(row0.add(48)), w6);
            let s07 = _mm256_mul_ps(_mm256_load_ps(row0.add(56)), w7);

            let s10 = _mm256_mul_ps(_mm256_load_ps(row1.add(0)), w0);
            let s11 = _mm256_mul_ps(_mm256_load_ps(row1.add(8)), w1);
            let s12 = _mm256_mul_ps(_mm256_load_ps(row1.add(16)), w2);
            let s13 = _mm256_mul_ps(_mm256_load_ps(row1.add(24)), w3);
            let s14 = _mm256_mul_ps(_mm256_load_ps(row1.add(32)), w4);
            let s15 = _mm256_mul_ps(_mm256_load_ps(row1.add(40)), w5);
            let s16 = _mm256_mul_ps(_mm256_load_ps(row1.add(48)), w6);
            let s17 = _mm256_mul_ps(_mm256_load_ps(row1.add(56)), w7);

            let s20 = _mm256_mul_ps(_mm256_load_ps(row2.add(0)), w0);
            let s21 = _mm256_mul_ps(_mm256_load_ps(row2.add(8)), w1);
            let s22 = _mm256_mul_ps(_mm256_load_ps(row2.add(16)), w2);
            let s23 = _mm256_mul_ps(_mm256_load_ps(row2.add(24)), w3);
            let s24 = _mm256_mul_ps(_mm256_load_ps(row2.add(32)), w4);
            let s25 = _mm256_mul_ps(_mm256_load_ps(row2.add(40)), w5);
            let s26 = _mm256_mul_ps(_mm256_load_ps(row2.add(48)), w6);
            let s27 = _mm256_mul_ps(_mm256_load_ps(row2.add(56)), w7);

            let s30 = _mm256_mul_ps(_mm256_load_ps(row3.add(0)), w0);
            let s31 = _mm256_mul_ps(_mm256_load_ps(row3.add(8)), w1);
            let s32 = _mm256_mul_ps(_mm256_load_ps(row3.add(16)), w2);
            let s33 = _mm256_mul_ps(_mm256_load_ps(row3.add(24)), w3);
            let s34 = _mm256_mul_ps(_mm256_load_ps(row3.add(32)), w4);
            let s35 = _mm256_mul_ps(_mm256_load_ps(row3.add(40)), w5);
            let s36 = _mm256_mul_ps(_mm256_load_ps(row3.add(48)), w6);
            let s37 = _mm256_mul_ps(_mm256_load_ps(row3.add(56)), w7);

            // Compute dot products with kk for all 4 rows
            let mut d0 = _mm256_mul_ps(s00, kk0);
            d0 = _mm256_fmadd_ps(s01, kk1, d0);
            d0 = _mm256_fmadd_ps(s02, kk2, d0);
            d0 = _mm256_fmadd_ps(s03, kk3, d0);
            d0 = _mm256_fmadd_ps(s04, kk4, d0);
            d0 = _mm256_fmadd_ps(s05, kk5, d0);
            d0 = _mm256_fmadd_ps(s06, kk6, d0);
            d0 = _mm256_fmadd_ps(s07, kk7, d0);

            let mut d1 = _mm256_mul_ps(s10, kk0);
            d1 = _mm256_fmadd_ps(s11, kk1, d1);
            d1 = _mm256_fmadd_ps(s12, kk2, d1);
            d1 = _mm256_fmadd_ps(s13, kk3, d1);
            d1 = _mm256_fmadd_ps(s14, kk4, d1);
            d1 = _mm256_fmadd_ps(s15, kk5, d1);
            d1 = _mm256_fmadd_ps(s16, kk6, d1);
            d1 = _mm256_fmadd_ps(s17, kk7, d1);

            let mut d2 = _mm256_mul_ps(s20, kk0);
            d2 = _mm256_fmadd_ps(s21, kk1, d2);
            d2 = _mm256_fmadd_ps(s22, kk2, d2);
            d2 = _mm256_fmadd_ps(s23, kk3, d2);
            d2 = _mm256_fmadd_ps(s24, kk4, d2);
            d2 = _mm256_fmadd_ps(s25, kk5, d2);
            d2 = _mm256_fmadd_ps(s26, kk6, d2);
            d2 = _mm256_fmadd_ps(s27, kk7, d2);

            let mut d3 = _mm256_mul_ps(s30, kk0);
            d3 = _mm256_fmadd_ps(s31, kk1, d3);
            d3 = _mm256_fmadd_ps(s32, kk2, d3);
            d3 = _mm256_fmadd_ps(s33, kk3, d3);
            d3 = _mm256_fmadd_ps(s34, kk4, d3);
            d3 = _mm256_fmadd_ps(s35, kk5, d3);
            d3 = _mm256_fmadd_ps(s36, kk6, d3);
            d3 = _mm256_fmadd_ps(s37, kk7, d3);

            // Horizontal sums
            let t0 = _mm256_set1_ps(hsum_avx(d0));
            let t1 = _mm256_set1_ps(hsum_avx(d1));
            let t2 = _mm256_set1_ps(hsum_avx(d2));
            let t3 = _mm256_set1_ps(hsum_avx(d3));

            // v scalars
            let v0 = _mm256_set1_ps(*v_h.add(i));
            let v1 = _mm256_set1_ps(*v_h.add(i + 1));
            let v2 = _mm256_set1_ps(*v_h.add(i + 2));
            let v3 = _mm256_set1_ps(*v_h.add(i + 3));

            // Update: s = s - tmp * kka + v * k
            let s00 = _mm256_fmadd_ps(v0, k0, _mm256_fnmadd_ps(t0, kka0, s00));
            let s01 = _mm256_fmadd_ps(v0, k1, _mm256_fnmadd_ps(t0, kka1, s01));
            let s02 = _mm256_fmadd_ps(v0, k2, _mm256_fnmadd_ps(t0, kka2, s02));
            let s03 = _mm256_fmadd_ps(v0, k3, _mm256_fnmadd_ps(t0, kka3, s03));
            let s04 = _mm256_fmadd_ps(v0, k4, _mm256_fnmadd_ps(t0, kka4, s04));
            let s05 = _mm256_fmadd_ps(v0, k5, _mm256_fnmadd_ps(t0, kka5, s05));
            let s06 = _mm256_fmadd_ps(v0, k6, _mm256_fnmadd_ps(t0, kka6, s06));
            let s07 = _mm256_fmadd_ps(v0, k7, _mm256_fnmadd_ps(t0, kka7, s07));

            let s10 = _mm256_fmadd_ps(v1, k0, _mm256_fnmadd_ps(t1, kka0, s10));
            let s11 = _mm256_fmadd_ps(v1, k1, _mm256_fnmadd_ps(t1, kka1, s11));
            let s12 = _mm256_fmadd_ps(v1, k2, _mm256_fnmadd_ps(t1, kka2, s12));
            let s13 = _mm256_fmadd_ps(v1, k3, _mm256_fnmadd_ps(t1, kka3, s13));
            let s14 = _mm256_fmadd_ps(v1, k4, _mm256_fnmadd_ps(t1, kka4, s14));
            let s15 = _mm256_fmadd_ps(v1, k5, _mm256_fnmadd_ps(t1, kka5, s15));
            let s16 = _mm256_fmadd_ps(v1, k6, _mm256_fnmadd_ps(t1, kka6, s16));
            let s17 = _mm256_fmadd_ps(v1, k7, _mm256_fnmadd_ps(t1, kka7, s17));

            let s20 = _mm256_fmadd_ps(v2, k0, _mm256_fnmadd_ps(t2, kka0, s20));
            let s21 = _mm256_fmadd_ps(v2, k1, _mm256_fnmadd_ps(t2, kka1, s21));
            let s22 = _mm256_fmadd_ps(v2, k2, _mm256_fnmadd_ps(t2, kka2, s22));
            let s23 = _mm256_fmadd_ps(v2, k3, _mm256_fnmadd_ps(t2, kka3, s23));
            let s24 = _mm256_fmadd_ps(v2, k4, _mm256_fnmadd_ps(t2, kka4, s24));
            let s25 = _mm256_fmadd_ps(v2, k5, _mm256_fnmadd_ps(t2, kka5, s25));
            let s26 = _mm256_fmadd_ps(v2, k6, _mm256_fnmadd_ps(t2, kka6, s26));
            let s27 = _mm256_fmadd_ps(v2, k7, _mm256_fnmadd_ps(t2, kka7, s27));

            let s30 = _mm256_fmadd_ps(v3, k0, _mm256_fnmadd_ps(t3, kka0, s30));
            let s31 = _mm256_fmadd_ps(v3, k1, _mm256_fnmadd_ps(t3, kka1, s31));
            let s32 = _mm256_fmadd_ps(v3, k2, _mm256_fnmadd_ps(t3, kka2, s32));
            let s33 = _mm256_fmadd_ps(v3, k3, _mm256_fnmadd_ps(t3, kka3, s33));
            let s34 = _mm256_fmadd_ps(v3, k4, _mm256_fnmadd_ps(t3, kka4, s34));
            let s35 = _mm256_fmadd_ps(v3, k5, _mm256_fnmadd_ps(t3, kka5, s35));
            let s36 = _mm256_fmadd_ps(v3, k6, _mm256_fnmadd_ps(t3, kka6, s36));
            let s37 = _mm256_fmadd_ps(v3, k7, _mm256_fnmadd_ps(t3, kka7, s37));

            // Store updated state
            _mm256_store_ps(row0.add(0), s00);
            _mm256_store_ps(row0.add(8), s01);
            _mm256_store_ps(row0.add(16), s02);
            _mm256_store_ps(row0.add(24), s03);
            _mm256_store_ps(row0.add(32), s04);
            _mm256_store_ps(row0.add(40), s05);
            _mm256_store_ps(row0.add(48), s06);
            _mm256_store_ps(row0.add(56), s07);

            _mm256_store_ps(row1.add(0), s10);
            _mm256_store_ps(row1.add(8), s11);
            _mm256_store_ps(row1.add(16), s12);
            _mm256_store_ps(row1.add(24), s13);
            _mm256_store_ps(row1.add(32), s14);
            _mm256_store_ps(row1.add(40), s15);
            _mm256_store_ps(row1.add(48), s16);
            _mm256_store_ps(row1.add(56), s17);

            _mm256_store_ps(row2.add(0), s20);
            _mm256_store_ps(row2.add(8), s21);
            _mm256_store_ps(row2.add(16), s22);
            _mm256_store_ps(row2.add(24), s23);
            _mm256_store_ps(row2.add(32), s24);
            _mm256_store_ps(row2.add(40), s25);
            _mm256_store_ps(row2.add(48), s26);
            _mm256_store_ps(row2.add(56), s27);

            _mm256_store_ps(row3.add(0), s30);
            _mm256_store_ps(row3.add(8), s31);
            _mm256_store_ps(row3.add(16), s32);
            _mm256_store_ps(row3.add(24), s33);
            _mm256_store_ps(row3.add(32), s34);
            _mm256_store_ps(row3.add(40), s35);
            _mm256_store_ps(row3.add(48), s36);
            _mm256_store_ps(row3.add(56), s37);

            // Compute y = s @ r
            let mut y0 = _mm256_mul_ps(s00, r0);
            y0 = _mm256_fmadd_ps(s01, r1, y0);
            y0 = _mm256_fmadd_ps(s02, r2, y0);
            y0 = _mm256_fmadd_ps(s03, r3, y0);
            y0 = _mm256_fmadd_ps(s04, r4, y0);
            y0 = _mm256_fmadd_ps(s05, r5, y0);
            y0 = _mm256_fmadd_ps(s06, r6, y0);
            y0 = _mm256_fmadd_ps(s07, r7, y0);

            let mut y1 = _mm256_mul_ps(s10, r0);
            y1 = _mm256_fmadd_ps(s11, r1, y1);
            y1 = _mm256_fmadd_ps(s12, r2, y1);
            y1 = _mm256_fmadd_ps(s13, r3, y1);
            y1 = _mm256_fmadd_ps(s14, r4, y1);
            y1 = _mm256_fmadd_ps(s15, r5, y1);
            y1 = _mm256_fmadd_ps(s16, r6, y1);
            y1 = _mm256_fmadd_ps(s17, r7, y1);

            let mut y2 = _mm256_mul_ps(s20, r0);
            y2 = _mm256_fmadd_ps(s21, r1, y2);
            y2 = _mm256_fmadd_ps(s22, r2, y2);
            y2 = _mm256_fmadd_ps(s23, r3, y2);
            y2 = _mm256_fmadd_ps(s24, r4, y2);
            y2 = _mm256_fmadd_ps(s25, r5, y2);
            y2 = _mm256_fmadd_ps(s26, r6, y2);
            y2 = _mm256_fmadd_ps(s27, r7, y2);

            let mut y3 = _mm256_mul_ps(s30, r0);
            y3 = _mm256_fmadd_ps(s31, r1, y3);
            y3 = _mm256_fmadd_ps(s32, r2, y3);
            y3 = _mm256_fmadd_ps(s33, r3, y3);
            y3 = _mm256_fmadd_ps(s34, r4, y3);
            y3 = _mm256_fmadd_ps(s35, r5, y3);
            y3 = _mm256_fmadd_ps(s36, r6, y3);
            y3 = _mm256_fmadd_ps(s37, r7, y3);

            *y_h.add(i) = hsum_avx(y0);
            *y_h.add(i + 1) = hsum_avx(y1);
            *y_h.add(i + 2) = hsum_avx(y2);
            *y_h.add(i + 3) = hsum_avx(y3);

            i += 4;
        }
    }
}
/// Softmax: computes softmax(x) and stores in y.
/// Returns log(sum(exp(x - max))).
#[inline(always)]
pub unsafe fn softmax_avx(x: *const f32, y: *mut f32, len: usize) -> f32 {
    // Find max
    let mut max_v = _mm256_set1_ps(f32::NEG_INFINITY);
    let mut i = 0;
    while i + 8 <= len {
        let xv = _mm256_load_ps(x.add(i));
        max_v = _mm256_max_ps(max_v, xv);
        i += 8;
    }
    // Reduce max_v
    let max128 = _mm_max_ps(
        _mm256_extractf128_ps(max_v, 1),
        _mm256_castps256_ps128(max_v),
    );
    let max64 = _mm_max_ps(max128, _mm_movehl_ps(max128, max128));
    let max32 = _mm_max_ss(max64, _mm_shuffle_ps(max64, max64, 0x55));
    let mut max_val = _mm_cvtss_f32(max32);
    while i < len {
        max_val = max_val.max(*x.add(i));
        i += 1;
    }

    // Compute exp(x - max) and sum
    let mut sum = 0.0f32;
    i = 0;
    while i < len {
        let exp_val = (*x.add(i) - max_val).exp();
        *y.add(i) = exp_val;
        sum += exp_val;
        i += 1;
    }

    // Normalize
    let inv_sum = 1.0 / sum;
    let inv_sum_v = _mm256_set1_ps(inv_sum);
    i = 0;
    while i + 8 <= len {
        let yv = _mm256_load_ps(y.add(i));
        _mm256_store_ps(y.add(i), _mm256_mul_ps(yv, inv_sum_v));
        i += 8;
    }
    while i < len {
        *y.add(i) *= inv_sum;
        i += 1;
    }

    sum.ln() + max_val
}
