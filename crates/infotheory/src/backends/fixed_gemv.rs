//! Shared crate-private fixed-shape GEMV specializations for neural backends.
//!
//! The current consumer is RWKV7. Shapes are intentionally limited to the
//! benchmark-stable hot path so we can demand a strict binary-size gate.

use wide::f32x8;

const LANES: usize = 8;

#[inline(always)]
unsafe fn load8(ptr: *const f32) -> f32x8 {
    unsafe { ptr.cast::<f32x8>().read_unaligned() }
}

#[inline(always)]
unsafe fn store8(ptr: *mut f32, v: f32x8) {
    unsafe { ptr.cast::<f32x8>().write_unaligned(v) }
}

#[inline(always)]
unsafe fn gemv_fixed<const ROWS: usize, const CHUNKS: usize>(
    a: *const f32,
    x: *const f32,
    y: *mut f32,
) {
    const ROW_BATCH: usize = 4;
    const fn cols_for<const CHUNKS: usize>() -> usize {
        CHUNKS * LANES
    }
    let cols = cols_for::<CHUNKS>();
    let mut r = 0usize;
    while r < ROWS {
        let row0 = unsafe { a.add(r * cols) };
        let row1 = unsafe { a.add((r + 1) * cols) };
        let row2 = unsafe { a.add((r + 2) * cols) };
        let row3 = unsafe { a.add((r + 3) * cols) };

        let mut sum0 = f32x8::ZERO;
        let mut sum1 = f32x8::ZERO;
        let mut sum2 = f32x8::ZERO;
        let mut sum3 = f32x8::ZERO;

        let mut c = 0usize;
        while c < cols {
            let xv = unsafe { load8(x.add(c)) };
            sum0 += unsafe { load8(row0.add(c)) } * xv;
            sum1 += unsafe { load8(row1.add(c)) } * xv;
            sum2 += unsafe { load8(row2.add(c)) } * xv;
            sum3 += unsafe { load8(row3.add(c)) } * xv;
            c += LANES;
        }

        unsafe {
            *y.add(r) = sum0.reduce_add();
            *y.add(r + 1) = sum1.reduce_add();
            *y.add(r + 2) = sum2.reduce_add();
            *y.add(r + 3) = sum3.reduce_add();
        }
        r += ROW_BATCH;
    }
}

#[inline(always)]
unsafe fn gemv_t_fixed<const ROWS: usize, const CHUNKS: usize>(
    a: *const f32,
    x: *const f32,
    y: *mut f32,
) {
    const fn cols_for<const CHUNKS: usize>() -> usize {
        CHUNKS * LANES
    }
    let cols = cols_for::<CHUNKS>();
    let mut c = 0usize;
    while c < cols {
        unsafe { store8(y.add(c), f32x8::ZERO) };
        c += LANES;
    }

    let mut r = 0usize;
    while r < ROWS {
        let x_r = f32x8::splat(unsafe { *x.add(r) });
        let row = unsafe { a.add(r * cols) };
        let mut c = 0usize;
        while c < cols {
            let yv = unsafe { load8(y.add(c)) };
            let av = unsafe { load8(row.add(c)) };
            unsafe { store8(y.add(c), yv + av * x_r) };
            c += LANES;
        }
        r += 1;
    }
}

#[inline(always)]
pub(crate) unsafe fn try_gemv(
    a: *const f32,
    x: *const f32,
    y: *mut f32,
    rows: usize,
    cols: usize,
) -> bool {
    match (rows, cols) {
        (256, 64) => unsafe { gemv_fixed::<256, 8>(a, x, y) },
        (64, 64) => unsafe { gemv_fixed::<64, 8>(a, x, y) },
        (16, 64) => unsafe { gemv_fixed::<16, 8>(a, x, y) },
        (64, 16) => unsafe { gemv_fixed::<64, 2>(a, x, y) },
        _ => return false,
    }
    true
}

#[inline(always)]
pub(crate) unsafe fn try_gemv_t(
    a: *const f32,
    x: *const f32,
    y: *mut f32,
    rows: usize,
    cols: usize,
) -> bool {
    match (rows, cols) {
        (256, 64) => unsafe { gemv_t_fixed::<256, 8>(a, x, y) },
        (64, 64) => unsafe { gemv_t_fixed::<64, 8>(a, x, y) },
        (16, 64) => unsafe { gemv_t_fixed::<16, 8>(a, x, y) },
        (64, 16) => unsafe { gemv_t_fixed::<64, 2>(a, x, y) },
        _ => return false,
    }
    true
}

#[cfg(test)]
mod tests {
    use super::{try_gemv, try_gemv_t};

    #[derive(Clone, Copy)]
    struct Lcg(u64);

    impl Lcg {
        fn new(seed: u64) -> Self {
            Self(seed)
        }

        fn next_f32(&mut self) -> f32 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let bits = ((self.0 >> 40) as u32) | 0x3f80_0000;
            f32::from_bits(bits) - 1.0
        }
    }

    fn fill_centered(dst: &mut [f32], rng: &mut Lcg, scale: f32) {
        for x in dst {
            *x = (rng.next_f32() * 2.0 - 1.0) * scale;
        }
    }

    fn gemv_scalar(a: &[f32], x: &[f32], rows: usize, cols: usize) -> Vec<f32> {
        let mut y = vec![0.0; rows];
        for r in 0..rows {
            let row = &a[r * cols..(r + 1) * cols];
            let mut acc = 0.0f32;
            for c in 0..cols {
                acc += row[c] * x[c];
            }
            y[r] = acc;
        }
        y
    }

    fn gemv_t_scalar(a: &[f32], x: &[f32], rows: usize, cols: usize) -> Vec<f32> {
        let mut y = vec![0.0; cols];
        for r in 0..rows {
            let row = &a[r * cols..(r + 1) * cols];
            for c in 0..cols {
                y[c] += row[c] * x[r];
            }
        }
        y
    }

    fn assert_close(lhs: &[f32], rhs: &[f32], tol: f32) {
        assert_eq!(lhs.len(), rhs.len());
        for idx in 0..lhs.len() {
            let diff = (lhs[idx] - rhs[idx]).abs();
            assert!(
                diff <= tol,
                "mismatch at {idx}: lhs={} rhs={} diff={diff}",
                lhs[idx],
                rhs[idx]
            );
        }
    }

    fn check_shape(rows: usize, cols: usize) {
        let mut rng = Lcg::new(((rows as u64) << 32) ^ (cols as u64) ^ 0xC0FFEEu64);
        let mut a = vec![0.0; rows * cols];
        let mut x = vec![0.0; cols];
        let mut xt = vec![0.0; rows];
        fill_centered(&mut a, &mut rng, 0.8);
        fill_centered(&mut x, &mut rng, 0.5);
        fill_centered(&mut xt, &mut rng, 0.4);

        let mut y = vec![0.0; rows];
        assert!(unsafe { try_gemv(a.as_ptr(), x.as_ptr(), y.as_mut_ptr(), rows, cols) });
        let y_ref = gemv_scalar(&a, &x, rows, cols);
        assert_close(&y, &y_ref, 2.5e-5);

        let mut yt = vec![0.0; cols];
        assert!(unsafe { try_gemv_t(a.as_ptr(), xt.as_ptr(), yt.as_mut_ptr(), rows, cols) });
        let yt_ref = gemv_t_scalar(&a, &xt, rows, cols);
        assert_close(&yt, &yt_ref, 2.5e-5);
    }

    #[test]
    fn fixed_shapes_match_scalar_reference() {
        check_shape(256, 64);
        check_shape(64, 64);
        check_shape(16, 64);
        check_shape(64, 16);
    }

    #[test]
    fn unsupported_shapes_fall_back() {
        let a = vec![0.0; 11 * 37];
        let x = vec![0.0; 37];
        let xt = vec![0.0; 11];
        let mut y = vec![0.0; 11];
        let mut yt = vec![0.0; 37];
        assert!(!unsafe { try_gemv(a.as_ptr(), x.as_ptr(), y.as_mut_ptr(), 11, 37) });
        assert!(!unsafe { try_gemv_t(a.as_ptr(), xt.as_ptr(), yt.as_mut_ptr(), 11, 37) });
    }
}
