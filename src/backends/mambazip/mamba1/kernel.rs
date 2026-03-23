//! Portable SIMD kernels for Mamba-1 CPU inference.

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

/// Dot product of two equal-length slices.
#[inline(always)]
pub unsafe fn dot(a: *const f32, b: *const f32, len: usize) -> f32 {
    let mut sum = f32x8::ZERO;
    let mut i = 0usize;
    while i + LANES <= len {
        // SAFETY: caller guarantees valid pointers and bounds.
        let av = unsafe { load8(a.add(i)) };
        // SAFETY: caller guarantees valid pointers and bounds.
        let bv = unsafe { load8(b.add(i)) };
        sum += av * bv;
        i += LANES;
    }
    let mut out = sum.reduce_add();
    while i < len {
        // SAFETY: caller guarantees valid pointers and bounds.
        out += unsafe { *a.add(i) * *b.add(i) };
        i += 1;
    }
    out
}

/// Matrix-vector multiply `y = A @ x`.
/// `A` is row-major with shape `(rows, cols)`.
#[inline(always)]
pub unsafe fn gemv(a: *const f32, x: *const f32, y: *mut f32, rows: usize, cols: usize) {
    let mut r = 0usize;
    while r + 4 <= rows {
        // SAFETY: caller guarantees valid pointers and matrix bounds.
        let row0 = unsafe { a.add(r * cols) };
        // SAFETY: caller guarantees valid pointers and matrix bounds.
        let row1 = unsafe { a.add((r + 1) * cols) };
        // SAFETY: caller guarantees valid pointers and matrix bounds.
        let row2 = unsafe { a.add((r + 2) * cols) };
        // SAFETY: caller guarantees valid pointers and matrix bounds.
        let row3 = unsafe { a.add((r + 3) * cols) };

        let mut sum0 = f32x8::ZERO;
        let mut sum1 = f32x8::ZERO;
        let mut sum2 = f32x8::ZERO;
        let mut sum3 = f32x8::ZERO;

        let mut c = 0usize;
        while c + LANES <= cols {
            // SAFETY: caller guarantees valid pointers and bounds.
            let xv = unsafe { load8(x.add(c)) };
            // SAFETY: caller guarantees valid pointers and bounds.
            sum0 += unsafe { load8(row0.add(c)) } * xv;
            // SAFETY: caller guarantees valid pointers and bounds.
            sum1 += unsafe { load8(row1.add(c)) } * xv;
            // SAFETY: caller guarantees valid pointers and bounds.
            sum2 += unsafe { load8(row2.add(c)) } * xv;
            // SAFETY: caller guarantees valid pointers and bounds.
            sum3 += unsafe { load8(row3.add(c)) } * xv;
            c += LANES;
        }

        let mut out0 = sum0.reduce_add();
        let mut out1 = sum1.reduce_add();
        let mut out2 = sum2.reduce_add();
        let mut out3 = sum3.reduce_add();

        while c < cols {
            // SAFETY: caller guarantees valid pointers and bounds.
            let xv = unsafe { *x.add(c) };
            // SAFETY: caller guarantees valid pointers and bounds.
            out0 += unsafe { *row0.add(c) } * xv;
            // SAFETY: caller guarantees valid pointers and bounds.
            out1 += unsafe { *row1.add(c) } * xv;
            // SAFETY: caller guarantees valid pointers and bounds.
            out2 += unsafe { *row2.add(c) } * xv;
            // SAFETY: caller guarantees valid pointers and bounds.
            out3 += unsafe { *row3.add(c) } * xv;
            c += 1;
        }

        // SAFETY: caller guarantees output pointer capacity.
        unsafe {
            *y.add(r) = out0;
            *y.add(r + 1) = out1;
            *y.add(r + 2) = out2;
            *y.add(r + 3) = out3;
        }
        r += 4;
    }

    while r < rows {
        // SAFETY: caller guarantees valid pointers and bounds.
        unsafe { *y.add(r) = dot(a.add(r * cols), x, cols) };
        r += 1;
    }
}

/// Matrix-vector multiply with transposed matrix: `y = A^T @ x`.
/// `A` is row-major with shape `(rows, cols)`, output `y` has length `cols`.
#[inline(always)]
pub unsafe fn gemv_t(a: *const f32, x: *const f32, y: *mut f32, rows: usize, cols: usize) {
    let mut c = 0usize;
    while c + LANES <= cols {
        // SAFETY: caller guarantees output pointer capacity.
        unsafe { store8(y.add(c), f32x8::ZERO) };
        c += LANES;
    }
    while c < cols {
        // SAFETY: caller guarantees output pointer capacity.
        unsafe { *y.add(c) = 0.0 };
        c += 1;
    }

    for r in 0..rows {
        // SAFETY: caller guarantees input pointer capacity.
        let xr = f32x8::splat(unsafe { *x.add(r) });
        // SAFETY: caller guarantees matrix pointer bounds.
        let row = unsafe { a.add(r * cols) };
        let mut c = 0usize;
        while c + LANES <= cols {
            // SAFETY: caller guarantees pointer bounds.
            let yv = unsafe { load8(y.add(c)) };
            // SAFETY: caller guarantees pointer bounds.
            let av = unsafe { load8(row.add(c)) };
            // SAFETY: caller guarantees output pointer capacity.
            unsafe { store8(y.add(c), yv + av * xr) };
            c += LANES;
        }
        while c < cols {
            // SAFETY: caller guarantees pointer bounds.
            unsafe { *y.add(c) += *row.add(c) * *x.add(r) };
            c += 1;
        }
    }
}

/// Vector add in-place: `a += b`.
#[inline(always)]
pub unsafe fn add_inplace(a: *mut f32, b: *const f32, len: usize) {
    let mut i = 0usize;
    while i + LANES <= len {
        // SAFETY: caller guarantees valid pointers and bounds.
        let av = unsafe { load8(a.add(i)) };
        // SAFETY: caller guarantees valid pointers and bounds.
        let bv = unsafe { load8(b.add(i)) };
        // SAFETY: caller guarantees valid pointers and bounds.
        unsafe { store8(a.add(i), av + bv) };
        i += LANES;
    }
    while i < len {
        // SAFETY: caller guarantees valid pointers and bounds.
        unsafe {
            *a.add(i) += *b.add(i);
        }
        i += 1;
    }
}
