use wide::f64x4;

#[allow(dead_code)]
#[inline]
pub(crate) fn dot_wide(lhs: &[f64], rhs: &[f64]) -> f64 {
    let n = lhs.len().min(rhs.len());
    let mut acc = f64x4::ZERO;
    let mut i = 0usize;
    while i + 4 <= n {
        let a = f64x4::new([lhs[i], lhs[i + 1], lhs[i + 2], lhs[i + 3]]);
        let b = f64x4::new([rhs[i], rhs[i + 1], rhs[i + 2], rhs[i + 3]]);
        acc += a * b;
        i += 4;
    }
    let lanes = acc.to_array();
    let mut out = lanes[0] + lanes[1] + lanes[2] + lanes[3];
    while i < n {
        out += lhs[i] * rhs[i];
        i += 1;
    }
    out
}

#[allow(dead_code)]
#[inline]
pub(crate) fn max_wide(xs: &[f64]) -> f64 {
    if xs.is_empty() {
        return f64::NEG_INFINITY;
    }
    let mut i = 0usize;
    let mut max4 = f64x4::splat(f64::NEG_INFINITY);
    while i + 4 <= xs.len() {
        let v = f64x4::new([xs[i], xs[i + 1], xs[i + 2], xs[i + 3]]);
        max4 = max4.max(v);
        i += 4;
    }
    let lanes = max4.to_array();
    let mut max_v = lanes[0].max(lanes[1]).max(lanes[2]).max(lanes[3]);
    while i < xs.len() {
        if xs[i] > max_v {
            max_v = xs[i];
        }
        i += 1;
    }
    max_v
}

#[allow(dead_code)]
#[inline]
pub(crate) fn logsumexp_wide(xs: &[f64]) -> f64 {
    let max_v = max_wide(xs);
    if !max_v.is_finite() {
        return max_v;
    }
    let mut sum = 0.0;
    for &v in xs {
        sum += (v - max_v).exp();
    }
    max_v + sum.ln()
}

#[allow(dead_code)]
#[inline]
pub(crate) fn axpy_wide(dst: &mut [f64], alpha: f64, src: &[f64]) {
    let n = dst.len().min(src.len());
    let mut i = 0usize;
    let a4 = f64x4::splat(alpha);
    while i + 4 <= n {
        let d = f64x4::new([dst[i], dst[i + 1], dst[i + 2], dst[i + 3]]);
        let s = f64x4::new([src[i], src[i + 1], src[i + 2], src[i + 3]]);
        let r = d + a4 * s;
        let lanes = r.to_array();
        dst[i] = lanes[0];
        dst[i + 1] = lanes[1];
        dst[i + 2] = lanes[2];
        dst[i + 3] = lanes[3];
        i += 4;
    }
    while i < n {
        dst[i] += alpha * src[i];
        i += 1;
    }
}

#[allow(dead_code)]
#[inline]
pub(crate) fn affine3_wide(
    dst: &mut [f64],
    bias: &[f64],
    weights: [f64; 3],
    src0: &[f64],
    src1: &[f64],
    src2: &[f64],
) {
    let n = dst.len();
    assert!(bias.len() >= n, "bias shorter than dst");
    assert!(src0.len() >= n, "src0 shorter than dst");
    assert!(src1.len() >= n, "src1 shorter than dst");
    assert!(src2.len() >= n, "src2 shorter than dst");
    let mut i = 0usize;
    let w0 = f64x4::splat(weights[0]);
    let w1 = f64x4::splat(weights[1]);
    let w2 = f64x4::splat(weights[2]);
    while i + 4 <= n {
        let b = f64x4::new([bias[i], bias[i + 1], bias[i + 2], bias[i + 3]]);
        let x0 = f64x4::new([src0[i], src0[i + 1], src0[i + 2], src0[i + 3]]);
        let x1 = f64x4::new([src1[i], src1[i + 1], src1[i + 2], src1[i + 3]]);
        let x2 = f64x4::new([src2[i], src2[i + 1], src2[i + 2], src2[i + 3]]);
        let r = b + w0 * x0 + w1 * x1 + w2 * x2;
        let lanes = r.to_array();
        dst[i] = lanes[0];
        dst[i + 1] = lanes[1];
        dst[i + 2] = lanes[2];
        dst[i + 3] = lanes[3];
        i += 4;
    }
    while i < n {
        dst[i] = bias[i] + weights[0] * src0[i] + weights[1] * src1[i] + weights[2] * src2[i];
        i += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_close(lhs: f64, rhs: f64, tol: f64) {
        let delta: f64 = (lhs - rhs).abs();
        assert!(
            delta <= tol,
            "lhs={lhs}, rhs={rhs}, delta={delta}, tol={tol}"
        );
    }

    #[test]
    fn dot_and_axpy_cover_vector_and_scalar_tails() {
        let lhs: Vec<f64> = vec![1.0, -2.0, 3.0, 4.0, 5.0, 9.0];
        let rhs: Vec<f64> = vec![0.5, 2.0, -1.0, 0.25, -3.0];
        let dot: f64 = dot_wide(&lhs, &rhs);
        let expected_dot: f64 = lhs.iter().zip(rhs.iter()).map(|(a, b)| a * b).sum::<f64>();
        assert_close(dot, expected_dot, 1e-12);

        let mut dst: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let src: Vec<f64> = vec![10.0, -1.0, 2.0, 0.5, -4.0];
        axpy_wide(&mut dst, 0.25, &src);
        let expected_dst: Vec<f64> = vec![
            1.0 + 0.25 * 10.0,
            2.0 - 0.25,
            3.0 + 0.25 * 2.0,
            4.0 + 0.25 * 0.5,
            5.0 + 0.25 * -4.0,
            6.0,
        ];
        assert_eq!(dst, expected_dst);
    }

    #[test]
    fn max_and_logsumexp_cover_empty_and_finite_inputs() {
        assert_eq!(max_wide(&[]), f64::NEG_INFINITY);
        assert_eq!(logsumexp_wide(&[]), f64::NEG_INFINITY);

        let xs: Vec<f64> = vec![-3.0, -1.0, -2.0, -4.0, -0.5];
        assert_close(max_wide(&xs), -0.5, 1e-12);

        let max_v: f64 = xs.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let expected_lse: f64 = max_v + xs.iter().map(|v| (v - max_v).exp()).sum::<f64>().ln();
        assert_close(logsumexp_wide(&xs), expected_lse, 1e-12);
    }

    #[test]
    fn affine3_wide_matches_scalar_reference() {
        let bias: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let src0: Vec<f64> = vec![0.0, 1.0, 0.5, -1.0, 2.0, 1.5];
        let src1: Vec<f64> = vec![2.0, 0.0, -1.0, 1.0, 0.5, -0.5];
        let src2: Vec<f64> = vec![1.0, -2.0, 3.0, 0.5, -1.5, 2.5];
        let weights: [f64; 3] = [0.25, -0.5, 1.5];
        let mut dst: Vec<f64> = vec![0.0; bias.len()];

        affine3_wide(&mut dst, &bias, weights, &src0, &src1, &src2);

        let expected: Vec<f64> = (0..bias.len())
            .map(|i| bias[i] + weights[0] * src0[i] + weights[1] * src1[i] + weights[2] * src2[i])
            .collect();
        assert_eq!(dst, expected);
    }

    #[test]
    fn affine3_wide_panics_on_short_inputs() {
        let mut dst: Vec<f64> = vec![0.0; 4];
        let bias: Vec<f64> = vec![0.0; 4];
        let src: Vec<f64> = vec![0.0; 4];

        let short_bias = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            affine3_wide(&mut dst, &bias[..3], [1.0, 1.0, 1.0], &src, &src, &src);
        }));
        assert!(short_bias.is_err());

        let short_src0 = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            affine3_wide(&mut dst, &bias, [1.0, 1.0, 1.0], &src[..3], &src, &src);
        }));
        assert!(short_src0.is_err());

        let short_src1 = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            affine3_wide(&mut dst, &bias, [1.0, 1.0, 1.0], &src, &src[..3], &src);
        }));
        assert!(short_src1.is_err());

        let short_src2 = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            affine3_wide(&mut dst, &bias, [1.0, 1.0, 1.0], &src, &src, &src[..3]);
        }));
        assert!(short_src2.is_err());
    }
}
