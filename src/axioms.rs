//! # Axioms: Mathematical Property Verifiers
//!
//! This module provides generic functions to verify mathematical properties
//! that should hold for any correct implementation of information-theoretic
//! measures.

// ============================================================================
// Metric Axioms
// ============================================================================

/// Verify that a distance function d(x,x) is close to 0 (allow for small overhead).
pub fn verify_identity<F>(metric: F, x: &[u8], tolerance: f64) -> bool
where
    F: Fn(&[u8], &[u8]) -> f64,
{
    let d = metric(x, x);
    d.abs() <= tolerance
}

/// Verify symmetry: d(x,y) ≈ d(y,x).
pub fn verify_symmetry<F>(metric: F, x: &[u8], y: &[u8], tolerance: f64) -> bool
where
    F: Fn(&[u8], &[u8]) -> f64,
{
    let d_xy = metric(x, y);
    let d_yx = metric(y, x);
    (d_xy - d_yx).abs() <= tolerance
}

/// Verify triangle inequality: d(x,z) ≤ d(x,y) + d(y,z).
pub fn verify_triangle_inequality<F>(
    metric: F,
    x: &[u8],
    y: &[u8],
    z: &[u8],
    tolerance: f64,
) -> bool
where
    F: Fn(&[u8], &[u8]) -> f64,
{
    let d_xy = metric(x, y);
    let d_yz = metric(y, z);
    let d_xz = metric(x, z);
    d_xz <= (d_xy + d_yz + tolerance)
}

/// Verify non-negativity: d(x,y) ≥ 0.
pub fn verify_non_negativity<F>(metric: F, x: &[u8], y: &[u8]) -> bool
where
    F: Fn(&[u8], &[u8]) -> f64,
{
    // Allow tiny floating point errors slightly below zero
    metric(x, y) >= -1e-12
}

// ============================================================================
// Information Inequalities
// ============================================================================

/// Verify mutual information non-negativity: I(X;Y) ≥ 0.
pub fn verify_mi_nonnegative<F>(mi: F, x: &[u8], y: &[u8]) -> bool
where
    F: Fn(&[u8], &[u8]) -> f64,
{
    mi(x, y) >= -1e-12
}

/// Verify subadditivity: H(X,Y) ≤ H(X) + H(Y).
///
/// This is equivalent to I(X;Y) ≥ 0.
pub fn verify_subadditivity<FJoint, FMarg>(
    joint_entropy: FJoint,
    marginal_entropy: FMarg,
    x: &[u8],
    y: &[u8],
    tolerance: f64,
) -> bool
where
    FJoint: Fn(&[u8], &[u8]) -> f64,
    FMarg: Fn(&[u8]) -> f64,
{
    let h_xy = joint_entropy(x, y);
    let h_x = marginal_entropy(x);
    let h_y = marginal_entropy(y);
    h_xy <= (h_x + h_y + tolerance)
}

/// Verify conditioning reduces entropy: H(X|Y) ≤ H(X).
pub fn verify_conditioning_reduces_entropy<FCond, FMarg>(
    conditional_entropy: FCond,
    marginal_entropy: FMarg,
    x: &[u8],
    y: &[u8],
    tolerance: f64,
) -> bool
where
    FCond: Fn(&[u8], &[u8]) -> f64,
    FMarg: Fn(&[u8]) -> f64,
{
    let h_x_given_y = conditional_entropy(x, y);
    let h_x = marginal_entropy(x);
    h_x_given_y <= (h_x + tolerance)
}

/// Verify chain rule: H(X,Y) = H(X) + H(Y|X).
pub fn verify_chain_rule<FJoint, FMarg, FCond>(
    joint: FJoint,
    marginal: FMarg,
    conditional: FCond,
    x: &[u8],
    y: &[u8],
    tolerance: f64,
) -> bool
where
    FJoint: Fn(&[u8], &[u8]) -> f64,
    FMarg: Fn(&[u8]) -> f64,
    FCond: Fn(&[u8], &[u8]) -> f64, // H(Y|X)
{
    let h_xy = joint(x, y);
    let h_x = marginal(x);
    let h_y_given_x = conditional(y, x);

    (h_xy - (h_x + h_y_given_x)).abs() <= tolerance
}

// ============================================================================
// Bounds
// ============================================================================

/// Verify NCD range: 0 ≤ NCD ≤ 1+epsilon.
///
/// NCD theoretically can slightly exceed 1 due to compression overhead, so we allow
/// a small margin or just check it's not egregiously large. Usually NCD <= 1.1 is safe.
pub fn verify_ncd_bounds<F>(ncd: F, x: &[u8], y: &[u8]) -> bool
where
    F: Fn(&[u8], &[u8]) -> f64,
{
    let val = ncd(x, y);
    val >= -1e-12 && val <= 1.1
}

/// Verify entropy is bounded by log2(alphabet_size).
/// For bytes, max entropy is 8.0 bits/byte.
pub fn verify_entropy_bounds<F>(entropy: F, data: &[u8]) -> bool
where
    F: Fn(&[u8]) -> f64,
{
    let h = entropy(data);
    h >= -1e-12 && h <= 8.0 + 1e-12
}
