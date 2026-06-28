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
pub fn verify_subadditivity<FJoint, FEmpirical>(
    joint_entropy: FJoint,
    empirical_entropy: FEmpirical,
    x: &[u8],
    y: &[u8],
    tolerance: f64,
) -> bool
where
    FJoint: Fn(&[u8], &[u8]) -> f64,
    FEmpirical: Fn(&[u8]) -> f64,
{
    let h_xy = joint_entropy(x, y);
    let h_x = empirical_entropy(x);
    let h_y = empirical_entropy(y);
    h_xy <= (h_x + h_y + tolerance)
}

/// Verify conditioning reduces entropy: H(X|Y) ≤ H(X).
pub fn verify_conditioning_reduces_entropy<FCond, FEmpirical>(
    conditional_entropy: FCond,
    empirical_entropy: FEmpirical,
    x: &[u8],
    y: &[u8],
    tolerance: f64,
) -> bool
where
    FCond: Fn(&[u8], &[u8]) -> f64,
    FEmpirical: Fn(&[u8]) -> f64,
{
    let h_x_given_y = conditional_entropy(x, y);
    let h_x = empirical_entropy(x);
    h_x_given_y <= (h_x + tolerance)
}

/// Verify chain rule: H(X,Y) = H(X) + H(Y|X).
pub fn verify_chain_rule<FJoint, FEmpirical, FCond>(
    joint: FJoint,
    empirical: FEmpirical,
    conditional: FCond,
    x: &[u8],
    y: &[u8],
    tolerance: f64,
) -> bool
where
    FJoint: Fn(&[u8], &[u8]) -> f64,
    FEmpirical: Fn(&[u8]) -> f64,
    FCond: Fn(&[u8], &[u8]) -> f64, // H(Y|X)
{
    let h_xy = joint(x, y);
    let h_x = empirical(x);
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
    (-1e-12..=1.1).contains(&val)
}

/// Verify entropy is bounded by log2(alphabet_size).
/// For bytes, max entropy is 8.0 bits/byte.
pub fn verify_entropy_bounds<F>(entropy: F, data: &[u8]) -> bool
where
    F: Fn(&[u8]) -> f64,
{
    let h = entropy(data);
    (-1e-12..=8.0 + 1e-12).contains(&h)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hamming_distance(x: &[u8], y: &[u8]) -> f64 {
        x.iter().zip(y.iter()).filter(|(a, b)| a != b).count() as f64
    }

    #[test]
    fn metric_axiom_verifiers_distinguish_valid_from_invalid_cases() {
        let x = b"abc";
        let y = b"abd";
        let z = b"acd";

        assert!(verify_identity(hamming_distance, x, 0.0));
        assert!(!verify_identity(|_, _| 0.2, x, 0.1));

        assert!(verify_symmetry(hamming_distance, x, y, 0.0));
        assert!(!verify_symmetry(
            |lhs, rhs| {
                if lhs == rhs {
                    0.0
                } else if lhs == x && rhs == y {
                    1.0
                } else {
                    3.0
                }
            },
            x,
            y,
            0.0
        ));

        assert!(verify_triangle_inequality(hamming_distance, x, y, z, 0.0));
        assert!(!verify_triangle_inequality(
            |lhs, rhs| { if lhs == x && rhs == z { 5.0 } else { 1.0 } },
            x,
            y,
            z,
            0.0
        ));

        assert!(verify_non_negativity(hamming_distance, x, y));
        assert!(verify_non_negativity(|_, _| -5e-13, x, y));
        assert!(!verify_non_negativity(|_, _| -1e-6, x, y));
    }

    #[test]
    fn information_inequality_verifiers_cover_positive_and_negative_examples() {
        let x = b"left";
        let y = b"right";

        let empirical = |data: &[u8]| data.len() as f64;
        let joint = |lhs: &[u8], rhs: &[u8]| (lhs.len() + rhs.len()) as f64 - 0.5;
        let conditional = |lhs: &[u8], _rhs: &[u8]| lhs.len() as f64 - 0.25;

        assert!(verify_mi_nonnegative(|_, _| 0.0, x, y));
        assert!(!verify_mi_nonnegative(|_, _| -1e-6, x, y));

        assert!(verify_subadditivity(joint, empirical, x, y, 0.0));
        assert!(!verify_subadditivity(
            |lhs, rhs| (lhs.len() + rhs.len()) as f64 + 2.0,
            empirical,
            x,
            y,
            0.0
        ));

        assert!(verify_conditioning_reduces_entropy(
            conditional,
            empirical,
            x,
            y,
            0.0
        ));
        assert!(!verify_conditioning_reduces_entropy(
            |lhs, _rhs| lhs.len() as f64 + 1.0,
            empirical,
            x,
            y,
            0.0
        ));

        assert!(verify_chain_rule(joint, empirical, conditional, x, y, 0.5));
        assert!(!verify_chain_rule(
            |lhs, rhs| (lhs.len() + rhs.len()) as f64 + 3.0,
            empirical,
            conditional,
            x,
            y,
            0.0
        ));
    }

    #[test]
    fn bound_verifiers_allow_expected_slack_only() {
        let x = b"x";
        let y = b"y";

        assert!(verify_ncd_bounds(|_, _| 0.0, x, y));
        assert!(verify_ncd_bounds(|_, _| 1.1, x, y));
        assert!(!verify_ncd_bounds(|_, _| 1.100_001, x, y));
        assert!(!verify_ncd_bounds(|_, _| -1e-6, x, y));

        assert!(verify_entropy_bounds(|_| 0.0, x));
        assert!(verify_entropy_bounds(|_| 8.0, x));
        assert!(verify_entropy_bounds(|_| -5e-13, x));
        assert!(!verify_entropy_bounds(|_| 8.1, x));
        assert!(!verify_entropy_bounds(|_| -1e-6, x));
    }
}
