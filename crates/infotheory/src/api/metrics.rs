//! Information-theoretic metric and scoring API surface.
//!
//! Functions in this module split cleanly into two families:
//!
//! - **Algorithmic**: take a [`CompiledRateBackend`] (or use the default context
//!   backend) and produce entropy-rate / cross-entropy / mutual-information /
//!   normalized-distance estimates driven by the rate model. Algorithm-specific
//!   parameters (such as ROSA's `max_order`) live inside the backend's variant.
//! - **Empirical** (`empirical_*`): order-0 / IID Shannon plug-in estimators.
//!   They treat the input as IID symbols and estimate entropy from observed
//!   symbol frequencies. They never take an order parameter; if higher-order
//!   structure matters, use a context-aware [`RateBackend`] (e.g. CTW).

use crate::error::{InfotheoryError, InfotheoryResult};
use crate::spec::CompiledRateBackend;

use crate::{aligned_prefix, with_default_ctx};

#[inline(always)]
/// Fallible entropy-rate estimate `Ĥ(X)` (bits per symbol) using the default context backend.
pub fn try_entropy_rate_bytes(data: &[u8]) -> InfotheoryResult<f64> {
    with_default_ctx(|ctx| ctx.try_entropy_rate_bytes(data))
}

#[inline(always)]
/// Fallible biased/plugin entropy-rate estimate using the default context backend.
pub fn try_biased_entropy_rate_bytes(data: &[u8]) -> InfotheoryResult<f64> {
    with_default_ctx(|ctx| ctx.try_biased_entropy_rate_bytes(data))
}

/// Mutual information rate estimate under an explicit `backend`.
///
/// Inputs are aligned to the shared prefix length.
pub fn try_mutual_information_rate_backend(
    x: &[u8],
    y: &[u8],
    backend: &CompiledRateBackend,
) -> InfotheoryResult<f64> {
    let (x, y) = aligned_prefix(x, y);
    if x.is_empty() {
        return Ok(0.0);
    }
    let h_x = try_entropy_rate_backend(x, backend)?;
    let h_y = try_entropy_rate_backend(y, backend)?;
    let h_xy = try_joint_entropy_rate_backend(x, y, backend)?;
    Ok((h_x + h_y - h_xy).max(0.0))
}

/// Normalized entropy distance under an explicit `backend`.
///
/// Returns a value in `[0, 1]` after clamping.
pub fn try_ned_rate_backend(
    x: &[u8],
    y: &[u8],
    backend: &CompiledRateBackend,
) -> InfotheoryResult<f64> {
    let (x, y) = aligned_prefix(x, y);
    if x.is_empty() {
        return Ok(0.0);
    }
    let h_x = try_entropy_rate_backend(x, backend)?;
    let h_y = try_entropy_rate_backend(y, backend)?;
    let h_xy = try_joint_entropy_rate_backend(x, y, backend)?;
    let min_h = h_x.min(h_y);
    let max_h = h_x.max(h_y);
    if max_h == 0.0 {
        Ok(0.0)
    } else {
        Ok(((h_xy - min_h) / max_h).clamp(0.0, 1.0))
    }
}

/// Normalized transform effort (variation-of-information form) under an explicit `backend`.
///
/// Returns a value in `[0, 2]` after clamping.
pub fn try_nte_rate_backend(
    x: &[u8],
    y: &[u8],
    backend: &CompiledRateBackend,
) -> InfotheoryResult<f64> {
    let (x, y) = aligned_prefix(x, y);
    if x.is_empty() {
        return Ok(0.0);
    }
    let h_x = try_entropy_rate_backend(x, backend)?;
    let h_y = try_entropy_rate_backend(y, backend)?;
    let h_xy = try_joint_entropy_rate_backend(x, y, backend)?;
    let max_h = h_x.max(h_y);
    if max_h == 0.0 {
        Ok(0.0)
    } else {
        let vi = (h_xy - h_x).max(0.0) + (h_xy - h_y).max(0.0);
        Ok((vi / max_h).clamp(0.0, 2.0))
    }
}

/// Fallible entropy-rate estimate of `data` using the explicit rate `backend`.
pub fn try_entropy_rate_backend(
    data: &[u8],
    backend: &CompiledRateBackend,
) -> InfotheoryResult<f64> {
    crate::runtime::try_entropy_rate_backend_direct(data, backend)
}

/// Fallible biased/plugin entropy rate of `data` using the explicit rate `backend`.
pub fn try_biased_entropy_rate_backend(
    data: &[u8],
    backend: &CompiledRateBackend,
) -> InfotheoryResult<f64> {
    if !backend.capabilities().supports_biased_entropy {
        Err(InfotheoryError::unsupported(
            "biased/plugin entropy is not supported for zpaq rate backends",
        ))
    } else {
        crate::try_frozen_plugin_rate_backend(data, &[data], backend)
    }
}

/// Fallible cross-entropy `H_{train}(test)` — score `test_data` under a model trained on `train_data`.
pub fn try_cross_entropy_rate_backend(
    test_data: &[u8],
    train_data: &[u8],
    backend: &CompiledRateBackend,
) -> InfotheoryResult<f64> {
    crate::runtime::try_cross_entropy_rate_backend_direct(test_data, train_data, backend)
}

/// Fallible joint entropy rate `H(X,Y)` using an explicit `backend`.
pub fn try_joint_entropy_rate_backend(
    x: &[u8],
    y: &[u8],
    backend: &CompiledRateBackend,
) -> InfotheoryResult<f64> {
    crate::runtime::try_joint_entropy_rate_backend_direct(x, y, backend)
}

/// Empirical (zero-order, IID) Shannon entropy `H₀(X)` in bits/symbol.
///
/// Treats the input as a sequence of IID byte symbols and returns the plug-in
/// Shannon entropy of the observed byte frequencies. Use a context-aware
/// [`crate::api::RateBackend`] via [`try_entropy_rate_bytes`] for higher-order estimation.
#[inline(always)]
pub fn empirical_entropy_bytes(data: &[u8]) -> f64 {
    if data.is_empty() {
        return 0.0;
    }

    let mut counts = [0u64; 256];
    for &b in data {
        counts[b as usize] += 1;
    }

    let n = data.len() as f64;
    let mut h = 0.0f64;
    for &count in &counts {
        if count > 0 {
            let p = count as f64 / n;
            h -= p * p.log2();
        }
    }
    h
}

/// Empirical (zero-order, IID) joint Shannon entropy `H₀(X,Y)` over aligned prefixes.
///
/// Treats aligned `(x[i], y[i])` pairs as IID samples from a joint distribution
/// over 65536 outcomes and returns the plug-in Shannon entropy.
#[inline(always)]
pub fn empirical_joint_entropy_bytes(x: &[u8], y: &[u8]) -> f64 {
    let (x, y) = aligned_prefix(x, y);
    let n = x.len();
    if n == 0 {
        return 0.0;
    }

    let mut counts = vec![0u64; 256 * 256];
    for i in 0..n {
        let pair_idx = (x[i] as usize) * 256 + (y[i] as usize);
        counts[pair_idx] += 1;
    }

    let n_f64 = n as f64;
    let mut h = 0.0f64;
    for &c in &counts {
        if c > 0 {
            let p = c as f64 / n_f64;
            h -= p * p.log2();
        }
    }
    h
}

#[inline(always)]
/// Fallible joint entropy-rate estimate `H(X,Y)` with the default context backend.
pub fn try_joint_entropy_rate_bytes(x: &[u8], y: &[u8]) -> InfotheoryResult<f64> {
    with_default_ctx(|ctx| ctx.try_joint_entropy_rate_bytes(x, y))
}

#[inline(always)]
/// Fallible conditional entropy-rate estimate `H(X|Y)` with the default context backend.
pub fn try_conditional_entropy_rate_bytes(x: &[u8], y: &[u8]) -> InfotheoryResult<f64> {
    with_default_ctx(|ctx| ctx.try_conditional_entropy_rate_bytes(x, y))
}

#[inline(always)]
/// Fallible conditional entropy estimate using the default context rate backend.
pub fn try_conditional_entropy_bytes(x: &[u8], y: &[u8]) -> InfotheoryResult<f64> {
    with_default_ctx(|ctx| ctx.try_conditional_entropy_bytes(x, y))
}

#[inline(always)]
/// Fallible mutual-information estimate `I(X;Y)` using the default context rate backend.
pub fn try_mutual_information_bytes(x: &[u8], y: &[u8]) -> InfotheoryResult<f64> {
    with_default_ctx(|ctx| ctx.try_mutual_information_bytes(x, y))
}

/// Empirical (zero-order, IID) mutual information `I₀(X;Y)` from byte histograms.
pub fn empirical_mutual_information_bytes(x: &[u8], y: &[u8]) -> f64 {
    let (x, y) = aligned_prefix(x, y);
    let h_x = empirical_entropy_bytes(x);
    let h_y = empirical_entropy_bytes(y);
    let h_xy = empirical_joint_entropy_bytes(x, y);
    (h_x + h_y - h_xy).max(0.0)
}

#[inline(always)]
/// Fallible mutual-information rate estimate with the default context backend.
pub fn try_mutual_information_rate_bytes(x: &[u8], y: &[u8]) -> InfotheoryResult<f64> {
    with_default_ctx(|ctx| ctx.try_mutual_information_rate_bytes(x, y))
}

#[inline(always)]
/// Fallible normalized entropy distance (NED) estimate with the default context backend.
pub fn try_ned_bytes(x: &[u8], y: &[u8]) -> InfotheoryResult<f64> {
    with_default_ctx(|ctx| ctx.try_ned_bytes(x, y))
}

/// Empirical (zero-order, IID) normalized entropy distance:
/// `(H₀(X,Y) - min(H₀(X), H₀(Y))) / max(H₀(X), H₀(Y))`.
pub fn empirical_ned_bytes(x: &[u8], y: &[u8]) -> f64 {
    let (x, y) = aligned_prefix(x, y);
    let h_x = empirical_entropy_bytes(x);
    let h_y = empirical_entropy_bytes(y);
    let h_xy = empirical_joint_entropy_bytes(x, y);
    let min_h = h_x.min(h_y);
    let max_h = h_x.max(h_y);
    if max_h == 0.0 {
        0.0
    } else {
        ((h_xy - min_h) / max_h).clamp(0.0, 1.0)
    }
}

#[inline(always)]
/// Fallible entropy-rate NED estimate with the default context backend.
pub fn try_ned_rate_bytes(x: &[u8], y: &[u8]) -> InfotheoryResult<f64> {
    with_default_ctx(|ctx| ctx.try_ned_bytes(x, y))
}

#[inline(always)]
/// Fallible constructive NED estimate with the default context backend.
pub fn try_ned_cons_bytes(x: &[u8], y: &[u8]) -> InfotheoryResult<f64> {
    with_default_ctx(|ctx| ctx.try_ned_cons_bytes(x, y))
}

/// Empirical (zero-order, IID) constructive normalized entropy distance:
/// `(H₀(X,Y) - min(H₀(X), H₀(Y))) / H₀(X,Y)`.
pub fn empirical_ned_cons_bytes(x: &[u8], y: &[u8]) -> f64 {
    let (x, y) = aligned_prefix(x, y);
    let h_x = empirical_entropy_bytes(x);
    let h_y = empirical_entropy_bytes(y);
    let h_xy = empirical_joint_entropy_bytes(x, y);
    let min_h = h_x.min(h_y);
    if h_xy == 0.0 {
        0.0
    } else {
        ((h_xy - min_h) / h_xy).clamp(0.0, 1.0)
    }
}

#[inline(always)]
/// Fallible entropy-rate constructive NED estimate with the default context backend.
pub fn try_ned_cons_rate_bytes(x: &[u8], y: &[u8]) -> InfotheoryResult<f64> {
    with_default_ctx(|ctx| ctx.try_ned_cons_bytes(x, y))
}

#[inline(always)]
/// Fallible normalized transform-effort (NTE/VI-based) estimate with the default context backend.
pub fn try_nte_bytes(x: &[u8], y: &[u8]) -> InfotheoryResult<f64> {
    with_default_ctx(|ctx| ctx.try_nte_bytes(x, y))
}

/// Empirical (zero-order, IID) NTE estimate using variation of information
/// normalized by `max(H₀(X), H₀(Y))`.
pub fn empirical_nte_bytes(x: &[u8], y: &[u8]) -> f64 {
    let (x, y) = aligned_prefix(x, y);
    let h_x = empirical_entropy_bytes(x);
    let h_y = empirical_entropy_bytes(y);
    let h_xy = empirical_joint_entropy_bytes(x, y);
    let vi = 2.0 * h_xy - h_x - h_y;
    let max_h = h_x.max(h_y);
    if max_h == 0.0 {
        0.0
    } else {
        (vi / max_h).clamp(0.0, 2.0)
    }
}

#[inline(always)]
/// Fallible entropy-rate NTE estimate with the default context backend.
pub fn try_nte_rate_bytes(x: &[u8], y: &[u8]) -> InfotheoryResult<f64> {
    with_default_ctx(|ctx| ctx.try_nte_bytes(x, y))
}

#[inline(always)]
pub(crate) fn byte_histogram(data: &[u8]) -> [f64; 256] {
    let mut counts = [0u64; 256];
    for &b in data {
        counts[b as usize] += 1;
    }
    let n = data.len() as f64;
    let mut probs = [0.0f64; 256];
    if n == 0.0 {
        return probs;
    }
    for i in 0..256 {
        probs[i] = counts[i] as f64 / n;
    }
    probs
}

#[inline(always)]
/// Total variation distance between the byte distributions of `x` and `y`.
///
/// TVD is an empirical, order-0 quantity by construction.
pub fn tvd_bytes(x: &[u8], y: &[u8]) -> f64 {
    if x.is_empty() || y.is_empty() {
        return 0.0;
    }
    let p_x = byte_histogram(x);
    let p_y = byte_histogram(y);

    let mut sum = 0.0f64;
    for i in 0..256 {
        sum += (p_x[i] - p_y[i]).abs();
    }

    (sum / 2.0).clamp(0.0, 1.0)
}

#[inline(always)]
/// Normalized Hellinger distance between the byte distributions of `x` and `y`.
///
/// NHD is an empirical, order-0 quantity by construction.
pub fn nhd_bytes(x: &[u8], y: &[u8]) -> f64 {
    if x.is_empty() || y.is_empty() {
        return 0.0;
    }
    let p_x = byte_histogram(x);
    let p_y = byte_histogram(y);

    let mut bc = 0.0f64;
    for i in 0..256 {
        bc += (p_x[i] * p_y[i]).sqrt();
    }

    (1.0 - bc).max(0.0).sqrt()
}

#[inline(always)]
/// Fallible cross-entropy estimate `H_train(test)` using the default context rate backend.
pub fn try_cross_entropy_bytes(test_data: &[u8], train_data: &[u8]) -> InfotheoryResult<f64> {
    with_default_ctx(|ctx| ctx.try_cross_entropy_bytes(test_data, train_data))
}

/// Empirical (zero-order, IID) cross-entropy `H₀_q(p) = -Σ p(x) log₂ q(x)` between
/// the byte histograms of `test_data` (treated as `p`) and `train_data` (treated as `q`).
pub fn empirical_cross_entropy_bytes(test_data: &[u8], train_data: &[u8]) -> f64 {
    if test_data.is_empty() {
        return 0.0;
    }
    let p_x = byte_histogram(test_data);
    let p_y = byte_histogram(train_data);
    let mut h = 0.0f64;
    for i in 0..256 {
        if p_x[i] > 0.0 {
            let q_y = p_y[i].max(1e-12);
            h -= p_x[i] * q_y.log2();
        }
    }
    h
}

#[inline(always)]
/// Fallible cross-entropy *rate* estimate with the default context backend.
pub fn try_cross_entropy_rate_bytes(test_data: &[u8], train_data: &[u8]) -> InfotheoryResult<f64> {
    with_default_ctx(|ctx| ctx.try_cross_entropy_rate_bytes(test_data, train_data))
}

/// KL divergence `D_KL(P || Q)` between the byte histograms of `x` and `y` (bits).
///
/// `D_KL` is an empirical, order-0 quantity by construction.
pub fn d_kl_bytes(x: &[u8], y: &[u8]) -> f64 {
    if x.is_empty() || y.is_empty() {
        return 0.0;
    }
    let p_x = byte_histogram(x);
    let p_y = byte_histogram(y);
    let mut d_kl = 0.0f64;
    for i in 0..256 {
        if p_x[i] > 0.0 {
            let q_y = p_y[i].max(1e-12);
            d_kl += p_x[i] * (p_x[i] / q_y).log2();
        }
    }
    d_kl.max(0.0)
}

/// Jensen-Shannon divergence between the byte histograms of `x` and `y` (bits).
///
/// JSD is an empirical, order-0 quantity by construction.
pub fn js_div_bytes(x: &[u8], y: &[u8]) -> f64 {
    if x.is_empty() || y.is_empty() {
        return 0.0;
    }
    let p_x = byte_histogram(x);
    let p_y = byte_histogram(y);
    let mut m = [0.0f64; 256];
    for i in 0..256 {
        m[i] = 0.5 * (p_x[i] + p_y[i]);
    }

    let mut kl_pm = 0.0f64;
    let mut kl_qm = 0.0f64;
    for i in 0..256 {
        if p_x[i] > 0.0 {
            kl_pm += p_x[i] * (p_x[i] / m[i]).log2();
        }
        if p_y[i] > 0.0 {
            kl_qm += p_y[i] * (p_y[i] / m[i]).log2();
        }
    }
    (0.5 * kl_pm + 0.5 * kl_qm).max(0.0)
}

#[inline(always)]
/// Fallible intrinsic dependence estimate:
/// `(H₀(X) - Ĥ(X)) / H₀(X)`, where `H₀` is the order-0 / empirical entropy and
/// `Ĥ` is the entropy rate produced by the default context's rate backend.
pub fn try_intrinsic_dependence_bytes(data: &[u8]) -> InfotheoryResult<f64> {
    with_default_ctx(|ctx| ctx.try_intrinsic_dependence_bytes(data))
}

#[inline(always)]
/// Fallible resistance-to-transformation estimate:
/// `I(X; T(X)) / H(X)` for `tx = T(x)`, using the default context rate backend.
pub fn try_resistance_to_transformation_bytes(x: &[u8], tx: &[u8]) -> InfotheoryResult<f64> {
    with_default_ctx(|ctx| ctx.try_resistance_to_transformation_bytes(x, tx))
}

/// Empirical (zero-order, IID) resistance-to-transformation estimate:
/// `I₀(X; T(X)) / H₀(X)` for `tx = T(x)`.
pub fn empirical_resistance_to_transformation_bytes(x: &[u8], tx: &[u8]) -> f64 {
    let (x, tx) = aligned_prefix(x, tx);
    let h_x = empirical_entropy_bytes(x);
    if h_x < 1e-9 {
        0.0
    } else {
        (empirical_mutual_information_bytes(x, tx) / h_x).clamp(0.0, 1.0)
    }
}
