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
/// Fallible bitwise intrinsic dependence:
/// `(H₀,bits(X) - Ĥ_per_bit(X)) / H₀,bits(X)`.
///
/// See [`InfotheoryCtx::try_intrinsic_dependence_bits`] for semantics: this is a
/// true binary-alphabet framing of ID, not a rescaling of
/// [`try_intrinsic_dependence_bytes`].
pub fn try_intrinsic_dependence_bits(data: &[u8]) -> InfotheoryResult<f64> {
    with_default_ctx(|ctx| ctx.try_intrinsic_dependence_bits(data))
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

/// Pooled marginal bit histogram: `[P(bit=0), P(bit=1)]` over all `8N` bits.
///
/// Bits from every byte are pooled together without regard for their position
/// within the byte. This computes a single marginal `P(bit=1)` across all bit
/// positions rather than a per-position histogram. On structured data (e.g.
/// ASCII text where bit 7 of every byte is almost always 0), the resulting
/// entropy will differ from any position-stratified estimate. This is the
/// correct interpretation for a flat binary-alphabet IID model.
#[inline(always)]
pub(crate) fn bit_histogram(data: &[u8]) -> [f64; 2] {
    let count_1: u64 = data.iter().map(|&b| u64::from(b.count_ones())).sum();
    // Cast to u64 before scaling so 32-bit `usize` lengths cannot overflow
    let total = (data.len() as u64) * 8;
    if total == 0 {
        return [0.0, 0.0];
    }
    let p1 = count_1 as f64 / total as f64;
    [1.0 - p1, p1]
}

/// Pooled joint bit histogram over aligned pairs: `[P(0,0), P(0,1), P(1,0), P(1,1)]`.
///
/// Iterates over corresponding byte pairs `(bx, by)` and accumulates joint bit
/// counts using three bitwise-AND popcounts per pair — no inner shift loop.
/// Index encoding: index `= (bit_x << 1) | bit_y`.
#[inline(always)]
pub(crate) fn joint_bit_histogram(x: &[u8], y: &[u8]) -> [f64; 4] {
    let (x, y) = aligned_prefix(x, y);
    let (mut c11, mut c10, mut c01) = (0u64, 0u64, 0u64);
    for (&bx, &by) in x.iter().zip(y.iter()) {
        c11 += u64::from((bx & by).count_ones());
        c10 += u64::from((bx & !by).count_ones());
        c01 += u64::from((!bx & by).count_ones());
    }
    // Cast before scaling so 32-bit `usize` lengths cannot overflow when
    // converting byte count to bit count.
    let total = (x.len() as u64) * 8;
    let c00 = total - c11 - c10 - c01;
    let total_f64 = total as f64;
    let mut probs = [0.0f64; 4];
    if total_f64 > 0.0 {
        // index = (bit_x << 1) | bit_y: 0=>(0,0), 1=>(0,1), 2=>(1,0), 3=>(1,1)
        probs[0] = c00 as f64 / total_f64;
        probs[1] = c01 as f64 / total_f64;
        probs[2] = c10 as f64 / total_f64;
        probs[3] = c11 as f64 / total_f64;
    }
    probs
}

// ─── Algorithmic: per-bit (unit-conversion wrappers) ─────────────────────────
//
// All `_per_bit` functions re-express the same total base-2 log-loss `L` in
// bits-per-bit instead of bits-per-byte. The relationship `rate_per_bit =
// rate_per_byte / 8` is an exact unit conversion valid because all algorithmic
// estimators in this crate return log-loss in base-2 bits (not nats).
//
// For normalized ratios (NED, NTE) the scale factor cancels; the wrappers
// forward directly to the corresponding `_bytes` function and return the same
// numerical value. They are present for API uniformity so callers using the
// bitwise surface never need per-function awareness of which measures are
// scale-invariant.

/// Fallible entropy-rate estimate in bits per bit using the default context backend.
///
/// Returns `try_entropy_rate_bytes(data).map(|v| v / 8.0)`.
pub fn try_entropy_rate_per_bit(data: &[u8]) -> InfotheoryResult<f64> {
    try_entropy_rate_bytes(data).map(|v| v / 8.0)
}

/// Fallible biased/plugin entropy rate in bits per bit using the default context backend.
pub fn try_biased_entropy_rate_per_bit(data: &[u8]) -> InfotheoryResult<f64> {
    try_biased_entropy_rate_bytes(data).map(|v| v / 8.0)
}

/// Fallible joint entropy rate in bits per bit using the default context backend.
pub fn try_joint_entropy_rate_per_bit(x: &[u8], y: &[u8]) -> InfotheoryResult<f64> {
    try_joint_entropy_rate_bytes(x, y).map(|v| v / 8.0)
}

/// Fallible cross-entropy rate in bits per bit using the default context backend.
pub fn try_cross_entropy_rate_per_bit(
    test_data: &[u8],
    train_data: &[u8],
) -> InfotheoryResult<f64> {
    try_cross_entropy_rate_bytes(test_data, train_data).map(|v| v / 8.0)
}

/// Fallible mutual information rate in bits per bit using the default context backend.
pub fn try_mutual_information_rate_per_bit(x: &[u8], y: &[u8]) -> InfotheoryResult<f64> {
    try_mutual_information_rate_bytes(x, y).map(|v| v / 8.0)
}

/// Fallible conditional entropy rate in bits per bit using the default context backend.
pub fn try_conditional_entropy_rate_per_bit(x: &[u8], y: &[u8]) -> InfotheoryResult<f64> {
    try_conditional_entropy_rate_bytes(x, y).map(|v| v / 8.0)
}

/// Fallible NED estimate using the default context backend.
///
/// NED is a dimensionless ratio; the `/8` scale cancels. This wrapper is
/// provided for naming consistency with the bitwise surface.
#[inline(always)]
pub fn try_ned_rate_per_bit(x: &[u8], y: &[u8]) -> InfotheoryResult<f64> {
    try_ned_bytes(x, y)
}

/// Fallible constructive NED estimate using the default context backend.
///
/// Constructive NED is a dimensionless ratio; the `/8` scale cancels.
#[inline(always)]
pub fn try_ned_cons_rate_per_bit(x: &[u8], y: &[u8]) -> InfotheoryResult<f64> {
    try_ned_cons_bytes(x, y)
}

/// Fallible NTE estimate using the default context backend.
///
/// NTE is a dimensionless ratio; the `/8` scale cancels. This wrapper is
/// provided for naming consistency with the bitwise surface.
#[inline(always)]
pub fn try_nte_rate_per_bit(x: &[u8], y: &[u8]) -> InfotheoryResult<f64> {
    try_nte_bytes(x, y)
}

/// Fallible resistance-to-transformation estimate using the default context backend.
///
/// Resistance is the dimensionless ratio `I(X;T(X)) / H(X)`; the `/8` scale
/// cancels. This wrapper forwards to `try_resistance_to_transformation_bytes` (algorithmic) and is provided for naming consistency with the bitwise surface.
#[inline(always)]
pub fn try_resistance_to_transformation_per_bit(x: &[u8], tx: &[u8]) -> InfotheoryResult<f64> {
    try_resistance_to_transformation_bytes(x, tx)
}

// ─── Empirical: bitwise (order-0, alphabet size 2) ───────────────────────────
//
// All `_bits` empirical functions treat the input as a flat sequence of `8N`
// independent binary symbols drawn from the alphabet {0, 1}. Because bits
// within a byte are correlated in real data, these quantities are genuinely
// different from their `_bytes` counterparts — not a simple rescaling.

/// Shannon entropy of a finite probability vector: `−Σᵢ pᵢ log₂ pᵢ` (zero bins skipped).
///
/// Accepts any size; callers pass `[f64; 2]` or `[f64; 4]` fixed arrays via
/// implicit coercion to `&[f64]`.
#[inline]
fn shannon_h(p: &[f64]) -> f64 {
    let mut h = 0.0f64;
    for &pi in p {
        if pi > 0.0 {
            h -= pi * pi.log2();
        }
    }
    h
}

/// Empirical (zero-order, IID) Shannon entropy `H₀(X)` in bits/bit over a binary alphabet.
///
/// Treats the input as `8N` independent bits drawn from a Bernoulli distribution
/// with the observed marginal frequency. Maximum entropy is 1 bit.
/// This is **not** a rescaling of [`empirical_entropy_bytes`]: bytes are
/// correlated across positions, so the pooled bit entropy is a distinct quantity.
pub fn empirical_entropy_bits(data: &[u8]) -> f64 {
    shannon_h(&bit_histogram(data))
}

/// Empirical (zero-order, IID) joint Shannon entropy `H₀(X,Y)` over aligned prefix bits.
///
/// Computes over the aligned byte prefix of `x` and `y`.
/// Treats aligned bit-pairs `(x_bit_k, y_bit_k)` as IID samples from a joint
/// distribution over `{(0,0),(0,1),(1,0),(1,1)}`. Maximum joint entropy is 2 bits.
pub fn empirical_joint_entropy_bits(x: &[u8], y: &[u8]) -> f64 {
    shannon_h(&joint_bit_histogram(x, y))
}

/// Empirical (zero-order, IID) mutual information `I₀(X;Y)` in bits/bit.
///
/// Computes over the aligned byte prefix of `x` and `y`.
/// `I₀(X;Y) = H₀(X) + H₀(Y) - H₀(X,Y)` where all quantities are over the
/// binary alphabet. Non-negativity is enforced by clamping.
pub fn empirical_mutual_information_bits(x: &[u8], y: &[u8]) -> f64 {
    let (x, y) = aligned_prefix(x, y);
    let h_x = empirical_entropy_bits(x);
    let h_y = empirical_entropy_bits(y);
    let h_xy = empirical_joint_entropy_bits(x, y);
    (h_x + h_y - h_xy).max(0.0)
}

/// Empirical (zero-order, IID) cross-entropy `H₀_q(p)` in bits/bit.
///
/// Scores the observed bit distribution of `test_data` (treated as `p`)
/// against the bit distribution of `train_data` (treated as `q`):
/// `H₀_q(p) = -Σ p(b) log₂ q(b)` over `b ∈ {0,1}`.
pub fn empirical_cross_entropy_bits(test_data: &[u8], train_data: &[u8]) -> f64 {
    if test_data.is_empty() {
        return 0.0;
    }
    let p = bit_histogram(test_data);
    let q = bit_histogram(train_data);
    let mut h = 0.0f64;
    for i in 0..2 {
        if p[i] > 0.0 {
            let qi = q[i].max(1e-12);
            h -= p[i] * qi.log2();
        }
    }
    h
}

/// Total variation distance between the bit distributions of `x` and `y`.
///
/// `TVD = ½ Σ |P(b) - Q(b)|` over `b ∈ {0,1}`.
pub fn tvd_bits(x: &[u8], y: &[u8]) -> f64 {
    if x.is_empty() || y.is_empty() {
        return 0.0;
    }
    let p = bit_histogram(x);
    let q = bit_histogram(y);
    let mut sum = 0.0f64;
    for i in 0..2 {
        sum += (p[i] - q[i]).abs();
    }
    (sum / 2.0).clamp(0.0, 1.0)
}

/// Normalized Hellinger distance between the bit distributions of `x` and `y`.
///
/// `NHD = sqrt(1 - BC(P,Q))` where `BC = Σ sqrt(P(b) Q(b))` over `b ∈ {0,1}`.
pub fn nhd_bits(x: &[u8], y: &[u8]) -> f64 {
    if x.is_empty() || y.is_empty() {
        return 0.0;
    }
    let p = bit_histogram(x);
    let q = bit_histogram(y);
    let mut bc = 0.0f64;
    for i in 0..2 {
        bc += (p[i] * q[i]).sqrt();
    }
    (1.0 - bc).max(0.0).sqrt()
}

/// KL divergence `D_KL(P || Q)` between the bit distributions of `x` (P) and `y` (Q).
///
/// `D_KL(P||Q) = Σ P(b) log₂(P(b)/Q(b))` over `b ∈ {0,1}`. Bits in `y` with
/// zero mass are regularized at `1e-12` to match the bytewise convention.
pub fn d_kl_bits(x: &[u8], y: &[u8]) -> f64 {
    if x.is_empty() || y.is_empty() {
        return 0.0;
    }
    let p = bit_histogram(x);
    let q = bit_histogram(y);
    let mut d_kl = 0.0f64;
    for i in 0..2 {
        if p[i] > 0.0 {
            let qi = q[i].max(1e-12);
            d_kl += p[i] * (p[i] / qi).log2();
        }
    }
    d_kl.max(0.0)
}

/// Jensen-Shannon divergence between the bit distributions of `x` and `y` (bits).
///
/// `JSD(P||Q) = ½ D_KL(P||M) + ½ D_KL(Q||M)` where `M = ½(P+Q)`.
/// Result lies in `[0, 1]` (when expressed in bits, JSD is bounded by `log₂(2) = 1`).
pub fn js_div_bits(x: &[u8], y: &[u8]) -> f64 {
    if x.is_empty() || y.is_empty() {
        return 0.0;
    }
    let p = bit_histogram(x);
    let q = bit_histogram(y);
    let mut m = [0.0f64; 2];
    for i in 0..2 {
        m[i] = 0.5 * (p[i] + q[i]);
    }
    let mut kl_pm = 0.0f64;
    let mut kl_qm = 0.0f64;
    for i in 0..2 {
        if p[i] > 0.0 {
            kl_pm += p[i] * (p[i] / m[i]).log2();
        }
        if q[i] > 0.0 {
            kl_qm += q[i] * (q[i] / m[i]).log2();
        }
    }
    (0.5 * kl_pm + 0.5 * kl_qm).clamp(0.0, 1.0)
}

/// Empirical (zero-order, IID) normalized entropy distance over bits:
/// `(H₀(X,Y) - min(H₀(X), H₀(Y))) / max(H₀(X), H₀(Y))`.
///
/// Computes over the aligned byte prefix of `x` and `y`.
pub fn empirical_ned_bits(x: &[u8], y: &[u8]) -> f64 {
    let (x, y) = aligned_prefix(x, y);
    let h_x = empirical_entropy_bits(x);
    let h_y = empirical_entropy_bits(y);
    let h_xy = empirical_joint_entropy_bits(x, y);
    let min_h = h_x.min(h_y);
    let max_h = h_x.max(h_y);
    if max_h == 0.0 {
        0.0
    } else {
        ((h_xy - min_h) / max_h).clamp(0.0, 1.0)
    }
}

/// Empirical (zero-order, IID) constructive normalized entropy distance over bits:
/// `(H₀(X,Y) - min(H₀(X), H₀(Y))) / H₀(X,Y)`.
///
/// Computes over the aligned byte prefix of `x` and `y`.
pub fn empirical_ned_cons_bits(x: &[u8], y: &[u8]) -> f64 {
    let (x, y) = aligned_prefix(x, y);
    let h_x = empirical_entropy_bits(x);
    let h_y = empirical_entropy_bits(y);
    let h_xy = empirical_joint_entropy_bits(x, y);
    let min_h = h_x.min(h_y);
    if h_xy == 0.0 {
        0.0
    } else {
        ((h_xy - min_h) / h_xy).clamp(0.0, 1.0)
    }
}

/// Empirical (zero-order, IID) NTE (variation of information) over bits,
/// normalized by `max(H₀(X), H₀(Y))`.
///
/// Computes over the aligned byte prefix of `x` and `y`.
pub fn empirical_nte_bits(x: &[u8], y: &[u8]) -> f64 {
    let (x, y) = aligned_prefix(x, y);
    let h_x = empirical_entropy_bits(x);
    let h_y = empirical_entropy_bits(y);
    let h_xy = empirical_joint_entropy_bits(x, y);
    let vi = 2.0 * h_xy - h_x - h_y;
    let max_h = h_x.max(h_y);
    if max_h == 0.0 {
        0.0
    } else {
        (vi / max_h).clamp(0.0, 2.0)
    }
}

/// Empirical (zero-order, IID) resistance-to-transformation over bits:
/// `I₀(X; T(X)) / H₀(X)` for `tx = T(x)`.
///
/// Computes over the aligned byte prefix of `x` and `tx`.
pub fn empirical_resistance_to_transformation_bits(x: &[u8], tx: &[u8]) -> f64 {
    let (x, tx) = aligned_prefix(x, tx);
    let h_x = empirical_entropy_bits(x);
    if h_x < 1e-9 {
        0.0
    } else {
        (empirical_mutual_information_bits(x, tx) / h_x).clamp(0.0, 1.0)
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod bitwise_tests {
    use super::*;

    // ── histogram primitives ──────────────────────────────────────────────────

    #[test]
    fn bit_histogram_all_zeros_is_p1_zero() {
        let [p0, p1] = bit_histogram(&[0x00u8; 64]);
        assert_eq!(p0, 1.0);
        assert_eq!(p1, 0.0);
    }

    #[test]
    fn bit_histogram_all_ones_is_p1_one() {
        let [p0, p1] = bit_histogram(&[0xFFu8; 64]);
        assert_eq!(p0, 0.0);
        assert_eq!(p1, 1.0);
    }

    #[test]
    fn bit_histogram_empty_is_zero() {
        let h = bit_histogram(&[]);
        assert_eq!(h, [0.0, 0.0]);
    }

    #[test]
    fn bit_histogram_sums_to_one() {
        let [p0, p1] = bit_histogram(b"hello world");
        assert!((p0 + p1 - 1.0).abs() < 1e-12);
    }

    #[test]
    fn joint_bit_histogram_empty_is_zero() {
        let h = joint_bit_histogram(&[], &[]);
        assert_eq!(h, [0.0; 4]);
    }

    #[test]
    fn joint_bit_histogram_sums_to_one() {
        let x = b"the quick brown fox";
        let y = b"jumps over the lazy";
        let probs = joint_bit_histogram(x, y);
        let sum: f64 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-12);
    }

    #[test]
    fn joint_bit_histogram_identical_has_no_cross_terms() {
        // When x == y, every bit pair must be (0,0) or (1,1).
        let data = b"infotheory";
        let probs = joint_bit_histogram(data, data);
        // probs[1] = P(0,1), probs[2] = P(1,0) must both be 0.
        assert_eq!(
            probs[1], 0.0,
            "P(x=0,y=1) must be zero for identical inputs"
        );
        assert_eq!(
            probs[2], 0.0,
            "P(x=1,y=0) must be zero for identical inputs"
        );
    }

    // ── empirical bitwise invariants ──────────────────────────────────────────

    #[test]
    fn empirical_entropy_bits_bounds() {
        let constant_0 = [0x00u8; 128];
        let constant_1 = [0xFFu8; 128];
        let text = b"hello world, this is an example with mixed bits!";

        assert_eq!(empirical_entropy_bits(&constant_0), 0.0, "all-zero: H=0");
        assert_eq!(empirical_entropy_bits(&constant_1), 0.0, "all-one: H=0");
        assert!(
            empirical_entropy_bits(text) <= 1.0 + 1e-12,
            "H <= 1 for text"
        );
        assert_eq!(empirical_entropy_bits(b""), 0.0, "empty: H=0");
    }

    #[test]
    fn empirical_entropy_bits_not_a_byte_rescaling() {
        // For ASCII-only text, empirical H_bits != empirical H_bytes / 8.
        // ASCII has bit 7 always 0, so the bit distribution is heavily skewed.
        let ascii = b"The quick brown fox jumps over the lazy dog";
        let h_bits = empirical_entropy_bits(ascii);
        let h_bytes_over_8 = empirical_entropy_bytes(ascii) / 8.0;
        assert!(
            (h_bits - h_bytes_over_8).abs() > 1e-4,
            "empirical H_bits != H_bytes/8 for ASCII: h_bits={h_bits}, h_bytes/8={h_bytes_over_8}"
        );
    }

    #[test]
    fn empirical_joint_entropy_bits_bounds() {
        let x = b"abcdef";
        let y = b"ghijkl";
        let h_xy = empirical_joint_entropy_bits(x, y);
        let h_x = empirical_entropy_bits(x);
        let h_y = empirical_entropy_bits(y);
        assert!(
            h_xy <= h_x + h_y + 1e-12,
            "subadditivity: H(X,Y) <= H(X)+H(Y)"
        );
        assert!(h_xy <= 2.0 + 1e-12, "H_joint_bits <= 2.0");
    }

    #[test]
    fn empirical_mi_bits_non_negative_and_bounded() {
        let x = b"abcabcabc";
        let y = b"xyzxyzxyz";
        let mi = empirical_mutual_information_bits(x, y);
        let h_x = empirical_entropy_bits(x);
        let h_y = empirical_entropy_bits(y);
        assert!(mi >= 0.0, "I(X;Y) >= 0");
        assert!(
            mi <= h_x.min(h_y) + 1e-12,
            "I(X;Y) <= min(H(X),H(Y)): mi={mi}, min_h={}",
            h_x.min(h_y)
        );
    }

    #[test]
    fn js_div_bits_bounds() {
        let zeros = [0x00u8; 64];
        let ones = [0xFFu8; 64];
        let jsd = js_div_bits(&zeros, &ones);
        assert!(
            (jsd - 1.0).abs() < 1e-12,
            "JSD(all-0, all-1) should equal 1 bit: {jsd}"
        );
        let identical = js_div_bits(b"hello", b"hello");
        assert!(identical.abs() < 1e-12, "JSD(x,x) = 0: {identical}");
        let text_jsd = js_div_bits(b"abcabc", b"xyzxyz");
        assert!((0.0..=1.0).contains(&text_jsd), "JSD in [0,1]: {text_jsd}");
    }

    #[test]
    fn d_kl_bits_degenerate_cases() {
        let x = b"hello world";
        let dkl_self = d_kl_bits(x, x);
        assert!(dkl_self.abs() < 1e-10, "D_KL(P||P) = 0: {dkl_self}");
        // Disjoint: regularisation keeps it finite.
        let zeros = [0x00u8; 32];
        let ones = [0xFFu8; 32];
        let dkl = d_kl_bits(&zeros, &ones);
        assert!(
            dkl.is_finite() && dkl > 0.0,
            "D_KL of disjoint dists is finite+positive: {dkl}"
        );
    }

    #[test]
    fn empirical_cross_entropy_bits_degenerate_cases() {
        assert_eq!(empirical_cross_entropy_bits(&[], b"abc"), 0.0);
        let x = b"abcdef";
        let h = empirical_entropy_bits(x);
        let hce = empirical_cross_entropy_bits(x, x);
        assert!(
            (h - hce).abs() < 1e-10,
            "H_self == cross_entropy_self: h={h}, hce={hce}"
        );
        // Degenerate source: all zeros against mixed reference.
        let zeros = [0x00u8; 32];
        let ce = empirical_cross_entropy_bits(&zeros, b"hello world");
        assert!(
            ce.is_finite(),
            "cross_entropy_bits finite for degenerate source: {ce}"
        );
    }

    #[test]
    fn tvd_bits_bounds() {
        let zeros = [0x00u8; 32];
        let ones = [0xFFu8; 32];
        assert!(
            (tvd_bits(&zeros, &ones) - 1.0).abs() < 1e-12,
            "TVD(all-0,all-1)=1"
        );
        assert!(tvd_bits(b"hello", b"hello").abs() < 1e-12, "TVD(x,x)=0");
    }

    #[test]
    fn nhd_bits_bounds() {
        let zeros = [0x00u8; 32];
        let ones = [0xFFu8; 32];
        assert!(
            (nhd_bits(&zeros, &ones) - 1.0).abs() < 1e-12,
            "NHD(all-0,all-1)=1"
        );
        assert!(nhd_bits(b"hello", b"hello").abs() < 1e-12, "NHD(x,x)=0");
    }

    #[test]
    fn empirical_ned_bits_bounds_and_symmetry() {
        let x = b"abcabcabc";
        let y = b"xyzxyzxyz";
        let ned = empirical_ned_bits(x, y);
        assert!((0.0..=1.0).contains(&ned), "NED_bits in [0,1]: {ned}");
        assert!(
            (ned - empirical_ned_bits(y, x)).abs() < 1e-12,
            "NED_bits symmetric"
        );
        assert_eq!(empirical_ned_bits(x, x), 0.0, "NED_bits(x,x)=0");
    }

    #[test]
    fn empirical_nte_bits_bounds_and_symmetry() {
        let x = b"abcabcabc";
        let y = b"xyzxyzxyz";
        let nte = empirical_nte_bits(x, y);
        assert!((0.0..=2.0).contains(&nte), "NTE_bits in [0,2]: {nte}");
        assert!(
            (nte - empirical_nte_bits(y, x)).abs() < 1e-12,
            "NTE_bits symmetric"
        );
    }

    // ── per-bit algorithmic unit conversion ───────────────────────────────────

    #[cfg(feature = "backend-ctw")]
    #[test]
    fn algorithmic_per_bit_is_per_byte_over_8() {
        let data = b"abcabcabcabcabc";
        let h_bytes = try_entropy_rate_bytes(data).expect("entropy_rate bytes");
        let h_per_bit = try_entropy_rate_per_bit(data).expect("entropy_rate per_bit");
        let expected = h_bytes / 8.0;
        assert!(
            (h_per_bit - expected).abs() < 1e-12,
            "per_bit={h_per_bit}, bytes/8={expected}"
        );
    }

    #[cfg(feature = "backend-ctw")]
    #[test]
    fn algorithmic_per_bit_nonnegative() {
        let rate = try_entropy_rate_per_bit(b"hello world").expect("rate");
        assert!(
            rate >= 0.0 && rate.is_finite(),
            "rate_per_bit must be finite non-negative: {rate}"
        );
    }

    #[cfg(feature = "backend-ctw")]
    #[test]
    fn algorithmic_per_bit_joint_rate_is_per_byte_over_8() {
        let x = b"abcabc";
        let y = b"xyzxyz";
        let h_bytes = try_joint_entropy_rate_bytes(x, y).expect("joint bytes");
        let h_per_bit = try_joint_entropy_rate_per_bit(x, y).expect("joint per_bit");
        assert!(
            (h_per_bit - h_bytes / 8.0).abs() < 1e-12,
            "joint per_bit={h_per_bit}, bytes/8={}",
            h_bytes / 8.0
        );
    }

    #[cfg(feature = "backend-ctw")]
    #[test]
    fn ned_rate_per_bit_equals_ned_rate_bytes() {
        let x = b"abcabcabc";
        let y = b"xyzxyzxyz";
        let ned_bytes = try_ned_bytes(x, y).expect("ned bytes");
        let ned_per_bit = try_ned_rate_per_bit(x, y).expect("ned per_bit");
        assert!(
            (ned_bytes - ned_per_bit).abs() < 1e-12,
            "NED_per_bit must equal NED_bytes: {ned_bytes} vs {ned_per_bit}"
        );
    }

    #[cfg(feature = "backend-ctw")]
    #[test]
    fn scale_invariant_per_bit_aliases_match_bytes() {
        let x = b"abcabcabc";
        let y = b"xyzxyzxyz";
        let ned_cons = try_ned_cons_bytes(x, y).expect("ned_cons");
        let ned_cons_per_bit = try_ned_cons_rate_per_bit(x, y).expect("ned_cons per_bit");
        assert!((ned_cons - ned_cons_per_bit).abs() < 1e-12);

        let nte = try_nte_bytes(x, y).expect("nte");
        let nte_per_bit = try_nte_rate_per_bit(x, y).expect("nte per_bit");
        assert!((nte - nte_per_bit).abs() < 1e-12);

        let rt = try_resistance_to_transformation_bytes(x, x).expect("rt");
        let rt_per_bit = try_resistance_to_transformation_per_bit(x, x).expect("rt per_bit");
        assert!((rt - rt_per_bit).abs() < 1e-12);
    }

    #[cfg(feature = "backend-ctw")]
    #[test]
    fn conditional_entropy_rate_per_bit_is_per_byte_over_8() {
        let x = b"abcabc";
        let y = b"xyzxyz";
        let h_bytes = try_conditional_entropy_rate_bytes(x, y).expect("ce bytes");
        let h_per_bit = try_conditional_entropy_rate_per_bit(x, y).expect("ce per_bit");
        assert!((h_per_bit - h_bytes / 8.0).abs() < 1e-12);
    }

    #[test]
    fn bit_histogram_total_uses_widening_multiply() {
        // Sanity: probabilities remain a valid distribution for ordinary sizes.
        // The widening cast is the portable fix for 32-bit `usize` overflow.
        let [p0, p1] = bit_histogram(&[0xAAu8; 1024]);
        assert!((p0 + p1 - 1.0).abs() < 1e-12);
        assert!((0.0..=1.0).contains(&p0));
        assert!((0.0..=1.0).contains(&p1));
    }

    #[cfg(feature = "backend-ctw")]
    #[test]
    fn intrinsic_dependence_bits_bounds_and_not_a_rescaling() {
        let zeros = [0u8; 64];
        assert_eq!(
            try_intrinsic_dependence_bits(&zeros).expect("id bits zeros"),
            0.0
        );

        // ASCII-heavy text: ID_bits is a distinct framing from ID_bytes (and
        // from ID_bytes/8), because H0_bits ≠ H0_bytes/8 while the rate
        // scales exactly by 1/8.
        let ascii = b"The quick brown fox jumps over the lazy dog. ";
        let id_bytes = try_intrinsic_dependence_bytes(ascii).expect("id bytes");
        let id_bits = try_intrinsic_dependence_bits(ascii).expect("id bits");
        assert!((0.0..=1.0).contains(&id_bytes), "id_bytes={id_bytes}");
        assert!((0.0..=1.0).contains(&id_bits), "id_bits={id_bits}");
        assert!(
            (id_bits - id_bytes).abs() > 1e-6 || (id_bits - id_bytes / 8.0).abs() > 1e-6,
            "ID_bits should not collapse to ID_bytes or ID_bytes/8: bits={id_bits} bytes={id_bytes}"
        );

        // Explicit identity of the bits formula against primitives.
        let h0 = empirical_entropy_bits(ascii);
        let h_rate = try_entropy_rate_per_bit(ascii).expect("rate per bit");
        let expected = ((h0 - h_rate) / h0).clamp(0.0, 1.0);
        assert!((id_bits - expected).abs() < 1e-12);
    }
}
