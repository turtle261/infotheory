//! Information-theoretic metric and scoring API surface.

use super::types::{RateBackend, validate_rate_backend};
use crate::error::{InfotheoryError, InfotheoryResult};

#[cfg(feature = "backend-ctw")]
use crate::backends::ctw::FacContextTree;
#[cfg(feature = "backend-particle")]
use crate::backends::particle::ParticleRuntime;
#[cfg(feature = "backend-rosa")]
use crate::backends::rosaplus::RosaPlus;
#[cfg(feature = "backend-zpaq")]
use crate::backends::zpaq_rate::ZpaqRateModel;
#[cfg(feature = "backend-mamba")]
use crate::with_mamba_method_tls;
#[cfg(feature = "backend-rwkv")]
use crate::with_rwkv_method_tls;
use crate::{aligned_prefix, with_default_ctx};

#[inline(always)]
pub fn try_entropy_rate_bytes(data: &[u8], max_order: i64) -> InfotheoryResult<f64> {
    with_default_ctx(|ctx| ctx.try_entropy_rate_bytes(data, max_order))
}

#[inline(always)]
pub fn try_biased_entropy_rate_bytes(data: &[u8], max_order: i64) -> InfotheoryResult<f64> {
    with_default_ctx(|ctx| ctx.try_biased_entropy_rate_bytes(data, max_order))
}

/// Mutual information rate estimate under an explicit `backend`.
///
/// Inputs are aligned to the shared prefix length.
pub fn try_mutual_information_rate_backend(
    x: &[u8],
    y: &[u8],
    max_order: i64,
    backend: &RateBackend,
) -> InfotheoryResult<f64> {
    let (x, y) = aligned_prefix(x, y);
    if x.is_empty() {
        return Ok(0.0);
    }
    let h_x = try_entropy_rate_backend(x, max_order, backend)?;
    let h_y = try_entropy_rate_backend(y, max_order, backend)?;
    let h_xy = try_joint_entropy_rate_backend(x, y, max_order, backend)?;
    Ok((h_x + h_y - h_xy).max(0.0))
}

/// Normalized entropy distance under an explicit `backend`.
///
/// Returns a value in `[0, 1]` after clamping.
pub fn try_ned_rate_backend(
    x: &[u8],
    y: &[u8],
    max_order: i64,
    backend: &RateBackend,
) -> InfotheoryResult<f64> {
    let (x, y) = aligned_prefix(x, y);
    if x.is_empty() {
        return Ok(0.0);
    }
    let h_x = try_entropy_rate_backend(x, max_order, backend)?;
    let h_y = try_entropy_rate_backend(y, max_order, backend)?;
    let h_xy = try_joint_entropy_rate_backend(x, y, max_order, backend)?;
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
    max_order: i64,
    backend: &RateBackend,
) -> InfotheoryResult<f64> {
    let (x, y) = aligned_prefix(x, y);
    if x.is_empty() {
        return Ok(0.0);
    }
    let h_x = try_entropy_rate_backend(x, max_order, backend)?;
    let h_y = try_entropy_rate_backend(y, max_order, backend)?;
    let h_xy = try_joint_entropy_rate_backend(x, y, max_order, backend)?;
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
    max_order: i64,
    backend: &RateBackend,
) -> InfotheoryResult<f64> {
    validate_rate_backend(backend)?;
    Ok(match backend {
        #[cfg(feature = "backend-rosa")]
        RateBackend::RosaPlus => {
            let mut m = RosaPlus::new(max_order, false, 0, 42);
            m.predictive_entropy_rate(data)
        }
        #[cfg(not(feature = "backend-rosa"))]
        RateBackend::RosaPlus => {
            return Err(InfotheoryError::invalid_backend_config(
                "backend 'rosaplus' requires infotheory feature 'backend-rosa'".to_string(),
            ));
        }
        RateBackend::Match { .. }
        | RateBackend::SparseMatch { .. }
        | RateBackend::Ppmd { .. }
        | RateBackend::Sequitur { .. }
        | RateBackend::Calibrated { .. } => {
            crate::try_prequential_rate_backend(data, &[], max_order, backend)?
        }
        #[cfg(feature = "backend-rwkv")]
        RateBackend::Rwkv7Method { method } => with_rwkv_method_tls(method, |c| {
            c.cross_entropy(data).map_err(|e| {
                InfotheoryError::runtime(format!("rwkv method entropy scoring failed: {e:#}"))
            })
        })?,
        #[cfg(feature = "backend-mamba")]
        RateBackend::MambaMethod { method } => with_mamba_method_tls(method, |c| {
            c.cross_entropy(data).map_err(|e| {
                InfotheoryError::runtime(format!("mamba method entropy scoring failed: {e:#}"))
            })
        })?,
        #[cfg(feature = "backend-zpaq")]
        RateBackend::Zpaq { method } => {
            if data.is_empty() {
                return Ok(0.0);
            }
            let mut model = ZpaqRateModel::new(method.clone(), 2f64.powi(-24));
            let bits = model.update_and_score(data);
            bits / (data.len() as f64)
        }
        #[cfg(not(feature = "backend-zpaq"))]
        RateBackend::Zpaq { .. } => {
            return Err(InfotheoryError::invalid_backend_config(
                "backend 'zpaq' requires infotheory feature 'backend-zpaq'".to_string(),
            ));
        }
        #[cfg(feature = "backend-mixture")]
        RateBackend::Mixture { spec } => {
            if data.is_empty() {
                return Ok(0.0);
            }
            let experts = spec.build_experts();
            let mut mix =
                crate::mixture::build_mixture_runtime(spec.as_ref(), &experts).map_err(|e| {
                    InfotheoryError::invalid_backend_config(format!("MixtureSpec invalid: {e}"))
                })?;
            mix.begin_stream(Some(data.len() as u64)).map_err(|e| {
                InfotheoryError::runtime(format!("Mixture stream init failed: {e}"))
            })?;
            let mut bits = 0.0;
            for &b in data {
                bits -= mix.step(b) / std::f64::consts::LN_2;
            }
            mix.finish_stream().map_err(|e| {
                InfotheoryError::runtime(format!("Mixture stream finalize failed: {e}"))
            })?;
            bits / (data.len() as f64)
        }
        #[cfg(not(feature = "backend-mixture"))]
        RateBackend::Mixture { .. } => {
            return Err(InfotheoryError::invalid_backend_config(
                "backend 'mixture' requires infotheory feature 'backend-mixture'".to_string(),
            ));
        }
        #[cfg(feature = "backend-particle")]
        RateBackend::Particle { spec } => {
            if data.is_empty() {
                return Ok(0.0);
            }
            let mut runtime = ParticleRuntime::new(spec.as_ref());
            let mut bits = 0.0;
            for &b in data {
                bits -= runtime.step(b) / std::f64::consts::LN_2;
            }
            bits / (data.len() as f64)
        }
        #[cfg(not(feature = "backend-particle"))]
        RateBackend::Particle { .. } => {
            return Err(InfotheoryError::invalid_backend_config(
                "backend 'particle' requires infotheory feature 'backend-particle'".to_string(),
            ));
        }
        #[cfg(feature = "backend-ctw")]
        RateBackend::Ctw { depth } => {
            if data.is_empty() {
                return Ok(0.0);
            }
            let mut fac = FacContextTree::new(*depth, 8);
            fac.reserve_for_symbols(data.len());
            for &b in data {
                fac.update_byte_msb(b);
            }
            let ln_p = fac.get_log_block_probability();
            let bits = -ln_p / std::f64::consts::LN_2;
            bits / (data.len() as f64)
        }
        #[cfg(not(feature = "backend-ctw"))]
        RateBackend::Ctw { .. } => {
            return Err(InfotheoryError::invalid_backend_config(
                "backend 'ctw' requires infotheory feature 'backend-ctw'".to_string(),
            ));
        }
        #[cfg(feature = "backend-ctw")]
        RateBackend::FacCtw {
            base_depth,
            num_percept_bits: _,
            encoding_bits,
        } => {
            if data.is_empty() {
                return Ok(0.0);
            }
            let bits_per_byte = (*encoding_bits).clamp(1, 8);
            let mut fac = FacContextTree::new(*base_depth, bits_per_byte);
            fac.reserve_for_symbols(data.len());
            for &b in data {
                fac.update_byte_lsb(b);
            }
            let ln_p = fac.get_log_block_probability();
            let bits = -ln_p / std::f64::consts::LN_2;
            bits / (data.len() as f64)
        }
        #[cfg(not(feature = "backend-ctw"))]
        RateBackend::FacCtw { .. } => {
            return Err(InfotheoryError::invalid_backend_config(
                "backend 'fac-ctw' requires infotheory feature 'backend-ctw'".to_string(),
            ));
        }
    })
}

/// Fallible biased/plugin entropy rate of `data` using the explicit rate `backend`.
pub fn try_biased_entropy_rate_backend(
    data: &[u8],
    max_order: i64,
    backend: &RateBackend,
) -> InfotheoryResult<f64> {
    match backend {
        #[cfg(feature = "backend-zpaq")]
        RateBackend::Zpaq { .. } => Err(InfotheoryError::unsupported(
            "biased/plugin entropy is not supported for zpaq rate backends in 1.1.1",
        )),
        #[cfg(not(feature = "backend-zpaq"))]
        RateBackend::Zpaq { .. } => Err(InfotheoryError::invalid_backend_config(
            "backend 'zpaq' requires infotheory feature 'backend-zpaq'".to_string(),
        )),
        _ => crate::try_frozen_plugin_rate_backend(data, &[data], max_order, backend),
    }
}

/// Fallible cross-entropy H_{train}(test) - score test_data under model trained on train_data.
pub fn try_cross_entropy_rate_backend(
    test_data: &[u8],
    train_data: &[u8],
    max_order: i64,
    backend: &RateBackend,
) -> InfotheoryResult<f64> {
    validate_rate_backend(backend)?;
    match backend {
        #[cfg(feature = "backend-zpaq")]
        RateBackend::Zpaq { method } => {
            if test_data.is_empty() {
                return Ok(0.0);
            }
            let mut model = ZpaqRateModel::new(method.clone(), 2f64.powi(-24));
            model.update_and_score(train_data);
            let bits = model.update_and_score(test_data);
            Ok(bits / (test_data.len() as f64))
        }
        #[cfg(not(feature = "backend-zpaq"))]
        RateBackend::Zpaq { .. } => Err(InfotheoryError::invalid_backend_config(
            "backend 'zpaq' requires infotheory feature 'backend-zpaq'".to_string(),
        )),
        _ => crate::try_frozen_plugin_rate_backend(test_data, &[train_data], max_order, backend),
    }
}

/// Fallible joint entropy rate `H(X,Y)` using an explicit `backend`.
pub fn try_joint_entropy_rate_backend(
    x: &[u8],
    y: &[u8],
    max_order: i64,
    backend: &RateBackend,
) -> InfotheoryResult<f64> {
    validate_rate_backend(backend)?;
    let (x, y) = aligned_prefix(x, y);
    if x.is_empty() {
        return Ok(0.0);
    }
    Ok(match backend {
        #[cfg(feature = "backend-rosa")]
        RateBackend::RosaPlus => {
            let joint_symbols: Vec<u32> = (0..x.len())
                .map(|i| (x[i] as u32) * 256 + (y[i] as u32))
                .collect();
            let mut m = RosaPlus::new(max_order, false, 0, 42);
            m.entropy_rate_cps(&joint_symbols)
        }
        #[cfg(not(feature = "backend-rosa"))]
        RateBackend::RosaPlus => {
            return Err(InfotheoryError::invalid_backend_config(
                "backend 'rosaplus' requires infotheory feature 'backend-rosa'".to_string(),
            ));
        }
        RateBackend::Match { .. }
        | RateBackend::SparseMatch { .. }
        | RateBackend::Ppmd { .. }
        | RateBackend::Sequitur { .. }
        | RateBackend::Calibrated { .. } => {
            let mut joint = Vec::with_capacity(x.len() * 2);
            for (&xb, &yb) in x.iter().zip(y.iter()) {
                joint.push(xb);
                joint.push(yb);
            }
            try_entropy_rate_backend(&joint, max_order, backend)? * 2.0
        }
        #[cfg(feature = "backend-rwkv")]
        RateBackend::Rwkv7Method { method } => with_rwkv_method_tls(method, |c| {
            c.joint_cross_entropy_aligned_min(x, y).map_err(|e| {
                InfotheoryError::runtime(format!("rwkv method joint-entropy scoring failed: {e:#}"))
            })
        })?,
        #[cfg(feature = "backend-mamba")]
        RateBackend::MambaMethod { method } => with_mamba_method_tls(method, |c| {
            c.joint_cross_entropy_aligned_min(x, y).map_err(|e| {
                InfotheoryError::runtime(format!(
                    "mamba method joint-entropy scoring failed: {e:#}"
                ))
            })
        })?,
        #[cfg(feature = "backend-zpaq")]
        RateBackend::Zpaq { method } => {
            let mut joint = Vec::with_capacity(x.len() * 2);
            for (&xb, &yb) in x.iter().zip(y.iter()) {
                joint.push(xb);
                joint.push(yb);
            }
            let mut model = ZpaqRateModel::new(method.clone(), 2f64.powi(-24));
            let bits = model.update_and_score(&joint);
            bits / (x.len() as f64)
        }
        #[cfg(not(feature = "backend-zpaq"))]
        RateBackend::Zpaq { .. } => {
            return Err(InfotheoryError::invalid_backend_config(
                "backend 'zpaq' requires infotheory feature 'backend-zpaq'".to_string(),
            ));
        }
        #[cfg(feature = "backend-mixture")]
        RateBackend::Mixture { spec } => {
            let mut joint = Vec::with_capacity(x.len() * 2);
            for (&xb, &yb) in x.iter().zip(y.iter()) {
                joint.push(xb);
                joint.push(yb);
            }
            let experts = spec.build_experts();
            let mut mix =
                crate::mixture::build_mixture_runtime(spec.as_ref(), &experts).map_err(|e| {
                    InfotheoryError::invalid_backend_config(format!("MixtureSpec invalid: {e}"))
                })?;
            mix.begin_stream(Some(joint.len() as u64)).map_err(|e| {
                InfotheoryError::runtime(format!("Mixture stream init failed: {e}"))
            })?;
            let mut bits = 0.0;
            for &b in &joint {
                bits -= mix.step(b) / std::f64::consts::LN_2;
            }
            mix.finish_stream().map_err(|e| {
                InfotheoryError::runtime(format!("Mixture stream finalize failed: {e}"))
            })?;
            bits / (x.len() as f64)
        }
        #[cfg(not(feature = "backend-mixture"))]
        RateBackend::Mixture { .. } => {
            return Err(InfotheoryError::invalid_backend_config(
                "backend 'mixture' requires infotheory feature 'backend-mixture'".to_string(),
            ));
        }
        #[cfg(feature = "backend-particle")]
        RateBackend::Particle { spec } => {
            let mut joint = Vec::with_capacity(x.len() * 2);
            for (&xb, &yb) in x.iter().zip(y.iter()) {
                joint.push(xb);
                joint.push(yb);
            }
            let mut runtime = ParticleRuntime::new(spec.as_ref());
            let mut bits = 0.0;
            for &b in &joint {
                bits -= runtime.step(b) / std::f64::consts::LN_2;
            }
            bits / (x.len() as f64)
        }
        #[cfg(not(feature = "backend-particle"))]
        RateBackend::Particle { .. } => {
            return Err(InfotheoryError::invalid_backend_config(
                "backend 'particle' requires infotheory feature 'backend-particle'".to_string(),
            ));
        }
        #[cfg(feature = "backend-ctw")]
        RateBackend::Ctw { depth } => {
            let mut fac = FacContextTree::new(*depth, 16);
            for k in 0..x.len() {
                let bx = x[k];
                let by = y[k];
                for bit_idx in 0..8 {
                    let bit_x = ((bx >> (7 - bit_idx)) & 1) == 1;
                    let bit_y = ((by >> (7 - bit_idx)) & 1) == 1;
                    fac.update(bit_x, bit_idx);
                    fac.update(bit_y, bit_idx + 8);
                }
            }
            let ln_p = fac.get_log_block_probability();
            let bits = -ln_p / std::f64::consts::LN_2;
            bits / (x.len() as f64)
        }
        #[cfg(not(feature = "backend-ctw"))]
        RateBackend::Ctw { .. } => {
            return Err(InfotheoryError::invalid_backend_config(
                "backend 'ctw' requires infotheory feature 'backend-ctw'".to_string(),
            ));
        }
        #[cfg(feature = "backend-ctw")]
        RateBackend::FacCtw {
            base_depth,
            num_percept_bits: _,
            encoding_bits,
        } => {
            let bits_per_byte = (*encoding_bits).clamp(1, 8);
            let mut fac = FacContextTree::new(*base_depth, bits_per_byte * 2);
            for k in 0..x.len() {
                let bx = x[k];
                let by = y[k];
                for i in 0..bits_per_byte {
                    let bit_idx_x = i * 2;
                    let bit_idx_y = bit_idx_x + 1;
                    fac.update(((bx >> i) & 1) == 1, bit_idx_x);
                    fac.update(((by >> i) & 1) == 1, bit_idx_y);
                }
            }
            let ln_p = fac.get_log_block_probability();
            let bits = -ln_p / std::f64::consts::LN_2;
            bits / (x.len() as f64)
        }
        #[cfg(not(feature = "backend-ctw"))]
        RateBackend::FacCtw { .. } => {
            return Err(InfotheoryError::invalid_backend_config(
                "backend 'fac-ctw' requires infotheory feature 'backend-ctw'".to_string(),
            ));
        }
    })
}

#[inline(always)]
pub fn marginal_entropy_bytes(data: &[u8]) -> f64 {
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

#[inline(always)]
pub fn joint_marginal_entropy_bytes(x: &[u8], y: &[u8]) -> f64 {
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
pub fn try_joint_entropy_rate_bytes(x: &[u8], y: &[u8], max_order: i64) -> InfotheoryResult<f64> {
    with_default_ctx(|ctx| ctx.try_joint_entropy_rate_bytes(x, y, max_order))
}

#[inline(always)]
pub fn try_conditional_entropy_rate_bytes(
    x: &[u8],
    y: &[u8],
    max_order: i64,
) -> InfotheoryResult<f64> {
    with_default_ctx(|ctx| ctx.try_conditional_entropy_rate_bytes(x, y, max_order))
}

#[inline(always)]
pub fn try_conditional_entropy_bytes(x: &[u8], y: &[u8], max_order: i64) -> InfotheoryResult<f64> {
    with_default_ctx(|ctx| ctx.try_conditional_entropy_bytes(x, y, max_order))
}

#[inline(always)]
pub fn try_mutual_information_bytes(x: &[u8], y: &[u8], max_order: i64) -> InfotheoryResult<f64> {
    with_default_ctx(|ctx| ctx.try_mutual_information_bytes(x, y, max_order))
}

pub fn mutual_information_marg_bytes(x: &[u8], y: &[u8]) -> f64 {
    let (x, y) = aligned_prefix(x, y);
    let h_x = marginal_entropy_bytes(x);
    let h_y = marginal_entropy_bytes(y);
    let h_xy = joint_marginal_entropy_bytes(x, y);
    (h_x + h_y - h_xy).max(0.0)
}

#[inline(always)]
pub fn try_mutual_information_rate_bytes(
    x: &[u8],
    y: &[u8],
    max_order: i64,
) -> InfotheoryResult<f64> {
    with_default_ctx(|ctx| ctx.try_mutual_information_rate_bytes(x, y, max_order))
}

#[inline(always)]
pub fn try_ned_bytes(x: &[u8], y: &[u8], max_order: i64) -> InfotheoryResult<f64> {
    with_default_ctx(|ctx| ctx.try_ned_bytes(x, y, max_order))
}

pub fn ned_marg_bytes(x: &[u8], y: &[u8]) -> f64 {
    let (x, y) = aligned_prefix(x, y);
    let h_x = marginal_entropy_bytes(x);
    let h_y = marginal_entropy_bytes(y);
    let h_xy = joint_marginal_entropy_bytes(x, y);
    let min_h = h_x.min(h_y);
    let max_h = h_x.max(h_y);
    if max_h == 0.0 {
        0.0
    } else {
        ((h_xy - min_h) / max_h).clamp(0.0, 1.0)
    }
}

#[inline(always)]
pub fn try_ned_rate_bytes(x: &[u8], y: &[u8], max_order: i64) -> InfotheoryResult<f64> {
    with_default_ctx(|ctx| ctx.try_ned_bytes(x, y, max_order))
}

#[inline(always)]
pub fn try_ned_cons_bytes(x: &[u8], y: &[u8], max_order: i64) -> InfotheoryResult<f64> {
    with_default_ctx(|ctx| ctx.try_ned_cons_bytes(x, y, max_order))
}

pub fn ned_cons_marg_bytes(x: &[u8], y: &[u8]) -> f64 {
    let h_x = marginal_entropy_bytes(x);
    let h_y = marginal_entropy_bytes(y);
    let h_xy = joint_marginal_entropy_bytes(x, y);
    let min_h = h_x.min(h_y);
    if h_xy == 0.0 {
        0.0
    } else {
        ((h_xy - min_h) / h_xy).clamp(0.0, 1.0)
    }
}

#[inline(always)]
pub fn try_ned_cons_rate_bytes(x: &[u8], y: &[u8], max_order: i64) -> InfotheoryResult<f64> {
    with_default_ctx(|ctx| ctx.try_ned_cons_bytes(x, y, max_order))
}

#[inline(always)]
pub fn try_nte_bytes(x: &[u8], y: &[u8], max_order: i64) -> InfotheoryResult<f64> {
    with_default_ctx(|ctx| ctx.try_nte_bytes(x, y, max_order))
}

pub fn nte_marg_bytes(x: &[u8], y: &[u8]) -> f64 {
    let (x, y) = aligned_prefix(x, y);
    let h_x = marginal_entropy_bytes(x);
    let h_y = marginal_entropy_bytes(y);
    let h_xy = joint_marginal_entropy_bytes(x, y);
    let vi = 2.0 * h_xy - h_x - h_y;
    let max_h = h_x.max(h_y);
    if max_h == 0.0 {
        0.0
    } else {
        (vi / max_h).clamp(0.0, 2.0)
    }
}

#[inline(always)]
pub fn try_nte_rate_bytes(x: &[u8], y: &[u8], max_order: i64) -> InfotheoryResult<f64> {
    with_default_ctx(|ctx| ctx.try_nte_bytes(x, y, max_order))
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
pub fn tvd_bytes(x: &[u8], y: &[u8], _max_order: i64) -> f64 {
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
pub fn nhd_bytes(x: &[u8], y: &[u8], _max_order: i64) -> f64 {
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
pub fn try_cross_entropy_bytes(
    test_data: &[u8],
    train_data: &[u8],
    max_order: i64,
) -> InfotheoryResult<f64> {
    with_default_ctx(|ctx| ctx.try_cross_entropy_bytes(test_data, train_data, max_order))
}

#[inline(always)]
pub fn try_cross_entropy_rate_bytes(
    test_data: &[u8],
    train_data: &[u8],
    max_order: i64,
) -> InfotheoryResult<f64> {
    with_default_ctx(|ctx| ctx.try_cross_entropy_rate_bytes(test_data, train_data, max_order))
}

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
pub fn try_intrinsic_dependence_bytes(data: &[u8], max_order: i64) -> InfotheoryResult<f64> {
    with_default_ctx(|ctx| ctx.try_intrinsic_dependence_bytes(data, max_order))
}

#[inline(always)]
pub fn try_resistance_to_transformation_bytes(
    x: &[u8],
    tx: &[u8],
    max_order: i64,
) -> InfotheoryResult<f64> {
    with_default_ctx(|ctx| ctx.try_resistance_to_transformation_bytes(x, tx, max_order))
}
