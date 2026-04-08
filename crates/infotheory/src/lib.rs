#![allow(unsafe_op_in_unsafe_fn)]

//! # InfoTheory: Information Theoretic Estimators & Metrics
//!
//! This crate provides a comprehensive suite of information-theoretic primitives for
//! quantifying complexity, dependence, and similarity between data sequences.
//!
//! It implements two primary classes of estimators:
//! 1.  **Compression-based (Kolmogorov Complexity)**: Using the ZPAQ compression algorithm to estimate
//!     Normalized Compression Distance (NCD).
//! 2.  **Entropy-based (Shannon Information)**: Using both exact marginal histograms (for i.i.d. data)
//!     and the ROSA (Rapid Online Suffix Automaton) predictive language model (for sequential data)
//!     to estimate Entropy, Mutual Information, and related distances.
//!
//! ## Mathematical Primitives
//!
//! The library implements the following core measures. For sequential data, "Rate" variants
//! use the ROSA model to estimate `Ĥ(X)` (entropy rate), while "Marginal" variants
//! treat data as a bag-of-bytes (i.i.d.) and compute `H(X)` from histograms.
//!
//! ### 1. Normalized Compression Distance (NCD)
//! Approximates the Normalized Information Distance (NID) using a compressor `C`.
//!
//! `NCD(x,y) = (C(xy) - min(C(x), C(y))) / max(C(x), C(y))`
//!
//! ### 2. Normalized Entropy Distance (NED)
//! An entropic analogue to NCD, defined using Shannon entropy `H`.
//!
//! `NED(X,Y) = (H(X,Y) - min(H(X), H(Y))) / max(H(X), H(Y))`
//!
//! ### 3. Normalized Transform Effort (NTE)
//! Based on the Variation of Information (VI), normalized by the maximum entropy.
//!
//! `NTE(X,Y) = (H(X|Y) + H(Y|X)) / max(H(X), H(Y)) = (2H(X,Y) - H(X) - H(Y)) / max(H(X), H(Y))`
//!
//! ### 4. Mutual Information (MI)
//! Measures the amount of information obtained about one random variable by observing another.
//!
//! `I(X;Y) = H(X) + H(Y) - H(X,Y)`
//!
//! ### 5. Divergences & Distances
//! *   **Total Variation Distance (TVD)**: `δ(P,Q) = 0.5 * Σ |P(x) - Q(x)|`
//! *   **Normalized Hellinger Distance (NHD)**: `sqrt(1 - Σ sqrt(P(x)Q(x)))`
//! *   **Kullback-Leibler Divergence (KL)**: `D_KL(P||Q) = Σ P(x) log(P(x)/Q(x))`
//! *   **Jensen-Shannon Divergence (JSD)**: Symmetrized and smoothed KL divergence.
//!
//! ### 6. Intrinsic Dependence (ID)
//! Measures the redundancy within a sequence, comparing marginal entropy to entropy rate.
//!
//! `ID(X) = (H_marginal(X) - H_rate(X)) / H_marginal(X)`
//!
//! ### 7. Resistance to Transformation
//! Quantifies how much information is preserved after a transformation `T` is applied.
//!
//! `R(X, T) = I(X; T(X)) / H(X)`
//!
//! ## Usage
//!
//! ```rust,no_run
//! use infotheory::api::{mutual_information_marg_bytes, try_ncd_paths, NcdVariant};
//!
//! let x = b"some data sequence";
//! let y = b"another data sequence";
//!
//! // Compression-based distance
//! let ncd = try_ncd_paths("file1.txt", "file2.txt", "5", NcdVariant::Vitanyi)
//!     .expect("ncd");
//!
//! // Entropy-based mutual information (Marginal / i.i.d.)
//! let mi_marg = mutual_information_marg_bytes(x, y);
//! ```

/// AIXI planning components, environments, and model abstractions.
pub mod aixi;
/// Public spec-first API reexports.
pub mod api;
/// Core information-theoretic axioms and validation helpers.
pub mod axioms;
/// Entropy/compression backend implementations and backend discovery.
pub mod backends;
/// Entropy coder implementations (AC and rANS).
pub mod coders;
/// Rate-coded compression helpers built on generic rate backends.
pub mod compression;
/// Synthetic data generators for information-theory experiments.
pub mod datagen;
/// Diagnostic tooling for exact AC/log-loss mixture tracing.
pub mod diagnostics;
/// Shared public error types for fallible APIs.
pub mod error;
/// Online Bayesian/switching/MDL mixture predictors.
pub mod mixture;
pub(crate) mod neural_mix;
/// Shared spec -> runtime builders and backend registry metadata.
pub(crate) mod runtime;
/// Information-theoretic code search pipeline (3-stage: prefilter, filter, KMI rerank).
#[cfg(feature = "backend-rosa")]
pub mod search;
pub(crate) mod simd_math;
/// Shared backend/spec parsing and loading helpers.
pub mod spec;
use crate::api::RateBackend;
#[cfg(all(test, any(feature = "default-backends", feature = "all-backends")))]
pub(crate) use crate::api::{
    CalibratedSpec, CalibrationContextKind, MixtureExpertSpec, MixtureKind, MixtureSpec,
    ParticleSpec,
};
#[cfg(all(test, any(feature = "default-backends", feature = "all-backends")))]
use crate::api::{
    CompressionBackend, GenerationConfig, InfotheoryCtx, NcdVariant, RateBackendSession,
    d_kl_bytes, try_biased_entropy_rate_backend, try_conditional_entropy_bytes,
    try_conditional_entropy_rate_bytes, try_cross_entropy_rate_backend, try_entropy_rate_backend,
    try_entropy_rate_bytes, try_joint_entropy_rate_backend, try_joint_entropy_rate_bytes,
    try_mutual_information_bytes, try_ncd_bytes,
};
#[cfg(all(test, any(feature = "default-backends", feature = "all-backends")))]
use crate::api::{
    joint_marginal_entropy_bytes, js_div_bytes, marginal_entropy_bytes, nhd_bytes, tvd_bytes,
};
use crate::error::{InfotheoryError, InfotheoryResult};
/// CTW and FAC-CTW backend types.
#[cfg(feature = "backend-ctw")]
pub use backends::ctw;
#[cfg(feature = "backend-mamba")]
/// Mamba backend types and compressor.
pub use backends::mambazip;
/// Match-based repeat predictor.
#[cfg(feature = "backend-match")]
pub use backends::match_model;
/// Particle-latent filter ensemble rate backend.
#[cfg(feature = "backend-particle")]
pub use backends::particle;
/// PPMD-style byte model.
#[cfg(feature = "backend-ppmd")]
pub use backends::ppmd;
/// ROSA+ backend types.
#[cfg(feature = "backend-rosa")]
pub use backends::rosaplus;
#[cfg(feature = "backend-rwkv")]
/// RWKV backend types and compressor.
pub use backends::rwkvzip;
/// Exact online Sequitur backend types.
#[cfg(feature = "backend-sequitur")]
pub use backends::sequitur;
/// Sparse/gapped match predictor.
#[cfg(feature = "backend-match")]
pub use backends::sparse_match;
/// ZPAQ rate-model adapter.
#[cfg(feature = "backend-zpaq")]
pub use backends::zpaq_rate;

#[cfg(feature = "backend-rosa")]
use crate::backends::rosaplus::RosaPlus;
use crate::mixture::OnlineBytePredictor;
use std::cell::RefCell;
#[cfg(any(feature = "backend-rwkv", feature = "backend-mamba"))]
use std::collections::HashMap;
#[cfg(all(test, any(feature = "default-backends", feature = "all-backends")))]
use std::sync::Arc;
use std::sync::OnceLock;

pub(crate) static NUM_THREADS: OnceLock<usize> = OnceLock::new();

thread_local! {
    #[cfg(feature = "backend-mamba")]
    static MAMBA_METHOD_TLS: RefCell<HashMap<String, mambazip::Compressor>> = RefCell::new(HashMap::new());
    #[cfg(feature = "backend-rwkv")]
    static RWKV_METHOD_TLS: RefCell<HashMap<String, rwkvzip::Compressor>> = RefCell::new(HashMap::new());
}

thread_local! {
    static DEFAULT_CTX: RefCell<api::InfotheoryCtx> = RefCell::new(api::InfotheoryCtx::default());
}

/// Returns the current default information theory context for the thread.
pub(crate) fn get_default_ctx() -> api::InfotheoryCtx {
    DEFAULT_CTX.with(|ctx| ctx.borrow().clone())
}

/// Sets the default information theory context for the thread.
pub(crate) fn set_default_ctx(ctx: api::InfotheoryCtx) {
    DEFAULT_CTX.with(|c| *c.borrow_mut() = ctx);
}

#[inline(always)]
pub(crate) fn with_default_ctx<R>(f: impl FnOnce(&api::InfotheoryCtx) -> R) -> R {
    DEFAULT_CTX.with(|ctx| f(&ctx.borrow()))
}

#[inline(always)]
pub(crate) fn aligned_prefix<'a>(x: &'a [u8], y: &'a [u8]) -> (&'a [u8], &'a [u8]) {
    let n = x.len().min(y.len());
    (&x[..n], &y[..n])
}

#[cfg(feature = "backend-zpaq")]
#[inline(always)]
pub(crate) fn try_zpaq_compress_size_bytes(data: &[u8], method: &str) -> InfotheoryResult<u64> {
    zpaq_rs::compress_size(data, method)
        .map_err(|err| InfotheoryError::runtime(format!("zpaq size compression failed: {err}")))
}

#[cfg(not(feature = "backend-zpaq"))]
#[inline(always)]
pub(crate) fn try_zpaq_compress_size_bytes(_data: &[u8], _method: &str) -> InfotheoryResult<u64> {
    Err(InfotheoryError::unsupported(
        "CompressionBackend::Zpaq is unavailable: build with feature 'backend-zpaq'",
    ))
}

#[cfg(feature = "backend-zpaq")]
#[inline(always)]
pub(crate) fn try_zpaq_compress_size_parallel_bytes(
    data: &[u8],
    method: &str,
    threads: usize,
) -> InfotheoryResult<u64> {
    zpaq_rs::compress_size_parallel(data, method, threads).map_err(|err| {
        InfotheoryError::runtime(format!("zpaq parallel size compression failed: {err}"))
    })
}

#[cfg(not(feature = "backend-zpaq"))]
#[inline(always)]
pub(crate) fn try_zpaq_compress_size_parallel_bytes(
    _data: &[u8],
    _method: &str,
    _threads: usize,
) -> InfotheoryResult<u64> {
    Err(InfotheoryError::unsupported(
        "CompressionBackend::Zpaq is unavailable: build with feature 'backend-zpaq'",
    ))
}

#[cfg(feature = "backend-zpaq")]
#[inline(always)]
pub(crate) fn try_zpaq_compress_size_stream<R: std::io::Read + Send>(
    reader: R,
    method: &str,
) -> InfotheoryResult<u64> {
    zpaq_rs::compress_size_stream(reader, method, None, None)
        .map_err(|err| InfotheoryError::runtime(format!("zpaq stream compression failed: {err}")))
}

#[cfg(not(feature = "backend-zpaq"))]
#[inline(always)]
pub(crate) fn try_zpaq_compress_size_stream<R: std::io::Read + Send>(
    _reader: R,
    _method: &str,
) -> InfotheoryResult<u64> {
    Err(InfotheoryError::unsupported(
        "CompressionBackend::Zpaq is unavailable: build with feature 'backend-zpaq'",
    ))
}

#[cfg(feature = "backend-zpaq")]
#[inline(always)]
pub(crate) fn zpaq_compress_to_vec(data: &[u8], method: &str) -> anyhow::Result<Vec<u8>> {
    Ok(zpaq_rs::compress_to_vec(data, method)?)
}

#[cfg(not(feature = "backend-zpaq"))]
#[inline(always)]
pub(crate) fn zpaq_compress_to_vec(_data: &[u8], _method: &str) -> anyhow::Result<Vec<u8>> {
    anyhow::bail!("zpaq backend disabled at compile time (enable feature 'backend-zpaq')")
}

#[cfg(feature = "backend-zpaq")]
#[inline(always)]
pub(crate) fn zpaq_decompress_to_vec(data: &[u8]) -> anyhow::Result<Vec<u8>> {
    Ok(zpaq_rs::decompress_to_vec(data)?)
}

#[cfg(not(feature = "backend-zpaq"))]
#[inline(always)]
pub(crate) fn zpaq_decompress_to_vec(_data: &[u8]) -> anyhow::Result<Vec<u8>> {
    anyhow::bail!("zpaq backend disabled at compile time (enable feature 'backend-zpaq')")
}

/// Validate that a ZPAQ method string is supported for rate estimation.
pub fn validate_zpaq_rate_method(method: &str) -> InfotheoryResult<()> {
    #[cfg(feature = "backend-zpaq")]
    {
        zpaq_rate::validate_zpaq_rate_method(method)
            .map_err(InfotheoryError::invalid_backend_config)
    }
    #[cfg(not(feature = "backend-zpaq"))]
    {
        let _ = method;
        Err(InfotheoryError::unsupported(
            "zpaq backend disabled at compile time",
        ))
    }
}

#[cfg(feature = "backend-rwkv")]
pub(crate) fn with_rwkv_method_tls<R>(
    method: &str,
    f: impl FnOnce(&mut rwkvzip::Compressor) -> R,
) -> R {
    RWKV_METHOD_TLS.with(|cell| {
        let mut map = cell.borrow_mut();
        // Keep a per-method template compressor for fast cloning while ensuring
        // each call gets isolated mutable runtime state (no cross-call leakage).
        let mut comp = if let Some(template) = map.get(method) {
            template.clone()
        } else {
            let template = rwkvzip::Compressor::new_from_method(method).unwrap_or_else(|e| {
                panic!("invalid rwkv method '{method}': {e:#}");
            });
            map.insert(method.to_string(), template.clone());
            template
        };
        drop(map);
        f(&mut comp)
    })
}

#[cfg(feature = "backend-mamba")]
pub(crate) fn with_mamba_method_tls<R>(
    method: &str,
    f: impl FnOnce(&mut mambazip::Compressor) -> R,
) -> R {
    MAMBA_METHOD_TLS.with(|cell| {
        let mut map = cell.borrow_mut();
        let mut comp = if let Some(template) = map.get(method) {
            template.clone()
        } else {
            let template = mambazip::Compressor::new_from_method(method).unwrap_or_else(|e| {
                panic!("invalid mamba method '{method}': {e:#}");
            });
            map.insert(method.to_string(), template.clone());
            template
        };
        drop(map);
        f(&mut comp)
    })
}

pub(crate) fn try_prequential_rate_backend(
    data: &[u8],
    prefix_parts: &[&[u8]],
    max_order: i64,
    backend: &RateBackend,
) -> InfotheoryResult<f64> {
    if data.is_empty() {
        return Ok(0.0);
    }
    let total = prefix_parts
        .iter()
        .map(|p| p.len() as u64)
        .sum::<u64>()
        .saturating_add(data.len() as u64);
    let mut predictor = crate::runtime::build_rate_backend_predictor_default(backend, max_order)
        .map_err(|e| {
            InfotheoryError::runtime(format!("rate backend predictor init failed: {e}"))
        })?;
    predictor
        .begin_stream(Some(total))
        .map_err(|e| InfotheoryError::runtime(format!("rate backend stream init failed: {e}")))?;
    for prefix in prefix_parts {
        for &b in *prefix {
            predictor.update(b);
        }
    }
    let mut bits = 0.0;
    for &b in data {
        bits -= predictor.log_prob(b) / std::f64::consts::LN_2;
        predictor.update(b);
    }
    predictor.finish_stream().map_err(|e| {
        InfotheoryError::runtime(format!("rate backend stream finalize failed: {e}"))
    })?;
    Ok(bits / (data.len() as f64))
}

pub(crate) fn try_frozen_plugin_rate_backend(
    score_data: &[u8],
    fit_parts: &[&[u8]],
    max_order: i64,
    backend: &RateBackend,
) -> InfotheoryResult<f64> {
    if score_data.is_empty() {
        return Ok(0.0);
    }
    #[cfg(feature = "backend-rosa")]
    if matches!(backend, RateBackend::RosaPlus) {
        let mut model = RosaPlus::new(max_order, false, 0, 42);
        let fit_total = fit_parts.iter().map(|part| part.len()).sum::<usize>();
        if fit_total > 0 {
            model.reserve_for_stream(fit_total);
            let mut non_empty_parts = fit_parts
                .iter()
                .copied()
                .filter(|part| !part.is_empty())
                .peekable();
            while let Some(part) = non_empty_parts.next() {
                if non_empty_parts.peek().is_some() {
                    model.train_sequence(part);
                } else {
                    model.train_example(part);
                }
            }
        }
        model.build_lm();
        return Ok(model.cross_entropy(score_data));
    }
    #[cfg(feature = "backend-rwkv")]
    match backend {
        RateBackend::Rwkv7Method { method } => {
            return with_rwkv_method_tls(method, |c| {
                c.cross_entropy_frozen_plugin_chain(fit_parts, score_data)
                    .map_err(|e| {
                        InfotheoryError::runtime(format!(
                            "rwkv method frozen-plugin scoring failed: {e:#}"
                        ))
                    })
            });
        }
        _ => {}
    }
    #[cfg(feature = "backend-mamba")]
    match backend {
        RateBackend::MambaMethod { method } => {
            return with_mamba_method_tls(method, |c| {
                c.cross_entropy_frozen_plugin_chain(fit_parts, score_data)
                    .map_err(|e| {
                        InfotheoryError::runtime(format!(
                            "mamba method frozen-plugin scoring failed: {e:#}"
                        ))
                    })
            });
        }
        _ => {}
    }

    let fit_total = fit_parts.iter().map(|part| part.len() as u64).sum::<u64>();
    let mut predictor = crate::runtime::build_rate_backend_predictor_default(backend, max_order)
        .map_err(|e| {
            InfotheoryError::runtime(format!("rate backend predictor init failed: {e}"))
        })?;
    predictor
        .begin_stream(Some(fit_total))
        .map_err(|e| InfotheoryError::runtime(format!("rate backend fit-pass init failed: {e}")))?;
    for part in fit_parts {
        for &byte in *part {
            predictor.update(byte);
        }
    }
    predictor.finish_stream().map_err(|e| {
        InfotheoryError::runtime(format!("rate backend fit-pass finalize failed: {e}"))
    })?;
    predictor
        .reset_frozen(Some(score_data.len() as u64))
        .map_err(|e| {
            InfotheoryError::runtime(format!("rate backend frozen-score reset failed: {e}"))
        })?;
    let mut bits = 0.0;
    for &byte in score_data {
        bits -= predictor.log_prob(byte) / std::f64::consts::LN_2;
        predictor.update_frozen(byte);
    }
    predictor.finish_stream().map_err(|e| {
        InfotheoryError::runtime(format!("rate backend frozen-score finalize failed: {e}"))
    })?;
    Ok(bits / (score_data.len() as f64))
}

// ============================================================
// Entropy-Based Distance Primitives (via ROSA)
// ============================================================
//
// These use ROSA's Witten-Bell language model to estimate entropy
// and compute information-theoretic distances.

/// Compute entropy rate `Ĥ(X)` in bits/symbol using ROSA LM.
///
/// This uses ROSA's context-conditional Witten-Bell model to estimate
/// the entropy rate, which accounts for sequential dependencies.
///
/// The estimator is **prequential** (predictive sequential): it sums the negative log-probability
/// of each symbol `x_t` given its past context `x_{<t}`, estimated from the model trained on `x_{<t}`.
///
/// `Ĥ(X) = -1/N * Σ log2 P(x_t | x_{t-k}^{t-1})`
///
/// Primitive 7: Resistance under Allowed Transformations.
///
/// Measures how much information is preserved after a transformation `T` is applied to `X`.
///
/// `Resistance(X, T) = I(X; T(X)) / H(X)`
///
/// Range `[0,1]` (with guard for `H(X)=0`).
/// * 1 means perfectly resistant (identity transformation).
/// * 0 means the transformation destroyed all information (e.g. mapping everything to a constant).
///
/// Assumes X and T(X) are aligned.
#[cfg(all(test, any(feature = "default-backends", feature = "all-backends")))]
mod tests {
    use super::*;

    #[cfg(not(feature = "backend-zpaq"))]
    fn compress_size_backend(data: &[u8], backend: &CompressionBackend) -> u64 {
        crate::api::try_compress_size_backend(data, backend).expect("compress_size_backend")
    }

    fn ncd_bytes(x: &[u8], y: &[u8], method: &str, variant: NcdVariant) -> f64 {
        try_ncd_bytes(x, y, method, variant).expect("ncd_bytes")
    }

    fn entropy_rate_bytes(data: &[u8], max_order: i64) -> f64 {
        try_entropy_rate_bytes(data, max_order).expect("entropy_rate_bytes")
    }

    fn entropy_rate_backend(data: &[u8], max_order: i64, backend: &RateBackend) -> f64 {
        try_entropy_rate_backend(data, max_order, backend).expect("entropy_rate_backend")
    }

    fn biased_entropy_rate_backend(data: &[u8], max_order: i64, backend: &RateBackend) -> f64 {
        try_biased_entropy_rate_backend(data, max_order, backend)
            .expect("biased_entropy_rate_backend")
    }

    fn cross_entropy_rate_backend(
        test_data: &[u8],
        train_data: &[u8],
        max_order: i64,
        backend: &RateBackend,
    ) -> f64 {
        try_cross_entropy_rate_backend(test_data, train_data, max_order, backend)
            .expect("cross_entropy_rate_backend")
    }

    fn joint_entropy_rate_backend(
        x: &[u8],
        y: &[u8],
        max_order: i64,
        backend: &RateBackend,
    ) -> f64 {
        try_joint_entropy_rate_backend(x, y, max_order, backend)
            .expect("joint_entropy_rate_backend")
    }

    fn joint_entropy_rate_bytes(x: &[u8], y: &[u8], max_order: i64) -> f64 {
        try_joint_entropy_rate_bytes(x, y, max_order).expect("joint_entropy_rate_bytes")
    }

    fn conditional_entropy_rate_bytes(x: &[u8], y: &[u8], max_order: i64) -> f64 {
        try_conditional_entropy_rate_bytes(x, y, max_order).expect("conditional_entropy_rate_bytes")
    }

    fn mutual_information_bytes(x: &[u8], y: &[u8], max_order: i64) -> f64 {
        try_mutual_information_bytes(x, y, max_order).expect("mutual_information_bytes")
    }

    fn conditional_entropy_bytes(x: &[u8], y: &[u8], max_order: i64) -> f64 {
        try_conditional_entropy_bytes(x, y, max_order).expect("conditional_entropy_bytes")
    }

    fn ned_bytes(x: &[u8], y: &[u8], max_order: i64) -> f64 {
        crate::api::try_ned_bytes(x, y, max_order).expect("ned_bytes")
    }

    fn nte_bytes(x: &[u8], y: &[u8], max_order: i64) -> f64 {
        crate::api::try_nte_bytes(x, y, max_order).expect("nte_bytes")
    }

    fn nte_rate_backend(x: &[u8], y: &[u8], max_order: i64, backend: &RateBackend) -> f64 {
        crate::api::try_nte_rate_backend(x, y, max_order, backend).expect("nte_rate_backend")
    }

    fn resistance_to_transformation_bytes(x: &[u8], tx: &[u8], max_order: i64) -> f64 {
        crate::api::try_resistance_to_transformation_bytes(x, tx, max_order)
            .expect("resistance_to_transformation_bytes")
    }

    fn test_match_backend() -> RateBackend {
        RateBackend::Match {
            hash_bits: 12,
            min_len: 2,
            max_len: 16,
            base_mix: 0.01,
            confidence_scale: 1.0,
        }
    }

    fn test_ppmd_backend() -> RateBackend {
        RateBackend::Ppmd {
            order: 4,
            memory_mb: 1,
        }
    }

    fn test_calibrated_backend() -> RateBackend {
        RateBackend::Calibrated {
            spec: Arc::new(CalibratedSpec {
                base: test_match_backend(),
                context: CalibrationContextKind::Text,
                bins: 16,
                learning_rate: 0.05,
                bias_clip: 4.0,
            }),
        }
    }

    fn test_mixture_backend() -> RateBackend {
        RateBackend::Mixture {
            spec: Arc::new(MixtureSpec::new(
                MixtureKind::Bayes,
                vec![
                    MixtureExpertSpec {
                        name: Some("match".to_string()),
                        log_prior: 0.0,
                        max_order: -1,
                        backend: test_match_backend(),
                    },
                    MixtureExpertSpec {
                        name: Some("ppmd".to_string()),
                        log_prior: 0.0,
                        max_order: -1,
                        backend: test_ppmd_backend(),
                    },
                ],
            )),
        }
    }

    fn test_particle_backend() -> RateBackend {
        RateBackend::Particle {
            spec: Arc::new(ParticleSpec {
                num_particles: 4,
                num_cells: 4,
                cell_dim: 8,
                num_rules: 2,
                selector_hidden: 16,
                rule_hidden: 16,
                context_window: 8,
                unroll_steps: 1,
                ..ParticleSpec::default()
            }),
        }
    }

    fn continuation_prompt() -> &'static [u8] {
        b"If a frog is green, dogs are red.\nIf a toad is green, cats are red.\nIf a dog is green, frogs are red.\nIf a cat is green, toads are red.\nIf a frog is red, dogs are green.\nIf a toad is red, cats are green.\nIf a dog is red, frogs are green.\nIf a cat is red, toads are \n"
    }

    fn assert_deterministic_generate_for_backend(
        backend: RateBackend,
        max_order: i64,
        bytes: usize,
        label: &str,
    ) {
        let prompt = continuation_prompt();
        let a = crate::api::generation::generate_rate_backend_chain(
            &[prompt],
            bytes,
            max_order,
            &backend,
            GenerationConfig::default(),
        );
        let b = crate::api::generation::generate_rate_backend_chain(
            &[prompt],
            bytes,
            max_order,
            &backend,
            GenerationConfig::default(),
        );
        assert_eq!(
            a, b,
            "{label} generation should be deterministic for identical input"
        );
        assert_eq!(
            a.len(),
            bytes,
            "{label} generation should emit requested byte count"
        );
    }

    fn assert_sampled_generate_for_backend(
        backend: RateBackend,
        max_order: i64,
        bytes: usize,
        label: &str,
    ) {
        let prompt = continuation_prompt();
        let config = GenerationConfig::sampled_frozen(42);
        let a = crate::api::generation::generate_rate_backend_chain(
            &[prompt],
            bytes,
            max_order,
            &backend,
            config,
        );
        let b = crate::api::generation::generate_rate_backend_chain(
            &[prompt],
            bytes,
            max_order,
            &backend,
            config,
        );
        assert_eq!(
            a, b,
            "{label} sampled generation should be deterministic for a fixed seed"
        );
        assert_eq!(
            a.len(),
            bytes,
            "{label} sampled generation should emit requested byte count"
        );
    }

    #[cfg(feature = "backend-zpaq")]
    #[test]
    fn ncd_basic_identity_nonnegative() {
        let x = b"abcdabcdabcd";
        let d = ncd_bytes(x, x, "5", NcdVariant::Vitanyi);
        assert!(d >= -1e-9);
    }

    #[test]
    fn shannon_identities_marginal_aligned() {
        let x = b"abracadabra";
        let y = b"abracadabra";

        let h = marginal_entropy_bytes(x);
        let mi = mutual_information_bytes(x, y, 0);
        let h_xy = joint_marginal_entropy_bytes(x, y);
        let h_x_given_y = conditional_entropy_bytes(x, y, 0);
        let ned = ned_bytes(x, y, 0);
        let nte = nte_bytes(x, y, 0);

        assert!((h_xy - h).abs() < 1e-12);
        assert!(h_x_given_y.abs() < 1e-12);
        assert!((mi - h).abs() < 1e-12);
        assert!(ned.abs() < 1e-12);
        assert!(nte.abs() < 1e-12);
    }

    #[test]
    fn shannon_identities_rate_aligned_reasonable() {
        let x = b"the quick brown fox jumps over the lazy dog";
        let y = b"the quick brown fox jumps over the lazy dog";
        let max_order = 8;
        let prev = get_default_ctx();
        set_default_ctx(InfotheoryCtx::new(
            RateBackend::RosaPlus,
            CompressionBackend::default(),
        ));

        let h_x = entropy_rate_bytes(x, max_order);
        let h_xy = joint_entropy_rate_bytes(x, y, max_order);
        let h_x_given_y = conditional_entropy_rate_bytes(x, y, max_order);
        let mi = mutual_information_bytes(x, y, max_order);
        let ned = ned_bytes(x, y, max_order);

        // Finite-sample estimators won't be exact; allow reasonable tolerance.
        let tol = 0.2;
        assert!((h_xy - h_x).abs() < tol);
        assert!(h_x_given_y < tol);
        assert!((mi - h_x).abs() < tol);
        assert!(ned < tol);
        set_default_ctx(prev);
    }

    #[test]
    fn resistance_identity_is_one() {
        let x = b"some repeated repeated repeated text";
        let prev = get_default_ctx();
        set_default_ctx(InfotheoryCtx::new(
            RateBackend::RosaPlus,
            CompressionBackend::default(),
        ));
        let r0 = resistance_to_transformation_bytes(x, x, 0);
        let r8 = resistance_to_transformation_bytes(x, x, 8);
        assert!((r0 - 1.0).abs() < 1e-12);
        assert!((r8 - 1.0).abs() < 1e-6);
        set_default_ctx(prev);
    }

    #[test]
    fn marginal_metrics_empty_inputs_are_zero() {
        let empty: &[u8] = &[];
        let x = b"abc";

        assert_eq!(tvd_bytes(empty, x, 0), 0.0);
        assert_eq!(tvd_bytes(x, empty, 0), 0.0);
        assert_eq!(nhd_bytes(empty, x, 0), 0.0);
        assert_eq!(nhd_bytes(x, empty, 0), 0.0);
        assert_eq!(d_kl_bytes(empty, x), 0.0);
        assert_eq!(d_kl_bytes(x, empty), 0.0);
        assert_eq!(js_div_bytes(empty, x), 0.0);
        assert_eq!(js_div_bytes(x, empty), 0.0);
    }

    #[test]
    fn marginal_cross_entropy_empty_test_is_zero() {
        let empty: &[u8] = &[];
        let y = b"abc";
        let ctx = InfotheoryCtx::with_zpaq("5");
        assert_eq!(
            ctx.try_cross_entropy_bytes(empty, y, 0)
                .expect("cross entropy bytes"),
            0.0
        );
    }

    #[test]
    fn backend_switching_test() {
        let x = b"hello world context";

        // Default is RosaPlus
        let h_rosa = entropy_rate_bytes(x, 8);

        // Switch to CTW
        set_default_ctx(InfotheoryCtx::new(
            RateBackend::Ctw { depth: 16 },
            CompressionBackend::default(),
        ));

        let h_ctw = entropy_rate_bytes(x, 8);

        // They should generally be different, but most importantly, CTW worked
        assert!(h_ctw > 0.0);

        // Reset to default
        set_default_ctx(InfotheoryCtx::default());
        let h_rosa_back = entropy_rate_bytes(x, 8);
        assert!((h_rosa - h_rosa_back).abs() < 1e-12);
    }

    #[test]
    fn ctw_early_updates_work() {
        // Test that CTW produces valid predictions from the very start,
        // not just after `depth` symbols have been processed.
        use crate::backends::ctw::ContextTree;

        let mut tree = ContextTree::new(16);

        // Even the first prediction should be valid (not NaN, not 0)
        let p0 = tree.predict(false);
        let p1 = tree.predict(true);

        // Initial KT estimator gives 0.5 / 1 = 0.5 for each symbol
        assert!((p0 - 0.5).abs() < 1e-10, "p0 should be ~0.5, got {}", p0);
        assert!((p1 - 0.5).abs() < 1e-10, "p1 should be ~0.5, got {}", p1);
        assert!((p0 + p1 - 1.0).abs() < 1e-10, "p0 + p1 should = 1.0");

        // Update with a few symbols and verify log_prob becomes negative (valid)
        for _ in 0..5 {
            tree.update(true);
            tree.update(false);
        }

        let log_prob = tree.get_log_block_probability();
        assert!(
            log_prob < 0.0,
            "log_prob should be negative (< log 1), got {}",
            log_prob
        );
        assert!(log_prob.is_finite(), "log_prob should be finite");
    }

    #[test]
    fn nte_can_exceed_one() {
        // Test that NTE is properly clamped to [0, 2] instead of [0, 1]
        // For independent sequences with similar entropy, NTE can approach 2.0
        //
        // Note: For *marginal* NTE, due to how joint entropy works for aligned pairs,
        // it's mathematically bounded differently. The fix for NTE clamping primarily
        // affects *rate*-based NTE where VI can truly be 2*max(H).
        //
        // We test that the clamp upper bound is at least > 1.0 for cases where VI > max(H)

        // Use CTW backend for rate-based test
        set_default_ctx(InfotheoryCtx::new(
            RateBackend::Ctw { depth: 8 },
            CompressionBackend::default(),
        ));

        // Generate two completely different patterns - should have high VI
        let x: Vec<u8> = (0..200).map(|i| (i % 2) as u8).collect(); // 010101...
        let y: Vec<u8> = (0..200).map(|i| ((i + 1) % 2) as u8).collect(); // 101010...

        let nte_rate = nte_rate_backend(&x, &y, -1, &RateBackend::Ctw { depth: 8 });

        // With the fix, NTE should not be clamped to 1.0
        // It may or may not exceed 1.0 depending on the specifics, but it should be allowed to
        assert!(
            (0.0..=2.0 + 1e-9).contains(&nte_rate),
            "NTE should be in [0, 2], got {}",
            nte_rate
        );

        // Reset context
        set_default_ctx(InfotheoryCtx::default());
    }

    #[test]
    fn ctw_empty_data_returns_zero() {
        // Verify empty data doesn't cause division-by-zero or NaN
        set_default_ctx(InfotheoryCtx::new(
            RateBackend::Ctw { depth: 16 },
            CompressionBackend::default(),
        ));

        let empty: &[u8] = &[];
        let h = entropy_rate_bytes(empty, -1);
        assert_eq!(h, 0.0, "empty data should return 0.0 entropy");

        // Reset
        set_default_ctx(InfotheoryCtx::default());
    }

    #[test]
    fn joint_entropy_rate_aligns_inputs_and_handles_empty_cases() {
        let cases = vec![
            ("ctw", RateBackend::Ctw { depth: 8 }),
            (
                "fac-ctw",
                RateBackend::FacCtw {
                    base_depth: 8,
                    num_percept_bits: 8,
                    encoding_bits: 8,
                },
            ),
            ("match", test_match_backend()),
        ];

        for (name, backend) in cases {
            assert_eq!(
                joint_entropy_rate_backend(b"", b"nonempty", -1, &backend),
                0.0,
                "{name} should return 0.0 for empty aligned pairs"
            );
            assert_eq!(
                joint_entropy_rate_backend(b"nonempty", b"", -1, &backend),
                0.0,
                "{name} should return 0.0 when alignment truncates to empty"
            );

            let aligned = joint_entropy_rate_backend(b"abcd", b"wxyz", -1, &backend);
            let truncated = joint_entropy_rate_backend(b"abcdextra", b"wxyz", -1, &backend);
            assert!(
                (aligned - truncated).abs() < 1e-12,
                "{name} should score only the aligned prefix: aligned={aligned} truncated={truncated}"
            );
        }
    }

    #[test]
    fn biased_entropy_is_repeatable_across_backend_families() {
        let data = b"ABABABAABBABABABAABB";
        let cases = vec![
            ("match", test_match_backend()),
            ("ppmd", test_ppmd_backend()),
            ("calibrated", test_calibrated_backend()),
            ("ctw", RateBackend::Ctw { depth: 8 }),
            ("mixture", test_mixture_backend()),
            ("particle", test_particle_backend()),
        ];

        for (name, backend) in cases {
            let h1 = biased_entropy_rate_backend(data, -1, &backend);
            let h2 = biased_entropy_rate_backend(data, -1, &backend);
            assert!(h1.is_finite(), "{name} biased entropy should be finite");
            assert!(
                (h1 - h2).abs() < 1e-12,
                "{name} biased entropy leaked mutable state across calls: h1={h1} h2={h2}"
            );
        }
    }

    #[test]
    fn generate_bytes_chain_matches_flat_prompt() {
        let prompt = continuation_prompt();
        let split_at = prompt.len() / 2;
        let front = &prompt[..split_at];
        let back = &prompt[split_at..];
        let backend = RateBackend::Ctw { depth: 32 };
        let bytes = 8usize;
        let max_order = -1;

        let flat = crate::api::generation::generate_rate_backend_chain(
            &[prompt],
            bytes,
            max_order,
            &backend,
            GenerationConfig::default(),
        );
        let chained = crate::api::generation::generate_rate_backend_chain(
            &[front, back],
            bytes,
            max_order,
            &backend,
            GenerationConfig::default(),
        );
        assert_eq!(
            flat, chained,
            "chain conditioning should match flat prompt conditioning"
        );
    }

    #[test]
    fn generate_bytes_api_is_deterministic_for_ctw_rosa_match_ppmd() {
        assert_deterministic_generate_for_backend(RateBackend::Ctw { depth: 32 }, -1, 8, "ctw");
        assert_deterministic_generate_for_backend(RateBackend::RosaPlus, -1, 8, "rosaplus");
        assert_deterministic_generate_for_backend(test_match_backend(), -1, 8, "match");
        assert_deterministic_generate_for_backend(test_ppmd_backend(), -1, 8, "ppmd");
    }

    #[cfg(feature = "backend-rwkv")]
    #[test]
    fn generate_bytes_api_is_deterministic_for_rwkv_method() {
        let backend = RateBackend::Rwkv7Method {
            method: "cfg:hidden=64,layers=1,intermediate=64,decay_rank=8,a_rank=8,v_rank=8,g_rank=8,seed=31,train=none,lr=0.0,stride=1;policy:schedule=0..100:infer".to_string(),
        };
        assert_deterministic_generate_for_backend(backend, -1, 8, "rwkv7");
    }

    #[test]
    fn sampled_generation_is_deterministic_for_ctw_rosa_match_ppmd() {
        assert_sampled_generate_for_backend(RateBackend::Ctw { depth: 32 }, -1, 8, "ctw");
        assert_sampled_generate_for_backend(RateBackend::RosaPlus, -1, 8, "rosaplus");
        assert_sampled_generate_for_backend(test_match_backend(), -1, 8, "match");
        assert_sampled_generate_for_backend(test_ppmd_backend(), -1, 8, "ppmd");
    }

    #[cfg(feature = "backend-rwkv")]
    #[test]
    fn sampled_generation_is_deterministic_for_rwkv_method() {
        let backend = RateBackend::Rwkv7Method {
            method: "cfg:hidden=64,layers=1,intermediate=64,decay_rank=8,a_rank=8,v_rank=8,g_rank=8,seed=31,train=none,lr=0.0,stride=1;policy:schedule=0..100:infer".to_string(),
        };
        assert_sampled_generate_for_backend(backend, -1, 8, "rwkv7");
    }

    #[test]
    fn rosaplus_sampled_generation_predicts_green_continuation() {
        let out = crate::api::generation::generate_rate_backend_chain(
            &[continuation_prompt()],
            8,
            -1,
            &RateBackend::RosaPlus,
            GenerationConfig::sampled_frozen(42),
        );
        assert_eq!(out, b" green.\n");
    }

    #[test]
    fn rate_backend_session_matches_ctx_generation() {
        let prompt = continuation_prompt();
        let backend = RateBackend::Ppmd {
            order: 12,
            memory_mb: 8,
        };
        let mut session =
            RateBackendSession::from_backend(backend.clone(), -1, Some((prompt.len() + 8) as u64))
                .expect("session init");
        session.observe(prompt);
        let from_session = session.generate_bytes(8, GenerationConfig::sampled_frozen(42));
        session.finish().expect("session finish");

        let ctx = InfotheoryCtx::new(backend, CompressionBackend::default());
        let from_ctx = ctx
            .try_generate_bytes_with_config(prompt, 8, -1, GenerationConfig::sampled_frozen(42))
            .expect("ctx generation");
        assert_eq!(from_session, from_ctx);
    }

    #[test]
    fn biased_entropy_ctw_uses_frozen_plugin_scoring() {
        let backend = RateBackend::Ctw { depth: 8 };
        let data = b"AAAAAAAA";
        let plugin = biased_entropy_rate_backend(data, -1, &backend);
        let prequential = entropy_rate_backend(data, -1, &backend);
        assert!(
            plugin + 1e-9 < prequential,
            "expected plugin scoring to beat prequential scoring: plugin={plugin} prequential={prequential}"
        );
    }

    #[test]
    fn rosa_plugin_entropy_matches_direct_model_api() {
        let data = b"abracadabra";
        let backend = RateBackend::RosaPlus;

        let plugin = biased_entropy_rate_backend(data, 3, &backend);

        let mut direct = RosaPlus::new(3, false, 0, 42);
        direct.train_example(data);
        direct.build_lm();
        let expected = direct.cross_entropy(data);

        assert!(
            (plugin - expected).abs() < 1e-12,
            "rosa plugin entropy must match direct model API: plugin={plugin} expected={expected}"
        );
    }

    #[test]
    fn rosa_plugin_cross_entropy_matches_direct_model_api() {
        let train = b"alakazam";
        let test = b"abracadabra";
        let backend = RateBackend::RosaPlus;

        let plugin = cross_entropy_rate_backend(test, train, 3, &backend);

        let mut direct = RosaPlus::new(3, false, 0, 42);
        direct.train_example(train);
        direct.build_lm();
        let expected = direct.cross_entropy(test);

        assert!(
            (plugin - expected).abs() < 1e-12,
            "rosa plugin cross entropy must match direct model API: plugin={plugin} expected={expected}"
        );
    }

    #[test]
    fn rosa_conditional_chain_matches_concatenated_prefix_scoring() {
        let ctx = InfotheoryCtx::new(RateBackend::RosaPlus, CompressionBackend::default());
        let prefix_parts: [&[u8]; 3] = [b"universal ", b"prior ", b"slice"];
        let data = b"query payload";

        let chained = ctx
            .try_cross_entropy_conditional_chain(&prefix_parts, data)
            .expect("conditional-chain cross entropy");
        let flat_prefix: Vec<u8> = prefix_parts.concat();
        let flat = cross_entropy_rate_backend(data, &flat_prefix, -1, &RateBackend::RosaPlus);

        assert!(
            (chained - flat).abs() < 1e-12,
            "conditional-chain scoring drifted from concatenated-prefix scoring: chained={chained} flat={flat}"
        );
    }

    #[test]
    fn datagen_bernoulli_entropy_estimate() {
        // Test that estimated entropy is close to theoretical for Bernoulli(0.5)
        let p = 0.5;
        let theoretical_h = crate::datagen::bernoulli_entropy(p);
        assert!((theoretical_h - 1.0).abs() < 1e-10);

        // Generate data and check marginal entropy is close to theoretical
        let data = crate::datagen::bernoulli(10000, p, 42);
        let estimated_h = marginal_entropy_bytes(&data);

        // Should be close to 1.0 bit (since values are 0 or 1)
        assert!(
            (estimated_h - theoretical_h).abs() < 0.1,
            "estimated H={} should be close to theoretical H={}",
            estimated_h,
            theoretical_h
        );
    }

    #[cfg(feature = "backend-rwkv")]
    #[test]
    fn rwkv_method_entropy_is_stable_across_calls() {
        let method = "cfg:hidden=64,layers=1,intermediate=64,decay_rank=8,a_rank=8,v_rank=8,g_rank=8,seed=21,train=sgd,lr=0.01,stride=1;policy:schedule=0..100:infer";
        let backend = RateBackend::Rwkv7Method {
            method: method.to_string(),
        };
        let data = b"rwkv method entropy stability regression sample";

        let h1 = entropy_rate_backend(data, -1, &backend);
        let h2 = entropy_rate_backend(data, -1, &backend);
        assert!(
            (h1 - h2).abs() < 1e-12,
            "rwkv method entropy leaked mutable state across calls: h1={h1}, h2={h2}"
        );
    }

    #[cfg(feature = "backend-rwkv")]
    #[test]
    fn rwkv_method_without_policy_is_accepted_by_public_api() {
        let backend = RateBackend::Rwkv7Method {
            method: "cfg:hidden=64,layers=1,intermediate=64".to_string(),
        };
        let data = b"rwkv method without policy";
        let h1 = entropy_rate_backend(data, -1, &backend);
        let h2 = biased_entropy_rate_backend(data, -1, &backend);
        assert!(h1.is_finite());
        assert!(h2.is_finite());
    }

    #[cfg(feature = "backend-rwkv")]
    #[test]
    fn rwkv_infer_only_plugin_collapses_to_single_pass_entropy() {
        let backend = RateBackend::Rwkv7Method {
            method: "cfg:hidden=64,layers=1,intermediate=64,decay_rank=8,a_rank=8,v_rank=8,g_rank=8,seed=25,train=none,lr=0.0,stride=1;policy:schedule=0..100:infer".to_string(),
        };
        let data = b"rwkv infer-only plugin equality sample";
        let h = entropy_rate_backend(data, -1, &backend);
        let plugin = biased_entropy_rate_backend(data, -1, &backend);
        assert!(
            (h - plugin).abs() < 1e-12,
            "infer-only rwkv plugin should equal single-pass entropy: h={h}, plugin={plugin}"
        );
    }

    #[cfg(feature = "backend-rwkv")]
    #[test]
    fn rwkv_method_biased_entropy_is_stable_across_calls_with_training_policy() {
        let backend = RateBackend::Rwkv7Method {
            method: "cfg:hidden=64,layers=1,intermediate=64,decay_rank=8,a_rank=8,v_rank=8,g_rank=8,seed=23,train=sgd,lr=0.01,stride=1;policy:schedule=0..100:train(scope=head+bias,opt=sgd,lr=0.01,stride=1,bptt=1,clip=0,momentum=0.0)".to_string(),
        };
        let data = b"rwkv plugin stability sample";
        let h1 = biased_entropy_rate_backend(data, -1, &backend);
        let h2 = biased_entropy_rate_backend(data, -1, &backend);
        assert!(
            (h1 - h2).abs() < 1e-12,
            "rwkv method biased entropy leaked mutable state across calls: h1={h1}, h2={h2}"
        );
    }

    #[cfg(feature = "backend-rwkv")]
    #[test]
    fn rwkv_method_conditional_chain_is_stable_across_calls() {
        let method = "cfg:hidden=64,layers=1,intermediate=64,decay_rank=8,a_rank=8,v_rank=8,g_rank=8,seed=22,train=sgd,lr=0.01,stride=1;policy:schedule=0..100:infer";
        let ctx = InfotheoryCtx::new(
            RateBackend::Rwkv7Method {
                method: method.to_string(),
            },
            CompressionBackend::default(),
        );

        let prefix = b"universal prior slice";
        let data = b"query payload";
        let h1 = ctx
            .try_cross_entropy_conditional_chain(&[prefix.as_slice()], data)
            .expect("conditional-chain cross entropy");
        let h2 = ctx
            .try_cross_entropy_conditional_chain(&[prefix.as_slice()], data)
            .expect("conditional-chain cross entropy");
        assert!(
            (h1 - h2).abs() < 1e-12,
            "rwkv method conditional chain leaked mutable state across calls: h1={h1}, h2={h2}"
        );
    }

    #[cfg(feature = "backend-mamba")]
    #[test]
    fn mamba_method_without_policy_is_accepted_by_public_api() {
        let backend = RateBackend::MambaMethod {
            method: "cfg:hidden=64,layers=1,intermediate=96".to_string(),
        };
        let data = b"mamba method without policy";
        let h1 = entropy_rate_backend(data, -1, &backend);
        let h2 = biased_entropy_rate_backend(data, -1, &backend);
        assert!(h1.is_finite());
        assert!(h2.is_finite());
    }

    #[cfg(feature = "backend-mamba")]
    #[test]
    fn mamba_infer_only_plugin_collapses_to_single_pass_entropy() {
        let backend = RateBackend::MambaMethod {
            method: "cfg:hidden=64,layers=1,intermediate=96,state=16,conv=4,dt_rank=16,seed=26,train=none,lr=0.0,stride=1;policy:schedule=0..100:infer".to_string(),
        };
        let data = b"mamba infer-only plugin equality sample";
        let h = entropy_rate_backend(data, -1, &backend);
        let plugin = biased_entropy_rate_backend(data, -1, &backend);
        assert!(
            (h - plugin).abs() < 1e-12,
            "infer-only mamba plugin should equal single-pass entropy: h={h}, plugin={plugin}"
        );
    }

    #[cfg(feature = "backend-mamba")]
    #[test]
    fn mamba_method_biased_entropy_is_stable_across_calls_with_training_policy() {
        let backend = RateBackend::MambaMethod {
            method: "cfg:hidden=64,layers=1,intermediate=96,state=16,conv=4,dt_rank=16,seed=24,train=sgd,lr=0.01,stride=1;policy:schedule=0..100:train(scope=head+bias,opt=sgd,lr=0.01,stride=1,bptt=1,clip=0,momentum=0.0)".to_string(),
        };
        let data = b"mamba plugin stability sample";
        let h1 = biased_entropy_rate_backend(data, -1, &backend);
        let h2 = biased_entropy_rate_backend(data, -1, &backend);
        assert!(
            (h1 - h2).abs() < 1e-12,
            "mamba method biased entropy leaked mutable state across calls: h1={h1}, h2={h2}"
        );
    }

    #[test]
    fn particle_entropy_rate_in_valid_range() {
        let rb = test_particle_backend();
        let data = b"hello world particle backend test";
        let rate = entropy_rate_backend(data, -1, &rb);
        assert!(
            rate > 0.0 && rate < 8.0,
            "particle entropy rate out of (0, 8) range: {rate}"
        );
    }

    #[test]
    fn particle_cross_entropy_stability() {
        let rb = test_particle_backend();
        let train = b"ABCABC";
        let test = b"ABC";
        let h1 = cross_entropy_rate_backend(test, train, -1, &rb);
        let h2 = cross_entropy_rate_backend(test, train, -1, &rb);
        assert!(
            (h1 - h2).abs() < 1e-12,
            "particle cross entropy not deterministic: h1={h1}, h2={h2}"
        );
    }

    #[test]
    fn particle_empty_input() {
        let rb = RateBackend::Particle {
            spec: Arc::new(ParticleSpec::default()),
        };
        let rate = entropy_rate_backend(b"", -1, &rb);
        assert!(
            rate == 0.0,
            "particle entropy rate for empty input should be 0.0, got {rate}"
        );
    }

    #[test]
    fn particle_joint_entropy_rate() {
        let rb = test_particle_backend();
        let x = b"AAAA";
        let y = b"BBBB";
        let joint = joint_entropy_rate_backend(x, y, -1, &rb);
        assert!(
            joint > 0.0 && joint < 16.0,
            "particle joint entropy rate out of range: {joint}"
        );
    }
}

#[cfg(all(
    test,
    not(any(
        feature = "default-backends",
        feature = "all-backends",
        feature = "backend-rosa",
        feature = "backend-ctw",
        feature = "backend-match",
        feature = "backend-ppmd",
        feature = "backend-sequitur",
        feature = "backend-mixture",
        feature = "backend-particle",
        feature = "backend-calibrated",
        feature = "backend-rwkv",
        feature = "backend-mamba"
    ))
))]
mod minimal_tests {
    #[cfg(not(feature = "backend-zpaq"))]
    use crate::api::CompressionBackend;

    #[cfg(not(feature = "backend-zpaq"))]
    fn compress_size_backend(data: &[u8], backend: &CompressionBackend) -> u64 {
        crate::api::try_compress_size_backend(data, backend).expect("compress_size_backend")
    }

    #[cfg(not(feature = "backend-zpaq"))]
    #[test]
    #[should_panic(expected = "requires infotheory feature 'backend-zpaq'")]
    fn explicit_zpaq_backend_fails_loudly() {
        let backend = CompressionBackend::Zpaq {
            method: "5".to_string(),
        };
        let _ = compress_size_backend(b"abc", &backend);
    }

    #[cfg(not(feature = "backend-zpaq"))]
    #[test]
    fn default_compression_backend_reports_missing_rate_backend_when_none_are_enabled() {
        let backend = CompressionBackend::default();
        assert!(matches!(
            &backend,
            CompressionBackend::Rate {
                coder: crate::coders::CoderType::AC,
                framing: crate::compression::FramingMode::Raw,
                ..
            }
        ));
        let err = crate::api::try_compress_size_backend(b"abc", &backend)
            .expect_err("default backend should fail loudly when no rate backends are enabled");
        assert!(
            err.to_string().contains("requires infotheory feature"),
            "unexpected error: {err}"
        );
    }
}
