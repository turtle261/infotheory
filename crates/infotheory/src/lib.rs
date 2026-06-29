#![allow(unsafe_op_in_unsafe_fn)]

//! # InfoTheory: Information Theoretic Estimators & Metrics
//!
//! This crate provides a comprehensive suite of information-theoretic primitives for
//! quantifying complexity, dependence, and similarity between data sequences.
//!
//! It implements two complementary classes of estimators:
//! 1.  **Algorithmic information theory (predictive / Kolmogorov
//!     complexity)**: estimates `K(·)`-flavored quantities by treating a
//!     model as a description-length functional — either a compressor
//!     `C(·)` or a sequential predictor used as a prequential code
//!     `-log₂ p(x_t | x_{<t})` — and taking the resulting code length as
//!     a finite, computable proxy for Kolmogorov complexity. Compression
//!     and prediction are unified in the library: any
//!     [`crate::api::RateBackend`] can drive the predictive metrics
//!     directly *and* power the generic rate-coded compressor
//!     `CompressionBackend::Rate`. The class carries the broadest
//!     surface in the library — the Normalized Compression Distance
//!     (NCD) family with its normalization variants and pairwise
//!     matrices, raw and chained compressed-size primitives, the entropy
//!     rate `Ĥ(X)`, joint and conditional entropy rates, cross-entropy,
//!     mutual information, normalized entropy distance (NED), normalized
//!     transform effort (NTE), intrinsic dependence, and resistance to
//!     transformation. Compressors are pluggable through
//!     [`crate::api::CompressionBackend`] (a dedicated ZPAQ family, an
//!     optional RWKV7 compressor, and the rate-coded compressor wrapping
//!     any rate backend). Rate backends are pluggable through
//!     [`crate::api::RateBackend`] — CTW, FAC-CTW, ROSA+, PPMD, Sequitur,
//!     contiguous and sparse local-match models, online RWKV7 and Mamba
//!     neural backends, calibrated wrappers, particle-filter backends,
//!     ZPAQ-as-rate, and arbitrary mixture / ensemble compositions
//!     thereof — and are interchangeable wherever a `RateBackend` is
//!     consumed.
//! 2.  **Shannon information theory (empirical / IID plug-in)**:
//!     estimates classical Shannon quantities directly from observed
//!     byte frequencies, with no learned model. The class is model-free:
//!     it plugs the empirical distribution into Shannon's formulae and
//!     returns an order-0 / IID estimator. It supplies the order-0
//!     entropy `H₀(X)`, joint and conditional `H₀`, `I₀(X;Y)`, and the
//!     `empirical_*` analogues of NED, NTE, cross-entropy, and
//!     resistance, plus the classical divergences and distances over
//!     byte distributions: total variation distance (TVD), normalized
//!     Hellinger distance (NHD), Kullback–Leibler divergence (KL), and
//!     Jensen–Shannon divergence (JSD). These are useful as model-free
//!     baselines, axiom test fixtures, and as the appropriate estimator
//!     when higher-order structure is absent by construction.
//!
//! Many of the same underlying quantities — entropy, mutual information,
//! NED, NTE, cross-entropy, resistance — are exposed in *both* classes,
//! so a caller can pick between an algorithmic / model-driven estimate
//! and a model-free Shannon plug-in for the same target. The algorithmic
//! side is correspondingly broader: it carries the entire backend and
//! compressor ecosystem and the metrics (NCD, compressed-size, the
//! entropy *rate*) that have no order-0 plug-in counterpart.
//!
//! ## Mathematical Primitives
//!
//! The library implements the following core measures. For sequential data,
//! `*_rate_*` and explicit-backend variants use the configured
//! [`crate::api::RateBackend`] to estimate the entropy rate `Ĥ(X)`, while
//! `empirical_*` variants compute the order-0 plug-in `H₀(X)` from byte
//! histograms.
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
//! Measures sequential redundancy by comparing the order-0 / empirical
//! entropy `H₀(X)` against the entropy rate `Ĥ(X)` produced by the
//! configured rate backend.
//!
//! `ID(X) = (H₀(X) - Ĥ(X)) / H₀(X)`
//!
//! ### 7. Resistance to Transformation
//! Quantifies how much information is preserved after a transformation `T` is applied.
//!
//! `R(X, T) = I(X; T(X)) / H(X)`
//!
//! ## Usage
//!
//! ```rust,no_run
//! use infotheory::api::{
//!     empirical_mutual_information_bytes, try_ncd_paths_backend, CompressionBackend, NcdVariant,
//! };
//!
//! let x = b"some data sequence";
//! let y = b"another data sequence";
//!
//! // Compression-based distance using an explicit compression backend.
//! let backend = CompressionBackend::zpaq("5");
//! let ncd = try_ncd_paths_backend("file1.txt", "file2.txt", &backend, NcdVariant::Vitanyi)
//!     .expect("ncd");
//!
//! // Order-0 / IID Shannon mutual information (model-free plug-in baseline).
//! let mi = empirical_mutual_information_bytes(x, y);
//! ```

#[cfg(test)]
extern crate self as infotheory;

/// AIXI planning components, environments, and model abstractions.
pub mod aixi;
/// Public spec-first API reexports.
pub mod api;
/// Core information-theoretic axioms and validation helpers.
pub mod axioms;
/// Entropy/compression backend implementations and backend discovery.
pub mod backends;
pub(crate) mod byte_prefix;
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
pub(crate) mod prediction;
pub(crate) mod rate_defaults;
/// Shared spec -> runtime builders and backend registry metadata.
pub(crate) mod runtime;
/// Information-theoretic code search pipeline (3-stage: prefilter, filter, KMI rerank).
#[cfg(feature = "backend-rosa")]
pub mod search;
#[cfg(feature = "backend-particle")]
pub(crate) mod simd_math;
/// Shared backend/spec parsing and loading helpers.
pub mod spec;
/// Tuner runtime execution controls and CLI-facing tuning entrypoints.
#[cfg(feature = "tuner")]
pub mod tuner;
use crate::api::CompiledRateBackend;
#[cfg(all(test, feature = "all-backends"))]
pub(crate) use crate::api::{
    CalibratedSpec, CalibrationContextKind, MixtureExpertSpec, MixtureKind, MixtureSpec,
    ParticleSpec,
};
#[cfg(all(test, feature = "all-backends"))]
use crate::api::{
    CompressionBackend, GenerationConfig, InfotheoryCtx, NcdVariant, RateBackend,
    RateBackendSession, d_kl_bytes, try_biased_entropy_rate_backend,
    try_conditional_entropy_rate_bytes, try_cross_entropy_rate_backend, try_entropy_rate_backend,
    try_entropy_rate_bytes, try_joint_entropy_rate_backend, try_joint_entropy_rate_bytes,
    try_mutual_information_bytes,
};
#[cfg(all(test, feature = "all-backends"))]
use crate::api::{
    empirical_entropy_bytes, empirical_joint_entropy_bytes, js_div_bytes, nhd_bytes, tvd_bytes,
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
#[cfg(all(test, feature = "all-backends"))]
use std::sync::Arc;

thread_local! {
    #[cfg(feature = "backend-mamba")]
    static MAMBA_METHOD_TLS: RefCell<HashMap<String, mambazip::Compressor>> = RefCell::new(HashMap::new());
    #[cfg(feature = "backend-rwkv")]
    static RWKV_METHOD_TLS: RefCell<HashMap<String, rwkvzip::Compressor>> = RefCell::new(HashMap::new());
}

thread_local! {
    static DEFAULT_CTX: RefCell<Option<api::InfotheoryCtx>> = const { RefCell::new(None) };
}

/// Returns the current default information theory context for the thread.
pub(crate) fn get_default_ctx() -> InfotheoryResult<api::InfotheoryCtx> {
    DEFAULT_CTX.with(|ctx| {
        let mut slot = ctx.borrow_mut();
        if slot.is_none() {
            *slot = Some(api::InfotheoryCtx::try_default()?);
        }
        Ok(slot
            .as_ref()
            .expect("default context initialized above")
            .clone())
    })
}

/// Sets the default information theory context for the thread.
pub(crate) fn set_default_ctx(ctx: api::InfotheoryCtx) {
    DEFAULT_CTX.with(|c| *c.borrow_mut() = Some(ctx));
}

#[inline(always)]
pub(crate) fn with_default_ctx<R>(
    f: impl FnOnce(&api::InfotheoryCtx) -> InfotheoryResult<R>,
) -> InfotheoryResult<R> {
    let ctx = get_default_ctx()?;
    f(&ctx)
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

#[cfg(feature = "backend-zpaq")]
#[inline(always)]
pub(crate) fn try_zpaq_compress_size_stream_parallel<R: std::io::Read + Send>(
    reader: R,
    method: &str,
    threads: usize,
) -> InfotheoryResult<u64> {
    zpaq_rs::compress_size_stream_parallel(reader, method, None, None, threads)
        .map_err(|err| InfotheoryError::runtime(format!("zpaq stream compression failed: {err}")))
}

#[cfg(not(feature = "backend-zpaq"))]
#[inline(always)]
pub(crate) fn try_zpaq_compress_size_stream_parallel<R: std::io::Read + Send>(
    _reader: R,
    _method: &str,
    _threads: usize,
) -> InfotheoryResult<u64> {
    Err(InfotheoryError::unsupported(
        "CompressionBackend::Zpaq is unavailable: build with feature 'backend-zpaq'",
    ))
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
pub(crate) fn with_rwkv_method_spec_tls<R>(
    method: &str,
    spec: &rwkvzip::MethodSpec,
    f: impl FnOnce(&mut rwkvzip::Compressor) -> R,
) -> R {
    RWKV_METHOD_TLS.with(|cell| {
        let mut map = cell.borrow_mut();
        let mut comp = if let Some(template) = map.get(method) {
            template.clone()
        } else {
            let template = rwkvzip::Compressor::new_from_method_spec(spec).unwrap_or_else(|e| {
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
pub(crate) fn with_mamba_method_spec_tls<R>(
    method: &str,
    spec: &mambazip::MethodSpec,
    f: impl FnOnce(&mut mambazip::Compressor) -> R,
) -> R {
    MAMBA_METHOD_TLS.with(|cell| {
        let mut map = cell.borrow_mut();
        let mut comp = if let Some(template) = map.get(method) {
            template.clone()
        } else {
            let template = mambazip::Compressor::new_from_method_spec(spec).unwrap_or_else(|e| {
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
    backend: &CompiledRateBackend,
) -> InfotheoryResult<f64> {
    if data.is_empty() {
        return Ok(0.0);
    }
    let total = prefix_parts
        .iter()
        .map(|p| p.len() as u64)
        .sum::<u64>()
        .saturating_add(data.len() as u64);
    let mut predictor =
        crate::runtime::build_rate_backend_predictor_default(backend).map_err(|e| {
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
    backend: &CompiledRateBackend,
) -> InfotheoryResult<f64> {
    if score_data.is_empty() {
        return Ok(0.0);
    }
    #[cfg(feature = "backend-rosa")]
    if let crate::spec::core::RateBackendPlan::RosaPlus { max_order } = backend.plan() {
        let mut model = RosaPlus::new(*max_order, false, 0, 42);
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
    if let crate::spec::core::RateBackendPlan::Rwkv7 {
        method,
        parsed_method,
        ..
    } = backend.plan()
    {
        return with_rwkv_method_spec_tls(method, parsed_method, |c| {
            c.cross_entropy_frozen_plugin_chain(fit_parts, score_data)
                .map_err(|e| {
                    InfotheoryError::runtime(format!(
                        "rwkv method frozen-plugin scoring failed: {e:#}"
                    ))
                })
        });
    }
    #[cfg(feature = "backend-mamba")]
    if let crate::spec::core::RateBackendPlan::Mamba {
        method,
        parsed_method,
        ..
    } = backend.plan()
    {
        return with_mamba_method_spec_tls(method, parsed_method, |c| {
            c.cross_entropy_frozen_plugin_chain(fit_parts, score_data)
                .map_err(|e| {
                    InfotheoryError::runtime(format!(
                        "mamba method frozen-plugin scoring failed: {e:#}"
                    ))
                })
        });
    }

    let fit_total = fit_parts.iter().map(|part| part.len() as u64).sum::<u64>();
    let mut predictor =
        crate::runtime::build_rate_backend_predictor_default(backend).map_err(|e| {
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

#[cfg(all(test, feature = "all-backends"))]
mod tests {
    use super::*;

    #[cfg(not(feature = "backend-zpaq"))]
    fn compress_size_backend(data: &[u8], backend: &CompressionBackend) -> u64 {
        let compiled = backend
            .compile()
            .unwrap_or_else(|err| panic!("failed to compile compression backend for test: {err}"));
        crate::api::try_compress_size_backend(data, &compiled).expect("compress_size_backend")
    }

    fn compiled_rate_backend(backend: &RateBackend) -> crate::spec::CompiledRateBackend {
        backend
            .compile()
            .unwrap_or_else(|err| panic!("failed to compile rate backend for test: {err}"))
    }

    fn ctx(rate_backend: RateBackend, compression_backend: CompressionBackend) -> InfotheoryCtx {
        InfotheoryCtx::from_specs(rate_backend, compression_backend)
            .unwrap_or_else(|err| panic!("failed to build infotheory test context: {err}"))
    }

    fn default_compression_backend() -> CompressionBackend {
        CompressionBackend::try_default()
            .unwrap_or_else(|err| panic!("failed to select default compression backend: {err}"))
    }

    fn default_ctx() -> InfotheoryCtx {
        InfotheoryCtx::try_default()
            .unwrap_or_else(|err| panic!("failed to build default test context: {err}"))
    }

    fn generate_rate_backend_chain(
        prefix_parts: &[&[u8]],
        bytes: usize,
        backend: &RateBackend,
        config: GenerationConfig,
    ) -> Vec<u8> {
        crate::api::generation::generate_rate_backend_chain(
            prefix_parts,
            bytes,
            &compiled_rate_backend(backend),
            config,
        )
    }

    fn ncd_bytes(x: &[u8], y: &[u8], method: &str, variant: NcdVariant) -> f64 {
        let backend = CompressionBackend::zpaq(method)
            .compile()
            .expect("compile zpaq backend");
        crate::api::try_ncd_bytes_backend(x, y, &backend, variant).expect("ncd_bytes")
    }

    fn entropy_rate_bytes(data: &[u8]) -> f64 {
        try_entropy_rate_bytes(data).expect("entropy_rate_bytes")
    }

    fn entropy_rate_backend(data: &[u8], backend: &RateBackend) -> f64 {
        try_entropy_rate_backend(data, &compiled_rate_backend(backend))
            .expect("entropy_rate_backend")
    }

    fn biased_entropy_rate_backend(data: &[u8], backend: &RateBackend) -> f64 {
        try_biased_entropy_rate_backend(data, &compiled_rate_backend(backend))
            .expect("biased_entropy_rate_backend")
    }

    fn cross_entropy_rate_backend(
        test_data: &[u8],
        train_data: &[u8],
        backend: &RateBackend,
    ) -> f64 {
        try_cross_entropy_rate_backend(test_data, train_data, &compiled_rate_backend(backend))
            .expect("cross_entropy_rate_backend")
    }

    fn joint_entropy_rate_backend(x: &[u8], y: &[u8], backend: &RateBackend) -> f64 {
        try_joint_entropy_rate_backend(x, y, &compiled_rate_backend(backend))
            .expect("joint_entropy_rate_backend")
    }

    fn joint_entropy_rate_bytes(x: &[u8], y: &[u8]) -> f64 {
        try_joint_entropy_rate_bytes(x, y).expect("joint_entropy_rate_bytes")
    }

    fn conditional_entropy_rate_bytes(x: &[u8], y: &[u8]) -> f64 {
        try_conditional_entropy_rate_bytes(x, y).expect("conditional_entropy_rate_bytes")
    }

    fn mutual_information_bytes(x: &[u8], y: &[u8]) -> f64 {
        try_mutual_information_bytes(x, y).expect("mutual_information_bytes")
    }

    fn ned_bytes(x: &[u8], y: &[u8]) -> f64 {
        crate::api::try_ned_bytes(x, y).expect("ned_bytes")
    }

    fn nte_rate_backend(x: &[u8], y: &[u8], backend: &RateBackend) -> f64 {
        crate::api::try_nte_rate_backend(x, y, &compiled_rate_backend(backend))
            .expect("nte_rate_backend")
    }

    fn resistance_to_transformation_bytes(x: &[u8], tx: &[u8]) -> f64 {
        crate::api::try_resistance_to_transformation_bytes(x, tx)
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
                        backend: test_match_backend(),
                    },
                    MixtureExpertSpec {
                        name: Some("ppmd".to_string()),
                        log_prior: 0.0,
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

    fn assert_deterministic_generate_for_backend(backend: RateBackend, bytes: usize, label: &str) {
        let prompt = continuation_prompt();
        let a =
            generate_rate_backend_chain(&[prompt], bytes, &backend, GenerationConfig::default());
        let b =
            generate_rate_backend_chain(&[prompt], bytes, &backend, GenerationConfig::default());
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

    fn assert_sampled_generate_for_backend(backend: RateBackend, bytes: usize, label: &str) {
        let prompt = continuation_prompt();
        let config = GenerationConfig::sampled_frozen(42);
        let a = generate_rate_backend_chain(&[prompt], bytes, &backend, config);
        let b = generate_rate_backend_chain(&[prompt], bytes, &backend, config);
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
    fn shannon_identities_empirical_aligned() {
        let x = b"abracadabra";
        let y = b"abracadabra";

        let h_x = empirical_entropy_bytes(x);
        let mi = crate::api::empirical_mutual_information_bytes(x, y);
        let h_xy = empirical_joint_entropy_bytes(x, y);
        let h_x_given_y = (h_xy - h_x).max(0.0);
        let ned = crate::api::empirical_ned_bytes(x, y);
        let nte = crate::api::empirical_nte_bytes(x, y);

        assert!((h_xy - h_x).abs() < 1e-12);
        assert!(h_x_given_y.abs() < 1e-12);
        assert!((mi - h_x).abs() < 1e-12);
        assert!(ned.abs() < 1e-12);
        assert!(nte.abs() < 1e-12);
    }

    #[test]
    fn shannon_identities_rate_aligned_reasonable() {
        let x = b"the quick brown fox jumps over the lazy dog";
        let y = b"the quick brown fox jumps over the lazy dog";
        let prev = get_default_ctx().expect("default ctx");
        set_default_ctx(ctx(
            RateBackend::RosaPlus { max_order: 8 },
            default_compression_backend(),
        ));

        let h_x = entropy_rate_bytes(x);
        let h_xy = joint_entropy_rate_bytes(x, y);
        let h_x_given_y = conditional_entropy_rate_bytes(x, y);
        let mi = mutual_information_bytes(x, y);
        let ned = ned_bytes(x, y);

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
        let prev = get_default_ctx().expect("default ctx");
        set_default_ctx(ctx(
            RateBackend::RosaPlus { max_order: 8 },
            default_compression_backend(),
        ));
        let r = resistance_to_transformation_bytes(x, x);
        assert!((r - 1.0).abs() < 1e-6);
        set_default_ctx(prev);
    }

    #[test]
    fn empirical_metrics_empty_inputs_are_zero() {
        let empty: &[u8] = &[];
        let x = b"abc";

        assert_eq!(tvd_bytes(empty, x), 0.0);
        assert_eq!(tvd_bytes(x, empty), 0.0);
        assert_eq!(nhd_bytes(empty, x), 0.0);
        assert_eq!(nhd_bytes(x, empty), 0.0);
        assert_eq!(d_kl_bytes(empty, x), 0.0);
        assert_eq!(d_kl_bytes(x, empty), 0.0);
        assert_eq!(js_div_bytes(empty, x), 0.0);
        assert_eq!(js_div_bytes(x, empty), 0.0);
    }

    #[test]
    fn empirical_cross_entropy_empty_test_is_zero() {
        let empty: &[u8] = &[];
        let y = b"abc";
        assert_eq!(crate::api::empirical_cross_entropy_bytes(empty, y), 0.0);
    }

    #[test]
    fn backend_switching_test() {
        let x = b"hello world context";

        // Default is RosaPlus
        let h_rosa = entropy_rate_bytes(x);

        // Switch to CTW
        set_default_ctx(ctx(
            RateBackend::Ctw { depth: 16 },
            default_compression_backend(),
        ));

        let h_ctw = entropy_rate_bytes(x);

        // They should generally be different, but most importantly, CTW worked
        assert!(h_ctw > 0.0);

        // Reset to default
        set_default_ctx(default_ctx());
        let h_rosa_back = entropy_rate_bytes(x);
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
        // Note: For *empirical* (order-0) NTE, due to how joint entropy works for aligned
        // pairs, it's mathematically bounded differently. The fix for NTE clamping primarily
        // affects *rate*-based NTE where VI can truly be 2*max(H).
        //
        // We test that the clamp upper bound is at least > 1.0 for cases where VI > max(H)

        // Use CTW backend for rate-based test
        set_default_ctx(ctx(
            RateBackend::Ctw { depth: 8 },
            default_compression_backend(),
        ));

        // Generate two completely different patterns - should have high VI
        let x: Vec<u8> = (0..200).map(|i| (i % 2) as u8).collect(); // 010101...
        let y: Vec<u8> = (0..200).map(|i| ((i + 1) % 2) as u8).collect(); // 101010...

        let nte_rate = nte_rate_backend(&x, &y, &RateBackend::Ctw { depth: 8 });

        // With the fix, NTE should not be clamped to 1.0
        // It may or may not exceed 1.0 depending on the specifics, but it should be allowed to
        assert!(
            (0.0..=2.0 + 1e-9).contains(&nte_rate),
            "NTE should be in [0, 2], got {}",
            nte_rate
        );

        // Reset context
        set_default_ctx(default_ctx());
    }

    #[test]
    fn ctw_empty_data_returns_zero() {
        // Verify empty data doesn't cause division-by-zero or NaN
        set_default_ctx(ctx(
            RateBackend::Ctw { depth: 16 },
            default_compression_backend(),
        ));

        let empty: &[u8] = &[];
        let h = entropy_rate_bytes(empty);
        assert_eq!(h, 0.0, "empty data should return 0.0 entropy");

        // Reset
        set_default_ctx(default_ctx());
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
                    msb_first: None,
                },
            ),
            ("match", test_match_backend()),
        ];

        for (name, backend) in cases {
            assert_eq!(
                joint_entropy_rate_backend(b"", b"nonempty", &backend),
                0.0,
                "{name} should return 0.0 for empty aligned pairs"
            );
            assert_eq!(
                joint_entropy_rate_backend(b"nonempty", b"", &backend),
                0.0,
                "{name} should return 0.0 when alignment truncates to empty"
            );

            let aligned = joint_entropy_rate_backend(b"abcd", b"wxyz", &backend);
            let truncated = joint_entropy_rate_backend(b"abcdextra", b"wxyz", &backend);
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
            let h1 = biased_entropy_rate_backend(data, &backend);
            let h2 = biased_entropy_rate_backend(data, &backend);
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

        let flat =
            generate_rate_backend_chain(&[prompt], bytes, &backend, GenerationConfig::default());
        let chained = generate_rate_backend_chain(
            &[front, back],
            bytes,
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
        assert_deterministic_generate_for_backend(RateBackend::Ctw { depth: 32 }, 8, "ctw");
        assert_deterministic_generate_for_backend(
            RateBackend::RosaPlus { max_order: -1 },
            8,
            "rosaplus",
        );
        assert_deterministic_generate_for_backend(test_match_backend(), 8, "match");
        assert_deterministic_generate_for_backend(test_ppmd_backend(), 8, "ppmd");
    }

    #[cfg(feature = "backend-rwkv")]
    #[test]
    fn generate_bytes_api_is_deterministic_for_rwkv_method() {
        let backend = RateBackend::Rwkv7Method {
            method: crate::rwkvzip::parse_method_spec("cfg:hidden=64,layers=1,intermediate=64,decay_rank=8,a_rank=8,v_rank=8,g_rank=8,seed=31,train=none,lr=0.0,stride=1;policy:schedule=0..100:infer").expect("rwkv method spec"),
        };
        assert_deterministic_generate_for_backend(backend, 8, "rwkv7");
    }

    #[test]
    fn sampled_generation_is_deterministic_for_ctw_rosa_match_ppmd() {
        assert_sampled_generate_for_backend(RateBackend::Ctw { depth: 32 }, 8, "ctw");
        assert_sampled_generate_for_backend(RateBackend::RosaPlus { max_order: -1 }, 8, "rosaplus");
        assert_sampled_generate_for_backend(test_match_backend(), 8, "match");
        assert_sampled_generate_for_backend(test_ppmd_backend(), 8, "ppmd");
    }

    #[cfg(feature = "backend-rwkv")]
    #[test]
    fn sampled_generation_is_deterministic_for_rwkv_method() {
        let backend = RateBackend::Rwkv7Method {
            method: crate::rwkvzip::parse_method_spec("cfg:hidden=64,layers=1,intermediate=64,decay_rank=8,a_rank=8,v_rank=8,g_rank=8,seed=31,train=none,lr=0.0,stride=1;policy:schedule=0..100:infer").expect("rwkv method spec"),
        };
        assert_sampled_generate_for_backend(backend, 8, "rwkv7");
    }

    #[test]
    fn rosaplus_sampled_generation_predicts_green_continuation() {
        let out = generate_rate_backend_chain(
            &[continuation_prompt()],
            8,
            &RateBackend::RosaPlus { max_order: -1 },
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
            RateBackendSession::from_spec(backend.clone(), Some((prompt.len() + 8) as u64))
                .expect("session init");
        session.observe(prompt);
        let from_session = session.generate_bytes(8, GenerationConfig::sampled_frozen(42));
        session.finish().expect("session finish");

        let ctx = ctx(backend, default_compression_backend());
        let from_ctx = ctx
            .try_generate_bytes_with_config(prompt, 8, GenerationConfig::sampled_frozen(42))
            .expect("ctx generation");
        assert_eq!(from_session, from_ctx);
    }

    #[test]
    fn biased_entropy_ctw_uses_frozen_plugin_scoring() {
        let backend = RateBackend::Ctw { depth: 8 };
        let data = b"AAAAAAAA";
        let plugin = biased_entropy_rate_backend(data, &backend);
        let prequential = entropy_rate_backend(data, &backend);
        assert!(
            plugin + 1e-9 < prequential,
            "expected plugin scoring to beat prequential scoring: plugin={plugin} prequential={prequential}"
        );
    }

    #[test]
    fn rosa_plugin_entropy_matches_direct_model_api() {
        let data = b"abracadabra";
        let backend = RateBackend::RosaPlus { max_order: 3 };

        let plugin = biased_entropy_rate_backend(data, &backend);

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
        let backend = RateBackend::RosaPlus { max_order: 3 };

        let plugin = cross_entropy_rate_backend(test, train, &backend);

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
        let ctx = ctx(
            RateBackend::RosaPlus { max_order: -1 },
            default_compression_backend(),
        );
        let prefix_parts: [&[u8]; 3] = [b"universal ", b"prior ", b"slice"];
        let data = b"query payload";

        let chained = ctx
            .try_cross_entropy_conditional_chain(&prefix_parts, data)
            .expect("conditional-chain cross entropy");
        let flat_prefix: Vec<u8> = prefix_parts.concat();
        let flat = cross_entropy_rate_backend(
            data,
            &flat_prefix,
            &RateBackend::RosaPlus { max_order: -1 },
        );

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

        // Generate data and check empirical entropy is close to theoretical
        let data = crate::datagen::bernoulli(10000, p, 42);
        let estimated_h = empirical_entropy_bytes(&data);

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
            method: crate::rwkvzip::parse_method_spec(method).expect("rwkv method spec"),
        };
        let data = b"rwkv method entropy stability regression sample";

        let h1 = entropy_rate_backend(data, &backend);
        let h2 = entropy_rate_backend(data, &backend);
        assert!(
            (h1 - h2).abs() < 1e-12,
            "rwkv method entropy leaked mutable state across calls: h1={h1}, h2={h2}"
        );
    }

    #[cfg(feature = "backend-rwkv")]
    #[test]
    fn rwkv_method_without_policy_is_accepted_by_public_api() {
        let backend = RateBackend::Rwkv7Method {
            method: crate::rwkvzip::parse_method_spec("cfg:hidden=64,layers=1,intermediate=64")
                .expect("rwkv method spec"),
        };
        let data = b"rwkv method without policy";
        let h1 = entropy_rate_backend(data, &backend);
        let h2 = biased_entropy_rate_backend(data, &backend);
        assert!(h1.is_finite());
        assert!(h2.is_finite());
    }

    #[cfg(feature = "backend-rwkv")]
    #[test]
    fn rwkv_infer_only_plugin_collapses_to_single_pass_entropy() {
        let backend = RateBackend::Rwkv7Method {
            method: crate::rwkvzip::parse_method_spec("cfg:hidden=64,layers=1,intermediate=64,decay_rank=8,a_rank=8,v_rank=8,g_rank=8,seed=25,train=none,lr=0.0,stride=1;policy:schedule=0..100:infer").expect("rwkv method spec"),
        };
        let data = b"rwkv infer-only plugin equality sample";
        let h = entropy_rate_backend(data, &backend);
        let plugin = biased_entropy_rate_backend(data, &backend);
        assert!(
            (h - plugin).abs() < 1e-12,
            "infer-only rwkv plugin should equal single-pass entropy: h={h}, plugin={plugin}"
        );
    }

    #[cfg(feature = "backend-rwkv")]
    #[test]
    fn rwkv_method_biased_entropy_is_stable_across_calls_with_training_policy() {
        let backend = RateBackend::Rwkv7Method {
            method: crate::rwkvzip::parse_method_spec("cfg:hidden=64,layers=1,intermediate=64,decay_rank=8,a_rank=8,v_rank=8,g_rank=8,seed=23,train=sgd,lr=0.01,stride=1;policy:schedule=0..100:train(scope=head+bias,opt=sgd,lr=0.01,stride=1,bptt=1,clip=0,momentum=0.0)").expect("rwkv method spec"),
        };
        let data = b"rwkv plugin stability sample";
        let h1 = biased_entropy_rate_backend(data, &backend);
        let h2 = biased_entropy_rate_backend(data, &backend);
        assert!(
            (h1 - h2).abs() < 1e-12,
            "rwkv method biased entropy leaked mutable state across calls: h1={h1}, h2={h2}"
        );
    }

    #[cfg(feature = "backend-rwkv")]
    #[test]
    fn rwkv_method_conditional_chain_is_stable_across_calls() {
        let method = "cfg:hidden=64,layers=1,intermediate=64,decay_rank=8,a_rank=8,v_rank=8,g_rank=8,seed=22,train=sgd,lr=0.01,stride=1;policy:schedule=0..100:infer";
        let ctx = ctx(
            RateBackend::Rwkv7Method {
                method: crate::rwkvzip::parse_method_spec(method).expect("rwkv method spec"),
            },
            default_compression_backend(),
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
            method: crate::mambazip::parse_method_spec("cfg:hidden=64,layers=1,intermediate=96")
                .expect("mamba method spec"),
        };
        let data = b"mamba method without policy";
        let h1 = entropy_rate_backend(data, &backend);
        let h2 = biased_entropy_rate_backend(data, &backend);
        assert!(h1.is_finite());
        assert!(h2.is_finite());
    }

    #[cfg(feature = "backend-mamba")]
    #[test]
    fn mamba_infer_only_plugin_collapses_to_single_pass_entropy() {
        let backend = RateBackend::MambaMethod {
            method: crate::mambazip::parse_method_spec("cfg:hidden=64,layers=1,intermediate=96,state=16,conv=4,dt_rank=16,seed=26,train=none,lr=0.0,stride=1;policy:schedule=0..100:infer").expect("mamba method spec"),
        };
        let data = b"mamba infer-only plugin equality sample";
        let h = entropy_rate_backend(data, &backend);
        let plugin = biased_entropy_rate_backend(data, &backend);
        assert!(
            (h - plugin).abs() < 1e-12,
            "infer-only mamba plugin should equal single-pass entropy: h={h}, plugin={plugin}"
        );
    }

    #[cfg(feature = "backend-mamba")]
    #[test]
    fn mamba_method_biased_entropy_is_stable_across_calls_with_training_policy() {
        let backend = RateBackend::MambaMethod {
            method: crate::mambazip::parse_method_spec("cfg:hidden=64,layers=1,intermediate=96,state=16,conv=4,dt_rank=16,seed=24,train=sgd,lr=0.01,stride=1;policy:schedule=0..100:train(scope=head+bias,opt=sgd,lr=0.01,stride=1,bptt=1,clip=0,momentum=0.0)").expect("mamba method spec"),
        };
        let data = b"mamba plugin stability sample";
        let h1 = biased_entropy_rate_backend(data, &backend);
        let h2 = biased_entropy_rate_backend(data, &backend);
        assert!(
            (h1 - h2).abs() < 1e-12,
            "mamba method biased entropy leaked mutable state across calls: h1={h1}, h2={h2}"
        );
    }

    #[test]
    fn particle_entropy_rate_in_valid_range() {
        let rb = test_particle_backend();
        let data = b"hello world particle backend test";
        let rate = entropy_rate_backend(data, &rb);
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
        let h1 = cross_entropy_rate_backend(test, train, &rb);
        let h2 = cross_entropy_rate_backend(test, train, &rb);
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
        let rate = entropy_rate_backend(b"", &rb);
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
        let joint = joint_entropy_rate_backend(x, y, &rb);
        assert!(
            joint > 0.0 && joint < 16.0,
            "particle joint entropy rate out of range: {joint}"
        );
    }
}

#[cfg(all(
    test,
    not(any(
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
    use crate::api::RateBackend;

    #[cfg(not(feature = "backend-zpaq"))]
    #[test]
    fn explicit_zpaq_backend_fails_to_compile_without_feature() {
        let backend = CompressionBackend::zpaq("5");
        let err = backend
            .compile()
            .err()
            .expect("zpaq backend should fail loudly at compile boundary");
        assert!(
            err.to_string()
                .contains("requires infotheory feature 'backend-zpaq'"),
            "unexpected error: {err}"
        );
    }

    #[cfg(not(feature = "backend-zpaq"))]
    #[test]
    fn default_rate_backend_selection_fails_when_no_rate_backends_are_enabled() {
        let err = match RateBackend::try_default() {
            Ok(_) => panic!("no default rate backend should exist"),
            Err(err) => err,
        };
        assert!(
            err.to_string()
                .contains("no default rate backend is available in this build"),
            "unexpected error: {err}"
        );
    }

    #[cfg(feature = "backend-zpaq")]
    #[test]
    fn default_rate_backend_is_zpaq_when_zpaq_is_enabled() {
        let backend = RateBackend::try_default()
            .expect("zpaq-enabled minimal build should expose a default rate backend");
        assert!(
            matches!(backend, RateBackend::Zpaq { .. }),
            "expected zpaq default backend variant"
        );
    }

    #[cfg(not(feature = "backend-zpaq"))]
    #[test]
    fn default_compression_backend_selection_fails_when_no_rate_backends_are_enabled() {
        let err = match CompressionBackend::try_default() {
            Ok(_) => panic!("no default compression backend exists"),
            Err(err) => err,
        };
        assert!(
            err.to_string()
                .contains("no default rate backend is available in this build"),
            "unexpected error: {err}"
        );
    }
}
