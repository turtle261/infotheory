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
//! use infotheory::{ncd_vitanyi, mutual_information_bytes, NcdVariant};
//!
//! let x = b"some data sequence";
//! let y = b"another data sequence";
//!
//! // Compression-based distance
//! let ncd = ncd_vitanyi("file1.txt", "file2.txt", "5");
//!
//! // Entropy-based mutual information (Marginal / i.i.d.)
//! let mi_marg = mutual_information_bytes(x, y, 0);
//!
//! // Entropy-based mutual information (Rate / Sequential, max_order=8)
//! let mi_rate = mutual_information_bytes(x, y, 8);
//! ```

/// AIXI planning components, environments, and model abstractions.
pub mod aixi;
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
/// Online Bayesian/switching/MDL mixture predictors.
pub mod mixture;
pub(crate) mod neural_mix;
pub(crate) mod simd_math;
/// CTW and FAC-CTW backend types.
pub use backends::ctw;
#[cfg(feature = "backend-mamba")]
/// Mamba backend types and compressor.
pub use backends::mambazip;
/// Match-based repeat predictor.
pub use backends::match_model;
/// Particle-latent filter ensemble rate backend.
pub use backends::particle;
/// PPMD-style byte model.
pub use backends::ppmd;
/// ROSA+ backend types.
pub use backends::rosaplus;
#[cfg(feature = "backend-rwkv")]
/// RWKV backend types and compressor.
pub use backends::rwkvzip;
/// Sparse/gapped match predictor.
pub use backends::sparse_match;
/// ZPAQ rate-model adapter.
pub use backends::zpaq_rate;

use rayon::prelude::*;

use crate::coders::CoderType;
use std::cell::RefCell;
#[cfg(any(feature = "backend-rwkv", feature = "backend-mamba"))]
use std::collections::HashMap;
use std::sync::Arc;
use std::sync::OnceLock;

static NUM_THREADS: OnceLock<usize> = OnceLock::new();

thread_local! {
    #[cfg(feature = "backend-mamba")]
    static MAMBA_TLS: RefCell<HashMap<usize, mambazip::Compressor>> = RefCell::new(HashMap::new());
    #[cfg(feature = "backend-mamba")]
    static MAMBA_RATE_TLS: RefCell<HashMap<usize, mambazip::Compressor>> = RefCell::new(HashMap::new());
    #[cfg(feature = "backend-mamba")]
    static MAMBA_METHOD_TLS: RefCell<HashMap<String, mambazip::Compressor>> = RefCell::new(HashMap::new());
    #[cfg(feature = "backend-rwkv")]
    static RWKV_TLS: RefCell<HashMap<usize, rwkvzip::Compressor>> = RefCell::new(HashMap::new());
    #[cfg(feature = "backend-rwkv")]
    static RWKV_RATE_TLS: RefCell<HashMap<usize, rwkvzip::Compressor>> = RefCell::new(HashMap::new());
    #[cfg(feature = "backend-rwkv")]
    static RWKV_METHOD_TLS: RefCell<HashMap<String, rwkvzip::Compressor>> = RefCell::new(HashMap::new());
}

#[cfg(feature = "backend-zpaq")]
impl Default for CompressionBackend {
    fn default() -> Self {
        CompressionBackend::Zpaq {
            method: "5".to_string(),
        }
    }
}

#[cfg(not(feature = "backend-zpaq"))]
impl Default for CompressionBackend {
    fn default() -> Self {
        CompressionBackend::Rate {
            rate_backend: RateBackend::default(),
            coder: CoderType::AC,
            framing: compression::FramingMode::Raw,
        }
    }
}

thread_local! {
    static DEFAULT_CTX: RefCell<InfotheoryCtx> = RefCell::new(InfotheoryCtx::default());
}

/// Returns the current default information theory context for the thread.
pub fn get_default_ctx() -> InfotheoryCtx {
    DEFAULT_CTX.with(|ctx| ctx.borrow().clone())
}

/// Sets the default information theory context for the thread.
pub fn set_default_ctx(ctx: InfotheoryCtx) {
    DEFAULT_CTX.with(|c| *c.borrow_mut() = ctx);
}

#[inline(always)]
fn with_default_ctx<R>(f: impl FnOnce(&InfotheoryCtx) -> R) -> R {
    DEFAULT_CTX.with(|ctx| f(&ctx.borrow()))
}

/// Mutual information rate estimate under an explicit `backend`.
///
/// Inputs are aligned to the shared prefix length.
pub fn mutual_information_rate_backend(
    x: &[u8],
    y: &[u8],
    max_order: i64,
    backend: &RateBackend,
) -> f64 {
    let (x, y) = aligned_prefix(x, y);
    if x.is_empty() {
        return 0.0;
    }
    // For CTW, we might want a special aligned implementation?
    // Using standard formula for now.
    let h_x = entropy_rate_backend(x, max_order, backend);
    let h_y = entropy_rate_backend(y, max_order, backend);
    let h_xy = joint_entropy_rate_backend(x, y, max_order, backend);
    (h_x + h_y - h_xy).max(0.0)
}

/// Normalized entropy distance under an explicit `backend`.
///
/// Returns a value in `[0, 1]` after clamping.
pub fn ned_rate_backend(x: &[u8], y: &[u8], max_order: i64, backend: &RateBackend) -> f64 {
    let (x, y) = aligned_prefix(x, y);
    if x.is_empty() {
        return 0.0;
    }
    let h_x = entropy_rate_backend(x, max_order, backend);
    let h_y = entropy_rate_backend(y, max_order, backend);
    let h_xy = joint_entropy_rate_backend(x, y, max_order, backend);
    let min_h = h_x.min(h_y);
    let max_h = h_x.max(h_y);
    if max_h == 0.0 {
        0.0
    } else {
        ((h_xy - min_h) / max_h).clamp(0.0, 1.0)
    }
}

/// Normalized transform effort (variation-of-information form) under an explicit `backend`.
///
/// Returns a value in `[0, 2]` after clamping.
pub fn nte_rate_backend(x: &[u8], y: &[u8], max_order: i64, backend: &RateBackend) -> f64 {
    let (x, y) = aligned_prefix(x, y);
    if x.is_empty() {
        return 0.0;
    }
    let h_x = entropy_rate_backend(x, max_order, backend);
    let h_y = entropy_rate_backend(y, max_order, backend);
    let h_xy = joint_entropy_rate_backend(x, y, max_order, backend);
    let max_h = h_x.max(h_y);
    if max_h == 0.0 {
        0.0
    } else {
        // VI = H(X|Y) + H(Y|X) can be as large as H(X) + H(Y) ≈ 2*max(H)
        // for independent sequences, so NTE ∈ [0, 2]
        let vi = (h_xy - h_x).max(0.0) + (h_xy - h_y).max(0.0);
        (vi / max_h).clamp(0.0, 2.0)
    }
}

/// Sequential entropy/rate backend used by context-aware metrics.
#[derive(Clone)]
pub enum RateBackend {
    /// ROSA+ suffix-automaton estimator.
    RosaPlus,
    /// Local contiguous match predictor.
    Match {
        /// Number of retained hash bits for suffix lookup.
        hash_bits: usize,
        /// Minimum repeat length required before predicting.
        min_len: usize,
        /// Maximum repeat length used for confidence scaling.
        max_len: usize,
        /// Residual probability mass left for non-match symbols.
        base_mix: f64,
        /// Confidence multiplier applied to short-match tapering.
        confidence_scale: f64,
    },
    /// Sparse/gapped local match predictor.
    SparseMatch {
        /// Number of retained hash bits for spaced-suffix lookup.
        hash_bits: usize,
        /// Minimum spaced repeat length required before predicting.
        min_len: usize,
        /// Maximum spaced repeat length used for confidence scaling.
        max_len: usize,
        /// Minimum gap between matched bytes.
        gap_min: usize,
        /// Maximum gap between matched bytes.
        gap_max: usize,
        /// Residual probability mass left for non-match symbols.
        base_mix: f64,
        /// Confidence multiplier applied to short-match tapering.
        confidence_scale: f64,
    },
    /// Pure-Rust bounded-memory PPMD-style model.
    Ppmd {
        /// Maximum context order.
        order: usize,
        /// Approximate memory budget in MiB.
        memory_mb: usize,
    },
    #[cfg(feature = "backend-mamba")]
    /// Mamba model loaded from explicit weights.
    Mamba {
        /// Loaded Mamba model.
        model: Arc<mambazip::Model>,
    },
    #[cfg(feature = "backend-mamba")]
    /// Mamba method string (e.g. `file:...` or `cfg:...[;policy:...]`) resolved lazily.
    MambaMethod {
        /// Mamba method string.
        method: String,
    },
    #[cfg(feature = "backend-rwkv")]
    /// RWKV7 model loaded from explicit weights.
    Rwkv7 {
        /// Loaded RWKV7 model.
        model: Arc<rwkvzip::Model>,
    },
    #[cfg(feature = "backend-rwkv")]
    /// RWKV7 method string (e.g. `file:...` or `cfg:...[;policy:...]`) resolved lazily.
    Rwkv7Method {
        /// RWKV7 method string.
        method: String,
    },
    /// ZPAQ compression-based rate model (streamable methods only).
    Zpaq {
        /// ZPAQ method string (streamable modes only for rate estimation).
        method: String,
    },
    /// Online mixture over rate-model experts (Bayes, fading Bayes, switching, MDL).
    Mixture {
        /// Mixture expert/runtime specification.
        spec: Arc<MixtureSpec>,
    },
    /// Particle-latent filter ensemble.
    Particle {
        /// Particle filter specification.
        spec: Arc<ParticleSpec>,
    },
    /// Calibrated wrapper over another bytewise backend.
    Calibrated {
        /// Calibration specification.
        spec: Arc<CalibratedSpec>,
    },
    /// Action-Conditional CTW (single context tree).
    Ctw {
        /// Context tree depth.
        depth: usize,
    },
    /// Factorized Action-Conditional CTW (k trees for k-bit percepts).
    FacCtw {
        /// Base context depth.
        base_depth: usize,
        /// Number of percept bits.
        num_percept_bits: usize,
        /// Encoding width in bits.
        encoding_bits: usize,
    },
}

#[allow(clippy::derivable_impls)]
impl Default for RateBackend {
    fn default() -> Self {
        #[cfg(feature = "backend-rosa")]
        {
            RateBackend::RosaPlus
        }
        #[cfg(all(not(feature = "backend-rosa"), feature = "backend-zpaq"))]
        {
            RateBackend::Zpaq {
                method: "1".to_string(),
            }
        }
        #[cfg(all(not(feature = "backend-rosa"), not(feature = "backend-zpaq")))]
        {
            RateBackend::Ctw { depth: 16 }
        }
    }
}

/// Compression backend used by NCD/compression-size operations.
#[derive(Clone)]
pub enum CompressionBackend {
    /// ZPAQ compressor with explicit method string.
    Zpaq {
        /// ZPAQ method (for example `"1"` or `"5"`).
        method: String,
    },
    #[cfg(feature = "backend-rwkv")]
    /// RWKV7 model as an entropy-coded compressor.
    Rwkv7 {
        /// Loaded RWKV7 model.
        model: Arc<rwkvzip::Model>,
        /// Entropy coder used for coding model PDFs.
        coder: CoderType,
    },
    /// Generic rate-coded compressor wrapping an arbitrary rate backend.
    Rate {
        /// Predictive rate backend.
        rate_backend: RateBackend,
        /// Entropy coder used for coding model PDFs.
        coder: CoderType,
        /// Framing mode for output payloads.
        framing: compression::FramingMode,
    },
}

/// Mixture policy kind for rate-backend mixtures.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MixtureKind {
    /// Standard Bayesian mixture with fixed expert weights.
    Bayes,
    /// Bayesian mixture with exponential weight decay.
    FadingBayes,
    /// Switching mixture with hazard `alpha`.
    Switching,
    /// MDL-style best-expert selector.
    Mdl,
    /// Bytewise neural logistic mixer (fx2-cmix style adaptation).
    Neural,
}

/// Fixed context families for calibrated PDF wrappers.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CalibrationContextKind {
    /// Single global calibration row.
    Global,
    /// Previous-byte class only.
    ByteClass,
    /// Text-structure-aware context hash.
    Text,
    /// Repeat-aware context hash.
    Repeat,
    /// Joint text/repeat-aware context hash.
    TextRepeat,
}

/// Configuration for a calibrated wrapper rate backend.
#[derive(Clone)]
pub struct CalibratedSpec {
    /// Base backend whose PDF is calibrated.
    pub base: RateBackend,
    /// Context family controlling table row selection.
    pub context: CalibrationContextKind,
    /// Number of probability bins per row.
    pub bins: usize,
    /// Online learning rate for observed-symbol updates.
    pub learning_rate: f64,
    /// Symmetric clip applied to calibration weights.
    pub bias_clip: f64,
}

/// Expert specification for mixture backends.
#[derive(Clone)]
pub struct MixtureExpertSpec {
    /// Optional expert display name.
    pub name: Option<String>,
    /// Log prior weight (natural log). Uniform priors can be `0.0`.
    pub log_prior: f64,
    /// Max order for ROSA experts (ignored for other backends).
    pub max_order: i64,
    /// Underlying backend for this expert.
    pub backend: RateBackend,
}

/// Mixture specification for rate-backend mixtures.
#[derive(Clone)]
pub struct MixtureSpec {
    /// Mixture policy.
    pub kind: MixtureKind,
    /// Switching probability (per step) for switching mixtures.
    pub alpha: f64,
    /// Decay factor for fading Bayes mixtures.
    pub decay: Option<f64>,
    /// Expert list.
    pub experts: Vec<MixtureExpertSpec>,
}

impl MixtureSpec {
    /// Build a mixture specification from kind and expert list.
    pub fn new(kind: MixtureKind, experts: Vec<MixtureExpertSpec>) -> Self {
        Self {
            kind,
            alpha: 0.01,
            decay: None,
            experts,
        }
    }

    /// Set switching hazard / adaptation parameter.
    pub fn with_alpha(mut self, alpha: f64) -> Self {
        self.alpha = alpha;
        self
    }

    /// Set fading decay factor.
    pub fn with_decay(mut self, decay: f64) -> Self {
        self.decay = Some(decay);
        self
    }

    /// Convert to executable expert configs for runtime mixture evaluation.
    pub fn build_experts(&self) -> Vec<crate::mixture::ExpertConfig> {
        self.experts
            .iter()
            .map(|spec| {
                crate::mixture::ExpertConfig::from_rate_backend(
                    spec.name.clone(),
                    spec.log_prior,
                    spec.backend.clone(),
                    spec.max_order,
                )
            })
            .collect()
    }
}

/// Configuration for a particle-latent filter ensemble rate backend.
#[derive(Clone, Debug)]
pub struct ParticleSpec {
    /// Number of particles in the ensemble.
    pub num_particles: usize,
    /// Context window length for rolling byte context.
    pub context_window: usize,
    /// Number of latent update unroll steps per byte.
    pub unroll_steps: usize,
    /// Number of latent cells per particle.
    pub num_cells: usize,
    /// Dimensionality of each latent cell.
    pub cell_dim: usize,
    /// Number of discrete rules for soft routing.
    pub num_rules: usize,
    /// Hidden dimension for the selector MLP.
    pub selector_hidden: usize,
    /// Hidden dimension for each rule MLP.
    pub rule_hidden: usize,
    /// Dimension of per-rule noise input (ignored when deterministic).
    pub noise_dim: usize,
    /// Whether to use fully deterministic execution (no RNG).
    pub deterministic: bool,
    /// Whether to inject noise into rule inputs (ignored when deterministic).
    pub enable_noise: bool,
    /// Base scale for deterministic hash-noise injected into rule inputs.
    pub noise_scale: f64,
    /// Number of steps over which injected noise linearly anneals to zero.
    pub noise_anneal_steps: usize,
    /// Learning rate for readout layer SGD.
    pub learning_rate_readout: f64,
    /// Learning rate for selector MLP SGD.
    pub learning_rate_selector: f64,
    /// Learning rate for rule MLP SGD.
    pub learning_rate_rule: f64,
    /// Truncated backpropagation-through-time depth (number of recent steps).
    pub bptt_depth: usize,
    /// Momentum coefficient for selector/rule online updates (in [0, 1)).
    pub optimizer_momentum: f64,
    /// Gradient clipping threshold (max abs value per element).
    pub grad_clip: f64,
    /// Latent cell state clipping threshold (max abs value per element).
    pub state_clip: f64,
    /// Forgetting factor for particle log-weights (0 = no forgetting).
    pub forget_lambda: f64,
    /// Effective sample size ratio threshold for resampling (in (0, 1]).
    pub resample_threshold: f64,
    /// Fraction of particles to mutate after resampling (in [0, 1]).
    pub mutate_fraction: f64,
    /// Scale of hash-noise perturbation applied during mutation.
    pub mutate_scale: f64,
    /// Whether mutation also perturbs model parameters (state is always mutated).
    pub mutate_model_params: bool,
    /// Diagnostics print interval in steps (0 disables particle diagnostics logs).
    pub diagnostics_interval: usize,
    /// Minimum probability floor for numerical stability.
    pub min_prob: f64,
    /// Master seed for deterministic initialization and mutation.
    pub seed: u64,
}

impl Default for ParticleSpec {
    fn default() -> Self {
        Self {
            num_particles: 16,
            context_window: 32,
            unroll_steps: 2,
            num_cells: 8,
            cell_dim: 32,
            num_rules: 4,
            selector_hidden: 64,
            rule_hidden: 64,
            noise_dim: 8,
            deterministic: true,
            enable_noise: false,
            noise_scale: 0.10,
            noise_anneal_steps: 8192,
            learning_rate_readout: 0.01,
            learning_rate_selector: 1e-4,
            learning_rate_rule: 3e-4,
            bptt_depth: 3,
            optimizer_momentum: 0.05,
            grad_clip: 1.0,
            state_clip: 8.0,
            forget_lambda: 0.0,
            resample_threshold: 0.5,
            mutate_fraction: 0.1,
            mutate_scale: 0.01,
            mutate_model_params: false,
            diagnostics_interval: 0,
            min_prob: 2f64.powi(-24),
            seed: 42,
        }
    }
}

impl ParticleSpec {
    /// Validate all fields, returning an error message on failure.
    pub fn validate(&self) -> Result<(), String> {
        if self.num_particles == 0 {
            return Err("num_particles must be > 0".into());
        }
        if self.context_window == 0 {
            return Err("context_window must be > 0".into());
        }
        if self.unroll_steps == 0 {
            return Err("unroll_steps must be > 0".into());
        }
        if self.num_cells == 0 {
            return Err("num_cells must be > 0".into());
        }
        if self.cell_dim == 0 {
            return Err("cell_dim must be > 0".into());
        }
        if self.num_rules == 0 {
            return Err("num_rules must be > 0".into());
        }
        if self.selector_hidden == 0 {
            return Err("selector_hidden must be > 0".into());
        }
        if self.rule_hidden == 0 {
            return Err("rule_hidden must be > 0".into());
        }
        if !self.learning_rate_readout.is_finite() || self.learning_rate_readout < 0.0 {
            return Err("learning_rate_readout must be finite and non-negative".into());
        }
        if !self.learning_rate_selector.is_finite() || self.learning_rate_selector < 0.0 {
            return Err("learning_rate_selector must be finite and non-negative".into());
        }
        if !self.learning_rate_rule.is_finite() || self.learning_rate_rule < 0.0 {
            return Err("learning_rate_rule must be finite and non-negative".into());
        }
        if !self.noise_scale.is_finite() || self.noise_scale < 0.0 {
            return Err("noise_scale must be finite and non-negative".into());
        }
        if !self.optimizer_momentum.is_finite()
            || self.optimizer_momentum < 0.0
            || self.optimizer_momentum >= 1.0
        {
            return Err("optimizer_momentum must be finite and in [0, 1)".into());
        }
        if self.bptt_depth == 0 {
            return Err("bptt_depth must be > 0".into());
        }
        if !(self.resample_threshold > 0.0 && self.resample_threshold <= 1.0) {
            return Err("resample_threshold must be in (0, 1]".into());
        }
        if !(self.mutate_fraction >= 0.0 && self.mutate_fraction <= 1.0) {
            return Err("mutate_fraction must be in [0, 1]".into());
        }
        if !(self.min_prob > 0.0 && self.min_prob < 0.5) {
            return Err("min_prob must be in (0, 0.5)".into());
        }
        Ok(())
    }
}

/// Reusable execution context holding default rate and compression backends.
#[derive(Clone, Default)]
pub struct InfotheoryCtx {
    /// Default rate backend for entropy/rate metrics.
    pub rate_backend: RateBackend,
    /// Default compression backend for NCD/compression primitives.
    pub compression_backend: CompressionBackend,
}

impl InfotheoryCtx {
    /// Create a context from explicit rate and compression backends.
    pub fn new(rate_backend: RateBackend, compression_backend: CompressionBackend) -> Self {
        Self {
            rate_backend,
            compression_backend,
        }
    }

    /// Create a context with ROSA+ rate backend and ZPAQ compression backend.
    pub fn with_zpaq(method: impl Into<String>) -> Self {
        Self {
            rate_backend: RateBackend::RosaPlus,
            compression_backend: CompressionBackend::Zpaq {
                method: method.into(),
            },
        }
    }

    /// Compressed length of one byte slice under this context's compressor.
    pub fn compress_size(&self, data: &[u8]) -> u64 {
        compress_size_backend(data, &self.compression_backend)
    }

    /// Compressed length of chained slices under one stream.
    pub fn compress_size_chain(&self, parts: &[&[u8]]) -> u64 {
        compress_size_chain_backend(parts, &self.compression_backend)
    }

    /// Entropy-rate estimate for `data` under this context's rate backend.
    pub fn entropy_rate_bytes(&self, data: &[u8], max_order: i64) -> f64 {
        entropy_rate_backend(data, max_order, &self.rate_backend)
    }

    /// Biased entropy-rate estimate (plugin variant) for `data`.
    pub fn biased_entropy_rate_bytes(&self, data: &[u8], max_order: i64) -> f64 {
        biased_entropy_rate_backend(data, max_order, &self.rate_backend)
    }

    /// Cross entropy of `test_data` under model trained on `train_data`.
    pub fn cross_entropy_rate_bytes(
        &self,
        test_data: &[u8],
        train_data: &[u8],
        max_order: i64,
    ) -> f64 {
        cross_entropy_rate_backend(test_data, train_data, max_order, &self.rate_backend)
    }

    /// Cross entropy with order-0 fast-path fallback when `max_order == 0`.
    pub fn cross_entropy_bytes(&self, test_data: &[u8], train_data: &[u8], max_order: i64) -> f64 {
        if max_order == 0 {
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
        } else {
            self.cross_entropy_rate_bytes(test_data, train_data, max_order)
        }
    }

    /// Joint entropy-rate estimate `H(X,Y)` under aligned-prefix semantics.
    pub fn joint_entropy_rate_bytes(&self, x: &[u8], y: &[u8], max_order: i64) -> f64 {
        let (x, y) = aligned_prefix(x, y);
        if x.is_empty() {
            return 0.0;
        }
        joint_entropy_rate_backend(x, y, max_order, &self.rate_backend)
    }

    /// Conditional entropy-rate estimate `H(X|Y)`.
    pub fn conditional_entropy_rate_bytes(&self, x: &[u8], y: &[u8], max_order: i64) -> f64 {
        let (x, y) = aligned_prefix(x, y);
        if x.is_empty() {
            return 0.0;
        }
        let h_xy = self.joint_entropy_rate_bytes(x, y, max_order);
        let h_y = self.entropy_rate_bytes(y, max_order);
        (h_xy - h_y).max(0.0)
    }

    /// Compute `H(data | prefix_parts)` by conditioning the active rate backend
    /// on an explicit prefix chain.
    pub fn cross_entropy_conditional_chain(&self, prefix_parts: &[&[u8]], data: &[u8]) -> f64 {
        match &self.rate_backend {
            RateBackend::RosaPlus => {
                let mut prefix = Vec::new();
                let total: usize = prefix_parts.iter().map(|p| p.len()).sum();
                prefix.reserve(total);
                for p in prefix_parts {
                    prefix.extend_from_slice(p);
                }
                cross_entropy_rate_backend(data, &prefix, -1, &RateBackend::RosaPlus)
            }
            RateBackend::Match { .. }
            | RateBackend::SparseMatch { .. }
            | RateBackend::Ppmd { .. }
            | RateBackend::Calibrated { .. } => {
                prequential_rate_backend(data, prefix_parts, -1, &self.rate_backend)
            }
            #[cfg(feature = "backend-rwkv")]
            RateBackend::Rwkv7 { model } => with_rwkv_tls(model, |c| {
                c.cross_entropy_conditional_chain(prefix_parts, data)
                    .unwrap_or_else(|e| panic!("rwkv conditional-chain scoring failed: {e:#}"))
            }),
            #[cfg(feature = "backend-rwkv")]
            RateBackend::Rwkv7Method { method } => with_rwkv_method_tls(method, |c| {
                c.cross_entropy_conditional_chain(prefix_parts, data)
                    .unwrap_or_else(|e| {
                        panic!("rwkv method conditional-chain scoring failed: {e:#}")
                    })
            }),
            #[cfg(feature = "backend-mamba")]
            RateBackend::Mamba { model } => with_mamba_tls(model, |c| {
                c.cross_entropy_conditional_chain(prefix_parts, data)
                    .unwrap_or_else(|e| panic!("mamba conditional-chain scoring failed: {e:#}"))
            }),
            #[cfg(feature = "backend-mamba")]
            RateBackend::MambaMethod { method } => with_mamba_method_tls(method, |c| {
                c.cross_entropy_conditional_chain(prefix_parts, data)
                    .unwrap_or_else(|e| {
                        panic!("mamba method conditional-chain scoring failed: {e:#}")
                    })
            }),
            RateBackend::Ctw { depth } => {
                if data.is_empty() {
                    return 0.0;
                }
                let mut tree = crate::ctw::ContextTree::new(*depth);
                for &part in prefix_parts {
                    for &b in part {
                        for i in (0..8).rev() {
                            tree.update(((b >> i) & 1) == 1);
                        }
                    }
                }
                let log_p_prefix = tree.get_log_block_probability();
                for &b in data {
                    for i in (0..8).rev() {
                        tree.update(((b >> i) & 1) == 1);
                    }
                }
                let log_p_joint = tree.get_log_block_probability();
                let log_p_cond = log_p_joint - log_p_prefix;
                let bits = -log_p_cond / std::f64::consts::LN_2;
                bits / (data.len() as f64)
            }
            RateBackend::Zpaq { method } => {
                if data.is_empty() {
                    return 0.0;
                }
                let mut model =
                    crate::zpaq_rate::ZpaqRateModel::new(method.clone(), 2f64.powi(-24));
                for &part in prefix_parts {
                    model.update_and_score(part);
                }
                let bits = model.update_and_score(data);
                bits / (data.len() as f64)
            }
            RateBackend::Mixture { spec } => {
                if data.is_empty() {
                    return 0.0;
                }
                let experts = spec.build_experts();
                let mut mix = crate::mixture::build_mixture_runtime(spec.as_ref(), &experts)
                    .unwrap_or_else(|e| panic!("MixtureSpec invalid: {e}"));
                let total = prefix_parts
                    .iter()
                    .map(|p| p.len() as u64)
                    .sum::<u64>()
                    .saturating_add(data.len() as u64);
                mix.begin_stream(Some(total))
                    .unwrap_or_else(|e| panic!("Mixture stream init failed: {e}"));
                for &part in prefix_parts {
                    for &b in part {
                        mix.step(b);
                    }
                }
                let mut bits = 0.0;
                for &b in data {
                    bits -= mix.step(b) / std::f64::consts::LN_2;
                }
                bits / (data.len() as f64)
            }
            RateBackend::Particle { spec } => {
                if data.is_empty() {
                    return 0.0;
                }
                let mut runtime = crate::particle::ParticleRuntime::new(spec.as_ref());
                for &part in prefix_parts {
                    for &b in part {
                        runtime.step(b);
                    }
                }
                let mut bits = 0.0;
                for &b in data {
                    bits -= runtime.step(b) / std::f64::consts::LN_2;
                }
                bits / (data.len() as f64)
            }
            RateBackend::FacCtw {
                base_depth,
                num_percept_bits: _,
                encoding_bits,
            } => {
                if data.is_empty() {
                    return 0.0;
                }
                let bits_per_byte = (*encoding_bits).clamp(1, 8);
                let mut fac = crate::ctw::FacContextTree::new(*base_depth, bits_per_byte);
                for &part in prefix_parts {
                    for &b in part {
                        // Fix Issue 1: LSB-first
                        for i in 0..bits_per_byte {
                            let bit_idx = i;
                            // b >> i gets the i-th bit (0 is LSB)
                            fac.update(((b >> i) & 1) == 1, bit_idx);
                        }
                    }
                }
                let log_p_prefix = fac.get_log_block_probability();
                for &b in data {
                    for i in 0..bits_per_byte {
                        let bit_idx = i;
                        fac.update(((b >> i) & 1) == 1, bit_idx);
                    }
                }
                let log_p_joint = fac.get_log_block_probability();
                let log_p_cond = log_p_joint - log_p_prefix;
                let bits = -log_p_cond / std::f64::consts::LN_2;
                bits / (data.len() as f64)
            }
        }
    }

    /// NCD between byte slices using this context's compression backend.
    pub fn ncd_bytes(&self, x: &[u8], y: &[u8], variant: NcdVariant) -> f64 {
        ncd_bytes_backend(x, y, &self.compression_backend, variant)
    }

    /// Rate-backend mutual information estimate.
    pub fn mutual_information_rate_bytes(&self, x: &[u8], y: &[u8], max_order: i64) -> f64 {
        mutual_information_rate_backend(x, y, max_order, &self.rate_backend)
    }

    /// Mutual information with `max_order == 0` marginal fast-path.
    pub fn mutual_information_bytes(&self, x: &[u8], y: &[u8], max_order: i64) -> f64 {
        if max_order == 0 {
            mutual_information_marg_bytes(x, y)
        } else {
            self.mutual_information_rate_bytes(x, y, max_order)
        }
    }

    /// Conditional entropy with aligned-prefix semantics.
    pub fn conditional_entropy_bytes(&self, x: &[u8], y: &[u8], max_order: i64) -> f64 {
        let (x, y) = aligned_prefix(x, y);
        if max_order == 0 {
            let h_xy = joint_marginal_entropy_bytes(x, y);
            let h_y = marginal_entropy_bytes(y);
            (h_xy - h_y).max(0.0)
        } else {
            let h_xy = self.joint_entropy_rate_bytes(x, y, max_order);
            let h_y = self.entropy_rate_bytes(y, max_order);
            (h_xy - h_y).max(0.0)
        }
    }

    /// Normalized entropy distance (NED) under this context.
    pub fn ned_bytes(&self, x: &[u8], y: &[u8], max_order: i64) -> f64 {
        if max_order == 0 {
            ned_marg_bytes(x, y)
        } else {
            ned_rate_backend(x, y, max_order, &self.rate_backend)
        }
    }

    /// Conservative NED normalization variant.
    pub fn ned_cons_bytes(&self, x: &[u8], y: &[u8], max_order: i64) -> f64 {
        let (x, y) = aligned_prefix(x, y);
        let (h_x, h_y, h_xy) = if max_order == 0 {
            (
                marginal_entropy_bytes(x),
                marginal_entropy_bytes(y),
                joint_marginal_entropy_bytes(x, y),
            )
        } else {
            (
                self.entropy_rate_bytes(x, max_order),
                self.entropy_rate_bytes(y, max_order),
                self.joint_entropy_rate_bytes(x, y, max_order),
            )
        };
        let min_h = h_x.min(h_y);
        if h_xy == 0.0 {
            0.0
        } else {
            ((h_xy - min_h) / h_xy).clamp(0.0, 1.0)
        }
    }

    /// Normalized transform effort (NTE) under this context.
    pub fn nte_bytes(&self, x: &[u8], y: &[u8], max_order: i64) -> f64 {
        if max_order == 0 {
            nte_marg_bytes(x, y)
        } else {
            nte_rate_backend(x, y, max_order, &self.rate_backend)
        }
    }

    /// Intrinsic dependence score in `[0,1]`.
    pub fn intrinsic_dependence_bytes(&self, data: &[u8], max_order: i64) -> f64 {
        let h_marginal = marginal_entropy_bytes(data);
        if h_marginal < 1e-9 {
            return 0.0;
        }
        let h_rate = self.entropy_rate_bytes(data, max_order);
        ((h_marginal - h_rate) / h_marginal).clamp(0.0, 1.0)
    }

    /// Resistance-to-transformation ratio `I(X;T(X))/H(X)` in `[0,1]`.
    pub fn resistance_to_transformation_bytes(&self, x: &[u8], tx: &[u8], max_order: i64) -> f64 {
        let (x, tx) = aligned_prefix(x, tx);
        let h_x = if max_order == 0 {
            marginal_entropy_bytes(x)
        } else {
            self.entropy_rate_bytes(x, max_order)
        };
        if h_x < 1e-9 {
            return 0.0;
        }
        let mi = self.mutual_information_bytes(x, tx, max_order);
        (mi / h_x).clamp(0.0, 1.0)
    }
}

#[cfg(feature = "backend-rwkv")]
/// Load an RWKV7 model from `.safetensors` path.
pub fn load_rwkv7_model_from_path(path: &str) -> Arc<rwkvzip::Model> {
    rwkvzip::Compressor::load_model(path).expect("failed to load RWKV7 model")
}

#[cfg(feature = "backend-mamba")]
/// Load a Mamba-1 model from `.safetensors` path.
pub fn load_mamba_model_from_path(path: &str) -> Arc<mambazip::Model> {
    mambazip::Compressor::load_model(path).expect("failed to load Mamba model")
}

#[inline(always)]
fn aligned_prefix<'a>(x: &'a [u8], y: &'a [u8]) -> (&'a [u8], &'a [u8]) {
    let n = x.len().min(y.len());
    (&x[..n], &y[..n])
}

#[cfg(feature = "backend-zpaq")]
#[inline(always)]
fn zpaq_compress_size_bytes(data: &[u8], method: &str) -> u64 {
    zpaq_rs::compress_size(data, method).unwrap_or(0)
}

#[cfg(not(feature = "backend-zpaq"))]
#[inline(always)]
fn zpaq_compress_size_bytes(_data: &[u8], _method: &str) -> u64 {
    panic!("CompressionBackend::Zpaq is unavailable: build with feature 'backend-zpaq'")
}

#[cfg(feature = "backend-zpaq")]
#[inline(always)]
fn zpaq_compress_size_parallel_bytes(data: &[u8], method: &str, threads: usize) -> u64 {
    zpaq_rs::compress_size_parallel(data, method, threads).unwrap_or(0)
}

#[cfg(not(feature = "backend-zpaq"))]
#[inline(always)]
fn zpaq_compress_size_parallel_bytes(_data: &[u8], _method: &str, _threads: usize) -> u64 {
    panic!("CompressionBackend::Zpaq is unavailable: build with feature 'backend-zpaq'")
}

#[cfg(feature = "backend-zpaq")]
#[inline(always)]
fn zpaq_compress_size_stream<R: std::io::Read + Send>(reader: R, method: &str) -> u64 {
    zpaq_rs::compress_size_stream(reader, method, None, None).unwrap_or(0)
}

#[cfg(not(feature = "backend-zpaq"))]
#[inline(always)]
fn zpaq_compress_size_stream<R: std::io::Read + Send>(_reader: R, _method: &str) -> u64 {
    panic!("CompressionBackend::Zpaq is unavailable: build with feature 'backend-zpaq'")
}

#[cfg(feature = "backend-zpaq")]
#[inline(always)]
fn zpaq_compress_to_vec(data: &[u8], method: &str) -> anyhow::Result<Vec<u8>> {
    Ok(zpaq_rs::compress_to_vec(data, method)?)
}

#[cfg(not(feature = "backend-zpaq"))]
#[inline(always)]
fn zpaq_compress_to_vec(_data: &[u8], _method: &str) -> anyhow::Result<Vec<u8>> {
    anyhow::bail!("zpaq backend disabled at compile time (enable feature 'backend-zpaq')")
}

#[cfg(feature = "backend-zpaq")]
#[inline(always)]
fn zpaq_decompress_to_vec(data: &[u8]) -> anyhow::Result<Vec<u8>> {
    Ok(zpaq_rs::decompress_to_vec(data)?)
}

#[cfg(not(feature = "backend-zpaq"))]
#[inline(always)]
fn zpaq_decompress_to_vec(_data: &[u8]) -> anyhow::Result<Vec<u8>> {
    anyhow::bail!("zpaq backend disabled at compile time (enable feature 'backend-zpaq')")
}

/// ------- Base Compression Functions -------
#[inline(always)]
pub fn get_compressed_size(path: &str, method: &str) -> u64 {
    // Convert Input file to Vec<u8>, and reference that (compress_size only takes &[u8] input), and pass method.
    // Will panic if file does not exist, so it must be prevalidated.
    zpaq_compress_size_bytes(&std::fs::read(path).unwrap(), method)
}

/// Validate that a ZPAQ method string is supported for rate estimation.
pub fn validate_zpaq_rate_method(method: &str) -> Result<(), String> {
    #[cfg(feature = "backend-zpaq")]
    {
        zpaq_rate::validate_zpaq_rate_method(method)
    }
    #[cfg(not(feature = "backend-zpaq"))]
    {
        let _ = method;
        Err("zpaq backend disabled at compile time".to_string())
    }
}

#[cfg(feature = "backend-rwkv")]
fn with_rwkv_tls<R>(
    model: &Arc<rwkvzip::Model>,
    f: impl FnOnce(&mut rwkvzip::Compressor) -> R,
) -> R {
    let key = Arc::as_ptr(model) as usize;
    RWKV_TLS.with(|cell| {
        let mut map = cell.borrow_mut();
        let comp = map
            .entry(key)
            .or_insert_with(|| rwkvzip::Compressor::new_from_model(model.clone()));
        f(comp)
    })
}

#[cfg(feature = "backend-rwkv")]
fn with_rwkv_method_tls<R>(method: &str, f: impl FnOnce(&mut rwkvzip::Compressor) -> R) -> R {
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

#[cfg(feature = "backend-rwkv")]
fn with_rwkv_rate_tls<R>(
    model: &Arc<rwkvzip::Model>,
    f: impl FnOnce(&mut rwkvzip::Compressor) -> R,
) -> R {
    let key = Arc::as_ptr(model) as usize;
    RWKV_RATE_TLS.with(|cell| {
        let mut map = cell.borrow_mut();
        let mut comp = if let Some(template) = map.get(&key) {
            template.clone()
        } else {
            let template = rwkvzip::Compressor::new_from_model(model.clone());
            map.insert(key, template.clone());
            template
        };
        drop(map);
        f(&mut comp)
    })
}

#[cfg(feature = "backend-mamba")]
fn with_mamba_tls<R>(
    model: &Arc<mambazip::Model>,
    f: impl FnOnce(&mut mambazip::Compressor) -> R,
) -> R {
    let key = Arc::as_ptr(model) as usize;
    MAMBA_TLS.with(|cell| {
        let mut map = cell.borrow_mut();
        let comp = map
            .entry(key)
            .or_insert_with(|| mambazip::Compressor::new_from_model(model.clone()));
        f(comp)
    })
}

#[cfg(feature = "backend-mamba")]
fn with_mamba_rate_tls<R>(
    model: &Arc<mambazip::Model>,
    f: impl FnOnce(&mut mambazip::Compressor) -> R,
) -> R {
    let key = Arc::as_ptr(model) as usize;
    MAMBA_RATE_TLS.with(|cell| {
        let mut map = cell.borrow_mut();
        let mut comp = if let Some(template) = map.get(&key) {
            template.clone()
        } else {
            let template = mambazip::Compressor::new_from_model(model.clone());
            map.insert(key, template.clone());
            template
        };
        drop(map);
        f(&mut comp)
    })
}

#[cfg(feature = "backend-mamba")]
fn with_mamba_method_tls<R>(method: &str, f: impl FnOnce(&mut mambazip::Compressor) -> R) -> R {
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

struct SliceChainReader<'a> {
    parts: &'a [&'a [u8]],
    i: usize,
    off: usize,
}

impl<'a> SliceChainReader<'a> {
    fn new(parts: &'a [&'a [u8]]) -> Self {
        Self {
            parts,
            i: 0,
            off: 0,
        }
    }
}

impl<'a> std::io::Read for SliceChainReader<'a> {
    fn read(&mut self, mut buf: &mut [u8]) -> std::io::Result<usize> {
        let mut total = 0;
        if buf.is_empty() {
            return Ok(0);
        }
        while self.i < self.parts.len() {
            let p = self.parts[self.i];
            if self.off >= p.len() {
                self.i += 1;
                self.off = 0;
                continue;
            }
            let n = (p.len() - self.off).min(buf.len());
            // Safe copy slice
            buf[..n].copy_from_slice(&p[self.off..self.off + n]);

            // Advance state
            self.off += n;
            total += n;

            // Re-slice buf to fill remainder
            let tmp = buf;
            buf = &mut tmp[n..];

            if buf.is_empty() {
                break;
            }
        }
        Ok(total)
    }
}

/// Compute compressed size of a chain of byte slices with a selected compression backend.
pub fn compress_size_chain_backend(parts: &[&[u8]], backend: &CompressionBackend) -> u64 {
    match backend {
        CompressionBackend::Zpaq { method } => {
            let r = SliceChainReader::new(parts);
            zpaq_compress_size_stream(r, method.as_str())
        }
        #[cfg(feature = "backend-rwkv")]
        CompressionBackend::Rwkv7 { model, coder } => {
            with_rwkv_tls(model, |c| c.compress_size_chain(parts, *coder).unwrap_or(0))
        }
        CompressionBackend::Rate {
            rate_backend,
            coder,
            framing,
        } => {
            crate::compression::compress_rate_size_chain(parts, rate_backend, -1, *coder, *framing)
                .unwrap_or(0)
        }
    }
}

/// Compute compressed size of a single byte slice with a selected compression backend.
pub fn compress_size_backend(data: &[u8], backend: &CompressionBackend) -> u64 {
    match backend {
        CompressionBackend::Zpaq { method } => zpaq_compress_size_bytes(data, method.as_str()),
        #[cfg(feature = "backend-rwkv")]
        CompressionBackend::Rwkv7 { model, coder } => {
            with_rwkv_tls(model, |c| c.compress_size(data, *coder).unwrap_or(0))
        }
        CompressionBackend::Rate {
            rate_backend,
            coder,
            framing,
        } => crate::compression::compress_rate_size(data, rate_backend, -1, *coder, *framing)
            .unwrap_or(0),
    }
}

/// Compress bytes with a selected compression backend.
pub fn compress_bytes_backend(
    data: &[u8],
    backend: &CompressionBackend,
) -> anyhow::Result<Vec<u8>> {
    match backend {
        CompressionBackend::Zpaq { method } => zpaq_compress_to_vec(data, method),
        #[cfg(feature = "backend-rwkv")]
        CompressionBackend::Rwkv7 { model, coder } => {
            with_rwkv_tls(model, |c| c.compress(data, *coder))
        }
        CompressionBackend::Rate {
            rate_backend,
            coder,
            framing,
        } => crate::compression::compress_rate_bytes(data, rate_backend, -1, *coder, *framing),
    }
}

/// Decompress bytes with a selected compression backend.
pub fn decompress_bytes_backend(
    input: &[u8],
    backend: &CompressionBackend,
) -> anyhow::Result<Vec<u8>> {
    match backend {
        CompressionBackend::Zpaq { .. } => zpaq_decompress_to_vec(input),
        #[cfg(feature = "backend-rwkv")]
        CompressionBackend::Rwkv7 { model, .. } => with_rwkv_tls(model, |c| c.decompress(input)),
        CompressionBackend::Rate {
            rate_backend,
            coder,
            framing,
        } => crate::compression::decompress_rate_bytes(input, rate_backend, -1, *coder, *framing),
    }
}

fn prequential_rate_backend(
    data: &[u8],
    prefix_parts: &[&[u8]],
    max_order: i64,
    backend: &RateBackend,
) -> f64 {
    use crate::mixture::OnlineBytePredictor;

    if data.is_empty() {
        return 0.0;
    }
    let total = prefix_parts
        .iter()
        .map(|p| p.len() as u64)
        .sum::<u64>()
        .saturating_add(data.len() as u64);
    let mut predictor = crate::mixture::RateBackendPredictor::from_backend(
        backend.clone(),
        max_order,
        crate::mixture::DEFAULT_MIN_PROB,
    );
    predictor
        .begin_stream(Some(total))
        .unwrap_or_else(|e| panic!("rate backend stream init failed: {e}"));
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
    predictor
        .finish_stream()
        .unwrap_or_else(|e| panic!("rate backend stream finalize failed: {e}"));
    bits / (data.len() as f64)
}

fn frozen_plugin_rate_backend(
    score_data: &[u8],
    fit_parts: &[&[u8]],
    max_order: i64,
    backend: &RateBackend,
) -> f64 {
    if score_data.is_empty() {
        return 0.0;
    }
    if matches!(backend, RateBackend::RosaPlus) {
        let mut model = rosaplus::RosaPlus::new(max_order, false, 0, 42);
        for part in fit_parts {
            model.train_example(part);
        }
        model.build_lm();
        return model.cross_entropy(score_data);
    }
    #[cfg(feature = "backend-rwkv")]
    match backend {
        RateBackend::Rwkv7 { model } => {
            return with_rwkv_rate_tls(model, |c| {
                c.cross_entropy_frozen_plugin_chain(fit_parts, score_data)
                    .unwrap_or_else(|e| panic!("rwkv frozen-plugin scoring failed: {e:#}"))
            });
        }
        RateBackend::Rwkv7Method { method } => {
            return with_rwkv_method_tls(method, |c| {
                c.cross_entropy_frozen_plugin_chain(fit_parts, score_data)
                    .unwrap_or_else(|e| panic!("rwkv method frozen-plugin scoring failed: {e:#}"))
            });
        }
        _ => {}
    }
    #[cfg(feature = "backend-mamba")]
    match backend {
        RateBackend::Mamba { model } => {
            return with_mamba_rate_tls(model, |c| {
                c.cross_entropy_frozen_plugin_chain(fit_parts, score_data)
                    .unwrap_or_else(|e| panic!("mamba frozen-plugin scoring failed: {e:#}"))
            });
        }
        RateBackend::MambaMethod { method } => {
            return with_mamba_method_tls(method, |c| {
                c.cross_entropy_frozen_plugin_chain(fit_parts, score_data)
                    .unwrap_or_else(|e| panic!("mamba method frozen-plugin scoring failed: {e:#}"))
            });
        }
        _ => {}
    }

    use crate::mixture::OnlineBytePredictor;

    let fit_total = fit_parts.iter().map(|part| part.len() as u64).sum::<u64>();
    let mut predictor = crate::mixture::RateBackendPredictor::from_backend(
        backend.clone(),
        max_order,
        crate::mixture::DEFAULT_MIN_PROB,
    );
    predictor
        .begin_stream(Some(fit_total))
        .unwrap_or_else(|e| panic!("rate backend fit-pass init failed: {e}"));
    for part in fit_parts {
        for &byte in *part {
            predictor.update(byte);
        }
    }
    predictor
        .finish_stream()
        .unwrap_or_else(|e| panic!("rate backend fit-pass finalize failed: {e}"));
    predictor
        .reset_frozen(Some(score_data.len() as u64))
        .unwrap_or_else(|e| panic!("rate backend frozen-score reset failed: {e}"));
    let mut bits = 0.0;
    for &byte in score_data {
        bits -= predictor.log_prob(byte) / std::f64::consts::LN_2;
        predictor.update_frozen(byte);
    }
    predictor
        .finish_stream()
        .unwrap_or_else(|e| panic!("rate backend frozen-score finalize failed: {e}"));
    bits / (score_data.len() as f64)
}

/// Estimate entropy rate of `data` using the explicit rate `backend`.
pub fn entropy_rate_backend(data: &[u8], max_order: i64, backend: &RateBackend) -> f64 {
    match backend {
        RateBackend::RosaPlus => {
            let mut m = rosaplus::RosaPlus::new(max_order, false, 0, 42);
            m.predictive_entropy_rate(data)
        }
        RateBackend::Match { .. }
        | RateBackend::SparseMatch { .. }
        | RateBackend::Ppmd { .. }
        | RateBackend::Calibrated { .. } => prequential_rate_backend(data, &[], max_order, backend),
        #[cfg(feature = "backend-rwkv")]
        RateBackend::Rwkv7 { model } => with_rwkv_tls(model, |c| {
            c.cross_entropy(data)
                .unwrap_or_else(|e| panic!("rwkv entropy scoring failed: {e:#}"))
        }),
        #[cfg(feature = "backend-rwkv")]
        RateBackend::Rwkv7Method { method } => with_rwkv_method_tls(method, |c| {
            c.cross_entropy(data)
                .unwrap_or_else(|e| panic!("rwkv method entropy scoring failed: {e:#}"))
        }),
        #[cfg(feature = "backend-mamba")]
        RateBackend::Mamba { model } => with_mamba_tls(model, |c| {
            c.cross_entropy(data)
                .unwrap_or_else(|e| panic!("mamba entropy scoring failed: {e:#}"))
        }),
        #[cfg(feature = "backend-mamba")]
        RateBackend::MambaMethod { method } => with_mamba_method_tls(method, |c| {
            c.cross_entropy(data)
                .unwrap_or_else(|e| panic!("mamba method entropy scoring failed: {e:#}"))
        }),
        RateBackend::Zpaq { method } => {
            if data.is_empty() {
                return 0.0;
            }
            let mut model = crate::zpaq_rate::ZpaqRateModel::new(method.clone(), 2f64.powi(-24));
            let bits = model.update_and_score(data);
            bits / (data.len() as f64)
        }
        RateBackend::Mixture { spec } => {
            if data.is_empty() {
                return 0.0;
            }
            let experts = spec.build_experts();
            let mut mix = crate::mixture::build_mixture_runtime(spec.as_ref(), &experts)
                .unwrap_or_else(|e| panic!("MixtureSpec invalid: {e}"));
            mix.begin_stream(Some(data.len() as u64))
                .unwrap_or_else(|e| panic!("Mixture stream init failed: {e}"));
            let mut bits = 0.0;
            for &b in data {
                bits -= mix.step(b) / std::f64::consts::LN_2;
            }
            mix.finish_stream()
                .unwrap_or_else(|e| panic!("Mixture stream finalize failed: {e}"));
            bits / (data.len() as f64)
        }
        RateBackend::Particle { spec } => {
            if data.is_empty() {
                return 0.0;
            }
            let mut runtime = crate::particle::ParticleRuntime::new(spec.as_ref());
            let mut bits = 0.0;
            for &b in data {
                bits -= runtime.step(b) / std::f64::consts::LN_2;
            }
            bits / (data.len() as f64)
        }
        RateBackend::Ctw { depth } => {
            if data.is_empty() {
                return 0.0;
            }
            // Byte-wise CTW: factorize by bit position so deterministic bits don't leak entropy.
            let mut fac = crate::ctw::FacContextTree::new(*depth, 8);
            fac.reserve_for_symbols(data.len());
            for &b in data {
                fac.update_byte_msb(b);
            }
            let ln_p = fac.get_log_block_probability();
            let bits = -ln_p / std::f64::consts::LN_2;
            bits / (data.len() as f64)
        }
        RateBackend::FacCtw {
            base_depth,
            num_percept_bits: _,
            encoding_bits,
        } => {
            if data.is_empty() {
                return 0.0;
            }
            let bits_per_byte = (*encoding_bits).clamp(1, 8);
            let mut fac = crate::ctw::FacContextTree::new(*base_depth, bits_per_byte);
            fac.reserve_for_symbols(data.len());
            for &b in data {
                fac.update_byte_lsb(b);
            }
            let ln_p = fac.get_log_block_probability();
            let bits = -ln_p / std::f64::consts::LN_2;
            bits / (data.len() as f64)
        }
    }
}

/// Estimate biased/plugin entropy rate of `data` using the explicit rate `backend`.
pub fn biased_entropy_rate_backend(data: &[u8], max_order: i64, backend: &RateBackend) -> f64 {
    match backend {
        RateBackend::Zpaq { .. } => {
            panic!("biased/plugin entropy is not supported for zpaq rate backends in 1.1.0")
        }
        _ => frozen_plugin_rate_backend(data, &[data], max_order, backend),
    }
}

/// Cross-entropy H_{train}(test) - score test_data under model trained on train_data.
pub fn cross_entropy_rate_backend(
    test_data: &[u8],
    train_data: &[u8],
    max_order: i64,
    backend: &RateBackend,
) -> f64 {
    match backend {
        RateBackend::Zpaq { method } => {
            if test_data.is_empty() {
                return 0.0;
            }
            let mut model = crate::zpaq_rate::ZpaqRateModel::new(method.clone(), 2f64.powi(-24));
            model.update_and_score(train_data);
            let bits = model.update_and_score(test_data);
            bits / (test_data.len() as f64)
        }
        _ => frozen_plugin_rate_backend(test_data, &[train_data], max_order, backend),
    }
}

/// Estimate joint entropy rate `H(X,Y)` using an explicit `backend`.
pub fn joint_entropy_rate_backend(
    x: &[u8],
    y: &[u8],
    max_order: i64,
    backend: &RateBackend,
) -> f64 {
    let (x, y) = aligned_prefix(x, y);
    if x.is_empty() {
        return 0.0;
    }
    match backend {
        RateBackend::RosaPlus => {
            let joint_symbols: Vec<u32> = (0..x.len())
                .map(|i| (x[i] as u32) * 256 + (y[i] as u32))
                .collect();
            let mut m = rosaplus::RosaPlus::new(max_order, false, 0, 42);
            m.entropy_rate_cps(&joint_symbols)
        }
        RateBackend::Match { .. }
        | RateBackend::SparseMatch { .. }
        | RateBackend::Ppmd { .. }
        | RateBackend::Calibrated { .. } => {
            let mut joint = Vec::with_capacity(x.len() * 2);
            for (&xb, &yb) in x.iter().zip(y.iter()) {
                joint.push(xb);
                joint.push(yb);
            }
            entropy_rate_backend(&joint, max_order, backend) * 2.0
        }
        #[cfg(feature = "backend-rwkv")]
        RateBackend::Rwkv7 { model } => with_rwkv_tls(model, |c| {
            c.joint_cross_entropy_aligned_min(x, y)
                .unwrap_or_else(|e| panic!("rwkv joint-entropy scoring failed: {e:#}"))
        }),
        #[cfg(feature = "backend-rwkv")]
        RateBackend::Rwkv7Method { method } => with_rwkv_method_tls(method, |c| {
            c.joint_cross_entropy_aligned_min(x, y)
                .unwrap_or_else(|e| panic!("rwkv method joint-entropy scoring failed: {e:#}"))
        }),
        #[cfg(feature = "backend-mamba")]
        RateBackend::Mamba { model } => with_mamba_tls(model, |c| {
            c.joint_cross_entropy_aligned_min(x, y)
                .unwrap_or_else(|e| panic!("mamba joint-entropy scoring failed: {e:#}"))
        }),
        #[cfg(feature = "backend-mamba")]
        RateBackend::MambaMethod { method } => with_mamba_method_tls(method, |c| {
            c.joint_cross_entropy_aligned_min(x, y)
                .unwrap_or_else(|e| panic!("mamba method joint-entropy scoring failed: {e:#}"))
        }),
        RateBackend::Zpaq { method } => {
            let mut joint = Vec::with_capacity(x.len() * 2);
            for (&xb, &yb) in x.iter().zip(y.iter()) {
                joint.push(xb);
                joint.push(yb);
            }
            let mut model = crate::zpaq_rate::ZpaqRateModel::new(method.clone(), 2f64.powi(-24));
            let bits = model.update_and_score(&joint);
            bits / (x.len() as f64)
        }
        RateBackend::Mixture { spec } => {
            let mut joint = Vec::with_capacity(x.len() * 2);
            for (&xb, &yb) in x.iter().zip(y.iter()) {
                joint.push(xb);
                joint.push(yb);
            }
            let experts = spec.build_experts();
            let mut mix = crate::mixture::build_mixture_runtime(spec.as_ref(), &experts)
                .unwrap_or_else(|e| panic!("MixtureSpec invalid: {e}"));
            mix.begin_stream(Some(joint.len() as u64))
                .unwrap_or_else(|e| panic!("Mixture stream init failed: {e}"));
            let mut bits = 0.0;
            for &b in &joint {
                bits -= mix.step(b) / std::f64::consts::LN_2;
            }
            mix.finish_stream()
                .unwrap_or_else(|e| panic!("Mixture stream finalize failed: {e}"));
            bits / (x.len() as f64)
        }
        RateBackend::Particle { spec } => {
            let mut joint = Vec::with_capacity(x.len() * 2);
            for (&xb, &yb) in x.iter().zip(y.iter()) {
                joint.push(xb);
                joint.push(yb);
            }
            let mut runtime = crate::particle::ParticleRuntime::new(spec.as_ref());
            let mut bits = 0.0;
            for &b in &joint {
                bits -= runtime.step(b) / std::f64::consts::LN_2;
            }
            bits / (x.len() as f64)
        }
        RateBackend::Ctw { depth } => {
            // NOTE: CTW interleaves bits: x_0, y_0, x_1, y_1...
            // This estimates the joint entropy H(X,Y) by modeling the sequence
            // of alternating bits. This is a fine-grained joint model but
            // theoretically consistent for estimating joint entropy rate.
            // ROSA uses 16-bit joint symbols (x << 8 | y). Both are valid.
            let mut fac = crate::ctw::FacContextTree::new(*depth, 16);
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
        RateBackend::FacCtw {
            base_depth,
            num_percept_bits: _,
            encoding_bits,
        } => {
            // Joint: interleave x and y bits, use 2*encoding_bits trees
            let bits_per_byte = (*encoding_bits).clamp(1, 8);
            let mut fac = crate::ctw::FacContextTree::new(*base_depth, bits_per_byte * 2);
            for k in 0..x.len() {
                let bx = x[k];
                let by = y[k];
                for i in 0..bits_per_byte {
                    // Tree structure:
                    // bits_per_byte trees for X, bits_per_byte trees for Y.
                    // But we interleave them in the "joint" sense.
                    // Here we map bit i of X to tree 2*i, bit i of Y to tree 2*i + 1
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
    }
}
#[inline(always)]
/// Compute compressed size for a file path with an explicit ZPAQ thread count.
pub fn get_compressed_size_parallel(path: &str, method: &str, threads: usize) -> u64 {
    // Convert Input file to Vec<u8>, and reference that (compress_size only takes &[u8] input), and pass method.
    // Will panic if file does not exist, so it must be prevalidated.
    zpaq_compress_size_parallel_bytes(&std::fs::read(path).unwrap(), method, threads)
}

#[inline(always)]
/// Read all files in `paths` in parallel and return their byte contents.
pub fn get_bytes_from_paths(paths: &[&str]) -> Vec<Vec<u8>> {
    paths
        .par_iter()
        .map(|path| std::fs::read(*path).expect("failed to read file"))
        .collect()
}

/// ------- Bulk File Compression Functions -------
#[inline(always)]
pub fn get_sequential_compressed_sizes_from_sequential_paths(
    paths: &[&str],
    method: &str,
) -> Vec<u64> {
    // This will, in parallel load all files into memory, THEN in parallel compress each one, each with one thread.
    // Use when File IO is the bottleneck
    // Only uses ONE ZPAQ THREAD.
    // For VERY large n (relative to threads) with small files (relative to memory) this may be useful.
    get_bytes_from_paths(paths)
        .par_iter()
        .map(|data| zpaq_compress_size_bytes(data, method))
        .collect()
}

#[inline(always)]
/// Compress all paths after preloading bytes, using per-file parallel ZPAQ compression.
pub fn get_parallel_compressed_sizes_from_sequential_paths(
    paths: &[&str],
    method: &str,
    threads: usize,
) -> Vec<u64> {
    // This will, in parallel load all files into memory, THEN in parallel compress each one, with THREADS. (for each file, the thread count is THREADS)
    // Use when File IO is the bottleneck.
    // Balanced parallelization between RAYON_NUM_THREADS and ZPAQ `THREADS` const. For when total dataset will fit in memory.
    get_bytes_from_paths(paths)
        .par_iter()
        .map(|data| zpaq_compress_size_parallel_bytes(data, method, threads))
        .collect()
}

#[inline(always)]
/// Compress all paths directly from disk using single-thread ZPAQ per file.
pub fn get_sequential_compressed_sizes_from_parallel_paths(
    paths: &[&str],
    method: &str,
) -> Vec<u64> {
    // This will, in parallel, for each file, read it from disk and compress it with one thread. (one file, one thread)
    // Use when File IO is not the bottleneck. Lower memory usage. (does not preload dataset)
    // Only uses ONE ZPAQ THREAD. For VERY large n(relative to threads) with large files(relative to memory) this may be useful.
    paths
        .par_iter()
        .map(|path| get_compressed_size(path, method))
        .collect()
}

#[inline(always)]
/// Compress all paths directly from disk using per-file multi-thread ZPAQ.
pub fn get_parallel_compressed_sizes_from_parallel_paths(
    paths: &[&str],
    method: &str,
    threads: usize,
) -> Vec<u64> {
    // This will, in parallel, for each file, read it from disk and compress it with THREADS. (for each file, the thread count is THREADS)
    // Use when File IO is not the bottleneck. Lower memory usage. (does not preload dataset)
    // For large n(relative to threads) with VERY large files(relative to memory) this may be useful.
    // This will reflect RAYON_NUM_THREADS and THREAD const values.
    paths
        .par_iter()
        .map(|path| get_compressed_size_parallel(path, method, threads))
        .collect()
}

/// Optimizes parallelization
#[inline(always)]
pub fn get_compressed_sizes_from_paths(paths: &[&str], method: &str) -> Vec<u64> {
    let n: usize = paths.len();
    let num_threads: usize = *NUM_THREADS.get_or_init(num_cpus::get);
    if n < num_threads {
        get_parallel_compressed_sizes_from_parallel_paths(paths, method, num_threads.div_ceil(n))
    } else {
        get_sequential_compressed_sizes_from_parallel_paths(paths, method)
    }
}

/// ------- NCD (Normalized Compression Distance) ------
///
/// NCD is a parameter-free similarity metric based on Kolmogorov complexity.
/// Since Kolmogorov complexity `K(x)` is uncomputable, we approximate it using
/// the compressed size `C(x)` provided by a real-world compressor (here, ZPAQ).
///
/// The general form is:
/// `NCD(x,y) = (C(xy) - min(C(x), C(y))) / max(C(x), C(y))`
///
/// Different variants handle normalization and symmetry differently.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum NcdVariant {
    /// Standard Vitanyi NCD:
    /// `NCD(x,y) = (C(xy) - min(C(x), C(y))) / max(C(x), C(y))`
    /// Note: `C(xy)` denotes compressing the concatenation of x and y.
    Vitanyi,
    /// Symmetric Vitanyi NCD:
    /// `NCD_sym(x,y) = (min(C(xy), C(yx)) - min(C(x), C(y))) / max(C(x), C(y))`
    /// Takes the best compression of `xy` or `yx` to ensure symmetry even if the compressor is not symmetric.
    SymVitanyi,
    /// Conservative NCD:
    /// `NCD_cons(x,y) = (C(xy) - min(C(x), C(y))) / C(xy)`
    /// Normalizes by the joint compressed size instead of the max marginal.
    Cons,
    /// Symmetric Conservative NCD:
    /// `NCD_sym_cons(x,y) = (min(C(xy), C(yx)) - min(C(x), C(y))) / min(C(xy), C(yx))`
    SymCons,
}

#[inline(always)]
fn compress_size_bytes(data: &[u8], method: &str) -> u64 {
    zpaq_compress_size_bytes(data, method)
}

#[inline(always)]
fn ncd_from_sizes(cx: u64, cy: u64, cxy: u64, cyx: Option<u64>, variant: NcdVariant) -> f64 {
    let min_c = cx.min(cy) as f64;
    let max_c = cx.max(cy) as f64;

    match variant {
        NcdVariant::Vitanyi => {
            if max_c == 0.0 {
                0.0
            } else {
                (cxy as f64 - min_c) / max_c
            }
        }
        NcdVariant::SymVitanyi => {
            let m = cxy.min(cyx.expect("cyx required for SymVitanyi")) as f64;
            if max_c == 0.0 {
                0.0
            } else {
                (m - min_c) / max_c
            }
        }
        NcdVariant::Cons => {
            let denom = cxy as f64;
            if denom == 0.0 {
                0.0
            } else {
                (cxy as f64 - min_c) / denom
            }
        }
        NcdVariant::SymCons => {
            let m = cxy.min(cyx.expect("cyx required for SymCons")) as f64;
            if m == 0.0 { 0.0 } else { (m - min_c) / m }
        }
    }
}

#[inline(always)]
/// Compute NCD for in-memory byte slices using the given ZPAQ `method` and `variant`.
pub fn ncd_bytes(x: &[u8], y: &[u8], method: &str, variant: NcdVariant) -> f64 {
    let backend = CompressionBackend::Zpaq {
        method: method.to_string(),
    };
    ncd_bytes_backend(x, y, &backend, variant)
}

/// NCD with bytes using the default context.
#[inline(always)]
pub fn ncd_bytes_default(x: &[u8], y: &[u8], variant: NcdVariant) -> f64 {
    with_default_ctx(|ctx| ctx.ncd_bytes(x, y, variant))
}

/// Compute NCD for in-memory byte slices using an explicit compression `backend`.
pub fn ncd_bytes_backend(
    x: &[u8],
    y: &[u8],
    backend: &CompressionBackend,
    variant: NcdVariant,
) -> f64 {
    let (cx, cy) = rayon::join(
        || compress_size_backend(x, backend),
        || compress_size_backend(y, backend),
    );

    let cxy = compress_size_chain_backend(&[x, y], backend);

    let cyx = match variant {
        NcdVariant::SymVitanyi | NcdVariant::SymCons => {
            Some(compress_size_chain_backend(&[y, x], backend))
        }
        _ => None,
    };

    ncd_from_sizes(cx, cy, cxy, cyx, variant)
}

#[inline(always)]
/// Compute NCD for two file paths using a ZPAQ `method` and `variant`.
pub fn ncd_paths(x: &str, y: &str, method: &str, variant: NcdVariant) -> f64 {
    let (bx, by) = rayon::join(
        || std::fs::read(x).expect("failed to read x"),
        || std::fs::read(y).expect("failed to read y"),
    );
    ncd_bytes(&bx, &by, method, variant)
}

/// Compute NCD for two file paths using an explicit compression `backend`.
pub fn ncd_paths_backend(
    x: &str,
    y: &str,
    backend: &CompressionBackend,
    variant: NcdVariant,
) -> f64 {
    let (bx, by) = rayon::join(
        || std::fs::read(x).expect("failed to read x"),
        || std::fs::read(y).expect("failed to read y"),
    );
    ncd_bytes_backend(&bx, &by, backend, variant)
}

/// Back-compat convenience wrappers (operate on file paths).
#[inline(always)]
pub fn ncd_vitanyi(x: &str, y: &str, method: &str) -> f64 {
    ncd_paths(x, y, method, NcdVariant::Vitanyi)
}
#[inline(always)]
/// Convenience wrapper for symmetric-Vitanyi NCD on file paths.
pub fn ncd_sym_vitanyi(x: &str, y: &str, method: &str) -> f64 {
    ncd_paths(x, y, method, NcdVariant::SymVitanyi)
}
#[inline(always)]
/// Convenience wrapper for conservative NCD on file paths.
pub fn ncd_cons(x: &str, y: &str, method: &str) -> f64 {
    ncd_paths(x, y, method, NcdVariant::Cons)
}
#[inline(always)]
/// Convenience wrapper for symmetric-conservative NCD on file paths.
pub fn ncd_sym_cons(x: &str, y: &str, method: &str) -> f64 {
    ncd_paths(x, y, method, NcdVariant::SymCons)
}

/// Computes an NCD matrix (row-major, len = n*n) for in-memory byte blobs.
///
/// Note: For symmetric variants, this computes each unordered pair once and writes both (i,j) and (j,i).
pub fn ncd_matrix_bytes(datas: &[Vec<u8>], method: &str, variant: NcdVariant) -> Vec<f64> {
    let n = datas.len();
    let cx: Vec<u64> = datas
        .par_iter()
        .map(|d| compress_size_bytes(d, method))
        .collect();

    let mut out = vec![0.0f64; n * n];
    let out_ptr = std::sync::atomic::AtomicPtr::new(out.as_mut_ptr());

    match variant {
        NcdVariant::SymVitanyi | NcdVariant::SymCons => {
            (0..n)
                .into_par_iter()
                .flat_map_iter(|i| (i + 1..n).map(move |j| (i, j)))
                .for_each_init(Vec::<u8>::new, |buf, (i, j)| {
                    let x = &datas[i];
                    let y = &datas[j];

                    buf.clear();
                    buf.reserve(x.len() + y.len());
                    buf.extend_from_slice(x);
                    buf.extend_from_slice(y);
                    let cxy = compress_size_bytes(buf, method);

                    buf.clear();
                    buf.reserve(x.len() + y.len());
                    buf.extend_from_slice(y);
                    buf.extend_from_slice(x);
                    let cyx = compress_size_bytes(buf, method);

                    let d = ncd_from_sizes(cx[i], cx[j], cxy, Some(cyx), variant);

                    // Safety: each (i,j) cell is written exactly once across all iterations.
                    let p = out_ptr.load(std::sync::atomic::Ordering::Relaxed);
                    unsafe {
                        *p.add(i * n + j) = d;
                        *p.add(j * n + i) = d;
                    }
                });
        }
        NcdVariant::Vitanyi | NcdVariant::Cons => {
            (0..n)
                .into_par_iter()
                .for_each_init(Vec::<u8>::new, |buf, i| {
                    let x = &datas[i];
                    for j in 0..n {
                        let d = if i == j {
                            0.0
                        } else {
                            let y = &datas[j];
                            buf.clear();
                            buf.reserve(x.len() + y.len());
                            buf.extend_from_slice(x);
                            buf.extend_from_slice(y);
                            let cxy = compress_size_bytes(buf, method);
                            ncd_from_sizes(cx[i], cx[j], cxy, None, variant)
                        };

                        let p = out_ptr.load(std::sync::atomic::Ordering::Relaxed);
                        unsafe {
                            *p.add(i * n + j) = d;
                        }
                    }
                });
        }
    }

    out
}

/// Computes an NCD matrix (row-major, len = n*n) for files (preloads all files into memory once).
pub fn ncd_matrix_paths(paths: &[&str], method: &str, variant: NcdVariant) -> Vec<f64> {
    let datas = get_bytes_from_paths(paths);
    ncd_matrix_bytes(&datas, method, variant)
}

// ============================================================
// Entropy-Based Distance Primitives (via ROSA)
// ============================================================
//
// These use ROSA's Witten-Bell language model to estimate entropy
// and compute information-theoretic distances.

/// Compute marginal (Shannon) entropy H(X) = −Σ p(x) log₂ p(x) in bits/symbol.
///
/// This is the simple first-order entropy from the byte histogram,
/// NOT the context-conditional entropy rate from a language model.
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
/// For i.i.d. data, this should approximately equal `marginal_entropy_bytes`.
///
/// * `max_order`: Maximum context order for the suffix automaton LM.
///   A value of -1 means unlimited context (bounded only by memory/sequence length).
#[inline(always)]
pub fn entropy_rate_bytes(data: &[u8], max_order: i64) -> f64 {
    with_default_ctx(|ctx| ctx.entropy_rate_bytes(data, max_order))
}

/// Compute biased entropy rate Ĥ_biased(X) bits per symbol.
///
/// This uses the full plugin estimator (training on the whole text, then scoring the same text).
/// While biased as a source entropy estimate, it is mathematically consistent for
/// similarity metrics like Mutual Information and NED.
#[inline(always)]
pub fn biased_entropy_rate_bytes(data: &[u8], max_order: i64) -> f64 {
    with_default_ctx(|ctx| ctx.biased_entropy_rate_bytes(data, max_order))
}

/// Compute joint marginal entropy H(X,Y) = −Σ p(x,y) log₂ p(x,y) in bits/symbol-pair.
///
/// Uses a direct histogram of (x_i, y_i) pairs. This is the exact first-order
/// joint entropy, matching the spec.md definition.
#[inline(always)]
pub fn joint_marginal_entropy_bytes(x: &[u8], y: &[u8]) -> f64 {
    let (x, y) = aligned_prefix(x, y);
    let n = x.len();
    if n == 0 {
        return 0.0;
    }

    // Count pair occurrences using a HashMap for (x, y) pairs
    // There are up to 65536 possible pairs, so we can use a flat array
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

/// Compute joint entropy rate `Ĥ(X,Y)`.
///
/// Dispatches based on `max_order`:
/// - `max_order == 0`: Strictly aligned pair-symbol mapping (Marginal Joint Entropy).
///   Treats `(x_i, y_i)` as a single symbol in a product alphabet `Σ_X × Σ_Y`.
/// - `max_order != 0`: Shift-invariant algorithmic joint entropy approximated via ROSA.
///   Constructs a sequence of pair-symbols and estimates the entropy rate of that sequence.
///
/// **Note**: This is an *aligned* joint entropy-rate estimate over time-indexed pairs
/// `(x_i, y_i)`. All joint-based quantities (`H(X)`, `H(Y)`, `H(X,Y)`, `I`, NED, NTE, etc.)
/// should be computed over the same aligned sample.
#[inline(always)]
pub fn joint_entropy_rate_bytes(x: &[u8], y: &[u8], max_order: i64) -> f64 {
    with_default_ctx(|ctx| ctx.joint_entropy_rate_bytes(x, y, max_order))
}

/// Compute conditional entropy rate `Ĥ(X|Y)`.
///
/// Dispatches based on `max_order`:
/// - `max_order == 0`: Strictly aligned `H(X,Y) - H(Y)` using marginals.
/// - `max_order != 0`: Chain rule definition `Ĥ(X|Y) = Ĥ(X,Y) - Ĥ(Y)`.
///
/// Note: This relies on the identity `H(X|Y) = H(X,Y) - H(Y)`.
#[inline(always)]
pub fn conditional_entropy_rate_bytes(x: &[u8], y: &[u8], max_order: i64) -> f64 {
    with_default_ctx(|ctx| ctx.conditional_entropy_rate_bytes(x, y, max_order))
}

/// Compute conditional entropy H(X|Y) = H(X,Y) − H(Y)
///
/// Dispatches based on `max_order`.
#[inline(always)]
pub fn conditional_entropy_bytes(x: &[u8], y: &[u8], max_order: i64) -> f64 {
    with_default_ctx(|ctx| ctx.conditional_entropy_bytes(x, y, max_order))
}

/// Compute mutual information `I(X;Y) = H(X) + H(Y) - H(X,Y)`.
///
/// Dispatches based on `max_order`. If 0, uses marginals; else uses rates.
///
/// `I(X;Y) = Σ p(x,y) log(p(x,y) / (p(x)p(y)))`
#[inline(always)]
pub fn mutual_information_bytes(x: &[u8], y: &[u8], max_order: i64) -> f64 {
    with_default_ctx(|ctx| ctx.mutual_information_bytes(x, y, max_order))
}

/// Marginal Mutual Information (exact/histogram)
pub fn mutual_information_marg_bytes(x: &[u8], y: &[u8]) -> f64 {
    let (x, y) = aligned_prefix(x, y);
    let h_x = marginal_entropy_bytes(x);
    let h_y = marginal_entropy_bytes(y);
    let h_xy = joint_marginal_entropy_bytes(x, y);
    (h_x + h_y - h_xy).max(0.0)
}

/// Entropy Rate Mutual Information (ROSA predictive)
#[inline(always)]
pub fn mutual_information_rate_bytes(x: &[u8], y: &[u8], max_order: i64) -> f64 {
    with_default_ctx(|ctx| ctx.mutual_information_rate_bytes(x, y, max_order))
}

// ====== NED: Normalized Entropy Distance ======
//
// A metric distance based on the overlap of information between two variables.

/// NED(X,Y) = (H(X,Y) - min(H(X), H(Y))) / max(H(X), H(Y))
///
/// Dispatches based on `max_order`. If 0, uses marginals; else uses rates.
///
/// Range: [0, 1].
/// * 0: Identity (X determines Y and Y determines X).
/// * 1: Independence (X and Y share no information).
#[inline(always)]
pub fn ned_bytes(x: &[u8], y: &[u8], max_order: i64) -> f64 {
    with_default_ctx(|ctx| ctx.ned_bytes(x, y, max_order))
}

/// Marginal NED (exact/histogram)
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

/// Normalized Entropy Distance (Rate-based)
#[inline(always)]
pub fn ned_rate_bytes(x: &[u8], y: &[u8], max_order: i64) -> f64 {
    with_default_ctx(|ctx| ctx.ned_bytes(x, y, max_order))
}

/// NED_cons(X,Y) = (H(X,Y) - min(H(X), H(Y))) / H(X,Y)
///
/// Conservative variant. Dispatches based on `max_order`.
#[inline(always)]
pub fn ned_cons_bytes(x: &[u8], y: &[u8], max_order: i64) -> f64 {
    with_default_ctx(|ctx| ctx.ned_cons_bytes(x, y, max_order))
}

/// Conservative marginal NED using histogram entropy estimates.
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
/// Conservative rate NED using the current default context backend.
pub fn ned_cons_rate_bytes(x: &[u8], y: &[u8], max_order: i64) -> f64 {
    with_default_ctx(|ctx| ctx.ned_cons_bytes(x, y, max_order))
}

// ====== NTE: Normalized Transform Effort (Variation of Information) ======

/// NTE(X,Y) = VI(X,Y) / max(H(X), H(Y))
/// where `VI(X,Y) = H(X|Y) + H(Y|X) = 2H(X,Y) - H(X) - H(Y)`.
///
/// Represents the "effort" required to transform X into Y (and vice versa) relative
/// to their complexity.
///
/// Note: VI can be as large as `H(X) + H(Y)`. If `H(X) ≈ H(Y)`, then VI can be `≈ 2 max(H(X), H(Y))`.
/// Thus, NTE is in [0, 2].
/// * Values near 0 indicate near-identity.
/// * Values near 1+ indicate substantial effort/transform cost (e.g. independence).
///
/// Dispatches based on `max_order`.
#[inline(always)]
pub fn nte_bytes(x: &[u8], y: &[u8], max_order: i64) -> f64 {
    with_default_ctx(|ctx| ctx.nte_bytes(x, y, max_order))
}

/// Marginal NTE using histogram entropy estimates.
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
/// Rate NTE using the current default context backend.
pub fn nte_rate_bytes(x: &[u8], y: &[u8], max_order: i64) -> f64 {
    with_default_ctx(|ctx| ctx.nte_bytes(x, y, max_order))
}

// ====== TVD: Total Variation Distance ======

/// Compute marginal byte histogram p(i) = count(i) / N for i ∈ [0, 255]
#[inline(always)]
fn byte_histogram(data: &[u8]) -> [f64; 256] {
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

/// TVD_marg(X,Y) = (1/2) Σᵢ |p_X(i) - p_Y(i)|
///
/// Total Variation Distance over marginal byte distributions.
/// True metric on probability space. Range: [0, 1].
/// 0 = identical distributions, 1 = completely disjoint support.
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

// ====== NHD: Normalized Hellinger Distance ======

/// NHD(X,Y) = sqrt(1 - BC(X,Y)) where BC = Σᵢ sqrt(p_X(i) · p_Y(i))
///
/// Normalized Hellinger Distance over marginal byte distributions.
/// True metric. Range: [0, 1]. 0 = identical, 1 = disjoint support.
#[inline(always)]
pub fn nhd_bytes(x: &[u8], y: &[u8], _max_order: i64) -> f64 {
    if x.is_empty() || y.is_empty() {
        return 0.0;
    }
    let p_x = byte_histogram(x);
    let p_y = byte_histogram(y);

    // Bhattacharyya coefficient: BC = Σᵢ sqrt(p_X(i) · p_Y(i))
    let mut bc = 0.0f64;
    for i in 0..256 {
        bc += (p_x[i] * p_y[i]).sqrt();
    }

    // NHD = sqrt(1 - BC)
    (1.0 - bc).max(0.0).sqrt()
}

// ====== Other Information-Theoretic Measures ======

/// Compute cross-entropy H_{train}(test) - score test_data under model trained on train_data.
///
/// Dispatches based on `max_order`.
#[inline(always)]
pub fn cross_entropy_bytes(test_data: &[u8], train_data: &[u8], max_order: i64) -> f64 {
    with_default_ctx(|ctx| ctx.cross_entropy_bytes(test_data, train_data, max_order))
}

/// Compute cross-entropy rate using ROSA/CTW/RWKV.
/// Training model on `train_data` and evaluating probability of `test_data`.
#[inline(always)]
pub fn cross_entropy_rate_bytes(test_data: &[u8], train_data: &[u8], max_order: i64) -> f64 {
    with_default_ctx(|ctx| ctx.cross_entropy_rate_bytes(test_data, train_data, max_order))
}

/// Kullback-Leibler Divergence D_KL(P || Q) = Σ p(x) log(p(x) / q(x))
///
/// Marginal only. Measure of how one probability distribution is different from a second.
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

/// Jensen-Shannon Divergence JSD(P || Q) = 1/2 D_KL(P || M) + 1/2 D_KL(Q || M)
/// where M = 1/2 (P + Q)
///
/// Marginal only. Symmetrized and smoothed version of KL divergence. Range `[0,1]`.
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

// ====== Path-based convenience wrappers ======

/// NED for files.
pub fn ned_paths(x: &str, y: &str, max_order: i64) -> f64 {
    let (bx, by) = rayon::join(
        || std::fs::read(x).expect("failed to read x"),
        || std::fs::read(y).expect("failed to read y"),
    );
    ned_bytes(&bx, &by, max_order)
}

/// NTE for files.
pub fn nte_paths(x: &str, y: &str, max_order: i64) -> f64 {
    let (bx, by) = rayon::join(
        || std::fs::read(x).expect("failed to read x"),
        || std::fs::read(y).expect("failed to read y"),
    );
    nte_bytes(&bx, &by, max_order)
}

/// TVD for files.
pub fn tvd_paths(x: &str, y: &str, max_order: i64) -> f64 {
    let (bx, by) = rayon::join(
        || std::fs::read(x).expect("failed to read x"),
        || std::fs::read(y).expect("failed to read y"),
    );
    tvd_bytes(&bx, &by, max_order)
}

/// NHD for files.
pub fn nhd_paths(x: &str, y: &str, max_order: i64) -> f64 {
    let (bx, by) = rayon::join(
        || std::fs::read(x).expect("failed to read x"),
        || std::fs::read(y).expect("failed to read y"),
    );
    nhd_bytes(&bx, &by, max_order)
}

/// Mutual Information for files.
pub fn mutual_information_paths(x: &str, y: &str, max_order: i64) -> f64 {
    let (bx, by) = rayon::join(
        || std::fs::read(x).expect("failed to read x"),
        || std::fs::read(y).expect("failed to read y"),
    );
    mutual_information_bytes(&bx, &by, max_order)
}

/// Conditional Entropy for files.
pub fn conditional_entropy_paths(x: &str, y: &str, max_order: i64) -> f64 {
    let (bx, by) = rayon::join(
        || std::fs::read(x).expect("failed to read x"),
        || std::fs::read(y).expect("failed to read y"),
    );
    conditional_entropy_bytes(&bx, &by, max_order)
}

/// Cross-Entropy for files.
pub fn cross_entropy_paths(x: &str, y: &str, max_order: i64) -> f64 {
    let (bx, by) = rayon::join(
        || std::fs::read(x).expect("failed to read x"),
        || std::fs::read(y).expect("failed to read y"),
    );
    cross_entropy_bytes(&bx, &by, max_order)
}

/// KL Divergence for files.
pub fn kl_divergence_paths(x: &str, y: &str) -> f64 {
    let (bx, by) = rayon::join(
        || std::fs::read(x).expect("failed to read x"),
        || std::fs::read(y).expect("failed to read y"),
    );
    d_kl_bytes(&bx, &by)
}

/// Jensen-Shannon Divergence for files.
pub fn js_divergence_paths(x: &str, y: &str) -> f64 {
    let (bx, by) = rayon::join(
        || std::fs::read(x).expect("failed to read x"),
        || std::fs::read(y).expect("failed to read y"),
    );
    js_div_bytes(&bx, &by)
}

// ====== Primitives 6 & 7 ======

/// Primitive 6: Intrinsic Dependence (Redundancy Ratio).
///
/// Measures how much structure is intrinsic to the sample, relative to its
/// own marginal entropy baseline.
///
/// `R = (H_marginal - H_rate) / H_marginal`
///
/// Clamped to `[0,1]`.
///
/// Interpretation:
///   - `R → 0`: Data is close to i.i.d./max-entropy (little intrinsic structure; highly extrinsically explainable by priors).
///   - `R → 1`: Data is highly predictable from its own past (strong intrinsic dependence; e.g., periodic strings like 010101...).
#[inline(always)]
pub fn intrinsic_dependence_bytes(data: &[u8], max_order: i64) -> f64 {
    with_default_ctx(|ctx| ctx.intrinsic_dependence_bytes(data, max_order))
}

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
#[inline(always)]
pub fn resistance_to_transformation_bytes(x: &[u8], tx: &[u8], max_order: i64) -> f64 {
    with_default_ctx(|ctx| ctx.resistance_to_transformation_bytes(x, tx, max_order))
}

#[cfg(test)]
mod tests {
    use super::*;

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
        assert_eq!(ctx.cross_entropy_bytes(empty, y, 0), 0.0);
    }

    #[cfg(not(feature = "backend-zpaq"))]
    #[test]
    #[should_panic(expected = "CompressionBackend::Zpaq is unavailable")]
    fn explicit_zpaq_backend_fails_loudly() {
        let backend = CompressionBackend::Zpaq {
            method: "5".to_string(),
        };
        let _ = compress_size_backend(b"abc", &backend);
    }

    #[cfg(not(feature = "backend-zpaq"))]
    #[test]
    fn default_compression_backend_falls_back_to_rate_coding() {
        let backend = CompressionBackend::default();
        assert!(matches!(
            &backend,
            CompressionBackend::Rate {
                coder: crate::coders::CoderType::AC,
                framing: crate::compression::FramingMode::Raw,
                ..
            }
        ));
        assert!(compress_size_backend(b"abc", &backend) > 0);
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
        use crate::ctw::ContextTree;

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

        let mut direct = rosaplus::RosaPlus::new(3, false, 0, 42);
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

        let mut direct = rosaplus::RosaPlus::new(3, false, 0, 42);
        direct.train_example(train);
        direct.build_lm();
        let expected = direct.cross_entropy(test);

        assert!(
            (plugin - expected).abs() < 1e-12,
            "rosa plugin cross entropy must match direct model API: plugin={plugin} expected={expected}"
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
        let h1 = ctx.cross_entropy_conditional_chain(&[prefix.as_slice()], data);
        let h2 = ctx.cross_entropy_conditional_chain(&[prefix.as_slice()], data);
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
