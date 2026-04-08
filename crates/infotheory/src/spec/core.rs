//! Canonical validated and compiled backend plans.
//!
//! This module is the internal/public bridge between wrapper AST specs
//! (`RateBackend`, `CompressionBackend`) and the compiled runtime plans used by
//! the generic Rust API.

use crate::api::{
    CalibratedSpec, CalibrationContextKind, CompressionBackend, MAX_MIXTURE_NESTING,
    MixtureExpertSpec, MixtureKind, MixtureScheduleMode, MixtureSpec, ParticleSpec, RateBackend,
};
use crate::coders::CoderType;
use crate::compression::FramingMode;
use crate::spec::{SpecError, SpecResult};
use std::fmt;
use std::path::{Path, PathBuf};
use std::sync::Arc;

/// Compilation environment for backend/spec validation and canonicalization.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct SpecEnvironment {
    base_dir: PathBuf,
}

impl SpecEnvironment {
    /// Create an environment rooted at `base_dir`.
    pub fn new(base_dir: impl Into<PathBuf>) -> Self {
        Self {
            base_dir: base_dir.into(),
        }
    }

    /// Base directory used to resolve relative asset references.
    pub fn base_dir(&self) -> &Path {
        &self.base_dir
    }
}

/// Typed external asset reference captured by a validated/compiled spec.
#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub enum AssetRef {
    /// Filesystem-backed asset resolved relative to a [`SpecEnvironment`].
    Filesystem(PathBuf),
}

/// Deterministic binary canonical code for a validated spec.
#[derive(Clone, Eq, PartialEq, Hash)]
pub struct CanonicalBytes(Arc<[u8]>);

impl CanonicalBytes {
    /// View the canonical bytes.
    pub fn as_slice(&self) -> &[u8] {
        &self.0
    }

    /// Length of the canonical code in bytes.
    pub fn len(&self) -> usize {
        self.0.len()
    }

    /// Whether the canonical code is empty.
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
}

impl fmt::Debug for CanonicalBytes {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("CanonicalBytes")
            .field(&self.0.len())
            .finish()
    }
}

impl From<Vec<u8>> for CanonicalBytes {
    fn from(value: Vec<u8>) -> Self {
        Self(Arc::<[u8]>::from(value))
    }
}

/// Shared execution family for method-backed neural backends.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum MethodBackendFamily {
    /// Mamba family.
    Mamba,
    /// RWKV-7 family.
    Rwkv7,
}

/// Trace-model execution strategy used by VM/AIXI adapters.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum RateBackendTraceStrategy {
    /// Direct ROSA-specific strategy.
    Rosa,
    /// Direct CTW strategy.
    Ctw,
    /// Direct FAC-CTW strategy.
    FacCtw,
    /// Generic predictor-backed strategy.
    PredictorBacked,
    /// ZPAQ-specific rate model strategy.
    Zpaq,
    /// Mamba compressor-backed strategy.
    Mamba,
    /// RWKV compressor-backed strategy.
    Rwkv7,
}

/// Shared capability metadata for rate backends.
#[derive(Clone, Debug)]
pub struct RateBackendCapabilities {
    /// Canonical backend family name.
    pub canonical_name: &'static str,
    /// Human-readable backend label.
    pub display_label: Arc<str>,
    /// VM/AIXI trace strategy for this backend family.
    pub trace_strategy: RateBackendTraceStrategy,
    /// Whether biased/plugin entropy is supported.
    pub supports_biased_entropy: bool,
    /// Whether generic frozen-conditioning is supported.
    pub supports_frozen_conditioning: bool,
    /// Whether generic rate-coded compression wrapping is supported.
    pub supports_rate_coded_compression: bool,
    /// Whether this backend can be adapted losslessly to bit-token mode.
    pub supports_bit_token_adaptation: bool,
    /// Whether the backend graph contains any ZPAQ component.
    pub contains_zpaq: bool,
    /// Whether this is a method-backed neural family.
    pub method_family: Option<MethodBackendFamily>,
}

/// Shared capability metadata for compression backends.
#[derive(Clone, Debug)]
pub struct CompressionBackendCapabilities {
    /// Canonical backend family name.
    pub canonical_name: &'static str,
    /// Human-readable backend label.
    pub display_label: Arc<str>,
    /// Whether this backend wraps a predictive rate backend.
    pub uses_rate_backend: bool,
    /// Whether this backend supports decompression.
    pub supports_decompression: bool,
}

#[derive(Clone, Debug)]
pub(crate) struct RateBackendPlanExpert {
    pub name: Option<String>,
    pub log_prior: f64,
    pub max_order: i64,
    pub backend: Arc<RateBackendPlan>,
}

#[derive(Clone, Debug)]
pub(crate) enum RateBackendPlan {
    RosaPlus,
    Match {
        hash_bits: usize,
        min_len: usize,
        max_len: usize,
        base_mix: f64,
        confidence_scale: f64,
    },
    SparseMatch {
        hash_bits: usize,
        min_len: usize,
        max_len: usize,
        gap_min: usize,
        gap_max: usize,
        base_mix: f64,
        confidence_scale: f64,
    },
    Ppmd {
        order: usize,
        memory_mb: usize,
    },
    Sequitur {
        context_bytes: usize,
    },
    Ctw {
        depth: usize,
    },
    FacCtw {
        base_depth: usize,
        num_percept_bits: usize,
        encoding_bits: usize,
    },
    Zpaq {
        method: String,
    },
    #[cfg(feature = "backend-mamba")]
    Mamba {
        method: String,
        parsed_method: crate::mambazip::MethodSpec,
        asset: Option<AssetRef>,
    },
    #[cfg(feature = "backend-rwkv")]
    Rwkv7 {
        method: String,
        parsed_method: crate::rwkvzip::MethodSpec,
        asset: Option<AssetRef>,
    },
    Mixture {
        kind: MixtureKind,
        schedule: MixtureScheduleMode,
        alpha: f64,
        decay: Option<f64>,
        experts: Box<[RateBackendPlanExpert]>,
    },
    Particle {
        spec: ParticleSpec,
    },
    Calibrated {
        context: CalibrationContextKind,
        bins: usize,
        learning_rate: f64,
        bias_clip: f64,
        base: Arc<RateBackendPlan>,
    },
}

impl RateBackendPlan {
    pub(crate) fn kind(&self) -> crate::runtime::RateBackendKind {
        match self {
            RateBackendPlan::RosaPlus => crate::runtime::RateBackendKind::RosaPlus,
            RateBackendPlan::Match { .. } => crate::runtime::RateBackendKind::Match,
            RateBackendPlan::SparseMatch { .. } => crate::runtime::RateBackendKind::SparseMatch,
            RateBackendPlan::Ppmd { .. } => crate::runtime::RateBackendKind::Ppmd,
            RateBackendPlan::Sequitur { .. } => crate::runtime::RateBackendKind::Sequitur,
            RateBackendPlan::Ctw { .. } => crate::runtime::RateBackendKind::Ctw,
            RateBackendPlan::FacCtw { .. } => crate::runtime::RateBackendKind::FacCtw,
            RateBackendPlan::Zpaq { .. } => crate::runtime::RateBackendKind::Zpaq,
            #[cfg(feature = "backend-mamba")]
            RateBackendPlan::Mamba { .. } => crate::runtime::RateBackendKind::Mamba,
            #[cfg(feature = "backend-rwkv")]
            RateBackendPlan::Rwkv7 { .. } => crate::runtime::RateBackendKind::Rwkv7,
            RateBackendPlan::Mixture { .. } => crate::runtime::RateBackendKind::Mixture,
            RateBackendPlan::Particle { .. } => crate::runtime::RateBackendKind::Particle,
            RateBackendPlan::Calibrated { .. } => crate::runtime::RateBackendKind::Calibrated,
        }
    }
}

#[derive(Clone, Debug)]
pub(crate) enum CompressionBackendPlan {
    Zpaq {
        method: String,
    },
    #[cfg(feature = "backend-rwkv")]
    Rwkv7 {
        method: String,
        parsed_method: crate::rwkvzip::MethodSpec,
        asset: Option<AssetRef>,
        coder: CoderType,
    },
    Rate {
        rate_backend: Arc<RateBackendPlan>,
        coder: CoderType,
        framing: FramingMode,
    },
}

impl CompressionBackendPlan {
    pub(crate) fn kind(&self) -> crate::runtime::CompressionBackendKind {
        match self {
            CompressionBackendPlan::Zpaq { .. } => crate::runtime::CompressionBackendKind::Zpaq,
            #[cfg(feature = "backend-rwkv")]
            CompressionBackendPlan::Rwkv7 { .. } => crate::runtime::CompressionBackendKind::Rwkv7,
            CompressionBackendPlan::Rate {
                coder: CoderType::AC,
                ..
            } => crate::runtime::CompressionBackendKind::RateAc,
            CompressionBackendPlan::Rate {
                coder: CoderType::RANS,
                ..
            } => crate::runtime::CompressionBackendKind::RateRans,
        }
    }
}

/// Canonicalized and feature-validated rate backend wrapper.
#[derive(Clone)]
pub struct ValidatedRateBackend {
    canonical_spec: Arc<RateBackend>,
    canonical_bytes: CanonicalBytes,
    capabilities: RateBackendCapabilities,
    plan: Arc<RateBackendPlan>,
}

impl ValidatedRateBackend {
    /// Canonical wrapper AST for this backend.
    pub fn canonical_spec(&self) -> &RateBackend {
        self.canonical_spec.as_ref()
    }

    /// Deterministic binary canonical code for this backend.
    pub fn canonical_bytes(&self) -> &CanonicalBytes {
        &self.canonical_bytes
    }

    /// Capability metadata derived from the canonicalized backend graph.
    pub fn capabilities(&self) -> &RateBackendCapabilities {
        &self.capabilities
    }

    /// Human-readable backend label derived from the compiled plan.
    pub fn display_label(&self, max_order: i64) -> String {
        rate_backend_plan_display_label(self.plan.as_ref(), max_order)
    }

    /// Short default backend name for logs, diagnostics, and model labels.
    pub fn default_name(&self, max_order: i64) -> String {
        rate_backend_plan_default_name(self.plan.as_ref(), max_order)
    }

    /// Compile the validated spec into an immutable runtime plan.
    pub fn compile(&self) -> SpecResult<CompiledRateBackend> {
        Ok(CompiledRateBackend {
            canonical_spec: self.canonical_spec.clone(),
            canonical_bytes: self.canonical_bytes.clone(),
            capabilities: self.capabilities.clone(),
            plan: self.plan.clone(),
        })
    }
}

/// Canonicalized and feature-validated compression backend wrapper.
#[derive(Clone)]
pub struct ValidatedCompressionBackend {
    canonical_spec: Arc<CompressionBackend>,
    canonical_bytes: CanonicalBytes,
    capabilities: CompressionBackendCapabilities,
    plan: Arc<CompressionBackendPlan>,
}

impl ValidatedCompressionBackend {
    /// Canonical wrapper AST for this backend.
    pub fn canonical_spec(&self) -> &CompressionBackend {
        self.canonical_spec.as_ref()
    }

    /// Deterministic binary canonical code for this backend.
    pub fn canonical_bytes(&self) -> &CanonicalBytes {
        &self.canonical_bytes
    }

    /// Capability metadata derived from the canonicalized backend graph.
    pub fn capabilities(&self) -> &CompressionBackendCapabilities {
        &self.capabilities
    }

    /// Compile the validated spec into an immutable runtime plan.
    pub fn compile(&self) -> SpecResult<CompiledCompressionBackend> {
        Ok(CompiledCompressionBackend {
            canonical_spec: self.canonical_spec.clone(),
            canonical_bytes: self.canonical_bytes.clone(),
            capabilities: self.capabilities.clone(),
            plan: self.plan.clone(),
        })
    }
}

/// Compiled immutable rate-backend runtime plan.
#[derive(Clone)]
pub struct CompiledRateBackend {
    canonical_spec: Arc<RateBackend>,
    canonical_bytes: CanonicalBytes,
    capabilities: RateBackendCapabilities,
    plan: Arc<RateBackendPlan>,
}

impl CompiledRateBackend {
    /// Canonical wrapper AST for this backend.
    pub fn canonical_spec(&self) -> &RateBackend {
        self.canonical_spec.as_ref()
    }

    /// Deterministic binary canonical code for this backend.
    pub fn canonical_bytes(&self) -> &CanonicalBytes {
        &self.canonical_bytes
    }

    /// Capability metadata derived from the compiled backend graph.
    pub fn capabilities(&self) -> &RateBackendCapabilities {
        &self.capabilities
    }

    /// Compile-friendly backend display label.
    pub fn display_label(&self, max_order: i64) -> String {
        rate_backend_plan_display_label(self.plan.as_ref(), max_order)
    }

    /// Short default backend name for logs, diagnostics, and model labels.
    pub fn default_name(&self, max_order: i64) -> String {
        rate_backend_plan_default_name(self.plan.as_ref(), max_order)
    }

    /// Canonical backend family name.
    pub fn canonical_name(&self) -> &'static str {
        self.capabilities.canonical_name
    }

    /// Whether the compiled backend graph contains any ZPAQ component.
    pub fn contains_zpaq(&self) -> bool {
        self.capabilities.contains_zpaq
    }

    /// Whether this backend supports generic frozen conditioning.
    pub fn supports_frozen_conditioning(&self) -> bool {
        self.capabilities.supports_frozen_conditioning
    }

    /// Whether this backend supports generic rate-coded compression wrapping.
    pub fn supports_rate_coded_compression(&self) -> bool {
        self.capabilities.supports_rate_coded_compression
    }

    /// Whether this backend can be adapted losslessly to bit-token mode.
    pub fn supports_bit_token_adaptation(&self) -> bool {
        self.capabilities.supports_bit_token_adaptation
    }

    /// Return a bit-token-adapted compiled backend when the transformation is defined.
    pub fn adapt_for_bit_tokens(&self) -> SpecResult<Self> {
        if !self.capabilities.supports_bit_token_adaptation {
            return Err(SpecError::new(format!(
                "backend '{}' cannot be adapted for bit-token mode",
                self.capabilities.canonical_name
            )));
        }
        compiled_rate_backend_from_plan(Arc::new(adapt_rate_plan_for_bit_tokens(
            self.plan.as_ref(),
        )))
    }

    pub(crate) fn plan(&self) -> &RateBackendPlan {
        self.plan.as_ref()
    }

    #[allow(dead_code)]
    pub(crate) fn method_string(&self) -> Option<&str> {
        match self.plan() {
            #[cfg(feature = "backend-mamba")]
            RateBackendPlan::Mamba { method, .. } => Some(method.as_str()),
            #[cfg(feature = "backend-rwkv")]
            RateBackendPlan::Rwkv7 { method, .. } => Some(method.as_str()),
            _ => None,
        }
    }
}

/// Compiled immutable compression-backend runtime plan.
#[derive(Clone)]
pub struct CompiledCompressionBackend {
    canonical_spec: Arc<CompressionBackend>,
    canonical_bytes: CanonicalBytes,
    capabilities: CompressionBackendCapabilities,
    plan: Arc<CompressionBackendPlan>,
}

impl CompiledCompressionBackend {
    /// Canonical wrapper AST for this backend.
    pub fn canonical_spec(&self) -> &CompressionBackend {
        self.canonical_spec.as_ref()
    }

    /// Deterministic binary canonical code for this backend.
    pub fn canonical_bytes(&self) -> &CanonicalBytes {
        &self.canonical_bytes
    }

    /// Capability metadata derived from the compiled backend graph.
    pub fn capabilities(&self) -> &CompressionBackendCapabilities {
        &self.capabilities
    }

    pub(crate) fn plan(&self) -> &CompressionBackendPlan {
        self.plan.as_ref()
    }
}

pub(crate) fn validate_rate_backend_in(
    backend: &RateBackend,
    env: &SpecEnvironment,
) -> SpecResult<ValidatedRateBackend> {
    let plan = Arc::new(build_rate_plan(backend, env, MAX_MIXTURE_NESTING)?);
    let canonical_spec = Arc::new(rate_plan_to_wrapper(plan.as_ref()));
    crate::api::validate_rate_backend(canonical_spec.as_ref())
        .map_err(|err| SpecError::new(err.to_string()))?;
    Ok(ValidatedRateBackend {
        canonical_spec,
        canonical_bytes: encode_rate_backend_plan(plan.as_ref()),
        capabilities: rate_backend_capabilities(plan.as_ref()),
        plan,
    })
}

pub(crate) fn validate_compression_backend_in(
    backend: &CompressionBackend,
    env: &SpecEnvironment,
) -> SpecResult<ValidatedCompressionBackend> {
    let plan = Arc::new(build_compression_plan(backend, env)?);
    let canonical_spec = Arc::new(compression_plan_to_wrapper(plan.as_ref()));
    crate::api::validate_compression_backend(canonical_spec.as_ref())
        .map_err(|err| SpecError::new(err.to_string()))?;
    Ok(ValidatedCompressionBackend {
        canonical_spec,
        canonical_bytes: encode_compression_backend_plan(plan.as_ref()),
        capabilities: compression_backend_capabilities(plan.as_ref()),
        plan,
    })
}

pub(crate) fn compiled_rate_backend_from_plan(
    plan: Arc<RateBackendPlan>,
) -> SpecResult<CompiledRateBackend> {
    Ok(compiled_rate_backend_from_plan_unchecked(plan))
}

pub(crate) fn compiled_compression_backend_from_plan(
    plan: Arc<CompressionBackendPlan>,
) -> SpecResult<CompiledCompressionBackend> {
    Ok(compiled_compression_backend_from_plan_unchecked(plan))
}

pub(crate) fn compiled_rate_backend_from_plan_unchecked(
    plan: Arc<RateBackendPlan>,
) -> CompiledRateBackend {
    let canonical_spec = Arc::new(rate_plan_to_wrapper(plan.as_ref()));
    CompiledRateBackend {
        canonical_bytes: encode_rate_backend_plan(plan.as_ref()),
        capabilities: rate_backend_capabilities(plan.as_ref()),
        canonical_spec,
        plan,
    }
}

pub(crate) fn compiled_compression_backend_from_plan_unchecked(
    plan: Arc<CompressionBackendPlan>,
) -> CompiledCompressionBackend {
    let canonical_spec = Arc::new(compression_plan_to_wrapper(plan.as_ref()));
    CompiledCompressionBackend {
        canonical_bytes: encode_compression_backend_plan(plan.as_ref()),
        capabilities: compression_backend_capabilities(plan.as_ref()),
        canonical_spec,
        plan,
    }
}

pub(crate) fn compile_rate_plan_rosa(
    backend: &RateBackend,
    _env: &SpecEnvironment,
    _depth: usize,
) -> SpecResult<RateBackendPlan> {
    match backend {
        RateBackend::RosaPlus => Ok(RateBackendPlan::RosaPlus),
        _ => unreachable!("rosa kernel used with non-rosa backend"),
    }
}

pub(crate) fn compile_rate_plan_match(
    backend: &RateBackend,
    _env: &SpecEnvironment,
    _depth: usize,
) -> SpecResult<RateBackendPlan> {
    match backend {
        RateBackend::Match {
            hash_bits,
            min_len,
            max_len,
            base_mix,
            confidence_scale,
        } => Ok(RateBackendPlan::Match {
            hash_bits: *hash_bits,
            min_len: *min_len,
            max_len: *max_len,
            base_mix: *base_mix,
            confidence_scale: *confidence_scale,
        }),
        _ => unreachable!("match kernel used with non-match backend"),
    }
}

pub(crate) fn compile_rate_plan_sparse_match(
    backend: &RateBackend,
    _env: &SpecEnvironment,
    _depth: usize,
) -> SpecResult<RateBackendPlan> {
    match backend {
        RateBackend::SparseMatch {
            hash_bits,
            min_len,
            max_len,
            gap_min,
            gap_max,
            base_mix,
            confidence_scale,
        } => Ok(RateBackendPlan::SparseMatch {
            hash_bits: *hash_bits,
            min_len: *min_len,
            max_len: *max_len,
            gap_min: *gap_min,
            gap_max: *gap_max,
            base_mix: *base_mix,
            confidence_scale: *confidence_scale,
        }),
        _ => unreachable!("sparse-match kernel used with non-sparse-match backend"),
    }
}

pub(crate) fn compile_rate_plan_ppmd(
    backend: &RateBackend,
    _env: &SpecEnvironment,
    _depth: usize,
) -> SpecResult<RateBackendPlan> {
    match backend {
        RateBackend::Ppmd { order, memory_mb } => Ok(RateBackendPlan::Ppmd {
            order: *order,
            memory_mb: *memory_mb,
        }),
        _ => unreachable!("ppmd kernel used with non-ppmd backend"),
    }
}

pub(crate) fn compile_rate_plan_sequitur(
    backend: &RateBackend,
    _env: &SpecEnvironment,
    _depth: usize,
) -> SpecResult<RateBackendPlan> {
    match backend {
        RateBackend::Sequitur { context_bytes } => {
            if *context_bytes < 2 {
                return Err(SpecError::new("sequitur context_bytes must be >= 2"));
            }
            Ok(RateBackendPlan::Sequitur {
                context_bytes: *context_bytes,
            })
        }
        _ => unreachable!("sequitur kernel used with non-sequitur backend"),
    }
}

pub(crate) fn compile_rate_plan_ctw(
    backend: &RateBackend,
    _env: &SpecEnvironment,
    _depth: usize,
) -> SpecResult<RateBackendPlan> {
    match backend {
        RateBackend::Ctw { depth } => Ok(RateBackendPlan::Ctw { depth: *depth }),
        _ => unreachable!("ctw kernel used with non-ctw backend"),
    }
}

pub(crate) fn compile_rate_plan_fac_ctw(
    backend: &RateBackend,
    _env: &SpecEnvironment,
    _depth: usize,
) -> SpecResult<RateBackendPlan> {
    match backend {
        RateBackend::FacCtw {
            base_depth,
            num_percept_bits,
            encoding_bits,
        } => Ok(RateBackendPlan::FacCtw {
            base_depth: *base_depth,
            num_percept_bits: *num_percept_bits,
            encoding_bits: *encoding_bits,
        }),
        _ => unreachable!("fac-ctw kernel used with non-fac-ctw backend"),
    }
}

pub(crate) fn compile_rate_plan_zpaq(
    backend: &RateBackend,
    _env: &SpecEnvironment,
    _depth: usize,
) -> SpecResult<RateBackendPlan> {
    match backend {
        RateBackend::Zpaq { method } => {
            crate::validate_zpaq_rate_method(method)
                .map_err(|err| SpecError::new(err.to_string()))?;
            Ok(RateBackendPlan::Zpaq {
                method: method.clone(),
            })
        }
        _ => unreachable!("zpaq kernel used with non-zpaq backend"),
    }
}

#[cfg(feature = "backend-mamba")]
pub(crate) fn compile_rate_plan_mamba(
    backend: &RateBackend,
    env: &SpecEnvironment,
    _depth: usize,
) -> SpecResult<RateBackendPlan> {
    match backend {
        RateBackend::MambaMethod { method } => {
            let normalized = super::normalize_mamba_method_for_base_dir(env.base_dir(), method)?;
            let parsed_method = crate::mambazip::parse_method_spec(&normalized)
                .map_err(|err| SpecError::new(err.to_string()))?;
            Ok(RateBackendPlan::Mamba {
                method: crate::mambazip::canonical_method_string(&parsed_method),
                asset: mamba_asset_ref(&parsed_method),
                parsed_method,
            })
        }
        _ => unreachable!("mamba kernel used with non-mamba backend"),
    }
}

#[cfg(not(feature = "backend-mamba"))]
pub(crate) fn compile_rate_plan_mamba(
    _backend: &RateBackend,
    _env: &SpecEnvironment,
    _depth: usize,
) -> SpecResult<RateBackendPlan> {
    unreachable!("mamba kernel should never compile without backend-mamba")
}

#[cfg(feature = "backend-rwkv")]
pub(crate) fn compile_rate_plan_rwkv7(
    backend: &RateBackend,
    env: &SpecEnvironment,
    _depth: usize,
) -> SpecResult<RateBackendPlan> {
    match backend {
        RateBackend::Rwkv7Method { method } => {
            let normalized = super::normalize_rwkv_method_for_base_dir(env.base_dir(), method)?;
            let parsed_method = crate::rwkvzip::parse_method_spec(&normalized)
                .map_err(|err| SpecError::new(err.to_string()))?;
            Ok(RateBackendPlan::Rwkv7 {
                method: crate::rwkvzip::canonical_method_string(&parsed_method),
                asset: rwkv_asset_ref(&parsed_method),
                parsed_method,
            })
        }
        _ => unreachable!("rwkv7 kernel used with non-rwkv7 backend"),
    }
}

#[cfg(not(feature = "backend-rwkv"))]
pub(crate) fn compile_rate_plan_rwkv7(
    _backend: &RateBackend,
    _env: &SpecEnvironment,
    _depth: usize,
) -> SpecResult<RateBackendPlan> {
    unreachable!("rwkv7 kernel should never compile without backend-rwkv")
}

pub(crate) fn compile_rate_plan_mixture(
    backend: &RateBackend,
    env: &SpecEnvironment,
    depth: usize,
) -> SpecResult<RateBackendPlan> {
    let RateBackend::Mixture { spec } = backend else {
        unreachable!("mixture kernel used with non-mixture backend");
    };
    let experts: Vec<_> = spec
        .experts
        .iter()
        .map(|expert| {
            Ok(RateBackendPlanExpert {
                name: expert.name.clone(),
                log_prior: expert.log_prior,
                max_order: expert.max_order,
                backend: Arc::new(build_rate_plan(&expert.backend, env, depth - 1)?),
            })
        })
        .collect::<SpecResult<_>>()?;
    let canonical = MixtureSpec {
        kind: spec.kind,
        schedule: spec.schedule,
        alpha: spec.alpha,
        decay: spec.decay,
        experts: experts
            .iter()
            .map(|expert| MixtureExpertSpec {
                name: expert.name.clone(),
                log_prior: expert.log_prior,
                max_order: expert.max_order,
                backend: rate_plan_to_wrapper(expert.backend.as_ref()),
            })
            .collect(),
    };
    canonical
        .validate()
        .map_err(|err| SpecError::new(err.to_string()))?;
    Ok(RateBackendPlan::Mixture {
        kind: canonical.kind,
        schedule: canonical.schedule,
        alpha: canonical.alpha,
        decay: canonical.decay,
        experts: experts.into_boxed_slice(),
    })
}

pub(crate) fn compile_rate_plan_particle(
    backend: &RateBackend,
    _env: &SpecEnvironment,
    _depth: usize,
) -> SpecResult<RateBackendPlan> {
    let RateBackend::Particle { spec } = backend else {
        unreachable!("particle kernel used with non-particle backend");
    };
    spec.validate()
        .map_err(|err| SpecError::new(err.to_string()))?;
    Ok(RateBackendPlan::Particle {
        spec: spec.as_ref().clone(),
    })
}

pub(crate) fn compile_rate_plan_calibrated(
    backend: &RateBackend,
    env: &SpecEnvironment,
    depth: usize,
) -> SpecResult<RateBackendPlan> {
    let RateBackend::Calibrated { spec } = backend else {
        unreachable!("calibrated kernel used with non-calibrated backend");
    };
    Ok(RateBackendPlan::Calibrated {
        context: spec.context,
        bins: spec.bins,
        learning_rate: spec.learning_rate,
        bias_clip: spec.bias_clip,
        base: Arc::new(build_rate_plan(&spec.base, env, depth - 1)?),
    })
}

pub(crate) fn compile_compression_plan_zpaq(
    backend: &CompressionBackend,
    _env: &SpecEnvironment,
) -> SpecResult<CompressionBackendPlan> {
    match backend {
        CompressionBackend::Zpaq { method } => Ok(CompressionBackendPlan::Zpaq {
            method: method.clone(),
        }),
        _ => unreachable!("zpaq compression kernel used with non-zpaq backend"),
    }
}

#[cfg(feature = "backend-rwkv")]
pub(crate) fn compile_compression_plan_rwkv7(
    backend: &CompressionBackend,
    env: &SpecEnvironment,
) -> SpecResult<CompressionBackendPlan> {
    match backend {
        CompressionBackend::Rwkv7 { method, coder } => {
            let normalized = super::normalize_rwkv_method_for_base_dir(env.base_dir(), method)?;
            let parsed_method = crate::rwkvzip::parse_method_spec(&normalized)
                .map_err(|err| SpecError::new(err.to_string()))?;
            Ok(CompressionBackendPlan::Rwkv7 {
                method: crate::rwkvzip::canonical_method_string(&parsed_method),
                asset: rwkv_asset_ref(&parsed_method),
                parsed_method,
                coder: *coder,
            })
        }
        _ => unreachable!("rwkv7 compression kernel used with non-rwkv7 backend"),
    }
}

#[cfg(not(feature = "backend-rwkv"))]
pub(crate) fn compile_compression_plan_rwkv7(
    _backend: &CompressionBackend,
    _env: &SpecEnvironment,
) -> SpecResult<CompressionBackendPlan> {
    unreachable!("rwkv7 compression kernel should never compile without backend-rwkv")
}

pub(crate) fn compile_compression_plan_rate(
    backend: &CompressionBackend,
    env: &SpecEnvironment,
) -> SpecResult<CompressionBackendPlan> {
    match backend {
        CompressionBackend::Rate {
            rate_backend,
            coder,
            framing,
        } => Ok(CompressionBackendPlan::Rate {
            rate_backend: Arc::new(build_rate_plan(rate_backend, env, MAX_MIXTURE_NESTING)?),
            coder: *coder,
            framing: *framing,
        }),
        _ => unreachable!("rate compression kernel used with non-rate backend"),
    }
}

fn build_rate_plan(
    backend: &RateBackend,
    env: &SpecEnvironment,
    depth: usize,
) -> SpecResult<RateBackendPlan> {
    if depth == 0 {
        return Err(SpecError::new("backend spec nesting too deep"));
    }

    let descriptor = backend
        .descriptor()
        .map_err(|err| SpecError::new(format!("{err} (while validating rate backend)")))?;
    if !descriptor.enabled {
        let Some(feature) = descriptor.feature else {
            return Err(SpecError::new(format!(
                "internal backend registry mismatch: disabled backend '{}' is missing required feature metadata",
                descriptor.canonical
            )));
        };
        return Err(SpecError::new(format!(
            "backend '{}' requires infotheory feature '{}'",
            descriptor.canonical, feature
        )));
    }

    crate::runtime::compile_rate_backend_plan_via_kernel(descriptor.kind, backend, env, depth)
}

fn build_compression_plan(
    backend: &CompressionBackend,
    env: &SpecEnvironment,
) -> SpecResult<CompressionBackendPlan> {
    let descriptor = backend
        .descriptor()
        .map_err(|err| SpecError::new(format!("{err} (while validating compression backend)")))?;
    if !descriptor.enabled {
        let Some(feature) = descriptor.feature else {
            return Err(SpecError::new(format!(
                "internal backend registry mismatch: disabled compression backend '{}' is missing required feature metadata",
                descriptor.canonical
            )));
        };
        return Err(SpecError::new(format!(
            "compression backend '{}' requires infotheory feature '{}'",
            descriptor.canonical, feature
        )));
    }

    crate::runtime::compile_compression_backend_plan_via_kernel(descriptor.kind, backend, env)
}

pub(crate) fn rate_plan_to_wrapper_rosa(plan: &RateBackendPlan) -> RateBackend {
    let RateBackendPlan::RosaPlus = plan else {
        unreachable!("rosa wrapper kernel used with non-rosa plan");
    };
    RateBackend::RosaPlus
}

pub(crate) fn rate_plan_to_wrapper_match(plan: &RateBackendPlan) -> RateBackend {
    let RateBackendPlan::Match {
        hash_bits,
        min_len,
        max_len,
        base_mix,
        confidence_scale,
    } = plan
    else {
        unreachable!("match wrapper kernel used with non-match plan");
    };
    RateBackend::Match {
        hash_bits: *hash_bits,
        min_len: *min_len,
        max_len: *max_len,
        base_mix: *base_mix,
        confidence_scale: *confidence_scale,
    }
}

pub(crate) fn rate_plan_to_wrapper_sparse_match(plan: &RateBackendPlan) -> RateBackend {
    let RateBackendPlan::SparseMatch {
        hash_bits,
        min_len,
        max_len,
        gap_min,
        gap_max,
        base_mix,
        confidence_scale,
    } = plan
    else {
        unreachable!("sparse-match wrapper kernel used with non-sparse-match plan");
    };
    RateBackend::SparseMatch {
        hash_bits: *hash_bits,
        min_len: *min_len,
        max_len: *max_len,
        gap_min: *gap_min,
        gap_max: *gap_max,
        base_mix: *base_mix,
        confidence_scale: *confidence_scale,
    }
}

pub(crate) fn rate_plan_to_wrapper_ppmd(plan: &RateBackendPlan) -> RateBackend {
    let RateBackendPlan::Ppmd { order, memory_mb } = plan else {
        unreachable!("ppmd wrapper kernel used with non-ppmd plan");
    };
    RateBackend::Ppmd {
        order: *order,
        memory_mb: *memory_mb,
    }
}

pub(crate) fn rate_plan_to_wrapper_sequitur(plan: &RateBackendPlan) -> RateBackend {
    let RateBackendPlan::Sequitur { context_bytes } = plan else {
        unreachable!("sequitur wrapper kernel used with non-sequitur plan");
    };
    RateBackend::Sequitur {
        context_bytes: *context_bytes,
    }
}

pub(crate) fn rate_plan_to_wrapper_ctw(plan: &RateBackendPlan) -> RateBackend {
    let RateBackendPlan::Ctw { depth } = plan else {
        unreachable!("ctw wrapper kernel used with non-ctw plan");
    };
    RateBackend::Ctw { depth: *depth }
}

pub(crate) fn rate_plan_to_wrapper_fac_ctw(plan: &RateBackendPlan) -> RateBackend {
    let RateBackendPlan::FacCtw {
        base_depth,
        num_percept_bits,
        encoding_bits,
    } = plan
    else {
        unreachable!("fac-ctw wrapper kernel used with non-fac-ctw plan");
    };
    RateBackend::FacCtw {
        base_depth: *base_depth,
        num_percept_bits: *num_percept_bits,
        encoding_bits: *encoding_bits,
    }
}

pub(crate) fn rate_plan_to_wrapper_zpaq(plan: &RateBackendPlan) -> RateBackend {
    let RateBackendPlan::Zpaq { method } = plan else {
        unreachable!("zpaq wrapper kernel used with non-zpaq plan");
    };
    RateBackend::Zpaq {
        method: method.clone(),
    }
}

#[cfg(feature = "backend-mamba")]
pub(crate) fn rate_plan_to_wrapper_mamba(plan: &RateBackendPlan) -> RateBackend {
    let RateBackendPlan::Mamba { method, .. } = plan else {
        unreachable!("mamba wrapper kernel used with non-mamba plan");
    };
    RateBackend::MambaMethod {
        method: method.clone(),
    }
}

#[cfg(not(feature = "backend-mamba"))]
pub(crate) fn rate_plan_to_wrapper_mamba(_plan: &RateBackendPlan) -> RateBackend {
    unreachable!("mamba wrapper kernel should never be used without backend-mamba")
}

#[cfg(feature = "backend-rwkv")]
pub(crate) fn rate_plan_to_wrapper_rwkv7(plan: &RateBackendPlan) -> RateBackend {
    let RateBackendPlan::Rwkv7 { method, .. } = plan else {
        unreachable!("rwkv7 wrapper kernel used with non-rwkv7 plan");
    };
    RateBackend::Rwkv7Method {
        method: method.clone(),
    }
}

#[cfg(not(feature = "backend-rwkv"))]
pub(crate) fn rate_plan_to_wrapper_rwkv7(_plan: &RateBackendPlan) -> RateBackend {
    unreachable!("rwkv7 wrapper kernel should never be used without backend-rwkv")
}

pub(crate) fn rate_plan_to_wrapper_mixture(plan: &RateBackendPlan) -> RateBackend {
    let RateBackendPlan::Mixture {
        kind,
        schedule,
        alpha,
        decay,
        experts,
    } = plan
    else {
        unreachable!("mixture wrapper kernel used with non-mixture plan");
    };
    RateBackend::Mixture {
        spec: Arc::new(MixtureSpec {
            kind: *kind,
            schedule: *schedule,
            alpha: *alpha,
            decay: *decay,
            experts: experts
                .iter()
                .map(|expert| MixtureExpertSpec {
                    name: expert.name.clone(),
                    log_prior: expert.log_prior,
                    max_order: expert.max_order,
                    backend: rate_plan_to_wrapper(expert.backend.as_ref()),
                })
                .collect(),
        }),
    }
}

pub(crate) fn rate_plan_to_wrapper_particle(plan: &RateBackendPlan) -> RateBackend {
    let RateBackendPlan::Particle { spec } = plan else {
        unreachable!("particle wrapper kernel used with non-particle plan");
    };
    RateBackend::Particle {
        spec: Arc::new(spec.clone()),
    }
}

pub(crate) fn rate_plan_to_wrapper_calibrated(plan: &RateBackendPlan) -> RateBackend {
    let RateBackendPlan::Calibrated {
        context,
        bins,
        learning_rate,
        bias_clip,
        base,
    } = plan
    else {
        unreachable!("calibrated wrapper kernel used with non-calibrated plan");
    };
    RateBackend::Calibrated {
        spec: Arc::new(CalibratedSpec {
            base: rate_plan_to_wrapper(base.as_ref()),
            context: *context,
            bins: *bins,
            learning_rate: *learning_rate,
            bias_clip: *bias_clip,
        }),
    }
}

pub(crate) fn compression_plan_to_wrapper_zpaq(
    plan: &CompressionBackendPlan,
) -> CompressionBackend {
    let CompressionBackendPlan::Zpaq { method } = plan else {
        unreachable!("zpaq compression wrapper kernel used with non-zpaq plan");
    };
    CompressionBackend::Zpaq {
        method: method.clone(),
    }
}

#[cfg(feature = "backend-rwkv")]
pub(crate) fn compression_plan_to_wrapper_rwkv7(
    plan: &CompressionBackendPlan,
) -> CompressionBackend {
    let CompressionBackendPlan::Rwkv7 { method, coder, .. } = plan else {
        unreachable!("rwkv7 compression wrapper kernel used with non-rwkv7 plan");
    };
    CompressionBackend::Rwkv7 {
        method: method.clone(),
        coder: *coder,
    }
}

#[cfg(not(feature = "backend-rwkv"))]
pub(crate) fn compression_plan_to_wrapper_rwkv7(
    _plan: &CompressionBackendPlan,
) -> CompressionBackend {
    unreachable!("rwkv7 compression wrapper kernel should never be used without backend-rwkv")
}

pub(crate) fn compression_plan_to_wrapper_rate(
    plan: &CompressionBackendPlan,
) -> CompressionBackend {
    let CompressionBackendPlan::Rate {
        rate_backend,
        coder,
        framing,
    } = plan
    else {
        unreachable!("rate compression wrapper kernel used with non-rate plan");
    };
    CompressionBackend::Rate {
        rate_backend: rate_plan_to_wrapper(rate_backend.as_ref()),
        coder: *coder,
        framing: *framing,
    }
}

pub(crate) fn rate_plan_contains_zpaq_false(_plan: &RateBackendPlan) -> bool {
    false
}

pub(crate) fn rate_plan_contains_zpaq_true(_plan: &RateBackendPlan) -> bool {
    true
}

pub(crate) fn rate_plan_contains_zpaq_mixture(plan: &RateBackendPlan) -> bool {
    let RateBackendPlan::Mixture { experts, .. } = plan else {
        unreachable!("mixture zpaq kernel used with non-mixture plan");
    };
    experts
        .iter()
        .any(|expert| rate_plan_contains_zpaq(expert.backend.as_ref()))
}

pub(crate) fn rate_plan_contains_zpaq_calibrated(plan: &RateBackendPlan) -> bool {
    let RateBackendPlan::Calibrated { base, .. } = plan else {
        unreachable!("calibrated zpaq kernel used with non-calibrated plan");
    };
    rate_plan_contains_zpaq(base.as_ref())
}

pub(crate) fn rate_plan_supports_bit_token_adaptation_true(_plan: &RateBackendPlan) -> bool {
    true
}

pub(crate) fn rate_plan_supports_bit_token_adaptation_false(_plan: &RateBackendPlan) -> bool {
    false
}

pub(crate) fn rate_plan_supports_bit_token_adaptation_mixture(plan: &RateBackendPlan) -> bool {
    let RateBackendPlan::Mixture { experts, .. } = plan else {
        unreachable!("mixture bit-token kernel used with non-mixture plan");
    };
    experts
        .iter()
        .all(|expert| rate_plan_supports_bit_token_adaptation(expert.backend.as_ref()))
}

pub(crate) fn rate_plan_supports_bit_token_adaptation_calibrated(plan: &RateBackendPlan) -> bool {
    let RateBackendPlan::Calibrated { base, .. } = plan else {
        unreachable!("calibrated bit-token kernel used with non-calibrated plan");
    };
    rate_plan_supports_bit_token_adaptation(base.as_ref())
}

fn rate_plan_supports_bit_token_adaptation(plan: &RateBackendPlan) -> bool {
    (crate::runtime::rate_backend_kernel(plan.kind()).supports_bit_token_adaptation)(plan)
}

pub(crate) fn adapt_rate_plan_identity(plan: &RateBackendPlan) -> RateBackendPlan {
    plan.clone()
}

pub(crate) fn adapt_rate_plan_ctw(plan: &RateBackendPlan) -> RateBackendPlan {
    let RateBackendPlan::Ctw { depth } = plan else {
        unreachable!("ctw bit-token adapter used with non-ctw plan");
    };
    RateBackendPlan::FacCtw {
        base_depth: *depth,
        num_percept_bits: 1,
        encoding_bits: 1,
    }
}

pub(crate) fn adapt_rate_plan_fac_ctw(plan: &RateBackendPlan) -> RateBackendPlan {
    let RateBackendPlan::FacCtw { base_depth, .. } = plan else {
        unreachable!("fac-ctw bit-token adapter used with non-fac-ctw plan");
    };
    RateBackendPlan::FacCtw {
        base_depth: *base_depth,
        num_percept_bits: 1,
        encoding_bits: 1,
    }
}

pub(crate) fn adapt_rate_plan_mixture(plan: &RateBackendPlan) -> RateBackendPlan {
    let RateBackendPlan::Mixture {
        kind,
        schedule,
        alpha,
        decay,
        experts,
    } = plan
    else {
        unreachable!("mixture bit-token adapter used with non-mixture plan");
    };
    RateBackendPlan::Mixture {
        kind: *kind,
        schedule: *schedule,
        alpha: *alpha,
        decay: *decay,
        experts: experts
            .iter()
            .map(|expert| RateBackendPlanExpert {
                name: expert.name.clone(),
                log_prior: expert.log_prior,
                max_order: expert.max_order,
                backend: Arc::new(adapt_rate_plan_for_bit_tokens(expert.backend.as_ref())),
            })
            .collect::<Vec<_>>()
            .into_boxed_slice(),
    }
}

pub(crate) fn adapt_rate_plan_calibrated(plan: &RateBackendPlan) -> RateBackendPlan {
    let RateBackendPlan::Calibrated {
        context,
        bins,
        learning_rate,
        bias_clip,
        base,
    } = plan
    else {
        unreachable!("calibrated bit-token adapter used with non-calibrated plan");
    };
    RateBackendPlan::Calibrated {
        context: *context,
        bins: *bins,
        learning_rate: *learning_rate,
        bias_clip: *bias_clip,
        base: Arc::new(adapt_rate_plan_for_bit_tokens(base.as_ref())),
    }
}

pub(crate) fn rate_plan_display_label_rosa(plan: &RateBackendPlan, max_order: i64) -> String {
    let RateBackendPlan::RosaPlus = plan else {
        unreachable!("rosa label kernel used with non-rosa plan");
    };
    format!("rosaplus(max_order={max_order})")
}

pub(crate) fn rate_plan_default_name_rosa(plan: &RateBackendPlan, max_order: i64) -> String {
    let RateBackendPlan::RosaPlus = plan else {
        unreachable!("rosa default-name kernel used with non-rosa plan");
    };
    format!("rosa(mo={max_order})")
}

pub(crate) fn rate_plan_display_label_match(plan: &RateBackendPlan, _max_order: i64) -> String {
    let RateBackendPlan::Match {
        hash_bits,
        min_len,
        max_len,
        base_mix,
        confidence_scale,
    } = plan
    else {
        unreachable!("match label kernel used with non-match plan");
    };
    format!(
        "match(hash_bits={hash_bits},min_len={min_len},max_len={max_len},base_mix={base_mix},confidence_scale={confidence_scale})"
    )
}

pub(crate) fn rate_plan_default_name_match(plan: &RateBackendPlan, _max_order: i64) -> String {
    let RateBackendPlan::Match { .. } = plan else {
        unreachable!("match default-name kernel used with non-match plan");
    };
    "match".to_string()
}

pub(crate) fn rate_plan_display_label_sparse_match(
    plan: &RateBackendPlan,
    _max_order: i64,
) -> String {
    let RateBackendPlan::SparseMatch {
        hash_bits,
        min_len,
        max_len,
        gap_min,
        gap_max,
        base_mix,
        confidence_scale,
    } = plan
    else {
        unreachable!("sparse-match label kernel used with non-sparse-match plan");
    };
    format!(
        "sparse-match(hash_bits={hash_bits},min_len={min_len},max_len={max_len},gap_min={gap_min},gap_max={gap_max},base_mix={base_mix},confidence_scale={confidence_scale})"
    )
}

pub(crate) fn rate_plan_default_name_sparse_match(
    plan: &RateBackendPlan,
    _max_order: i64,
) -> String {
    let RateBackendPlan::SparseMatch { .. } = plan else {
        unreachable!("sparse-match default-name kernel used with non-sparse-match plan");
    };
    "sparse-match".to_string()
}

pub(crate) fn rate_plan_display_label_ppmd(plan: &RateBackendPlan, _max_order: i64) -> String {
    let RateBackendPlan::Ppmd { order, memory_mb } = plan else {
        unreachable!("ppmd label kernel used with non-ppmd plan");
    };
    format!("ppmd(order={order},memory_mb={memory_mb})")
}

pub(crate) fn rate_plan_default_name_ppmd(plan: &RateBackendPlan, _max_order: i64) -> String {
    let RateBackendPlan::Ppmd { order, memory_mb } = plan else {
        unreachable!("ppmd default-name kernel used with non-ppmd plan");
    };
    format!("ppmd(o={order},m={memory_mb}MiB)")
}

pub(crate) fn rate_plan_display_label_sequitur(plan: &RateBackendPlan, _max_order: i64) -> String {
    let RateBackendPlan::Sequitur { context_bytes } = plan else {
        unreachable!("sequitur label kernel used with non-sequitur plan");
    };
    format!("sequitur(context_bytes={context_bytes})")
}

pub(crate) fn rate_plan_default_name_sequitur(plan: &RateBackendPlan, _max_order: i64) -> String {
    let RateBackendPlan::Sequitur { context_bytes } = plan else {
        unreachable!("sequitur default-name kernel used with non-sequitur plan");
    };
    format!("sequitur(ctx={context_bytes})")
}

pub(crate) fn rate_plan_display_label_ctw(plan: &RateBackendPlan, _max_order: i64) -> String {
    let RateBackendPlan::Ctw { depth } = plan else {
        unreachable!("ctw label kernel used with non-ctw plan");
    };
    format!("ctw(depth={depth})")
}

pub(crate) fn rate_plan_default_name_ctw(plan: &RateBackendPlan, _max_order: i64) -> String {
    let RateBackendPlan::Ctw { depth } = plan else {
        unreachable!("ctw default-name kernel used with non-ctw plan");
    };
    format!("ctw(d={depth})")
}

pub(crate) fn rate_plan_display_label_fac_ctw(plan: &RateBackendPlan, _max_order: i64) -> String {
    let RateBackendPlan::FacCtw {
        base_depth,
        num_percept_bits,
        encoding_bits,
    } = plan
    else {
        unreachable!("fac-ctw label kernel used with non-fac-ctw plan");
    };
    format!(
        "fac-ctw(base_depth={base_depth},num_percept_bits={num_percept_bits},encoding_bits={encoding_bits})"
    )
}

pub(crate) fn rate_plan_default_name_fac_ctw(plan: &RateBackendPlan, _max_order: i64) -> String {
    let RateBackendPlan::FacCtw {
        base_depth,
        encoding_bits,
        ..
    } = plan
    else {
        unreachable!("fac-ctw default-name kernel used with non-fac-ctw plan");
    };
    format!("fac-ctw(d={base_depth},b={encoding_bits})")
}

pub(crate) fn rate_plan_display_label_zpaq(plan: &RateBackendPlan, _max_order: i64) -> String {
    let RateBackendPlan::Zpaq { method } = plan else {
        unreachable!("zpaq label kernel used with non-zpaq plan");
    };
    format!("zpaq(method={method})")
}

pub(crate) fn rate_plan_default_name_zpaq(plan: &RateBackendPlan, _max_order: i64) -> String {
    let RateBackendPlan::Zpaq { method } = plan else {
        unreachable!("zpaq default-name kernel used with non-zpaq plan");
    };
    format!("zpaq(m={method})")
}

#[cfg(feature = "backend-mamba")]
pub(crate) fn rate_plan_display_label_mamba(plan: &RateBackendPlan, _max_order: i64) -> String {
    let RateBackendPlan::Mamba { method, .. } = plan else {
        unreachable!("mamba label kernel used with non-mamba plan");
    };
    format!("mamba(method={method})")
}

#[cfg(not(feature = "backend-mamba"))]
pub(crate) fn rate_plan_display_label_mamba(_plan: &RateBackendPlan, _max_order: i64) -> String {
    unreachable!("mamba label kernel should never be used without backend-mamba")
}

#[cfg(feature = "backend-mamba")]
pub(crate) fn rate_plan_default_name_mamba(plan: &RateBackendPlan, _max_order: i64) -> String {
    let RateBackendPlan::Mamba { method, .. } = plan else {
        unreachable!("mamba default-name kernel used with non-mamba plan");
    };
    format!("mamba({method})")
}

#[cfg(not(feature = "backend-mamba"))]
pub(crate) fn rate_plan_default_name_mamba(_plan: &RateBackendPlan, _max_order: i64) -> String {
    unreachable!("mamba default-name kernel should never be used without backend-mamba")
}

#[cfg(feature = "backend-rwkv")]
pub(crate) fn rate_plan_display_label_rwkv7(plan: &RateBackendPlan, _max_order: i64) -> String {
    let RateBackendPlan::Rwkv7 { method, .. } = plan else {
        unreachable!("rwkv7 label kernel used with non-rwkv7 plan");
    };
    format!("rwkv7(method={method})")
}

#[cfg(not(feature = "backend-rwkv"))]
pub(crate) fn rate_plan_display_label_rwkv7(_plan: &RateBackendPlan, _max_order: i64) -> String {
    unreachable!("rwkv7 label kernel should never be used without backend-rwkv")
}

#[cfg(feature = "backend-rwkv")]
pub(crate) fn rate_plan_default_name_rwkv7(plan: &RateBackendPlan, _max_order: i64) -> String {
    let RateBackendPlan::Rwkv7 { method, .. } = plan else {
        unreachable!("rwkv7 default-name kernel used with non-rwkv7 plan");
    };
    format!("rwkv7({method})")
}

#[cfg(not(feature = "backend-rwkv"))]
pub(crate) fn rate_plan_default_name_rwkv7(_plan: &RateBackendPlan, _max_order: i64) -> String {
    unreachable!("rwkv7 default-name kernel should never be used without backend-rwkv")
}

pub(crate) fn rate_plan_display_label_mixture(plan: &RateBackendPlan, _max_order: i64) -> String {
    let RateBackendPlan::Mixture { kind, .. } = plan else {
        unreachable!("mixture label kernel used with non-mixture plan");
    };
    match kind {
        MixtureKind::Bayes => "mixture:bayes".to_string(),
        MixtureKind::FadingBayes => "mixture:fading-bayes".to_string(),
        MixtureKind::Switching => "mixture:switching".to_string(),
        MixtureKind::Convex => "mixture:convex".to_string(),
        MixtureKind::Mdl => "mixture:mdl".to_string(),
        MixtureKind::Neural => "mixture:neural".to_string(),
    }
}

pub(crate) fn rate_plan_default_name_mixture(plan: &RateBackendPlan, _max_order: i64) -> String {
    let RateBackendPlan::Mixture { kind, .. } = plan else {
        unreachable!("mixture default-name kernel used with non-mixture plan");
    };
    let kind = match kind {
        MixtureKind::Bayes => "bayes",
        MixtureKind::FadingBayes => "fading",
        MixtureKind::Switching => "switch",
        MixtureKind::Convex => "convex",
        MixtureKind::Mdl => "mdl",
        MixtureKind::Neural => "neural",
    };
    format!("mix({kind})")
}

pub(crate) fn rate_plan_display_label_particle(plan: &RateBackendPlan, _max_order: i64) -> String {
    let RateBackendPlan::Particle { spec } = plan else {
        unreachable!("particle label kernel used with non-particle plan");
    };
    format!(
        "particle(num_particles={},num_cells={})",
        spec.num_particles, spec.num_cells
    )
}

pub(crate) fn rate_plan_default_name_particle(plan: &RateBackendPlan, _max_order: i64) -> String {
    let RateBackendPlan::Particle { spec } = plan else {
        unreachable!("particle default-name kernel used with non-particle plan");
    };
    format!("particle(n={},c={})", spec.num_particles, spec.num_cells)
}

pub(crate) fn rate_plan_display_label_calibrated(
    plan: &RateBackendPlan,
    _max_order: i64,
) -> String {
    let RateBackendPlan::Calibrated {
        context,
        bins,
        learning_rate,
        bias_clip,
        ..
    } = plan
    else {
        unreachable!("calibrated label kernel used with non-calibrated plan");
    };
    format!(
        "calibrated(context={context:?},bins={bins},learning_rate={learning_rate},bias_clip={bias_clip})"
    )
}

pub(crate) fn rate_plan_default_name_calibrated(plan: &RateBackendPlan, max_order: i64) -> String {
    let RateBackendPlan::Calibrated { base, .. } = plan else {
        unreachable!("calibrated default-name kernel used with non-calibrated plan");
    };
    format!(
        "calibrated({})",
        rate_backend_plan_default_name(base.as_ref(), max_order)
    )
}

pub(crate) fn compression_plan_display_label_zpaq(plan: &CompressionBackendPlan) -> String {
    let CompressionBackendPlan::Zpaq { method } = plan else {
        unreachable!("zpaq compression label kernel used with non-zpaq plan");
    };
    format!("zpaq(method={method})")
}

#[cfg(feature = "backend-rwkv")]
pub(crate) fn compression_plan_display_label_rwkv7(plan: &CompressionBackendPlan) -> String {
    let CompressionBackendPlan::Rwkv7 { method, coder, .. } = plan else {
        unreachable!("rwkv7 compression label kernel used with non-rwkv7 plan");
    };
    format!("rwkv7(coder={coder:?},method={method})")
}

#[cfg(not(feature = "backend-rwkv"))]
pub(crate) fn compression_plan_display_label_rwkv7(_plan: &CompressionBackendPlan) -> String {
    unreachable!("rwkv7 compression label kernel should never be used without backend-rwkv")
}

pub(crate) fn compression_plan_display_label_rate(plan: &CompressionBackendPlan) -> String {
    let CompressionBackendPlan::Rate {
        rate_backend,
        coder,
        framing,
    } = plan
    else {
        unreachable!("rate compression label kernel used with non-rate plan");
    };
    format!(
        "{}(coder={coder:?},framing={framing:?})",
        crate::runtime::rate_backend_canonical_name(rate_backend.kind())
    )
}

pub(crate) fn encode_rate_payload_rosa(plan: &RateBackendPlan, out: &mut Vec<u8>) {
    let RateBackendPlan::RosaPlus = plan else {
        unreachable!("rosa encoder kernel used with non-rosa plan");
    };
    out.push(0);
}

pub(crate) fn encode_rate_payload_match(plan: &RateBackendPlan, out: &mut Vec<u8>) {
    let RateBackendPlan::Match {
        hash_bits,
        min_len,
        max_len,
        base_mix,
        confidence_scale,
    } = plan
    else {
        unreachable!("match encoder kernel used with non-match plan");
    };
    out.push(1);
    push_usize(out, *hash_bits);
    push_usize(out, *min_len);
    push_usize(out, *max_len);
    push_f64(out, *base_mix);
    push_f64(out, *confidence_scale);
}

pub(crate) fn encode_rate_payload_sparse_match(plan: &RateBackendPlan, out: &mut Vec<u8>) {
    let RateBackendPlan::SparseMatch {
        hash_bits,
        min_len,
        max_len,
        gap_min,
        gap_max,
        base_mix,
        confidence_scale,
    } = plan
    else {
        unreachable!("sparse-match encoder kernel used with non-sparse-match plan");
    };
    out.push(2);
    push_usize(out, *hash_bits);
    push_usize(out, *min_len);
    push_usize(out, *max_len);
    push_usize(out, *gap_min);
    push_usize(out, *gap_max);
    push_f64(out, *base_mix);
    push_f64(out, *confidence_scale);
}

pub(crate) fn encode_rate_payload_ppmd(plan: &RateBackendPlan, out: &mut Vec<u8>) {
    let RateBackendPlan::Ppmd { order, memory_mb } = plan else {
        unreachable!("ppmd encoder kernel used with non-ppmd plan");
    };
    out.push(3);
    push_usize(out, *order);
    push_usize(out, *memory_mb);
}

pub(crate) fn encode_rate_payload_sequitur(plan: &RateBackendPlan, out: &mut Vec<u8>) {
    let RateBackendPlan::Sequitur { context_bytes } = plan else {
        unreachable!("sequitur encoder kernel used with non-sequitur plan");
    };
    out.push(4);
    push_usize(out, *context_bytes);
}

pub(crate) fn encode_rate_payload_ctw(plan: &RateBackendPlan, out: &mut Vec<u8>) {
    let RateBackendPlan::Ctw { depth } = plan else {
        unreachable!("ctw encoder kernel used with non-ctw plan");
    };
    out.push(5);
    push_usize(out, *depth);
}

pub(crate) fn encode_rate_payload_fac_ctw(plan: &RateBackendPlan, out: &mut Vec<u8>) {
    let RateBackendPlan::FacCtw {
        base_depth,
        num_percept_bits,
        encoding_bits,
    } = plan
    else {
        unreachable!("fac-ctw encoder kernel used with non-fac-ctw plan");
    };
    out.push(6);
    push_usize(out, *base_depth);
    push_usize(out, *num_percept_bits);
    push_usize(out, *encoding_bits);
}

pub(crate) fn encode_rate_payload_zpaq(plan: &RateBackendPlan, out: &mut Vec<u8>) {
    let RateBackendPlan::Zpaq { method } = plan else {
        unreachable!("zpaq encoder kernel used with non-zpaq plan");
    };
    out.push(7);
    push_string(out, method);
}

#[cfg(feature = "backend-mamba")]
pub(crate) fn encode_rate_payload_mamba(plan: &RateBackendPlan, out: &mut Vec<u8>) {
    let RateBackendPlan::Mamba { method, asset, .. } = plan else {
        unreachable!("mamba encoder kernel used with non-mamba plan");
    };
    out.push(8);
    push_string(out, method);
    push_asset_ref(out, asset.as_ref());
}

#[cfg(not(feature = "backend-mamba"))]
pub(crate) fn encode_rate_payload_mamba(_plan: &RateBackendPlan, _out: &mut Vec<u8>) {
    unreachable!("mamba encoder kernel should never be used without backend-mamba")
}

#[cfg(feature = "backend-rwkv")]
pub(crate) fn encode_rate_payload_rwkv7(plan: &RateBackendPlan, out: &mut Vec<u8>) {
    let RateBackendPlan::Rwkv7 { method, asset, .. } = plan else {
        unreachable!("rwkv7 encoder kernel used with non-rwkv7 plan");
    };
    out.push(9);
    push_string(out, method);
    push_asset_ref(out, asset.as_ref());
}

#[cfg(not(feature = "backend-rwkv"))]
pub(crate) fn encode_rate_payload_rwkv7(_plan: &RateBackendPlan, _out: &mut Vec<u8>) {
    unreachable!("rwkv7 encoder kernel should never be used without backend-rwkv")
}

pub(crate) fn encode_rate_payload_mixture(plan: &RateBackendPlan, out: &mut Vec<u8>) {
    let RateBackendPlan::Mixture {
        kind,
        schedule,
        alpha,
        decay,
        experts,
    } = plan
    else {
        unreachable!("mixture encoder kernel used with non-mixture plan");
    };
    out.push(10);
    out.push(mixture_kind_tag(*kind));
    out.push(mixture_schedule_tag(*schedule));
    push_f64(out, *alpha);
    push_option_f64(out, *decay);
    push_varint(out, experts.len() as u64);
    for expert in experts.iter() {
        push_option_string(out, expert.name.as_deref());
        push_f64(out, expert.log_prior);
        push_i64(out, expert.max_order);
        encode_rate_backend_payload(expert.backend.as_ref(), out);
    }
}

pub(crate) fn encode_rate_payload_particle(plan: &RateBackendPlan, out: &mut Vec<u8>) {
    let RateBackendPlan::Particle { spec } = plan else {
        unreachable!("particle encoder kernel used with non-particle plan");
    };
    out.push(11);
    encode_particle_spec(spec, out);
}

pub(crate) fn encode_rate_payload_calibrated(plan: &RateBackendPlan, out: &mut Vec<u8>) {
    let RateBackendPlan::Calibrated {
        context,
        bins,
        learning_rate,
        bias_clip,
        base,
    } = plan
    else {
        unreachable!("calibrated encoder kernel used with non-calibrated plan");
    };
    out.push(12);
    out.push(calibration_context_tag(*context));
    push_usize(out, *bins);
    push_f64(out, *learning_rate);
    push_f64(out, *bias_clip);
    encode_rate_backend_payload(base.as_ref(), out);
}

pub(crate) fn encode_compression_payload_zpaq(plan: &CompressionBackendPlan, out: &mut Vec<u8>) {
    let CompressionBackendPlan::Zpaq { method } = plan else {
        unreachable!("zpaq compression encoder kernel used with non-zpaq plan");
    };
    out.push(0);
    push_string(out, method);
}

#[cfg(feature = "backend-rwkv")]
pub(crate) fn encode_compression_payload_rwkv7(plan: &CompressionBackendPlan, out: &mut Vec<u8>) {
    let CompressionBackendPlan::Rwkv7 {
        method,
        asset,
        coder,
        ..
    } = plan
    else {
        unreachable!("rwkv7 compression encoder kernel used with non-rwkv7 plan");
    };
    out.push(1);
    push_string(out, method);
    push_asset_ref(out, asset.as_ref());
    out.push(coder_tag(*coder));
}

#[cfg(not(feature = "backend-rwkv"))]
pub(crate) fn encode_compression_payload_rwkv7(_plan: &CompressionBackendPlan, _out: &mut Vec<u8>) {
    unreachable!("rwkv7 compression encoder kernel should never be used without backend-rwkv")
}

pub(crate) fn encode_compression_payload_rate(plan: &CompressionBackendPlan, out: &mut Vec<u8>) {
    let CompressionBackendPlan::Rate {
        rate_backend,
        coder,
        framing,
    } = plan
    else {
        unreachable!("rate compression encoder kernel used with non-rate plan");
    };
    out.push(2);
    out.push(coder_tag(*coder));
    out.push(framing_tag(*framing));
    encode_rate_backend_payload(rate_backend.as_ref(), out);
}

pub(crate) fn rate_plan_to_wrapper(plan: &RateBackendPlan) -> RateBackend {
    crate::runtime::rate_backend_wrapper_via_kernel(plan)
}

fn compression_plan_to_wrapper(plan: &CompressionBackendPlan) -> CompressionBackend {
    crate::runtime::compression_backend_wrapper_via_kernel(plan)
}

pub(crate) fn rate_backend_capabilities(plan: &RateBackendPlan) -> RateBackendCapabilities {
    crate::runtime::rate_backend_capabilities_via_kernel(plan)
}

fn compression_backend_capabilities(
    plan: &CompressionBackendPlan,
) -> CompressionBackendCapabilities {
    crate::runtime::compression_backend_capabilities_via_kernel(plan)
}

fn rate_plan_contains_zpaq(plan: &RateBackendPlan) -> bool {
    (crate::runtime::rate_backend_kernel(plan.kind()).contains_zpaq)(plan)
}

fn rate_backend_plan_display_label(plan: &RateBackendPlan, max_order: i64) -> String {
    crate::runtime::rate_backend_display_label_via_kernel(plan, max_order)
}

fn rate_backend_plan_default_name(plan: &RateBackendPlan, max_order: i64) -> String {
    crate::runtime::rate_backend_default_name_via_kernel(plan, max_order)
}

fn adapt_rate_plan_for_bit_tokens(plan: &RateBackendPlan) -> RateBackendPlan {
    crate::runtime::adapt_rate_backend_for_bit_tokens_via_kernel(plan)
}

#[cfg(feature = "backend-rwkv")]
fn rwkv_asset_ref(spec: &crate::rwkvzip::MethodSpec) -> Option<AssetRef> {
    match spec {
        crate::rwkvzip::MethodSpec::File { path, .. } => Some(AssetRef::Filesystem(path.clone())),
        crate::rwkvzip::MethodSpec::Online { .. } => None,
    }
}

#[cfg(feature = "backend-mamba")]
fn mamba_asset_ref(spec: &crate::mambazip::MethodSpec) -> Option<AssetRef> {
    match spec {
        crate::mambazip::MethodSpec::File { path, .. } => Some(AssetRef::Filesystem(path.clone())),
        crate::mambazip::MethodSpec::Online { .. } => None,
    }
}

pub(crate) fn encode_rate_backend_plan(plan: &RateBackendPlan) -> CanonicalBytes {
    let mut payload = Vec::new();
    payload.extend_from_slice(b"itrb");
    payload.push(1);
    encode_rate_backend_payload(plan, &mut payload);
    frame_canonical_payload(payload)
}

fn encode_compression_backend_plan(plan: &CompressionBackendPlan) -> CanonicalBytes {
    let mut payload = Vec::new();
    payload.extend_from_slice(b"itcb");
    payload.push(1);
    encode_compression_backend_payload(plan, &mut payload);
    frame_canonical_payload(payload)
}

fn frame_canonical_payload(payload: Vec<u8>) -> CanonicalBytes {
    let mut framed = Vec::with_capacity(payload.len() + 10);
    push_varint(&mut framed, payload.len() as u64);
    framed.extend_from_slice(&payload);
    CanonicalBytes::from(framed)
}

fn encode_rate_backend_payload(plan: &RateBackendPlan, out: &mut Vec<u8>) {
    crate::runtime::encode_rate_backend_payload_via_kernel(plan, out)
}

fn encode_compression_backend_payload(plan: &CompressionBackendPlan, out: &mut Vec<u8>) {
    crate::runtime::encode_compression_backend_payload_via_kernel(plan, out)
}

fn encode_particle_spec(spec: &ParticleSpec, out: &mut Vec<u8>) {
    push_usize(out, spec.num_particles);
    push_usize(out, spec.context_window);
    push_usize(out, spec.unroll_steps);
    push_usize(out, spec.num_cells);
    push_usize(out, spec.cell_dim);
    push_usize(out, spec.num_rules);
    push_usize(out, spec.selector_hidden);
    push_usize(out, spec.rule_hidden);
    push_usize(out, spec.noise_dim);
    push_bool(out, spec.deterministic);
    push_bool(out, spec.enable_noise);
    push_f64(out, spec.noise_scale);
    push_usize(out, spec.noise_anneal_steps);
    push_f64(out, spec.learning_rate_readout);
    push_f64(out, spec.learning_rate_selector);
    push_f64(out, spec.learning_rate_rule);
    push_usize(out, spec.bptt_depth);
    push_f64(out, spec.optimizer_momentum);
    push_f64(out, spec.grad_clip);
    push_f64(out, spec.state_clip);
    push_f64(out, spec.forget_lambda);
    push_f64(out, spec.resample_threshold);
    push_f64(out, spec.mutate_fraction);
    push_f64(out, spec.mutate_scale);
    push_bool(out, spec.mutate_model_params);
    push_usize(out, spec.diagnostics_interval);
    push_f64(out, spec.min_prob);
    push_varint(out, spec.seed);
}

fn push_varint(out: &mut Vec<u8>, mut value: u64) {
    while value >= 0x80 {
        out.push((value as u8) | 0x80);
        value >>= 7;
    }
    out.push(value as u8);
}

fn push_i64(out: &mut Vec<u8>, value: i64) {
    let zigzag = ((value << 1) ^ (value >> 63)) as u64;
    push_varint(out, zigzag);
}

fn push_usize(out: &mut Vec<u8>, value: usize) {
    push_varint(out, value as u64);
}

fn push_bool(out: &mut Vec<u8>, value: bool) {
    out.push(u8::from(value));
}

fn push_f64(out: &mut Vec<u8>, value: f64) {
    out.extend_from_slice(&value.to_bits().to_le_bytes());
}

fn push_string(out: &mut Vec<u8>, value: &str) {
    push_varint(out, value.len() as u64);
    out.extend_from_slice(value.as_bytes());
}

fn push_option_string(out: &mut Vec<u8>, value: Option<&str>) {
    match value {
        Some(value) => {
            out.push(1);
            push_string(out, value);
        }
        None => out.push(0),
    }
}

fn push_option_f64(out: &mut Vec<u8>, value: Option<f64>) {
    match value {
        Some(value) => {
            out.push(1);
            push_f64(out, value);
        }
        None => out.push(0),
    }
}

#[cfg(any(feature = "backend-mamba", feature = "backend-rwkv"))]
fn push_asset_ref(out: &mut Vec<u8>, value: Option<&AssetRef>) {
    match value {
        Some(AssetRef::Filesystem(path)) => {
            out.push(1);
            push_string(out, &path.to_string_lossy());
        }
        None => out.push(0),
    }
}

fn coder_tag(coder: CoderType) -> u8 {
    match coder {
        CoderType::AC => 0,
        CoderType::RANS => 1,
    }
}

fn framing_tag(framing: FramingMode) -> u8 {
    match framing {
        FramingMode::Raw => 0,
        FramingMode::Framed => 1,
    }
}

fn mixture_kind_tag(kind: MixtureKind) -> u8 {
    match kind {
        MixtureKind::Bayes => 0,
        MixtureKind::FadingBayes => 1,
        MixtureKind::Switching => 2,
        MixtureKind::Convex => 3,
        MixtureKind::Mdl => 4,
        MixtureKind::Neural => 5,
    }
}

fn mixture_schedule_tag(schedule: MixtureScheduleMode) -> u8 {
    match schedule {
        MixtureScheduleMode::Default => 0,
        MixtureScheduleMode::Theorem => 1,
    }
}

fn calibration_context_tag(context: CalibrationContextKind) -> u8 {
    match context {
        CalibrationContextKind::Global => 0,
        CalibrationContextKind::ByteClass => 1,
        CalibrationContextKind::Text => 2,
        CalibrationContextKind::Repeat => 3,
        CalibrationContextKind::TextRepeat => 4,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_not_prefix(a: &CanonicalBytes, b: &CanonicalBytes) {
        assert!(
            !b.as_slice().starts_with(a.as_slice()),
            "canonical encoding must be prefix-free"
        );
    }

    #[test]
    fn canonical_rate_plan_bytes_are_prefix_free_for_sample_corpus() {
        let env = SpecEnvironment::default();
        let samples = [
            RateBackend::RosaPlus,
            RateBackend::Ctw { depth: 8 },
            RateBackend::Match {
                hash_bits: 20,
                min_len: 4,
                max_len: 255,
                base_mix: 0.02,
                confidence_scale: 1.0,
            },
            RateBackend::FacCtw {
                base_depth: 6,
                num_percept_bits: 8,
                encoding_bits: 8,
            },
        ];
        let encodings: Vec<_> = samples
            .iter()
            .map(|backend| {
                validate_rate_backend_in(backend, &env)
                    .unwrap()
                    .canonical_bytes
                    .clone()
            })
            .collect();
        for (i, left) in encodings.iter().enumerate() {
            for (j, right) in encodings.iter().enumerate() {
                if i != j {
                    assert_not_prefix(left, right);
                }
            }
        }
    }

    #[test]
    fn compiled_rate_backend_clone_is_o1_arc_backed() {
        let compiled =
            validate_rate_backend_in(&RateBackend::RosaPlus, &SpecEnvironment::default())
                .unwrap()
                .compile()
                .unwrap();
        let cloned = compiled.clone();
        assert!(Arc::ptr_eq(&compiled.plan, &cloned.plan));
        assert!(Arc::ptr_eq(
            &compiled.canonical_spec,
            &cloned.canonical_spec
        ));
    }

    #[test]
    fn bit_token_adaptation_rewrites_ctw_family_without_revalidation_failure() {
        let compiled =
            validate_rate_backend_in(&RateBackend::Ctw { depth: 7 }, &SpecEnvironment::default())
                .unwrap()
                .compile()
                .unwrap();
        let adapted = compiled.adapt_for_bit_tokens().unwrap();
        assert!(matches!(
            adapted.canonical_spec(),
            RateBackend::FacCtw {
                base_depth: 7,
                num_percept_bits: 1,
                encoding_bits: 1
            }
        ));
    }

    #[cfg(feature = "backend-mixture")]
    #[cfg(feature = "backend-zpaq")]
    #[test]
    fn compiled_capabilities_track_nested_zpaq_components() {
        let backend = RateBackend::Mixture {
            spec: Arc::new(MixtureSpec::new(
                MixtureKind::Bayes,
                vec![
                    MixtureExpertSpec {
                        name: Some("ctw".to_string()),
                        log_prior: 0.0,
                        max_order: -1,
                        backend: RateBackend::Ctw { depth: 6 },
                    },
                    MixtureExpertSpec {
                        name: Some("zpaq".to_string()),
                        log_prior: -0.1,
                        max_order: -1,
                        backend: RateBackend::Zpaq {
                            method: "1".to_string(),
                        },
                    },
                ],
            )),
        };
        let compiled = validate_rate_backend_in(&backend, &SpecEnvironment::default())
            .unwrap()
            .compile()
            .unwrap();
        assert!(compiled.contains_zpaq());
        assert!(!compiled.supports_bit_token_adaptation());
    }
}
