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

mod adapters;
mod compile;
pub(crate) use adapters::*;
pub(crate) use compile::*;

/// Compilation environment for backend/spec validation and canonicalization.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
#[non_exhaustive]
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
#[non_exhaustive]
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
#[non_exhaustive]
pub enum MethodBackendFamily {
    /// Mamba family.
    Mamba,
    /// RWKV-7 family.
    Rwkv7,
}

/// Trace-model execution strategy used by VM/AIXI adapters.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
#[non_exhaustive]
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
#[non_exhaustive]
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
    /// Whether this backend has a direct native bit predictor rather than a
    /// byte-symbol adaptation over `{0,1}`.
    pub supports_native_bit_prediction: bool,
    /// Whether this backend can expose its byte PDF as a lazy binary prefix mass.
    pub supports_byte_prefix_mass: bool,
    /// Whether repeated byte-packed bit-session prefix queries remain practical
    /// without a replay-heavy fallback.
    pub supports_efficient_byte_packed_bit_sessions: bool,
    /// Whether bit observations can be undone exactly after update.
    pub supports_reversible_bit_updates: bool,
    /// Whether the backend graph contains any ZPAQ component.
    pub contains_zpaq: bool,
    /// Whether this is a method-backed neural family.
    pub method_family: Option<MethodBackendFamily>,
}

/// Shared capability metadata for compression backends.
#[derive(Clone, Debug)]
#[non_exhaustive]
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
    pub backend: Arc<RateBackendPlan>,
}

#[derive(Clone, Debug)]
pub(crate) enum RateBackendPlan {
    RosaPlus {
        max_order: i64,
    },
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
        msb_first: bool,
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
            RateBackendPlan::RosaPlus { .. } => crate::runtime::RateBackendKind::RosaPlus,
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
        threads: usize,
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
    pub fn display_label(&self) -> String {
        rate_backend_plan_display_label(self.plan.as_ref())
    }

    /// Short default backend name for logs, diagnostics, and model labels.
    pub fn default_name(&self) -> String {
        rate_backend_plan_default_name(self.plan.as_ref())
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
    pub fn display_label(&self) -> String {
        rate_backend_plan_display_label(self.plan.as_ref())
    }

    /// Short default backend name for logs, diagnostics, and model labels.
    pub fn default_name(&self) -> String {
        rate_backend_plan_default_name(self.plan.as_ref())
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

    /// Whether this backend has a direct native bit predictor rather than a
    /// byte-symbol adaptation over `{0,1}`.
    pub fn supports_native_bit_prediction(&self) -> bool {
        self.capabilities.supports_native_bit_prediction
    }

    /// Whether this backend can expose its byte PDF as a lazy binary prefix mass.
    pub fn supports_byte_prefix_mass(&self) -> bool {
        self.capabilities.supports_byte_prefix_mass
    }

    /// Whether this backend can support repeated byte-packed bit-session prefix
    /// queries without pathological replay-heavy fallback.
    pub fn supports_efficient_byte_packed_bit_sessions(&self) -> bool {
        self.capabilities
            .supports_efficient_byte_packed_bit_sessions
    }

    /// Whether bit observations can be undone exactly after update.
    pub fn supports_reversible_bit_updates(&self) -> bool {
        self.capabilities.supports_reversible_bit_updates
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
    let canonical_spec = Arc::new(rate_plan_to_wrapper(plan.as_ref()));
    crate::api::validate_rate_backend(canonical_spec.as_ref())
        .map_err(|err| SpecError::new(err.to_string()))?;
    Ok(CompiledRateBackend {
        canonical_bytes: encode_rate_backend_plan(plan.as_ref()),
        capabilities: rate_backend_capabilities(plan.as_ref()),
        canonical_spec,
        plan,
    })
}

pub(crate) fn compiled_compression_backend_from_plan(
    plan: Arc<CompressionBackendPlan>,
) -> SpecResult<CompiledCompressionBackend> {
    let canonical_spec = Arc::new(compression_plan_to_wrapper(plan.as_ref()));
    crate::api::validate_compression_backend(canonical_spec.as_ref())
        .map_err(|err| SpecError::new(err.to_string()))?;
    Ok(CompiledCompressionBackend {
        canonical_bytes: encode_compression_backend_plan(plan.as_ref()),
        capabilities: compression_backend_capabilities(plan.as_ref()),
        canonical_spec,
        plan,
    })
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

fn rate_backend_plan_display_label(plan: &RateBackendPlan) -> String {
    crate::runtime::rate_backend_display_label_via_kernel(plan)
}

fn rate_backend_plan_default_name(plan: &RateBackendPlan) -> String {
    crate::runtime::rate_backend_default_name_via_kernel(plan)
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

    fn read_varint_prefix(bytes: &[u8]) -> Result<(u64, usize), String> {
        let mut shift: u32 = 0;
        let mut out: u64 = 0;
        let mut index: usize = 0;
        loop {
            let byte = *bytes.get(index).ok_or_else(|| {
                "unexpected end while decoding canonical frame length".to_string()
            })?;
            out |= ((byte & 0x7f) as u64) << shift;
            index = index.saturating_add(1);
            if byte & 0x80 == 0 {
                return Ok((out, index));
            }
            shift = shift.saturating_add(7);
            if shift > 63 {
                return Err("invalid canonical frame varint length".to_string());
            }
        }
    }

    fn parse_framed_canonical_payload(bytes: &[u8]) -> Result<&[u8], String> {
        let (declared_len_u64, header_len) = read_varint_prefix(bytes)?;
        let declared_len = usize::try_from(declared_len_u64)
            .map_err(|_| "canonical frame length exceeds usize::MAX".to_string())?;
        let payload_end = header_len
            .checked_add(declared_len)
            .ok_or_else(|| "canonical frame length overflow".to_string())?;
        if bytes.len() < payload_end {
            return Err("truncated canonical frame payload".to_string());
        }
        if bytes.len() > payload_end {
            return Err("canonical frame has trailing bytes".to_string());
        }
        Ok(&bytes[header_len..payload_end])
    }

    fn sample_compression_backend_corpus() -> Vec<CompressionBackend> {
        let mut out = Vec::<CompressionBackend>::new();
        let leaf = crate::runtime::first_enabled_default_rate_backend_spec();
        for descriptor in crate::runtime::COMPRESSION_BACKEND_REGISTRY {
            if !descriptor.enabled {
                continue;
            }
            match descriptor.kind {
                crate::runtime::CompressionBackendKind::Zpaq => {
                    out.push(CompressionBackend::zpaq("5"));
                }
                crate::runtime::CompressionBackendKind::RateAc => {
                    if let Some(rate_backend) = leaf.clone() {
                        out.push(CompressionBackend::Rate {
                            rate_backend,
                            coder: crate::coders::CoderType::AC,
                            framing: crate::compression::FramingMode::Framed,
                        });
                    }
                }
                crate::runtime::CompressionBackendKind::RateRans => {
                    if let Some(rate_backend) = leaf.clone() {
                        out.push(CompressionBackend::Rate {
                            rate_backend,
                            coder: crate::coders::CoderType::RANS,
                            framing: crate::compression::FramingMode::Raw,
                        });
                    }
                }
                #[cfg(feature = "backend-rwkv")]
                crate::runtime::CompressionBackendKind::Rwkv7 => {
                    let options = crate::spec::CompressionBackendShorthandOptions {
                        default_framing: crate::compression::FramingMode::Raw,
                        ..Default::default()
                    };
                    let candidate = crate::spec::parse_compression_backend_name_method(
                        "rwkv7",
                        Some(
                            "cfg:hidden=64,intermediate=64,layers=1,train=sgd,lr=0.01;policy:schedule=0..100:infer",
                        ),
                        None,
                        &options,
                    )
                    .expect("rwkv shorthand candidate");
                    out.push(candidate);
                }
                #[cfg(not(feature = "backend-rwkv"))]
                crate::runtime::CompressionBackendKind::Rwkv7 => {}
            }
        }
        out
    }

    #[test]
    fn canonical_compression_code_frame_is_self_delimiting_and_typed() {
        let env = SpecEnvironment::default();
        let corpus = sample_compression_backend_corpus();
        if corpus.is_empty() {
            return;
        }
        for candidate in corpus {
            let validated =
                validate_compression_backend_in(&candidate, &env).expect("validate candidate");
            let bytes = validated.canonical_bytes.as_slice();
            let payload = parse_framed_canonical_payload(bytes).expect("valid canonical frame");
            assert!(
                payload.len() >= 5,
                "canonical payload must contain magic + version"
            );
            assert_eq!(&payload[..4], b"itcb");
            assert_eq!(payload[4], 1_u8);
        }
    }

    #[test]
    fn canonical_compression_code_frame_rejects_trailing_bytes_in_contract_parser() {
        let env = SpecEnvironment::default();
        let Some(candidate) = sample_compression_backend_corpus().into_iter().next() else {
            return;
        };
        let validated = validate_compression_backend_in(&candidate, &env).expect("validate");
        let bytes = validated.canonical_bytes.as_slice().to_vec();
        parse_framed_canonical_payload(&bytes).expect("valid canonical frame");

        let mut extended = bytes.clone();
        extended.extend_from_slice(&[0x00, 0x01]);
        let err = parse_framed_canonical_payload(&extended)
            .expect_err("trailing bytes must be rejected by canonical frame parser");
        assert!(err.contains("trailing bytes"), "{err}");
    }

    #[test]
    fn canonical_rate_plan_bytes_are_prefix_free_for_sample_corpus() {
        let env = SpecEnvironment::default();
        let samples: Vec<_> = crate::runtime::RATE_BACKEND_REGISTRY
            .iter()
            .filter(|descriptor| descriptor.enabled)
            .filter_map(|descriptor| crate::runtime::default_rate_backend_spec(descriptor.kind))
            .take(4)
            .collect();
        if samples.len() < 2 {
            return;
        }
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
        let Some(default_backend) = crate::runtime::first_enabled_default_rate_backend_spec()
        else {
            return;
        };
        let compiled = validate_rate_backend_in(&default_backend, &SpecEnvironment::default())
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

    #[cfg(feature = "backend-ctw")]
    #[test]
    fn compiled_ctw_preserves_family_identity() {
        let compiled =
            validate_rate_backend_in(&RateBackend::Ctw { depth: 7 }, &SpecEnvironment::default())
                .unwrap()
                .compile()
                .unwrap();
        assert!(matches!(
            compiled.canonical_spec(),
            RateBackend::Ctw { depth: 7 }
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
                        backend: RateBackend::Ctw { depth: 6 },
                    },
                    MixtureExpertSpec {
                        name: Some("zpaq".to_string()),
                        log_prior: -0.1,
                        backend: RateBackend::Zpaq {
                            method: crate::api::ZpaqMethodSpec::literal("1"),
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
    }
}
