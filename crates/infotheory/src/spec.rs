//! Canonical backend/spec parsing shared by Rust, CLI, and Python surfaces.

pub mod core;
mod document;

pub use self::core::{
    AssetRef, CanonicalBytes, CompiledCompressionBackend, CompiledRateBackend,
    CompressionBackendCapabilities, MethodBackendFamily, RateBackendCapabilities,
    RateBackendTraceStrategy, SpecEnvironment, ValidatedCompressionBackend, ValidatedRateBackend,
};
#[cfg(feature = "tuner")]
pub(crate) use self::document::TuneInvalidReason;
#[cfg(feature = "aixi")]
pub use self::document::WarmStartExactJhControllerSpec;
pub use self::document::{
    AiqiDiscountedControllerSpec, AssetBinding, AssetId, BuiltinEnvironmentSpec,
    CompiledPlannerController, CompiledPlannerRunSpec, CompiledSpecDocument, ControllerSpec,
    EnvironmentSpec, McAixiControllerSpec, ParsedSpecDocument, PlannerInterfaceSpec,
    PlannerRunSpec, PlannerRuntimeSpec, ResolvedAssetBinding, SharedMemoryPolicySpec, SpecDocument,
    ValidatedPlannerRunSpec, ValidatedSpecDocument, VmActionFilterSpec, VmEnvironmentSpec,
    VmFuzzMutatorSpec, VmObservationPolicySpec, VmObservationStreamModeSpec, VmPayloadEncodingSpec,
    VmRewardPolicySpec, VmRewardShapingSpec, VmRuntimeActionSourceSpec, VmTraceSpec,
    load_spec_document,
};
#[cfg(feature = "tuner")]
pub use self::document::{
    AiqiDiscountedTuneControllerSpec, AnnealedHillClimbingTuneControllerSpec,
    CompiledTuneController, CompiledTuneSpec, McAixiFacCtwTuneControllerSpec, TuneBoundsSpec,
    TuneControllerKind, TuneControllerSpec, TuneParameterRangeSpec, TunePlannerInterfaceSpec,
    TuneSpec, ValidatedTuneSpec, WarmStartExactJhTuneControllerSpec,
};

#[cfg(feature = "backend-bit-reservoir")]
use crate::api::BitReservoirConfig;
use crate::api::{
    CalibratedSpec, CalibrationContextKind, CompressionBackend, MAX_MIXTURE_NESTING,
    MixtureExpertSpec, MixtureKind, MixtureScheduleMode, MixtureSpec, ParticleSpec, RateBackend,
    parse_mixture_kind_name, parse_mixture_schedule_name,
};
use crate::validate_zpaq_rate_method;
use std::error::Error;
use std::fmt;
use std::path::{Path, PathBuf};
use std::sync::Arc;

/// Result type used by the shared spec/parsing layer.
pub type SpecResult<T> = Result<T, SpecError>;

/// Serialize JSON hash payloads with recursive lexicographic object-key order.
///
/// This is the crate-local byte contract for CRC/SHA commitments over ad-hoc
/// JSON payloads. It deliberately avoids relying on `serde_json::Map`'s backing
/// type or feature-unified insertion-order behavior.
#[cfg(feature = "aixi")]
pub(crate) fn canonical_json_bytes(
    value: &serde_json::Value,
) -> Result<Vec<u8>, serde_json::Error> {
    let mut bytes = Vec::<u8>::new();
    write_canonical_json_value(value, &mut bytes)?;
    Ok(bytes)
}

#[cfg(feature = "aixi")]
fn write_canonical_json_value(
    value: &serde_json::Value,
    out: &mut Vec<u8>,
) -> Result<(), serde_json::Error> {
    match value {
        serde_json::Value::Array(items) => {
            out.push(b'[');
            for (index, item) in items.iter().enumerate() {
                if index > 0 {
                    out.push(b',');
                }
                write_canonical_json_value(item, out)?;
            }
            out.push(b']');
            Ok(())
        }
        serde_json::Value::Object(object) => {
            let mut entries = object.iter().collect::<Vec<_>>();
            entries.sort_unstable_by(|(left, _), (right, _)| left.cmp(right));
            out.push(b'{');
            for (index, (key, item)) in entries.into_iter().enumerate() {
                if index > 0 {
                    out.push(b',');
                }
                serde_json::to_writer(&mut *out, key)?;
                out.push(b':');
                write_canonical_json_value(item, out)?;
            }
            out.push(b'}');
            Ok(())
        }
        scalar => serde_json::to_writer(out, scalar),
    }
}

/// Trait for deterministic canonical JSON serialization.
pub trait CanonicalJson {
    /// Serialize this value into canonical JSON value form.
    fn to_canonical_json_value(&self) -> SpecResult<serde_json::Value>;

    /// Serialize this value into deterministic canonical JSON text.
    fn to_canonical_json(&self) -> SpecResult<String> {
        serde_json::to_string_pretty(&self.to_canonical_json_value()?).map_err(SpecError::from)
    }
}

/// Lightweight error type for spec/config parsing and loading.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SpecError {
    message: String,
}

impl SpecError {
    /// Create a new spec error from a message.
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

impl fmt::Display for SpecError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.message)
    }
}

impl Error for SpecError {}

impl From<&str> for SpecError {
    fn from(value: &str) -> Self {
        Self::new(value)
    }
}

impl From<String> for SpecError {
    fn from(value: String) -> Self {
        Self::new(value)
    }
}

impl From<std::io::Error> for SpecError {
    fn from(value: std::io::Error) -> Self {
        Self::new(value.to_string())
    }
}

impl From<serde_json::Error> for SpecError {
    fn from(value: serde_json::Error) -> Self {
        Self::new(value.to_string())
    }
}

/// Defaults for shorthand backend parsing such as CLI `--rate-backend ... --method ...`.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct RateBackendShorthandOptions {
    /// Base directory used to resolve relative spec/model paths.
    pub base_dir: PathBuf,
    /// Default CTW depth when no numeric method is supplied.
    pub ctw_depth: usize,
    /// Default FAC-CTW base depth when no numeric method is supplied.
    pub fac_ctw_base_depth: usize,
    /// Default FAC-CTW percept width.
    pub fac_ctw_num_percept_bits: usize,
    /// Default FAC-CTW symbol encoding width.
    pub fac_ctw_encoding_bits: usize,
    /// Optional FAC-CTW MSB-first override for shorthand CLI parsing.
    ///
    /// `None` defers to compile-time default (`encoding_bits == 8` → MSB-first).
    pub fac_ctw_msb_first: Option<bool>,
    /// Default PPMD order.
    pub ppmd_order: usize,
    /// Default PPMD memory budget in MiB.
    pub ppmd_memory_mb: usize,
    /// Default Sequitur context window.
    pub sequitur_context_bytes: usize,
    /// Default ZPAQ rate method.
    pub zpaq_method: String,
    /// Optional default Mamba model path when no method is supplied.
    pub default_mamba_model_path: Option<String>,
    /// Optional default RWKV7 model path when no method is supplied.
    pub default_rwkv_model_path: Option<String>,
    /// Whether `particle` without a method should build `ParticleSpec::default()`.
    pub particle_default_if_missing_method: bool,
}

impl Default for RateBackendShorthandOptions {
    fn default() -> Self {
        Self {
            base_dir: PathBuf::from("."),
            ctw_depth: crate::rate_defaults::SHORTHAND_DEFAULT_CTW_DEPTH,
            fac_ctw_base_depth: crate::rate_defaults::SHORTHAND_DEFAULT_FAC_CTW_BASE_DEPTH,
            fac_ctw_num_percept_bits:
                crate::rate_defaults::SHORTHAND_DEFAULT_FAC_CTW_NUM_PERCEPT_BITS,
            fac_ctw_encoding_bits: crate::rate_defaults::SHORTHAND_DEFAULT_FAC_CTW_ENCODING_BITS,
            fac_ctw_msb_first: None,
            ppmd_order: crate::rate_defaults::SHORTHAND_DEFAULT_PPMD_ORDER,
            ppmd_memory_mb: crate::rate_defaults::SHORTHAND_DEFAULT_PPMD_MEMORY_MB,
            sequitur_context_bytes: crate::rate_defaults::SHORTHAND_DEFAULT_SEQUITUR_CONTEXT_BYTES,
            zpaq_method: crate::rate_defaults::SHORTHAND_DEFAULT_ZPAQ_RATE_METHOD.to_string(),
            default_mamba_model_path: None,
            default_rwkv_model_path: None,
            particle_default_if_missing_method: true,
        }
    }
}

/// Defaults for shorthand compression-backend parsing such as
/// CLI/Python `--compression-backend ... --method ...`.
#[derive(Clone)]
#[non_exhaustive]
pub struct CompressionBackendShorthandOptions {
    /// Base directory used to resolve relative model/spec paths.
    pub base_dir: PathBuf,
    /// Default ZPAQ compression method.
    pub zpaq_method: String,
    /// Default rate backend for `rate-ac`/`rate-rans` shorthands.
    pub default_rate_backend: Option<RateBackend>,
    /// Default framing mode for generic rate-coded compression backends.
    pub default_framing: crate::compression::FramingMode,
    /// Optional default RWKV7 model path when no method is supplied.
    pub default_rwkv_model_path: Option<String>,
}

impl Default for CompressionBackendShorthandOptions {
    fn default() -> Self {
        Self {
            base_dir: PathBuf::from("."),
            zpaq_method: "5".to_string(),
            default_rate_backend: None,
            default_framing: crate::compression::FramingMode::Framed,
            default_rwkv_model_path: None,
        }
    }
}

/// Resolve a spec path against a base directory.
///
/// Absolute paths are returned unchanged; relative paths are joined to `base_dir`.
pub fn resolve_spec_path(base_dir: &Path, path: impl AsRef<Path>) -> PathBuf {
    let path = path.as_ref();
    if path.is_absolute() {
        path.to_path_buf()
    } else {
        base_dir.join(path)
    }
}

/// Resolve a rate backend alias and require that it is enabled in the current build.
fn resolve_enabled_rate_backend_kind(input: &str) -> SpecResult<crate::runtime::RateBackendKind> {
    match crate::runtime::find_backend_descriptor_in_registry(
        crate::runtime::RATE_BACKEND_REGISTRY,
        input,
    ) {
        Some(descriptor) if descriptor.enabled => Ok(descriptor.kind),
        Some(descriptor) => Err(SpecError::new(format!(
            "backend '{}' requires infotheory feature '{}'",
            descriptor.canonical,
            descriptor
                .feature
                .unwrap_or("__internal-registry-mismatch__")
        ))),
        None => Err(SpecError::new(format!("unknown backend '{input}'"))),
    }
}

/// Resolve a rate backend alias and require that it is enabled in the current build.
pub fn resolve_enabled_rate_backend_name(input: &str) -> SpecResult<&'static str> {
    let kind = resolve_enabled_rate_backend_kind(input)?;
    Ok(crate::runtime::describe_rate_backend_kind(kind)
        .map_err(SpecError::new)?
        .canonical)
}

/// Resolve a compression backend alias and require that it is enabled in the current build.
fn resolve_enabled_compression_backend_kind(
    input: &str,
) -> SpecResult<crate::runtime::CompressionBackendKind> {
    match crate::runtime::find_backend_descriptor_in_registry(
        crate::runtime::COMPRESSION_BACKEND_REGISTRY,
        input,
    ) {
        Some(descriptor) if descriptor.enabled => Ok(descriptor.kind),
        Some(descriptor) => Err(SpecError::new(format!(
            "compression backend '{}' requires infotheory feature '{}'",
            descriptor.canonical,
            descriptor
                .feature
                .unwrap_or("__internal-registry-mismatch__")
        ))),
        None => Err(SpecError::new(format!(
            "unknown compression backend '{input}'"
        ))),
    }
}

/// Resolve a compression backend alias and require that it is enabled in the current build.
pub fn resolve_enabled_compression_backend_name(input: &str) -> SpecResult<&'static str> {
    let kind = resolve_enabled_compression_backend_kind(input)?;
    Ok(crate::runtime::describe_compression_backend_kind(kind)
        .map_err(SpecError::new)?
        .canonical)
}

fn resolve_default_rate_backend_spec(
    default_rate_backend: Option<RateBackend>,
) -> SpecResult<RateBackend> {
    default_rate_backend.map(Ok).unwrap_or_else(|| {
        RateBackend::try_default().map_err(|err| SpecError::new(err.to_string()))
    })
}

/// Load a plain JSON value from disk, resolving relative paths against `base_dir`.
pub fn load_json_value_from_path(
    base_dir: &Path,
    path: &str,
    label: &str,
) -> SpecResult<(serde_json::Value, PathBuf)> {
    let full = resolve_spec_path(base_dir, path);
    let raw = std::fs::read(&full)
        .map_err(|e| SpecError::new(format!("failed to read {label} '{}': {e}", full.display())))?;
    let value = serde_json::from_slice(&raw)
        .map_err(|e| SpecError::new(format!("invalid {label} JSON '{}': {e}", full.display())))?;
    Ok((value, full))
}

fn parse_calibration_context_kind(value: Option<&str>) -> SpecResult<CalibrationContextKind> {
    match value.unwrap_or("text").trim().to_ascii_lowercase().as_str() {
        "global" => Ok(CalibrationContextKind::Global),
        "byteclass" => Ok(CalibrationContextKind::ByteClass),
        "text" => Ok(CalibrationContextKind::Text),
        "repeat" => Ok(CalibrationContextKind::Repeat),
        "textrepeat" => Ok(CalibrationContextKind::TextRepeat),
        other => Err(SpecError::new(format!(
            "unknown calibration context '{other}'"
        ))),
    }
}

fn parse_mixture_kind(kind: &str) -> SpecResult<MixtureKind> {
    parse_mixture_kind_name(kind).map_err(SpecError::from)
}

fn parse_mixture_schedule(schedule: &str) -> SpecResult<MixtureScheduleMode> {
    parse_mixture_schedule_name(schedule).map_err(SpecError::from)
}

fn parse_framing_mode(value: Option<&str>) -> SpecResult<crate::compression::FramingMode> {
    match value
        .unwrap_or("framed")
        .trim()
        .to_ascii_lowercase()
        .as_str()
    {
        "framed" => Ok(crate::compression::FramingMode::Framed),
        "raw" => Ok(crate::compression::FramingMode::Raw),
        other => Err(SpecError::new(format!("unknown framing mode '{other}'"))),
    }
}

fn mixture_kind_name(kind: MixtureKind) -> &'static str {
    match kind {
        MixtureKind::Bayes => "bayes",
        MixtureKind::FadingBayes => "fading-bayes",
        MixtureKind::Switching => "switching",
        MixtureKind::Convex => "convex",
        MixtureKind::Mdl => "mdl",
        MixtureKind::Neural => "neural",
    }
}

fn mixture_schedule_name(schedule: MixtureScheduleMode) -> &'static str {
    match schedule {
        MixtureScheduleMode::Default => "default",
        MixtureScheduleMode::Theorem => "theorem",
    }
}

fn calibration_context_kind_name(kind: CalibrationContextKind) -> &'static str {
    match kind {
        CalibrationContextKind::Global => "global",
        CalibrationContextKind::ByteClass => "byteclass",
        CalibrationContextKind::Text => "text",
        CalibrationContextKind::Repeat => "repeat",
        CalibrationContextKind::TextRepeat => "textrepeat",
    }
}

fn framing_mode_name(mode: crate::compression::FramingMode) -> &'static str {
    match mode {
        crate::compression::FramingMode::Raw => "raw",
        crate::compression::FramingMode::Framed => "framed",
    }
}

#[cfg(any(feature = "backend-rwkv", feature = "backend-mamba"))]
fn canonicalize_explicit_file_method(
    base_dir: &Path,
    method: &str,
    backend_label: &str,
) -> SpecResult<Option<String>> {
    let (base, policy) = crate::backends::llm_policy::split_method_policy_segments(method)
        .map_err(|err| SpecError::new(err.to_string()))?;
    let Some(path) = base.strip_prefix("file:") else {
        return Ok(None);
    };
    let path = crate::backends::llm_policy::parse_method_file_path(path.trim());
    if path.as_os_str().is_empty() {
        return Err(SpecError::new(format!(
            "empty file path in {backend_label} method"
        )));
    }
    let full = resolve_spec_path(base_dir, &path);
    let mut canonical = format!(
        "file:{}",
        crate::backends::llm_policy::render_method_file_path(&full)
    );
    if let Some(policy) = policy {
        canonical.push_str(";policy:");
        canonical.push_str(policy.trim());
    }
    Ok(Some(canonical))
}

#[cfg(feature = "backend-rwkv")]
fn validate_rwkv_method_eager(method: &str) -> SpecResult<()> {
    crate::rwkvzip::Compressor::new_from_method(method)
        .map(|_| ())
        .map_err(|err| SpecError::new(err.to_string()))
}

#[cfg(feature = "backend-mamba")]
fn validate_mamba_method_eager(method: &str) -> SpecResult<()> {
    crate::mambazip::Compressor::new_from_method(method)
        .map(|_| ())
        .map_err(|err| SpecError::new(err.to_string()))
}

#[cfg(feature = "backend-rwkv")]
fn normalize_rwkv_method_spec_for_base_dir(
    base_dir: &Path,
    method: &crate::rwkvzip::MethodSpec,
) -> SpecResult<crate::rwkvzip::MethodSpec> {
    let normalized = match method {
        crate::rwkvzip::MethodSpec::File { path, policy } => crate::rwkvzip::MethodSpec::File {
            path: resolve_spec_path(base_dir, path),
            policy: policy.clone(),
        },
        crate::rwkvzip::MethodSpec::Online { cfg, policy } => crate::rwkvzip::MethodSpec::Online {
            cfg: cfg.clone(),
            policy: policy.clone(),
        },
    };
    let canonical = crate::rwkvzip::canonical_method_string(&normalized)
        .map_err(|err| SpecError::new(err.to_string()))?;
    validate_rwkv_method_eager(&canonical)?;
    Ok(normalized)
}

#[cfg(feature = "backend-rwkv")]
fn normalize_rwkv_path_method(
    base_dir: &Path,
    model_path: &str,
) -> SpecResult<crate::rwkvzip::MethodSpec> {
    normalize_rwkv_method_spec_for_base_dir(
        base_dir,
        &crate::rwkvzip::MethodSpec::File {
            path: PathBuf::from(model_path),
            policy: None,
        },
    )
}

#[cfg(feature = "backend-mamba")]
fn normalize_mamba_method_spec_for_base_dir(
    base_dir: &Path,
    method: &crate::mambazip::MethodSpec,
) -> SpecResult<crate::mambazip::MethodSpec> {
    let normalized = match method {
        crate::mambazip::MethodSpec::File { path, policy } => crate::mambazip::MethodSpec::File {
            path: resolve_spec_path(base_dir, path),
            policy: policy.clone(),
        },
        crate::mambazip::MethodSpec::Online { cfg, policy } => {
            crate::mambazip::MethodSpec::Online {
                cfg: cfg.clone(),
                policy: policy.clone(),
            }
        }
    };
    let canonical = crate::mambazip::canonical_method_string(&normalized)
        .map_err(|err| SpecError::new(err.to_string()))?;
    validate_mamba_method_eager(&canonical)?;
    Ok(normalized)
}

#[cfg(feature = "backend-mamba")]
fn normalize_mamba_path_method(
    base_dir: &Path,
    model_path: &str,
) -> SpecResult<crate::mambazip::MethodSpec> {
    normalize_mamba_method_spec_for_base_dir(
        base_dir,
        &crate::mambazip::MethodSpec::File {
            path: PathBuf::from(model_path),
            policy: None,
        },
    )
}

#[cfg(feature = "backend-rwkv")]
fn normalize_rwkv_method_for_base_dir(
    base_dir: &Path,
    method: &str,
) -> SpecResult<crate::rwkvzip::MethodSpec> {
    if let Some(canonical) = canonicalize_explicit_file_method(base_dir, method, "rwkv")? {
        let parsed = crate::rwkvzip::parse_method_spec(&canonical)
            .map_err(|err| SpecError::new(err.to_string()))?;
        normalize_rwkv_method_spec_for_base_dir(base_dir, &parsed)
    } else {
        let parsed = crate::rwkvzip::parse_method_spec(method)
            .map_err(|err| SpecError::new(err.to_string()))?;
        normalize_rwkv_method_spec_for_base_dir(base_dir, &parsed)
    }
}

#[cfg(feature = "backend-mamba")]
fn normalize_mamba_method_for_base_dir(
    base_dir: &Path,
    method: &str,
) -> SpecResult<crate::mambazip::MethodSpec> {
    if let Some(canonical) = canonicalize_explicit_file_method(base_dir, method, "mamba")? {
        let parsed = crate::mambazip::parse_method_spec(&canonical)
            .map_err(|err| SpecError::new(err.to_string()))?;
        normalize_mamba_method_spec_for_base_dir(base_dir, &parsed)
    } else {
        let parsed = crate::mambazip::parse_method_spec(method)
            .map_err(|err| SpecError::new(err.to_string()))?;
        normalize_mamba_method_spec_for_base_dir(base_dir, &parsed)
    }
}

#[cfg(feature = "backend-rwkv")]
const RWKV_POLICY_SCOPES: &[&str] = &[
    "embed",
    "pre_norm",
    "attn_norm",
    "ffn_norm",
    "attn",
    "ffn",
    "head",
    "bias",
    "all",
    "none",
];

#[cfg(feature = "backend-mamba")]
const MAMBA_POLICY_SCOPES: &[&str] = &[
    "embed",
    "layer_norm",
    "mixer_conv",
    "mixer_ssm",
    "mixer_proj",
    "head",
    "bias",
    "all",
    "none",
];

fn zpaq_method_to_json_value(method: &crate::api::ZpaqMethodSpec) -> serde_json::Value {
    match method {
        crate::api::ZpaqMethodSpec::Literal { value } => serde_json::json!({
            "kind": "literal",
            "value": value,
        }),
    }
}

fn parse_zpaq_method_json_value(
    value: &serde_json::Value,
    default: &str,
) -> SpecResult<crate::api::ZpaqMethodSpec> {
    if value.is_null() {
        return Ok(crate::api::ZpaqMethodSpec::literal(default));
    }
    if value.is_string() {
        return Err(SpecError::new(
            "zpaq method must use object form {'kind':'literal','value':'...'}",
        ));
    }

    let kind = value["kind"]
        .as_str()
        .ok_or_else(|| SpecError::new("zpaq method.kind is required for object form"))?;
    if kind != "literal" {
        return Err(SpecError::new(format!("unknown zpaq method kind '{kind}'")));
    }

    let method = value["value"]
        .as_str()
        .ok_or_else(|| SpecError::new("zpaq method.value must be a string"))?;
    Ok(crate::api::ZpaqMethodSpec::literal(method))
}

#[cfg(feature = "backend-rwkv")]
fn rwkv_online_config_to_json_value(cfg: &crate::rwkvzip::OnlineConfig) -> serde_json::Value {
    serde_json::json!({
        "hidden": cfg.hidden,
        "layers": cfg.layers,
        "intermediate": cfg.intermediate,
        "decay_rank": cfg.decay_rank,
        "a_rank": cfg.a_rank,
        "v_rank": cfg.v_rank,
        "g_rank": cfg.g_rank,
        "seed": cfg.seed,
        "train_mode": match cfg.train_mode {
            crate::rwkvzip::OnlineTrainMode::None => "none",
            crate::rwkvzip::OnlineTrainMode::Sgd => "sgd",
            crate::rwkvzip::OnlineTrainMode::Adam => "adam",
        },
        "lr": cfg.lr,
        "stride": cfg.stride,
    })
}

#[cfg(feature = "backend-rwkv")]
fn rwkv_online_config_from_json_value(
    value: &serde_json::Value,
) -> SpecResult<crate::rwkvzip::OnlineConfig> {
    let defaults = crate::rwkvzip::OnlineConfig::default();
    let train_mode = match value["train_mode"].as_str().unwrap_or("none") {
        "none" => crate::rwkvzip::OnlineTrainMode::None,
        "sgd" => crate::rwkvzip::OnlineTrainMode::Sgd,
        "adam" => crate::rwkvzip::OnlineTrainMode::Adam,
        other => {
            return Err(SpecError::new(format!("unknown rwkv train_mode '{other}'")));
        }
    };
    Ok(crate::rwkvzip::OnlineConfig {
        hidden: value["hidden"].as_u64().unwrap_or(defaults.hidden as u64) as usize,
        layers: value["layers"].as_u64().unwrap_or(defaults.layers as u64) as usize,
        intermediate: value["intermediate"]
            .as_u64()
            .unwrap_or(defaults.intermediate as u64) as usize,
        decay_rank: value["decay_rank"]
            .as_u64()
            .unwrap_or(defaults.decay_rank as u64) as usize,
        a_rank: value["a_rank"].as_u64().unwrap_or(defaults.a_rank as u64) as usize,
        v_rank: value["v_rank"].as_u64().unwrap_or(defaults.v_rank as u64) as usize,
        g_rank: value["g_rank"].as_u64().unwrap_or(defaults.g_rank as u64) as usize,
        seed: value["seed"].as_u64().unwrap_or(defaults.seed),
        train_mode,
        lr: value["lr"].as_f64().unwrap_or(defaults.lr as f64) as f32,
        stride: value["stride"].as_u64().unwrap_or(defaults.stride as u64) as usize,
    })
}

#[cfg(feature = "backend-rwkv")]
fn rwkv_method_to_json_value(method: &crate::rwkvzip::MethodSpec) -> SpecResult<serde_json::Value> {
    Ok(match method {
        crate::rwkvzip::MethodSpec::File { path, policy } => serde_json::json!({
            "kind": "file",
            "path": path.to_string_lossy(),
            "policy": policy.as_ref().map(crate::backends::llm_policy::LlmPolicy::canonical),
        }),
        crate::rwkvzip::MethodSpec::Online { cfg, policy } => serde_json::json!({
            "kind": "online",
            "cfg": rwkv_online_config_to_json_value(cfg),
            "policy": policy.as_ref().map(crate::backends::llm_policy::LlmPolicy::canonical),
        }),
    })
}

#[cfg(feature = "backend-rwkv")]
fn parse_rwkv_method_json_value(
    value: &serde_json::Value,
    base_dir: &Path,
) -> SpecResult<crate::rwkvzip::MethodSpec> {
    if let Some(method) = value.as_str() {
        return normalize_rwkv_method_for_base_dir(base_dir, method);
    }
    match value["kind"].as_str().unwrap_or("file") {
        "file" => {
            let path = value["path"]
                .as_str()
                .ok_or_else(|| SpecError::new("rwkv method.path is required"))?;
            let policy = value["policy"]
                .as_str()
                .map(|raw| {
                    crate::backends::llm_policy::parse_policy_segment(raw, RWKV_POLICY_SCOPES)
                })
                .transpose()
                .map_err(|err| SpecError::new(err.to_string()))?;
            normalize_rwkv_method_spec_for_base_dir(
                base_dir,
                &crate::rwkvzip::MethodSpec::File {
                    path: resolve_spec_path(base_dir, path),
                    policy,
                },
            )
        }
        "online" => {
            let cfg = rwkv_online_config_from_json_value(&value["cfg"])?;
            let policy = value["policy"]
                .as_str()
                .map(|raw| {
                    crate::backends::llm_policy::parse_policy_segment(raw, RWKV_POLICY_SCOPES)
                })
                .transpose()
                .map_err(|err| SpecError::new(err.to_string()))?;
            normalize_rwkv_method_spec_for_base_dir(
                base_dir,
                &crate::rwkvzip::MethodSpec::Online { cfg, policy },
            )
        }
        other => Err(SpecError::new(format!(
            "unknown rwkv method kind '{other}'"
        ))),
    }
}

#[cfg(feature = "backend-mamba")]
fn mamba_online_config_to_json_value(cfg: &crate::mambazip::OnlineConfig) -> serde_json::Value {
    serde_json::json!({
        "hidden": cfg.hidden,
        "layers": cfg.layers,
        "intermediate": cfg.intermediate,
        "state": cfg.state,
        "conv": cfg.conv,
        "dt_rank": cfg.dt_rank,
        "seed": cfg.seed,
        "train_mode": match cfg.train_mode {
            crate::mambazip::OnlineTrainMode::None => "none",
            crate::mambazip::OnlineTrainMode::Sgd => "sgd",
            crate::mambazip::OnlineTrainMode::Adam => "adam",
        },
        "lr": cfg.lr,
        "stride": cfg.stride,
    })
}

#[cfg(feature = "backend-mamba")]
fn mamba_online_config_from_json_value(
    value: &serde_json::Value,
) -> SpecResult<crate::mambazip::OnlineConfig> {
    let defaults = crate::mambazip::OnlineConfig::default();
    let train_mode = match value["train_mode"].as_str().unwrap_or("none") {
        "none" => crate::mambazip::OnlineTrainMode::None,
        "sgd" => crate::mambazip::OnlineTrainMode::Sgd,
        "adam" => crate::mambazip::OnlineTrainMode::Adam,
        other => {
            return Err(SpecError::new(format!(
                "unknown mamba train_mode '{other}'"
            )));
        }
    };
    Ok(crate::mambazip::OnlineConfig {
        hidden: value["hidden"].as_u64().unwrap_or(defaults.hidden as u64) as usize,
        layers: value["layers"].as_u64().unwrap_or(defaults.layers as u64) as usize,
        intermediate: value["intermediate"]
            .as_u64()
            .unwrap_or(defaults.intermediate as u64) as usize,
        state: value["state"].as_u64().unwrap_or(defaults.state as u64) as usize,
        conv: value["conv"].as_u64().unwrap_or(defaults.conv as u64) as usize,
        dt_rank: value["dt_rank"].as_u64().unwrap_or(defaults.dt_rank as u64) as usize,
        seed: value["seed"].as_u64().unwrap_or(defaults.seed),
        train_mode,
        lr: value["lr"].as_f64().unwrap_or(defaults.lr as f64) as f32,
        stride: value["stride"].as_u64().unwrap_or(defaults.stride as u64) as usize,
    })
}

#[cfg(feature = "backend-mamba")]
fn mamba_method_to_json_value(
    method: &crate::mambazip::MethodSpec,
) -> SpecResult<serde_json::Value> {
    Ok(match method {
        crate::mambazip::MethodSpec::File { path, policy } => serde_json::json!({
            "kind": "file",
            "path": path.to_string_lossy(),
            "policy": policy.as_ref().map(crate::backends::llm_policy::LlmPolicy::canonical),
        }),
        crate::mambazip::MethodSpec::Online { cfg, policy } => serde_json::json!({
            "kind": "online",
            "cfg": mamba_online_config_to_json_value(cfg),
            "policy": policy.as_ref().map(crate::backends::llm_policy::LlmPolicy::canonical),
        }),
    })
}

#[cfg(feature = "backend-mamba")]
fn parse_mamba_method_json_value(
    value: &serde_json::Value,
    base_dir: &Path,
) -> SpecResult<crate::mambazip::MethodSpec> {
    if let Some(method) = value.as_str() {
        return normalize_mamba_method_for_base_dir(base_dir, method);
    }
    match value["kind"].as_str().unwrap_or("file") {
        "file" => {
            let path = value["path"]
                .as_str()
                .ok_or_else(|| SpecError::new("mamba method.path is required"))?;
            let policy = value["policy"]
                .as_str()
                .map(|raw| {
                    crate::backends::llm_policy::parse_policy_segment(raw, MAMBA_POLICY_SCOPES)
                })
                .transpose()
                .map_err(|err| SpecError::new(err.to_string()))?;
            normalize_mamba_method_spec_for_base_dir(
                base_dir,
                &crate::mambazip::MethodSpec::File {
                    path: resolve_spec_path(base_dir, path),
                    policy,
                },
            )
        }
        "online" => {
            let cfg = mamba_online_config_from_json_value(&value["cfg"])?;
            let policy = value["policy"]
                .as_str()
                .map(|raw| {
                    crate::backends::llm_policy::parse_policy_segment(raw, MAMBA_POLICY_SCOPES)
                })
                .transpose()
                .map_err(|err| SpecError::new(err.to_string()))?;
            normalize_mamba_method_spec_for_base_dir(
                base_dir,
                &crate::mambazip::MethodSpec::Online { cfg, policy },
            )
        }
        other => Err(SpecError::new(format!(
            "unknown mamba method kind '{other}'"
        ))),
    }
}

/// Parse an RWKV7 compression backend from a method string or configured model path,
/// preserving the shared direct-vs-rate-coded lowering semantics across CLI, JSON,
/// and binding surfaces.
#[cfg(feature = "backend-rwkv")]
fn lower_rwkv7_compression_backend_method(
    method: crate::rwkvzip::MethodSpec,
    coder: crate::coders::CoderType,
    framing: crate::compression::FramingMode,
) -> CompressionBackend {
    match method {
        crate::rwkvzip::MethodSpec::File { policy: None, .. } => {
            CompressionBackend::Rwkv7 { method, coder }
        }
        crate::rwkvzip::MethodSpec::File {
            policy: Some(_), ..
        }
        | crate::rwkvzip::MethodSpec::Online { .. } => CompressionBackend::Rate {
            rate_backend: RateBackend::Rwkv7Method { method },
            coder,
            framing,
        },
    }
}

/// Parse RWKV7 compression backend shorthand into a canonical backend configuration.
///
/// When `method` is empty or absent, this falls back to
/// [`CompressionBackendShorthandOptions::default_rwkv_model_path`].
/// If the `backend-rwkv` feature is disabled, this returns a spec error.
pub fn parse_rwkv7_compression_backend_method(
    method: Option<&str>,
    coder: crate::coders::CoderType,
    options: &CompressionBackendShorthandOptions,
) -> SpecResult<CompressionBackend> {
    #[cfg(feature = "backend-rwkv")]
    {
        let method = if let Some(method) = method.filter(|value| !value.is_empty()) {
            normalize_rwkv_method_for_base_dir(&options.base_dir, method)?
        } else {
            let model_path = options.default_rwkv_model_path.as_deref().ok_or_else(|| {
                SpecError::new(
                    "rwkv7 compression backend requires a method string or a configured model path",
                )
            })?;
            normalize_rwkv_path_method(&options.base_dir, model_path)?
        };
        Ok(lower_rwkv7_compression_backend_method(
            method,
            coder,
            options.default_framing,
        ))
    }
    #[cfg(not(feature = "backend-rwkv"))]
    {
        let _ = method;
        let _ = coder;
        let _ = options;
        Err(SpecError::new(
            "rwkv7 compression backend disabled at compile time",
        ))
    }
}

/// Serialize a `ParticleSpec` into the canonical JSON representation.
pub fn particle_spec_to_json_value(spec: &ParticleSpec) -> serde_json::Value {
    serde_json::json!({
        "num_particles": spec.num_particles,
        "context_window": spec.context_window,
        "unroll_steps": spec.unroll_steps,
        "num_cells": spec.num_cells,
        "cell_dim": spec.cell_dim,
        "num_rules": spec.num_rules,
        "selector_hidden": spec.selector_hidden,
        "rule_hidden": spec.rule_hidden,
        "noise_dim": spec.noise_dim,
        "deterministic": spec.deterministic,
        "enable_noise": spec.enable_noise,
        "noise_scale": spec.noise_scale,
        "noise_anneal_steps": spec.noise_anneal_steps,
        "learning_rate_readout": spec.learning_rate_readout,
        "learning_rate_selector": spec.learning_rate_selector,
        "learning_rate_rule": spec.learning_rate_rule,
        "bptt_depth": spec.bptt_depth,
        "optimizer_momentum": spec.optimizer_momentum,
        "grad_clip": spec.grad_clip,
        "state_clip": spec.state_clip,
        "forget_lambda": spec.forget_lambda,
        "resample_threshold": spec.resample_threshold,
        "mutate_fraction": spec.mutate_fraction,
        "mutate_scale": spec.mutate_scale,
        "mutate_model_params": spec.mutate_model_params,
        "diagnostics_interval": spec.diagnostics_interval,
        "min_prob": spec.min_prob,
        "seed": spec.seed,
    })
}

/// Serialize a `MixtureExpertSpec` into the canonical JSON representation.
pub fn mixture_expert_spec_to_json_value(
    spec: &MixtureExpertSpec,
) -> SpecResult<serde_json::Value> {
    let mut value = rate_backend_to_json_value(&spec.backend)?;
    let object = value
        .as_object_mut()
        .expect("rate backend serialization must produce a JSON object");
    if let Some(name) = &spec.name {
        object.insert("name".to_string(), serde_json::Value::String(name.clone()));
    }
    object.insert("log_prior".to_string(), serde_json::json!(spec.log_prior));
    Ok(value)
}

/// Serialize a `MixtureSpec` into the canonical JSON representation.
pub fn mixture_spec_to_json_value(spec: &MixtureSpec) -> SpecResult<serde_json::Value> {
    let mut experts = Vec::with_capacity(spec.experts.len());
    for expert in &spec.experts {
        experts.push(mixture_expert_spec_to_json_value(expert)?);
    }
    Ok(serde_json::json!({
        "kind": mixture_kind_name(spec.kind),
        "schedule": mixture_schedule_name(spec.schedule),
        "alpha": spec.alpha,
        "decay": spec.decay,
        "experts": experts,
    }))
}

/// Serialize a `CalibratedSpec` into the canonical JSON representation.
pub fn calibrated_spec_to_json_value(spec: &CalibratedSpec) -> SpecResult<serde_json::Value> {
    Ok(serde_json::json!({
        "base": rate_backend_to_json_value(&spec.base)?,
        "context": calibration_context_kind_name(spec.context),
        "bins": spec.bins,
        "learning_rate": spec.learning_rate,
        "bias_clip": spec.bias_clip,
    }))
}

fn rate_backend_to_json_leaf_value(
    canonical: &str,
    backend: &RateBackend,
) -> Option<SpecResult<serde_json::Value>> {
    match backend {
        RateBackend::RosaPlus { max_order } => Some(Ok(serde_json::json!({
            "kind": canonical,
            "max_order": max_order,
        }))),
        RateBackend::Match {
            hash_bits,
            min_len,
            max_len,
            base_mix,
            confidence_scale,
        } => Some(Ok(serde_json::json!({
            "kind": canonical,
            "hash_bits": hash_bits,
            "min_len": min_len,
            "max_len": max_len,
            "base_mix": base_mix,
            "confidence_scale": confidence_scale,
        }))),
        RateBackend::SparseMatch {
            hash_bits,
            min_len,
            max_len,
            gap_min,
            gap_max,
            base_mix,
            confidence_scale,
        } => Some(Ok(serde_json::json!({
            "kind": canonical,
            "hash_bits": hash_bits,
            "min_len": min_len,
            "max_len": max_len,
            "gap_min": gap_min,
            "gap_max": gap_max,
            "base_mix": base_mix,
            "confidence_scale": confidence_scale,
        }))),
        RateBackend::Ppmd { order, memory_mb } => Some(Ok(serde_json::json!({
            "kind": canonical,
            "order": order,
            "memory_mb": memory_mb,
        }))),
        RateBackend::Sequitur { context_bytes } => Some(Ok(serde_json::json!({
            "kind": canonical,
            "context_bytes": context_bytes,
        }))),
        RateBackend::Zpaq { method } => Some(Ok(serde_json::json!({
            "kind": canonical,
            "method": zpaq_method_to_json_value(method),
        }))),
        #[cfg(feature = "backend-bit-reservoir")]
        RateBackend::BitReservoir { config } => {
            Some(Ok(bit_reservoir_config_to_json_value(canonical, config)))
        }
        RateBackend::Ctw { depth } => Some(Ok(serde_json::json!({
            "kind": canonical,
            "depth": depth,
        }))),
        RateBackend::FacCtw {
            base_depth,
            num_percept_bits,
            encoding_bits,
            msb_first,
        } => {
            let mut value = serde_json::json!({
                "kind": canonical,
                "base_depth": base_depth,
                "num_percept_bits": num_percept_bits,
                "encoding_bits": encoding_bits,
            });
            if let Some(msb_first) = msb_first {
                value["msb_first"] = serde_json::Value::Bool(*msb_first);
            }
            Some(Ok(value))
        }
        _ => None,
    }
}

/// Serialize a `RateBackend` into the canonical JSON representation.
pub fn rate_backend_to_json_value(backend: &RateBackend) -> SpecResult<serde_json::Value> {
    let canonical = backend.descriptor().map_err(SpecError::new)?.canonical;
    if let Some(value) = rate_backend_to_json_leaf_value(canonical, backend) {
        return value;
    }
    match backend {
        #[cfg(feature = "backend-mamba")]
        RateBackend::MambaMethod { method } => Ok(serde_json::json!({
            "kind": canonical,
            "method": mamba_method_to_json_value(method)?,
        })),
        #[cfg(feature = "backend-rwkv")]
        RateBackend::Rwkv7Method { method } => Ok(serde_json::json!({
            "kind": canonical,
            "method": rwkv_method_to_json_value(method)?,
        })),
        RateBackend::Mixture { spec } => Ok(serde_json::json!({
            "kind": canonical,
            "spec": mixture_spec_to_json_value(spec.as_ref())?,
        })),
        RateBackend::Particle { spec } => Ok(serde_json::json!({
            "kind": canonical,
            "spec": particle_spec_to_json_value(spec.as_ref()),
        })),
        RateBackend::Calibrated { spec } => Ok(serde_json::json!({
            "kind": canonical,
            "spec": calibrated_spec_to_json_value(spec.as_ref())?,
        })),
        _ => Err(SpecError::new(
            "internal backend serialization mismatch for current feature set",
        )),
    }
}

/// Serialize a `CompressionBackend` into the canonical JSON representation.
pub fn compression_backend_to_json_value(
    backend: &CompressionBackend,
) -> SpecResult<serde_json::Value> {
    let canonical = backend.descriptor().map_err(SpecError::new)?.canonical;
    match backend {
        CompressionBackend::Zpaq { method, threads } => Ok(serde_json::json!({
            "kind": canonical,
            "method": zpaq_method_to_json_value(method),
            "threads": threads.get(),
        })),
        #[cfg(feature = "backend-rwkv")]
        CompressionBackend::Rwkv7 { method, coder } => Ok(serde_json::json!({
            "kind": canonical,
            "method": rwkv_method_to_json_value(method)?,
            "coder": match coder {
                crate::coders::CoderType::AC => "ac",
                crate::coders::CoderType::RANS => "rans",
            },
        })),
        CompressionBackend::Rate {
            rate_backend,
            coder: _,
            framing,
        } => Ok(serde_json::json!({
            "kind": canonical,
            "rate_backend": rate_backend_to_json_value(rate_backend)?,
            "framing": framing_mode_name(*framing),
        })),
    }
}

/// Parse a `ParticleSpec` from JSON.
pub fn parse_particle_spec_value(v: &serde_json::Value) -> SpecResult<ParticleSpec> {
    if v.get("experts").is_some() {
        return Err(SpecError::new(
            "looks like a mixture spec (found 'experts'); expected ParticleSpec JSON",
        ));
    }
    if let Some(kind) = v.get("kind").and_then(|k| k.as_str()) {
        let k = kind.to_ascii_lowercase();
        if matches!(
            k.as_str(),
            "bayes"
                | "fading"
                | "fading-bayes"
                | "switch"
                | "switching"
                | "mdl"
                | "neural"
                | "mixture"
        ) {
            return Err(SpecError::new(format!(
                "looks like a mixture spec (kind='{kind}'); expected ParticleSpec JSON"
            )));
        }
    }
    let d = ParticleSpec::default();
    Ok(ParticleSpec {
        num_particles: v["num_particles"]
            .as_u64()
            .unwrap_or(d.num_particles as u64) as usize,
        context_window: v["context_window"]
            .as_u64()
            .unwrap_or(d.context_window as u64) as usize,
        unroll_steps: v["unroll_steps"].as_u64().unwrap_or(d.unroll_steps as u64) as usize,
        num_cells: v["num_cells"].as_u64().unwrap_or(d.num_cells as u64) as usize,
        cell_dim: v["cell_dim"].as_u64().unwrap_or(d.cell_dim as u64) as usize,
        num_rules: v["num_rules"].as_u64().unwrap_or(d.num_rules as u64) as usize,
        selector_hidden: v["selector_hidden"]
            .as_u64()
            .unwrap_or(d.selector_hidden as u64) as usize,
        rule_hidden: v["rule_hidden"].as_u64().unwrap_or(d.rule_hidden as u64) as usize,
        noise_dim: v["noise_dim"].as_u64().unwrap_or(d.noise_dim as u64) as usize,
        deterministic: v["deterministic"].as_bool().unwrap_or(d.deterministic),
        enable_noise: v["enable_noise"].as_bool().unwrap_or(d.enable_noise),
        noise_scale: v["noise_scale"].as_f64().unwrap_or(d.noise_scale),
        noise_anneal_steps: v["noise_anneal_steps"]
            .as_u64()
            .unwrap_or(d.noise_anneal_steps as u64) as usize,
        learning_rate_readout: v["learning_rate_readout"]
            .as_f64()
            .unwrap_or(d.learning_rate_readout),
        learning_rate_selector: v["learning_rate_selector"]
            .as_f64()
            .unwrap_or(d.learning_rate_selector),
        learning_rate_rule: v["learning_rate_rule"]
            .as_f64()
            .unwrap_or(d.learning_rate_rule),
        bptt_depth: v["bptt_depth"].as_u64().unwrap_or(d.bptt_depth as u64) as usize,
        optimizer_momentum: v["optimizer_momentum"]
            .as_f64()
            .unwrap_or(d.optimizer_momentum),
        grad_clip: v["grad_clip"].as_f64().unwrap_or(d.grad_clip),
        state_clip: v["state_clip"].as_f64().unwrap_or(d.state_clip),
        forget_lambda: v["forget_lambda"].as_f64().unwrap_or(d.forget_lambda),
        resample_threshold: v["resample_threshold"]
            .as_f64()
            .unwrap_or(d.resample_threshold),
        mutate_fraction: v["mutate_fraction"].as_f64().unwrap_or(d.mutate_fraction),
        mutate_scale: v["mutate_scale"].as_f64().unwrap_or(d.mutate_scale),
        mutate_model_params: v["mutate_model_params"]
            .as_bool()
            .unwrap_or(d.mutate_model_params),
        diagnostics_interval: v["diagnostics_interval"]
            .as_u64()
            .unwrap_or(d.diagnostics_interval as u64) as usize,
        min_prob: v["min_prob"].as_f64().unwrap_or(d.min_prob),
        seed: v["seed"].as_u64().unwrap_or(d.seed),
    })
}

#[cfg(feature = "backend-bit-reservoir")]
fn bit_reservoir_config_to_json_value(
    canonical: &str,
    config: &BitReservoirConfig,
) -> serde_json::Value {
    serde_json::json!({
        "kind": canonical,
        "hidden": config.hidden,
        "delay_bits": config.delay_bits,
        "embedding_bits": config.embedding_bits,
        "learning_rate": config.learning_rate,
        "learning_rate_decay": config.learning_rate_decay,
        "weight_decay": config.weight_decay,
        "state_decay": config.state_decay,
        "recurrent_scale": config.recurrent_scale,
        "input_scale": config.input_scale,
        "phase_scale": config.phase_scale,
        "grad_clip": config.grad_clip,
        "seed": config.seed,
    })
}

#[cfg(feature = "backend-bit-reservoir")]
fn json_usize_field(object: &serde_json::Value, field: &str, default: usize) -> SpecResult<usize> {
    match object.get(field).filter(|value| !value.is_null()) {
        Some(value) => value
            .as_u64()
            .and_then(|raw| usize::try_from(raw).ok())
            .ok_or_else(|| SpecError::new(format!("bit-reservoir {field} must be an integer"))),
        None => Ok(default),
    }
}

#[cfg(feature = "backend-bit-reservoir")]
fn json_f64_field(object: &serde_json::Value, field: &str, default: f64) -> SpecResult<f64> {
    match object.get(field).filter(|value| !value.is_null()) {
        Some(value) => value
            .as_f64()
            .ok_or_else(|| SpecError::new(format!("bit-reservoir {field} must be numeric"))),
        None => Ok(default),
    }
}

#[cfg(feature = "backend-bit-reservoir")]
fn json_u64_field(object: &serde_json::Value, field: &str, default: u64) -> SpecResult<u64> {
    match object.get(field).filter(|value| !value.is_null()) {
        Some(value) => value
            .as_u64()
            .ok_or_else(|| SpecError::new(format!("bit-reservoir {field} must be an integer"))),
        None => Ok(default),
    }
}

#[cfg(feature = "backend-bit-reservoir")]
fn parse_bit_reservoir_config_value(v: &serde_json::Value) -> SpecResult<BitReservoirConfig> {
    let defaults = BitReservoirConfig::default();
    let config = BitReservoirConfig {
        hidden: json_usize_field(v, "hidden", defaults.hidden)?,
        delay_bits: json_usize_field(v, "delay_bits", defaults.delay_bits)?,
        embedding_bits: json_usize_field(v, "embedding_bits", defaults.embedding_bits)?,
        learning_rate: json_f64_field(v, "learning_rate", defaults.learning_rate)?,
        learning_rate_decay: json_f64_field(
            v,
            "learning_rate_decay",
            defaults.learning_rate_decay,
        )?,
        weight_decay: json_f64_field(v, "weight_decay", defaults.weight_decay)?,
        state_decay: json_f64_field(v, "state_decay", defaults.state_decay)?,
        recurrent_scale: json_f64_field(v, "recurrent_scale", defaults.recurrent_scale)?,
        input_scale: json_f64_field(v, "input_scale", defaults.input_scale)?,
        phase_scale: json_f64_field(v, "phase_scale", defaults.phase_scale)?,
        grad_clip: json_f64_field(v, "grad_clip", defaults.grad_clip)?,
        seed: json_u64_field(v, "seed", defaults.seed)?,
    };
    config
        .validate()
        .map_err(|err| SpecError::new(err.to_string()))?;
    Ok(config)
}

#[cfg(feature = "backend-bit-reservoir")]
fn parse_bit_reservoir_shorthand_method(method: Option<&str>) -> SpecResult<RateBackend> {
    let mut config = BitReservoirConfig::default();
    let Some(method) = method else {
        return Ok(RateBackend::BitReservoir { config });
    };
    if let Ok(hidden) = method.parse::<usize>() {
        config.hidden = hidden;
        config
            .validate()
            .map_err(|err| SpecError::new(err.to_string()))?;
        return Ok(RateBackend::BitReservoir { config });
    }

    for raw_pair in method.split(',').map(str::trim).filter(|s| !s.is_empty()) {
        let Some((key, value)) = raw_pair.split_once('=') else {
            return Err(SpecError::new(format!(
                "bit-reservoir shorthand option '{raw_pair}' must be key=value"
            )));
        };
        let key = key.trim();
        let value = value.trim();
        match key {
            "hidden" | "h" => {
                config.hidden = value.parse::<usize>().map_err(|_| {
                    SpecError::new("bit-reservoir hidden shorthand value must be an integer")
                })?;
            }
            "delay_bits" | "delay" | "d" => {
                config.delay_bits = value.parse::<usize>().map_err(|_| {
                    SpecError::new("bit-reservoir delay_bits shorthand value must be an integer")
                })?;
            }
            "embedding_bits" | "embed_bits" | "eb" => {
                config.embedding_bits = value.parse::<usize>().map_err(|_| {
                    SpecError::new(
                        "bit-reservoir embedding_bits shorthand value must be an integer",
                    )
                })?;
            }
            "learning_rate" | "lr" => {
                config.learning_rate = value.parse::<f64>().map_err(|_| {
                    SpecError::new("bit-reservoir learning_rate shorthand value must be numeric")
                })?;
            }
            "learning_rate_decay" | "lr_decay" => {
                config.learning_rate_decay = value.parse::<f64>().map_err(|_| {
                    SpecError::new(
                        "bit-reservoir learning_rate_decay shorthand value must be numeric",
                    )
                })?;
            }
            "weight_decay" | "wd" => {
                config.weight_decay = value.parse::<f64>().map_err(|_| {
                    SpecError::new("bit-reservoir weight_decay shorthand value must be numeric")
                })?;
            }
            "state_decay" => {
                config.state_decay = value.parse::<f64>().map_err(|_| {
                    SpecError::new("bit-reservoir state_decay shorthand value must be numeric")
                })?;
            }
            "recurrent_scale" | "rec" => {
                config.recurrent_scale = value.parse::<f64>().map_err(|_| {
                    SpecError::new("bit-reservoir recurrent_scale shorthand value must be numeric")
                })?;
            }
            "input_scale" | "input" => {
                config.input_scale = value.parse::<f64>().map_err(|_| {
                    SpecError::new("bit-reservoir input_scale shorthand value must be numeric")
                })?;
            }
            "phase_scale" | "phase" => {
                config.phase_scale = value.parse::<f64>().map_err(|_| {
                    SpecError::new("bit-reservoir phase_scale shorthand value must be numeric")
                })?;
            }
            "grad_clip" | "clip" => {
                config.grad_clip = value.parse::<f64>().map_err(|_| {
                    SpecError::new("bit-reservoir grad_clip shorthand value must be numeric")
                })?;
            }
            "seed" => {
                config.seed = value.parse::<u64>().map_err(|_| {
                    SpecError::new("bit-reservoir seed shorthand value must be an integer")
                })?;
            }
            other => {
                return Err(SpecError::new(format!(
                    "unknown bit-reservoir shorthand option '{other}'"
                )));
            }
        }
    }
    config
        .validate()
        .map_err(|err| SpecError::new(err.to_string()))?;
    Ok(RateBackend::BitReservoir { config })
}

/// Parse a canonical rate-backend JSON object.
pub fn parse_rate_backend_json(
    v: &serde_json::Value,
    base_dir: &Path,
    depth: usize,
) -> SpecResult<RateBackend> {
    if depth == 0 {
        return Err(SpecError::new("backend spec nesting too deep"));
    }

    let raw_kind = v["kind"]
        .as_str()
        .ok_or_else(|| SpecError::new("backend spec missing 'kind'"))?;
    let kind = resolve_enabled_rate_backend_kind(raw_kind)?;
    if let Some(backend) = parse_rate_backend_json_leaf(kind, v)? {
        return Ok(backend);
    }

    match kind {
        crate::runtime::RateBackendKind::Mamba => {
            #[cfg(feature = "backend-mamba")]
            {
                if !v["method"].is_null() {
                    Ok(RateBackend::MambaMethod {
                        method: parse_mamba_method_json_value(&v["method"], base_dir)?,
                    })
                } else {
                    let model_path = v["model_path"].as_str().ok_or_else(|| {
                        SpecError::new("mamba backend requires 'method' or 'model_path'")
                    })?;
                    Ok(RateBackend::MambaMethod {
                        method: normalize_mamba_path_method(base_dir, model_path)?,
                    })
                }
            }
            #[cfg(not(feature = "backend-mamba"))]
            {
                Err(SpecError::new("mamba backend disabled at compile time"))
            }
        }
        crate::runtime::RateBackendKind::Rwkv7 => {
            #[cfg(feature = "backend-rwkv")]
            {
                if !v["method"].is_null() {
                    Ok(RateBackend::Rwkv7Method {
                        method: parse_rwkv_method_json_value(&v["method"], base_dir)?,
                    })
                } else {
                    let model_path = v["model_path"].as_str().ok_or_else(|| {
                        SpecError::new("rwkv7 backend requires 'method' or 'model_path'")
                    })?;
                    Ok(RateBackend::Rwkv7Method {
                        method: normalize_rwkv_path_method(base_dir, model_path)?,
                    })
                }
            }
            #[cfg(not(feature = "backend-rwkv"))]
            {
                Err(SpecError::new("rwkv backend disabled at compile time"))
            }
        }
        crate::runtime::RateBackendKind::Mixture => {
            let spec = if let Some(spec_v) = v.get("spec").filter(|value| value.is_object()) {
                parse_mixture_spec_value(spec_v, base_dir, depth - 1)?
            } else if let Some(path) = v["spec_path"].as_str() {
                let full = resolve_spec_path(base_dir, path);
                load_mixture_spec_with_depth(full.to_string_lossy().as_ref(), depth - 1)?
            } else {
                parse_mixture_spec_value(v, base_dir, depth - 1)?
            };
            Ok(RateBackend::Mixture {
                spec: Arc::new(spec),
            })
        }
        crate::runtime::RateBackendKind::Particle => {
            let spec = if let Some(spec_v) = v.get("spec").filter(|value| value.is_object()) {
                parse_particle_spec_value(spec_v)?
            } else if let Some(path) = v["spec_path"].as_str() {
                let full = resolve_spec_path(base_dir, path);
                load_particle_spec(full.to_string_lossy().as_ref())?
            } else {
                parse_particle_spec_value(v)?
            };
            spec.validate()
                .map_err(|err| SpecError::new(err.to_string()))?;
            Ok(RateBackend::Particle {
                spec: Arc::new(spec),
            })
        }
        crate::runtime::RateBackendKind::Calibrated => {
            let spec = if let Some(spec_v) = v.get("spec").filter(|value| value.is_object()) {
                parse_calibrated_spec_value(spec_v, base_dir, depth - 1)?
            } else if let Some(path) = v["spec_path"].as_str() {
                let full = resolve_spec_path(base_dir, path);
                load_calibrated_spec(full.to_string_lossy().as_ref())?
            } else {
                parse_calibrated_spec_value(v, base_dir, depth - 1)?
            };
            Ok(RateBackend::Calibrated {
                spec: Arc::new(spec),
            })
        }
        _ => Err(SpecError::new(
            "internal backend parse mismatch for current feature set",
        )),
    }
}

fn parse_rate_backend_json_leaf(
    kind: crate::runtime::RateBackendKind,
    v: &serde_json::Value,
) -> SpecResult<Option<RateBackend>> {
    let backend = match kind {
        crate::runtime::RateBackendKind::RosaPlus => RateBackend::RosaPlus {
            max_order: v["max_order"]
                .as_i64()
                .or_else(|| v["order"].as_i64())
                .unwrap_or(-1),
        },
        crate::runtime::RateBackendKind::Ctw => RateBackend::Ctw {
            depth: v["depth"]
                .as_u64()
                .unwrap_or(crate::rate_defaults::JSON_DEFAULT_CTW_DEPTH as u64)
                as usize,
        },
        crate::runtime::RateBackendKind::FacCtw => {
            let base_depth = v["base_depth"]
                .as_u64()
                .unwrap_or(crate::rate_defaults::JSON_DEFAULT_FAC_CTW_BASE_DEPTH as u64)
                as usize;
            let encoding_bits = v["encoding_bits"]
                .as_u64()
                .unwrap_or(crate::rate_defaults::JSON_DEFAULT_FAC_CTW_ENCODING_BITS as u64)
                as usize;
            let num_percept_bits = v["num_percept_bits"]
                .as_u64()
                .unwrap_or(encoding_bits as u64) as usize;
            let msb_first = v.get("msb_first").and_then(serde_json::Value::as_bool);
            RateBackend::FacCtw {
                base_depth,
                num_percept_bits,
                encoding_bits,
                msb_first,
            }
        }
        crate::runtime::RateBackendKind::Match => RateBackend::Match {
            hash_bits: v["hash_bits"]
                .as_u64()
                .unwrap_or(crate::rate_defaults::JSON_DEFAULT_MATCH_HASH_BITS as u64)
                as usize,
            min_len: v["min_len"]
                .as_u64()
                .unwrap_or(crate::rate_defaults::JSON_DEFAULT_MATCH_MIN_LEN as u64)
                as usize,
            max_len: v["max_len"]
                .as_u64()
                .unwrap_or(crate::rate_defaults::JSON_DEFAULT_MATCH_MAX_LEN as u64)
                as usize,
            base_mix: v["base_mix"]
                .as_f64()
                .unwrap_or(crate::rate_defaults::JSON_DEFAULT_MATCH_BASE_MIX),
            confidence_scale: v["confidence_scale"]
                .as_f64()
                .unwrap_or(crate::rate_defaults::JSON_DEFAULT_MATCH_CONFIDENCE_SCALE),
        },
        crate::runtime::RateBackendKind::SparseMatch => RateBackend::SparseMatch {
            hash_bits: v["hash_bits"]
                .as_u64()
                .unwrap_or(crate::rate_defaults::JSON_DEFAULT_SPARSE_MATCH_HASH_BITS as u64)
                as usize,
            min_len: v["min_len"]
                .as_u64()
                .unwrap_or(crate::rate_defaults::JSON_DEFAULT_SPARSE_MATCH_MIN_LEN as u64)
                as usize,
            max_len: v["max_len"]
                .as_u64()
                .unwrap_or(crate::rate_defaults::JSON_DEFAULT_SPARSE_MATCH_MAX_LEN as u64)
                as usize,
            gap_min: v["gap_min"]
                .as_u64()
                .unwrap_or(crate::rate_defaults::JSON_DEFAULT_SPARSE_MATCH_GAP_MIN as u64)
                as usize,
            gap_max: v["gap_max"]
                .as_u64()
                .unwrap_or(crate::rate_defaults::JSON_DEFAULT_SPARSE_MATCH_GAP_MAX as u64)
                as usize,
            base_mix: v["base_mix"]
                .as_f64()
                .unwrap_or(crate::rate_defaults::JSON_DEFAULT_SPARSE_MATCH_BASE_MIX),
            confidence_scale: v["confidence_scale"]
                .as_f64()
                .unwrap_or(crate::rate_defaults::JSON_DEFAULT_SPARSE_MATCH_CONFIDENCE_SCALE),
        },
        crate::runtime::RateBackendKind::Ppmd => RateBackend::Ppmd {
            order: v["order"]
                .as_u64()
                .unwrap_or(crate::rate_defaults::JSON_DEFAULT_PPMD_ORDER as u64)
                as usize,
            memory_mb: v["memory_mb"]
                .as_u64()
                .unwrap_or(crate::rate_defaults::JSON_DEFAULT_PPMD_MEMORY_MB as u64)
                as usize,
        },
        crate::runtime::RateBackendKind::Sequitur => RateBackend::Sequitur {
            context_bytes: v["context_bytes"]
                .as_u64()
                .unwrap_or(crate::rate_defaults::JSON_DEFAULT_SEQUITUR_CONTEXT_BYTES as u64)
                as usize,
        },
        crate::runtime::RateBackendKind::Zpaq => {
            let method = parse_zpaq_method_json_value(
                &v["method"],
                crate::rate_defaults::JSON_DEFAULT_ZPAQ_RATE_METHOD,
            )?;
            validate_zpaq_rate_method(method.value())
                .map_err(|err| SpecError::new(err.to_string()))?;
            RateBackend::Zpaq { method }
        }
        crate::runtime::RateBackendKind::BitReservoir => {
            #[cfg(feature = "backend-bit-reservoir")]
            {
                RateBackend::BitReservoir {
                    config: parse_bit_reservoir_config_value(v)?,
                }
            }
            #[cfg(not(feature = "backend-bit-reservoir"))]
            {
                return Err(SpecError::new(
                    "bit-reservoir backend disabled at compile time",
                ));
            }
        }
        _ => return Ok(None),
    };
    Ok(Some(backend))
}

/// Parse a canonical compression-backend JSON object.
pub fn parse_compression_backend_json(
    v: &serde_json::Value,
    base_dir: &Path,
    default_rate_backend: Option<RateBackend>,
    default_framing: crate::compression::FramingMode,
) -> SpecResult<CompressionBackend> {
    let raw_kind = v["kind"]
        .as_str()
        .ok_or_else(|| SpecError::new("compression backend spec missing 'kind'"))?;
    let kind = resolve_enabled_compression_backend_kind(raw_kind)?;
    let framing = v["framing"]
        .as_str()
        .map(|value| parse_framing_mode(Some(value)))
        .transpose()?
        .unwrap_or(default_framing);

    match kind {
        crate::runtime::CompressionBackendKind::Zpaq => {
            let method = parse_zpaq_method_json_value(&v["method"], "5")?;
            let threads = if let Some(raw_value) = v.get("threads").filter(|value| !value.is_null())
            {
                let raw_u64 = raw_value
                    .as_u64()
                    .ok_or_else(|| SpecError::new("zpaq threads must be an integer >= 1"))?;
                let raw = usize::try_from(raw_u64)
                    .map_err(|_| SpecError::new("zpaq threads exceeds usize::MAX"))?;
                std::num::NonZeroUsize::new(raw)
                    .ok_or_else(|| SpecError::new("zpaq threads must be >= 1"))?
            } else {
                std::num::NonZeroUsize::MIN
            };
            crate::zpaq_compress_to_vec(&[], method.value()).map_err(|err| {
                SpecError::new(format!(
                    "invalid zpaq compression method '{}': {err}",
                    method.value()
                ))
            })?;
            Ok(CompressionBackend::Zpaq { method, threads })
        }
        crate::runtime::CompressionBackendKind::RateAc
        | crate::runtime::CompressionBackendKind::RateRans => {
            let rate_backend = if let Some(rate_backend_v) = v.get("rate_backend") {
                parse_rate_backend_json(rate_backend_v, base_dir, MAX_MIXTURE_NESTING)?
            } else if let Some(backend_v) = v.get("backend_spec") {
                parse_rate_backend_json(backend_v, base_dir, MAX_MIXTURE_NESTING)?
            } else {
                resolve_default_rate_backend_spec(default_rate_backend.clone())?
            };
            let coder = if kind == crate::runtime::CompressionBackendKind::RateAc {
                crate::coders::CoderType::AC
            } else {
                crate::coders::CoderType::RANS
            };
            Ok(CompressionBackend::Rate {
                rate_backend,
                coder,
                framing,
            })
        }
        crate::runtime::CompressionBackendKind::Rwkv7 => {
            #[cfg(feature = "backend-rwkv")]
            {
                let coder = if let Some(value) = v["coder"].as_str() {
                    crate::backends::parse_rwkv7_coder(value).ok_or_else(|| {
                        SpecError::new("rwkv7 compression coder must be 'ac' or 'rans'")
                    })?
                } else {
                    crate::coders::CoderType::AC
                };
                let parsed_method = if !v["method"].is_null() {
                    Some(parse_rwkv_method_json_value(&v["method"], base_dir)?)
                } else {
                    None
                };
                let model_path = v["model_path"].as_str();
                if parsed_method.is_none() && model_path.is_none() {
                    Err(SpecError::new(
                        "rwkv7 compression backend requires 'method' or 'model_path'",
                    ))
                } else if let Some(method) = parsed_method {
                    Ok(lower_rwkv7_compression_backend_method(
                        method, coder, framing,
                    ))
                } else {
                    let opts = CompressionBackendShorthandOptions {
                        base_dir: base_dir.to_path_buf(),
                        default_framing: framing,
                        default_rwkv_model_path: model_path.map(ToOwned::to_owned),
                        ..Default::default()
                    };
                    parse_rwkv7_compression_backend_method(None, coder, &opts)
                }
            }
            #[cfg(not(feature = "backend-rwkv"))]
            {
                Err(SpecError::new(
                    "rwkv7 compression backend disabled at compile time",
                ))
            }
        }
    }
}

/// Parse a calibrated backend specification.
pub fn parse_calibrated_spec_value(
    v: &serde_json::Value,
    base_dir: &Path,
    depth: usize,
) -> SpecResult<CalibratedSpec> {
    if depth == 0 {
        return Err(SpecError::new("calibrated spec nesting too deep"));
    }

    let base_backend = if let Some(base_v) = v.get("base") {
        parse_rate_backend_json(base_v, base_dir, depth - 1)?
    } else if let Some(path) = v["base_path"].as_str() {
        let (value, full) = load_json_value_from_path(base_dir, path, "calibrated base backend")?;
        parse_rate_backend_json(&value, full.parent().unwrap_or(base_dir), depth - 1)?
    } else {
        return Err(SpecError::new(
            "calibrated backend requires 'base' or 'base_path'",
        ));
    };

    Ok(CalibratedSpec {
        base: base_backend,
        context: parse_calibration_context_kind(v["context"].as_str())?,
        bins: v["bins"].as_u64().unwrap_or(32) as usize,
        learning_rate: v["learning_rate"].as_f64().unwrap_or(1.0 / 32.0),
        bias_clip: v["bias_clip"].as_f64().unwrap_or(16.0),
    })
}

/// Parse one mixture expert entry.
pub fn parse_mixture_expert_value(
    v: &serde_json::Value,
    base_dir: &Path,
    depth: usize,
) -> SpecResult<MixtureExpertSpec> {
    if depth == 0 {
        return Err(SpecError::new("mixture spec nesting too deep"));
    }

    let backend = parse_rate_backend_json(v, base_dir, depth - 1)?;

    Ok(MixtureExpertSpec {
        name: v["name"].as_str().map(|s| s.to_string()),
        log_prior: v["log_prior"]
            .as_f64()
            .or_else(|| v["prior"].as_f64())
            .unwrap_or(0.0),
        backend,
    })
}

/// Parse a `MixtureSpec` from JSON.
pub fn parse_mixture_spec_value(
    v: &serde_json::Value,
    base_dir: &Path,
    depth: usize,
) -> SpecResult<MixtureSpec> {
    if depth == 0 {
        return Err(SpecError::new("mixture spec nesting too deep"));
    }

    let kind_str = v["kind"]
        .as_str()
        .or_else(|| v["mixture_kind"].as_str())
        .or_else(|| v["mix_kind"].as_str())
        .unwrap_or("bayes");
    let kind = parse_mixture_kind(kind_str)?;
    let schedule = v["schedule"]
        .as_str()
        .or_else(|| v["schedule_mode"].as_str())
        .or_else(|| v["mixture_schedule"].as_str())
        .map(parse_mixture_schedule)
        .transpose()?
        .unwrap_or(MixtureScheduleMode::Default);

    let experts_v = v["experts"]
        .as_array()
        .ok_or_else(|| SpecError::new("mixture spec missing 'experts' array"))?;
    if experts_v.is_empty() {
        return Err(SpecError::new(
            "mixture spec must include at least one expert",
        ));
    }

    let mut experts = Vec::with_capacity(experts_v.len());
    for expert in experts_v {
        experts.push(parse_mixture_expert_value(expert, base_dir, depth - 1)?);
    }

    let mut spec = MixtureSpec::new(kind, experts)
        .with_schedule(schedule)
        .with_alpha(v["alpha"].as_f64().unwrap_or(0.01));
    if let Some(decay) = v["decay"].as_f64() {
        spec = spec.with_decay(decay);
    }
    spec.validate()
        .map_err(|err| SpecError::new(err.to_string()))?;
    Ok(spec)
}

fn read_mixture_value_with_zpaq_fallback(path: &Path) -> SpecResult<serde_json::Value> {
    let raw = std::fs::read(path).map_err(|e| {
        SpecError::new(format!(
            "failed to read mixture spec '{}': {e}",
            path.display()
        ))
    })?;
    match serde_json::from_slice(&raw) {
        Ok(value) => Ok(value),
        Err(json_err) => {
            #[cfg(feature = "backend-zpaq")]
            {
                let decompressed = zpaq_rs::decompress_to_vec(&raw).map_err(|_| {
                    SpecError::new(format!(
                        "failed to parse mixture JSON '{}': {json_err}",
                        path.display()
                    ))
                })?;
                serde_json::from_slice(&decompressed).map_err(|e| {
                    SpecError::new(format!("invalid mixture JSON '{}': {e}", path.display()))
                })
            }
            #[cfg(not(feature = "backend-zpaq"))]
            {
                Err(SpecError::new(format!(
                    "failed to parse mixture JSON '{}', and zpaq support is disabled at compile time: {json_err}",
                    path.display()
                )))
            }
        }
    }
}

/// Load a mixture spec from disk.
pub fn load_mixture_spec(path: &str) -> SpecResult<MixtureSpec> {
    load_mixture_spec_with_depth(path, MAX_MIXTURE_NESTING)
}

/// Load a mixture spec from disk with an explicit remaining nesting budget.
pub fn load_mixture_spec_with_depth(path: &str, depth: usize) -> SpecResult<MixtureSpec> {
    let full = Path::new(path);
    let value = read_mixture_value_with_zpaq_fallback(full)?;
    let base_dir = full.parent().unwrap_or_else(|| Path::new("."));
    parse_mixture_spec_value(&value, base_dir, depth)
}

/// Load a particle spec from disk.
pub fn load_particle_spec(path: &str) -> SpecResult<ParticleSpec> {
    let full = Path::new(path);
    let raw = std::fs::read(full).map_err(|e| {
        SpecError::new(format!(
            "failed to read particle spec '{}': {e}",
            full.display()
        ))
    })?;
    let value: serde_json::Value = serde_json::from_slice(&raw).map_err(|e| {
        SpecError::new(format!(
            "invalid particle spec JSON '{}': {e}",
            full.display()
        ))
    })?;
    let spec = parse_particle_spec_value(&value)?;
    spec.validate()
        .map_err(|err| SpecError::new(err.to_string()))?;
    Ok(spec)
}

/// Load a calibrated spec from disk.
pub fn load_calibrated_spec(path: &str) -> SpecResult<CalibratedSpec> {
    let full = Path::new(path);
    let raw = std::fs::read(full).map_err(|e| {
        SpecError::new(format!(
            "failed to read calibrated spec '{}': {e}",
            full.display()
        ))
    })?;
    let value: serde_json::Value = serde_json::from_slice(&raw).map_err(|e| {
        SpecError::new(format!(
            "invalid calibrated spec JSON '{}': {e}",
            full.display()
        ))
    })?;
    let base_dir = full.parent().unwrap_or_else(|| Path::new("."));
    parse_calibrated_spec_value(&value, base_dir, MAX_MIXTURE_NESTING)
}

/// Load a single expert spec from disk.
pub fn load_expert_spec(path: &str) -> SpecResult<MixtureExpertSpec> {
    let full = Path::new(path);
    let raw = std::fs::read(full).map_err(|e| {
        SpecError::new(format!(
            "failed to read expert spec '{}': {e}",
            full.display()
        ))
    })?;
    let value: serde_json::Value = serde_json::from_slice(&raw).map_err(|e| {
        SpecError::new(format!(
            "invalid expert spec JSON '{}': {e}",
            full.display()
        ))
    })?;
    let base_dir = full.parent().unwrap_or_else(|| Path::new("."));
    parse_mixture_expert_value(&value, base_dir, MAX_MIXTURE_NESTING)
}

/// Build a backend from shorthand CLI/Python-style `name` + optional `method` inputs.
pub fn parse_rate_backend_name_method(
    name: &str,
    method: Option<&str>,
    options: &RateBackendShorthandOptions,
) -> SpecResult<RateBackend> {
    let kind = resolve_enabled_rate_backend_kind(name)?;
    let method = method.filter(|value| !value.is_empty());
    if let Some(backend) = parse_rate_backend_name_method_leaf(kind, method, options)? {
        return Ok(backend);
    }

    match kind {
        crate::runtime::RateBackendKind::Mamba => {
            #[cfg(feature = "backend-mamba")]
            {
                if let Some(method) = method {
                    Ok(RateBackend::MambaMethod {
                        method: normalize_mamba_method_for_base_dir(&options.base_dir, method)?,
                    })
                } else if let Some(path) = options.default_mamba_model_path.as_deref() {
                    Ok(RateBackend::MambaMethod {
                        method: normalize_mamba_path_method(&options.base_dir, path)?,
                    })
                } else {
                    Err(SpecError::new(
                        "mamba backend requires method string (cfg:...;policy:... or file:...) or a configured model path",
                    ))
                }
            }
            #[cfg(not(feature = "backend-mamba"))]
            {
                Err(SpecError::new("mamba backend disabled at compile time"))
            }
        }
        crate::runtime::RateBackendKind::Rwkv7 => {
            #[cfg(feature = "backend-rwkv")]
            {
                if let Some(method) = method {
                    Ok(RateBackend::Rwkv7Method {
                        method: normalize_rwkv_method_for_base_dir(&options.base_dir, method)?,
                    })
                } else if let Some(path) = options.default_rwkv_model_path.as_deref() {
                    Ok(RateBackend::Rwkv7Method {
                        method: normalize_rwkv_path_method(&options.base_dir, path)?,
                    })
                } else {
                    Err(SpecError::new(
                        "rwkv backend requires method string (cfg:...;policy:... or file:...) or a configured model path",
                    ))
                }
            }
            #[cfg(not(feature = "backend-rwkv"))]
            {
                Err(SpecError::new("rwkv backend disabled at compile time"))
            }
        }
        crate::runtime::RateBackendKind::Mixture => {
            let path = method.ok_or_else(|| {
                SpecError::new("mixture backend requires a path to a MixtureSpec JSON file")
            })?;
            let full = resolve_spec_path(&options.base_dir, path);
            Ok(RateBackend::Mixture {
                spec: Arc::new(load_mixture_spec(full.to_string_lossy().as_ref())?),
            })
        }
        crate::runtime::RateBackendKind::Particle => {
            let spec = if let Some(path) = method {
                let full = resolve_spec_path(&options.base_dir, path);
                load_particle_spec(full.to_string_lossy().as_ref())?
            } else if options.particle_default_if_missing_method {
                ParticleSpec::default()
            } else {
                return Err(SpecError::new(
                    "particle backend requires a path to a ParticleSpec JSON file",
                ));
            };
            spec.validate()
                .map_err(|err| SpecError::new(err.to_string()))?;
            Ok(RateBackend::Particle {
                spec: Arc::new(spec),
            })
        }
        crate::runtime::RateBackendKind::Calibrated => {
            let path = method.ok_or_else(|| {
                SpecError::new("calibrated backend requires a path to a CalibratedSpec JSON file")
            })?;
            let full = resolve_spec_path(&options.base_dir, path);
            Ok(RateBackend::Calibrated {
                spec: Arc::new(load_calibrated_spec(full.to_string_lossy().as_ref())?),
            })
        }
        _ => Err(SpecError::new(
            "internal backend shorthand parse mismatch for current feature set",
        )),
    }
}

fn parse_rate_backend_name_method_leaf(
    kind: crate::runtime::RateBackendKind,
    method: Option<&str>,
    options: &RateBackendShorthandOptions,
) -> SpecResult<Option<RateBackend>> {
    let backend = match kind {
        crate::runtime::RateBackendKind::RosaPlus => RateBackend::RosaPlus {
            max_order: method
                .and_then(|value| value.parse::<i64>().ok())
                .unwrap_or(-1),
        },
        crate::runtime::RateBackendKind::Match => RateBackend::Match {
            hash_bits: crate::rate_defaults::JSON_DEFAULT_MATCH_HASH_BITS,
            min_len: crate::rate_defaults::JSON_DEFAULT_MATCH_MIN_LEN,
            max_len: crate::rate_defaults::JSON_DEFAULT_MATCH_MAX_LEN,
            base_mix: crate::rate_defaults::JSON_DEFAULT_MATCH_BASE_MIX,
            confidence_scale: crate::rate_defaults::JSON_DEFAULT_MATCH_CONFIDENCE_SCALE,
        },
        crate::runtime::RateBackendKind::SparseMatch => RateBackend::SparseMatch {
            hash_bits: crate::rate_defaults::JSON_DEFAULT_SPARSE_MATCH_HASH_BITS,
            min_len: crate::rate_defaults::JSON_DEFAULT_SPARSE_MATCH_MIN_LEN,
            max_len: crate::rate_defaults::JSON_DEFAULT_SPARSE_MATCH_MAX_LEN,
            gap_min: crate::rate_defaults::JSON_DEFAULT_SPARSE_MATCH_GAP_MIN,
            gap_max: crate::rate_defaults::JSON_DEFAULT_SPARSE_MATCH_GAP_MAX,
            base_mix: crate::rate_defaults::JSON_DEFAULT_SPARSE_MATCH_BASE_MIX,
            confidence_scale: crate::rate_defaults::JSON_DEFAULT_SPARSE_MATCH_CONFIDENCE_SCALE,
        },
        crate::runtime::RateBackendKind::Ppmd => RateBackend::Ppmd {
            order: method
                .and_then(|value| value.parse::<usize>().ok())
                .unwrap_or(options.ppmd_order),
            memory_mb: options.ppmd_memory_mb,
        },
        crate::runtime::RateBackendKind::Sequitur => RateBackend::Sequitur {
            context_bytes: method
                .and_then(|value| value.parse::<usize>().ok())
                .unwrap_or(options.sequitur_context_bytes),
        },
        crate::runtime::RateBackendKind::Ctw => RateBackend::Ctw {
            depth: method
                .and_then(|value| value.parse::<usize>().ok())
                .unwrap_or(options.ctw_depth),
        },
        crate::runtime::RateBackendKind::FacCtw => {
            let base_depth = method
                .and_then(|value| value.parse::<usize>().ok())
                .unwrap_or(options.fac_ctw_base_depth);
            RateBackend::FacCtw {
                base_depth,
                num_percept_bits: options.fac_ctw_num_percept_bits,
                encoding_bits: options.fac_ctw_encoding_bits,
                msb_first: options.fac_ctw_msb_first,
            }
        }
        crate::runtime::RateBackendKind::Zpaq => {
            let method = method
                .map(ToOwned::to_owned)
                .unwrap_or_else(|| options.zpaq_method.clone());
            validate_zpaq_rate_method(&method).map_err(|err| SpecError::new(err.to_string()))?;
            RateBackend::Zpaq {
                method: crate::api::ZpaqMethodSpec::literal(method),
            }
        }
        crate::runtime::RateBackendKind::BitReservoir => {
            #[cfg(feature = "backend-bit-reservoir")]
            {
                parse_bit_reservoir_shorthand_method(method)?
            }
            #[cfg(not(feature = "backend-bit-reservoir"))]
            {
                return Err(SpecError::new(
                    "bit-reservoir backend disabled at compile time",
                ));
            }
        }
        _ => return Ok(None),
    };
    Ok(Some(backend))
}

/// Parse and compile a shorthand CLI/Python-style rate backend.
pub fn compile_rate_backend_name_method(
    name: &str,
    method: Option<&str>,
    options: &RateBackendShorthandOptions,
) -> SpecResult<CompiledRateBackend> {
    let env = SpecEnvironment::new(options.base_dir.clone());
    parse_rate_backend_name_method(name, method, options)?.compile_in(&env)
}

/// Build a compression backend from shorthand CLI/Python-style
/// `name` + optional `method` inputs.
pub fn parse_compression_backend_name_method(
    name: &str,
    method: Option<&str>,
    rate_backend: Option<RateBackend>,
    options: &CompressionBackendShorthandOptions,
) -> SpecResult<CompressionBackend> {
    let kind = resolve_enabled_compression_backend_kind(name)?;
    let method = method.filter(|value| !value.is_empty());

    match kind {
        crate::runtime::CompressionBackendKind::Zpaq => {
            let method = method
                .map(ToOwned::to_owned)
                .unwrap_or_else(|| options.zpaq_method.clone());
            crate::zpaq_compress_to_vec(&[], &method).map_err(|err| {
                SpecError::new(format!("invalid zpaq compression method '{method}': {err}"))
            })?;
            Ok(CompressionBackend::zpaq(method))
        }
        crate::runtime::CompressionBackendKind::RateAc => Ok(CompressionBackend::Rate {
            rate_backend: rate_backend
                .or_else(|| options.default_rate_backend.clone())
                .map(Ok)
                .unwrap_or_else(|| resolve_default_rate_backend_spec(None))?,
            coder: crate::coders::CoderType::AC,
            framing: options.default_framing,
        }),
        crate::runtime::CompressionBackendKind::RateRans => Ok(CompressionBackend::Rate {
            rate_backend: rate_backend
                .or_else(|| options.default_rate_backend.clone())
                .map(Ok)
                .unwrap_or_else(|| resolve_default_rate_backend_spec(None))?,
            coder: crate::coders::CoderType::RANS,
            framing: options.default_framing,
        }),
        crate::runtime::CompressionBackendKind::Rwkv7 => {
            #[cfg(feature = "backend-rwkv")]
            {
                match method {
                    Some(m) if crate::backends::parse_rwkv7_coder(m).is_some() => {
                        let model_path = options.default_rwkv_model_path.as_deref().ok_or_else(|| {
                            SpecError::new(
                                "rwkv7 compression backend requires a configured model path when only a coder alias is provided",
                            )
                        })?;
                        Ok(CompressionBackend::Rwkv7 {
                            method: normalize_rwkv_path_method(&options.base_dir, model_path)?,
                            coder: crate::backends::parse_rwkv7_coder(m)
                                .expect("coder alias already validated"),
                        })
                    }
                    Some(m) => parse_rwkv7_compression_backend_method(
                        Some(m),
                        crate::coders::CoderType::AC,
                        options,
                    ),
                    None => parse_rwkv7_compression_backend_method(
                        None,
                        crate::coders::CoderType::AC,
                        options,
                    ),
                }
            }
            #[cfg(not(feature = "backend-rwkv"))]
            {
                Err(SpecError::new(
                    "rwkv7 compression backend disabled at compile time",
                ))
            }
        }
    }
}

/// Parse and compile a shorthand CLI/Python-style compression backend.
pub fn compile_compression_backend_name_method(
    name: &str,
    method: Option<&str>,
    rate_backend: Option<RateBackend>,
    options: &CompressionBackendShorthandOptions,
) -> SpecResult<CompiledCompressionBackend> {
    let env = SpecEnvironment::new(options.base_dir.clone());
    parse_compression_backend_name_method(name, method, rate_backend, options)?.compile_in(&env)
}

#[cfg(test)]
mod tests {
    #[allow(unused_imports)]
    use super::*;
    use crate::api::{
        CalibratedSpec, CalibrationContextKind, CompressionBackend, MAX_MIXTURE_NESTING,
        MixtureExpertSpec, MixtureKind, MixtureSpec, RateBackend,
    };
    #[cfg(any(
        feature = "all-backends",
        feature = "backend-rwkv",
        feature = "backend-mamba"
    ))]
    use crate::coders::CoderType;
    use std::fs;
    use std::path::Path;
    use std::sync::Arc;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn unique_temp_dir(label: &str) -> std::path::PathBuf {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system clock must be after unix epoch")
            .as_nanos();
        std::env::temp_dir().join(format!(
            "infotheory-spec-tests-{label}-{}-{nonce}",
            std::process::id()
        ))
    }

    fn sample_self_contained_rate_backend(
        kind: crate::runtime::RateBackendKind,
    ) -> Option<RateBackend> {
        crate::runtime::default_rate_backend_spec(kind)
    }

    #[cfg(feature = "aixi")]
    #[test]
    fn canonical_json_bytes_sort_object_keys_recursively_and_preserve_array_order() {
        let value = serde_json::json!({
            "b": 1,
            "a": {
                "z": [2, 1],
                "a": false
            },
            "c": [
                {
                    "b": 2,
                    "a": 1
                },
                null
            ]
        });
        let bytes = canonical_json_bytes(&value).expect("canonical JSON bytes");
        assert_eq!(
            std::str::from_utf8(&bytes).expect("canonical JSON is UTF-8"),
            r#"{"a":{"a":false,"z":[2,1]},"b":1,"c":[{"a":1,"b":2},null]}"#
        );
    }

    #[cfg(feature = "backend-rosa")]
    #[test]
    fn shorthand_rate_aliases_compile_to_identical_canonical_bytes() {
        let opts = RateBackendShorthandOptions::default();
        let rosa =
            compile_rate_backend_name_method("rosa", None, &opts).expect("compile rosa alias");
        let rosaplus = compile_rate_backend_name_method("rosaplus", None, &opts)
            .expect("compile rosaplus canonical");
        assert_eq!(
            rosa.canonical_bytes().as_slice(),
            rosaplus.canonical_bytes().as_slice()
        );
        assert_eq!(
            rosa.canonical_spec().to_canonical_json().unwrap(),
            rosaplus.canonical_spec().to_canonical_json().unwrap()
        );
    }

    #[test]
    fn shorthand_compression_requires_canonical_names() {
        let Some(default_rate_backend) = sample_enabled_leaf_rate_backend() else {
            return;
        };
        let opts = CompressionBackendShorthandOptions {
            default_rate_backend: Some(default_rate_backend),
            ..CompressionBackendShorthandOptions::default()
        };
        let canonical = compile_compression_backend_name_method("rate-ac", None, None, &opts)
            .expect("compile rate-ac canonical");
        let canonical_kind = canonical.canonical_spec().kind();
        assert_eq!(
            canonical_kind,
            crate::runtime::CompressionBackendKind::RateAc
        );

        let err = match compile_compression_backend_name_method("rate_ac", None, None, &opts) {
            Ok(_) => panic!("legacy alias must be rejected"),
            Err(err) => err,
        };
        let msg = err.to_string();
        assert!(
            msg.contains("unknown compression backend") || msg.contains("not available"),
            "unexpected error: {msg}"
        );
    }

    fn sample_enabled_leaf_rate_backend() -> Option<RateBackend> {
        crate::runtime::RATE_BACKEND_REGISTRY
            .iter()
            .filter(|descriptor| descriptor.enabled)
            .find_map(|descriptor| sample_self_contained_rate_backend(descriptor.kind))
    }

    fn sample_roundtrip_rate_backends() -> Vec<RateBackend> {
        let mut backends = Vec::new();
        let leaf = sample_enabled_leaf_rate_backend();

        for descriptor in crate::runtime::RATE_BACKEND_REGISTRY {
            if !descriptor.enabled {
                continue;
            }

            match descriptor.kind {
                crate::runtime::RateBackendKind::Mixture => {
                    if let Some(base) = leaf.clone() {
                        backends.push(RateBackend::Mixture {
                            spec: Arc::new(MixtureSpec::new(
                                MixtureKind::Bayes,
                                vec![MixtureExpertSpec {
                                    name: Some("leaf".to_string()),
                                    log_prior: 0.0,
                                    backend: base,
                                }],
                            )),
                        });
                    }
                }
                crate::runtime::RateBackendKind::Calibrated => {
                    if let Some(base) = leaf.clone() {
                        backends.push(RateBackend::Calibrated {
                            spec: Arc::new(CalibratedSpec {
                                base,
                                context: CalibrationContextKind::Text,
                                bins: 17,
                                learning_rate: 0.05,
                                bias_clip: 3.0,
                            }),
                        });
                    }
                }
                _ => {
                    if let Some(backend) = sample_self_contained_rate_backend(descriptor.kind) {
                        backends.push(backend);
                    }
                }
            }
        }

        backends
    }

    fn sample_roundtrip_compression_backends() -> Vec<CompressionBackend> {
        let mut backends = Vec::new();
        let leaf = sample_enabled_leaf_rate_backend();

        if cfg!(feature = "backend-zpaq") {
            backends.push(CompressionBackend::zpaq("5"));
        }

        if let Some(rate_backend) = leaf.clone() {
            backends.push(CompressionBackend::Rate {
                rate_backend: rate_backend.clone(),
                coder: crate::coders::CoderType::AC,
                framing: crate::compression::FramingMode::Raw,
            });
            backends.push(CompressionBackend::Rate {
                rate_backend,
                coder: crate::coders::CoderType::RANS,
                framing: crate::compression::FramingMode::Framed,
            });
        }

        #[cfg(feature = "backend-rwkv")]
        {
            let opts = CompressionBackendShorthandOptions {
                default_framing: crate::compression::FramingMode::Raw,
                ..Default::default()
            };
            let backend = parse_compression_backend_name_method(
                "rwkv7",
                Some(
                    "cfg:hidden=64,intermediate=64,layers=1,train=sgd,lr=0.01;policy:schedule=0..100:infer",
                ),
                None,
                &opts,
            )
            .expect("rwkv shorthand should parse");
            backends.push(backend);
        }

        backends
    }

    #[test]
    fn rate_backend_aliases_share_registry_resolution_and_feature_errors() {
        for descriptor in crate::runtime::RATE_BACKEND_REGISTRY {
            for alias in descriptor.aliases {
                if descriptor.enabled {
                    assert_eq!(
                        resolve_enabled_rate_backend_name(alias)
                            .expect("enabled alias should resolve"),
                        descriptor.canonical
                    );
                } else {
                    let err = resolve_enabled_rate_backend_name(alias)
                        .expect_err("disabled alias should fail");
                    let message = err.to_string();
                    assert!(message.contains(descriptor.canonical));
                    assert!(
                        message.contains(descriptor.feature.expect("disabled feature metadata"))
                    );
                }
            }
        }
    }

    #[test]
    fn compression_backend_aliases_share_registry_resolution_and_feature_errors() {
        for descriptor in crate::runtime::COMPRESSION_BACKEND_REGISTRY {
            for alias in descriptor.aliases {
                if descriptor.enabled {
                    assert_eq!(
                        resolve_enabled_compression_backend_name(alias)
                            .expect("enabled alias should resolve"),
                        descriptor.canonical
                    );
                } else {
                    let err = resolve_enabled_compression_backend_name(alias)
                        .expect_err("disabled alias should fail");
                    let message = err.to_string();
                    assert!(message.contains(descriptor.canonical));
                    assert!(
                        message.contains(descriptor.feature.expect("disabled feature metadata"))
                    );
                }
            }
        }
    }

    #[test]
    fn enabled_self_contained_rate_backends_validate_and_roundtrip() {
        for backend in sample_roundtrip_rate_backends() {
            crate::api::validate_rate_backend(&backend).expect("sample backend should validate");
            let json = rate_backend_to_json_value(&backend).expect("serialize backend");
            let reparsed = parse_rate_backend_json(&json, Path::new("."), MAX_MIXTURE_NESTING)
                .expect("parse backend");
            let roundtrip = rate_backend_to_json_value(&reparsed).expect("re-serialize backend");
            assert_eq!(json, roundtrip);
        }
    }

    #[test]
    fn enabled_compression_parse_paths_validate_and_roundtrip() {
        for backend in sample_roundtrip_compression_backends() {
            crate::api::validate_compression_backend(&backend)
                .expect("sample compression backend should validate");
            let json =
                compression_backend_to_json_value(&backend).expect("serialize compression backend");
            let reparsed = parse_compression_backend_json(
                &json,
                Path::new("."),
                None,
                crate::compression::FramingMode::Framed,
            )
            .expect("parse compression backend");
            let roundtrip = compression_backend_to_json_value(&reparsed)
                .expect("re-serialize compression backend");
            assert_eq!(json, roundtrip);
        }
    }

    #[cfg(feature = "all-backends")]
    #[test]
    fn rate_backend_json_roundtrip_handles_nested_specs() {
        let backend = RateBackend::Calibrated {
            spec: Arc::new(CalibratedSpec {
                base: RateBackend::Mixture {
                    spec: Arc::new(
                        MixtureSpec::new(
                            MixtureKind::Switching,
                            vec![
                                MixtureExpertSpec {
                                    name: Some("rosa".to_string()),
                                    log_prior: -0.2,
                                    backend: RateBackend::RosaPlus { max_order: 8 },
                                },
                                MixtureExpertSpec {
                                    name: Some("ctw".to_string()),
                                    log_prior: -1.4,
                                    backend: RateBackend::Ctw { depth: 12 },
                                },
                                MixtureExpertSpec {
                                    name: Some("particle".to_string()),
                                    log_prior: -2.0,
                                    backend: RateBackend::Particle {
                                        spec: Arc::new(ParticleSpec {
                                            num_particles: 4,
                                            ..ParticleSpec::default()
                                        }),
                                    },
                                },
                            ],
                        )
                        .with_schedule(MixtureScheduleMode::Theorem)
                        .with_alpha(0.25),
                    ),
                },
                context: CalibrationContextKind::TextRepeat,
                bins: 31,
                learning_rate: 0.05,
                bias_clip: 3.0,
            }),
        };

        let json = rate_backend_to_json_value(&backend).expect("serialize backend");
        let reparsed = parse_rate_backend_json(&json, Path::new("."), MAX_MIXTURE_NESTING)
            .expect("parse backend");
        let roundtrip = rate_backend_to_json_value(&reparsed).expect("re-serialize backend");
        assert_eq!(json, roundtrip);
    }

    #[cfg(feature = "all-backends")]
    #[test]
    fn compression_backend_json_roundtrip_handles_rate_wrappers() {
        let backend = CompressionBackend::Rate {
            rate_backend: RateBackend::Match {
                hash_bits: 18,
                min_len: 5,
                max_len: 128,
                base_mix: 0.03,
                confidence_scale: 0.8,
            },
            coder: CoderType::RANS,
            framing: crate::compression::FramingMode::Framed,
        };

        let json =
            compression_backend_to_json_value(&backend).expect("serialize compression backend");
        let reparsed = parse_compression_backend_json(
            &json,
            Path::new("."),
            None,
            crate::compression::FramingMode::Framed,
        )
        .expect("parse compression backend");
        let roundtrip =
            compression_backend_to_json_value(&reparsed).expect("re-serialize compression backend");
        assert_eq!(json, roundtrip);
    }

    #[cfg(feature = "all-backends")]
    #[test]
    fn parse_compression_backend_name_method_uses_shared_shorthand_defaults() {
        let rate_backend = RateBackend::Ppmd {
            order: 7,
            memory_mb: 32,
        };
        let opts = CompressionBackendShorthandOptions {
            default_rate_backend: Some(rate_backend.clone()),
            default_framing: crate::compression::FramingMode::Framed,
            ..Default::default()
        };

        let ac = parse_compression_backend_name_method("rate-ac", None, None, &opts)
            .expect("parse rate-ac");
        let rans = parse_compression_backend_name_method("rate-rans", None, None, &opts)
            .expect("parse rate-rans");
        let zpaq =
            parse_compression_backend_name_method("zpaq", None, None, &opts).expect("parse zpaq");

        match ac {
            CompressionBackend::Rate {
                rate_backend: RateBackend::Ppmd { order, memory_mb },
                coder,
                framing,
            } => {
                assert_eq!(order, 7);
                assert_eq!(memory_mb, 32);
                assert_eq!(coder, CoderType::AC);
                assert_eq!(framing, crate::compression::FramingMode::Framed);
            }
            _ => panic!("unexpected rate-ac backend"),
        }

        match rans {
            CompressionBackend::Rate { coder, .. } => {
                assert_eq!(coder, CoderType::RANS);
            }
            _ => panic!("unexpected rate-rans backend"),
        }

        match zpaq {
            CompressionBackend::Zpaq { method, .. } => assert_eq!(method.value(), "5"),
            _ => panic!("unexpected zpaq backend"),
        }
    }

    #[cfg(feature = "backend-rwkv")]
    #[test]
    fn parse_compression_backend_name_method_wraps_rwkv_cfg_methods_as_rate_backend() {
        let opts = CompressionBackendShorthandOptions {
            default_framing: crate::compression::FramingMode::Raw,
            ..Default::default()
        };

        let backend = parse_compression_backend_name_method(
            "rwkv7",
            Some(
                "cfg:hidden=64,intermediate=64,layers=1,train=sgd,lr=0.01;policy:schedule=0..100:infer",
            ),
            None,
            &opts,
        )
        .expect("parse rwkv cfg compression backend");

        match backend {
            CompressionBackend::Rate {
                rate_backend: RateBackend::Rwkv7Method { .. },
                coder,
                framing,
            } => {
                assert_eq!(coder, CoderType::AC);
                assert_eq!(framing, crate::compression::FramingMode::Raw);
            }
            _ => panic!("expected rate-coded RWKV backend"),
        }
    }

    #[cfg(feature = "backend-rwkv")]
    #[test]
    fn parse_compression_backend_json_wraps_rwkv_cfg_methods_as_rate_backend() {
        let json = serde_json::json!({
            "kind": "rwkv7",
            "method": "cfg:hidden=64,intermediate=64,layers=1,train=sgd,lr=0.01;policy:schedule=0..100:infer",
            "coder": "rans",
            "framing": "raw"
        });

        let backend = parse_compression_backend_json(
            &json,
            Path::new("."),
            None,
            crate::compression::FramingMode::Framed,
        )
        .expect("parse rwkv cfg compression backend json");

        match backend {
            CompressionBackend::Rate {
                rate_backend: RateBackend::Rwkv7Method { .. },
                coder,
                framing,
            } => {
                assert_eq!(coder, CoderType::RANS);
                assert_eq!(framing, crate::compression::FramingMode::Raw);
            }
            _ => panic!("expected rate-coded RWKV backend"),
        }
    }

    #[cfg(feature = "backend-rwkv")]
    #[test]
    fn parse_compression_backend_json_wraps_typed_rwkv_methods_as_rate_backend() {
        let json = serde_json::json!({
            "kind": "rwkv7",
            "method": {
                "kind": "online",
                "cfg": {
                    "hidden": 64,
                    "intermediate": 64,
                    "layers": 1,
                    "train_mode": "sgd",
                    "lr": 0.01
                },
                "policy": "schedule=0..100:infer"
            },
            "coder": "ac",
            "framing": "framed"
        });

        let backend = parse_compression_backend_json(
            &json,
            Path::new("."),
            None,
            crate::compression::FramingMode::Raw,
        )
        .expect("parse typed rwkv compression backend json");

        match backend {
            CompressionBackend::Rate {
                rate_backend: RateBackend::Rwkv7Method { .. },
                coder,
                framing,
            } => {
                assert_eq!(coder, CoderType::AC);
                assert_eq!(framing, crate::compression::FramingMode::Framed);
            }
            _ => panic!("expected rate-coded RWKV backend"),
        }
    }

    #[cfg(feature = "backend-rwkv")]
    #[test]
    fn normalize_rwkv_method_for_base_dir_decodes_reserved_file_escapes_once() {
        let base_dir = Path::new("/tmp/spec-base");
        let method = "file:weights/model%3Bv1%25done.safetensors";
        let normalized = canonicalize_explicit_file_method(base_dir, method, "rwkv")
            .expect("canonicalize rwkv method")
            .expect("file method");
        assert_eq!(
            normalized,
            "file:/tmp/spec-base/weights/model%3Bv1%25done.safetensors"
        );
    }

    #[cfg(all(feature = "backend-rwkv", windows))]
    #[test]
    fn normalize_rwkv_method_for_base_dir_renders_forward_slashes_on_windows() {
        let base_dir = Path::new(r"C:\tmp\spec-base");
        let method = "file:weights/model%3Bv1%25done.safetensors";
        let normalized = canonicalize_explicit_file_method(base_dir, method, "rwkv")
            .expect("canonicalize rwkv method")
            .expect("file method");
        assert_eq!(
            normalized,
            "file:C:/tmp/spec-base/weights/model%3Bv1%25done.safetensors"
        );
    }

    #[cfg(feature = "backend-rwkv")]
    #[test]
    fn normalize_rwkv_method_for_base_dir_rejects_ambiguous_file_suffixes() {
        let err = normalize_rwkv_method_for_base_dir(
            Path::new("/tmp/spec-base"),
            "file:weights/model;polciy:infer",
        )
        .unwrap_err();
        assert!(
            err.message
                .contains("ambiguous file method segment ';polciy:'")
        );
    }

    #[cfg(feature = "backend-mamba")]
    #[test]
    fn parse_compression_backend_json_wraps_mamba_cfg_rate_backend() {
        let json = serde_json::json!({
            "kind": "rate-ac",
            "framing": "raw",
            "rate_backend": {
                "kind": "mamba",
                "method": "cfg:hidden=64,layers=1,intermediate=96,state=16,conv=4,dt_rank=16,seed=26,train=none,lr=0.0,stride=1;policy:schedule=0..100:infer"
            }
        });

        let backend = parse_compression_backend_json(
            &json,
            Path::new("."),
            None,
            crate::compression::FramingMode::Framed,
        )
        .expect("parse mamba rate-coded compression backend json");

        match backend {
            CompressionBackend::Rate {
                rate_backend: RateBackend::MambaMethod { .. },
                coder,
                framing,
            } => {
                assert_eq!(coder, CoderType::AC);
                assert_eq!(framing, crate::compression::FramingMode::Raw);
            }
            _ => panic!("expected rate-coded mamba backend"),
        }
    }

    #[cfg(feature = "backend-zpaq")]
    #[test]
    fn canonical_json_emits_typed_zpaq_method_objects() {
        let rate = serde_json::from_str::<serde_json::Value>(
            &RateBackend::Zpaq {
                method: crate::api::ZpaqMethodSpec::literal("5"),
            }
            .to_canonical_json()
            .expect("rate json"),
        )
        .expect("valid rate json");
        assert_eq!(rate["kind"], "zpaq");
        assert_eq!(rate["method"]["kind"], "literal");
        assert_eq!(rate["method"]["value"], "5");

        let compression = serde_json::from_str::<serde_json::Value>(
            &CompressionBackend::zpaq("5")
                .to_canonical_json()
                .expect("compression json"),
        )
        .expect("valid compression json");
        assert_eq!(compression["kind"], "zpaq");
        assert_eq!(compression["method"]["kind"], "literal");
        assert_eq!(compression["method"]["value"], "5");
    }

    #[cfg(feature = "backend-zpaq")]
    #[test]
    fn parse_typed_zpaq_method_objects_are_strictly_validated() {
        let err = parse_rate_backend_json(
            &serde_json::json!({
                "kind": "zpaq",
                "method": {"kind": "literal"}
            }),
            Path::new("."),
            MAX_MIXTURE_NESTING,
        )
        .err()
        .expect("missing typed zpaq value must fail");
        assert!(err.message.contains("method.value"), "{err}");

        let err = parse_rate_backend_json(
            &serde_json::json!({
                "kind": "zpaq",
                "method": {"value": "2"}
            }),
            Path::new("."),
            MAX_MIXTURE_NESTING,
        )
        .err()
        .expect("missing typed zpaq kind must fail");
        assert!(err.message.contains("method.kind"), "{err}");

        let err = parse_compression_backend_json(
            &serde_json::json!({
                "kind": "zpaq",
                "method": {"kind": "nonliteral", "value": "5"}
            }),
            Path::new("."),
            None,
            crate::compression::FramingMode::Framed,
        )
        .err()
        .expect("unknown typed zpaq kind must fail");
        assert!(err.message.contains("unknown zpaq method kind"), "{err}");
    }

    #[cfg(feature = "backend-zpaq")]
    #[test]
    fn parse_zpaq_method_requires_typed_object_form() {
        let err = parse_rate_backend_json(
            &serde_json::json!({"kind": "zpaq", "method": "2"}),
            Path::new("."),
            MAX_MIXTURE_NESTING,
        )
        .err()
        .expect("legacy zpaq method string must be rejected");
        assert!(
            err.message
                .contains("zpaq method must use object form {'kind':'literal','value':'...'}"),
            "{err}"
        );

        let compression = parse_compression_backend_json(
            &serde_json::json!({"kind": "zpaq"}),
            Path::new("."),
            None,
            crate::compression::FramingMode::Framed,
        )
        .expect("missing method should use default compression method");
        assert!(
            matches!(compression, CompressionBackend::Zpaq { method, .. } if method.value() == "5")
        );
    }

    #[cfg(feature = "backend-rwkv")]
    #[test]
    fn canonical_json_emits_typed_rwkv_method_objects() {
        let value = serde_json::from_str::<serde_json::Value>(
            &RateBackend::Rwkv7Method {
                method: crate::rwkvzip::parse_method_spec("cfg:hidden=64,intermediate=64,layers=1")
                    .expect("rwkv method spec"),
            }
            .to_canonical_json()
            .expect("rate json"),
        )
        .expect("valid rate json");

        assert_eq!(value["kind"], "rwkv7");
        assert_eq!(value["method"]["kind"], "online");
        assert_eq!(value["method"]["cfg"]["hidden"], 64);
        assert_eq!(value["method"]["cfg"]["layers"], 1);
        assert_eq!(value["method"]["cfg"]["intermediate"], 64);
    }

    #[cfg(feature = "backend-mamba")]
    #[test]
    fn canonical_json_emits_typed_mamba_method_objects() {
        let value = serde_json::from_str::<serde_json::Value>(
            &RateBackend::MambaMethod {
                method: crate::mambazip::parse_method_spec(
                    "cfg:hidden=64,layers=1,intermediate=96,state=16,conv=4,dt_rank=16",
                )
                .expect("mamba method spec"),
            }
            .to_canonical_json()
            .expect("rate json"),
        )
        .expect("valid rate json");

        assert_eq!(value["kind"], "mamba");
        assert_eq!(value["method"]["kind"], "online");
        assert_eq!(value["method"]["cfg"]["hidden"], 64);
        assert_eq!(value["method"]["cfg"]["layers"], 1);
        assert_eq!(value["method"]["cfg"]["intermediate"], 96);
    }

    #[test]
    fn helper_parsers_and_name_renderers_use_canonical_forms() {
        assert_eq!(
            resolve_spec_path(Path::new("/tmp/base"), "child/spec.json"),
            Path::new("/tmp/base").join("child/spec.json")
        );
        assert_eq!(
            resolve_spec_path(Path::new("/tmp/base"), Path::new("/tmp/absolute.json")),
            Path::new("/tmp/absolute.json")
        );

        assert_eq!(
            parse_calibration_context_kind(None).expect("default calibration context"),
            CalibrationContextKind::Text
        );
        assert_eq!(
            parse_calibration_context_kind(Some("repeat")).expect("repeat context"),
            CalibrationContextKind::Repeat
        );
        assert!(parse_calibration_context_kind(Some("legacy")).is_err());

        assert_eq!(
            parse_mixture_kind("switching").expect("switching"),
            MixtureKind::Switching
        );
        assert_eq!(
            parse_mixture_schedule("theorem").expect("theorem schedule"),
            MixtureScheduleMode::Theorem
        );
        assert_eq!(mixture_kind_name(MixtureKind::Neural), "neural");
        assert_eq!(
            mixture_schedule_name(MixtureScheduleMode::Default),
            "default"
        );
        assert_eq!(
            calibration_context_kind_name(CalibrationContextKind::Text),
            "text"
        );

        assert_eq!(
            parse_framing_mode(None).expect("default framing"),
            crate::compression::FramingMode::Framed
        );
        assert_eq!(
            parse_framing_mode(Some("raw")).expect("raw framing"),
            crate::compression::FramingMode::Raw
        );
        assert!(parse_framing_mode(Some("legacy")).is_err());
        assert_eq!(
            framing_mode_name(crate::compression::FramingMode::Framed),
            "framed"
        );

        let zpaq_json = zpaq_method_to_json_value(&crate::api::ZpaqMethodSpec::literal("3"));
        assert_eq!(zpaq_json["kind"], "literal");
        assert_eq!(
            parse_zpaq_method_json_value(&zpaq_json, "5")
                .expect("typed zpaq method")
                .value(),
            "3"
        );
    }

    #[test]
    fn load_json_value_from_path_reports_read_and_parse_context() {
        let dir = unique_temp_dir("load-json");
        fs::create_dir_all(&dir).expect("create temp dir");
        let valid = dir.join("valid.json");
        let invalid = dir.join("invalid.json");
        fs::write(&valid, br#"{ "alpha": 1 }"#).expect("write valid json");
        fs::write(&invalid, b"{ invalid").expect("write invalid json");

        let (value, full) =
            load_json_value_from_path(&dir, "valid.json", "spec fixture").expect("load valid");
        assert_eq!(value["alpha"], 1);
        assert_eq!(full, valid);

        let err = load_json_value_from_path(&dir, "missing.json", "spec fixture")
            .expect_err("missing json must fail");
        assert!(err.to_string().contains("failed to read spec fixture"));
        assert!(err.to_string().contains("missing.json"));

        let err = load_json_value_from_path(&dir, "invalid.json", "spec fixture")
            .expect_err("invalid json must fail");
        assert!(err.to_string().contains("invalid spec fixture JSON"));
        assert!(err.to_string().contains("invalid.json"));

        let _ = fs::remove_file(valid);
        let _ = fs::remove_file(invalid);
        let _ = fs::remove_dir(dir);
    }

    #[cfg(feature = "backend-ctw")]
    #[test]
    fn shorthand_rate_backend_parsers_load_file_backed_specs_and_respect_particle_default_policy() {
        let dir = unique_temp_dir("backend-files");
        fs::create_dir_all(&dir).expect("create temp dir");

        let mixture_path = dir.join("mixture.json");
        let calibrated_path = dir.join("calibrated.json");
        let particle_path = dir.join("particle.json");

        let mixture = MixtureSpec::new(
            MixtureKind::Bayes,
            vec![MixtureExpertSpec {
                name: Some("ctw".to_string()),
                log_prior: 0.0,
                backend: RateBackend::Ctw { depth: 4 },
            }],
        );
        let calibrated = CalibratedSpec {
            base: RateBackend::Ctw { depth: 5 },
            context: CalibrationContextKind::Text,
            bins: 17,
            learning_rate: 0.05,
            bias_clip: 3.0,
        };
        let particle = ParticleSpec::default();

        fs::write(
            &mixture_path,
            mixture.to_canonical_json().expect("mixture canonical json"),
        )
        .expect("write mixture spec");
        fs::write(
            &calibrated_path,
            calibrated
                .to_canonical_json()
                .expect("calibrated canonical json"),
        )
        .expect("write calibrated spec");
        fs::write(
            &particle_path,
            particle
                .to_canonical_json()
                .expect("particle canonical json"),
        )
        .expect("write particle spec");

        let options = RateBackendShorthandOptions {
            base_dir: dir.clone(),
            particle_default_if_missing_method: false,
            ..RateBackendShorthandOptions::default()
        };

        let mixture_result =
            parse_rate_backend_name_method("mixture", Some("mixture.json"), &options);
        #[cfg(feature = "backend-mixture")]
        {
            let mixture_backend = mixture_result.expect("mixture shorthand should load JSON file");
            assert!(matches!(mixture_backend, RateBackend::Mixture { .. }));
        }
        #[cfg(not(feature = "backend-mixture"))]
        {
            let err = match mixture_result {
                Ok(_) => {
                    panic!("mixture shorthand should fail when feature is disabled")
                }
                Err(err) => err,
            };
            assert!(
                err.to_string()
                    .contains("backend 'mixture' requires infotheory feature 'backend-mixture'"),
                "unexpected mixture error: {err}"
            );
        }

        let particle_result =
            parse_rate_backend_name_method("particle", Some("particle.json"), &options);
        #[cfg(feature = "backend-particle")]
        {
            let particle_backend =
                particle_result.expect("particle shorthand should load JSON file");
            assert!(matches!(particle_backend, RateBackend::Particle { .. }));
        }
        #[cfg(not(feature = "backend-particle"))]
        {
            let err = match particle_result {
                Ok(_) => {
                    panic!("particle shorthand should fail when feature is disabled")
                }
                Err(err) => err,
            };
            assert!(
                err.to_string()
                    .contains("backend 'particle' requires infotheory feature 'backend-particle'"),
                "unexpected particle error: {err}"
            );
        }

        let calibrated_result =
            parse_rate_backend_name_method("calibrated", Some("calibrated.json"), &options);
        #[cfg(feature = "backend-calibrated")]
        {
            let calibrated_backend =
                calibrated_result.expect("calibrated shorthand should load JSON file");
            assert!(matches!(calibrated_backend, RateBackend::Calibrated { .. }));
        }
        #[cfg(not(feature = "backend-calibrated"))]
        {
            let err = match calibrated_result {
                Ok(_) => {
                    panic!("calibrated shorthand should fail when feature is disabled")
                }
                Err(err) => err,
            };
            assert!(
                err.to_string().contains(
                    "backend 'calibrated' requires infotheory feature 'backend-calibrated'"
                ),
                "unexpected calibrated error: {err}"
            );
        }

        let particle_missing_method_result =
            parse_rate_backend_name_method("particle", None, &options);
        #[cfg(feature = "backend-particle")]
        {
            let err = match particle_missing_method_result {
                Ok(_) => {
                    panic!("particle shorthand should require path when disabled")
                }
                Err(err) => err,
            };
            assert!(
                err.to_string()
                    .contains("particle backend requires a path to a ParticleSpec JSON file"),
                "unexpected particle missing-method error: {err}"
            );
        }
        #[cfg(not(feature = "backend-particle"))]
        {
            let err = match particle_missing_method_result {
                Ok(_) => {
                    panic!("particle shorthand without feature should report capability error")
                }
                Err(err) => err,
            };
            assert!(
                err.to_string()
                    .contains("backend 'particle' requires infotheory feature 'backend-particle'"),
                "unexpected particle feature error: {err}"
            );
        }

        let _ = fs::remove_file(mixture_path);
        let _ = fs::remove_file(calibrated_path);
        let _ = fs::remove_file(particle_path);
        let _ = fs::remove_dir(dir);
    }

    #[test]
    fn parse_particle_spec_value_preserves_defaults_and_rejects_mixture_shapes() {
        let defaults = ParticleSpec::default();
        let parsed = parse_particle_spec_value(&serde_json::json!({
            "num_particles": defaults.num_particles + 7,
            "deterministic": !defaults.deterministic,
        }))
        .expect("particle subset should parse with defaults");
        assert_eq!(parsed.num_particles, defaults.num_particles + 7);
        assert_eq!(parsed.context_window, defaults.context_window);
        assert_eq!(parsed.seed, defaults.seed);
        assert_eq!(parsed.deterministic, !defaults.deterministic);

        let err = parse_particle_spec_value(&serde_json::json!({
            "kind": "mixture",
            "num_particles": 8,
        }))
        .expect_err("mixture-looking kind must be rejected");
        assert!(
            err.to_string()
                .contains("looks like a mixture spec (kind='mixture')"),
            "unexpected error: {err}"
        );

        let err = parse_particle_spec_value(&serde_json::json!({
            "experts": [],
        }))
        .expect_err("mixture-shaped object must be rejected");
        assert!(
            err.to_string()
                .contains("looks like a mixture spec (found 'experts')"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn fac_ctw_default_projections_remain_explicit_and_consistent() {
        let shorthand = RateBackendShorthandOptions::default();
        assert_eq!(
            shorthand.fac_ctw_num_percept_bits,
            crate::rate_defaults::FAC_CTW_DEFAULT_NUM_PERCEPT_BITS
        );
        assert_eq!(
            shorthand.fac_ctw_encoding_bits,
            crate::rate_defaults::JSON_DEFAULT_FAC_CTW_ENCODING_BITS
        );

        #[cfg(feature = "backend-ctw")]
        {
            let parsed = parse_rate_backend_json(
                &serde_json::json!({"kind":"fac-ctw"}),
                Path::new("."),
                MAX_MIXTURE_NESTING,
            )
            .expect("fac-ctw json parse");
            let parsed_for_compile = parsed.clone();
            match parsed {
                RateBackend::FacCtw {
                    base_depth,
                    num_percept_bits,
                    encoding_bits,
                    msb_first,
                } => {
                    assert_eq!(
                        base_depth,
                        crate::rate_defaults::JSON_DEFAULT_FAC_CTW_BASE_DEPTH
                    );
                    assert_eq!(
                        encoding_bits,
                        crate::rate_defaults::JSON_DEFAULT_FAC_CTW_ENCODING_BITS
                    );
                    assert_eq!(num_percept_bits, encoding_bits);
                    assert_eq!(msb_first, None);
                }
                _ => panic!("expected fac-ctw backend"),
            }

            let explicit_lsb = parse_rate_backend_json(
                &serde_json::json!({
                    "kind": "fac-ctw",
                    "base_depth": 9,
                    "encoding_bits": 8,
                    "num_percept_bits": 8,
                    "msb_first": false,
                }),
                Path::new("."),
                MAX_MIXTURE_NESTING,
            )
            .expect("fac-ctw explicit LSB json parse");
            let compiled_lsb = explicit_lsb
                .compile()
                .expect("fac-ctw explicit LSB compiles");
            match compiled_lsb.plan() {
                crate::spec::core::RateBackendPlan::FacCtw { msb_first, .. } => {
                    assert!(!*msb_first, "explicit msb_first=false must survive compile");
                }
                _ => panic!("expected fac-ctw compiled plan"),
            }

            let compiled_default = parsed_for_compile
                .compile()
                .expect("fac-ctw default compiles");
            match compiled_default.plan() {
                crate::spec::core::RateBackendPlan::FacCtw {
                    encoding_bits,
                    msb_first,
                    ..
                } => {
                    assert_eq!(*encoding_bits, 8);
                    assert!(
                        *msb_first,
                        "omitted msb_first defaults to MSB-first for byte-width FacCtw"
                    );
                }
                _ => panic!("expected fac-ctw compiled plan"),
            }

            let invalid_width = parse_rate_backend_json(
                &serde_json::json!({
                    "kind": "fac-ctw",
                    "base_depth": 9,
                    "encoding_bits": 9,
                    "num_percept_bits": 9,
                }),
                Path::new("."),
                MAX_MIXTURE_NESTING,
            )
            .expect("fac-ctw invalid-width json parses before semantic compile validation");
            let err = match invalid_width.compile() {
                Ok(_) => panic!("fac-ctw encoding_bits outside 1..=8 must be rejected"),
                Err(err) => err,
            };
            assert!(
                err.to_string()
                    .contains("fac-ctw encoding_bits must be in 1..=8, got 9"),
                "unexpected fac-ctw encoding_bits error: {err}"
            );
        }

        #[cfg(not(feature = "backend-ctw"))]
        {
            let err = match parse_rate_backend_json(
                &serde_json::json!({"kind":"fac-ctw"}),
                Path::new("."),
                MAX_MIXTURE_NESTING,
            ) {
                Ok(_) => panic!("disabled fac-ctw backend must report a feature error"),
                Err(err) => err,
            };
            assert!(
                err.to_string()
                    .contains("backend 'fac-ctw' requires infotheory feature 'backend-ctw'"),
                "unexpected fac-ctw feature error: {err}"
            );
        }

        let runtime_default =
            crate::runtime::default_rate_backend_spec(crate::runtime::RateBackendKind::FacCtw)
                .expect("runtime fac-ctw default");
        match runtime_default {
            RateBackend::FacCtw {
                base_depth,
                num_percept_bits,
                encoding_bits,
                msb_first,
            } => {
                assert_eq!(base_depth, 8);
                assert_eq!(
                    encoding_bits,
                    crate::rate_defaults::JSON_DEFAULT_FAC_CTW_ENCODING_BITS
                );
                assert_eq!(
                    num_percept_bits,
                    crate::rate_defaults::FAC_CTW_DEFAULT_NUM_PERCEPT_BITS
                );
                assert_eq!(msb_first, None);
            }
            _ => panic!("expected runtime fac-ctw default backend"),
        }
        let fac_ctw_default_json = rate_backend_to_json_value(&runtime_default)
            .expect("serialize runtime fac-ctw default backend");
        assert!(
            fac_ctw_default_json.get("msb_first").is_none(),
            "canonical fac-ctw json must omit msb_first when unset"
        );

        #[cfg(feature = "backend-ctw")]
        {
            let msb_shorthand = RateBackendShorthandOptions {
                fac_ctw_msb_first: Some(true),
                ..RateBackendShorthandOptions::default()
            };
            let parsed_msb = parse_rate_backend_name_method("fac-ctw", Some("9"), &msb_shorthand)
                .expect("fac-ctw shorthand with msb_first");
            match parsed_msb {
                RateBackend::FacCtw { msb_first, .. } => {
                    assert_eq!(msb_first, Some(true));
                }
                _ => panic!("expected fac-ctw backend"),
            }
            let compiled_msb = parsed_msb.compile().expect("fac-ctw msb compiles");
            match compiled_msb.plan() {
                crate::spec::core::RateBackendPlan::FacCtw { msb_first, .. } => {
                    assert!(*msb_first, "shorthand msb_first=true must survive compile");
                }
                _ => panic!("expected fac-ctw compiled plan"),
            }

            let lsb_shorthand = RateBackendShorthandOptions {
                fac_ctw_msb_first: Some(false),
                ..RateBackendShorthandOptions::default()
            };
            let parsed_lsb = parse_rate_backend_name_method("fac-ctw", Some("9"), &lsb_shorthand)
                .expect("fac-ctw shorthand with lsb_first");
            match parsed_lsb.compile().expect("fac-ctw lsb compiles").plan() {
                crate::spec::core::RateBackendPlan::FacCtw { msb_first, .. } => {
                    assert!(
                        !*msb_first,
                        "shorthand msb_first=false must survive compile"
                    );
                }
                _ => panic!("expected fac-ctw compiled plan"),
            }

            let factory_json = crate::rate_defaults::fac_ctw_spec_json(9, 8, 8, Some(false));
            assert_eq!(factory_json["msb_first"], serde_json::json!(false));
            let factory_default = crate::rate_defaults::fac_ctw_spec_json(9, 8, 8, None);
            assert!(
                factory_default.get("msb_first").is_none(),
                "factory omits msb_first when None"
            );
        }
    }

    #[cfg(feature = "all-backends")]
    #[test]
    fn load_sidecar_specs_report_read_and_json_error_context() {
        let dir = unique_temp_dir("load-sidecar-errors");
        fs::create_dir_all(&dir).expect("create temp dir");

        let particle_invalid = dir.join("particle-invalid.json");
        let calibrated_invalid = dir.join("calibrated-invalid.json");
        let expert_invalid = dir.join("expert-invalid.json");
        fs::write(&particle_invalid, b"{ invalid").expect("write invalid particle json");
        fs::write(&calibrated_invalid, b"{ invalid").expect("write invalid calibrated json");
        fs::write(&expert_invalid, b"{ invalid").expect("write invalid expert json");

        let missing_particle = dir.join("particle-missing.json");
        let err = match load_particle_spec(missing_particle.to_str().expect("utf8 path")) {
            Ok(_) => panic!("missing particle spec should fail"),
            Err(err) => err,
        };
        assert!(
            err.to_string().contains("failed to read particle spec"),
            "{err}"
        );

        let err = match load_particle_spec(particle_invalid.to_str().expect("utf8 path")) {
            Ok(_) => panic!("invalid particle spec JSON should fail"),
            Err(err) => err,
        };
        assert!(
            err.to_string().contains("invalid particle spec JSON"),
            "{err}"
        );

        let missing_calibrated = dir.join("calibrated-missing.json");
        let err = match load_calibrated_spec(missing_calibrated.to_str().expect("utf8 path")) {
            Ok(_) => panic!("missing calibrated spec should fail"),
            Err(err) => err,
        };
        assert!(
            err.to_string().contains("failed to read calibrated spec"),
            "{err}"
        );

        let err = match load_calibrated_spec(calibrated_invalid.to_str().expect("utf8 path")) {
            Ok(_) => panic!("invalid calibrated spec JSON should fail"),
            Err(err) => err,
        };
        assert!(
            err.to_string().contains("invalid calibrated spec JSON"),
            "{err}"
        );

        let missing_expert = dir.join("expert-missing.json");
        let err = match load_expert_spec(missing_expert.to_str().expect("utf8 path")) {
            Ok(_) => panic!("missing expert spec should fail"),
            Err(err) => err,
        };
        assert!(
            err.to_string().contains("failed to read expert spec"),
            "{err}"
        );

        let err = match load_expert_spec(expert_invalid.to_str().expect("utf8 path")) {
            Ok(_) => panic!("invalid expert spec JSON should fail"),
            Err(err) => err,
        };
        assert!(
            err.to_string().contains("invalid expert spec JSON"),
            "{err}"
        );

        let _ = fs::remove_file(particle_invalid);
        let _ = fs::remove_file(calibrated_invalid);
        let _ = fs::remove_file(expert_invalid);
        let _ = fs::remove_dir(dir);
    }

    #[cfg(feature = "all-backends")]
    #[test]
    fn shorthand_rate_backend_parser_covers_leaf_defaults_and_model_path_contracts() {
        let options = RateBackendShorthandOptions::default();

        let sparse_match = parse_rate_backend_name_method("sparse-match", None, &options)
            .expect("sparse-match shorthand should parse");
        match sparse_match {
            RateBackend::SparseMatch {
                hash_bits,
                min_len,
                max_len,
                gap_min,
                gap_max,
                base_mix,
                confidence_scale,
            } => {
                assert_eq!(hash_bits, 19);
                assert_eq!(min_len, 3);
                assert_eq!(max_len, 64);
                assert_eq!(gap_min, 1);
                assert_eq!(gap_max, 2);
                assert!((base_mix - 0.05).abs() < f64::EPSILON);
                assert!((confidence_scale - 1.0).abs() < f64::EPSILON);
            }
            _ => panic!("expected sparse-match backend"),
        }

        let fac_ctw = parse_rate_backend_name_method("fac-ctw", Some("11"), &options)
            .expect("fac-ctw shorthand should parse");
        match fac_ctw {
            RateBackend::FacCtw {
                base_depth,
                num_percept_bits,
                encoding_bits,
                msb_first,
            } => {
                assert_eq!(base_depth, 11);
                assert_eq!(num_percept_bits, options.fac_ctw_num_percept_bits);
                assert_eq!(encoding_bits, options.fac_ctw_encoding_bits);
                assert_eq!(msb_first, None);
            }
            _ => panic!("expected fac-ctw backend"),
        }

        let zpaq = parse_rate_backend_name_method("zpaq", None, &options)
            .expect("zpaq shorthand should parse");
        assert!(
            matches!(zpaq, RateBackend::Zpaq { method } if method.value() == options.zpaq_method)
        );

        let particle = parse_rate_backend_name_method("particle", None, &options)
            .expect("particle shorthand should use default spec");
        assert!(matches!(particle, RateBackend::Particle { .. }));

        let mixture_err = match parse_rate_backend_name_method("mixture", None, &options) {
            Ok(_) => panic!("mixture shorthand without path should fail"),
            Err(err) => err,
        };
        assert!(
            mixture_err
                .to_string()
                .contains("mixture backend requires a path to a MixtureSpec JSON file"),
            "{mixture_err}"
        );

        let calibrated_err = match parse_rate_backend_name_method("calibrated", None, &options) {
            Ok(_) => panic!("calibrated shorthand without path should fail"),
            Err(err) => err,
        };
        assert!(
            calibrated_err
                .to_string()
                .contains("calibrated backend requires a path to a CalibratedSpec JSON file"),
            "{calibrated_err}"
        );

        let mamba_err = match parse_rate_backend_name_method("mamba", None, &options) {
            Ok(_) => panic!("mamba shorthand without method/model path should fail"),
            Err(err) => err,
        };
        assert!(
            mamba_err
                .to_string()
                .contains("mamba backend requires method string"),
            "{mamba_err}"
        );

        let rwkv_err = match parse_rate_backend_name_method("rwkv7", None, &options) {
            Ok(_) => panic!("rwkv shorthand without method/model path should fail"),
            Err(err) => err,
        };
        assert!(
            rwkv_err
                .to_string()
                .contains("rwkv backend requires method string"),
            "{rwkv_err}"
        );
    }

    #[cfg(feature = "backend-ctw")]
    #[test]
    fn parse_rate_backend_json_loads_nested_spec_paths_relative_to_base_dir() {
        let dir = unique_temp_dir("nested-backend-specs");
        fs::create_dir_all(&dir).expect("create temp dir");

        let nested_dir = dir.join("nested");
        fs::create_dir_all(&nested_dir).expect("create nested dir");

        let particle_path = nested_dir.join("particle.json");
        let calibrated_path = nested_dir.join("calibrated.json");
        let mixture_path = nested_dir.join("mixture.json");

        let particle = ParticleSpec {
            num_particles: 11,
            context_window: 19,
            ..ParticleSpec::default()
        };
        let calibrated = CalibratedSpec {
            base: RateBackend::Ctw { depth: 9 },
            context: CalibrationContextKind::Repeat,
            bins: 21,
            learning_rate: 0.03,
            bias_clip: 2.5,
        };
        let mixture = MixtureSpec::new(
            MixtureKind::Bayes,
            vec![MixtureExpertSpec {
                name: Some("ctw-nine".to_string()),
                log_prior: -0.5,
                backend: RateBackend::Ctw { depth: 9 },
            }],
        );

        fs::write(
            &particle_path,
            particle.to_canonical_json().expect("particle json"),
        )
        .expect("write particle spec");
        fs::write(
            &calibrated_path,
            calibrated.to_canonical_json().expect("calibrated json"),
        )
        .expect("write calibrated spec");
        fs::write(
            &mixture_path,
            mixture.to_canonical_json().expect("mixture json"),
        )
        .expect("write mixture spec");

        let particle_result = parse_rate_backend_json(
            &serde_json::json!({
                "kind": "particle",
                "spec_path": "nested/particle.json",
            }),
            &dir,
            MAX_MIXTURE_NESTING,
        );
        #[cfg(feature = "backend-particle")]
        {
            let particle_backend =
                particle_result.expect("particle spec_path should resolve relative to base dir");
            match particle_backend {
                RateBackend::Particle { spec } => {
                    assert_eq!(spec.num_particles, 11);
                    assert_eq!(spec.context_window, 19);
                }
                _ => panic!("expected particle backend"),
            }
        }
        #[cfg(not(feature = "backend-particle"))]
        {
            let err = match particle_result {
                Ok(_) => panic!("particle backend should report feature gate in this slice"),
                Err(err) => err,
            };
            assert!(
                err.to_string()
                    .contains("backend 'particle' requires infotheory feature 'backend-particle'"),
                "unexpected particle error: {err}"
            );
        }

        let calibrated_result = parse_rate_backend_json(
            &serde_json::json!({
                "kind": "calibrated",
                "spec_path": "nested/calibrated.json",
            }),
            &dir,
            MAX_MIXTURE_NESTING,
        );
        #[cfg(feature = "backend-calibrated")]
        {
            let calibrated_backend = calibrated_result
                .expect("calibrated spec_path should resolve relative to base dir");
            match calibrated_backend {
                RateBackend::Calibrated { spec } => {
                    assert_eq!(spec.bins, 21);
                    assert_eq!(spec.context, CalibrationContextKind::Repeat);
                    match spec.base {
                        RateBackend::Ctw { depth } => assert_eq!(depth, 9),
                        _ => panic!("expected ctw base backend"),
                    }
                }
                _ => panic!("expected calibrated backend"),
            }
        }
        #[cfg(not(feature = "backend-calibrated"))]
        {
            let err = match calibrated_result {
                Ok(_) => panic!("calibrated backend should report feature gate in this slice"),
                Err(err) => err,
            };
            assert!(
                err.to_string().contains(
                    "backend 'calibrated' requires infotheory feature 'backend-calibrated'"
                ),
                "unexpected calibrated error: {err}"
            );
        }

        let mixture_result = parse_rate_backend_json(
            &serde_json::json!({
                "kind": "mixture",
                "spec_path": "nested/mixture.json",
            }),
            &dir,
            MAX_MIXTURE_NESTING,
        );
        #[cfg(feature = "backend-mixture")]
        {
            let mixture_backend =
                mixture_result.expect("mixture spec_path should resolve relative to base dir");
            match mixture_backend {
                RateBackend::Mixture { spec } => {
                    assert_eq!(spec.kind, MixtureKind::Bayes);
                    assert_eq!(spec.experts.len(), 1);
                    assert_eq!(spec.experts[0].name.as_deref(), Some("ctw-nine"));
                }
                _ => panic!("expected mixture backend"),
            }
        }
        #[cfg(not(feature = "backend-mixture"))]
        {
            let err = match mixture_result {
                Ok(_) => panic!("mixture backend should report feature gate in this slice"),
                Err(err) => err,
            };
            assert!(
                err.to_string()
                    .contains("backend 'mixture' requires infotheory feature 'backend-mixture'"),
                "unexpected mixture error: {err}"
            );
        }

        let _ = fs::remove_file(particle_path);
        let _ = fs::remove_file(calibrated_path);
        let _ = fs::remove_file(mixture_path);
        let _ = fs::remove_dir(nested_dir);
        let _ = fs::remove_dir(dir);
    }

    #[cfg(feature = "backend-rwkv")]
    #[test]
    fn parse_rwkv7_compression_backend_json_requires_method_or_model_path() {
        let err = match parse_compression_backend_json(
            &serde_json::json!({
                "kind": "rwkv7",
            }),
            Path::new("."),
            None,
            crate::compression::FramingMode::Framed,
        ) {
            Ok(_) => panic!("rwkv7 compression json without method/model_path must fail"),
            Err(err) => err,
        };
        assert!(
            err.to_string()
                .contains("rwkv7 compression backend requires 'method' or 'model_path'"),
            "unexpected error: {err}"
        );
    }

    #[cfg(feature = "backend-rwkv")]
    #[test]
    fn parse_rwkv7_compression_backend_json_lowers_typed_method_and_coder() {
        let backend = parse_compression_backend_json(
            &serde_json::json!({
                "kind": "rwkv7",
                "coder": "rans",
                "framing": "raw",
                "method": {
                    "kind": "online",
                    "cfg": {
                        "hidden": 64,
                        "layers": 1,
                        "intermediate": 64,
                        "decay_rank": 8,
                        "a_rank": 8,
                        "v_rank": 8,
                        "g_rank": 8,
                        "seed": 5,
                        "train": "none",
                        "lr": 0.01,
                        "stride": 2
                    },
                    "policy": "schedule=0..100:infer"
                }
            }),
            Path::new("."),
            None,
            crate::compression::FramingMode::Framed,
        )
        .expect("typed rwkv7 compression backend should parse");

        match backend {
            CompressionBackend::Rate {
                rate_backend,
                coder,
                framing,
            } => {
                assert_eq!(coder, crate::coders::CoderType::RANS);
                assert_eq!(framing, crate::compression::FramingMode::Raw);
                match rate_backend {
                    RateBackend::Rwkv7Method { method } => match method {
                        crate::rwkvzip::MethodSpec::Online { cfg, policy } => {
                            assert_eq!(cfg.hidden, 64);
                            assert_eq!(cfg.layers, 1);
                            assert_eq!(cfg.stride, 2);
                            assert!(policy.is_some(), "policy should be preserved");
                        }
                        _ => panic!("expected online rwkv method"),
                    },
                    _ => panic!("expected rwkv7 rate backend"),
                }
            }
            CompressionBackend::Rwkv7 { method, coder } => {
                assert_eq!(coder, crate::coders::CoderType::RANS);
                match method {
                    crate::rwkvzip::MethodSpec::Online { cfg, policy } => {
                        assert_eq!(cfg.hidden, 64);
                        assert_eq!(cfg.layers, 1);
                        assert_eq!(cfg.stride, 2);
                        assert!(policy.is_some(), "policy should be preserved");
                    }
                    _ => panic!("expected online rwkv method"),
                }
            }
            _ => panic!("expected rwkv7-derived compression backend"),
        }
    }

    #[cfg(feature = "backend-rwkv")]
    #[test]
    fn parse_rwkv7_compression_shorthand_coder_alias_requires_or_uses_model_path() {
        let base_dir = unique_temp_dir("rwkv-coder-alias");
        fs::create_dir_all(&base_dir).expect("create temp dir");

        let missing_path_options = CompressionBackendShorthandOptions {
            base_dir: base_dir.clone(),
            ..CompressionBackendShorthandOptions::default()
        };
        let err = match parse_compression_backend_name_method(
            "rwkv7",
            Some("ac"),
            None,
            &missing_path_options,
        ) {
            Ok(_) => panic!("coder alias without default model path should fail"),
            Err(err) => err,
        };
        assert!(
            err.to_string().contains(
                "rwkv7 compression backend requires a configured model path when only a coder alias is provided"
            ),
            "{err}"
        );

        let with_path_options = CompressionBackendShorthandOptions {
            base_dir: base_dir.clone(),
            default_rwkv_model_path: Some("weights/model.safetensors".to_string()),
            ..CompressionBackendShorthandOptions::default()
        };
        let with_path_err = match parse_compression_backend_name_method(
            "rwkv7",
            Some("rans"),
            None,
            &with_path_options,
        ) {
            Ok(_) => panic!("missing RWKV model weights should fail deterministically"),
            Err(err) => err,
        };
        assert!(
            with_path_err
                .to_string()
                .contains("Failed to load model weights"),
            "{with_path_err}"
        );
        assert!(
            with_path_err
                .to_string()
                .contains("weights/model.safetensors"),
            "{with_path_err}"
        );

        let _ = fs::remove_dir(base_dir);
    }
}
