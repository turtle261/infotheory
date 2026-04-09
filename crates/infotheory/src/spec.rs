//! Canonical backend/spec parsing shared by Rust, CLI, and Python surfaces.

pub mod core;

pub use self::core::{
    AssetRef, CanonicalBytes, CompiledCompressionBackend, CompiledRateBackend,
    CompressionBackendCapabilities, MethodBackendFamily, RateBackendCapabilities,
    RateBackendTraceStrategy, SpecEnvironment, ValidatedCompressionBackend, ValidatedRateBackend,
};

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
            ctw_depth: 16,
            fac_ctw_base_depth: 16,
            fac_ctw_num_percept_bits: 8,
            fac_ctw_encoding_bits: 8,
            ppmd_order: 10,
            ppmd_memory_mb: 64,
            sequitur_context_bytes: 64,
            zpaq_method: "2".to_string(),
            default_mamba_model_path: None,
            default_rwkv_model_path: None,
            particle_default_if_missing_method: true,
        }
    }
}

/// Defaults for shorthand compression-backend parsing such as
/// CLI/Python `--compression-backend ... --method ...`.
#[derive(Clone)]
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

/// Resolve a relative spec path against a base directory.
pub fn resolve_spec_path(base_dir: &Path, path: &str) -> PathBuf {
    let path = Path::new(path);
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
        "byteclass" | "byte-class" | "byte_class" | "class" => {
            Ok(CalibrationContextKind::ByteClass)
        }
        "text" => Ok(CalibrationContextKind::Text),
        "repeat" => Ok(CalibrationContextKind::Repeat),
        "textrepeat" | "text-repeat" | "text_repeat" => Ok(CalibrationContextKind::TextRepeat),
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
        "framed" | "frame" => Ok(crate::compression::FramingMode::Framed),
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
fn split_method_policy_suffix(method: &str) -> (&str, Option<&str>) {
    if let Some((base, policy)) = method.split_once(";policy:") {
        (base, Some(policy))
    } else {
        (method, None)
    }
}

#[cfg(any(feature = "backend-rwkv", feature = "backend-mamba"))]
fn canonicalize_explicit_file_method(
    base_dir: &Path,
    method: &str,
    backend_label: &str,
) -> SpecResult<Option<String>> {
    let (base, policy) = split_method_policy_suffix(method);
    let Some(path) = base.strip_prefix("file:") else {
        return Ok(None);
    };
    let path = path.trim();
    if path.is_empty() {
        return Err(SpecError::new(format!(
            "empty file path in {backend_label} method"
        )));
    }
    let full = resolve_spec_path(base_dir, path);
    let mut canonical = format!("file:{}", full.display());
    if let Some(policy) = policy {
        canonical.push_str(";policy:");
        canonical.push_str(policy);
    }
    Ok(Some(canonical))
}

#[cfg(feature = "backend-rwkv")]
fn rwkv_file_method(path: &Path) -> SpecResult<String> {
    crate::rwkvzip::canonical_method_string(&crate::rwkvzip::MethodSpec::File {
        path: path.to_path_buf(),
        policy: None,
    })
    .map_err(|err| SpecError::new(err.to_string()))
}

#[cfg(feature = "backend-mamba")]
fn mamba_file_method(path: &Path) -> SpecResult<String> {
    crate::mambazip::canonical_method_string(&crate::mambazip::MethodSpec::File {
        path: path.to_path_buf(),
        policy: None,
    })
    .map_err(|err| SpecError::new(err.to_string()))
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
fn normalize_rwkv_path_method(base_dir: &Path, model_path: &str) -> SpecResult<String> {
    let full = resolve_spec_path(base_dir, model_path);
    let method = rwkv_file_method(&full)?;
    validate_rwkv_method_eager(&method)?;
    Ok(method)
}

#[cfg(feature = "backend-mamba")]
fn normalize_mamba_path_method(base_dir: &Path, model_path: &str) -> SpecResult<String> {
    let full = resolve_spec_path(base_dir, model_path);
    let method = mamba_file_method(&full)?;
    validate_mamba_method_eager(&method)?;
    Ok(method)
}

#[cfg(feature = "backend-rwkv")]
fn normalize_rwkv_method_for_base_dir(base_dir: &Path, method: &str) -> SpecResult<String> {
    if let Some(canonical) = canonicalize_explicit_file_method(base_dir, method, "rwkv")? {
        validate_rwkv_method_eager(&canonical)?;
        Ok(canonical)
    } else {
        Ok(method.to_string())
    }
}

#[cfg(feature = "backend-mamba")]
fn normalize_mamba_method_for_base_dir(base_dir: &Path, method: &str) -> SpecResult<String> {
    if let Some(canonical) = canonicalize_explicit_file_method(base_dir, method, "mamba")? {
        validate_mamba_method_eager(&canonical)?;
        Ok(canonical)
    } else {
        Ok(method.to_string())
    }
}

/// Parse an RWKV7 compression backend from a method string or configured model path,
/// preserving the shared direct-vs-rate-coded lowering semantics across CLI, JSON,
/// and binding surfaces.
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
        let parsed = crate::rwkvzip::parse_method_spec(&method).map_err(|err| {
            SpecError::new(format!(
                "invalid rwkv7 compression method '{method}': {err}"
            ))
        })?;
        match parsed {
            crate::rwkvzip::MethodSpec::File { policy: None, .. } => {
                Ok(CompressionBackend::Rwkv7 { method, coder })
            }
            crate::rwkvzip::MethodSpec::File {
                policy: Some(_), ..
            }
            | crate::rwkvzip::MethodSpec::Online { .. } => Ok(CompressionBackend::Rate {
                rate_backend: RateBackend::Rwkv7Method { method },
                coder,
                framing: options.default_framing,
            }),
        }
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
    if spec.max_order >= 0 || matches!(spec.backend, RateBackend::RosaPlus) {
        object.insert("max_order".to_string(), serde_json::json!(spec.max_order));
    }
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

/// Serialize a `RateBackend` into the canonical JSON representation.
pub fn rate_backend_to_json_value(backend: &RateBackend) -> SpecResult<serde_json::Value> {
    let canonical = backend.descriptor().map_err(SpecError::new)?.canonical;
    match backend {
        RateBackend::RosaPlus => Ok(serde_json::json!({ "kind": canonical })),
        RateBackend::Match {
            hash_bits,
            min_len,
            max_len,
            base_mix,
            confidence_scale,
        } => Ok(serde_json::json!({
            "kind": canonical,
            "hash_bits": hash_bits,
            "min_len": min_len,
            "max_len": max_len,
            "base_mix": base_mix,
            "confidence_scale": confidence_scale,
        })),
        RateBackend::SparseMatch {
            hash_bits,
            min_len,
            max_len,
            gap_min,
            gap_max,
            base_mix,
            confidence_scale,
        } => Ok(serde_json::json!({
            "kind": canonical,
            "hash_bits": hash_bits,
            "min_len": min_len,
            "max_len": max_len,
            "gap_min": gap_min,
            "gap_max": gap_max,
            "base_mix": base_mix,
            "confidence_scale": confidence_scale,
        })),
        RateBackend::Ppmd { order, memory_mb } => Ok(serde_json::json!({
            "kind": canonical,
            "order": order,
            "memory_mb": memory_mb,
        })),
        RateBackend::Sequitur { context_bytes } => Ok(serde_json::json!({
            "kind": canonical,
            "context_bytes": context_bytes,
        })),
        #[cfg(feature = "backend-mamba")]
        RateBackend::MambaMethod { method } => Ok(serde_json::json!({
            "kind": canonical,
            "method": method,
        })),
        #[cfg(feature = "backend-rwkv")]
        RateBackend::Rwkv7Method { method } => Ok(serde_json::json!({
            "kind": canonical,
            "method": method,
        })),
        RateBackend::Zpaq { method } => Ok(serde_json::json!({
            "kind": canonical,
            "method": method,
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
        RateBackend::Ctw { depth } => Ok(serde_json::json!({
            "kind": canonical,
            "depth": depth,
        })),
        RateBackend::FacCtw {
            base_depth,
            num_percept_bits,
            encoding_bits,
        } => Ok(serde_json::json!({
            "kind": canonical,
            "base_depth": base_depth,
            "num_percept_bits": num_percept_bits,
            "encoding_bits": encoding_bits,
        })),
    }
}

/// Serialize a `CompressionBackend` into the canonical JSON representation.
pub fn compression_backend_to_json_value(
    backend: &CompressionBackend,
) -> SpecResult<serde_json::Value> {
    let canonical = backend.descriptor().map_err(SpecError::new)?.canonical;
    match backend {
        CompressionBackend::Zpaq { method } => Ok(serde_json::json!({
            "kind": canonical,
            "method": method,
        })),
        #[cfg(feature = "backend-rwkv")]
        CompressionBackend::Rwkv7 { method, coder } => Ok(serde_json::json!({
            "kind": canonical,
            "method": method,
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

/// Serialize a `RateBackend` to deterministic canonical JSON text.
pub fn rate_backend_to_canonical_json(backend: &RateBackend) -> SpecResult<String> {
    serde_json::to_string_pretty(&rate_backend_to_json_value(backend)?).map_err(SpecError::from)
}

/// Serialize a `CompressionBackend` to deterministic canonical JSON text.
pub fn compression_backend_to_canonical_json(backend: &CompressionBackend) -> SpecResult<String> {
    serde_json::to_string_pretty(&compression_backend_to_json_value(backend)?)
        .map_err(SpecError::from)
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
        .or_else(|| v["type"].as_str())
        .or_else(|| v["backend"].as_str())
        .ok_or_else(|| SpecError::new("backend spec missing 'kind'"))?;
    let kind = resolve_enabled_rate_backend_kind(raw_kind)?;

    match kind {
        crate::runtime::RateBackendKind::RosaPlus => Ok(RateBackend::RosaPlus),
        crate::runtime::RateBackendKind::Ctw => Ok(RateBackend::Ctw {
            depth: v["depth"]
                .as_u64()
                .or_else(|| v["ct_depth"].as_u64())
                .unwrap_or(16) as usize,
        }),
        crate::runtime::RateBackendKind::FacCtw => {
            let base_depth = v["base_depth"]
                .as_u64()
                .or_else(|| v["ct_depth"].as_u64())
                .unwrap_or(16) as usize;
            let encoding_bits = v["encoding_bits"].as_u64().unwrap_or(8) as usize;
            let num_percept_bits = v["num_percept_bits"]
                .as_u64()
                .unwrap_or(encoding_bits as u64) as usize;
            Ok(RateBackend::FacCtw {
                base_depth,
                num_percept_bits,
                encoding_bits,
            })
        }
        crate::runtime::RateBackendKind::Match => Ok(RateBackend::Match {
            hash_bits: v["hash_bits"].as_u64().unwrap_or(20) as usize,
            min_len: v["min_len"].as_u64().unwrap_or(4) as usize,
            max_len: v["max_len"].as_u64().unwrap_or(255) as usize,
            base_mix: v["base_mix"].as_f64().unwrap_or(0.02),
            confidence_scale: v["confidence_scale"].as_f64().unwrap_or(1.0),
        }),
        crate::runtime::RateBackendKind::SparseMatch => Ok(RateBackend::SparseMatch {
            hash_bits: v["hash_bits"].as_u64().unwrap_or(19) as usize,
            min_len: v["min_len"].as_u64().unwrap_or(3) as usize,
            max_len: v["max_len"].as_u64().unwrap_or(64) as usize,
            gap_min: v["gap_min"].as_u64().unwrap_or(1) as usize,
            gap_max: v["gap_max"].as_u64().unwrap_or(2) as usize,
            base_mix: v["base_mix"].as_f64().unwrap_or(0.05),
            confidence_scale: v["confidence_scale"].as_f64().unwrap_or(1.0),
        }),
        crate::runtime::RateBackendKind::Ppmd => Ok(RateBackend::Ppmd {
            order: v["order"].as_u64().unwrap_or(10) as usize,
            memory_mb: v["memory_mb"].as_u64().unwrap_or(64) as usize,
        }),
        crate::runtime::RateBackendKind::Sequitur => Ok(RateBackend::Sequitur {
            context_bytes: v["context_bytes"].as_u64().unwrap_or(64) as usize,
        }),
        crate::runtime::RateBackendKind::Zpaq => {
            let method = v["method"].as_str().unwrap_or("2").to_string();
            validate_zpaq_rate_method(&method).map_err(|err| SpecError::new(err.to_string()))?;
            Ok(RateBackend::Zpaq { method })
        }
        crate::runtime::RateBackendKind::Mamba => {
            #[cfg(feature = "backend-mamba")]
            {
                if let Some(method) = v["method"].as_str().or_else(|| v["mamba_method"].as_str()) {
                    Ok(RateBackend::MambaMethod {
                        method: normalize_mamba_method_for_base_dir(base_dir, method)?,
                    })
                } else {
                    let model_path = v["mamba_model_path"]
                        .as_str()
                        .or_else(|| v["model_path"].as_str())
                        .ok_or_else(|| {
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
                if let Some(method) = v["method"].as_str().or_else(|| v["rwkv_method"].as_str()) {
                    Ok(RateBackend::Rwkv7Method {
                        method: normalize_rwkv_method_for_base_dir(base_dir, method)?,
                    })
                } else {
                    let model_path = v["rwkv_model_path"]
                        .as_str()
                        .or_else(|| v["model_path"].as_str())
                        .ok_or_else(|| {
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
            } else if let Some(path) = v["spec_path"]
                .as_str()
                .or_else(|| v["path"].as_str())
                .or_else(|| v["spec"].as_str())
            {
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
            } else if let Some(path) = v["spec_path"]
                .as_str()
                .or_else(|| v["path"].as_str())
                .or_else(|| v["spec"].as_str())
            {
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
            } else if let Some(path) = v["spec_path"]
                .as_str()
                .or_else(|| v["path"].as_str())
                .or_else(|| v["spec"].as_str())
            {
                let full = resolve_spec_path(base_dir, path);
                load_calibrated_spec(full.to_string_lossy().as_ref())?
            } else {
                parse_calibrated_spec_value(v, base_dir, depth - 1)?
            };
            Ok(RateBackend::Calibrated {
                spec: Arc::new(spec),
            })
        }
    }
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
        .or_else(|| v["type"].as_str())
        .or_else(|| v["backend"].as_str())
        .ok_or_else(|| SpecError::new("compression backend spec missing 'kind'"))?;
    let kind = resolve_enabled_compression_backend_kind(raw_kind)?;
    let framing = v["framing"]
        .as_str()
        .map(|value| parse_framing_mode(Some(value)))
        .transpose()?
        .unwrap_or(default_framing);

    match kind {
        crate::runtime::CompressionBackendKind::Zpaq => Ok(CompressionBackend::Zpaq {
            method: v["method"].as_str().unwrap_or("5").to_string(),
        }),
        crate::runtime::CompressionBackendKind::RateAc
        | crate::runtime::CompressionBackendKind::RateRans => {
            let rate_backend = if let Some(rate_backend_v) = v.get("rate_backend") {
                parse_rate_backend_json(rate_backend_v, base_dir, MAX_MIXTURE_NESTING)?
            } else if let Some(backend_v) = v.get("backend_spec") {
                parse_rate_backend_json(backend_v, base_dir, MAX_MIXTURE_NESTING)?
            } else {
                default_rate_backend.unwrap_or_default()
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
                let method = v["method"].as_str().or_else(|| v["rwkv_method"].as_str());
                let model_path = v["rwkv_model_path"]
                    .as_str()
                    .or_else(|| v["model_path"].as_str());
                if method.is_none() && model_path.is_none() {
                    Err(SpecError::new(
                        "rwkv7 compression backend requires 'method' or 'model_path'",
                    ))
                } else {
                    let opts = CompressionBackendShorthandOptions {
                        base_dir: base_dir.to_path_buf(),
                        default_framing: framing,
                        default_rwkv_model_path: model_path.map(ToOwned::to_owned),
                        ..Default::default()
                    };
                    parse_rwkv7_compression_backend_method(method, coder, &opts)
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
    } else if let Some(path) = v["base_path"].as_str().or_else(|| v["path"].as_str()) {
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
        bins: v["bins"].as_u64().unwrap_or(33) as usize,
        learning_rate: v["learning_rate"].as_f64().unwrap_or(0.02),
        bias_clip: v["bias_clip"].as_f64().unwrap_or(4.0),
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
    let max_order = if matches!(backend, RateBackend::RosaPlus) {
        v["max_order"]
            .as_i64()
            .or_else(|| v["order"].as_i64())
            .unwrap_or(8)
    } else {
        -1
    };

    Ok(MixtureExpertSpec {
        name: v["name"].as_str().map(|s| s.to_string()),
        log_prior: v["log_prior"]
            .as_f64()
            .or_else(|| v["prior"].as_f64())
            .unwrap_or(0.0),
        max_order,
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
    parse_calibrated_spec_value(&value, base_dir, 4)
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

    match kind {
        crate::runtime::RateBackendKind::RosaPlus => Ok(RateBackend::RosaPlus),
        crate::runtime::RateBackendKind::Match => Ok(RateBackend::Match {
            hash_bits: 20,
            min_len: 4,
            max_len: 255,
            base_mix: 0.02,
            confidence_scale: 1.0,
        }),
        crate::runtime::RateBackendKind::SparseMatch => Ok(RateBackend::SparseMatch {
            hash_bits: 19,
            min_len: 3,
            max_len: 64,
            gap_min: 1,
            gap_max: 2,
            base_mix: 0.05,
            confidence_scale: 1.0,
        }),
        crate::runtime::RateBackendKind::Ppmd => Ok(RateBackend::Ppmd {
            order: method
                .and_then(|value| value.parse::<usize>().ok())
                .unwrap_or(options.ppmd_order),
            memory_mb: options.ppmd_memory_mb,
        }),
        crate::runtime::RateBackendKind::Sequitur => Ok(RateBackend::Sequitur {
            context_bytes: method
                .and_then(|value| value.parse::<usize>().ok())
                .unwrap_or(options.sequitur_context_bytes),
        }),
        crate::runtime::RateBackendKind::Ctw => Ok(RateBackend::Ctw {
            depth: method
                .and_then(|value| value.parse::<usize>().ok())
                .unwrap_or(options.ctw_depth),
        }),
        crate::runtime::RateBackendKind::FacCtw => {
            let base_depth = method
                .and_then(|value| value.parse::<usize>().ok())
                .unwrap_or(options.fac_ctw_base_depth);
            Ok(RateBackend::FacCtw {
                base_depth,
                num_percept_bits: options.fac_ctw_num_percept_bits,
                encoding_bits: options.fac_ctw_encoding_bits,
            })
        }
        crate::runtime::RateBackendKind::Zpaq => {
            let method = method
                .map(ToOwned::to_owned)
                .unwrap_or_else(|| options.zpaq_method.clone());
            validate_zpaq_rate_method(&method).map_err(|err| SpecError::new(err.to_string()))?;
            Ok(RateBackend::Zpaq { method })
        }
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
    }
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
        crate::runtime::CompressionBackendKind::Zpaq => Ok(CompressionBackend::Zpaq {
            method: method
                .map(ToOwned::to_owned)
                .unwrap_or_else(|| options.zpaq_method.clone()),
        }),
        crate::runtime::CompressionBackendKind::RateAc => Ok(CompressionBackend::Rate {
            rate_backend: rate_backend
                .or_else(|| options.default_rate_backend.clone())
                .unwrap_or_default(),
            coder: crate::coders::CoderType::AC,
            framing: options.default_framing,
        }),
        crate::runtime::CompressionBackendKind::RateRans => Ok(CompressionBackend::Rate {
            rate_backend: rate_backend
                .or_else(|| options.default_rate_backend.clone())
                .unwrap_or_default(),
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
        MixtureExpertSpec, MixtureKind, MixtureSpec, ParticleSpec, RateBackend,
    };
    #[cfg(any(
        feature = "all-backends",
        feature = "backend-rwkv",
        feature = "backend-mamba"
    ))]
    use crate::coders::CoderType;
    use std::path::Path;
    use std::sync::Arc;

    fn sample_self_contained_rate_backend(
        kind: crate::runtime::RateBackendKind,
    ) -> Option<RateBackend> {
        match kind {
            crate::runtime::RateBackendKind::RosaPlus => Some(RateBackend::RosaPlus),
            crate::runtime::RateBackendKind::Ctw => Some(RateBackend::Ctw { depth: 8 }),
            crate::runtime::RateBackendKind::FacCtw => Some(RateBackend::FacCtw {
                base_depth: 8,
                num_percept_bits: 8,
                encoding_bits: 8,
            }),
            crate::runtime::RateBackendKind::Match => Some(RateBackend::Match {
                hash_bits: 18,
                min_len: 4,
                max_len: 96,
                base_mix: 0.02,
                confidence_scale: 1.0,
            }),
            crate::runtime::RateBackendKind::SparseMatch => Some(RateBackend::SparseMatch {
                hash_bits: 17,
                min_len: 3,
                max_len: 48,
                gap_min: 1,
                gap_max: 2,
                base_mix: 0.05,
                confidence_scale: 1.0,
            }),
            crate::runtime::RateBackendKind::Ppmd => Some(RateBackend::Ppmd {
                order: 6,
                memory_mb: 16,
            }),
            crate::runtime::RateBackendKind::Sequitur => {
                Some(RateBackend::Sequitur { context_bytes: 32 })
            }
            crate::runtime::RateBackendKind::Zpaq => Some(RateBackend::Zpaq {
                method: "2".to_string(),
            }),
            crate::runtime::RateBackendKind::Particle => Some(RateBackend::Particle {
                spec: Arc::new(ParticleSpec::default()),
            }),
            crate::runtime::RateBackendKind::Mixture
            | crate::runtime::RateBackendKind::Calibrated => None,
            #[cfg(feature = "backend-mamba")]
            crate::runtime::RateBackendKind::Mamba => Some(RateBackend::MambaMethod {
                method: "cfg:hidden=64,layers=1,intermediate=96,state=16,conv=4,dt_rank=16,seed=26,train=none,lr=0.0,stride=1;policy:schedule=0..100:infer".to_string(),
            }),
            #[cfg(not(feature = "backend-mamba"))]
            crate::runtime::RateBackendKind::Mamba => None,
            #[cfg(feature = "backend-rwkv")]
            crate::runtime::RateBackendKind::Rwkv7 => Some(RateBackend::Rwkv7Method {
                method: "cfg:hidden=64,intermediate=64,layers=1,train=sgd,lr=0.01;policy:schedule=0..100:infer".to_string(),
            }),
            #[cfg(not(feature = "backend-rwkv"))]
            crate::runtime::RateBackendKind::Rwkv7 => None,
        }
    }

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
    fn shorthand_compression_aliases_compile_to_identical_canonical_bytes() {
        let opts = CompressionBackendShorthandOptions {
            default_rate_backend: Some(RateBackend::Ctw { depth: 8 }),
            ..CompressionBackendShorthandOptions::default()
        };
        let rate_ac = compile_compression_backend_name_method("rate_ac", None, None, &opts)
            .expect("compile rate_ac alias");
        let rate_ac_canonical =
            compile_compression_backend_name_method("rate-ac", None, None, &opts)
                .expect("compile rate-ac canonical");
        assert_eq!(
            rate_ac.canonical_bytes().as_slice(),
            rate_ac_canonical.canonical_bytes().as_slice()
        );
        assert_eq!(
            rate_ac.canonical_spec().to_canonical_json().unwrap(),
            rate_ac_canonical
                .canonical_spec()
                .to_canonical_json()
                .unwrap()
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
                                    max_order: -1,
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
            backends.push(CompressionBackend::Zpaq {
                method: "5".to_string(),
            });
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
                                    max_order: 8,
                                    backend: RateBackend::RosaPlus,
                                },
                                MixtureExpertSpec {
                                    name: Some("ctw".to_string()),
                                    log_prior: -1.4,
                                    max_order: -1,
                                    backend: RateBackend::Ctw { depth: 12 },
                                },
                                MixtureExpertSpec {
                                    name: Some("particle".to_string()),
                                    log_prior: -2.0,
                                    max_order: -1,
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
            CompressionBackend::Zpaq { method } => assert_eq!(method, "5"),
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
}
