//! Canonical top-level specification documents for planner runs and tuning.

#[cfg(feature = "tuner")]
use super::core::CompiledCompressionBackend;
use super::core::{CanonicalBytes, CompiledRateBackend};
use super::{
    SpecEnvironment, SpecError, SpecResult, compression_backend_to_json_value,
    parse_compression_backend_json, parse_rate_backend_json, rate_backend_to_json_value,
};
use crate::aixi::common::ObservationKeyMode;
use crate::aixi::common::resolve_random_seed;
use crate::api::{CompressionBackend, RateBackend};
use crate::spec::CanonicalJson;
use std::fmt;
use std::path::Path;
use std::sync::Arc;

/// Schema version for canonical top-level spec documents.
pub const SPEC_DOCUMENT_SCHEMA_VERSION: u32 = 1;

const DOCUMENT_MAGIC: &[u8; 4] = b"itsd";
const DOCUMENT_BINARY_VERSION: u8 = 1;
#[cfg(feature = "tuner")]
const TUNE_CANONICALIZATION_CLASSIFICATION_VERSION: &str = "bounds-v1";

#[cfg(feature = "tuner")]
pub(crate) const CANDIDATE_EXTERNAL_ASSET_FORBIDDEN: &str = "candidate_external_asset_forbidden";

#[cfg(feature = "tuner")]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum TuneInvalidReason {
    CandidateExternalAssetForbidden,
    CandidateOutOfBounds,
    CandidateCompileError,
    InvalidActionIndex,
    InapplicableAction,
}

#[cfg(feature = "tuner")]
impl TuneInvalidReason {
    pub(crate) const fn as_str(self) -> &'static str {
        match self {
            Self::CandidateExternalAssetForbidden => CANDIDATE_EXTERNAL_ASSET_FORBIDDEN,
            Self::CandidateOutOfBounds => "candidate_out_of_bounds",
            Self::CandidateCompileError => "candidate_compile_error",
            Self::InvalidActionIndex => "invalid_action_index",
            Self::InapplicableAction => "inapplicable_action",
        }
    }
}

mod binary;
mod io;
mod parser;
mod pipeline;
mod serializer;
mod types;

use binary::{builtin_environment_name, observation_key_mode_name};
#[cfg(feature = "vm")]
use binary::{
    shared_memory_policy_name, vm_fuzz_mutator_name, vm_observation_policy_name,
    vm_observation_stream_mode_name, vm_payload_encoding_name,
};
pub use io::load_spec_document;
pub use types::*;

impl ValidatedPlannerRunSpec {
    /// Canonical validated planner-run spec.
    pub fn canonical_spec(&self) -> &PlannerRunSpec {
        self.canonical_spec.as_ref()
    }

    /// Deterministic binary representation of the canonical planner-run spec.
    pub fn canonical_bytes(&self) -> &CanonicalBytes {
        &self.canonical_bytes
    }

    /// Compile the validated planner-run spec into resolved assets and compiled backends.
    pub fn compile(&self) -> SpecResult<CompiledPlannerRunSpec> {
        pipeline::compile_validated_planner_run_spec(self)
    }
}

#[cfg(feature = "tuner")]
impl ValidatedTuneSpec {
    /// Canonical validated tune spec.
    pub fn canonical_spec(&self) -> &TuneSpec {
        self.canonical_spec.as_ref()
    }

    /// Deterministic binary representation of the canonical tune spec.
    pub fn canonical_bytes(&self) -> &CanonicalBytes {
        &self.canonical_bytes
    }

    /// Compile the validated tune request into resolved assets and compiled backends.
    pub fn compile(&self) -> SpecResult<CompiledTuneSpec> {
        pipeline::compile_validated_tune_spec(self)
    }
}

impl ParsedSpecDocument {
    /// Parsed document before validation.
    pub fn document(&self) -> &SpecDocument {
        &self.document
    }

    /// Base directory captured at parse time.
    pub fn base_dir(&self) -> &Path {
        self.base_dir.as_path()
    }

    /// Consume this stage wrapper and return the parsed document.
    pub fn into_document(self) -> SpecDocument {
        self.document
    }

    /// Validate this parsed document in the captured parse environment.
    pub fn validate(self) -> SpecResult<ValidatedSpecDocument> {
        let env = SpecEnvironment::new(self.base_dir);
        self.document.validate_in(&env)
    }

    /// Validate and compile this parsed document in the captured parse environment.
    pub fn compile(self) -> SpecResult<CompiledSpecDocument> {
        self.validate()?.compile()
    }
}

impl ValidatedSpecDocument {
    /// Deterministic canonical bytes for this validated document.
    pub fn canonical_bytes(&self) -> &CanonicalBytes {
        match self {
            Self::PlannerRun(validated) => validated.canonical_bytes(),
            #[cfg(feature = "tuner")]
            Self::Tune(validated) => validated.canonical_bytes(),
            Self::RateBackend(validated) => validated.canonical_bytes(),
            Self::CompressionBackend(validated) => validated.canonical_bytes(),
        }
    }

    /// Compile this validated document into runtime-ready form.
    pub fn compile(&self) -> SpecResult<CompiledSpecDocument> {
        match self {
            Self::PlannerRun(validated) => {
                Ok(CompiledSpecDocument::PlannerRun(validated.compile()?))
            }
            #[cfg(feature = "tuner")]
            Self::Tune(validated) => Ok(CompiledSpecDocument::Tune(validated.compile()?)),
            Self::RateBackend(validated) => {
                Ok(CompiledSpecDocument::RateBackend(validated.compile()?))
            }
            Self::CompressionBackend(validated) => Ok(CompiledSpecDocument::CompressionBackend(
                validated.compile()?,
            )),
        }
    }
}

impl CompiledSpecDocument {
    /// Deterministic canonical bytes for this compiled document.
    pub fn canonical_bytes(&self) -> &CanonicalBytes {
        match self {
            Self::PlannerRun(compiled) => compiled.canonical_bytes(),
            #[cfg(feature = "tuner")]
            Self::Tune(compiled) => compiled.canonical_bytes(),
            Self::RateBackend(compiled) => compiled.canonical_bytes(),
            Self::CompressionBackend(compiled) => compiled.canonical_bytes(),
        }
    }
}

/// Compatibility alias for a planner-run top-level spec document.
pub type PlannerRunDocument = PlannerRunSpec;

/// Compatibility alias for a tune top-level spec document.
#[cfg(feature = "tuner")]
pub type TuneDocument = TuneSpec;

impl PlannerRunSpec {
    /// Validate this planner-run spec and return its canonical binary encoding.
    pub fn validate_in(&self, env: &SpecEnvironment) -> SpecResult<ValidatedPlannerRunSpec> {
        let canonical = pipeline::canonicalize_planner_run(self, env)?;
        Ok(ValidatedPlannerRunSpec {
            canonical_bytes: CanonicalBytes::from(binary::encode_spec_document_payload(
                &SpecDocument::PlannerRun(canonical.clone()),
            )),
            canonical_spec: Arc::new(canonical),
            base_dir: env.base_dir().to_path_buf(),
        })
    }

    /// Validate this planner-run spec using the default compilation environment.
    pub fn validate(&self) -> SpecResult<ValidatedPlannerRunSpec> {
        self.validate_in(&SpecEnvironment::default())
    }

    /// Validate and compile this planner-run spec using the supplied environment.
    pub fn compile_in(&self, env: &SpecEnvironment) -> SpecResult<CompiledPlannerRunSpec> {
        pipeline::compile_planner_run_spec(self, env.base_dir())
    }

    /// Validate and compile this planner-run spec using the default environment.
    pub fn compile(&self) -> SpecResult<CompiledPlannerRunSpec> {
        self.compile_in(&SpecEnvironment::default())
    }
}

impl EnvironmentSpec {
    /// Validate and canonicalize this environment spec against the supplied asset bindings.
    pub fn validate_in(
        &self,
        assets: &[AssetBinding],
        env: &SpecEnvironment,
    ) -> SpecResult<EnvironmentSpec> {
        pipeline::canonicalize_environment_spec(self, assets, env)
    }

    /// Validate and canonicalize this environment spec using the default environment.
    pub fn validate(&self, assets: &[AssetBinding]) -> SpecResult<EnvironmentSpec> {
        self.validate_in(assets, &SpecEnvironment::default())
    }

    /// Stable canonical kind name for this environment variant.
    ///
    /// Used by tooling that prints canonical names without dispatching on the
    /// payload of each variant.
    pub fn kind_str(&self) -> &'static str {
        match self {
            Self::Builtin { .. } => "builtin",
            #[cfg(feature = "vm")]
            Self::NyxVm(_) => "vm",
        }
    }
}

impl BuiltinEnvironmentSpec {
    /// Stable canonical name string for this builtin environment.
    ///
    /// This is the canonical document/CLI identifier used in serialized specs;
    /// it does not change when new builtins are added.
    pub fn canonical_name(&self) -> &'static str {
        match self {
            Self::TunerBridge => "tuner_bridge",
            Self::CoinFlip => "coin_flip",
            Self::BiasedRockPaperScissor => "biased_rock_paper_scissor",
            Self::KuhnPoker => "kuhn_poker",
            Self::ExtendedTiger => "extended_tiger",
            Self::TicTacToe => "tic_tac_toe",
            Self::Blackjack => "blackjack",
            Self::Platformer => "platformer",
        }
    }
}

#[cfg(feature = "tuner")]
impl TuneSpec {
    /// Validate this tune request and return its canonical binary encoding.
    pub fn validate_in(&self, env: &SpecEnvironment) -> SpecResult<ValidatedTuneSpec> {
        let canonical = pipeline::canonicalize_tune_spec(self, env)?;
        Ok(ValidatedTuneSpec {
            canonical_bytes: CanonicalBytes::from(binary::encode_spec_document_payload(
                &SpecDocument::Tune(canonical.clone()),
            )),
            canonical_spec: Arc::new(canonical),
            base_dir: env.base_dir().to_path_buf(),
        })
    }

    /// Validate this tune request using the default compilation environment.
    pub fn validate(&self) -> SpecResult<ValidatedTuneSpec> {
        self.validate_in(&SpecEnvironment::default())
    }

    /// Validate and compile this tune request using the supplied environment.
    pub fn compile_in(&self, env: &SpecEnvironment) -> SpecResult<CompiledTuneSpec> {
        pipeline::compile_tune_spec(self, env.base_dir())
    }

    /// Validate and compile this tune request using the default environment.
    pub fn compile(&self) -> SpecResult<CompiledTuneSpec> {
        self.compile_in(&SpecEnvironment::default())
    }
}

impl SpecDocument {
    /// Encode this document in the versioned binary document envelope.
    pub fn to_binary(&self) -> Vec<u8> {
        binary::encode_spec_document_payload(self)
    }

    /// Parse a canonical JSON document from a raw JSON value.
    pub fn parse_json_value(value: &serde_json::Value, base_dir: &Path) -> SpecResult<Self> {
        parser::parse_spec_document_json_value(value, base_dir)
    }

    /// Parse a canonical JSON document and return the parsed-stage wrapper.
    pub fn parse_json_value_staged(
        value: &serde_json::Value,
        base_dir: &Path,
    ) -> SpecResult<ParsedSpecDocument> {
        let document = Self::parse_json_value(value, base_dir)?;
        Ok(ParsedSpecDocument {
            document,
            base_dir: base_dir.to_path_buf(),
        })
    }

    /// Decode a binary spec document.
    pub fn from_binary(bytes: &[u8], base_dir: &Path) -> SpecResult<Self> {
        binary::decode_spec_document(bytes, base_dir)
    }

    /// Decode a binary spec document and return the parsed-stage wrapper.
    pub fn from_binary_staged(bytes: &[u8], base_dir: &Path) -> SpecResult<ParsedSpecDocument> {
        let document = Self::from_binary(bytes, base_dir)?;
        Ok(ParsedSpecDocument {
            document,
            base_dir: base_dir.to_path_buf(),
        })
    }

    /// Validate this top-level document in the supplied environment.
    pub fn validate_in(&self, env: &SpecEnvironment) -> SpecResult<ValidatedSpecDocument> {
        match self {
            Self::PlannerRun(spec) => Ok(ValidatedSpecDocument::PlannerRun(spec.validate_in(env)?)),
            #[cfg(feature = "tuner")]
            Self::Tune(spec) => Ok(ValidatedSpecDocument::Tune(spec.validate_in(env)?)),
            Self::RateBackend(backend) => Ok(ValidatedSpecDocument::RateBackend(
                backend.validate_in(env)?,
            )),
            Self::CompressionBackend(backend) => Ok(ValidatedSpecDocument::CompressionBackend(
                backend.validate_in(env)?,
            )),
        }
    }

    /// Validate this top-level document using the default compilation environment.
    pub fn validate(&self) -> SpecResult<ValidatedSpecDocument> {
        self.validate_in(&SpecEnvironment::default())
    }

    /// Validate and compile this top-level document in the supplied environment.
    pub fn compile_in(&self, env: &SpecEnvironment) -> SpecResult<CompiledSpecDocument> {
        self.validate_in(env)?.compile()
    }

    /// Validate and compile this top-level document using the default environment.
    pub fn compile(&self) -> SpecResult<CompiledSpecDocument> {
        self.compile_in(&SpecEnvironment::default())
    }

    /// Stable canonical kind name (matches the document's `"kind"` JSON field).
    pub fn kind_str(&self) -> &'static str {
        match self {
            Self::PlannerRun(_) => "planner_run",
            #[cfg(feature = "tuner")]
            Self::Tune(_) => "tune",
            Self::RateBackend(_) => "rate_backend",
            Self::CompressionBackend(_) => "compression_backend",
        }
    }
}

impl CanonicalJson for PlannerRunSpec {
    fn to_canonical_json_value(&self) -> SpecResult<serde_json::Value> {
        serializer::planner_run_to_json_value(self)
    }
}

#[cfg(feature = "tuner")]
impl CanonicalJson for TuneSpec {
    fn to_canonical_json_value(&self) -> SpecResult<serde_json::Value> {
        serializer::tune_spec_to_json_value(self)
    }
}

impl CanonicalJson for SpecDocument {
    fn to_canonical_json_value(&self) -> SpecResult<serde_json::Value> {
        serializer::spec_document_to_json_value(self)
    }
}

impl CompiledPlannerController {
    /// Compiled predictor backend used by this planner controller.
    pub fn predictor(&self) -> &CompiledRateBackend {
        match self {
            Self::McAixi { predictor, .. } => predictor,
            Self::AiqiDiscounted { predictor, .. } => predictor,
            Self::AiqiWarmstartExactJh { predictor, .. } => predictor,
        }
    }

    /// Stable canonical kind name string for this controller variant.
    pub fn kind_str(&self) -> &'static str {
        match self {
            Self::McAixi { .. } => "mc_aixi",
            Self::AiqiDiscounted { .. } => "aiqi_discounted",
            Self::AiqiWarmstartExactJh { .. } => "aiqi_warmstart_exact_jh",
        }
    }

    /// Human-readable predictor backend label.
    ///
    /// Backend-local algorithm parameters (such as ROSA's `max_order`) are
    /// derived from the backend's variant directly.
    pub fn backend_label(&self) -> String {
        self.predictor().display_label()
    }
}

impl CompiledPlannerRunSpec {
    /// Canonical planner-run spec used to build this compiled form.
    pub fn canonical_spec(&self) -> &PlannerRunSpec {
        self.canonical_spec.as_ref()
    }

    /// Deterministic canonical bytes for the planner-run document.
    pub fn canonical_bytes(&self) -> &CanonicalBytes {
        &self.canonical_bytes
    }

    /// Resolved assets used when compiling the planner-run document.
    pub fn resolved_assets(&self) -> &[ResolvedAssetBinding] {
        self.resolved_assets.as_ref()
    }

    /// Canonical planner interface metadata.
    pub fn interface(&self) -> &PlannerInterfaceSpec {
        &self.interface
    }

    /// Operational runtime controls.
    pub fn runtime(&self) -> &PlannerRuntimeSpec {
        &self.runtime
    }

    /// Returns the canonical resolved planner runtime seed.
    pub fn resolved_random_seed(&self) -> u64 {
        resolve_random_seed(self.runtime.random_seed)
    }

    /// Compiled planner controller.
    pub fn controller(&self) -> &CompiledPlannerController {
        &self.controller
    }

    /// Number of bits required to encode agent actions.
    pub fn action_bits(&self) -> usize {
        self.action_bits
    }
}

#[cfg(feature = "tuner")]
impl CompiledTuneSpec {
    /// Canonical tune request used to build this compiled form.
    pub fn canonical_spec(&self) -> &TuneSpec {
        self.canonical_spec.as_ref()
    }

    /// Deterministic canonical bytes for the tune request document.
    pub fn canonical_bytes(&self) -> &CanonicalBytes {
        &self.canonical_bytes
    }

    /// Base directory used to resolve relative paths during tune compilation.
    pub fn base_dir(&self) -> &Path {
        self.base_dir.as_path()
    }

    /// Resolved assets used when compiling the tune request.
    pub fn resolved_assets(&self) -> &[ResolvedAssetBinding] {
        self.resolved_assets.as_ref()
    }

    /// Compiled baseline candidate used by the future tuner.
    pub fn baseline_candidate(&self) -> &CompiledCompressionBackend {
        &self.baseline_candidate
    }

    /// Runtime-selectable compiled controller settings.
    pub fn controller(&self) -> &CompiledTuneController {
        &self.controller
    }

    /// Stable identifier for the current canonicalization classification rules.
    pub fn candidate_canonicalization_version(&self) -> &'static str {
        self.candidate_canonicalization_version
    }

    /// Canonical byte length of the baseline candidate model code.
    pub fn baseline_candidate_model_bytes(&self) -> usize {
        self.baseline_candidate.canonical_bytes().len()
    }
}

#[cfg(feature = "vm")]
fn canonicalize_vm_observation_policy_name(name: &str) -> SpecResult<VmObservationPolicySpec> {
    match name {
        "from_guest" => Ok(VmObservationPolicySpec::FromGuest),
        "output_hash" => Ok(VmObservationPolicySpec::OutputHash),
        "raw_output" => Ok(VmObservationPolicySpec::RawOutput),
        "shared_memory" => Ok(VmObservationPolicySpec::SharedMemory),
        other => Err(SpecError::new(format!(
            "unknown VM observation_policy '{other}'"
        ))),
    }
}

#[cfg(feature = "vm")]
fn canonicalize_vm_observation_stream_mode_name(
    name: &str,
) -> SpecResult<VmObservationStreamModeSpec> {
    match name {
        "pad_truncate" => Ok(VmObservationStreamModeSpec::PadTruncate),
        "pad" => Ok(VmObservationStreamModeSpec::Pad),
        "truncate" => Ok(VmObservationStreamModeSpec::Truncate),
        other => Err(SpecError::new(format!(
            "unknown VM observation_stream_mode '{other}'"
        ))),
    }
}

#[cfg(feature = "vm")]
fn canonicalize_vm_payload_encoding(
    name: &str,
    field_name: &str,
) -> SpecResult<VmPayloadEncodingSpec> {
    match name {
        "utf8" => Ok(VmPayloadEncodingSpec::Utf8),
        "hex" => Ok(VmPayloadEncodingSpec::Hex),
        other => Err(SpecError::new(format!(
            "unknown VM payload encoding '{other}' for {field_name}"
        ))),
    }
}

#[cfg(feature = "vm")]
fn canonicalize_vm_fuzz_mutator_name(name: &str) -> SpecResult<VmFuzzMutatorSpec> {
    match name {
        "flip_bit" => Ok(VmFuzzMutatorSpec::FlipBit),
        "flip_byte" => Ok(VmFuzzMutatorSpec::FlipByte),
        "insert_byte" => Ok(VmFuzzMutatorSpec::InsertByte),
        "delete_byte" => Ok(VmFuzzMutatorSpec::DeleteByte),
        "splice_seed" => Ok(VmFuzzMutatorSpec::SpliceSeed),
        "reset_seed" => Ok(VmFuzzMutatorSpec::ResetSeed),
        "havoc" => Ok(VmFuzzMutatorSpec::Havoc),
        other => Err(SpecError::new(format!("unknown VM fuzz mutator '{other}'"))),
    }
}

impl fmt::Debug for ValidatedPlannerRunSpec {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ValidatedPlannerRunSpec")
            .field("canonical_bytes_len", &self.canonical_bytes.len())
            .finish()
    }
}

#[cfg(feature = "tuner")]
impl fmt::Debug for ValidatedTuneSpec {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ValidatedTuneSpec")
            .field("canonical_bytes_len", &self.canonical_bytes.len())
            .finish()
    }
}

#[cfg(test)]
mod tests;
