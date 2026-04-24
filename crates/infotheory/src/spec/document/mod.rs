//! Canonical top-level specification documents for planner runs and tuning.

use super::core::{CanonicalBytes, CompiledCompressionBackend, CompiledRateBackend};
use super::{
    SpecEnvironment, SpecError, SpecResult, compression_backend_to_json_value,
    parse_compression_backend_json, parse_rate_backend_json, rate_backend_to_json_value,
};
use crate::aixi::common::ObservationKeyMode;
use crate::aixi::common::resolve_random_seed;
use crate::api::{CompressionBackend, RateBackend};
use std::fmt;
use std::path::Path;
use std::sync::Arc;

/// Schema version for canonical top-level spec documents.
pub const SPEC_DOCUMENT_SCHEMA_VERSION: u32 = 1;

const DOCUMENT_MAGIC: &[u8; 4] = b"itsd";
const DOCUMENT_BINARY_VERSION: u8 = 1;
const TUNE_CANONICALIZATION_CLASSIFICATION_VERSION: &str = "bounds-v1";

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
            Self::Tune(compiled) => compiled.canonical_bytes(),
            Self::RateBackend(compiled) => compiled.canonical_bytes(),
            Self::CompressionBackend(compiled) => compiled.canonical_bytes(),
        }
    }
}

/// Compatibility alias for a planner-run top-level spec document.
pub type PlannerRunDocument = PlannerRunSpec;

/// Compatibility alias for a tune top-level spec document.
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

    /// Serialize this planner-run spec to deterministic canonical JSON.
    pub fn to_canonical_json(&self) -> SpecResult<String> {
        serde_json::to_string_pretty(&serializer::planner_run_to_json_value(self)?)
            .map_err(SpecError::from)
    }

    /// Serialize this planner-run spec to canonical JSON value form.
    pub fn to_canonical_json_value(&self) -> SpecResult<serde_json::Value> {
        serializer::planner_run_to_json_value(self)
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
}

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

    /// Serialize this tune request to deterministic canonical JSON.
    pub fn to_canonical_json(&self) -> SpecResult<String> {
        serde_json::to_string_pretty(&serializer::tune_spec_to_json_value(self)?)
            .map_err(SpecError::from)
    }

    /// Serialize this tune request to canonical JSON value form.
    pub fn to_canonical_json_value(&self) -> SpecResult<serde_json::Value> {
        serializer::tune_spec_to_json_value(self)
    }
}

impl SpecDocument {
    /// Serialize this document to deterministic canonical JSON.
    pub fn to_canonical_json(&self) -> SpecResult<String> {
        let value = self.to_canonical_json_value()?;
        serde_json::to_string_pretty(&value).map_err(SpecError::from)
    }

    /// Serialize this document to canonical JSON value form.
    pub fn to_canonical_json_value(&self) -> SpecResult<serde_json::Value> {
        serializer::spec_document_to_json_value(self)
    }

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

impl CompiledTuneSpec {
    /// Canonical tune request used to build this compiled form.
    pub fn canonical_spec(&self) -> &TuneSpec {
        self.canonical_spec.as_ref()
    }

    /// Deterministic canonical bytes for the tune request document.
    pub fn canonical_bytes(&self) -> &CanonicalBytes {
        &self.canonical_bytes
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
        "from_guest" | "guest" | "from-guest" => Ok(VmObservationPolicySpec::FromGuest),
        "output_hash" | "hash" | "output-hash" => Ok(VmObservationPolicySpec::OutputHash),
        "raw_output" | "raw" | "raw-output" => Ok(VmObservationPolicySpec::RawOutput),
        "shared_memory" | "shared-memory" | "shm" => Ok(VmObservationPolicySpec::SharedMemory),
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
        "pad_truncate" | "pad-truncate" => Ok(VmObservationStreamModeSpec::PadTruncate),
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
        "utf8" | "text" => Ok(VmPayloadEncodingSpec::Utf8),
        "hex" => Ok(VmPayloadEncodingSpec::Hex),
        other => Err(SpecError::new(format!(
            "unknown VM payload encoding '{other}' for {field_name}"
        ))),
    }
}

#[cfg(feature = "vm")]
fn canonicalize_vm_fuzz_mutator_name(name: &str) -> SpecResult<VmFuzzMutatorSpec> {
    match name {
        "flip_bit" | "flipbit" => Ok(VmFuzzMutatorSpec::FlipBit),
        "flip_byte" | "flipbyte" => Ok(VmFuzzMutatorSpec::FlipByte),
        "insert_byte" | "insertbyte" => Ok(VmFuzzMutatorSpec::InsertByte),
        "delete_byte" | "deletebyte" => Ok(VmFuzzMutatorSpec::DeleteByte),
        "splice_seed" | "splice-seed" | "splice" => Ok(VmFuzzMutatorSpec::SpliceSeed),
        "reset_seed" | "reset-seed" | "reset" => Ok(VmFuzzMutatorSpec::ResetSeed),
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

impl fmt::Debug for ValidatedTuneSpec {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ValidatedTuneSpec")
            .field("canonical_bytes_len", &self.canonical_bytes.len())
            .finish()
    }
}

#[cfg(test)]
mod tests;
