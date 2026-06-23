//! Warm-start exact finite-horizon objective controller for AIXI-family runs.

use crate::aixi::common::{
    Action, ActionAlphabet, PerceptVal, RandomGenerator, Reward, RewardEncodingError,
    resolve_random_seed, validate_reward_encoding_bounds,
};
use crate::aixi::model::{Predictor, PredictorBuildError, build_aiqi_predictor};
use crate::aixi::planner_agent::PlannerActionProvenance;
use crate::aixi::planner_spec::{PlannerInterfaceConfig, build_default_planner_run_spec};
use crate::aixi::return_law::{
    ReturnLabelCodec, ReturnLawEvaluator, ReturnPrefixUpdate, expected_decoded_return,
    predict_return_law,
};
use crate::aixi::warmstart_contract::{
    TaskFingerprint, WARMSTART_STANDALONE_OBSERVATION_ADAPTER_SPEC_REF,
    WARMSTART_STANDALONE_SCALAR_REPRESENTATION, WARMSTART_TEACHER_CONTRACT_SCHEMA_VERSION,
    observation_key_mode_name, standalone_exact_reward_encoding_certificate_hash,
    standalone_observation_adapter_content_crc32, warmstart_exact_jh_planner_task_fingerprint,
};
use crate::api::{BitStreamSemantics, RateBackend, validate_rate_backend};
use crate::spec::{
    AssetBinding, BuiltinEnvironmentSpec, CompiledPlannerController, CompiledPlannerRunSpec,
    ControllerSpec, EnvironmentSpec, PlannerRunSpec, SpecError, WarmStartExactJhControllerSpec,
};
use serde_json::{Value, json};
use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::fmt;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

/// One observed environment transition in a same-task warm-start trace.
#[derive(Clone, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub struct WarmStartExactJhTransition {
    /// Action selected by the teacher/controller.
    pub action: Action,
    /// Observation stream emitted after the action.
    pub observations: Vec<PerceptVal>,
    /// Exact integer reward emitted after the action.
    pub reward: Reward,
}

impl WarmStartExactJhTransition {
    /// Construct one warm-start teacher transition.
    pub fn new(action: Action, observations: Vec<PerceptVal>, reward: Reward) -> Self {
        Self {
            action,
            observations,
            reward,
        }
    }
}

/// Same-task trace used to initialize a warm-start exact-J_H controller.
#[derive(Clone, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub struct WarmStartExactJhTeacherTrace {
    /// Chronological transition sequence.
    pub transitions: Vec<WarmStartExactJhTransition>,
}

impl WarmStartExactJhTeacherTrace {
    /// Construct a same-task teacher trace from chronological transitions.
    pub fn new(transitions: Vec<WarmStartExactJhTransition>) -> Self {
        Self { transitions }
    }
}

/// Validates standalone planner-run provenance hashes against canonical standalone declarations.
///
/// Used by the corresponding method on the private WarmStartExactJhRuntimeConfig
/// and by `validate_warmstart_teacher_against_compiled_planner_run`.
pub fn validate_standalone_warmstart_provenance(
    contract: &WarmStartExactJhTeacherContract,
    observation_bits: usize,
    observation_stream_len: usize,
    reward_bits: usize,
) -> Result<(), WarmStartExactJhError> {
    if contract.observation_adapter_spec_ref != WARMSTART_STANDALONE_OBSERVATION_ADAPTER_SPEC_REF {
        return Err(WarmStartExactJhError::InvalidTeacherDataset {
            reason: format!(
                "teacher observation_adapter_spec_ref '{}' does not match standalone direct-percept adapter declaration '{}'",
                contract.observation_adapter_spec_ref,
                WARMSTART_STANDALONE_OBSERVATION_ADAPTER_SPEC_REF
            ),
        });
    }
    let expected_adapter_crc = standalone_observation_adapter_content_crc32(
        observation_bits,
        observation_stream_len,
        reward_bits,
    )
    .map_err(|err| WarmStartExactJhError::InvalidTeacherDataset {
        reason: format!("failed to compute standalone observation adapter content hash: {err}"),
    })?;
    if contract.observation_adapter_content_crc32 != expected_adapter_crc {
        return Err(WarmStartExactJhError::InvalidTeacherDataset {
            reason: format!(
                "teacher observation_adapter_content_crc32 '{}' does not match canonical standalone adapter spec '{}'",
                contract.observation_adapter_content_crc32, expected_adapter_crc
            ),
        });
    }
    if contract.scalar_representation != WARMSTART_STANDALONE_SCALAR_REPRESENTATION {
        return Err(WarmStartExactJhError::InvalidTeacherDataset {
            reason: format!(
                "teacher scalar_representation '{}' does not match standalone nonnegative integer declaration '{}'",
                contract.scalar_representation, WARMSTART_STANDALONE_SCALAR_REPRESENTATION
            ),
        });
    }
    let expected_reward_cert = standalone_exact_reward_encoding_certificate_hash(reward_bits)
        .map_err(|err| WarmStartExactJhError::InvalidTeacherDataset {
            reason: format!(
                "failed to compute standalone exact reward encoding certificate hash: {err}"
            ),
        })?;
    if contract.exact_reward_encoding_certificate != expected_reward_cert {
        return Err(WarmStartExactJhError::InvalidTeacherDataset {
            reason: format!(
                "teacher exact_reward_encoding_certificate '{}' does not match canonical standalone reward encoder certificate '{}'",
                contract.exact_reward_encoding_certificate, expected_reward_cert
            ),
        });
    }
    Ok(())
}

/// Expected warm-start teacher contract fields for comparison against a parsed contract.
pub(crate) struct WarmStartTeacherContractExpectation<'a> {
    /// Expected schema version.
    pub schema_version: u64,
    /// Expected planner task fingerprint.
    pub task_fingerprint: TaskFingerprint,
    /// Expected action alphabet size.
    pub action_alphabet_size: usize,
    /// Expected observation bit width.
    pub observation_bits: usize,
    /// Expected observation stream length.
    pub observation_stream_len: usize,
    /// Expected observation key mode label.
    pub observation_key_mode: &'a str,
    /// Expected reward bit width.
    pub reward_bits: usize,
    /// Expected return horizon.
    pub return_horizon: usize,
    /// Expected delayed-label phase period.
    pub label_phase_period: usize,
    /// Whether standalone planner-run provenance hashes must match.
    pub validate_standalone_provenance: bool,
}

/// Validate a parsed teacher contract against an explicit field expectation.
pub(crate) fn validate_warmstart_teacher_contract_against_expectation(
    contract: &WarmStartExactJhTeacherContract,
    expected: &WarmStartTeacherContractExpectation<'_>,
) -> Result<(), WarmStartExactJhError> {
    if contract.schema_version != expected.schema_version {
        return Err(WarmStartExactJhError::InvalidTeacherDataset {
            reason: format!("teacher schema_version must be {}", expected.schema_version),
        });
    }
    if contract.task_fingerprint != expected.task_fingerprint {
        return Err(WarmStartExactJhError::InvalidTeacherDataset {
            reason: format!(
                "teacher task_fingerprint '{}' does not match current planner_run '{}'",
                contract.task_fingerprint, expected.task_fingerprint
            ),
        });
    }
    if contract.action_alphabet_size != expected.action_alphabet_size {
        return Err(WarmStartExactJhError::InvalidTeacherDataset {
            reason: format!(
                "teacher action_alphabet_size {} does not match configured {}",
                contract.action_alphabet_size, expected.action_alphabet_size
            ),
        });
    }
    if contract.observation_bits != expected.observation_bits {
        return Err(WarmStartExactJhError::InvalidTeacherDataset {
            reason: format!(
                "teacher observation_bits {} does not match configured {}",
                contract.observation_bits, expected.observation_bits
            ),
        });
    }
    if contract.observation_stream_len != expected.observation_stream_len {
        return Err(WarmStartExactJhError::InvalidTeacherDataset {
            reason: format!(
                "teacher observation_stream_len {} does not match configured {}",
                contract.observation_stream_len, expected.observation_stream_len
            ),
        });
    }
    if contract.observation_key_mode != expected.observation_key_mode {
        return Err(WarmStartExactJhError::InvalidTeacherDataset {
            reason: format!(
                "teacher observation_key_mode '{}' does not match configured planner interface '{}'",
                contract.observation_key_mode, expected.observation_key_mode
            ),
        });
    }
    if contract.reward_bits != expected.reward_bits {
        return Err(WarmStartExactJhError::InvalidTeacherDataset {
            reason: format!(
                "teacher reward_bits {} does not match configured {}",
                contract.reward_bits, expected.reward_bits
            ),
        });
    }
    if contract.return_horizon != expected.return_horizon {
        return Err(WarmStartExactJhError::InvalidTeacherDataset {
            reason: format!(
                "teacher return_horizon {} does not match configured {}",
                contract.return_horizon, expected.return_horizon
            ),
        });
    }
    if contract.label_phase_period != expected.label_phase_period {
        return Err(WarmStartExactJhError::InvalidTeacherDataset {
            reason: format!(
                "teacher label_phase_period {} does not match configured {}",
                contract.label_phase_period, expected.label_phase_period
            ),
        });
    }
    if expected.validate_standalone_provenance {
        validate_standalone_warmstart_provenance(
            contract,
            expected.observation_bits,
            expected.observation_stream_len,
            expected.reward_bits,
        )?;
    }
    Ok(())
}

/// Validates teacher [`WarmStartExactJhTeacherContract::schema_version`] and
/// [`WarmStartExactJhTeacherContract::task_fingerprint`] against a compiled planner run.
///
/// Used by [`validate_warmstart_teacher_against_compiled_planner_run`] (standalone CLI / assets)
/// and direct fingerprint probes. The tuner bridge validates complete teacher datasets through
/// [`validate_warmstart_teacher_dataset_for_compiled_planner_run`], which includes this
/// fingerprint check via the compiled runtime contract and then validates trace payloads.
/// On mismatch the reason string includes
/// `current planner_run '<hex>'` for stable integration-test probing.
pub fn validate_warmstart_teacher_planner_task_fingerprint(
    compiled: &CompiledPlannerRunSpec,
    contract: &WarmStartExactJhTeacherContract,
) -> Result<(), WarmStartExactJhError> {
    if contract.schema_version != WARMSTART_TEACHER_CONTRACT_SCHEMA_VERSION {
        return Err(WarmStartExactJhError::InvalidTeacherDataset {
            reason: format!(
                "teacher schema_version must be {WARMSTART_TEACHER_CONTRACT_SCHEMA_VERSION}"
            ),
        });
    }
    let task_fingerprint =
        warmstart_exact_jh_planner_task_fingerprint(compiled).map_err(|err| {
            WarmStartExactJhError::InvalidTeacherDataset {
                reason: format!("failed to compute planner task fingerprint: {err}"),
            }
        })?;
    if contract.task_fingerprint != task_fingerprint {
        return Err(WarmStartExactJhError::InvalidTeacherDataset {
            reason: format!(
                "teacher task_fingerprint '{}' does not match current planner_run '{}'",
                contract.task_fingerprint, task_fingerprint
            ),
        });
    }
    Ok(())
}

/// Validates a parsed teacher contract against a compiled standalone [`PlannerRunSpec`] (CLI / asset loader).
///
/// This is the single authoritative check for filesystem-loaded teachers before runtime construction.
pub fn validate_warmstart_teacher_against_compiled_planner_run(
    compiled: &CompiledPlannerRunSpec,
    contract: &WarmStartExactJhTeacherContract,
) -> Result<(), WarmStartExactJhError> {
    let interface = compiled.interface();
    let (return_horizon, label_phase_period, planner_simulations_per_step) =
        match compiled.controller() {
            CompiledPlannerController::AiqiWarmstartExactJh {
                return_horizon,
                label_phase_period,
                planner_simulations_per_step,
                ..
            } => (
                *return_horizon,
                *label_phase_period,
                *planner_simulations_per_step,
            ),
            _ => {
                return Err(WarmStartExactJhError::InvalidTeacherDataset {
                reason:
                    "warm-start teacher contract can only be validated for aiqi_warmstart_exact_jh"
                        .to_string(),
            });
            }
        };
    let task_fingerprint =
        warmstart_exact_jh_planner_task_fingerprint(compiled).map_err(|err| {
            WarmStartExactJhError::InvalidTeacherDataset {
                reason: format!("failed to compute planner task fingerprint: {err}"),
            }
        })?;
    validate_warmstart_teacher_contract_against_expectation(
        contract,
        &WarmStartTeacherContractExpectation {
            schema_version: WARMSTART_TEACHER_CONTRACT_SCHEMA_VERSION,
            task_fingerprint,
            action_alphabet_size: interface.agent_actions.get(),
            observation_bits: interface.observation_bits,
            observation_stream_len: interface.observation_stream_len.max(1),
            observation_key_mode: observation_key_mode_name(interface.observation_key_mode),
            reward_bits: interface.reward_bits,
            return_horizon,
            label_phase_period,
            validate_standalone_provenance: true,
        },
    )?;
    if planner_simulations_per_step == 0 {
        return Err(WarmStartExactJhError::PlannerSimulationsZero);
    }
    if planner_simulations_per_step != 1 {
        return Err(WarmStartExactJhError::PlannerSimulationsUnsupported {
            configured: planner_simulations_per_step,
        });
    }
    Ok(())
}

/// Validate a complete warm-start teacher dataset against a compiled planner run.
///
/// This is the authoritative ingestion/export gate for teacher assets. It checks
/// the same runtime contract used by [`WarmStartExactJhAgent`]: controller
/// compatibility, task fingerprint, provenance policy, transition bounds, exact
/// finite-horizon label encodability, and that every trace contributes at least
/// one complete \(H\)-step label.
pub fn validate_warmstart_teacher_dataset_for_compiled_planner_run(
    compiled: &CompiledPlannerRunSpec,
    teacher: &WarmStartExactJhTeacherDataset,
) -> Result<(), WarmStartExactJhError> {
    let config = WarmStartExactJhRuntimeConfig::from_compiled(compiled)?;
    config.validate_teacher_contract(&teacher.contract)?;
    let predictor = match compiled.controller() {
        CompiledPlannerController::AiqiWarmstartExactJh { predictor, .. } => predictor,
        _ => return Err(WarmStartExactJhError::ControllerKindMismatch),
    };
    if !predictor.supports_frozen_conditioning() {
        return Err(WarmStartExactJhError::UnsupportedRateBackend {
            reason: "warm-start exact-J_H strict mode requires frozen context conditioning; configured rate_backend does not provide strict frozen conditioning",
        });
    }

    let mut total_labels: usize = 0;
    for (trace_index, trace) in teacher.traces.iter().enumerate() {
        let trace_len = trace.transitions.len();
        if trace_len < config.return_horizon {
            return Err(WarmStartExactJhError::InvalidTeacherDataset {
                reason: format!(
                    "traces[{trace_index}] contains {trace_len} transitions but return_horizon is {}",
                    config.return_horizon
                ),
            });
        }
        for (step_index, transition) in trace.transitions.iter().enumerate() {
            validate_runtime_transition(
                &config,
                transition.action,
                &transition.observations,
                transition.reward,
            )
            .map_err(|err| WarmStartExactJhError::InvalidTeacherDataset {
                reason: format!(
                    "traces[{trace_index}].transitions[{step_index}] violates runtime contract: {err}"
                ),
            })?;
        }
        let labels = exact_return_labels_for_trace(&config, &trace.transitions)?;
        let label_count = labels.iter().filter(|label| label.is_some()).count();
        if label_count == 0 {
            return Err(WarmStartExactJhError::InvalidTeacherDataset {
                reason: format!("traces[{trace_index}] did not contain any complete H-step labels"),
            });
        }
        total_labels = total_labels.saturating_add(label_count);
    }

    if total_labels == 0 {
        return Err(WarmStartExactJhError::InvalidTeacherDataset {
            reason: "teacher dataset did not contain any complete H-step labels".to_string(),
        });
    }
    Ok(())
}

/// Validate one reconstructed teacher transition against a warm-start bridge contract.
pub fn validate_warmstart_teacher_transition_against_contract(
    contract: &WarmStartExactJhTeacherContract,
    action: Action,
    observations: &[PerceptVal],
    reward: Reward,
) -> Result<(), WarmStartExactJhError> {
    let agent_actions =
        ActionAlphabet::try_from_usize(contract.action_alphabet_size).map_err(|_| {
            WarmStartExactJhError::InvalidTeacherDataset {
                reason: "teacher contract action_alphabet_size is zero".to_string(),
            }
        })?;
    if action as usize >= contract.action_alphabet_size {
        return Err(WarmStartExactJhError::ActionOutOfRange {
            action,
            agent_actions,
        });
    }
    if observations.len() != contract.observation_stream_len {
        return Err(WarmStartExactJhError::ObservationStreamLengthMismatch {
            expected: contract.observation_stream_len,
            actual: observations.len(),
        });
    }
    let obs_max = max_value_for_bits(contract.observation_bits);
    for &observation in observations {
        if observation > obs_max {
            return Err(WarmStartExactJhError::ObservationValueOutOfRange {
                observation,
                observation_bits: contract.observation_bits,
                maximum: obs_max,
            });
        }
    }
    let max_reward = max_value_for_bits(contract.reward_bits) as i64;
    if reward < 0 || reward > max_reward {
        return Err(WarmStartExactJhError::RewardOutOfRange {
            reward,
            min_reward: 0,
            max_reward,
        });
    }
    Ok(())
}

/// Same-task teacher dataset for [`WarmStartExactJhAgent`].
#[derive(Clone, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub struct WarmStartExactJhTeacherDataset {
    /// Versioned same-task contract metadata for the teacher traces.
    pub contract: WarmStartExactJhTeacherContract,
    /// Canonical teacher traces. Each trace is treated as an independent
    /// same-task rollout, and dataset construction sorts and deduplicates this
    /// set by transition content.
    pub traces: Vec<WarmStartExactJhTeacherTrace>,
}

impl WarmStartExactJhTeacherDataset {
    /// Construct a teacher dataset from its contract and trace set.
    ///
    /// The supplied traces may be in arbitrary order and may contain
    /// duplicates; the dataset stores the canonical deterministic set.
    pub fn new(
        contract: WarmStartExactJhTeacherContract,
        mut traces: Vec<WarmStartExactJhTeacherTrace>,
    ) -> Self {
        canonicalize_warmstart_teacher_traces(&mut traces);
        Self { contract, traces }
    }
}

/// Versioned same-task contract attached to a warm-start teacher dataset.
#[derive(Clone, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub struct WarmStartExactJhTeacherContract {
    /// Teacher dataset schema version. Version 1 is the v1 exact-J_H contract.
    pub schema_version: u64,
    /// Fingerprint of the exact tuning task that produced the traces.
    pub task_fingerprint: TaskFingerprint,
    /// Number of actions in the compiled planner alphabet.
    pub action_alphabet_size: usize,
    /// Observation bit width.
    pub observation_bits: usize,
    /// Number of observation symbols per step.
    pub observation_stream_len: usize,
    /// Observation keying mode used by the planner-visible history.
    pub observation_key_mode: String,
    /// Observation adapter declaration reference used to encode raw tuner observations.
    pub observation_adapter_spec_ref: String,
    /// Content hash of the concrete observation adapter schema.
    pub observation_adapter_content_crc32: String,
    /// Reward bit width.
    pub reward_bits: usize,
    /// Return horizon used to compute exact labels.
    pub return_horizon: usize,
    /// Delayed-label phase period.
    pub label_phase_period: usize,
    /// Scalar representation declaration used by the exact reward encoder.
    pub scalar_representation: String,
    /// Hash or ref for the verified exact reward encoder.
    pub exact_reward_encoding_certificate: String,
}

impl WarmStartExactJhTeacherDataset {
    /// Parse a JSON teacher dataset.
    pub fn from_json_slice(bytes: &[u8]) -> Result<Self, WarmStartExactJhError> {
        let value = serde_json::from_slice::<Value>(bytes).map_err(|err| {
            WarmStartExactJhError::InvalidTeacherDataset {
                reason: format!("invalid teacher JSON: {err}"),
            }
        })?;
        Self::from_json_value(&value)
    }

    /// Parse a JSON teacher dataset value.
    pub fn from_json_value(value: &Value) -> Result<Self, WarmStartExactJhError> {
        let object =
            value
                .as_object()
                .ok_or_else(|| WarmStartExactJhError::InvalidTeacherDataset {
                    reason: "teacher dataset must be a JSON object".to_string(),
                })?;
        ensure_teacher_fields(
            object,
            &["schema_version", "contract", "traces"],
            "teacher dataset",
        )?;
        let schema_version = object
            .get("schema_version")
            .and_then(Value::as_u64)
            .ok_or_else(|| WarmStartExactJhError::InvalidTeacherDataset {
                reason: format!(
                    "teacher dataset requires schema_version={WARMSTART_TEACHER_CONTRACT_SCHEMA_VERSION}"
                ),
            })?;
        if schema_version != WARMSTART_TEACHER_CONTRACT_SCHEMA_VERSION {
            return Err(WarmStartExactJhError::InvalidTeacherDataset {
                reason: format!(
                    "teacher dataset schema_version must be {WARMSTART_TEACHER_CONTRACT_SCHEMA_VERSION}, got {schema_version}"
                ),
            });
        }
        let contract = parse_teacher_contract(object, schema_version)?;
        let traces_value =
            object
                .get("traces")
                .ok_or_else(|| WarmStartExactJhError::InvalidTeacherDataset {
                    reason: "teacher dataset requires a 'traces' array".to_string(),
                })?;
        let traces_array = traces_value.as_array().ok_or_else(|| {
            WarmStartExactJhError::InvalidTeacherDataset {
                reason: "teacher dataset field 'traces' must be an array".to_string(),
            }
        })?;
        let mut traces = Vec::with_capacity(traces_array.len());
        for (trace_index, trace_value) in traces_array.iter().enumerate() {
            traces.push(parse_teacher_trace(trace_value, trace_index)?);
        }
        if traces.is_empty() {
            return Err(WarmStartExactJhError::InvalidTeacherDataset {
                reason: "teacher dataset must contain at least one trace".to_string(),
            });
        }
        canonicalize_warmstart_teacher_traces(&mut traces);
        Ok(Self { contract, traces })
    }

    /// Convert this teacher dataset to its JSON representation.
    pub fn to_json_value(&self) -> Value {
        json!({
            "schema_version": WARMSTART_TEACHER_CONTRACT_SCHEMA_VERSION,
            "contract": teacher_contract_to_json_value(&self.contract),
            "traces": self.traces.iter().map(teacher_trace_to_json_value).collect::<Vec<_>>(),
        })
    }

    /// Count labels constructible from this dataset for the supplied horizon.
    pub fn label_count_for_horizon(&self, return_horizon: usize) -> usize {
        if return_horizon == 0 {
            return 0;
        }
        self.traces
            .iter()
            .map(|trace| {
                trace
                    .transitions
                    .len()
                    .saturating_add(1)
                    .saturating_sub(return_horizon)
            })
            .sum()
    }
}

fn teacher_contract_to_json_value(contract: &WarmStartExactJhTeacherContract) -> Value {
    json!({
        "task_fingerprint": contract.task_fingerprint.to_string(),
        "action_alphabet_size": contract.action_alphabet_size,
        "observation_bits": contract.observation_bits,
        "observation_stream_len": contract.observation_stream_len,
        "observation_key_mode": contract.observation_key_mode,
        "observation_adapter_spec_ref": contract.observation_adapter_spec_ref,
        "observation_adapter_content_crc32": contract.observation_adapter_content_crc32,
        "reward_bits": contract.reward_bits,
        "return_horizon": contract.return_horizon,
        "label_phase_period": contract.label_phase_period,
        "scalar_representation": contract.scalar_representation,
        "exact_reward_encoding_certificate": contract.exact_reward_encoding_certificate,
    })
}

fn teacher_trace_to_json_value(trace: &WarmStartExactJhTeacherTrace) -> Value {
    json!({
        "transitions": trace.transitions.iter().map(|transition| {
            json!({
                "action": transition.action,
                "observations": transition.observations,
                "reward": transition.reward,
            })
        }).collect::<Vec<_>>(),
    })
}

fn compare_warmstart_teacher_traces(
    left: &WarmStartExactJhTeacherTrace,
    right: &WarmStartExactJhTeacherTrace,
) -> Ordering {
    let mut left_iter = left.transitions.iter();
    let mut right_iter = right.transitions.iter();
    loop {
        match (left_iter.next(), right_iter.next()) {
            (Some(left_transition), Some(right_transition)) => {
                let ordering = left_transition
                    .action
                    .cmp(&right_transition.action)
                    .then_with(|| {
                        left_transition
                            .observations
                            .as_slice()
                            .cmp(right_transition.observations.as_slice())
                    })
                    .then_with(|| left_transition.reward.cmp(&right_transition.reward));
                if !ordering.is_eq() {
                    return ordering;
                }
            }
            (None, Some(_)) => return Ordering::Less,
            (Some(_), None) => return Ordering::Greater,
            (None, None) => return Ordering::Equal,
        }
    }
}

fn canonicalize_warmstart_teacher_traces(traces: &mut Vec<WarmStartExactJhTeacherTrace>) {
    traces.sort_by(compare_warmstart_teacher_traces);
    traces.dedup_by(|right, left| compare_warmstart_teacher_traces(left, right).is_eq());
}

fn insert_warmstart_teacher_trace_canonical(
    traces: &mut Vec<WarmStartExactJhTeacherTrace>,
    trace: WarmStartExactJhTeacherTrace,
) -> Option<usize> {
    match traces.binary_search_by(|existing| compare_warmstart_teacher_traces(existing, &trace)) {
        Ok(_) => None,
        Err(index) => {
            traces.insert(index, trace);
            Some(index)
        }
    }
}

/// Merge one teacher trace into `traces`, preserving deterministic lexicographic order.
pub fn merge_warmstart_teacher_trace_deterministic(
    traces: &mut Vec<WarmStartExactJhTeacherTrace>,
    trace: WarmStartExactJhTeacherTrace,
) -> bool {
    canonicalize_warmstart_teacher_traces(traces);
    insert_warmstart_teacher_trace_canonical(traces, trace).is_some()
}

/// Merge teacher traces into `traces`, returning the number and payload of inserted traces.
///
/// Existing and incoming traces are canonicalized as a deterministic set before
/// insertion, so callers may pass traces in arbitrary order.
pub fn merge_warmstart_teacher_traces_deterministic<I>(
    traces: &mut Vec<WarmStartExactJhTeacherTrace>,
    incoming: I,
) -> (usize, Vec<WarmStartExactJhTeacherTrace>)
where
    I: IntoIterator<Item = WarmStartExactJhTeacherTrace>,
{
    canonicalize_warmstart_teacher_traces(traces);
    let mut inserted = Vec::new();
    for trace in incoming {
        if let Some(index) = insert_warmstart_teacher_trace_canonical(traces, trace) {
            inserted.push(traces[index].clone());
        }
    }
    (inserted.len(), inserted)
}

/// Records normalized warm-start action/percept telemetry into one teacher trace.
#[derive(Clone, Debug, Default)]
pub struct WarmStartExactJhTraceRecorder {
    actions: BTreeMap<usize, Action>,
    percepts: BTreeMap<usize, (Vec<PerceptVal>, Reward)>,
}

impl WarmStartExactJhTraceRecorder {
    /// Construct an empty trace recorder.
    pub fn new() -> Self {
        Self::default()
    }

    /// Record an action at step `step`.
    pub fn record_action(
        &mut self,
        step: usize,
        action: Action,
    ) -> Result<(), WarmStartExactJhError> {
        if self.actions.insert(step, action).is_some() {
            return Err(invalid_telemetry(format!(
                "duplicate action record for step {step}"
            )));
        }
        Ok(())
    }

    /// Record a post-action percept at step `step`.
    pub fn record_percept(
        &mut self,
        step: usize,
        observations: &[PerceptVal],
        reward: Reward,
    ) -> Result<(), WarmStartExactJhError> {
        if self
            .percepts
            .insert(step, (observations.to_vec(), reward))
            .is_some()
        {
            return Err(invalid_telemetry(format!(
                "duplicate percept record for step {step}"
            )));
        }
        Ok(())
    }

    /// Convert the recorded telemetry into a validated teacher trace.
    pub fn into_teacher_trace(
        self,
        contract: &WarmStartExactJhTeacherContract,
        return_horizon: usize,
    ) -> Result<WarmStartExactJhTeacherTrace, WarmStartExactJhError> {
        if self.actions.len() != self.percepts.len() {
            return Err(invalid_telemetry(
                "action/percept record counts do not match",
            ));
        }
        let action_steps = self.actions.keys().copied().collect::<BTreeSet<_>>();
        let percept_steps = self.percepts.keys().copied().collect::<BTreeSet<_>>();
        if action_steps != percept_steps {
            return Err(invalid_telemetry("action/percept step sets do not match"));
        }
        ensure_dense_step_set(&action_steps, "recorded action/percept")?;
        let mut transitions = Vec::with_capacity(self.actions.len());
        for (step, action) in self.actions {
            let (observations, reward) = self
                .percepts
                .get(&step)
                .ok_or_else(|| invalid_telemetry(format!("missing percept for step {step}")))?;
            validate_warmstart_teacher_transition_against_contract(
                contract,
                action,
                observations,
                *reward,
            )?;
            transitions.push(WarmStartExactJhTransition {
                action,
                observations: observations.clone(),
                reward: *reward,
            });
        }
        if transitions.len() < return_horizon {
            return Err(invalid_telemetry(format!(
                "trace contains {} transitions but return_horizon is {return_horizon}",
                transitions.len()
            )));
        }
        Ok(WarmStartExactJhTeacherTrace { transitions })
    }
}

/// Build a normalized JSONL action record.
pub fn warmstart_jsonl_action_record(
    step: usize,
    action: Action,
    provenance: PlannerActionProvenance,
) -> Value {
    json!({
        "kind": "action",
        "t": step,
        "action": action,
        "provenance": provenance.as_str(),
    })
}

/// Build a normalized JSONL percept record.
pub fn warmstart_jsonl_percept_record(
    step: usize,
    observations: &[PerceptVal],
    reward: Reward,
) -> Value {
    json!({
        "kind": "percept",
        "t": step,
        "observations": observations,
        "reward": reward,
    })
}

fn invalid_telemetry(reason: impl Into<String>) -> WarmStartExactJhError {
    WarmStartExactJhError::InvalidTelemetry {
        reason: reason.into(),
    }
}

fn ensure_dense_step_set(
    steps: &BTreeSet<usize>,
    label: &str,
) -> Result<(), WarmStartExactJhError> {
    let Some(&first) = steps.first() else {
        return Ok(());
    };
    let mut previous = first;
    for &step in steps.iter().skip(1) {
        let expected = previous.checked_add(1).ok_or_else(|| {
            invalid_telemetry(format!(
                "{label} step set cannot be dense after usize::MAX step {previous}"
            ))
        })?;
        if step != expected {
            return Err(invalid_telemetry(format!(
                "{label} step set is not contiguous: expected step {expected} before step {step}"
            )));
        }
        previous = step;
    }
    Ok(())
}

fn ensure_jsonl_fields(
    object: &serde_json::Map<String, Value>,
    allowed: &[&str],
    line_number: usize,
) -> Result<(), WarmStartExactJhError> {
    for key in object.keys() {
        if !allowed.contains(&key.as_str()) {
            return Err(invalid_telemetry(format!(
                "line {line_number}: unknown field '{key}'"
            )));
        }
    }
    Ok(())
}

fn jsonl_required_u64(
    value: &Value,
    field: &str,
    line_number: usize,
) -> Result<u64, WarmStartExactJhError> {
    value.get(field).and_then(Value::as_u64).ok_or_else(|| {
        invalid_telemetry(format!("line {line_number}: missing u64 field '{field}'"))
    })
}

fn jsonl_required_i64(
    value: &Value,
    field: &str,
    line_number: usize,
) -> Result<i64, WarmStartExactJhError> {
    value.get(field).and_then(Value::as_i64).ok_or_else(|| {
        invalid_telemetry(format!("line {line_number}: missing i64 field '{field}'"))
    })
}

fn jsonl_required_observations(
    value: &Value,
    line_number: usize,
) -> Result<Vec<PerceptVal>, WarmStartExactJhError> {
    let observations = value
        .get("observations")
        .and_then(Value::as_array)
        .ok_or_else(|| {
            invalid_telemetry(format!("line {line_number}: missing observations array"))
        })?;
    observations
        .iter()
        .map(|item| {
            item.as_u64().ok_or_else(|| {
                invalid_telemetry(format!("line {line_number}: observation must be a u64"))
            })
        })
        .collect()
}

#[derive(Clone, Debug)]
struct JsonlActionRecord {
    action: Action,
    line_number: usize,
}

#[derive(Clone, Debug)]
struct JsonlPerceptRecord {
    observations: Vec<PerceptVal>,
    reward: Reward,
    line_number: usize,
}

#[derive(Clone, Debug, Default)]
struct JsonlStepRecords {
    action: Option<JsonlActionRecord>,
    percept: Option<JsonlPerceptRecord>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum JsonlTraceConvention {
    ActionThenPostPercept,
    DecisionPerceptThenAction,
}

fn infer_jsonl_trace_convention(
    steps: &BTreeMap<usize, JsonlStepRecords>,
) -> Result<JsonlTraceConvention, WarmStartExactJhError> {
    let mut saw_action_then_percept = false;
    let mut saw_percept_then_action = false;
    for records in steps.values() {
        let (Some(action), Some(percept)) = (&records.action, &records.percept) else {
            continue;
        };
        match action.line_number.cmp(&percept.line_number) {
            std::cmp::Ordering::Less => saw_action_then_percept = true,
            std::cmp::Ordering::Greater => saw_percept_then_action = true,
            std::cmp::Ordering::Equal => {
                return Err(invalid_telemetry(
                    "action and percept records cannot originate from the same JSONL line",
                ));
            }
        }
    }
    match (saw_action_then_percept, saw_percept_then_action) {
        (true, false) => Ok(JsonlTraceConvention::ActionThenPostPercept),
        (false, true) => Ok(JsonlTraceConvention::DecisionPerceptThenAction),
        (true, true) => Err(invalid_telemetry(
            "mixed JSONL action/percept conventions in one trace",
        )),
        (false, false) => Err(invalid_telemetry(
            "cannot infer JSONL action/percept convention",
        )),
    }
}

fn push_validated_jsonl_transition(
    transitions: &mut Vec<WarmStartExactJhTransition>,
    contract: &WarmStartExactJhTeacherContract,
    action_step: usize,
    action: Action,
    percept: &JsonlPerceptRecord,
) -> Result<(), WarmStartExactJhError> {
    validate_warmstart_teacher_transition_against_contract(
        contract,
        action,
        &percept.observations,
        percept.reward,
    )
    .map_err(|err| {
        invalid_telemetry(format!(
            "action step {action_step} paired with percept line {} violates contract: {err}",
            percept.line_number
        ))
    })?;
    transitions.push(WarmStartExactJhTransition {
        action,
        observations: percept.observations.clone(),
        reward: percept.reward,
    });
    Ok(())
}

fn jsonl_steps_into_teacher_trace(
    steps: BTreeMap<usize, JsonlStepRecords>,
    contract: &WarmStartExactJhTeacherContract,
    return_horizon: usize,
) -> Result<WarmStartExactJhTeacherTrace, WarmStartExactJhError> {
    let convention = infer_jsonl_trace_convention(&steps)?;
    let mut transitions = Vec::new();
    match convention {
        JsonlTraceConvention::ActionThenPostPercept => {
            let all_steps = steps.keys().copied().collect::<BTreeSet<_>>();
            ensure_dense_step_set(&all_steps, "action-then-percept JSONL")?;
            for (step, records) in &steps {
                let action = records.action.as_ref().ok_or_else(|| {
                    invalid_telemetry(format!("missing action record for step {step}"))
                })?;
                let percept = records.percept.as_ref().ok_or_else(|| {
                    invalid_telemetry(format!("missing percept record for step {step}"))
                })?;
                push_validated_jsonl_transition(
                    &mut transitions,
                    contract,
                    *step,
                    action.action,
                    percept,
                )?;
            }
        }
        JsonlTraceConvention::DecisionPerceptThenAction => {
            let action_steps = steps
                .iter()
                .filter_map(|(step, records)| records.action.as_ref().map(|_| *step))
                .collect::<BTreeSet<_>>();
            let Some(&first_action_step) = action_steps.first() else {
                return Err(invalid_telemetry("trace contains no action records"));
            };
            ensure_dense_step_set(&action_steps, "decision-percept JSONL action")?;
            let max_action_step = *action_steps
                .last()
                .ok_or_else(|| invalid_telemetry("trace contains no action records"))?;
            let final_percept_step = max_action_step.checked_add(1).ok_or_else(|| {
                invalid_telemetry(format!(
                    "decision-percept JSONL action step {max_action_step} cannot have a successor"
                ))
            })?;
            for step in steps.keys() {
                if *step < first_action_step || *step > final_percept_step {
                    return Err(invalid_telemetry(format!(
                        "unexpected JSONL step {step} outside dense decision trace domain {first_action_step}..={final_percept_step}"
                    )));
                }
                if !action_steps.contains(step) && *step != final_percept_step {
                    return Err(invalid_telemetry(format!(
                        "unexpected percept-only JSONL step {step} inside dense decision trace"
                    )));
                }
            }
            for step in &action_steps {
                let records = steps
                    .get(step)
                    .expect("action_steps contains only keys present in steps");
                let action = records
                    .action
                    .as_ref()
                    .expect("action_steps contains only records with actions");
                if records.percept.is_none() {
                    return Err(invalid_telemetry(format!(
                        "missing decision percept record for action step {step}"
                    )));
                }
                let next_step = step.checked_add(1).ok_or_else(|| {
                    invalid_telemetry(format!(
                        "decision-percept JSONL action step {step} cannot have a successor"
                    ))
                })?;
                let next_records = steps.get(&next_step).ok_or_else(|| {
                    invalid_telemetry(format!(
                        "missing post-action percept for action step {step}"
                    ))
                })?;
                let Some(percept) = next_records.percept.as_ref() else {
                    return Err(invalid_telemetry(format!(
                        "missing post-action percept for action step {step}"
                    )));
                };
                push_validated_jsonl_transition(
                    &mut transitions,
                    contract,
                    *step,
                    action.action,
                    percept,
                )?;
            }
        }
    }
    if transitions.len() < return_horizon {
        return Err(invalid_telemetry(format!(
            "trace contains {} complete transitions but return_horizon is {return_horizon}",
            transitions.len()
        )));
    }
    Ok(WarmStartExactJhTeacherTrace { transitions })
}

/// Parse a normalized planner JSONL trace into a warm-start teacher trace.
pub fn warmstart_teacher_trace_from_jsonl_reader<R: BufRead>(
    reader: R,
    contract: &WarmStartExactJhTeacherContract,
    return_horizon: usize,
) -> Result<WarmStartExactJhTeacherTrace, WarmStartExactJhError> {
    let mut steps = BTreeMap::<usize, JsonlStepRecords>::new();
    for (line_index, line) in reader.lines().enumerate() {
        let line_number = line_index + 1;
        let line = line.map_err(|err| invalid_telemetry(format!("line {line_number}: {err}")))?;
        if line.trim().is_empty() {
            return Err(invalid_telemetry(format!(
                "line {line_number}: empty JSONL records are not allowed"
            )));
        }
        let value = serde_json::from_str::<Value>(&line)
            .map_err(|err| invalid_telemetry(format!("line {line_number}: invalid JSON: {err}")))?;
        let object = value.as_object().ok_or_else(|| {
            invalid_telemetry(format!("line {line_number}: record must be an object"))
        })?;
        let kind = value
            .get("kind")
            .and_then(Value::as_str)
            .ok_or_else(|| invalid_telemetry(format!("line {line_number}: missing kind")))?;
        let step = usize::try_from(jsonl_required_u64(&value, "t", line_number)?)
            .map_err(|_| invalid_telemetry(format!("line {line_number}: t exceeds usize")))?;
        match kind {
            "action" => {
                ensure_jsonl_fields(object, &["kind", "t", "action", "provenance"], line_number)?;
                if let Some(provenance_value) = value.get("provenance") {
                    let provenance = provenance_value.as_str().ok_or_else(|| {
                        invalid_telemetry(format!(
                            "line {line_number}: action provenance must be a string"
                        ))
                    })?;
                    PlannerActionProvenance::from_jsonl_str(provenance)?;
                }
                let action = jsonl_required_u64(&value, "action", line_number)?;
                let entry = steps.entry(step).or_default();
                if entry
                    .action
                    .replace(JsonlActionRecord {
                        action,
                        line_number,
                    })
                    .is_some()
                {
                    return Err(invalid_telemetry(format!(
                        "duplicate action record for step {step}"
                    )));
                }
            }
            "percept" => {
                ensure_jsonl_fields(
                    object,
                    &["kind", "t", "observations", "reward"],
                    line_number,
                )?;
                let observations = jsonl_required_observations(&value, line_number)?;
                let reward = jsonl_required_i64(&value, "reward", line_number)?;
                let entry = steps.entry(step).or_default();
                if entry
                    .percept
                    .replace(JsonlPerceptRecord {
                        observations,
                        reward,
                        line_number,
                    })
                    .is_some()
                {
                    return Err(invalid_telemetry(format!(
                        "duplicate percept record for step {step}"
                    )));
                }
            }
            other => {
                return Err(invalid_telemetry(format!(
                    "line {line_number}: unknown record kind '{other}'"
                )));
            }
        }
    }
    jsonl_steps_into_teacher_trace(steps, contract, return_horizon)
}

/// Parse a normalized planner JSONL trace from bytes.
pub fn warmstart_teacher_trace_from_jsonl_slice(
    bytes: &[u8],
    contract: &WarmStartExactJhTeacherContract,
    return_horizon: usize,
) -> Result<WarmStartExactJhTeacherTrace, WarmStartExactJhError> {
    warmstart_teacher_trace_from_jsonl_reader(BufReader::new(bytes), contract, return_horizon)
}

/// Parse a normalized planner JSONL trace from a filesystem path.
pub fn warmstart_teacher_trace_from_jsonl_path(
    path: impl AsRef<Path>,
    contract: &WarmStartExactJhTeacherContract,
    return_horizon: usize,
) -> Result<WarmStartExactJhTeacherTrace, WarmStartExactJhError> {
    let file = File::open(path.as_ref()).map_err(|err| {
        invalid_telemetry(format!(
            "failed to open JSONL trace '{}': {err}",
            path.as_ref().display()
        ))
    })?;
    warmstart_teacher_trace_from_jsonl_reader(BufReader::new(file), contract, return_horizon)
}

/// Build the standalone same-task teacher contract for a compiled warm-start planner run.
pub fn standalone_warmstart_teacher_contract_for_compiled_planner_run(
    compiled: &CompiledPlannerRunSpec,
) -> Result<WarmStartExactJhTeacherContract, WarmStartExactJhError> {
    let interface = compiled.interface();
    let (return_horizon, label_phase_period, planner_simulations_per_step) =
        match compiled.controller() {
            CompiledPlannerController::AiqiWarmstartExactJh {
                return_horizon,
                label_phase_period,
                planner_simulations_per_step,
                ..
            } => (
                *return_horizon,
                *label_phase_period,
                *planner_simulations_per_step,
            ),
            _ => return Err(WarmStartExactJhError::ControllerKindMismatch),
        };
    if planner_simulations_per_step == 0 {
        return Err(WarmStartExactJhError::PlannerSimulationsZero);
    }
    if planner_simulations_per_step != 1 {
        return Err(WarmStartExactJhError::PlannerSimulationsUnsupported {
            configured: planner_simulations_per_step,
        });
    }
    let task_fingerprint =
        warmstart_exact_jh_planner_task_fingerprint(compiled).map_err(|err| {
            WarmStartExactJhError::InvalidTeacherDataset {
                reason: format!("failed to compute planner task fingerprint: {err}"),
            }
        })?;
    let (observation_adapter_content_crc32, exact_reward_encoding_certificate) =
        crate::aixi::warmstart_contract::standalone_teacher_provenance_crc32_pair(
            interface.observation_bits,
            interface.observation_stream_len.max(1),
            interface.reward_bits,
        )
        .map_err(|err| WarmStartExactJhError::InvalidTeacherDataset {
            reason: format!("failed to compute standalone teacher provenance: {err}"),
        })?;
    Ok(WarmStartExactJhTeacherContract {
        schema_version: WARMSTART_TEACHER_CONTRACT_SCHEMA_VERSION,
        task_fingerprint,
        action_alphabet_size: interface.agent_actions.get(),
        observation_bits: interface.observation_bits,
        observation_stream_len: interface.observation_stream_len.max(1),
        observation_key_mode: observation_key_mode_name(interface.observation_key_mode).to_string(),
        observation_adapter_spec_ref: WARMSTART_STANDALONE_OBSERVATION_ADAPTER_SPEC_REF.to_string(),
        observation_adapter_content_crc32,
        reward_bits: interface.reward_bits,
        return_horizon,
        label_phase_period,
        scalar_representation: WARMSTART_STANDALONE_SCALAR_REPRESENTATION.to_string(),
        exact_reward_encoding_certificate,
    })
}

/// Return the target exact-return horizon for a compiled warm-start planner run.
pub fn warmstart_target_return_horizon(
    compiled: &CompiledPlannerRunSpec,
) -> Result<usize, WarmStartExactJhError> {
    match compiled.controller() {
        CompiledPlannerController::AiqiWarmstartExactJh { return_horizon, .. } => {
            Ok(*return_horizon)
        }
        _ => Err(WarmStartExactJhError::ControllerKindMismatch),
    }
}

/// Read a warm-start teacher dataset from a filesystem path.
pub fn read_warmstart_teacher_dataset_path(
    path: impl AsRef<Path>,
) -> Result<WarmStartExactJhTeacherDataset, WarmStartExactJhError> {
    let path_ref = path.as_ref();
    let bytes =
        std::fs::read(path_ref).map_err(|err| WarmStartExactJhError::InvalidTeacherDataset {
            reason: format!(
                "failed to read teacher dataset '{}': {err}",
                path_ref.display()
            ),
        })?;
    WarmStartExactJhTeacherDataset::from_json_slice(&bytes)
}

/// Write a warm-start teacher dataset to a filesystem path.
pub fn write_warmstart_teacher_dataset_path(
    path: impl AsRef<Path>,
    dataset: &WarmStartExactJhTeacherDataset,
) -> Result<(), WarmStartExactJhError> {
    let path_ref = path.as_ref();
    let bytes = serde_json::to_vec_pretty(&dataset.to_json_value()).map_err(|err| {
        WarmStartExactJhError::InvalidTeacherDataset {
            reason: format!("failed to serialize teacher dataset: {err}"),
        }
    })?;
    std::fs::write(path_ref, bytes).map_err(|err| WarmStartExactJhError::InvalidTeacherDataset {
        reason: format!(
            "failed to write teacher dataset '{}': {err}",
            path_ref.display()
        ),
    })
}

/// Configuration parameters for the warm-start exact-J_H controller.
#[derive(Clone)]
#[non_exhaustive]
pub struct WarmStartExactJhConfig {
    /// Predictive backend used by the return-label model.
    pub rate_backend: RateBackend,
    /// Number of bits used to encode observations.
    pub observation_bits: usize,
    /// Number of observation symbols per environment step.
    pub observation_stream_len: usize,
    /// Number of bits used to encode rewards.
    pub reward_bits: usize,
    /// Number of valid actions.
    pub agent_actions: ActionAlphabet,
    /// Exact finite-horizon return length H.
    pub return_horizon: usize,
    /// Exact return-label alphabet cardinality.
    ///
    /// For this nonnegative-reward controller the valid shape is
    /// `return_horizon * max_reward + 1`; slack labels are rejected because
    /// they would not correspond to any exact finite-horizon return.
    pub return_bins: usize,
    /// Delayed-label phase period.
    pub label_phase_period: usize,
    /// Canonical direct-evaluator budget marker.
    ///
    /// Warm-start exact-\(J_H\) uses deterministic full return-law evaluation,
    /// not a simulation loop. The only truthful value is therefore `1`; larger
    /// values are rejected instead of being silently ignored.
    pub planner_simulations_per_step: usize,
    /// Bit-stream semantics used to adapt the return-label predictor.
    pub bit_stream_semantics: BitStreamSemantics,
    /// Optional deterministic RNG seed.
    pub random_seed: Option<u64>,
}

impl Default for WarmStartExactJhConfig {
    fn default() -> Self {
        Self {
            rate_backend: RateBackend::Ctw { depth: 8 },
            observation_bits: 1,
            observation_stream_len: 1,
            reward_bits: 1,
            agent_actions: ActionAlphabet::try_from_usize(2)
                .expect("default action alphabet must be non-zero"),
            return_horizon: 1,
            return_bins: 2,
            label_phase_period: 1,
            planner_simulations_per_step: 1,
            bit_stream_semantics: BitStreamSemantics::BinaryTokens,
            random_seed: None,
        }
    }
}

impl WarmStartExactJhConfig {
    fn canonical_planner_run_spec(&self) -> PlannerRunSpec {
        let mut spec = build_default_planner_run_spec(
            PlannerInterfaceConfig {
                observation_bits: self.observation_bits,
                observation_stream_len: self.observation_stream_len,
                observation_key_mode: crate::aixi::common::ObservationKeyMode::FullStream,
                reward_bits: self.reward_bits,
                agent_actions: self.agent_actions,
            },
            ControllerSpec::AiqiWarmstartExactJh(WarmStartExactJhControllerSpec {
                predictor: self.rate_backend.clone(),
                bit_stream_semantics: self.bit_stream_semantics,
                return_horizon: self.return_horizon,
                return_bins: self.return_bins,
                label_phase_period: self.label_phase_period,
                teacher_dataset_asset: "programmatic_warmstart_teacher".to_string(),
                planner_simulations_per_step: self.planner_simulations_per_step,
            }),
            self.random_seed,
        );
        spec.assets.push(AssetBinding {
            id: "programmatic_warmstart_teacher".to_string(),
            path: "programmatic_warmstart_teacher.json".to_string(),
        });
        spec
    }

    fn compile_planner_run_spec(&self) -> Result<CompiledPlannerRunSpec, WarmStartExactJhError> {
        self.canonical_planner_run_spec()
            .compile()
            .map_err(WarmStartExactJhError::from)
    }

    fn validate_runtime_invariants(&self) -> Result<(), WarmStartExactJhError> {
        if self.return_horizon == 0 {
            return Err(WarmStartExactJhError::ReturnHorizonZero);
        }
        if self.return_bins == 0 {
            return Err(WarmStartExactJhError::ReturnBinsZero);
        }
        if self.label_phase_period < self.return_horizon {
            return Err(WarmStartExactJhError::LabelPhasePeriodTooShort {
                label_phase_period: self.label_phase_period,
                return_horizon: self.return_horizon,
            });
        }
        if self.planner_simulations_per_step == 0 {
            return Err(WarmStartExactJhError::PlannerSimulationsZero);
        }
        if self.planner_simulations_per_step != 1 {
            return Err(WarmStartExactJhError::PlannerSimulationsUnsupported {
                configured: self.planner_simulations_per_step,
            });
        }
        let (min_reward, max_reward, _reward_offset) = reward_bounds_from_exact_return_bins(
            self.return_horizon,
            self.return_bins,
            self.reward_bits,
        )?;
        validate_exact_return_alphabet(
            min_reward,
            max_reward,
            self.return_horizon,
            self.return_bins,
        )?;
        validate_rate_backend(&self.rate_backend)
            .map_err(WarmStartExactJhError::InvalidRateBackend)?;
        let compiled = self
            .rate_backend
            .compile()
            .map_err(WarmStartExactJhError::Spec)?;
        if !compiled.supports_frozen_conditioning() {
            return Err(WarmStartExactJhError::UnsupportedRateBackend {
                reason: "warm-start exact-J_H strict mode requires frozen context conditioning; configured rate_backend does not provide strict frozen conditioning",
            });
        }
        Ok(())
    }

    /// Validate this configuration.
    pub fn validate(&self) -> Result<(), WarmStartExactJhError> {
        self.validate_runtime_invariants()?;
        self.compile_planner_run_spec().map(|_| ())
    }
}

#[derive(Clone)]
struct WarmStartExactJhRuntimeConfig {
    task_fingerprint: TaskFingerprint,
    observation_bits: usize,
    observation_stream_len: usize,
    observation_key_mode: &'static str,
    reward_bits: usize,
    agent_actions: ActionAlphabet,
    min_reward: Reward,
    max_reward: Reward,
    reward_offset: Reward,
    return_horizon: usize,
    return_bins: usize,
    label_phase_period: usize,
    planner_simulations_per_step: usize,
    random_seed: u64,
    provenance_policy: TeacherProvenancePolicy,
}

#[derive(Clone, Copy)]
enum TeacherProvenancePolicy {
    StandalonePlannerRun,
    ExternallyValidatedTunerBridge,
}

impl WarmStartExactJhRuntimeConfig {
    /// Build a canonicalized runtime contract from a compiled planner run.
    ///
    /// The resulting config captures the exact planner/interface contract that
    /// teacher traces are expected to match. This includes the planner task
    /// fingerprint plus runtime-visible interface dimensions that must remain
    /// aligned with any warm-start dataset.
    fn from_compiled(compiled: &CompiledPlannerRunSpec) -> Result<Self, WarmStartExactJhError> {
        let planner = compiled.canonical_spec();
        let interface = compiled.interface();
        let runtime = compiled.runtime();
        let (return_horizon, return_bins, label_phase_period, planner_simulations_per_step) =
            match compiled.controller() {
                CompiledPlannerController::AiqiWarmstartExactJh {
                    return_horizon,
                    return_bins,
                    label_phase_period,
                    planner_simulations_per_step,
                    ..
                } => (
                    *return_horizon,
                    *return_bins,
                    *label_phase_period,
                    *planner_simulations_per_step,
                ),
                _ => return Err(WarmStartExactJhError::ControllerKindMismatch),
            };
        if return_horizon == 0 {
            return Err(WarmStartExactJhError::ReturnHorizonZero);
        }
        if return_bins == 0 {
            return Err(WarmStartExactJhError::ReturnBinsZero);
        }
        if label_phase_period < return_horizon {
            return Err(WarmStartExactJhError::LabelPhasePeriodTooShort {
                label_phase_period,
                return_horizon,
            });
        }
        if planner_simulations_per_step == 0 {
            return Err(WarmStartExactJhError::PlannerSimulationsZero);
        }
        if planner_simulations_per_step != 1 {
            return Err(WarmStartExactJhError::PlannerSimulationsUnsupported {
                configured: planner_simulations_per_step,
            });
        }
        let (min_reward, max_reward, reward_offset) = reward_bounds_from_exact_return_bins(
            return_horizon,
            return_bins,
            interface.reward_bits,
        )?;
        validate_exact_return_alphabet(min_reward, max_reward, return_horizon, return_bins)?;
        let task_fingerprint =
            warmstart_exact_jh_planner_task_fingerprint(compiled).map_err(|err| {
                WarmStartExactJhError::InvalidTeacherDataset {
                    reason: format!("failed to compute planner task fingerprint: {err}"),
                }
            })?;
        let provenance_policy = match &planner.environment {
            EnvironmentSpec::Builtin {
                builtin: BuiltinEnvironmentSpec::TunerBridge,
            } => TeacherProvenancePolicy::ExternallyValidatedTunerBridge,
            _ => TeacherProvenancePolicy::StandalonePlannerRun,
        };
        Ok(Self {
            task_fingerprint,
            observation_bits: interface.observation_bits,
            observation_stream_len: interface.observation_stream_len.max(1),
            observation_key_mode: observation_key_mode_name(interface.observation_key_mode),
            reward_bits: interface.reward_bits,
            agent_actions: interface.agent_actions,
            min_reward,
            max_reward,
            reward_offset,
            return_horizon,
            return_bins,
            label_phase_period,
            planner_simulations_per_step,
            random_seed: resolve_random_seed(runtime.random_seed),
            provenance_policy,
        })
    }

    /// Validate that a parsed teacher contract matches the active planner runtime
    /// contract for this agent configuration.
    ///
    /// This comparison is authoritative for schema/contract compatibility and is
    /// intentionally strict on planner-task fields that influence trace encoding.
    fn validate_teacher_contract(
        &self,
        contract: &WarmStartExactJhTeacherContract,
    ) -> Result<(), WarmStartExactJhError> {
        validate_warmstart_teacher_contract_against_expectation(
            contract,
            &WarmStartTeacherContractExpectation {
                schema_version: WARMSTART_TEACHER_CONTRACT_SCHEMA_VERSION,
                task_fingerprint: self.task_fingerprint,
                action_alphabet_size: self.agent_actions.get(),
                observation_bits: self.observation_bits,
                observation_stream_len: self.observation_stream_len,
                observation_key_mode: self.observation_key_mode,
                reward_bits: self.reward_bits,
                return_horizon: self.return_horizon,
                label_phase_period: self.label_phase_period,
                validate_standalone_provenance: matches!(
                    self.provenance_policy,
                    TeacherProvenancePolicy::StandalonePlannerRun
                ),
            },
        )
    }
}

fn reward_bounds_from_exact_return_bins(
    return_horizon: usize,
    return_bins: usize,
    reward_bits: usize,
) -> Result<(Reward, Reward, Reward), WarmStartExactJhError> {
    let span = return_bins
        .checked_sub(1)
        .ok_or(WarmStartExactJhError::ExactReturnRangeOverflow)?;
    if span % return_horizon != 0 {
        return Err(WarmStartExactJhError::ReturnBinsNotExactHorizon {
            return_bins,
            return_horizon,
        });
    }
    let max_reward = i64::try_from(span / return_horizon)
        .map_err(|_| WarmStartExactJhError::ExactReturnRangeOverflow)?;
    validate_reward_encoding_bounds(0, max_reward, 0, reward_bits)?;
    Ok((0, max_reward, 0))
}

#[derive(Clone, Debug)]
struct StepRecord {
    action: Action,
    observations: Vec<PerceptVal>,
    reward: Reward,
}

struct PhaseModel {
    predictor: Box<dyn Predictor>,
    last_augmented_step: usize,
}

/// Warm-start exact finite-horizon objective controller.
pub struct WarmStartExactJhAgent {
    config: WarmStartExactJhRuntimeConfig,
    phases: Vec<PhaseModel>,
    steps: Vec<StepRecord>,
    return_labels_by_step: Vec<Option<u64>>,
    total_steps_observed: usize,
    action_bits: usize,
    return_label_codec: ReturnLabelCodec,
    teacher_label_count: usize,
    rng: RandomGenerator,
}

impl WarmStartExactJhAgent {
    /// Construct a new warm-start exact-J_H agent.
    pub fn new(
        config: WarmStartExactJhConfig,
        teacher: WarmStartExactJhTeacherDataset,
    ) -> Result<Self, WarmStartExactJhError> {
        config.validate_runtime_invariants()?;
        let compiled = config.compile_planner_run_spec()?;
        Self::from_compiled_planner_run(&compiled, teacher)
    }

    /// Construct from a compiled planner-run spec and same-task teacher data.
    pub fn from_compiled_planner_run(
        compiled: &CompiledPlannerRunSpec,
        teacher: WarmStartExactJhTeacherDataset,
    ) -> Result<Self, WarmStartExactJhError> {
        // Validate contract mismatch upfront so invalid metadata fails before costly
        // predictor allocation and before trace replay.
        let config = WarmStartExactJhRuntimeConfig::from_compiled(compiled)?;
        config.validate_teacher_contract(&teacher.contract)?;
        let (predictor, bit_stream_semantics) = match compiled.controller() {
            CompiledPlannerController::AiqiWarmstartExactJh {
                predictor,
                bit_stream_semantics,
                ..
            } => (predictor, *bit_stream_semantics),
            _ => return Err(WarmStartExactJhError::ControllerKindMismatch),
        };
        if !predictor.supports_frozen_conditioning() {
            return Err(WarmStartExactJhError::UnsupportedRateBackend {
                reason: "warm-start exact-J_H strict mode requires frozen context conditioning; configured rate_backend does not provide strict frozen conditioning",
            });
        }
        let action_bits = compiled.action_bits();
        let return_label_codec = ReturnLabelCodec::value_monotone(config.return_bins);
        let return_bits = return_label_codec.bits();
        let mut phases = Vec::with_capacity(config.label_phase_period);
        for _ in 0..config.label_phase_period {
            phases.push(PhaseModel {
                predictor: build_aiqi_predictor(predictor, return_bits, bit_stream_semantics)
                    .map_err(WarmStartExactJhError::Predictor)?,
                last_augmented_step: 0,
            });
        }
        let rng = RandomGenerator::from_seed(config.random_seed);
        let mut agent = Self {
            action_bits,
            return_label_codec,
            phases,
            steps: Vec::new(),
            return_labels_by_step: Vec::new(),
            total_steps_observed: 0,
            teacher_label_count: 0,
            rng,
            config,
        };
        agent.warm_start_from_teacher(&teacher)?;
        Ok(agent)
    }

    /// Number of transitions incorporated from live interaction.
    pub fn steps_observed(&self) -> usize {
        self.total_steps_observed
    }

    /// Number of warm-start labels incorporated from the teacher dataset.
    pub fn teacher_label_count(&self) -> usize {
        self.teacher_label_count
    }

    /// Extract the current live same-task trajectory as a teacher trace.
    ///
    /// The trace is admissible under the same runtime validator used for
    /// teacher datasets because it was produced through `observe_transition`.
    pub fn same_task_live_trace(&self) -> Option<WarmStartExactJhTeacherTrace> {
        if self.steps.len() < self.config.return_horizon {
            return None;
        }
        Some(WarmStartExactJhTeacherTrace {
            transitions: self
                .steps
                .iter()
                .map(|step| WarmStartExactJhTransition {
                    action: step.action,
                    observations: step.observations.clone(),
                    reward: step.reward,
                })
                .collect(),
        })
    }

    /// Configured action alphabet cardinality.
    pub fn num_actions(&self) -> ActionAlphabet {
        self.config.agent_actions
    }

    /// Canonical direct-evaluator budget marker.
    ///
    /// This is always `1`; warm-start exact-\(J_H\) has no simulation loop.
    pub fn planner_simulations_per_step(&self) -> usize {
        self.config.planner_simulations_per_step
    }

    /// Resolved deterministic seed.
    pub fn resolved_random_seed(&self) -> u64 {
        self.config.random_seed
    }

    pub(crate) fn reseed_random(&mut self, seed: u64) {
        self.config.random_seed = seed;
        self.rng = RandomGenerator::from_seed(seed);
    }

    /// Select the next greedy action from the current exact-return model.
    pub fn get_planned_action(&mut self) -> Action {
        let q_values = self.estimate_q_values();
        argmax_with_fixed_tie_break(&q_values) as u64
    }

    /// Estimate exact finite-horizon action values at the current decision state.
    pub fn estimate_action_values(&mut self) -> Result<Vec<f64>, WarmStartExactJhError> {
        Ok(self.estimate_q_values())
    }

    /// Select the next action with optional epsilon exploration.
    ///
    /// Warm-start exact-\(J_H\) has no baseline exploration parameter; this
    /// method's argument is the entire exploration probability.
    pub fn get_planned_action_with_extra_exploration(&mut self, extra_exploration: f64) -> Action {
        self.get_planned_action_with_extra_exploration_flag(extra_exploration)
            .0
    }

    /// Select the next action with optional epsilon exploration and return whether exploration fired.
    ///
    /// Warm-start exact-\(J_H\) has no baseline exploration parameter; this
    /// method's argument is the entire exploration probability.
    pub fn get_planned_action_with_extra_exploration_flag(
        &mut self,
        extra_exploration: f64,
    ) -> (Action, bool) {
        let extra = extra_exploration.clamp(0.0, 1.0);
        if extra > 0.0 && self.rng.gen_bool(extra) {
            (
                self.rng.gen_range(self.config.agent_actions.get()) as u64,
                true,
            )
        } else {
            (self.get_planned_action(), false)
        }
    }

    /// Record one live environment transition.
    pub fn observe_transition(
        &mut self,
        action: Action,
        observations: &[PerceptVal],
        reward: Reward,
    ) -> Result<(), WarmStartExactJhError> {
        self.validate_transition(action, observations, reward)?;
        self.steps.push(StepRecord {
            action,
            observations: observations.to_vec(),
            reward,
        });
        self.total_steps_observed = self.total_steps_observed.saturating_add(1);
        self.return_labels_by_step.push(None);
        self.maybe_learn_new_return()
    }

    /// Warm-start from teacher traces (trace payload validation only).
    ///
    /// Contract-level validation is performed before this method is called in the
    /// constructor hot path; this keeps construction cheap on malformed contracts.
    fn warm_start_from_teacher(
        &mut self,
        teacher: &WarmStartExactJhTeacherDataset,
    ) -> Result<(), WarmStartExactJhError> {
        let mut label_count = 0usize;
        for trace in &teacher.traces {
            self.validate_teacher_trace(trace)?;
            label_count = label_count.saturating_add(self.commit_teacher_trace(trace)?);
            for phase in &mut self.phases {
                phase
                    .predictor
                    .reset_conditioning_history()
                    .map_err(|reason| WarmStartExactJhError::PredictorConditioningReset {
                        reason,
                    })?;
            }
        }
        if label_count == 0 {
            return Err(WarmStartExactJhError::InvalidTeacherDataset {
                reason: "teacher dataset did not contain any complete H-step labels".to_string(),
            });
        }
        self.teacher_label_count = label_count;
        Ok(())
    }

    fn validate_teacher_trace(
        &self,
        trace: &WarmStartExactJhTeacherTrace,
    ) -> Result<(), WarmStartExactJhError> {
        for step in &trace.transitions {
            self.validate_transition(step.action, &step.observations, step.reward)?;
        }
        Ok(())
    }

    fn validate_transition(
        &self,
        action: Action,
        observations: &[PerceptVal],
        reward: Reward,
    ) -> Result<(), WarmStartExactJhError> {
        validate_runtime_transition(&self.config, action, observations, reward)
    }

    fn commit_teacher_trace(
        &mut self,
        trace: &WarmStartExactJhTeacherTrace,
    ) -> Result<usize, WarmStartExactJhError> {
        let labels = exact_return_labels_for_trace(&self.config, &trace.transitions)?;
        let mut committed = 0usize;
        for phase in 0..self.config.label_phase_period {
            let model = &mut self.phases[phase];
            for (idx0, step) in trace.transitions.iter().enumerate() {
                let step_index = idx0 + 1;
                push_action_tokens_commit_history(
                    model.predictor.as_mut(),
                    step.action,
                    self.action_bits,
                );
                if step_index % self.config.label_phase_period == phase
                    && let Some(label) = labels[idx0]
                {
                    self.return_label_codec
                        .push_label_commit(model.predictor.as_mut(), label);
                    committed = committed.saturating_add(1);
                }
                push_percept_tokens_commit_history(
                    &self.config,
                    model.predictor.as_mut(),
                    &step.observations,
                    step.reward,
                )?;
            }
        }
        Ok(committed)
    }

    fn maybe_learn_new_return(&mut self) -> Result<(), WarmStartExactJhError> {
        let t = self.total_steps_observed;
        let h = self.config.return_horizon;
        if t < h {
            return Ok(());
        }
        let start_step = t + 1 - h;
        let label = self.compute_return_label(start_step)?;
        self.return_labels_by_step[start_step - 1] = Some(label);
        let phase = start_step % self.config.label_phase_period;
        self.advance_phase_model_to_step(phase, start_step)
    }

    fn estimate_q_values(&mut self) -> Vec<f64> {
        let predictors = self
            .action_conditioned_predictors()
            .expect("warm-start history encoding invariant");
        let config = self.config.clone();
        let codec = self.return_label_codec;
        predictors
            .into_iter()
            .map(|mut predictor| {
                let distribution = predict_return_law(
                    predictor.as_mut(),
                    codec,
                    ReturnPrefixUpdate::Training,
                    ReturnLawEvaluator::SharedPrefix,
                );
                expected_decoded_return(&distribution.probabilities, |label| {
                    exact_return_from_label(&config, label)
                })
            })
            .collect()
    }

    fn action_conditioned_predictors(
        &mut self,
    ) -> Result<Vec<Box<dyn Predictor>>, WarmStartExactJhError> {
        let step = self.total_steps_observed + 1;
        let phase = step % self.config.label_phase_period;
        let config = &self.config;
        let steps = &self.steps;
        let return_labels_by_step = &self.return_labels_by_step;
        let action_bits = self.action_bits;
        let token_ctx = WarmStartAugmentedTokenContext {
            config,
            steps,
            return_labels_by_step,
            action_bits,
            return_label_codec: self.return_label_codec,
            phase,
        };
        let mut predictors = Vec::with_capacity(self.config.agent_actions.get());
        let mut pushed_history = 0usize;
        {
            let model = &mut self.phases[phase];
            let start = model.last_augmented_step + 1;
            let end = step.saturating_sub(1);
            if start <= end {
                for idx in start..=end {
                    pushed_history +=
                        push_step_tokens_history(&token_ctx, model.predictor.as_mut(), idx)?;
                }
            }
            for action in 0..self.config.agent_actions.get() {
                let pushed_action =
                    push_encoded_bits_history(model.predictor.as_mut(), action as u64, action_bits);
                predictors.push(model.predictor.boxed_clone());
                pop_history_bits(model.predictor.as_mut(), pushed_action);
            }
            pop_history_bits(model.predictor.as_mut(), pushed_history);
        }
        Ok(predictors)
    }

    fn advance_phase_model_to_step(
        &mut self,
        phase: usize,
        target_step: usize,
    ) -> Result<(), WarmStartExactJhError> {
        let token_ctx = WarmStartAugmentedTokenContext {
            config: &self.config,
            steps: &self.steps,
            return_labels_by_step: &self.return_labels_by_step,
            action_bits: self.action_bits,
            return_label_codec: self.return_label_codec,
            phase,
        };
        let model = &mut self.phases[phase];
        if target_step <= model.last_augmented_step {
            return Ok(());
        }
        let start = model.last_augmented_step + 1;
        for idx in start..=target_step {
            push_augmented_step_tokens_commit(&token_ctx, model.predictor.as_mut(), idx)?;
        }
        model.last_augmented_step = target_step;
        Ok(())
    }

    fn compute_return_label(&self, start_step: usize) -> Result<u64, WarmStartExactJhError> {
        let mut total = 0i128;
        for offset in 0..self.config.return_horizon {
            let idx = start_step + offset;
            let step =
                self.steps
                    .get(idx - 1)
                    .ok_or(WarmStartExactJhError::HistoryIndexOutOfRange {
                        global_step: idx,
                        total_steps_observed: self.total_steps_observed,
                    })?;
            total += step.reward as i128;
        }
        label_for_exact_return(&self.config, total)
    }
}

/// Error type for warm-start exact-J_H agent construction and execution.
#[derive(Debug)]
#[non_exhaustive]
pub enum WarmStartExactJhError {
    /// The compiled planner controller was not a warm-start exact-J_H controller.
    ControllerKindMismatch,
    /// The return horizon was zero.
    ReturnHorizonZero,
    /// The return-label alphabet was empty.
    ReturnBinsZero,
    /// The label phase period was smaller than the return horizon.
    LabelPhasePeriodTooShort {
        /// Configured label phase period.
        label_phase_period: usize,
        /// Configured return horizon.
        return_horizon: usize,
    },
    /// The direct-evaluator budget marker was zero.
    PlannerSimulationsZero,
    /// The direct-evaluator budget marker was not the canonical value.
    PlannerSimulationsUnsupported {
        /// Configured unsupported value.
        configured: usize,
    },
    /// The exact return range cannot be represented by `return_bins`.
    ReturnBinsTooSmall {
        /// Required exact labels.
        required: u128,
        /// Configured labels.
        configured: usize,
    },
    /// `return_bins` would leave unreachable exact-return labels.
    ReturnBinsNotExactHorizon {
        /// Configured label count.
        return_bins: usize,
        /// Configured return horizon.
        return_horizon: usize,
    },
    /// The exact return range overflowed the supported integer domain.
    ExactReturnRangeOverflow,
    /// The configured reward range is not representable.
    RewardEncoding(RewardEncodingError),
    /// Invalid rate backend.
    InvalidRateBackend(crate::error::InfotheoryError),
    /// Unsupported rate backend semantics.
    UnsupportedRateBackend {
        /// Human-readable reason.
        reason: &'static str,
    },
    /// Spec compilation failed.
    Spec(SpecError),
    /// Predictor construction failed.
    Predictor(PredictorBuildError),
    /// Predictor conditioning history reset failed.
    PredictorConditioningReset {
        /// Human-readable reason.
        reason: String,
    },
    /// Teacher dataset was malformed or semantically inadmissible.
    InvalidTeacherDataset {
        /// Human-readable reason.
        reason: String,
    },
    /// JSONL telemetry was malformed.
    InvalidTelemetry {
        /// Human-readable reason.
        reason: String,
    },
    /// Action outside the configured alphabet.
    ActionOutOfRange {
        /// Invalid action.
        action: Action,
        /// Configured alphabet.
        agent_actions: ActionAlphabet,
    },
    /// Observation stream length mismatch.
    ObservationStreamLengthMismatch {
        /// Expected stream length.
        expected: usize,
        /// Actual stream length.
        actual: usize,
    },
    /// Observation value exceeded its bit width.
    ObservationValueOutOfRange {
        /// Invalid observation.
        observation: PerceptVal,
        /// Configured observation bits.
        observation_bits: usize,
        /// Maximum representable value.
        maximum: PerceptVal,
    },
    /// Reward outside the configured exact reward range.
    RewardOutOfRange {
        /// Invalid reward.
        reward: Reward,
        /// Minimum reward.
        min_reward: Reward,
        /// Maximum reward.
        max_reward: Reward,
    },
    /// Live history index was unavailable.
    HistoryIndexOutOfRange {
        /// Requested 1-based step index.
        global_step: usize,
        /// Total observed steps.
        total_steps_observed: usize,
    },
    /// A delayed label was required but absent.
    MissingReturnLabel {
        /// Step index.
        step: usize,
        /// Phase index.
        phase: usize,
    },
}

impl fmt::Display for WarmStartExactJhError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ControllerKindMismatch => {
                f.write_str("compiled controller kind is not aiqi_warmstart_exact_jh")
            }
            Self::ReturnHorizonZero => f.write_str("return_horizon must be >= 1"),
            Self::ReturnBinsZero => f.write_str("return_bins must be >= 1"),
            Self::LabelPhasePeriodTooShort {
                label_phase_period,
                return_horizon,
            } => write!(
                f,
                "label_phase_period ({label_phase_period}) must be >= return_horizon ({return_horizon})"
            ),
            Self::PlannerSimulationsZero => f.write_str(
                "planner_simulations_per_step must be exactly 1 for warm-start exact-J_H direct evaluation",
            ),
            Self::PlannerSimulationsUnsupported { configured } => write!(
                f,
                "planner_simulations_per_step must be exactly 1 for warm-start exact-J_H direct evaluation, got {configured}"
            ),
            Self::ReturnBinsTooSmall {
                required,
                configured,
            } => write!(
                f,
                "return_bins too small for exact J_H labels: required {required}, configured {configured}"
            ),
            Self::ReturnBinsNotExactHorizon {
                return_bins,
                return_horizon,
            } => write!(
                f,
                "return_bins must be exactly H * max_reward + 1 for warm-start exact-J_H; got return_bins={return_bins}, return_horizon={return_horizon}"
            ),
            Self::ExactReturnRangeOverflow => {
                f.write_str("exact finite-horizon return range overflowed supported integer domain")
            }
            Self::RewardEncoding(err) => write!(f, "{err}"),
            Self::InvalidRateBackend(err) => write!(f, "invalid rate_backend: {err}"),
            Self::UnsupportedRateBackend { reason } => f.write_str(reason),
            Self::Spec(err) => write!(f, "{err}"),
            Self::Predictor(err) => write!(f, "failed to construct predictor: {err}"),
            Self::PredictorConditioningReset { reason } => {
                write!(
                    f,
                    "failed to reset predictor conditioning history: {reason}"
                )
            }
            Self::InvalidTeacherDataset { reason } => {
                write!(f, "invalid teacher dataset: {reason}")
            }
            Self::InvalidTelemetry { reason } => write!(f, "invalid telemetry: {reason}"),
            Self::ActionOutOfRange {
                action,
                agent_actions,
            } => write!(
                f,
                "action {action} is outside configured action alphabet {agent_actions}"
            ),
            Self::ObservationStreamLengthMismatch { expected, actual } => write!(
                f,
                "observation stream length mismatch: expected {expected}, got {actual}"
            ),
            Self::ObservationValueOutOfRange {
                observation,
                observation_bits,
                maximum,
            } => write!(
                f,
                "observation value {observation} does not fit observation_bits={observation_bits} (max={maximum})"
            ),
            Self::RewardOutOfRange {
                reward,
                min_reward,
                max_reward,
            } => write!(
                f,
                "reward {reward} outside configured range [{min_reward}, {max_reward}]"
            ),
            Self::HistoryIndexOutOfRange {
                global_step,
                total_steps_observed,
            } => write!(
                f,
                "global step {global_step} out of observed history range [1, {total_steps_observed}]"
            ),
            Self::MissingReturnLabel { step, phase } => {
                write!(
                    f,
                    "missing exact return label for step {step} in phase {phase}"
                )
            }
        }
    }
}

impl Error for WarmStartExactJhError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::RewardEncoding(err) => Some(err),
            Self::InvalidRateBackend(err) => Some(err),
            Self::Spec(err) => Some(err),
            Self::Predictor(err) => Some(err),
            Self::PredictorConditioningReset { .. } => None,
            _ => None,
        }
    }
}

impl From<RewardEncodingError> for WarmStartExactJhError {
    fn from(value: RewardEncodingError) -> Self {
        Self::RewardEncoding(value)
    }
}

impl From<SpecError> for WarmStartExactJhError {
    fn from(value: SpecError) -> Self {
        Self::Spec(value)
    }
}

fn parse_teacher_contract(
    object: &serde_json::Map<String, Value>,
    schema_version: u64,
) -> Result<WarmStartExactJhTeacherContract, WarmStartExactJhError> {
    let contract = object
        .get("contract")
        .and_then(Value::as_object)
        .ok_or_else(|| WarmStartExactJhError::InvalidTeacherDataset {
            reason: "teacher dataset requires a 'contract' object".to_string(),
        })?;
    ensure_teacher_fields(
        contract,
        &[
            "task_fingerprint",
            "action_alphabet_size",
            "observation_bits",
            "observation_stream_len",
            "observation_key_mode",
            "observation_adapter_spec_ref",
            "observation_adapter_content_crc32",
            "reward_bits",
            "return_horizon",
            "label_phase_period",
            "scalar_representation",
            "exact_reward_encoding_certificate",
        ],
        "contract",
    )?;
    Ok(WarmStartExactJhTeacherContract {
        schema_version,
        task_fingerprint: required_teacher_task_fingerprint(contract, "task_fingerprint")?,
        action_alphabet_size: required_teacher_usize(contract, "action_alphabet_size")?,
        observation_bits: required_teacher_usize(contract, "observation_bits")?,
        observation_stream_len: required_teacher_usize(contract, "observation_stream_len")?,
        observation_key_mode: required_teacher_string(contract, "observation_key_mode")?,
        observation_adapter_spec_ref: required_teacher_string(
            contract,
            "observation_adapter_spec_ref",
        )?,
        observation_adapter_content_crc32: required_teacher_string(
            contract,
            "observation_adapter_content_crc32",
        )?,
        reward_bits: required_teacher_usize(contract, "reward_bits")?,
        return_horizon: required_teacher_usize(contract, "return_horizon")?,
        label_phase_period: required_teacher_usize(contract, "label_phase_period")?,
        scalar_representation: required_teacher_string(contract, "scalar_representation")?,
        exact_reward_encoding_certificate: required_teacher_string(
            contract,
            "exact_reward_encoding_certificate",
        )?,
    })
}

fn required_teacher_task_fingerprint(
    object: &serde_json::Map<String, Value>,
    field: &str,
) -> Result<TaskFingerprint, WarmStartExactJhError> {
    let value = object.get(field).and_then(Value::as_str).ok_or_else(|| {
        WarmStartExactJhError::InvalidTeacherDataset {
            reason: format!("teacher contract field '{field}' must be a string"),
        }
    })?;
    TaskFingerprint::parse_hex(value).ok_or_else(|| WarmStartExactJhError::InvalidTeacherDataset {
        reason: format!(
            "teacher contract field '{field}' must be a 64-digit lowercase hexadecimal SHA-256 digest, got '{value}'"
        ),
    })
}

fn required_teacher_string(
    object: &serde_json::Map<String, Value>,
    field: &str,
) -> Result<String, WarmStartExactJhError> {
    object
        .get(field)
        .and_then(Value::as_str)
        .map(str::to_string)
        .ok_or_else(|| WarmStartExactJhError::InvalidTeacherDataset {
            reason: format!("teacher contract field '{field}' must be a string"),
        })
}

fn required_teacher_usize(
    object: &serde_json::Map<String, Value>,
    field: &str,
) -> Result<usize, WarmStartExactJhError> {
    let value = object.get(field).and_then(Value::as_u64).ok_or_else(|| {
        WarmStartExactJhError::InvalidTeacherDataset {
            reason: format!("teacher contract field '{field}' must be an unsigned integer"),
        }
    })?;
    usize::try_from(value).map_err(|_| WarmStartExactJhError::InvalidTeacherDataset {
        reason: format!("teacher contract field '{field}' does not fit usize"),
    })
}

fn parse_teacher_trace(
    value: &Value,
    trace_index: usize,
) -> Result<WarmStartExactJhTeacherTrace, WarmStartExactJhError> {
    let object = value
        .as_object()
        .ok_or_else(|| WarmStartExactJhError::InvalidTeacherDataset {
            reason: format!("traces[{trace_index}] must be an object with transitions"),
        })?;
    ensure_teacher_fields(object, &["transitions"], &format!("traces[{trace_index}]"))?;
    let transitions_value = object
        .get("transitions")
        .and_then(Value::as_array)
        .ok_or_else(|| WarmStartExactJhError::InvalidTeacherDataset {
            reason: format!("traces[{trace_index}].transitions must be an array"),
        })?;
    let mut transitions = Vec::with_capacity(transitions_value.len());
    for (step_index, transition) in transitions_value.iter().enumerate() {
        transitions.push(parse_teacher_transition(
            transition,
            trace_index,
            step_index,
        )?);
    }
    Ok(WarmStartExactJhTeacherTrace { transitions })
}

fn parse_teacher_transition(
    value: &Value,
    trace_index: usize,
    step_index: usize,
) -> Result<WarmStartExactJhTransition, WarmStartExactJhError> {
    let object = value
        .as_object()
        .ok_or_else(|| WarmStartExactJhError::InvalidTeacherDataset {
            reason: format!("traces[{trace_index}].transitions[{step_index}] must be an object"),
        })?;
    ensure_teacher_fields(
        object,
        &["action", "observations", "reward"],
        &format!("traces[{trace_index}].transitions[{step_index}]"),
    )?;
    let action = object
        .get("action")
        .and_then(Value::as_u64)
        .ok_or_else(|| WarmStartExactJhError::InvalidTeacherDataset {
            reason: format!(
                "traces[{trace_index}].transitions[{step_index}].action must be an integer"
            ),
        })?;
    let observations_value =
        object
            .get("observations")
            .ok_or_else(|| WarmStartExactJhError::InvalidTeacherDataset {
                reason: format!(
                    "traces[{trace_index}].transitions[{step_index}] requires observations"
                ),
            })?;
    let observations = observations_value
        .as_array()
        .ok_or_else(|| WarmStartExactJhError::InvalidTeacherDataset {
            reason: format!(
                "traces[{trace_index}].transitions[{step_index}].observations must be an array"
            ),
        })?
        .iter()
        .enumerate()
        .map(|(obs_index, obs)| {
            obs.as_u64().ok_or_else(|| WarmStartExactJhError::InvalidTeacherDataset {
                reason: format!(
                    "traces[{trace_index}].transitions[{step_index}].observations[{obs_index}] must be an integer"
                ),
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
    let reward = object
        .get("reward")
        .and_then(Value::as_i64)
        .ok_or_else(|| WarmStartExactJhError::InvalidTeacherDataset {
            reason: format!(
                "traces[{trace_index}].transitions[{step_index}].reward must be an integer"
            ),
        })?;
    Ok(WarmStartExactJhTransition {
        action,
        observations,
        reward,
    })
}

fn ensure_teacher_fields(
    object: &serde_json::Map<String, Value>,
    allowed: &[&str],
    label: &str,
) -> Result<(), WarmStartExactJhError> {
    for key in object.keys() {
        if !allowed.contains(&key.as_str()) {
            return Err(WarmStartExactJhError::InvalidTeacherDataset {
                reason: format!("{label} contains unknown teacher field '{key}'"),
            });
        }
    }
    Ok(())
}

fn validate_exact_return_alphabet(
    min_reward: Reward,
    max_reward: Reward,
    return_horizon: usize,
    return_bins: usize,
) -> Result<(), WarmStartExactJhError> {
    let min_return = (min_reward as i128)
        .checked_mul(return_horizon as i128)
        .ok_or(WarmStartExactJhError::ExactReturnRangeOverflow)?;
    let max_return = (max_reward as i128)
        .checked_mul(return_horizon as i128)
        .ok_or(WarmStartExactJhError::ExactReturnRangeOverflow)?;
    let span = max_return
        .checked_sub(min_return)
        .and_then(|value| value.checked_add(1))
        .ok_or(WarmStartExactJhError::ExactReturnRangeOverflow)?;
    let required =
        u128::try_from(span).map_err(|_| WarmStartExactJhError::ExactReturnRangeOverflow)?;
    if required > return_bins as u128 {
        return Err(WarmStartExactJhError::ReturnBinsTooSmall {
            required,
            configured: return_bins,
        });
    }
    Ok(())
}

fn exact_return_labels_for_trace(
    config: &WarmStartExactJhRuntimeConfig,
    steps: &[WarmStartExactJhTransition],
) -> Result<Vec<Option<u64>>, WarmStartExactJhError> {
    let mut labels = vec![None; steps.len()];
    if config.return_horizon == 0 || steps.len() < config.return_horizon {
        return Ok(labels);
    }

    let horizon = config.return_horizon;
    let mut window_sum = 0_i128;
    for (index, step) in steps.iter().enumerate() {
        window_sum += step.reward as i128;
        if index >= horizon {
            window_sum -= steps[index - horizon].reward as i128;
        }
        if index + 1 >= horizon {
            let start0 = index + 1 - horizon;
            labels[start0] = Some(label_for_exact_return(config, window_sum)?);
        }
    }
    Ok(labels)
}

fn label_for_exact_return(
    config: &WarmStartExactJhRuntimeConfig,
    exact_return: i128,
) -> Result<u64, WarmStartExactJhError> {
    let min_return = (config.min_reward as i128)
        .checked_mul(config.return_horizon as i128)
        .ok_or(WarmStartExactJhError::ExactReturnRangeOverflow)?;
    let label = exact_return
        .checked_sub(min_return)
        .ok_or(WarmStartExactJhError::ExactReturnRangeOverflow)?;
    if label < 0 || label >= config.return_bins as i128 {
        return Err(WarmStartExactJhError::ReturnBinsTooSmall {
            required: (label + 1).max(0) as u128,
            configured: config.return_bins,
        });
    }
    u64::try_from(label).map_err(|_| WarmStartExactJhError::ExactReturnRangeOverflow)
}

fn exact_return_from_label(config: &WarmStartExactJhRuntimeConfig, label: u64) -> f64 {
    let min_return = (config.min_reward as i128) * (config.return_horizon as i128);
    (min_return + label as i128) as f64
}

fn validate_runtime_transition(
    config: &WarmStartExactJhRuntimeConfig,
    action: Action,
    observations: &[PerceptVal],
    reward: Reward,
) -> Result<(), WarmStartExactJhError> {
    if action as usize >= config.agent_actions.get() {
        return Err(WarmStartExactJhError::ActionOutOfRange {
            action,
            agent_actions: config.agent_actions,
        });
    }
    if observations.len() != config.observation_stream_len {
        return Err(WarmStartExactJhError::ObservationStreamLengthMismatch {
            expected: config.observation_stream_len,
            actual: observations.len(),
        });
    }
    let obs_max = max_value_for_bits(config.observation_bits);
    for &observation in observations {
        if observation > obs_max {
            return Err(WarmStartExactJhError::ObservationValueOutOfRange {
                observation,
                observation_bits: config.observation_bits,
                maximum: obs_max,
            });
        }
    }
    if reward < config.min_reward || reward > config.max_reward {
        return Err(WarmStartExactJhError::RewardOutOfRange {
            reward,
            min_reward: config.min_reward,
            max_reward: config.max_reward,
        });
    }
    Ok(())
}

struct WarmStartAugmentedTokenContext<'a> {
    config: &'a WarmStartExactJhRuntimeConfig,
    steps: &'a [StepRecord],
    return_labels_by_step: &'a [Option<u64>],
    action_bits: usize,
    return_label_codec: ReturnLabelCodec,
    phase: usize,
}

fn push_augmented_step_tokens_commit(
    ctx: &WarmStartAugmentedTokenContext<'_>,
    predictor: &mut dyn Predictor,
    idx: usize,
) -> Result<usize, WarmStartExactJhError> {
    let step = &ctx.steps[idx - 1];
    let mut pushed = 0usize;
    pushed += push_action_tokens_commit_history(predictor, step.action, ctx.action_bits);
    if idx % ctx.config.label_phase_period == ctx.phase {
        let label = ctx.return_labels_by_step[idx - 1].ok_or(
            WarmStartExactJhError::MissingReturnLabel {
                step: idx,
                phase: ctx.phase,
            },
        )?;
        pushed += ctx.return_label_codec.push_label_commit(predictor, label);
    }
    pushed +=
        push_percept_tokens_commit_history(ctx.config, predictor, &step.observations, step.reward)?;
    Ok(pushed)
}

fn push_step_tokens_history(
    ctx: &WarmStartAugmentedTokenContext<'_>,
    predictor: &mut dyn Predictor,
    idx: usize,
) -> Result<usize, WarmStartExactJhError> {
    let step = &ctx.steps[idx - 1];
    let mut pushed = 0usize;
    pushed += push_encoded_bits_history(predictor, step.action, ctx.action_bits);
    if idx % ctx.config.label_phase_period == ctx.phase
        && let Some(label) = ctx.return_labels_by_step[idx - 1]
    {
        pushed += ctx.return_label_codec.push_label_history(predictor, label);
    }
    pushed += push_percept_tokens_history(ctx.config, predictor, &step.observations, step.reward)?;
    Ok(pushed)
}

fn push_percept_tokens_commit_history(
    config: &WarmStartExactJhRuntimeConfig,
    predictor: &mut dyn Predictor,
    observations: &[PerceptVal],
    reward: Reward,
) -> Result<usize, WarmStartExactJhError> {
    let mut pushed = 0usize;
    for &observation in observations {
        pushed += push_encoded_bits_commit_history(predictor, observation, config.observation_bits);
    }
    pushed += push_encoded_reward_commit_history(
        predictor,
        reward,
        config.reward_bits,
        config.reward_offset,
    )?;
    Ok(pushed)
}

fn push_percept_tokens_history(
    config: &WarmStartExactJhRuntimeConfig,
    predictor: &mut dyn Predictor,
    observations: &[PerceptVal],
    reward: Reward,
) -> Result<usize, WarmStartExactJhError> {
    let mut pushed = 0usize;
    for &observation in observations {
        pushed += push_encoded_bits_history(predictor, observation, config.observation_bits);
    }
    pushed +=
        push_encoded_reward_history(predictor, reward, config.reward_bits, config.reward_offset)?;
    Ok(pushed)
}

fn push_action_tokens_commit_history(
    predictor: &mut dyn Predictor,
    action: Action,
    action_bits: usize,
) -> usize {
    push_encoded_bits_commit_history(predictor, action, action_bits)
}

fn push_encoded_bits_history(predictor: &mut dyn Predictor, value: u64, bits: usize) -> usize {
    let mut v = value;
    for _ in 0..bits {
        predictor.update_history((v & 1) == 1);
        v >>= 1;
    }
    bits
}

fn push_encoded_bits_commit_history(
    predictor: &mut dyn Predictor,
    value: u64,
    bits: usize,
) -> usize {
    let mut v = value;
    for _ in 0..bits {
        predictor.commit_update_history((v & 1) == 1);
        v >>= 1;
    }
    bits
}

fn push_encoded_reward_history(
    predictor: &mut dyn Predictor,
    reward: Reward,
    bits: usize,
    offset: Reward,
) -> Result<usize, WarmStartExactJhError> {
    validate_reward_encoding_bounds(reward, reward, offset, bits)
        .map_err(WarmStartExactJhError::from)?;
    let shifted = (reward as i128) + (offset as i128);
    debug_assert!(
        shifted >= 0,
        "validate_reward_encoding_bounds implies shifted minimum >= 0"
    );
    let value = shifted as u64;
    Ok(push_encoded_bits_history(predictor, value, bits))
}

fn push_encoded_reward_commit_history(
    predictor: &mut dyn Predictor,
    reward: Reward,
    bits: usize,
    offset: Reward,
) -> Result<usize, WarmStartExactJhError> {
    validate_reward_encoding_bounds(reward, reward, offset, bits)
        .map_err(WarmStartExactJhError::from)?;
    let shifted = (reward as i128) + (offset as i128);
    debug_assert!(
        shifted >= 0,
        "validate_reward_encoding_bounds implies shifted minimum >= 0"
    );
    let value = shifted as u64;
    Ok(push_encoded_bits_commit_history(predictor, value, bits))
}

fn pop_history_bits(predictor: &mut dyn Predictor, bits: usize) {
    for _ in 0..bits {
        predictor.pop_history();
    }
}

fn max_value_for_bits(bits: usize) -> u64 {
    if bits >= 64 {
        u64::MAX
    } else if bits == 0 {
        0
    } else {
        (1u64 << bits) - 1
    }
}

fn argmax_with_fixed_tie_break(values: &[f64]) -> usize {
    let mut best_value = f64::NEG_INFINITY;
    let mut best_index = 0usize;
    for (index, &value) in values.iter().enumerate() {
        if value > best_value {
            best_value = value;
            best_index = index;
        }
    }
    best_index
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::aixi::warmstart_contract::standalone_teacher_provenance_crc32_pair;
    use std::sync::{Arc, Mutex};

    const TEST_TASK_FINGERPRINT_HEX: &str =
        "0102030401020304010203040102030401020304010203040102030401020304";
    const ZERO_TASK_FINGERPRINT_HEX: &str =
        "0000000000000000000000000000000000000000000000000000000000000000";
    const MISMATCH_TASK_FINGERPRINT_HEX: &str =
        "deadbeefdeadbeefdeadbeefdeadbeefdeadbeefdeadbeefdeadbeefdeadbeef";

    fn action_alphabet(n: usize) -> ActionAlphabet {
        ActionAlphabet::try_from_usize(n).expect("test action alphabet must be non-zero")
    }

    fn config() -> WarmStartExactJhConfig {
        WarmStartExactJhConfig {
            rate_backend: RateBackend::Ctw { depth: 4 },
            observation_bits: 2,
            observation_stream_len: 1,
            reward_bits: 2,
            agent_actions: action_alphabet(2),
            return_horizon: 1,
            return_bins: 4,
            label_phase_period: 1,
            planner_simulations_per_step: 1,
            bit_stream_semantics: BitStreamSemantics::BinaryTokens,
            random_seed: Some(9),
        }
    }

    #[test]
    fn reward_bounds_from_exact_return_bins_matches_default_test_config() {
        let cfg = config();
        assert_eq!(
            reward_bounds_from_exact_return_bins(
                cfg.return_horizon,
                cfg.return_bins,
                cfg.reward_bits
            )
            .expect("bounds"),
            (0, 3, 0)
        );
    }

    #[test]
    fn exact_return_labels_cover_each_dense_window() {
        let runtime = WarmStartExactJhRuntimeConfig {
            task_fingerprint: TaskFingerprint::parse_hex(TEST_TASK_FINGERPRINT_HEX)
                .expect("test fingerprint"),
            observation_bits: 2,
            observation_stream_len: 1,
            observation_key_mode: "full_stream",
            reward_bits: 2,
            agent_actions: action_alphabet(2),
            min_reward: 0,
            max_reward: 3,
            reward_offset: 0,
            return_horizon: 3,
            return_bins: 10,
            label_phase_period: 3,
            planner_simulations_per_step: 1,
            random_seed: 11,
            provenance_policy: TeacherProvenancePolicy::StandalonePlannerRun,
        };
        let steps = [1, 2, 0, 3, 1]
            .into_iter()
            .map(|reward| WarmStartExactJhTransition {
                action: 0,
                observations: vec![0],
                reward,
            })
            .collect::<Vec<_>>();

        let labels = exact_return_labels_for_trace(&runtime, &steps).expect("dense return labels");

        assert_eq!(labels, vec![Some(3), Some(5), Some(4), None, None]);
    }

    #[cfg(feature = "backend-ctw")]
    #[test]
    fn validate_warmstart_teacher_against_compiled_accepts_matching_contract() {
        let cfg = config();
        let compiled = cfg.compile_planner_run_spec().expect("compile planner run");
        let teacher = teacher_for_config(&cfg);
        validate_warmstart_teacher_against_compiled_planner_run(&compiled, &teacher.contract)
            .expect("matching teacher must validate");
        validate_warmstart_teacher_dataset_for_compiled_planner_run(&compiled, &teacher)
            .expect("matching teacher dataset must validate");
    }

    #[cfg(feature = "backend-ctw")]
    #[test]
    fn full_teacher_validator_rejects_short_trace_before_export() {
        let mut cfg = config();
        cfg.return_horizon = 2;
        cfg.return_bins = 5;
        cfg.label_phase_period = 2;
        let compiled = cfg.compile_planner_run_spec().expect("compile planner run");
        let mut teacher = teacher_for_config(&cfg);
        teacher.traces[0].transitions = vec![WarmStartExactJhTransition {
            action: 0,
            observations: vec![1],
            reward: 0,
        }];

        let err = validate_warmstart_teacher_dataset_for_compiled_planner_run(&compiled, &teacher)
            .expect_err("short trace must fail before write/load");
        assert!(err.to_string().contains("return_horizon is 2"), "{err}");
    }

    #[cfg(feature = "backend-ctw")]
    #[test]
    fn full_teacher_validator_rejects_bit_width_valid_but_runtime_invalid_reward() {
        let mut cfg = config();
        cfg.return_horizon = 2;
        cfg.return_bins = 5;
        cfg.label_phase_period = 2;
        let compiled = cfg.compile_planner_run_spec().expect("compile planner run");
        let mut teacher = teacher_for_config(&cfg);
        teacher.traces[0].transitions = vec![
            WarmStartExactJhTransition {
                action: 0,
                observations: vec![1],
                reward: 3,
            },
            WarmStartExactJhTransition {
                action: 1,
                observations: vec![2],
                reward: 0,
            },
        ];

        let err = validate_warmstart_teacher_dataset_for_compiled_planner_run(&compiled, &teacher)
            .expect_err("reward valid for reward_bits but outside exact runtime range must fail");
        assert!(err.to_string().contains("runtime contract"), "{err}");
        assert!(
            err.to_string().contains("outside configured range"),
            "{err}"
        );
    }

    #[cfg(feature = "backend-ctw")]
    #[test]
    fn validate_warmstart_teacher_against_compiled_rejects_reward_certificate_crc_mismatch() {
        let cfg = config();
        let compiled = cfg.compile_planner_run_spec().expect("compile planner run");
        let mut teacher = teacher_for_config(&cfg);
        teacher.contract.exact_reward_encoding_certificate = "00000000".to_string();
        let err =
            validate_warmstart_teacher_against_compiled_planner_run(&compiled, &teacher.contract)
                .expect_err("corrupted certificate hash must fail");
        assert!(matches!(
            err,
            WarmStartExactJhError::InvalidTeacherDataset { .. }
        ));
    }

    #[cfg(feature = "backend-ctw")]
    #[test]
    fn validate_warmstart_teacher_planner_task_fingerprint_rejects_mismatch_with_stable_markers() {
        let cfg = config();
        let compiled = cfg.compile_planner_run_spec().expect("compile planner run");
        let mut teacher = teacher_for_config(&cfg);
        teacher.contract.task_fingerprint = TaskFingerprint::parse_hex(ZERO_TASK_FINGERPRINT_HEX)
            .expect("valid mismatch fingerprint");
        let err = validate_warmstart_teacher_planner_task_fingerprint(&compiled, &teacher.contract)
            .expect_err("wrong fingerprint must fail");
        let msg = err.to_string();
        assert!(msg.contains("task_fingerprint"), "{msg}");
        assert!(msg.contains("current planner_run '"), "{msg}");
    }

    fn teacher_for_config(cfg: &WarmStartExactJhConfig) -> WarmStartExactJhTeacherDataset {
        let compiled = cfg
            .compile_planner_run_spec()
            .expect("test planner run must compile");
        let task_fingerprint = warmstart_exact_jh_planner_task_fingerprint(&compiled)
            .expect("test planner fingerprint");
        let observation_key_mode =
            observation_key_mode_name(compiled.interface().observation_key_mode);
        let observation_stream_len = cfg.observation_stream_len.max(1);
        let (adapter_crc, reward_cert) = standalone_teacher_provenance_crc32_pair(
            cfg.observation_bits,
            observation_stream_len,
            cfg.reward_bits,
        )
        .expect("standalone teacher provenance crc pair");
        WarmStartExactJhTeacherDataset {
            contract: WarmStartExactJhTeacherContract {
                schema_version: WARMSTART_TEACHER_CONTRACT_SCHEMA_VERSION,
                task_fingerprint,
                action_alphabet_size: cfg.agent_actions.get(),
                observation_bits: cfg.observation_bits,
                observation_stream_len,
                observation_key_mode: observation_key_mode.to_string(),
                observation_adapter_spec_ref: WARMSTART_STANDALONE_OBSERVATION_ADAPTER_SPEC_REF
                    .to_string(),
                observation_adapter_content_crc32: adapter_crc,
                reward_bits: cfg.reward_bits,
                return_horizon: cfg.return_horizon,
                label_phase_period: cfg.label_phase_period,
                scalar_representation: WARMSTART_STANDALONE_SCALAR_REPRESENTATION.to_string(),
                exact_reward_encoding_certificate: reward_cert,
            },
            traces: vec![WarmStartExactJhTeacherTrace {
                transitions: vec![
                    WarmStartExactJhTransition {
                        action: 0,
                        observations: vec![1],
                        reward: 0,
                    },
                    WarmStartExactJhTransition {
                        action: 1,
                        observations: vec![2],
                        reward: 3,
                    },
                    WarmStartExactJhTransition {
                        action: 1,
                        observations: vec![2],
                        reward: 3,
                    },
                ],
            }],
        }
    }

    fn teacher() -> WarmStartExactJhTeacherDataset {
        teacher_for_config(&config())
    }

    #[test]
    fn jsonl_trace_converter_round_trips_action_percept_pairs() {
        let teacher = teacher();
        let jsonl = [
            warmstart_jsonl_action_record(0, 0, PlannerActionProvenance::Greedy).to_string(),
            warmstart_jsonl_percept_record(0, &[1], 0).to_string(),
            warmstart_jsonl_action_record(1, 1, PlannerActionProvenance::Exploratory).to_string(),
            warmstart_jsonl_percept_record(1, &[2], 3).to_string(),
        ]
        .join("\n");
        let trace = warmstart_teacher_trace_from_jsonl_slice(
            jsonl.as_bytes(),
            &teacher.contract,
            teacher.contract.return_horizon,
        )
        .expect("jsonl trace should parse");
        assert_eq!(
            trace.transitions,
            vec![
                WarmStartExactJhTransition {
                    action: 0,
                    observations: vec![1],
                    reward: 0,
                },
                WarmStartExactJhTransition {
                    action: 1,
                    observations: vec![2],
                    reward: 3,
                },
            ]
        );
    }

    #[test]
    fn jsonl_trace_converter_converts_mcaixi_decision_percept_order() {
        let teacher = teacher();
        let jsonl = [
            warmstart_jsonl_percept_record(0, &[0], 0).to_string(),
            warmstart_jsonl_action_record(0, 0, PlannerActionProvenance::Greedy).to_string(),
            warmstart_jsonl_percept_record(1, &[1], 3).to_string(),
            warmstart_jsonl_action_record(1, 1, PlannerActionProvenance::Greedy).to_string(),
            warmstart_jsonl_percept_record(2, &[2], 0).to_string(),
        ]
        .join("\n");
        let trace = warmstart_teacher_trace_from_jsonl_slice(
            jsonl.as_bytes(),
            &teacher.contract,
            teacher.contract.return_horizon,
        )
        .expect("MC-AIXI-order jsonl trace should parse");
        assert_eq!(
            trace.transitions,
            vec![
                WarmStartExactJhTransition {
                    action: 0,
                    observations: vec![1],
                    reward: 3,
                },
                WarmStartExactJhTransition {
                    action: 1,
                    observations: vec![2],
                    reward: 0,
                },
            ]
        );
    }

    #[test]
    fn jsonl_trace_converter_rejects_sparse_action_then_percept_steps() {
        let teacher = teacher();
        let jsonl = [
            warmstart_jsonl_action_record(0, 0, PlannerActionProvenance::Greedy).to_string(),
            warmstart_jsonl_percept_record(0, &[1], 0).to_string(),
            warmstart_jsonl_action_record(2, 1, PlannerActionProvenance::Greedy).to_string(),
            warmstart_jsonl_percept_record(2, &[2], 3).to_string(),
        ]
        .join("\n");

        let err = warmstart_teacher_trace_from_jsonl_slice(
            jsonl.as_bytes(),
            &teacher.contract,
            teacher.contract.return_horizon,
        )
        .expect_err("sparse action/percept JSONL trace must fail");

        assert!(err.to_string().contains("not contiguous"), "{err}");
        assert!(err.to_string().contains("step 1"), "{err}");
    }

    #[test]
    fn jsonl_trace_converter_rejects_sparse_decision_percept_steps() {
        let teacher = teacher();
        let jsonl = [
            warmstart_jsonl_percept_record(0, &[0], 0).to_string(),
            warmstart_jsonl_action_record(0, 0, PlannerActionProvenance::Greedy).to_string(),
            warmstart_jsonl_percept_record(1, &[1], 3).to_string(),
            warmstart_jsonl_percept_record(2, &[2], 0).to_string(),
            warmstart_jsonl_action_record(2, 1, PlannerActionProvenance::Greedy).to_string(),
            warmstart_jsonl_percept_record(3, &[2], 0).to_string(),
        ]
        .join("\n");

        let err = warmstart_teacher_trace_from_jsonl_slice(
            jsonl.as_bytes(),
            &teacher.contract,
            teacher.contract.return_horizon,
        )
        .expect_err("sparse decision-percept JSONL trace must fail");

        assert!(err.to_string().contains("not contiguous"), "{err}");
        assert!(err.to_string().contains("step 1"), "{err}");
    }

    #[test]
    fn jsonl_trace_converter_rejects_malformed_and_inconsistent_records() {
        let teacher = teacher();
        let contract = &teacher.contract;
        let return_horizon = teacher.contract.return_horizon;
        for (jsonl, expected) in [
            ("{", "invalid JSON"),
            (
                r#"{"kind":"action","t":0,"action":0,"provenance":"unknown"}"#,
                "unknown action provenance",
            ),
            (
                r#"{"kind":"action","t":0,"action":0,"provenance":7}"#,
                "action provenance must be a string",
            ),
            (
                r#"{"kind":"action","t":0,"action":0}"#,
                "cannot infer JSONL action/percept convention",
            ),
            (
                r#"{"kind":"percept","t":0,"observations":[1],"reward":0}"#,
                "cannot infer JSONL action/percept convention",
            ),
            (
                concat!(
                    r#"{"kind":"action","t":0,"action":0}"#,
                    "\n",
                    r#"{"kind":"percept","t":0,"observations":[4],"reward":0}"#
                ),
                "observation value",
            ),
            (
                r#"{"kind":"action","t":0,"action":0,"extra":true}"#,
                "unknown field 'extra'",
            ),
            ("\n", "empty JSONL records are not allowed"),
            (
                concat!(
                    r#"{"kind":"action","t":0,"action":0}"#,
                    "\n",
                    r#"{"kind":"percept","t":0,"observations":[1],"reward":0}"#,
                    "\n",
                    r#"{"kind":"percept","t":1,"observations":[1],"reward":0}"#,
                    "\n",
                    r#"{"kind":"action","t":1,"action":0}"#
                ),
                "mixed JSONL action/percept conventions",
            ),
        ] {
            let err = warmstart_teacher_trace_from_jsonl_slice(
                jsonl.as_bytes(),
                contract,
                return_horizon,
            )
            .expect_err("invalid JSONL trace should fail");
            assert!(
                err.to_string().contains(expected),
                "expected '{expected}' in {err}"
            );
        }
    }

    #[test]
    fn jsonl_trace_converter_accepts_absent_provenance() {
        let contract = teacher().contract;
        let absent = concat!(
            r#"{"kind":"action","t":0,"action":1}"#,
            "\n",
            r#"{"kind":"percept","t":0,"observations":[2],"reward":1}"#,
            "\n",
            r#"{"kind":"action","t":1,"action":0}"#,
            "\n",
            r#"{"kind":"percept","t":1,"observations":[1],"reward":0}"#
        );
        let trace = warmstart_teacher_trace_from_jsonl_slice(absent.as_bytes(), &contract, 1)
            .expect("legacy trace without provenance should parse");
        assert_eq!(trace.transitions.len(), 2);
        assert_eq!(trace.transitions[0].action, 1);
        assert_eq!(trace.transitions[1].action, 0);
    }

    #[test]
    fn jsonl_trace_converter_rejects_duplicate_action_and_percept_records() {
        let contract = teacher().contract;
        let duplicate_action = concat!(
            r#"{"kind":"action","t":0,"action":0}"#,
            "\n",
            r#"{"kind":"action","t":0,"action":1}"#,
            "\n",
            r#"{"kind":"percept","t":0,"observations":[1],"reward":0}"#
        );
        let err =
            warmstart_teacher_trace_from_jsonl_slice(duplicate_action.as_bytes(), &contract, 1)
                .expect_err("duplicate action must fail");
        assert!(err.to_string().contains("duplicate action record"), "{err}");

        let duplicate_percept = concat!(
            r#"{"kind":"action","t":0,"action":0}"#,
            "\n",
            r#"{"kind":"percept","t":0,"observations":[1],"reward":0}"#,
            "\n",
            r#"{"kind":"percept","t":0,"observations":[2],"reward":1}"#
        );
        let err =
            warmstart_teacher_trace_from_jsonl_slice(duplicate_percept.as_bytes(), &contract, 1)
                .expect_err("duplicate percept must fail");
        assert!(
            err.to_string().contains("duplicate percept record"),
            "{err}"
        );
    }

    #[test]
    fn trace_recorder_requires_a_complete_return_horizon_window() {
        let contract = teacher().contract;
        let mut recorder = WarmStartExactJhTraceRecorder::new();
        recorder
            .record_action(0, 1)
            .expect("record action at step 0");
        recorder
            .record_percept(0, &[2], 3)
            .expect("record percept at step 0");
        let err = recorder
            .into_teacher_trace(&contract, 2)
            .expect_err("single transition must fail for return_horizon 2");
        assert!(err.to_string().contains("return_horizon is 2"), "{err}");

        let mut recorder = WarmStartExactJhTraceRecorder::new();
        recorder
            .record_action(0, 1)
            .expect("record action at step 0");
        recorder
            .record_percept(0, &[2], 3)
            .expect("record percept at step 0");
        let trace = recorder
            .into_teacher_trace(&contract, 1)
            .expect("one transition covers horizon one");
        assert_eq!(trace.transitions.len(), 1);
        assert_eq!(trace.transitions[0].action, 1);
        assert_eq!(trace.transitions[0].observations, vec![2]);
        assert_eq!(trace.transitions[0].reward, 3);
    }

    #[test]
    fn trace_recorder_rejects_sparse_step_sets() {
        let contract = teacher().contract;
        let mut recorder = WarmStartExactJhTraceRecorder::new();
        recorder
            .record_action(0, 0)
            .expect("record action at step 0");
        recorder
            .record_percept(0, &[1], 0)
            .expect("record percept at step 0");
        recorder
            .record_action(2, 1)
            .expect("record action at step 2");
        recorder
            .record_percept(2, &[2], 3)
            .expect("record percept at step 2");

        let err = recorder
            .into_teacher_trace(&contract, 2)
            .expect_err("sparse recorder trace must fail");

        assert!(err.to_string().contains("not contiguous"), "{err}");
        assert!(err.to_string().contains("step 1"), "{err}");
    }

    #[derive(Clone, Debug, Default, Eq, PartialEq)]
    struct PhaseUpdateCounts {
        commit_label_bits: usize,
        commit_history_bits: usize,
    }

    #[derive(Clone)]
    struct PhaseUpdateCountingPredictor {
        counts: Arc<Mutex<PhaseUpdateCounts>>,
    }

    impl Predictor for PhaseUpdateCountingPredictor {
        fn update(&mut self, _sym: bool) {}

        fn commit_update(&mut self, _sym: bool) {
            self.counts
                .lock()
                .expect("counts mutex poisoned")
                .commit_label_bits += 1;
        }

        fn update_history(&mut self, _sym: bool) {}

        fn commit_update_history(&mut self, _sym: bool) {
            self.counts
                .lock()
                .expect("counts mutex poisoned")
                .commit_history_bits += 1;
        }

        fn revert(&mut self) {}

        fn pop_history(&mut self) {}

        fn predict_prob(&mut self, sym: bool) -> f64 {
            if sym { 0.75 } else { 0.25 }
        }

        fn model_name(&self) -> String {
            "PhaseUpdateCountingPredictor".to_string()
        }

        fn boxed_clone(&self) -> Box<dyn Predictor> {
            Box::new(self.clone())
        }
    }

    #[test]
    fn warmstart_offline_phase_stream_updates_one_phase_per_closed_horizon_window() {
        let counts = (0..3)
            .map(|_| Arc::new(Mutex::new(PhaseUpdateCounts::default())))
            .collect::<Vec<_>>();
        let cfg = WarmStartExactJhRuntimeConfig {
            task_fingerprint: TaskFingerprint::parse_hex(TEST_TASK_FINGERPRINT_HEX)
                .expect("test fingerprint"),
            observation_bits: 2,
            observation_stream_len: 1,
            observation_key_mode: "full_stream",
            reward_bits: 2,
            agent_actions: action_alphabet(2),
            min_reward: 0,
            max_reward: 3,
            reward_offset: 0,
            return_horizon: 2,
            return_bins: 7,
            label_phase_period: 3,
            planner_simulations_per_step: 1,
            random_seed: 11,
            provenance_policy: TeacherProvenancePolicy::StandalonePlannerRun,
        };
        let teacher = WarmStartExactJhTeacherDataset {
            contract: WarmStartExactJhTeacherContract {
                schema_version: 1,
                task_fingerprint: TaskFingerprint::parse_hex(TEST_TASK_FINGERPRINT_HEX)
                    .expect("test fingerprint"),
                action_alphabet_size: 2,
                observation_bits: 2,
                observation_stream_len: 1,
                observation_key_mode: "full_stream".to_string(),
                observation_adapter_spec_ref: "test-observation-adapter".to_string(),
                observation_adapter_content_crc32: "test-observation-adapter-crc32".to_string(),
                reward_bits: 2,
                return_horizon: 2,
                label_phase_period: 3,
                scalar_representation: "test-scalar".to_string(),
                exact_reward_encoding_certificate: "test-cert".to_string(),
            },
            traces: vec![WarmStartExactJhTeacherTrace {
                transitions: vec![
                    WarmStartExactJhTransition {
                        action: 0,
                        observations: vec![1],
                        reward: 1,
                    },
                    WarmStartExactJhTransition {
                        action: 1,
                        observations: vec![2],
                        reward: 2,
                    },
                    WarmStartExactJhTransition {
                        action: 0,
                        observations: vec![3],
                        reward: 0,
                    },
                ],
            }],
        };
        let mut agent = WarmStartExactJhAgent {
            config: cfg,
            phases: counts
                .iter()
                .map(|counts| PhaseModel {
                    predictor: Box::new(PhaseUpdateCountingPredictor {
                        counts: counts.clone(),
                    }),
                    last_augmented_step: 0,
                })
                .collect(),
            steps: Vec::new(),
            return_labels_by_step: Vec::new(),
            total_steps_observed: 0,
            action_bits: 1,
            return_label_codec: ReturnLabelCodec::value_monotone(7),
            teacher_label_count: 0,
            rng: RandomGenerator::from_seed(11),
        };

        agent
            .warm_start_from_teacher(&teacher)
            .expect("offline teacher trace should warm-start");

        let snapshots = counts
            .iter()
            .map(|counts| counts.lock().expect("counts mutex poisoned").clone())
            .collect::<Vec<_>>();
        assert_eq!(agent.teacher_label_count(), 2);
        assert_eq!(agent.steps_observed(), 0);
        assert!(agent.same_task_live_trace().is_none());
        assert_eq!(
            snapshots,
            vec![
                PhaseUpdateCounts {
                    commit_label_bits: 0,
                    commit_history_bits: 15,
                },
                PhaseUpdateCounts {
                    commit_label_bits: 3,
                    commit_history_bits: 15,
                },
                PhaseUpdateCounts {
                    commit_label_bits: 3,
                    commit_history_bits: 15,
                },
            ]
        );
    }

    #[cfg(feature = "backend-ctw")]
    #[test]
    fn standalone_warmstart_teacher_contract_matches_compiled_validation() {
        let cfg = config();
        let compiled = cfg.compile_planner_run_spec().expect("compile planner run");
        let contract = standalone_warmstart_teacher_contract_for_compiled_planner_run(&compiled)
            .expect("standalone contract");
        validate_warmstart_teacher_against_compiled_planner_run(&compiled, &contract)
            .expect("standalone contract must validate against compiled planner run");
    }

    #[test]
    fn merge_warmstart_teacher_traces_preserves_deterministic_order_and_dedups() {
        let middle = WarmStartExactJhTeacherTrace {
            transitions: vec![WarmStartExactJhTransition {
                action: 0,
                observations: vec![2],
                reward: 0,
            }],
        };
        let mut traces = vec![middle.clone()];
        let high = WarmStartExactJhTeacherTrace {
            transitions: vec![WarmStartExactJhTransition {
                action: 1,
                observations: vec![2],
                reward: 3,
            }],
        };
        let low = WarmStartExactJhTeacherTrace {
            transitions: vec![WarmStartExactJhTransition {
                action: 0,
                observations: vec![1],
                reward: 0,
            }],
        };
        let (inserted, _) =
            merge_warmstart_teacher_traces_deterministic(&mut traces, vec![high.clone(), low]);
        assert_eq!(inserted, 2);
        assert!(!merge_warmstart_teacher_trace_deterministic(
            &mut traces,
            high
        ));
        assert!(!merge_warmstart_teacher_trace_deterministic(
            &mut traces,
            middle
        ));
        assert_eq!(traces.len(), 3);
        assert_eq!(traces[0].transitions[0].action, 0);
        assert_eq!(traces[0].transitions[0].observations, vec![1]);
        assert_eq!(traces[1].transitions[0].observations, vec![2]);
        assert_eq!(traces[2].transitions[0].action, 1);
    }

    #[test]
    fn teacher_dataset_new_canonicalizes_trace_order_and_dedups() {
        let high = WarmStartExactJhTeacherTrace {
            transitions: vec![WarmStartExactJhTransition {
                action: 1,
                observations: vec![2],
                reward: 3,
            }],
        };
        let low = WarmStartExactJhTeacherTrace {
            transitions: vec![WarmStartExactJhTransition {
                action: 0,
                observations: vec![1],
                reward: 0,
            }],
        };
        let dataset = WarmStartExactJhTeacherDataset::new(
            teacher().contract,
            vec![high.clone(), low.clone(), high.clone()],
        );
        assert_eq!(dataset.traces, vec![low, high]);
    }

    #[test]
    fn json_teacher_dataset_requires_same_task_traces() {
        let value = serde_json::json!({
            "schema_version": 1,
            "contract": {
                "task_fingerprint": TEST_TASK_FINGERPRINT_HEX,
                "action_alphabet_size": 2,
                "observation_bits": 2,
                "observation_stream_len": 1,
                "observation_key_mode": "full_stream",
                "observation_adapter_spec_ref": "test-observation-adapter",
                "observation_adapter_content_crc32": "test-observation-adapter-crc32",
                "reward_bits": 2,
                "return_horizon": 1,
                "label_phase_period": 1,
                "scalar_representation": "test-scalar",
                "exact_reward_encoding_certificate": "test-cert"
            },
            "traces": [{
                "transitions": [{"action": 1, "observations": [2], "reward": 3}]
            }]
        });
        let parsed = WarmStartExactJhTeacherDataset::from_json_value(&value)
            .expect("teacher trace should parse");
        assert_eq!(parsed.traces.len(), 1);
        assert_eq!(parsed.traces[0].transitions[0].action, 1);
    }

    #[test]
    fn json_teacher_dataset_rejects_malformed_task_fingerprint() {
        let value = serde_json::json!({
            "schema_version": 1,
            "contract": {
                "task_fingerprint": "not-a-fingerprint",
                "action_alphabet_size": 2,
                "observation_bits": 2,
                "observation_stream_len": 1,
                "observation_key_mode": "full_stream",
                "observation_adapter_spec_ref": "test-observation-adapter",
                "observation_adapter_content_crc32": "test-observation-adapter-crc32",
                "reward_bits": 2,
                "return_horizon": 1,
                "label_phase_period": 1,
                "scalar_representation": "test-scalar",
                "exact_reward_encoding_certificate": "test-cert"
            },
            "traces": [{
                "transitions": [{"action": 1, "observations": [2], "reward": 3}]
            }]
        });
        let err = WarmStartExactJhTeacherDataset::from_json_value(&value)
            .expect_err("malformed task fingerprint must fail at parse time");
        assert!(
            err.to_string()
                .contains("must be a 64-digit lowercase hexadecimal SHA-256 digest"),
            "{err}"
        );
    }

    #[test]
    fn json_teacher_dataset_rejects_legacy_trace_and_observation_aliases() {
        let mut value = serde_json::json!({
            "schema_version": 1,
            "contract": {
                "task_fingerprint": TEST_TASK_FINGERPRINT_HEX,
                "action_alphabet_size": 2,
                "observation_bits": 2,
                "observation_stream_len": 1,
                "observation_key_mode": "full_stream",
                "observation_adapter_spec_ref": "test-observation-adapter",
                "observation_adapter_content_crc32": "test-observation-adapter-crc32",
                "reward_bits": 2,
                "return_horizon": 1,
                "label_phase_period": 1,
                "scalar_representation": "test-scalar",
                "exact_reward_encoding_certificate": "test-cert"
            },
            "traces": [[{"action": 1, "observations": [2], "reward": 3}]]
        });
        let err = WarmStartExactJhTeacherDataset::from_json_value(&value)
            .expect_err("bare trace arrays must be rejected");
        assert!(
            err.to_string()
                .contains("must be an object with transitions")
        );

        value["traces"] = serde_json::json!([{
            "transitions": [{"action": 1, "obs": [2], "reward": 3}]
        }]);
        let err = WarmStartExactJhTeacherDataset::from_json_value(&value)
            .expect_err("obs alias must be rejected");
        assert!(err.to_string().contains("unknown teacher field 'obs'"));

        let mut top_extra = value.clone();
        top_extra["traces"] = serde_json::json!([{
            "transitions": [{"action": 1, "observations": [2], "reward": 3}]
        }]);
        top_extra["extra"] = serde_json::json!(true);
        let err = WarmStartExactJhTeacherDataset::from_json_value(&top_extra)
            .expect_err("top-level unknown fields must be rejected");
        assert!(err.to_string().contains("unknown teacher field 'extra'"));

        let mut contract_extra = top_extra;
        contract_extra
            .as_object_mut()
            .expect("object")
            .remove("extra");
        contract_extra["contract"]["extra"] = serde_json::json!(true);
        let err = WarmStartExactJhTeacherDataset::from_json_value(&contract_extra)
            .expect_err("contract unknown fields must be rejected");
        assert!(err.to_string().contains("unknown teacher field 'extra'"));
    }

    #[test]
    fn teacher_dataset_slice_parser_and_label_count_cover_horizon_windows() {
        let err = WarmStartExactJhTeacherDataset::from_json_slice(b"{")
            .expect_err("invalid json must be rejected");
        assert!(err.to_string().contains("invalid teacher JSON"), "{err}");

        let value = serde_json::json!({
            "schema_version": 1,
            "contract": {
                "task_fingerprint": TEST_TASK_FINGERPRINT_HEX,
                "action_alphabet_size": 2,
                "observation_bits": 2,
                "observation_stream_len": 1,
                "observation_key_mode": "full_stream",
                "observation_adapter_spec_ref": "test-observation-adapter",
                "observation_adapter_content_crc32": "test-observation-adapter-crc32",
                "reward_bits": 2,
                "return_horizon": 2,
                "label_phase_period": 2,
                "scalar_representation": "test-scalar",
                "exact_reward_encoding_certificate": "test-cert"
            },
            "traces": [
                {"transitions": [
                    {"action": 0, "observations": [1], "reward": 0},
                    {"action": 1, "observations": [2], "reward": 3},
                    {"action": 1, "observations": [2], "reward": 3}
                ]},
                {"transitions": [
                    {"action": 0, "observations": [0], "reward": 1}
                ]}
            ]
        });
        let bytes = serde_json::to_vec(&value).expect("teacher json");
        let parsed = WarmStartExactJhTeacherDataset::from_json_slice(&bytes)
            .expect("teacher dataset should parse from slice");

        assert_eq!(parsed.label_count_for_horizon(0), 0);
        assert_eq!(parsed.label_count_for_horizon(1), 4);
        assert_eq!(parsed.label_count_for_horizon(2), 2);
        assert_eq!(parsed.label_count_for_horizon(4), 0);
    }

    #[test]
    fn config_validation_reports_local_contract_errors_before_backend_use() {
        let mut cfg = config();
        cfg.return_horizon = 0;
        assert!(matches!(
            cfg.validate(),
            Err(WarmStartExactJhError::ReturnHorizonZero)
        ));

        let mut cfg = config();
        cfg.return_bins = 0;
        assert!(matches!(
            cfg.validate(),
            Err(WarmStartExactJhError::ReturnBinsZero)
        ));

        let mut cfg = config();
        cfg.return_horizon = 2;
        cfg.label_phase_period = 1;
        assert!(matches!(
            cfg.validate(),
            Err(WarmStartExactJhError::LabelPhasePeriodTooShort { .. })
        ));

        let mut cfg = config();
        cfg.planner_simulations_per_step = 0;
        assert!(matches!(
            cfg.validate(),
            Err(WarmStartExactJhError::PlannerSimulationsZero)
        ));

        let mut cfg = config();
        cfg.planner_simulations_per_step = 2;
        assert!(matches!(
            cfg.validate(),
            Err(WarmStartExactJhError::PlannerSimulationsUnsupported { configured: 2 })
        ));

        let mut cfg = config();
        cfg.return_horizon = 4;
        cfg.return_bins = 8;
        cfg.label_phase_period = 4;
        assert!(matches!(
            cfg.validate(),
            Err(WarmStartExactJhError::ReturnBinsNotExactHorizon {
                return_bins: 8,
                return_horizon: 4
            })
        ));

        let mut cfg = config();
        cfg.reward_bits = 1;
        assert!(matches!(
            cfg.validate(),
            Err(WarmStartExactJhError::RewardEncoding(_))
        ));
    }

    #[cfg(feature = "backend-ctw")]
    #[test]
    fn warmstart_agent_rejects_teacher_transitions_outside_interface_contract() {
        let mut invalid = teacher();
        invalid.traces[0].transitions[0].action = 2;
        let err = match WarmStartExactJhAgent::new(config(), invalid) {
            Ok(_) => panic!("teacher action outside alphabet must fail"),
            Err(err) => err,
        };
        assert!(matches!(
            err,
            WarmStartExactJhError::ActionOutOfRange { .. }
        ));

        let mut invalid = teacher();
        invalid.traces[0].transitions[0].observations = vec![1, 2];
        let err = match WarmStartExactJhAgent::new(config(), invalid) {
            Ok(_) => panic!("teacher observation stream length must fail"),
            Err(err) => err,
        };
        assert!(matches!(
            err,
            WarmStartExactJhError::ObservationStreamLengthMismatch { .. }
        ));

        let mut invalid = teacher();
        invalid.traces[0].transitions[0].observations = vec![4];
        let err = match WarmStartExactJhAgent::new(config(), invalid) {
            Ok(_) => panic!("teacher observation value must fail"),
            Err(err) => err,
        };
        assert!(matches!(
            err,
            WarmStartExactJhError::ObservationValueOutOfRange { .. }
        ));

        let mut invalid = teacher();
        invalid.traces[0].transitions[0].reward = 4;
        let err = match WarmStartExactJhAgent::new(config(), invalid) {
            Ok(_) => panic!("teacher reward outside range must fail"),
            Err(err) => err,
        };
        assert!(matches!(
            err,
            WarmStartExactJhError::RewardOutOfRange { .. }
        ));
    }

    #[cfg(feature = "backend-ctw")]
    #[test]
    fn warmstart_agent_rejects_teacher_contract_task_fingerprint_mismatch() {
        let mut invalid = teacher();
        invalid.contract.task_fingerprint =
            TaskFingerprint::parse_hex(MISMATCH_TASK_FINGERPRINT_HEX)
                .expect("valid mismatch fingerprint");
        let err = match WarmStartExactJhAgent::new(config(), invalid) {
            Ok(_) => panic!("teacher task fingerprint mismatch must fail"),
            Err(err) => err,
        };
        assert!(matches!(
            err,
            WarmStartExactJhError::InvalidTeacherDataset { .. }
        ));
        assert!(err.to_string().contains("task_fingerprint"), "{err}");
    }

    #[cfg(feature = "backend-ctw")]
    #[test]
    fn warmstart_agent_rejects_teacher_contract_interface_mismatch() {
        let mut invalid = teacher();
        invalid.contract.action_alphabet_size = 3;
        let err = match WarmStartExactJhAgent::new(config(), invalid) {
            Ok(_) => panic!("teacher contract interface mismatch must fail"),
            Err(err) => err,
        };
        assert!(matches!(
            err,
            WarmStartExactJhError::InvalidTeacherDataset { .. }
        ));
        assert!(err.to_string().contains("action_alphabet_size"), "{err}");
    }

    #[cfg(feature = "backend-ctw")]
    #[test]
    fn warmstart_agent_rejects_teacher_contract_return_horizon_mismatch() {
        let mut invalid = teacher();
        invalid.contract.return_horizon = 2;
        let err = match WarmStartExactJhAgent::new(config(), invalid) {
            Ok(_) => panic!("teacher contract return_horizon mismatch must fail"),
            Err(err) => err,
        };
        assert!(matches!(
            err,
            WarmStartExactJhError::InvalidTeacherDataset { .. }
        ));
        assert!(err.to_string().contains("return_horizon"), "{err}");
    }

    #[cfg(feature = "backend-ctw")]
    #[test]
    fn warmstart_agent_rejects_teacher_contract_label_phase_period_mismatch() {
        let mut invalid = teacher();
        invalid.contract.label_phase_period = 2;
        let err = match WarmStartExactJhAgent::new(config(), invalid) {
            Ok(_) => panic!("teacher contract label_phase_period mismatch must fail"),
            Err(err) => err,
        };
        assert!(matches!(
            err,
            WarmStartExactJhError::InvalidTeacherDataset { .. }
        ));
        assert!(err.to_string().contains("label_phase_period"), "{err}");
    }

    #[cfg(feature = "backend-ctw")]
    #[test]
    fn warmstart_agent_rejects_teacher_contract_observation_key_mode_mismatch() {
        let mut invalid = teacher();
        invalid.contract.observation_key_mode = "definitely-not-a-mode".to_string();
        let err = match WarmStartExactJhAgent::new(config(), invalid) {
            Ok(_) => panic!("teacher contract observation_key_mode mismatch must fail"),
            Err(err) => err,
        };
        assert!(matches!(
            err,
            WarmStartExactJhError::InvalidTeacherDataset { .. }
        ));
        assert!(err.to_string().contains("observation_key_mode"), "{err}");
    }

    #[cfg(feature = "backend-ctw")]
    #[test]
    fn warmstart_agent_rejects_teacher_contract_scalar_provenance_mismatch() {
        let mut invalid = teacher();
        invalid.contract.scalar_representation = "different-scalar".to_string();
        let err = match WarmStartExactJhAgent::new(config(), invalid) {
            Ok(_) => panic!("teacher contract scalar provenance mismatch must fail"),
            Err(err) => err,
        };
        assert!(matches!(
            err,
            WarmStartExactJhError::InvalidTeacherDataset { .. }
        ));
        assert!(err.to_string().contains("scalar_representation"), "{err}");
    }

    #[cfg(feature = "backend-ctw")]
    #[test]
    fn warmstart_agent_delays_live_trace_until_complete_return_horizon() {
        let mut cfg = config();
        cfg.return_horizon = 2;
        cfg.return_bins = 7;
        cfg.label_phase_period = 2;
        cfg.random_seed = Some(16);
        let mut agent = WarmStartExactJhAgent::new(cfg.clone(), teacher_for_config(&cfg))
            .expect("warmstart agent should initialize");

        assert_eq!(agent.teacher_label_count(), 2);
        assert_eq!(agent.num_actions(), action_alphabet(2));
        assert_eq!(agent.planner_simulations_per_step(), 1);
        assert_eq!(agent.resolved_random_seed(), 16);
        assert!(agent.same_task_live_trace().is_none());

        agent
            .observe_transition(0, &[1], 1)
            .expect("first live transition");
        assert!(agent.same_task_live_trace().is_none());

        agent
            .observe_transition(1, &[2], 2)
            .expect("second live transition");
        let live = agent
            .same_task_live_trace()
            .expect("complete live trace should be available");
        assert_eq!(live.transitions.len(), 2);
        assert_eq!(live.transitions[0].action, 0);
        assert_eq!(live.transitions[1].reward, 2);

        let greedy = agent.get_planned_action();
        let first_exploratory = agent.get_planned_action_with_extra_exploration(1.0);
        let second_exploratory = agent.get_planned_action_with_extra_exploration(1.0);
        assert_ne!(
            first_exploratory, second_exploratory,
            "test seed must make forced exploration distinguishable from a fixed action"
        );
        assert!(first_exploratory != greedy || second_exploratory != greedy);
    }

    #[cfg(feature = "backend-ctw")]
    #[test]
    fn warmstart_bytepacked_ctw_teacher_replay_and_live_step() {
        use crate::api::BitOrder;

        let cfg = WarmStartExactJhConfig {
            rate_backend: RateBackend::Ctw { depth: 8 },
            bit_stream_semantics: BitStreamSemantics::BytePacked {
                order: BitOrder::MsbFirst,
            },
            observation_bits: 8,
            observation_stream_len: 1,
            reward_bits: 8,
            agent_actions: action_alphabet(256),
            return_horizon: 1,
            return_bins: 256,
            label_phase_period: 1,
            planner_simulations_per_step: 1,
            random_seed: Some(42),
        };
        let teacher = WarmStartExactJhTeacherDataset {
            contract: teacher_for_config(&cfg).contract,
            traces: vec![WarmStartExactJhTeacherTrace {
                transitions: vec![WarmStartExactJhTransition {
                    action: 0,
                    observations: vec![5],
                    reward: 17,
                }],
            }],
        };
        let mut agent_a = WarmStartExactJhAgent::new(cfg.clone(), teacher.clone())
            .expect("byte-packed warmstart agent");
        let mut agent_b =
            WarmStartExactJhAgent::new(cfg, teacher).expect("byte-packed warmstart replay agent");
        assert_eq!(agent_a.teacher_label_count(), 1);
        let planned_a = agent_a.get_planned_action();
        let planned_b = agent_b.get_planned_action();
        assert_eq!(
            planned_a, planned_b,
            "byte-packed warmstart planning must be deterministic under identical seed"
        );
        assert!(planned_a < 256);
        agent_a
            .observe_transition(planned_a, &[3], 10)
            .expect("byte-packed live transition");
        assert_eq!(agent_a.steps_observed(), 1);
    }

    #[cfg(feature = "backend-ctw")]
    #[test]
    fn warmstart_binarytokens_fac_ctw_teacher_replay_and_live_step() {
        let cfg = WarmStartExactJhConfig {
            rate_backend: RateBackend::FacCtw {
                base_depth: 8,
                num_percept_bits: 8,
                encoding_bits: 8,
                msb_first: Some(true),
            },
            bit_stream_semantics: BitStreamSemantics::BinaryTokens,
            observation_bits: 2,
            observation_stream_len: 1,
            reward_bits: 2,
            agent_actions: action_alphabet(2),
            return_horizon: 1,
            return_bins: 4,
            label_phase_period: 1,
            planner_simulations_per_step: 1,
            random_seed: Some(7),
        };
        let teacher = teacher_for_config(&cfg);
        let mut agent_a = WarmStartExactJhAgent::new(cfg.clone(), teacher.clone())
            .expect("BinaryTokens FAC-CTW warmstart agent");
        let mut agent_b = WarmStartExactJhAgent::new(cfg, teacher)
            .expect("BinaryTokens FAC-CTW warmstart replay agent");
        assert_eq!(agent_a.teacher_label_count(), 3);
        let planned_a = agent_a.get_planned_action();
        let planned_b = agent_b.get_planned_action();
        assert_eq!(
            planned_a, planned_b,
            "BinaryTokens FAC-CTW warmstart planning must be deterministic under identical seed"
        );
        agent_a
            .observe_transition(planned_a, &[1], 1)
            .expect("BinaryTokens FAC-CTW live transition");
        assert_eq!(agent_a.steps_observed(), 1);
    }

    #[cfg(feature = "backend-ctw")]
    #[test]
    fn warmstart_agent_learns_teacher_labels_and_observes_live_steps() {
        let mut agent = WarmStartExactJhAgent::new(config(), teacher())
            .expect("warmstart agent should initialize");
        assert_eq!(agent.teacher_label_count(), 3);
        let action = agent.get_planned_action();
        assert!(action < 2);
        agent
            .observe_transition(action, &[1], 1)
            .expect("first transition");
        assert_eq!(agent.steps_observed(), 1);
    }

    #[test]
    fn validate_rejects_reward_bits_too_narrow_for_derived_instantaneous_bounds() {
        let mut cfg = config();
        cfg.return_horizon = 1;
        cfg.label_phase_period = 1;
        cfg.return_bins = 100;
        cfg.reward_bits = 1;
        let err = cfg
            .validate()
            .expect_err("derived max instantaneous reward must fit reward_bits");
        assert!(matches!(err, WarmStartExactJhError::RewardEncoding(_)));
    }

    #[derive(Clone, Default)]
    struct ResetSpyCounts {
        reset_calls: usize,
    }

    #[derive(Clone)]
    struct ResetSpyPredictor {
        counts: Arc<Mutex<ResetSpyCounts>>,
    }

    impl Predictor for ResetSpyPredictor {
        fn update(&mut self, _sym: bool) {}

        fn update_history(&mut self, _sym: bool) {}

        fn revert(&mut self) {}

        fn pop_history(&mut self) {}

        fn predict_prob(&mut self, sym: bool) -> f64 {
            if sym { 0.75 } else { 0.25 }
        }

        fn model_name(&self) -> String {
            "ResetSpyPredictor".to_string()
        }

        fn boxed_clone(&self) -> Box<dyn Predictor> {
            Box::new(self.clone())
        }

        fn reset_conditioning_history(&mut self) -> Result<(), String> {
            self.counts
                .lock()
                .expect("counts mutex poisoned")
                .reset_calls += 1;
            Ok(())
        }
    }

    #[test]
    fn warmstart_resets_predictor_conditioning_between_teacher_traces() {
        let counts = Arc::new(Mutex::new(ResetSpyCounts::default()));
        let spy = ResetSpyPredictor {
            counts: counts.clone(),
        };

        let cfg = WarmStartExactJhRuntimeConfig {
            task_fingerprint: TaskFingerprint::parse_hex(TEST_TASK_FINGERPRINT_HEX)
                .expect("test fingerprint"),
            observation_bits: 2,
            observation_stream_len: 1,
            observation_key_mode: "full_stream",
            reward_bits: 2,
            agent_actions: action_alphabet(2),
            min_reward: 0,
            max_reward: 3,
            reward_offset: 0,
            return_horizon: 1,
            return_bins: 4,
            label_phase_period: 1,
            planner_simulations_per_step: 1,
            random_seed: 7,
            provenance_policy: TeacherProvenancePolicy::StandalonePlannerRun,
        };

        let teacher = WarmStartExactJhTeacherDataset {
            contract: WarmStartExactJhTeacherContract {
                schema_version: 1,
                task_fingerprint: TaskFingerprint::parse_hex(TEST_TASK_FINGERPRINT_HEX)
                    .expect("test fingerprint"),
                action_alphabet_size: 2,
                observation_bits: 2,
                observation_stream_len: 1,
                observation_key_mode: "full_stream".to_string(),
                observation_adapter_spec_ref: "test-observation-adapter".to_string(),
                observation_adapter_content_crc32: "test-observation-adapter-crc32".to_string(),
                reward_bits: 2,
                return_horizon: 1,
                label_phase_period: 1,
                scalar_representation: "test-scalar".to_string(),
                exact_reward_encoding_certificate: "test-cert".to_string(),
            },
            traces: vec![
                WarmStartExactJhTeacherTrace {
                    transitions: vec![WarmStartExactJhTransition {
                        action: 0,
                        observations: vec![1],
                        reward: 1,
                    }],
                },
                WarmStartExactJhTeacherTrace {
                    transitions: vec![WarmStartExactJhTransition {
                        action: 1,
                        observations: vec![2],
                        reward: 2,
                    }],
                },
            ],
        };

        let mut agent = WarmStartExactJhAgent {
            config: cfg,
            phases: vec![PhaseModel {
                predictor: Box::new(spy),
                last_augmented_step: 0,
            }],
            steps: Vec::new(),
            return_labels_by_step: Vec::new(),
            total_steps_observed: 0,
            action_bits: 1,
            return_label_codec: ReturnLabelCodec::value_monotone(4),
            teacher_label_count: 0,
            rng: RandomGenerator::from_seed(7),
        };

        agent
            .warm_start_from_teacher(&teacher)
            .expect("warm-start should succeed");

        let snapshot = counts.lock().expect("counts mutex poisoned").clone();
        assert_eq!(snapshot.reset_calls, teacher.traces.len());
        assert_eq!(agent.teacher_label_count(), 2);
    }
}
