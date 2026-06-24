//! Tuner execution profile and CLI-facing tune runner.
//!
//! `SpecDocument::Tune` remains the canonical candidate/request surface.
//! Runtime controls in this module are executor-side and must not mutate
//! canonical candidate identity.

use crate::aixi::agent::Agent;
use crate::aixi::aiqi::AiqiAgent;
use crate::aixi::common::{
    Action, MctsStrategy, ObservationKeyMode, PerceptVal, RandomGenerator, Reward,
};
use crate::aixi::warmstart::{
    WarmStartExactJhAgent, WarmStartExactJhTeacherDataset, WarmStartExactJhTeacherTrace,
    merge_warmstart_teacher_trace_deterministic,
};
use crate::api::RateBackend;
use crate::runtime::CompressionRuntime;
use crate::spec::{
    AiqiDiscountedControllerSpec, AssetRef, BuiltinEnvironmentSpec, CanonicalJson,
    CompiledPlannerRunSpec, ControllerSpec, EnvironmentSpec, McAixiControllerSpec,
    PlannerInterfaceSpec, PlannerRunSpec, PlannerRuntimeSpec, SpecDocument, SpecEnvironment,
    TuneInvalidReason, WarmStartExactJhControllerSpec, load_spec_document,
};
use crc32fast::Hasher;
use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::fs;
use std::io::Write;
use std::path::Path;
use std::time::{Duration, Instant};

mod annealer;
mod causal_dataset;
mod certificates;
use certificates::validate_complete_finite_reward_interval;
#[cfg(test)]
use certificates::{parse_finite_reward_map, project_observation_output};
mod config;
mod eval;
mod planner_bridge;
mod report;
use crate::aixi::common::max_nonnegative_reward_for_bits;
use annealer::{
    annealer_acceptance_probability, annealer_active_radius, annealer_progress,
    annealer_runtime_path_name, annealer_temperature, collect_numeric_leaves,
    sample_annealed_proposal,
};
#[cfg(test)]
use annealer::{annealer_progress_from_elapsed, compile_canonical_proposal_kernel};
use causal_dataset::load_dataset;
pub use config::{
    AnnealerKernelProfile, PeakMemoryMode, TimingCertificationTier, TuneCommandRequest,
    TuneExecutionConfig, TuneTheoremConfig, parse_tune_command_args,
};
use config::{
    annealer_kernel_profile_name, compiled_feature_set, peak_memory_mode_name, timing_tier_name,
};
#[cfg(test)]
use eval::evaluate_candidate_causal_loss;
pub use eval::run_tuner_eval_worker_from_env;
use eval::{
    ResolvedEvaluatorRuntimeProfile, cache_key_for_candidate, evaluate_candidate,
    resolve_evaluator_runtime_profile, timeout_eval_result,
};
#[cfg(test)]
use planner_bridge::{
    TunerRawObservation, compile_tuner_planner_run_spec, encode_tuner_planner_percept,
    merge_warmstart_trace_deterministic, planner_controller_contract,
    validate_theorem_planner_mutation_domain,
};
use planner_bridge::{
    exact_nonnegative_i64_from_f64, key_less, normalized_clipped_improvement,
    run_planner_family_controller,
};
#[cfg(test)]
use report::exact_finite_mdp_missing_prereqs;
use report::{
    causal_profile_report, controller_kind_name, dataset_kind_name, diagnostic_chunking_report,
    evaluator_execution_model, executor_controls_report, objective_target_name,
    observation_key_mode_name, planner_completed_status, planner_deployability_report,
    planner_runtime_path_name, theorem_claims_report, theorem_timing_basis,
};

const PASSIVE_DATASET_LOWERING_VERSION: &str = "passive-bytes-v1";
const INTERACTIVE_TRACE_LOWERING_VERSION: &str = "interactive-trace-events-v1";
const CAUSAL_PREFIX_LOWERING_VERSION: &str = "causal-prefix-examples-v1";
const TUNER_EVALUATOR_INTERFACE_VERSION: &str = "typed-causal-evaluator-v1";
const OBSERVATION_ADAPTER_DECLARATION: &str = "single-channel-conditional-byte-adapter-v1";
const SCALAR_REPRESENTATION_DECLARATION: &str = "finite-ieee754-f64-nonfinite-forbidden-v1";
const TUNER_MCAIXI_HORIZON: usize = 5;
const TUNER_MCAIXI_FAC_CTW_BASE_DEPTH: usize = 8;
const ANNEALER_T0_BITS: f64 = 1.0;
const ANNEALER_T_MIN_BITS: f64 = 1.0e-3;

enum ExactObjectiveDifferenceController<'a> {
    McAixiFacCtw(&'a crate::spec::McAixiFacCtwTuneControllerSpec),
    AiqiWarmstartExactJh(&'a crate::spec::WarmStartExactJhTuneControllerSpec),
}

impl ExactObjectiveDifferenceController<'_> {
    fn reward_bits(&self) -> usize {
        match self {
            Self::McAixiFacCtw(controller) => controller.interface.reward_bits,
            Self::AiqiWarmstartExactJh(controller) => controller.interface.reward_bits,
        }
    }
}

impl crate::spec::CompiledTuneController {
    fn exact_objective_difference_controller(
        &self,
    ) -> Option<ExactObjectiveDifferenceController<'_>> {
        match self {
            Self::McAixiFacCtw(controller) => {
                Some(ExactObjectiveDifferenceController::McAixiFacCtw(controller))
            }
            Self::AiqiWarmstartExactJh(controller) => Some(
                ExactObjectiveDifferenceController::AiqiWarmstartExactJh(controller),
            ),
            Self::AnnealedHillClimbing(_) | Self::AiqiDiscounted(_) => None,
        }
    }
}

fn observation_adapter_spec_value() -> Value {
    serde_json::json!({
        "kind": OBSERVATION_ADAPTER_DECLARATION,
        "schema_version": 1,
        "stream": "fixed_len_packed_u64_little_endian",
        "fields": [
            "fail_flag",
            "normalized_physical_size",
            "normalized_target_loss",
            "normalized_eval_time",
            "physical_size_delta",
            "eval_time_delta",
            "candidate_signature_crc32",
            "terminal"
        ],
        "missing_sentinel": 0xff_u8,
        "nonfinite_float_encoding": "forbidden_before_encoding",
        "delta_time_epsilon": 1.0e-9_f64,
    })
}

fn observation_adapter_content_hash() -> Result<String, String> {
    serde_json::to_vec(&observation_adapter_spec_value())
        .map(|bytes| crc32_hex(&bytes))
        .map_err(|err| format!("failed to encode observation adapter spec: {err}"))
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum DatasetKind {
    PassiveBytes,
    InteractiveTrace,
    CausalPrefixDataset,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ObjectiveTarget {
    PassiveAc,
    InteractiveCausalAc,
    PlannerDeployableModel,
}

#[derive(Clone, Debug)]
struct LoadedDataset {
    kind: DatasetKind,
    objective_target: ObjectiveTarget,
    lowering_version: &'static str,
    codec_hash: String,
    event_grammar_hash: String,
    target_domain_support_hash: String,
    causal_header_profile_hash: String,
    target_size_function: &'static str,
    canonical_content_hash: String,
    lowered_skeleton_hash: String,
    resolved_path: String,
    source_size_bytes: usize,
    raw_bytes: Vec<u8>,
    events: Vec<LoweredCausalEvent>,
    causal_profile: Option<CausalEvaluationProfile>,
    dataset_units: f64,
    target_events: usize,
}

#[derive(Clone, Debug)]
struct CausalEvaluationProfile {
    domains: BTreeMap<String, CausalTargetDomain>,
    channel_set: BTreeSet<String>,
    domain_support_hash: String,
    byte_alphabet_symbol_width: usize,
    header_profile_hash: String,
    event_grammar: CausalEventGrammar,
    action_alphabet_size: usize,
    collection_policy: String,
    percept_channels: BTreeSet<CausalChannelDomain>,
    reward_channel: CausalChannelDomain,
    terminal_channel: CausalChannelDomain,
}

#[derive(Clone, Debug)]
enum CausalTargetDomain {
    ByteAlphabet,
    EnumeratedPayloads { payloads: Vec<Vec<u8>> },
}

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd)]
struct CausalChannelDomain {
    channel: String,
    domain: String,
}

#[derive(Clone, Debug)]
struct CausalEventGrammar {
    context_channels: BTreeSet<String>,
    observe_target_no_score: BTreeSet<CausalChannelDomain>,
    target: BTreeSet<CausalChannelDomain>,
}

struct PreparedTuneContext {
    compiled: crate::spec::CompiledTuneSpec,
    dataset: LoadedDataset,
    runtime_profile: ResolvedEvaluatorRuntimeProfile,
    evaluator_profile: EvaluatorProfile,
    initial_eval_limit: f64,
    tune_started: Instant,
}

fn prepare_tune_context(request: &TuneCommandRequest) -> Result<PreparedTuneContext, String> {
    request.execution.validate()?;
    let config_path = Path::new(&request.spec_path);
    let config_dir = config_path.parent().unwrap_or(Path::new("."));
    let document = load_spec_document(&request.spec_path).map_err(|err| err.to_string())?;
    let SpecDocument::Tune(spec) = document else {
        return Err(format!(
            "tune expects a tune document, found kind '{}'",
            document.kind_str()
        ));
    };
    let tune_env = SpecEnvironment::new(config_dir);
    let compiled = spec.compile_in(&tune_env).map_err(|err| err.to_string())?;

    validate_candidate_against_tune_bounds(
        compiled.baseline_candidate().canonical_spec(),
        &compiled.canonical_spec().bounds,
    )?;
    reject_candidate_local_external_artifacts(compiled.baseline_candidate().canonical_spec())
        .map_err(|err| err.diagnostic)?;

    let dataset = load_dataset(resolve_input_asset_path(
        &compiled,
        &compiled.canonical_spec().input_asset,
    )?)?;
    let tune_started = Instant::now();
    let initial_eval_limit = effective_eval_limit_seconds(
        &compiled,
        tune_started,
        Some(compiled.canonical_spec().eval_time_limit_seconds),
        None,
    );
    let runtime_profile = resolve_evaluator_runtime_profile(
        &request.execution,
        request
            .execution
            .theorem
            .deterministic_evaluator_table
            .is_some(),
    )?;
    let evaluator_profile = EvaluatorProfile {
        dataset_kind: dataset.kind,
        objective_target: if request.execution.planner_deployable_model {
            ObjectiveTarget::PlannerDeployableModel
        } else {
            dataset.objective_target
        },
        dataset_lowering_version: dataset.lowering_version,
        dataset_codec_hash: dataset.codec_hash.clone(),
        event_grammar_hash: dataset.event_grammar_hash.clone(),
        target_domain_support_hash: dataset.target_domain_support_hash.clone(),
        causal_header_profile_hash: dataset.causal_header_profile_hash.clone(),
        target_size_function: dataset.target_size_function,
        evaluator_interface_version: TUNER_EVALUATOR_INTERFACE_VERSION,
        candidate_canonicalization_version: compiled
            .candidate_canonicalization_version()
            .to_string(),
        warmup_baseline_runs: request.execution.warmup_baseline_runs,
        diagnostic_chunk_bytes: request.execution.diagnostic_chunk_bytes,
        eval_time_limit_seconds: initial_eval_limit,
        evaluator_threads: request.execution.evaluator_threads(),
        worker_isolation_mode: "spawn_exec_worker",
        worker_executable_identity: runtime_profile.worker_executable_identity.clone(),
        resolved_memory_accounting_kind: runtime_profile.memory_accounting_kind.name(),
        resolved_memory_accounting_strict_theorem_facing: runtime_profile
            .strict_theorem_memory_certified(),
        resolved_evaluator_cgroup_parent: runtime_profile.resolved_cgroup_parent_string(),
        backend_report_component_policy: runtime_profile
            .memory_accounting_kind
            .backend_report_component_policy(),
        evaluator_determinism: request.execution.evaluator_determinism(),
        rss_mode: request.execution.rss_mode,
        timing_certification_tier: request.execution.theorem.timing_certification_tier,
        build_profile: option_env!("PROFILE").unwrap_or("unknown"),
        feature_set: compiled_feature_set(),
    };

    Ok(PreparedTuneContext {
        compiled,
        dataset,
        runtime_profile,
        evaluator_profile,
        initial_eval_limit,
        tune_started,
    })
}

fn emit_exact_reward_encoding_certificate(
    request: &TuneCommandRequest,
    path: &str,
    prepared: &PreparedTuneContext,
) -> Result<(), String> {
    let controller_kind = controller_kind_name(prepared.compiled.controller());
    let scalar_representation = request
        .execution
        .theorem
        .scalar_representation_ref
        .as_deref()
        .unwrap_or(SCALAR_REPRESENTATION_DECLARATION);
    let Some(exact_controller) = prepared
        .compiled
        .controller()
        .exact_objective_difference_controller()
    else {
        return Err(format!(
            "exact reward-encoding certificate emission is only supported for exact-objective controller families (mc_aixi_fac_ctw, aiqi_warmstart_exact_jh); found '{controller_kind}'"
        ));
    };
    let reward_bits = exact_controller.reward_bits();
    let action_alphabet_size = planner_action_count(&prepared.compiled)?;
    let cert = serde_json::json!({
        "schema_version": 1,
        "kind": "exact_reward_encoding",
        "dataset_crc32": prepared.dataset.canonical_content_hash,
        "bounds_crc32": bounds_hash(&prepared.compiled.canonical_spec().bounds)?,
        "evaluator_profile_crc32": prepared.evaluator_profile.hash()?,
        "controller_kind": controller_kind,
        "action_alphabet_size": action_alphabet_size,
        "encoding": "integer_objective_difference",
        "scalar_representation": scalar_representation,
        "reward_bits": reward_bits,
        "max_reward": max_nonnegative_reward_for_bits(reward_bits)?,
    });
    let bytes = serde_json::to_vec_pretty(&cert)
        .map_err(|err| format!("failed to serialize exact reward certificate JSON: {err}"))?;
    fs::write(path, bytes)
        .map_err(|err| format!("failed to write exact reward certificate '{}': {err}", path))?;
    Ok(())
}

#[derive(Clone, Debug)]
struct CausalHeaderProfile {
    action_alphabet_size: usize,
    collection_policy: String,
    percept_channels: BTreeSet<CausalChannelDomain>,
    reward_channel: CausalChannelDomain,
    terminal_channel: CausalChannelDomain,
    event_grammar: CausalEventGrammar,
    profile_hash: String,
}

#[derive(Clone, Debug, PartialEq)]
struct EvaluatorProfile {
    dataset_kind: DatasetKind,
    objective_target: ObjectiveTarget,
    dataset_lowering_version: &'static str,
    dataset_codec_hash: String,
    event_grammar_hash: String,
    target_domain_support_hash: String,
    causal_header_profile_hash: String,
    target_size_function: &'static str,
    evaluator_interface_version: &'static str,
    candidate_canonicalization_version: String,
    warmup_baseline_runs: usize,
    diagnostic_chunk_bytes: Option<usize>,
    eval_time_limit_seconds: f64,
    evaluator_threads: usize,
    worker_isolation_mode: &'static str,
    worker_executable_identity: Option<String>,
    resolved_memory_accounting_kind: &'static str,
    resolved_memory_accounting_strict_theorem_facing: bool,
    resolved_evaluator_cgroup_parent: Option<String>,
    backend_report_component_policy: &'static str,
    evaluator_determinism: &'static str,
    rss_mode: PeakMemoryMode,
    timing_certification_tier: TimingCertificationTier,
    build_profile: &'static str,
    feature_set: Vec<&'static str>,
}

impl EvaluatorProfile {
    fn with_eval_time_limit(&self, eval_time_limit_seconds: f64) -> Self {
        let mut profile = self.clone();
        profile.eval_time_limit_seconds = eval_time_limit_seconds;
        profile
    }

    fn to_json_value(&self) -> Value {
        serde_json::json!({
            "dataset_kind": dataset_kind_name(self.dataset_kind),
            "objective_target": objective_target_name(self.objective_target),
            "dataset_lowering_version": self.dataset_lowering_version,
            "dataset_codec_hash": self.dataset_codec_hash,
            "event_grammar_hash": self.event_grammar_hash,
            "target_domain_support_hash": self.target_domain_support_hash,
            "causal_header_profile_hash": self.causal_header_profile_hash,
            "target_size_function": self.target_size_function,
            "evaluator_interface_version": self.evaluator_interface_version,
            "candidate_canonicalization_version": self.candidate_canonicalization_version,
            "warmup_baseline_runs": self.warmup_baseline_runs,
            "diagnostic_chunk_bytes": self.diagnostic_chunk_bytes,
            "effective_eval_time_limit_seconds": self.eval_time_limit_seconds,
            "evaluator_threads": self.evaluator_threads,
            "worker_isolation_mode": self.worker_isolation_mode,
            "worker_executable_identity": self.worker_executable_identity.as_deref(),
            "resolved_memory_accounting_kind": self.resolved_memory_accounting_kind,
            "resolved_memory_accounting_strict_theorem_facing": self.resolved_memory_accounting_strict_theorem_facing,
            "resolved_evaluator_cgroup_parent": self.resolved_evaluator_cgroup_parent.as_deref(),
            "backend_report_component_policy": self.backend_report_component_policy,
            "evaluator_determinism": self.evaluator_determinism,
            "rss_mode": peak_memory_mode_name(self.rss_mode),
            "timing_certification_tier": timing_tier_name(self.timing_certification_tier),
            "build_profile": self.build_profile,
            "feature_set": self.feature_set,
        })
    }

    fn hash(&self) -> Result<String, String> {
        Ok(crc32_hex(&self.cache_identity_bytes()?))
    }

    fn cache_identity_bytes(&self) -> Result<Vec<u8>, String> {
        serde_json::to_vec(&serde_json::json!({
            "dataset_kind": dataset_kind_name(self.dataset_kind),
            "objective_target": objective_target_name(self.objective_target),
            "dataset_lowering_version": self.dataset_lowering_version,
            "dataset_codec_hash": self.dataset_codec_hash,
            "event_grammar_hash": self.event_grammar_hash,
            "target_domain_support_hash": self.target_domain_support_hash,
            "causal_header_profile_hash": self.causal_header_profile_hash,
            "target_size_function": self.target_size_function,
            "evaluator_interface_version": self.evaluator_interface_version,
            "candidate_canonicalization_version": self.candidate_canonicalization_version,
            "warmup_baseline_runs": self.warmup_baseline_runs,
            "diagnostic_chunk_bytes": self.diagnostic_chunk_bytes,
            "effective_eval_time_limit_seconds_bits": self.eval_time_limit_seconds.to_bits(),
            "evaluator_threads": self.evaluator_threads,
            "worker_isolation_mode": self.worker_isolation_mode,
            "worker_executable_identity": self.worker_executable_identity.as_deref(),
            "resolved_memory_accounting_kind": self.resolved_memory_accounting_kind,
            "resolved_memory_accounting_strict_theorem_facing": self.resolved_memory_accounting_strict_theorem_facing,
            "resolved_evaluator_cgroup_parent": self.resolved_evaluator_cgroup_parent.as_deref(),
            "backend_report_component_policy": self.backend_report_component_policy,
            "evaluator_determinism": self.evaluator_determinism,
            "rss_mode": peak_memory_mode_name(self.rss_mode),
            "timing_certification_tier": timing_tier_name(self.timing_certification_tier),
            "build_profile": self.build_profile,
            "feature_set": self.feature_set,
        }))
        .map_err(|err| format!("failed to encode evaluator profile JSON: {err}"))
    }
}

#[derive(Clone, Debug)]
struct CandidateEvalResult {
    status: CandidateEvalStatus,
    compressed_bytes: usize,
    elapsed_seconds: f64,
    effective_eval_time_limit_seconds: f64,
    throughput_bytes_per_second: f64,
    peak_memory_bytes: u64,
    target_loss_bits: f64,
    objective_bits: f64,
    deployable: bool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum CandidateEvalStatus {
    Success,
    Timeout,
    Invalid,
    Error,
}

impl CandidateEvalStatus {
    fn name(self) -> &'static str {
        match self {
            CandidateEvalStatus::Success => "success",
            CandidateEvalStatus::Timeout => "timeout",
            CandidateEvalStatus::Invalid => "invalid",
            CandidateEvalStatus::Error => "error",
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
enum CandidateEvalFailure {
    FatalEvaluatorFailure { diagnostic: String },
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct CandidateInvalidDiagnostic {
    reason: TuneInvalidReason,
    diagnostic: String,
}

#[derive(Clone, Debug, Default)]
struct CandidateResultCounts {
    success_deployable: usize,
    success_non_deployable: usize,
    timeout: usize,
    invalid: usize,
    error_recoverable: usize,
}

impl CandidateResultCounts {
    fn record_admitted_result(&mut self, value: &CandidateEvalResult) {
        match value.status {
            CandidateEvalStatus::Success if value.deployable => {
                self.success_deployable = self.success_deployable.saturating_add(1);
            }
            CandidateEvalStatus::Success => {
                self.success_non_deployable = self.success_non_deployable.saturating_add(1);
            }
            CandidateEvalStatus::Timeout => {
                self.timeout = self.timeout.saturating_add(1);
            }
            CandidateEvalStatus::Invalid => {
                self.invalid = self.invalid.saturating_add(1);
            }
            CandidateEvalStatus::Error => {
                self.error_recoverable = self.error_recoverable.saturating_add(1);
            }
        }
    }
}

#[derive(Clone, Debug, Default)]
struct InvalidReasonCounts {
    candidate_external_asset_forbidden: usize,
    candidate_out_of_bounds: usize,
    candidate_compile_error: usize,
    invalid_action_index: usize,
    inapplicable_action: usize,
}

impl InvalidReasonCounts {
    fn record(&mut self, reason: TuneInvalidReason) {
        match reason {
            TuneInvalidReason::CandidateExternalAssetForbidden => {
                self.candidate_external_asset_forbidden =
                    self.candidate_external_asset_forbidden.saturating_add(1);
            }
            TuneInvalidReason::CandidateOutOfBounds => {
                self.candidate_out_of_bounds = self.candidate_out_of_bounds.saturating_add(1);
            }
            TuneInvalidReason::CandidateCompileError => {
                self.candidate_compile_error = self.candidate_compile_error.saturating_add(1);
            }
            TuneInvalidReason::InvalidActionIndex => {
                self.invalid_action_index = self.invalid_action_index.saturating_add(1);
            }
            TuneInvalidReason::InapplicableAction => {
                self.inapplicable_action = self.inapplicable_action.saturating_add(1);
            }
        }
    }
}

#[derive(Clone, Debug)]
struct CandidateTopologyStats {
    backend_families: BTreeSet<String>,
    max_mixture_nesting_depth: usize,
    mixture_node_expert_counts: Vec<usize>,
}

#[derive(Clone)]
struct SearchSummary {
    status: &'static str,
    warning: Option<String>,
    fatal_evaluator_failure: Option<String>,
    fatal_evaluator_failures: usize,
    best_candidate: crate::api::CompressionBackend,
    best_candidate_crc32: String,
    best_eval: CandidateEvalResult,
    cache_key_digest: String,
    cache_hits: usize,
    cache_misses: usize,
    candidate_evaluations_executed: usize,
    non_warmup_candidate_results_seen: usize,
    post_baseline_candidate_results_seen: usize,
    proposals_attempted: usize,
    proposals_invalid: usize,
    self_loop_proposals: usize,
    invalid_reason_counts: InvalidReasonCounts,
    successful_non_deployable: usize,
    candidate_result_counts: CandidateResultCounts,
    final_best_move_reward: f64,
    realized_trace_counts_by_round: Option<Vec<usize>>,
    trace_refresh_merges_by_round: Option<Vec<usize>>,
    controller_report: Value,
}

#[derive(Clone, Debug)]
struct NumericLeaf {
    path: String,
    pointer: String,
    kind: NumericKind,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum NumericKind {
    Unsigned,
    Signed,
    Float,
}

#[derive(Clone)]
struct CanonicalProposalKernel {
    transitions: Vec<CanonicalProposal>,
    total_raw_actions: u64,
}

#[derive(Clone)]
struct CanonicalProposal {
    candidate: crate::api::CompressionBackend,
    candidate_canonical_bytes: Vec<u8>,
    raw_action_count: u64,
}

#[derive(Clone)]
struct AnnealedProposal {
    candidate: crate::api::CompressionBackend,
    forward_raw_action_count: u64,
    forward_total_raw_actions: u64,
    reverse_raw_action_count: u64,
    reverse_total_raw_actions: u64,
}

#[derive(Clone)]
enum AnnealedProposalDraw {
    Proposal(AnnealedProposal),
    SelfLoop,
    Exhausted,
}

impl CanonicalProposalKernel {
    fn proposal_mass_to_canonical_bytes(&self, candidate_canonical_bytes: &[u8]) -> u64 {
        self.transitions
            .iter()
            .find(|proposal| proposal.candidate_canonical_bytes == candidate_canonical_bytes)
            .map(|proposal| proposal.raw_action_count)
            .unwrap_or(0)
    }

    fn sample<'a>(&'a self, rng: &mut RandomGenerator) -> Option<&'a CanonicalProposal> {
        if self.total_raw_actions == 0 {
            return None;
        }
        let total_raw_actions = usize::try_from(self.total_raw_actions).ok()?;
        let mut draw = rng.gen_range(total_raw_actions) as u64;
        for proposal in &self.transitions {
            if draw < proposal.raw_action_count {
                return Some(proposal);
            }
            draw -= proposal.raw_action_count;
        }
        None
    }
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
struct CandidateCacheKey {
    candidate_canonical_bytes: Vec<u8>,
    evaluator_profile_bytes: Vec<u8>,
    dataset_identity: String,
}

impl CandidateCacheKey {
    fn digest_crc32(&self) -> String {
        let mut payload = Vec::with_capacity(
            self.candidate_canonical_bytes.len()
                + self.evaluator_profile_bytes.len()
                + self.dataset_identity.len(),
        );
        payload.extend_from_slice(&self.candidate_canonical_bytes);
        payload.extend_from_slice(&self.evaluator_profile_bytes);
        payload.extend_from_slice(self.dataset_identity.as_bytes());
        crc32_hex(&payload)
    }
}

#[derive(Clone, Debug, Default)]
struct VerifiedTheoremInputs {
    finite_planner_state: Option<VerifiedCertificate>,
    no_hidden_state: Option<VerifiedCertificate>,
    exact_reward_encoding: Option<VerifiedExactRewardEncodingCertificate>,
    exact_state_observation: Option<VerifiedExactStateObservationCertificate>,
    determinism_deadline: Option<VerifiedCertificate>,
    deterministic_table: Option<VerifiedDeterministicEvaluatorTable>,
}

#[derive(Clone, Debug)]
struct VerifiedCertificate {
    ref_value: String,
    content_hash: String,
}

#[derive(Clone, Debug)]
struct VerifiedExactRewardEncodingCertificate {
    base: VerifiedCertificate,
    max_reward: Reward,
    reward_bits: usize,
    scalar_representation: String,
    mode: VerifiedRewardEncodingMode,
}

#[derive(Clone, Debug)]
enum VerifiedRewardEncodingMode {
    IntegerObjectiveDifferenceInterval,
    FiniteRewardMap { map: VerifiedFiniteRewardMap },
}

#[derive(Clone, Debug)]
struct VerifiedFiniteRewardMap {
    objective_difference_to_symbol: BTreeMap<Reward, Reward>,
    complete_nonnegative_interval_max: Option<Reward>,
}

#[derive(Clone, Debug)]
struct VerifiedExactStateObservationCertificate {
    base: VerifiedCertificate,
    observation_key_mode: String,
    exact_state_encoder_spec_ref: String,
    observation_adapter_spec_ref: String,
    observation_adapter_content_hash: String,
    finite_planner_state_certificate_hash: String,
    finite_state_count: usize,
}

#[derive(Clone, Debug)]
struct VerifiedDeterministicEvaluatorTable {
    base: VerifiedCertificate,
    rows: HashMap<String, DeterministicEvaluatorRow>,
}

#[derive(Clone, Debug)]
struct DeterministicEvaluatorRow {
    status: CandidateEvalStatus,
    compressed_bytes: usize,
    target_loss_bits: f64,
    elapsed_seconds: f64,
    peak_memory_bytes: u64,
}

#[derive(Clone, Debug)]
enum LoweredCausalEvent {
    Reset,
    Context {
        channel: String,
        bytes: Vec<u8>,
    },
    ObserveTargetNoScore {
        channel: String,
        domain: String,
        bytes: Vec<u8>,
    },
    Target {
        channel: String,
        domain: String,
        bytes: Vec<u8>,
        weight: f64,
    },
}

#[derive(Clone, Debug)]
enum PlannerMutationAction {
    NumericStep {
        path: String,
        pointer: String,
        kind: NumericKind,
        delta: f64,
        range: Option<(f64, f64)>,
    },
    Noop,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum PlannerRewardSemantics {
    ExactObjectiveDifference,
    NormalizedClippedImprovement,
}

impl PlannerRewardSemantics {
    fn name(self) -> &'static str {
        match self {
            PlannerRewardSemantics::ExactObjectiveDifference => "exact_objective_difference",
            PlannerRewardSemantics::NormalizedClippedImprovement => {
                "normalized_clipped_improvement"
            }
        }
    }
}

#[derive(Clone, Debug)]
struct WarmstartTeacherDataset {
    asset_id: String,
    resolved_path: String,
    content_hash: String,
    records: usize,
    traces: WarmStartExactJhTeacherDataset,
}

#[derive(Clone, Debug)]
struct PlannerControllerContract {
    interface: crate::spec::TunePlannerInterfaceSpec,
    planner_simulations_per_step: usize,
    return_horizon: Option<usize>,
    label_phase_period: Option<usize>,
    discount_factor: f64,
    reward_semantics: PlannerRewardSemantics,
    clipping_interval: Option<(f64, f64)>,
    teacher: Option<WarmstartTeacherDataset>,
    warmstart_self_improvement: bool,
}

#[derive(Clone, Debug)]
struct PlannerEncodedPercept {
    observations: Vec<PerceptVal>,
    reward: Reward,
}

impl PlannerEncodedPercept {
    fn observation_slice(&self) -> &[PerceptVal] {
        &self.observations
    }
}

impl PlannerControllerContract {
    fn reward_encoder(
        &self,
        dataset: &LoadedDataset,
        baseline_eval: &CandidateEvalResult,
        verified_theorem: &VerifiedTheoremInputs,
    ) -> Result<TunerRewardEncoder, String> {
        match self.reward_semantics {
            PlannerRewardSemantics::ExactObjectiveDifference => {
                TunerRewardEncoder::exact_integer_objective_difference(
                    self.interface.reward_bits,
                    dataset,
                    baseline_eval,
                    verified_theorem.exact_reward_encoding.as_ref(),
                )
            }
            PlannerRewardSemantics::NormalizedClippedImprovement => {
                let (min_improvement, max_improvement) =
                    self.clipping_interval.ok_or_else(|| {
                        "normalized clipped reward contract was not initialized".to_string()
                    })?;
                TunerRewardEncoder::normalized_clipped(
                    self.interface.reward_bits,
                    min_improvement,
                    max_improvement,
                )
            }
        }
    }
}

#[derive(Clone, Debug)]
enum TunerRewardEncoder {
    ExactIntegerObjectiveDifference {
        max_reward: Reward,
        objective_difference_to_symbol: Option<BTreeMap<Reward, Reward>>,
    },
    NormalizedClipped {
        min_improvement: f64,
        max_improvement: f64,
        max_reward: Reward,
    },
}

impl TunerRewardEncoder {
    fn exact_integer_objective_difference(
        reward_bits: usize,
        dataset: &LoadedDataset,
        baseline_eval: &CandidateEvalResult,
        certificate: Option<&VerifiedExactRewardEncodingCertificate>,
    ) -> Result<Self, String> {
        let certificate = certificate.ok_or_else(|| {
            "reward_encoding_unsafe: exact objective-difference planner controllers require a verified exact_reward_encoding_certificate"
                .to_string()
        })?;
        if certificate.reward_bits != reward_bits {
            return Err(format!(
                "reward_encoding_unsafe: certificate reward_bits={} does not match controller reward_bits={reward_bits}",
                certificate.reward_bits
            ));
        }
        let baseline_objective =
            exact_nonnegative_i64_from_f64(baseline_eval.objective_bits, "baseline objective")?;
        let max_encoded = max_nonnegative_reward_for_bits(reward_bits)?;
        if baseline_objective > max_encoded {
            return Err(format!(
                "reward_encoding_unsafe: reward_bits={reward_bits} cannot injectively encode reachable exact objective differences up to baseline objective {baseline_objective} (max encoded {max_encoded})"
            ));
        }
        if baseline_objective > certificate.max_reward {
            return Err(format!(
                "reward_encoding_unsafe: baseline objective {baseline_objective} exceeds verified exact reward maximum {}",
                certificate.max_reward
            ));
        }
        let objective_difference_to_symbol = match &certificate.mode {
            VerifiedRewardEncodingMode::IntegerObjectiveDifferenceInterval => None,
            VerifiedRewardEncodingMode::FiniteRewardMap { map } => {
                let objective_difference_to_symbol = &map.objective_difference_to_symbol;
                if !objective_difference_to_symbol.contains_key(&0) {
                    return Err(
                        "reward_encoding_unsafe: finite reward map must encode zero improvement"
                            .to_string(),
                    );
                }
                if !objective_difference_to_symbol.contains_key(&baseline_objective) {
                    return Err(format!(
                        "reward_encoding_unsafe: finite reward map must include baseline objective difference {baseline_objective}"
                    ));
                }
                let complete_max = map.complete_nonnegative_interval_max.ok_or_else(|| {
                    "reward_encoding_unsafe: finite reward map certificates used by exact controllers must declare complete_nonnegative_interval_max"
                        .to_string()
                })?;
                if complete_max < baseline_objective {
                    return Err(format!(
                        "reward_encoding_unsafe: finite reward map complete_nonnegative_interval_max {complete_max} is below baseline objective difference {baseline_objective}"
                    ));
                }
                validate_complete_finite_reward_interval(
                    objective_difference_to_symbol,
                    complete_max,
                )?;
                Some(objective_difference_to_symbol.clone())
            }
        };
        if dataset.kind != DatasetKind::PassiveBytes
            && certificate.scalar_representation == SCALAR_REPRESENTATION_DECLARATION
        {
            return Err(
                "reward_encoding_unsafe: non-passive exact objective-difference runs require a task-specific finite scalar certificate, not the passive default declaration"
                    .to_string(),
            );
        }
        Ok(Self::ExactIntegerObjectiveDifference {
            max_reward: certificate.max_reward.min(baseline_objective),
            objective_difference_to_symbol,
        })
    }

    fn normalized_clipped(
        reward_bits: usize,
        min_improvement: f64,
        max_improvement: f64,
    ) -> Result<Self, String> {
        if !min_improvement.is_finite() || !max_improvement.is_finite() {
            return Err("normalized clipped reward bounds must be finite".to_string());
        }
        if max_improvement <= min_improvement {
            return Err(
                "normalized clipped reward contract requires max_improvement > min_improvement"
                    .to_string(),
            );
        }
        Ok(Self::NormalizedClipped {
            min_improvement,
            max_improvement,
            max_reward: max_nonnegative_reward_for_bits(reward_bits)?,
        })
    }

    fn max_reward(&self) -> Reward {
        match self {
            Self::ExactIntegerObjectiveDifference { max_reward, .. }
            | Self::NormalizedClipped { max_reward, .. } => *max_reward,
        }
    }

    fn encode(&self, raw_improvement: f64) -> Result<Reward, String> {
        if !raw_improvement.is_finite() {
            return Err("planner reward improvement must be finite".to_string());
        }
        match self {
            Self::ExactIntegerObjectiveDifference {
                max_reward,
                objective_difference_to_symbol,
            } => {
                if raw_improvement < 0.0 {
                    return Err("exact objective-difference reward cannot be negative".to_string());
                }
                let reward =
                    exact_nonnegative_i64_from_f64(raw_improvement, "planner reward improvement")?;
                if let Some(map) = objective_difference_to_symbol {
                    return map.get(&reward).copied().ok_or_else(|| {
                        format!(
                            "reward_encoding_unsafe: exact objective difference {reward} is missing from verified finite reward map"
                        )
                    });
                }
                if reward > *max_reward {
                    return Err(format!(
                        "reward_encoding_unsafe: exact reward {reward} exceeds certified maximum {max_reward}"
                    ));
                }
                Ok(reward)
            }
            Self::NormalizedClipped {
                min_improvement,
                max_improvement,
                max_reward,
            } => {
                let normalized = normalized_clipped_improvement(
                    raw_improvement,
                    *min_improvement,
                    *max_improvement,
                )?;
                Ok((normalized * *max_reward as f64).round() as Reward)
            }
        }
    }
}

enum TunerPlannerAgentRuntime {
    McAixi {
        agent: Agent,
        prev_action: Action,
        prev_percept: PlannerEncodedPercept,
    },
    AiqiDiscounted {
        agent: AiqiAgent,
    },
    WarmStartExactJh {
        agent: WarmStartExactJhAgent,
    },
}

impl TunerPlannerAgentRuntime {
    fn select_action(&mut self) -> Action {
        match self {
            Self::McAixi {
                agent,
                prev_action,
                prev_percept,
            } => {
                agent.model_update_percept_stream(
                    prev_percept.observation_slice(),
                    prev_percept.reward,
                );
                let action = agent.get_planned_action(
                    prev_percept.observation_slice(),
                    prev_percept.reward,
                    *prev_action,
                );
                agent.model_update_action_external(action);
                *prev_action = action;
                action
            }
            Self::AiqiDiscounted { agent } => agent.get_planned_action(),
            Self::WarmStartExactJh { agent } => agent.get_planned_action(),
        }
    }

    fn observe_transition(
        &mut self,
        action: Action,
        percept: PlannerEncodedPercept,
    ) -> Result<(), String> {
        match self {
            Self::McAixi { prev_percept, .. } => {
                *prev_percept = percept;
                Ok(())
            }
            Self::AiqiDiscounted { agent } => agent
                .observe_transition(action, percept.observation_slice(), percept.reward)
                .map_err(|err| err.to_string()),
            Self::WarmStartExactJh { agent } => agent
                .observe_transition(action, percept.observation_slice(), percept.reward)
                .map_err(|err| err.to_string()),
        }
    }

    fn same_task_live_trace(&self) -> Option<WarmStartExactJhTeacherTrace> {
        match self {
            Self::WarmStartExactJh { agent } => agent.same_task_live_trace(),
            Self::McAixi { .. } | Self::AiqiDiscounted { .. } => None,
        }
    }

    fn rebuild_warmstart_agent(
        &mut self,
        planner_run: &CompiledPlannerRunSpec,
        teacher: WarmStartExactJhTeacherDataset,
    ) -> Result<(), String> {
        match self {
            Self::WarmStartExactJh { agent } => {
                *agent = WarmStartExactJhAgent::from_compiled_planner_run(planner_run, teacher)
                    .map_err(|err| err.to_string())?;
                Ok(())
            }
            Self::McAixi { .. } | Self::AiqiDiscounted { .. } => Err(
                "trace refresh is only defined for warm-start exact-J_H controllers".to_string(),
            ),
        }
    }
}

/// Execute tuning for one canonical tune document using executor-side controls.
///
/// The runtime enforces canonical/executor separation, baseline deployability
/// preconditions, and controller-specific bounded search semantics.
pub fn run_tune(request: &TuneCommandRequest) -> Result<(), String> {
    let prepared = prepare_tune_context(request)?;
    if let Some(path) = request.emit_exact_reward_encoding_certificate.as_deref() {
        emit_exact_reward_encoding_certificate(request, path, &prepared)?;
        return Ok(());
    }
    let PreparedTuneContext {
        compiled,
        dataset,
        runtime_profile,
        evaluator_profile,
        initial_eval_limit,
        tune_started,
    } = prepared;
    let verified_theorem = VerifiedTheoremInputs::load(
        &request.execution.theorem,
        &compiled,
        &dataset,
        &evaluator_profile,
        compiled.base_dir(),
    )?;
    apply_executor_controls(&request.execution)?;

    for _ in 0..request.execution.warmup_baseline_runs {
        match evaluate_candidate(
            compiled.baseline_candidate(),
            &dataset,
            compiled.baseline_candidate_model_bytes(),
            compiled.canonical_spec().min_throughput_bytes_per_second,
            compiled.canonical_spec().max_memory_bytes,
            initial_eval_limit,
            request.execution.evaluator_threads(),
            &runtime_profile,
            verified_theorem.deterministic_table.as_ref(),
        ) {
            Ok(_) => {}
            Err(CandidateEvalFailure::FatalEvaluatorFailure { diagnostic }) => {
                return Err(format!(
                    "unrecoverable evaluator failure during baseline warmup: {diagnostic}"
                ));
            }
        }
    }

    let baseline_eval = match evaluate_candidate(
        compiled.baseline_candidate(),
        &dataset,
        compiled.baseline_candidate_model_bytes(),
        compiled.canonical_spec().min_throughput_bytes_per_second,
        compiled.canonical_spec().max_memory_bytes,
        initial_eval_limit,
        request.execution.evaluator_threads(),
        &runtime_profile,
        verified_theorem.deterministic_table.as_ref(),
    ) {
        Ok(value) => value,
        Err(CandidateEvalFailure::FatalEvaluatorFailure { diagnostic }) => {
            return Err(format!(
                "unrecoverable evaluator failure during baseline evaluation: {diagnostic}"
            ));
        }
    };
    let baseline_key = cache_key_for_candidate(
        compiled.baseline_candidate().canonical_bytes().as_slice(),
        &evaluator_profile,
        &dataset.canonical_content_hash,
    )?;
    let baseline_hash = crc32_hex(compiled.baseline_candidate().canonical_bytes().as_slice());
    let baseline_bytes = compiled
        .baseline_candidate()
        .canonical_bytes()
        .as_slice()
        .to_vec();
    let mut cache = HashMap::<CandidateCacheKey, CandidateEvalResult>::new();
    cache.insert(baseline_key.clone(), baseline_eval.clone());
    let doc_hash = crc32_hex(compiled.canonical_bytes().as_slice());
    let evaluator_profile_hash = evaluator_profile.hash()?;
    let search_summary = if baseline_eval.deployable {
        run_controller_search(
            &compiled,
            request,
            &dataset,
            &evaluator_profile,
            &verified_theorem,
            tune_started,
            baseline_eval.clone(),
            baseline_hash.clone(),
            baseline_bytes.clone(),
            baseline_key.clone(),
            &runtime_profile,
            &mut cache,
        )?
    } else {
        SearchSummary {
            status: "baseline_not_deployable",
            warning: None,
            fatal_evaluator_failure: None,
            fatal_evaluator_failures: 0,
            best_candidate: compiled.canonical_spec().baseline_candidate.clone(),
            best_candidate_crc32: baseline_hash.clone(),
            best_eval: baseline_eval.clone(),
            cache_key_digest: baseline_key.digest_crc32(),
            cache_hits: 0,
            cache_misses: 1,
            candidate_evaluations_executed: 1,
            non_warmup_candidate_results_seen: 1,
            post_baseline_candidate_results_seen: 0,
            proposals_attempted: 0,
            proposals_invalid: 0,
            self_loop_proposals: 0,
            invalid_reason_counts: InvalidReasonCounts::default(),
            successful_non_deployable: if baseline_eval.status == CandidateEvalStatus::Success
                && !baseline_eval.deployable
            {
                1
            } else {
                0
            },
            candidate_result_counts: {
                let mut counts = CandidateResultCounts::default();
                counts.record_admitted_result(&baseline_eval);
                counts
            },
            final_best_move_reward: 0.0,
            realized_trace_counts_by_round: None,
            trace_refresh_merges_by_round: None,
            controller_report: serde_json::json!({
                "kind": controller_kind_name(compiled.controller()),
                "runtime_path": "baseline_precondition_failed",
                "baseline_deployable_precondition": false,
            }),
        }
    };

    let output_path = compiled.canonical_spec().output_config_path.as_str();
    let output_written = if search_summary.best_eval.deployable {
        let output_json = search_summary
            .best_candidate
            .to_canonical_json()
            .map_err(|err| format!("failed to serialize output candidate: {err}"))?;
        fs::write(output_path, output_json).map_err(|err| {
            format!(
                "failed to write output_config_path '{}': {err}",
                output_path
            )
        })?;
        true
    } else {
        false
    };
    let theorem_claims = theorem_claims_report(
        &request.execution.theorem,
        &verified_theorem,
        compiled.controller(),
        &dataset,
        &search_summary,
        runtime_profile.strict_theorem_memory_certified(),
    );
    let bounds_hash = bounds_hash(&compiled.canonical_spec().bounds)?;
    let observation_adapter_hash = observation_adapter_content_hash()?;
    let evaluator_execution_model =
        evaluator_execution_model(verified_theorem.deterministic_table.as_ref());
    let theorem_timing_basis = theorem_timing_basis(&request.execution.theorem, &verified_theorem);
    let best_model_bytes = search_summary
        .best_candidate
        .compile_in(&SpecEnvironment::new(compiled.base_dir()))
        .map_err(|err| format!("failed to compile best candidate for reporting: {err}"))?
        .canonical_bytes()
        .len();
    let throughput_runtime_cap_seconds =
        dataset.dataset_units / compiled.canonical_spec().min_throughput_bytes_per_second;
    let warmstart_self_improvement_enabled = matches!(
        compiled.controller(),
        crate::spec::CompiledTuneController::AiqiWarmstartExactJh(_)
    );
    let same_task_trace_refresh_enabled =
        warmstart_self_improvement_enabled && request.execution.warmstart_trace_refresh;
    let effective_self_improvement_rounds = if warmstart_self_improvement_enabled {
        request.execution.self_improvement_rounds.max(1)
    } else {
        1
    };
    let self_improvement_round_deadlines_seconds =
        if warmstart_self_improvement_enabled && effective_self_improvement_rounds > 1 {
            Some(
                (1..=effective_self_improvement_rounds)
                    .map(|round| {
                        ((round as f64) / (effective_self_improvement_rounds as f64))
                            * compiled.canonical_spec().time_budget_seconds
                    })
                    .collect::<Vec<f64>>(),
            )
        } else {
            None
        };

    let report = serde_json::json!({
        "schema_version": 1,
        "kind": "tune_report",
        "status": search_summary.status,
        "warning": search_summary.warning,
        "crate_version": env!("CARGO_PKG_VERSION"),
        "feature_set": compiled_feature_set(),
        "spec_path": request.spec_path,
        "spec_crc32": doc_hash,
        "seed": compiled.canonical_spec().seed,
        "timing_certification_tier": timing_tier_name(request.execution.theorem.timing_certification_tier),
        "determinism_deadline_certificate": request.execution.theorem.determinism_deadline_certificate,
        "evaluator_execution_model": evaluator_execution_model,
        "theorem_timing_basis": theorem_timing_basis,
        "theorem_claims": theorem_claims,
        "execution_profile": request.execution.to_json_value(),
        "evaluator_profile": evaluator_profile.to_json_value(),
        "evaluator_profile_crc32": evaluator_profile_hash,
        "provenance": {
            "bounds_crc32": bounds_hash,
            "candidate_canonicalization_version": compiled.candidate_canonicalization_version(),
            "observation_adapter_spec_ref": request.execution.theorem.observation_adapter_spec_ref.as_deref().unwrap_or(OBSERVATION_ADAPTER_DECLARATION),
            "observation_adapter_content_crc32": observation_adapter_hash,
            "exact_state_encoder_spec_ref": request.execution.theorem.exact_state_encoder_spec_ref,
            "exact_state_observation_certified": verified_theorem.exact_state_observation.is_some(),
            "scalar_representation_ref": request.execution.theorem.scalar_representation_ref.as_deref().unwrap_or(SCALAR_REPRESENTATION_DECLARATION),
            "evaluator_execution_model": evaluator_execution_model,
            "theorem_timing_basis": theorem_timing_basis,
            "verified_theorem_inputs": verified_theorem.to_json_value(),
            "resolved_evaluator_runtime_profile": runtime_profile.to_provenance_value(),
            "canonical_code_certification": {
                "basis": "structural_self_delimiting_binary_encoding_plus_tests",
                "mechanized": false,
                "sample_corpus_checked": true,
                "trailing_bytes_rejected": true,
                "top_level_length_prefix": true,
            },
            "warmup_policy": {
                "warmup_baseline_runs": request.execution.warmup_baseline_runs,
                "excluded_from_cache": true,
                "excluded_from_optimization_metrics": true,
            },
            "self_improvement_policy": {
                "rounds": effective_self_improvement_rounds,
                "same_task_trace_refresh_enabled": same_task_trace_refresh_enabled,
                "online_delayed_label_update_enabled": warmstart_self_improvement_enabled && !same_task_trace_refresh_enabled,
                "deterministic_round_deadlines_seconds": self_improvement_round_deadlines_seconds,
                "realized_trace_counts_by_round": search_summary.realized_trace_counts_by_round.clone(),
                "trace_refresh_merges_by_round": search_summary.trace_refresh_merges_by_round.clone(),
            },
            "stagnation_policy": {
                "stagnation_reset_evals": request.execution.stagnation_reset_evals,
            },
            "executor_controls": executor_controls_report(
                &request.execution,
                &runtime_profile,
            ),
            "diagnostic_chunking": diagnostic_chunking_report(
                &dataset,
                request.execution.diagnostic_chunk_bytes,
            ),
        },
        "cache": {
            "key_candidate_crc32": search_summary.best_candidate_crc32,
            "key_digest_crc32": search_summary.cache_key_digest,
            "dataset_content_crc32": dataset.canonical_content_hash,
            "warmup_runs_excluded_from_cache": true,
            "actual_evaluator_calls_excluding_warmups": search_summary.candidate_evaluations_executed,
            "candidate_evaluations_executed": search_summary.candidate_evaluations_executed,
            "cache_hits": search_summary.cache_hits,
            "cache_misses": search_summary.cache_misses,
        },
        "search": {
            "termination_reason": search_summary.status,
            "fatal_evaluator_failures": search_summary.fatal_evaluator_failures,
            "fatal_evaluator_failure": search_summary.fatal_evaluator_failure.clone(),
            "proposals_attempted": search_summary.proposals_attempted,
            "proposals_invalid": search_summary.proposals_invalid,
            "self_loop_proposals": search_summary.self_loop_proposals,
            "invalid_reason_counts": {
                "candidate_external_asset_forbidden": search_summary.invalid_reason_counts.candidate_external_asset_forbidden,
                "candidate_out_of_bounds": search_summary.invalid_reason_counts.candidate_out_of_bounds,
                "candidate_compile_error": search_summary.invalid_reason_counts.candidate_compile_error,
                "invalid_action_index": search_summary.invalid_reason_counts.invalid_action_index,
                "inapplicable_action": search_summary.invalid_reason_counts.inapplicable_action,
            },
            "successful_non_deployable": search_summary.successful_non_deployable,
            "candidate_result_counts": {
                "success_deployable": search_summary.candidate_result_counts.success_deployable,
                "success_non_deployable": search_summary.candidate_result_counts.success_non_deployable,
                "timeout": search_summary.candidate_result_counts.timeout,
                "invalid": search_summary.candidate_result_counts.invalid,
                "error_recoverable": search_summary.candidate_result_counts.error_recoverable,
            },
            "self_improvement_rounds": effective_self_improvement_rounds,
            "max_evaluations": request.execution.max_evaluations,
            "max_evaluations_semantics": "baseline_included_warmups_excluded_cache_hits_included_for_search_steps",
            "baseline_counts_toward_max_evaluations": true,
            "non_warmup_candidate_results_seen": search_summary.non_warmup_candidate_results_seen,
            "post_baseline_candidate_results_seen": search_summary.post_baseline_candidate_results_seen,
            "time_budget_seconds": compiled.canonical_spec().time_budget_seconds,
            "final_best_move_reward": search_summary.final_best_move_reward,
            "controller": search_summary.controller_report,
            "planner_deployability": planner_deployability_report(
                request.execution.planner_deployable_model,
                best_model_bytes,
                search_summary.best_eval.elapsed_seconds,
                search_summary.best_eval.deployable,
            ),
        },
        "input_asset": {
            "id": compiled.canonical_spec().input_asset,
            "resolved_path": dataset.resolved_path,
            "content_crc32": dataset.canonical_content_hash,
            "lowered_skeleton_crc32": dataset.lowered_skeleton_hash,
            "target_domain_support_crc32": dataset.target_domain_support_hash,
            "causal_header_profile_crc32": dataset.causal_header_profile_hash,
            "causal_profile": causal_profile_report(&dataset),
            "dataset_kind": dataset_kind_name(dataset.kind),
            "source_size_bytes": dataset.source_size_bytes,
            "charged_target_bytes": dataset.raw_bytes.len(),
            "dataset_units": dataset.dataset_units,
            "target_events": dataset.target_events,
            "target_size_function": dataset.target_size_function,
            "diagnostic_chunking": diagnostic_chunking_report(
                &dataset,
                request.execution.diagnostic_chunk_bytes,
            ),
        },
        "baseline": {
            "candidate_crc32": baseline_hash,
            "status": baseline_eval.status.name(),
            "model_bytes": compiled.baseline_candidate_model_bytes(),
            "compressed_bytes": baseline_eval.compressed_bytes,
            "physical_compressed_bytes_diagnostic_only": true,
            "target_loss_bits": baseline_eval.target_loss_bits,
            "objective_bits": baseline_eval.objective_bits,
            "elapsed_seconds": baseline_eval.elapsed_seconds,
            "throughput_bytes_per_second": baseline_eval.throughput_bytes_per_second,
            "peak_memory_bytes": baseline_eval.peak_memory_bytes,
            "min_throughput_bytes_per_second": compiled.canonical_spec().min_throughput_bytes_per_second,
            "throughput_runtime_cap_seconds": throughput_runtime_cap_seconds,
            "max_memory_bytes": compiled.canonical_spec().max_memory_bytes,
            "effective_eval_time_limit_seconds": baseline_eval.effective_eval_time_limit_seconds,
            "deployable": baseline_eval.deployable,
        },
        "best": {
            "candidate_crc32": search_summary.best_candidate_crc32,
            "status": search_summary.best_eval.status.name(),
            "model_bytes": best_model_bytes,
            "compressed_bytes": search_summary.best_eval.compressed_bytes,
            "physical_compressed_bytes_diagnostic_only": true,
            "target_loss_bits": search_summary.best_eval.target_loss_bits,
            "objective_bits": search_summary.best_eval.objective_bits,
            "elapsed_seconds": search_summary.best_eval.elapsed_seconds,
            "throughput_bytes_per_second": search_summary.best_eval.throughput_bytes_per_second,
            "peak_memory_bytes": search_summary.best_eval.peak_memory_bytes,
            "effective_eval_time_limit_seconds": search_summary.best_eval.effective_eval_time_limit_seconds,
            "deployable": search_summary.best_eval.deployable,
        },
        "output": {
            "output_config_path": output_path,
            "output_written": output_written,
            "output_candidate_crc32": if output_written {
                Some(search_summary.best_candidate_crc32.clone())
            } else {
                None::<String>
            },
        },
    });

    if let Some(path) = compiled.canonical_spec().report_path.as_deref() {
        let text = serde_json::to_string_pretty(&report)
            .map_err(|err| format!("failed to serialize report: {err}"))?;
        fs::write(path, text)
            .map_err(|err| format!("failed to write report_path '{}': {err}", path))?;
    }
    log_executor_event(
        &request.execution,
        "finish",
        serde_json::json!({
            "status": search_summary.status,
            "output_written": output_written,
            "best_eval_status": search_summary.best_eval.status.name(),
        }),
    )?;

    if let Some(diagnostic) = search_summary.fatal_evaluator_failure.as_deref() {
        Err(format!(
            "tuning terminated due to unrecoverable evaluator failure: {diagnostic}"
        ))
    } else if search_summary.best_eval.deployable {
        Ok(())
    } else if search_summary.best_eval.status == CandidateEvalStatus::Timeout {
        Err(format!(
            "baseline candidate timed out under effective evaluator limit {:.6} seconds",
            initial_eval_limit
        ))
    } else {
        Err(format!(
            "baseline candidate is not deployable under tune constraints: throughput {:.3} B/s (required >= {:.3}), peak memory {} bytes (required <= {})",
            search_summary.best_eval.throughput_bytes_per_second,
            compiled.canonical_spec().min_throughput_bytes_per_second,
            search_summary.best_eval.peak_memory_bytes,
            compiled.canonical_spec().max_memory_bytes
        ))
    }
}

fn resolve_input_asset_path<'a>(
    compiled: &'a crate::spec::CompiledTuneSpec,
    input_asset_id: &str,
) -> Result<&'a Path, String> {
    let binding = compiled
        .resolved_assets()
        .iter()
        .find(|entry| entry.id == input_asset_id)
        .ok_or_else(|| format!("unknown input_asset id '{}'", input_asset_id))?;
    let AssetRef::Filesystem(path) = &binding.asset;
    Ok(path.as_path())
}

fn effective_eval_limit_seconds(
    compiled: &crate::spec::CompiledTuneSpec,
    tune_started: Instant,
    full_limit_seconds: Option<f64>,
    round_deadline_seconds: Option<f64>,
) -> f64 {
    let elapsed = tune_started.elapsed().as_secs_f64();
    let remaining_total = (compiled.canonical_spec().time_budget_seconds - elapsed).max(0.0);
    let remaining_round = round_deadline_seconds
        .map(|deadline| (deadline - elapsed).max(0.0))
        .unwrap_or(f64::INFINITY);
    full_limit_seconds
        .unwrap_or(compiled.canonical_spec().eval_time_limit_seconds)
        .min(remaining_total)
        .min(remaining_round)
        .max(0.0)
}

fn apply_executor_controls(config: &TuneExecutionConfig) -> Result<(), String> {
    if let Some(cpu_affinity) = config.cpu_affinity.as_deref() {
        apply_cpu_affinity(cpu_affinity)?;
    }
    log_executor_event(config, "start", serde_json::json!({}))?;
    Ok(())
}

#[cfg(target_os = "linux")]
fn apply_cpu_affinity(raw: &str) -> Result<(), String> {
    // SAFETY: `cpu_set_t` is a plain C bitset type for the Linux affinity API.
    // Zero-initialization is the documented starting state before `CPU_ZERO`.
    let mut set = unsafe { std::mem::zeroed::<libc::cpu_set_t>() };
    // SAFETY: `set` is a valid, writable `cpu_set_t` local variable.
    unsafe {
        libc::CPU_ZERO(&mut set);
    }
    let mut count = 0usize;
    for part in raw.split(',') {
        let cpu = part
            .trim()
            .parse::<usize>()
            .map_err(|_| format!("invalid cpu_affinity entry '{}'", part.trim()))?;
        // SAFETY: `set` is valid and writable for the duration of the call; CPU_SET
        // only mutates the bitset. Kernel validation of out-of-range CPU indices is
        // handled by the subsequent `sched_setaffinity` call.
        unsafe {
            libc::CPU_SET(cpu, &mut set);
        }
        count = count.saturating_add(1);
    }
    if count == 0 {
        return Err("cpu_affinity must name at least one CPU".to_string());
    }
    // SAFETY: The pointer references a live `cpu_set_t`, the size matches that type,
    // and pid 0 intentionally targets the current process per sched_setaffinity(2).
    let result =
        unsafe { libc::sched_setaffinity(0, std::mem::size_of::<libc::cpu_set_t>(), &set) };
    if result == 0 {
        Ok(())
    } else {
        Err(format!(
            "failed to apply cpu_affinity '{}': {}",
            raw,
            std::io::Error::last_os_error()
        ))
    }
}

#[cfg(not(target_os = "linux"))]
fn apply_cpu_affinity(raw: &str) -> Result<(), String> {
    Err(format!(
        "cpu_affinity '{}' is only supported on Linux by this tuner runtime",
        raw
    ))
}

fn log_executor_event(
    config: &TuneExecutionConfig,
    event: &str,
    payload: Value,
) -> Result<(), String> {
    let Some(path) = config.log_path.as_deref() else {
        return Ok(());
    };
    let mut file = fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .map_err(|err| format!("failed to open log_path '{}': {err}", path))?;
    let line = serde_json::to_string(&serde_json::json!({
        "kind": "tune_executor_event",
        "event": event,
        "payload": payload,
    }))
    .map_err(|err| format!("failed to serialize executor log event: {err}"))?;
    writeln!(file, "{line}").map_err(|err| format!("failed to write log_path '{}': {err}", path))
}

fn planner_action_count(compiled: &crate::spec::CompiledTuneSpec) -> Result<usize, String> {
    match compiled.controller() {
        crate::spec::CompiledTuneController::AnnealedHillClimbing(_) => Ok(0),
        crate::spec::CompiledTuneController::McAixiFacCtw(inner) => {
            Ok(inner.interface.agent_actions.get())
        }
        crate::spec::CompiledTuneController::AiqiDiscounted(inner) => {
            Ok(inner.interface.agent_actions.get())
        }
        crate::spec::CompiledTuneController::AiqiWarmstartExactJh(inner) => {
            Ok(inner.interface.agent_actions.get())
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn run_controller_search(
    compiled: &crate::spec::CompiledTuneSpec,
    request: &TuneCommandRequest,
    dataset: &LoadedDataset,
    evaluator_profile: &EvaluatorProfile,
    verified_theorem: &VerifiedTheoremInputs,
    tune_started: Instant,
    baseline_eval: CandidateEvalResult,
    baseline_hash: String,
    baseline_bytes: Vec<u8>,
    baseline_key: CandidateCacheKey,
    runtime_profile: &ResolvedEvaluatorRuntimeProfile,
    cache: &mut HashMap<CandidateCacheKey, CandidateEvalResult>,
) -> Result<SearchSummary, String> {
    match compiled.controller() {
        crate::spec::CompiledTuneController::AnnealedHillClimbing(inner) => {
            run_annealed_hill_climbing(
                compiled,
                request,
                dataset,
                evaluator_profile,
                verified_theorem,
                tune_started,
                baseline_eval,
                baseline_hash,
                baseline_bytes,
                baseline_key,
                runtime_profile,
                cache,
                inner.max_mutation_radius,
            )
        }
        crate::spec::CompiledTuneController::McAixiFacCtw(_)
        | crate::spec::CompiledTuneController::AiqiDiscounted(_)
        | crate::spec::CompiledTuneController::AiqiWarmstartExactJh(_) => {
            run_planner_family_controller(
                compiled,
                request,
                dataset,
                evaluator_profile,
                verified_theorem,
                tune_started,
                baseline_eval,
                baseline_hash,
                baseline_bytes,
                baseline_key,
                runtime_profile,
                cache,
            )
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn run_annealed_hill_climbing(
    compiled: &crate::spec::CompiledTuneSpec,
    request: &TuneCommandRequest,
    dataset: &LoadedDataset,
    evaluator_profile: &EvaluatorProfile,
    verified_theorem: &VerifiedTheoremInputs,
    tune_started: Instant,
    baseline_eval: CandidateEvalResult,
    baseline_hash: String,
    baseline_bytes: Vec<u8>,
    baseline_key: CandidateCacheKey,
    runtime_profile: &ResolvedEvaluatorRuntimeProfile,
    cache: &mut HashMap<CandidateCacheKey, CandidateEvalResult>,
    max_mutation_radius: usize,
) -> Result<SearchSummary, String> {
    let env = SpecEnvironment::new(compiled.base_dir());
    let mut rng = RandomGenerator::from_seed(compiled.canonical_spec().seed);

    let mut best_candidate = compiled.canonical_spec().baseline_candidate.clone();
    let mut best_eval = baseline_eval.clone();
    let mut best_hash = baseline_hash;
    let mut best_bytes = baseline_bytes;
    let mut best_key = baseline_key;

    let mut current_candidate = best_candidate.clone();
    let mut current_eval = baseline_eval;
    let mut candidate_result_counts = CandidateResultCounts::default();
    candidate_result_counts.record_admitted_result(&current_eval);
    let mut cache_hits: usize = 0;
    let mut cache_misses: usize = 1;
    let mut candidate_evaluations_executed: usize = 1;
    let mut proposals_attempted: usize = 0;
    let mut proposals_invalid: usize = 0;
    let mut self_loop_proposals: usize = 0;
    let mut invalid_reason_counts = InvalidReasonCounts::default();
    let mut successful_non_deployable: usize = 0;
    let mut final_best_move_reward: f64 = 0.0;
    let mut evaluations_seen: usize = 1;
    let max_evaluations = request.execution.max_evaluations.unwrap_or(usize::MAX);
    let mut fatal_evaluator_failure: Option<String> = None;

    let mut stagnation_counter: usize = 0;
    'search: loop {
        if evaluations_seen >= max_evaluations {
            break;
        }
        if tune_started.elapsed().as_secs_f64() >= compiled.canonical_spec().time_budget_seconds {
            break;
        }
        proposals_attempted = proposals_attempted.saturating_add(1);

        let progress =
            annealer_progress(tune_started, compiled.canonical_spec().time_budget_seconds);
        let temperature = annealer_temperature(progress);
        let active_radius = annealer_active_radius(max_mutation_radius, temperature);
        let proposal = match sample_annealed_proposal(
            &current_candidate,
            &compiled.canonical_spec().bounds,
            max_mutation_radius,
            active_radius,
            &env,
            &mut rng,
        )? {
            AnnealedProposalDraw::Proposal(proposal) => proposal,
            AnnealedProposalDraw::SelfLoop => {
                self_loop_proposals = self_loop_proposals.saturating_add(1);
                continue;
            }
            AnnealedProposalDraw::Exhausted => {
                self_loop_proposals = self_loop_proposals.saturating_add(1);
                break;
            }
        };
        let proposed_candidate = proposal.candidate.clone();

        if let Err(err) = reject_candidate_local_external_artifacts(&proposed_candidate) {
            proposals_invalid = proposals_invalid.saturating_add(1);
            invalid_reason_counts.record(err.reason);
            continue;
        }
        if validate_candidate_against_tune_bounds(
            &proposed_candidate,
            &compiled.canonical_spec().bounds,
        )
        .is_err()
        {
            proposals_invalid = proposals_invalid.saturating_add(1);
            invalid_reason_counts.record(TuneInvalidReason::CandidateOutOfBounds);
            continue;
        }

        let compiled_candidate = match proposed_candidate.compile_in(&env) {
            Ok(value) => value,
            Err(_) => {
                proposals_invalid = proposals_invalid.saturating_add(1);
                invalid_reason_counts.record(TuneInvalidReason::CandidateCompileError);
                continue;
            }
        };

        let candidate_bytes = compiled_candidate.canonical_bytes().as_slice().to_vec();
        let candidate_hash = crc32_hex(&candidate_bytes);
        let effective_limit = effective_eval_limit_seconds(
            compiled,
            tune_started,
            Some(compiled.canonical_spec().eval_time_limit_seconds),
            None,
        );
        let candidate_profile = evaluator_profile.with_eval_time_limit(effective_limit);
        let cache_key = cache_key_for_candidate(
            compiled_candidate.canonical_bytes().as_slice(),
            &candidate_profile,
            &dataset.canonical_content_hash,
        )?;
        let candidate_eval = if let Some(cached) = cache.get(&cache_key) {
            cache_hits = cache_hits.saturating_add(1);
            cached.clone()
        } else {
            let evaluated = match evaluate_candidate(
                &compiled_candidate,
                dataset,
                compiled_candidate.canonical_bytes().len(),
                compiled.canonical_spec().min_throughput_bytes_per_second,
                compiled.canonical_spec().max_memory_bytes,
                effective_limit,
                request.execution.evaluator_threads(),
                runtime_profile,
                verified_theorem.deterministic_table.as_ref(),
            ) {
                Ok(value) => value,
                Err(CandidateEvalFailure::FatalEvaluatorFailure { diagnostic }) => {
                    fatal_evaluator_failure = Some(diagnostic);
                    break 'search;
                }
            };
            cache.insert(cache_key.clone(), evaluated.clone());
            cache_misses = cache_misses.saturating_add(1);
            candidate_evaluations_executed = candidate_evaluations_executed.saturating_add(1);
            evaluated
        };
        evaluations_seen = evaluations_seen.saturating_add(1);
        candidate_result_counts.record_admitted_result(&candidate_eval);

        if candidate_eval.status == CandidateEvalStatus::Success && !candidate_eval.deployable {
            successful_non_deployable = successful_non_deployable.saturating_add(1);
        }

        if !candidate_eval.deployable {
            continue;
        }

        let delta = candidate_eval.objective_bits - current_eval.objective_bits;
        let accept_probability = annealer_acceptance_probability(
            request.execution.annealer_kernel_profile,
            delta,
            temperature,
            &proposal,
        )?;
        let accept = rng.gen_f64() < accept_probability;

        if accept {
            current_candidate = proposed_candidate.clone();
            current_eval = candidate_eval.clone();
        }

        if key_less(&candidate_eval, &candidate_bytes, &best_eval, &best_bytes) {
            final_best_move_reward =
                (best_eval.objective_bits - candidate_eval.objective_bits).max(0.0);
            best_candidate = proposed_candidate;
            best_eval = candidate_eval;
            best_hash = candidate_hash;
            best_bytes = candidate_bytes;
            best_key = cache_key;
            stagnation_counter = 0;
        } else {
            stagnation_counter = stagnation_counter.saturating_add(1);
        }

        if let Some(reset_after) = request.execution.stagnation_reset_evals
            && stagnation_counter >= reset_after
        {
            current_candidate = best_candidate.clone();
            current_eval = best_eval.clone();
            stagnation_counter = 0;
        }
    }

    let status = if fatal_evaluator_failure.is_some() {
        "terminated_unrecoverable_evaluator_failure"
    } else {
        "completed_annealed"
    };
    let warning = fatal_evaluator_failure
        .as_ref()
        .map(|_| "terminated due to unrecoverable evaluator failure".to_string());

    Ok(SearchSummary {
        status,
        warning,
        fatal_evaluator_failure: fatal_evaluator_failure.clone(),
        fatal_evaluator_failures: usize::from(fatal_evaluator_failure.is_some()),
        best_candidate,
        best_candidate_crc32: best_hash,
        best_eval,
        cache_key_digest: best_key.digest_crc32(),
        cache_hits,
        cache_misses,
        candidate_evaluations_executed,
        non_warmup_candidate_results_seen: evaluations_seen,
        post_baseline_candidate_results_seen: evaluations_seen.saturating_sub(1),
        proposals_attempted,
        proposals_invalid,
        self_loop_proposals,
        invalid_reason_counts,
        successful_non_deployable,
        candidate_result_counts,
        final_best_move_reward,
        realized_trace_counts_by_round: None,
        trace_refresh_merges_by_round: None,
        controller_report: serde_json::json!({
            "kind": "annealed_hill_climbing",
            "runtime_path": annealer_runtime_path_name(request.execution.annealer_kernel_profile),
            "max_mutation_radius": max_mutation_radius,
            "annealer_kernel_profile": annealer_kernel_profile_name(request.execution.annealer_kernel_profile),
            "proposal_mass_accounting": request.execution.annealer_kernel_profile == AnnealerKernelProfile::CompiledUniformMetropolisHastings,
            "proposal_action_distribution": "uniform_finite_bounded_numeric_elementary_descriptors",
        }),
    })
}

#[allow(clippy::too_many_arguments)]
fn validate_candidate_against_tune_bounds(
    candidate: &crate::api::CompressionBackend,
    bounds: &crate::spec::TuneBoundsSpec,
) -> Result<(), String> {
    let topology = collect_candidate_topology(candidate)?;
    validate_candidate_backend_family_bounds(&topology, bounds)?;
    if bounds.allow_duplicate_experts == Some(false) && candidate_has_duplicate_experts(candidate) {
        return Err(
            "candidate violates bounds.allow_duplicate_experts=false due to duplicate mixture experts"
                .to_string(),
        );
    }
    validate_candidate_parameter_ranges(candidate, bounds)?;
    Ok(())
}

fn reject_candidate_local_external_artifacts(
    candidate: &crate::api::CompressionBackend,
) -> Result<(), CandidateInvalidDiagnostic> {
    if candidate_contains_external_artifact(candidate) {
        Err(CandidateInvalidDiagnostic {
            reason: TuneInvalidReason::CandidateExternalAssetForbidden,
            diagnostic: format!(
                "{}: candidate-local external filesystem/model path references are not allowed in tune candidates",
                TuneInvalidReason::CandidateExternalAssetForbidden.as_str()
            ),
        })
    } else {
        Ok(())
    }
}

fn collect_candidate_topology(
    candidate: &crate::api::CompressionBackend,
) -> Result<CandidateTopologyStats, String> {
    let mut stats = CandidateTopologyStats {
        backend_families: BTreeSet::new(),
        max_mixture_nesting_depth: 0,
        mixture_node_expert_counts: Vec::new(),
    };
    match candidate {
        crate::api::CompressionBackend::Zpaq { .. } => {
            stats.backend_families.insert("zpaq".to_string());
        }
        #[cfg(feature = "backend-rwkv")]
        crate::api::CompressionBackend::Rwkv7 { .. } => {
            stats.backend_families.insert("rwkv7".to_string());
        }
        crate::api::CompressionBackend::Rate { rate_backend, .. } => {
            collect_rate_backend_topology(rate_backend, 0, &mut stats)?
        }
    }
    Ok(stats)
}

fn collect_rate_backend_topology(
    backend: &crate::api::RateBackend,
    mixture_depth: usize,
    stats: &mut CandidateTopologyStats,
) -> Result<(), String> {
    let canonical_name = backend
        .descriptor()
        .map_err(|err| format!("failed to resolve backend descriptor: {err}"))?
        .canonical
        .to_string();
    stats.backend_families.insert(canonical_name);

    match backend {
        crate::api::RateBackend::Mixture { spec } => {
            let depth = mixture_depth + 1;
            if depth > stats.max_mixture_nesting_depth {
                stats.max_mixture_nesting_depth = depth;
            }
            stats.mixture_node_expert_counts.push(spec.experts.len());
            for expert in &spec.experts {
                collect_rate_backend_topology(&expert.backend, depth, stats)?;
            }
        }
        crate::api::RateBackend::Calibrated { spec } => {
            collect_rate_backend_topology(&spec.base, mixture_depth, stats)?;
        }
        _ => {}
    }
    Ok(())
}

fn validate_candidate_backend_family_bounds(
    topology: &CandidateTopologyStats,
    bounds: &crate::spec::TuneBoundsSpec,
) -> Result<(), String> {
    if topology.max_mixture_nesting_depth > bounds.max_mixture_nesting_depth {
        return Err(format!(
            "candidate mixture nesting depth {} exceeds bounds.max_mixture_nesting_depth {}",
            topology.max_mixture_nesting_depth, bounds.max_mixture_nesting_depth
        ));
    }
    if topology
        .mixture_node_expert_counts
        .iter()
        .any(|count| *count > bounds.max_experts)
    {
        return Err(format!(
            "candidate mixture expert count exceeds bounds.max_experts {}",
            bounds.max_experts
        ));
    }
    if let Some(min_experts) = bounds.min_experts
        && topology
            .mixture_node_expert_counts
            .iter()
            .any(|count| *count < min_experts)
    {
        return Err(format!(
            "candidate mixture expert count is below bounds.min_experts {}",
            min_experts
        ));
    }

    if !bounds.allowed_backends.is_empty()
        && topology.backend_families.iter().any(|name| {
            !bounds
                .allowed_backends
                .iter()
                .any(|allowed| allowed == name)
        })
    {
        return Err(
            "candidate contains backend family not listed in bounds.allowed_backends".into(),
        );
    }
    if topology.backend_families.iter().any(|name| {
        bounds
            .forbidden_backends
            .iter()
            .any(|blocked| blocked == name)
    }) {
        return Err("candidate contains backend family listed in bounds.forbidden_backends".into());
    }
    if bounds
        .required_experts
        .iter()
        .any(|required| !topology.backend_families.contains(required))
    {
        return Err(
            "candidate is missing at least one backend listed in bounds.required_experts".into(),
        );
    }
    if bounds.forbidden_expert_pairs.iter().any(|(left, right)| {
        topology.backend_families.contains(left) && topology.backend_families.contains(right)
    }) {
        return Err("candidate violates bounds.forbidden_expert_pairs".into());
    }
    Ok(())
}

fn validate_candidate_parameter_ranges(
    candidate: &crate::api::CompressionBackend,
    bounds: &crate::spec::TuneBoundsSpec,
) -> Result<(), String> {
    if bounds.parameter_ranges.is_empty() {
        return Ok(());
    }
    let json = crate::spec::compression_backend_to_json_value(candidate)
        .map_err(|err| format!("failed to materialize candidate JSON for range checks: {err}"))?;
    let mut values = BTreeMap::<String, f64>::new();
    collect_numeric_parameter_paths("", &json, &mut values);
    for range in &bounds.parameter_ranges {
        let Some(value) = values.get(&range.parameter) else {
            return Err(format!(
                "candidate is missing bounded parameter path '{}'",
                range.parameter
            ));
        };
        if *value < range.min || *value > range.max {
            return Err(format!(
                "candidate parameter '{}' = {} violates bounds [{}, {}]",
                range.parameter, value, range.min, range.max
            ));
        }
    }
    Ok(())
}

fn collect_numeric_parameter_paths(
    prefix: &str,
    value: &Value,
    output: &mut BTreeMap<String, f64>,
) {
    match value {
        Value::Object(map) => {
            for (key, child) in map {
                let next = if prefix.is_empty() {
                    key.clone()
                } else {
                    format!("{prefix}.{key}")
                };
                collect_numeric_parameter_paths(&next, child, output);
            }
        }
        Value::Array(items) => {
            for (idx, child) in items.iter().enumerate() {
                let next = format!("{prefix}[{idx}]");
                collect_numeric_parameter_paths(&next, child, output);
            }
        }
        Value::Number(number) => {
            if let Some(value) = number.as_f64() {
                output.insert(prefix.to_string(), value);
            }
        }
        _ => {}
    }
}

fn candidate_contains_external_artifact(candidate: &crate::api::CompressionBackend) -> bool {
    match candidate {
        crate::api::CompressionBackend::Zpaq { method, .. } => {
            zpaq_method_contains_external_file(method)
        }
        #[cfg(feature = "backend-rwkv")]
        crate::api::CompressionBackend::Rwkv7 { method, .. } => {
            rwkv_method_contains_external_artifact(method)
        }
        crate::api::CompressionBackend::Rate { rate_backend, .. } => {
            rate_backend_contains_external_artifact(rate_backend)
        }
    }
}

fn candidate_has_duplicate_experts(candidate: &crate::api::CompressionBackend) -> bool {
    match candidate {
        crate::api::CompressionBackend::Rate { rate_backend, .. } => {
            rate_backend_has_duplicate_experts(rate_backend)
        }
        _ => false,
    }
}

fn rate_backend_has_duplicate_experts(backend: &crate::api::RateBackend) -> bool {
    match backend {
        crate::api::RateBackend::Mixture { spec } => {
            let mut seen = BTreeSet::<String>::new();
            for expert in &spec.experts {
                let key = expert
                    .backend
                    .to_canonical_json()
                    .unwrap_or_else(|_| "<invalid-backend-canonical>".to_string());
                if !seen.insert(key) {
                    return true;
                }
            }
            spec.experts
                .iter()
                .any(|expert| rate_backend_has_duplicate_experts(&expert.backend))
        }
        crate::api::RateBackend::Calibrated { spec } => {
            rate_backend_has_duplicate_experts(&spec.base)
        }
        _ => false,
    }
}

fn rate_backend_contains_external_artifact(backend: &crate::api::RateBackend) -> bool {
    match backend {
        #[cfg(feature = "backend-rwkv")]
        crate::api::RateBackend::Rwkv7Method { method } => {
            rwkv_method_contains_external_artifact(method)
        }
        #[cfg(feature = "backend-mamba")]
        crate::api::RateBackend::MambaMethod { method } => {
            mamba_method_contains_external_artifact(method)
        }
        crate::api::RateBackend::Mixture { spec } => spec
            .experts
            .iter()
            .any(|expert| rate_backend_contains_external_artifact(&expert.backend)),
        crate::api::RateBackend::Calibrated { spec } => {
            rate_backend_contains_external_artifact(&spec.base)
        }
        _ => false,
    }
}

#[cfg(feature = "backend-rwkv")]
fn rwkv_method_contains_external_artifact(method: &crate::rwkvzip::MethodSpec) -> bool {
    match method {
        crate::rwkvzip::MethodSpec::File { .. } => true,
        crate::rwkvzip::MethodSpec::Online { policy, .. } => policy
            .as_ref()
            .and_then(|policy| policy.load_from.as_ref())
            .is_some(),
    }
}

#[cfg(feature = "backend-mamba")]
fn mamba_method_contains_external_artifact(method: &crate::mambazip::MethodSpec) -> bool {
    match method {
        crate::mambazip::MethodSpec::File { .. } => true,
        crate::mambazip::MethodSpec::Online { policy, .. } => policy
            .as_ref()
            .and_then(|policy| policy.load_from.as_ref())
            .is_some(),
    }
}

fn zpaq_method_contains_external_file(method: &crate::api::ZpaqMethodSpec) -> bool {
    match method {
        crate::api::ZpaqMethodSpec::Literal { value } => {
            let trimmed = value.trim_start();
            trimmed.starts_with("file:") || trimmed.contains("://")
        }
    }
}

fn bounds_hash(bounds: &crate::spec::TuneBoundsSpec) -> Result<String, String> {
    let value = serde_json::json!({
        "allowed_backends": bounds.allowed_backends,
        "forbidden_backends": bounds.forbidden_backends,
        "parameter_ranges": bounds.parameter_ranges.iter().map(|range| serde_json::json!({
            "parameter": range.parameter,
            "min_bits": range.min.to_bits(),
            "max_bits": range.max.to_bits(),
        })).collect::<Vec<_>>(),
        "max_experts": bounds.max_experts,
        "max_mixture_nesting_depth": bounds.max_mixture_nesting_depth,
        "min_experts": bounds.min_experts,
        "allow_duplicate_experts": bounds.allow_duplicate_experts,
        "required_experts": bounds.required_experts,
        "forbidden_expert_pairs": bounds.forbidden_expert_pairs,
    });
    serde_json::to_vec(&value)
        .map(|bytes| crc32_hex(&bytes))
        .map_err(|err| format!("failed to serialize bounds for hash: {err}"))
}

fn crc32_hex(bytes: &[u8]) -> String {
    let mut hasher = Hasher::new();
    hasher.update(bytes);
    format!("{:08x}", hasher.finalize())
}

fn peak_rss_bytes() -> u64 {
    #[cfg(target_os = "linux")]
    {
        if let Ok(status) = fs::read_to_string("/proc/self/status") {
            for line in status.lines() {
                if let Some(raw) = line.strip_prefix("VmHWM:") {
                    let kb = raw
                        .split_whitespace()
                        .next()
                        .and_then(|token| token.parse::<u64>().ok());
                    if let Some(kb) = kb {
                        return kb.saturating_mul(1024);
                    }
                }
            }
        }
    }
    0
}

fn peak_memory_bytes(mode: PeakMemoryMode) -> u64 {
    let process = peak_rss_bytes();
    let cgroup = cgroup_peak_memory_bytes();
    match mode {
        PeakMemoryMode::ProcessRssPeak => process,
        PeakMemoryMode::BackendReported => cgroup.unwrap_or(process),
        PeakMemoryMode::HybridStrictMax => cgroup.unwrap_or(process).max(process),
    }
}

#[cfg(target_os = "linux")]
fn cgroup_peak_memory_bytes() -> Option<u64> {
    cgroup_v2_peak_memory_bytes().or_else(cgroup_v1_peak_memory_bytes)
}

#[cfg(not(target_os = "linux"))]
fn cgroup_peak_memory_bytes() -> Option<u64> {
    None
}

#[cfg(target_os = "linux")]
fn cgroup_v2_peak_memory_bytes() -> Option<u64> {
    let relative = current_cgroup_path("0::")?;
    let path = Path::new("/sys/fs/cgroup")
        .join(relative.trim_start_matches('/'))
        .join("memory.peak");
    read_u64_from_file(&path).ok()
}

#[cfg(target_os = "linux")]
fn cgroup_v1_peak_memory_bytes() -> Option<u64> {
    let relative = current_cgroup_memory_path()?;
    let path = Path::new("/sys/fs/cgroup/memory")
        .join(relative.trim_start_matches('/'))
        .join("memory.max_usage_in_bytes");
    read_u64_from_file(&path).ok()
}

#[cfg(target_os = "linux")]
fn current_cgroup_path(prefix: &str) -> Option<String> {
    let content = fs::read_to_string("/proc/self/cgroup").ok()?;
    content
        .lines()
        .find_map(|line| line.strip_prefix(prefix).map(ToOwned::to_owned))
}

#[cfg(target_os = "linux")]
fn current_cgroup_memory_path() -> Option<String> {
    let content = fs::read_to_string("/proc/self/cgroup").ok()?;
    content.lines().find_map(|line| {
        let mut parts = line.splitn(3, ':');
        let _hierarchy = parts.next()?;
        let controllers = parts.next()?;
        let path = parts.next()?;
        controllers
            .split(',')
            .any(|controller| controller == "memory")
            .then(|| path.to_string())
    })
}

#[cfg(target_os = "linux")]
fn read_u64_from_file(path: &Path) -> Result<u64, String> {
    let raw = fs::read_to_string(path)
        .map_err(|err| format!("failed to read '{}': {err}", path.display()))?;
    let trimmed = raw.trim();
    if trimmed == "max" {
        return Err(format!(
            "'{}' contains unbounded sentinel 'max'",
            path.display()
        ));
    }
    trimmed
        .parse::<u64>()
        .map_err(|err| format!("failed to parse '{}': {err}", path.display()))
}

#[cfg(target_os = "linux")]
fn peak_rss_bytes_for_pid(pid: u32) -> Option<u64> {
    let path = format!("/proc/{pid}/status");
    let status = fs::read_to_string(path).ok()?;
    for line in status.lines() {
        if let Some(raw) = line.strip_prefix("VmHWM:") {
            let kb = raw
                .split_whitespace()
                .next()
                .and_then(|token| token.parse::<u64>().ok())?;
            return Some(kb.saturating_mul(1024));
        }
    }
    None
}

#[cfg(all(unix, not(target_os = "linux")))]
fn peak_memory_bytes_for_pid(_pid: u32, _mode: PeakMemoryMode) -> Option<u64> {
    None
}

#[cfg(test)]
mod tests;
