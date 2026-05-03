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
};
use crate::api::RateBackend;
use crate::runtime::CompressionRuntime;
use crate::spec::{
    AiqiDiscountedControllerSpec, AssetRef, BuiltinEnvironmentSpec, CanonicalJson,
    CompiledPlannerRunSpec, ControllerSpec, EnvironmentSpec, McAixiControllerSpec,
    PlannerInterfaceSpec, PlannerRunSpec, PlannerRuntimeSpec, SpecDocument, SpecEnvironment,
    WarmStartExactJhControllerSpec, load_spec_document,
};
use crc32fast::Hasher;
use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::fs;
use std::io::{Read, Write};
#[cfg(unix)]
use std::os::fd::{FromRawFd, RawFd};
use std::path::Path;
use std::time::{Duration, Instant};

/// Executor-side tuning request assembled from CLI inputs.
#[derive(Clone, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub struct TuneCommandRequest {
    /// Canonical tune document path (`.json` or `.itsd`).
    pub spec_path: String,
    /// Non-canonical execution controls and theorem claim inputs.
    pub execution: TuneExecutionConfig,
}

/// Executor-side tuning controls (non-canonical).
#[derive(Clone, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub struct TuneExecutionConfig {
    pub max_evaluations: Option<usize>,
    pub annealer_kernel_profile: AnnealerKernelProfile,
    pub cpu_affinity: Option<String>,
    pub threads: Option<usize>,
    pub warmup_baseline_runs: usize,
    pub self_improvement_rounds: usize,
    pub stagnation_reset_evals: Option<usize>,
    pub log_path: Option<String>,
    pub diagnostic_chunk_bytes: Option<usize>,
    pub rss_mode: PeakMemoryMode,
    pub planner_deployable_model: bool,
    pub warmstart_trace_refresh: bool,
    pub theorem: TuneTheoremConfig,
}

/// Theorem-facing runtime claims and certification references (non-canonical).
#[derive(Clone, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub struct TuneTheoremConfig {
    pub claim_exact_finite_mdp: bool,
    pub claim_exact_observed_markov: bool,
    pub claim_planner_convergence: bool,
    pub timing_certification_tier: TimingCertificationTier,
    pub determinism_deadline_certificate: Option<String>,
    pub observation_adapter_spec_ref: Option<String>,
    pub exact_state_encoder_spec_ref: Option<String>,
    pub scalar_representation_ref: Option<String>,
    pub finite_planner_state_certificate: Option<String>,
    pub no_hidden_state_certificate: Option<String>,
    pub exact_reward_encoding_certificate: Option<String>,
    pub exact_state_observation_certificate: Option<String>,
    pub deterministic_evaluator_table: Option<String>,
}

/// Executor-selected annealer kernel profile.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum AnnealerKernelProfile {
    ReversibleElementaryMetropolis,
    CompiledUniformMetropolisHastings,
}

/// Peak-memory accounting mode.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum PeakMemoryMode {
    ProcessRssPeak,
    BackendReported,
    HybridStrictMax,
}

/// Timing certification tier declaration for theorem-facing claims.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum TimingCertificationTier {
    BestEffort,
    Isolated,
    RealTime,
    DeterministicTable,
}

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
    successful_non_deployable: usize,
    final_best_move_reward: f64,
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

    fn min_reward(&self) -> Reward {
        0
    }

    fn max_reward(&self) -> Reward {
        match self {
            Self::ExactIntegerObjectiveDifference { max_reward, .. }
            | Self::NormalizedClipped { max_reward, .. } => *max_reward,
        }
    }

    fn reward_offset(&self) -> Reward {
        0
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

impl Default for TuneExecutionConfig {
    fn default() -> Self {
        Self {
            max_evaluations: None,
            annealer_kernel_profile: AnnealerKernelProfile::ReversibleElementaryMetropolis,
            cpu_affinity: None,
            threads: None,
            warmup_baseline_runs: 0,
            self_improvement_rounds: 1,
            stagnation_reset_evals: None,
            log_path: None,
            diagnostic_chunk_bytes: None,
            rss_mode: PeakMemoryMode::ProcessRssPeak,
            planner_deployable_model: false,
            warmstart_trace_refresh: false,
            theorem: TuneTheoremConfig::default(),
        }
    }
}

impl Default for TuneTheoremConfig {
    fn default() -> Self {
        Self {
            claim_exact_finite_mdp: false,
            claim_exact_observed_markov: false,
            claim_planner_convergence: false,
            timing_certification_tier: TimingCertificationTier::BestEffort,
            determinism_deadline_certificate: None,
            observation_adapter_spec_ref: None,
            exact_state_encoder_spec_ref: None,
            scalar_representation_ref: None,
            finite_planner_state_certificate: None,
            no_hidden_state_certificate: None,
            exact_reward_encoding_certificate: None,
            exact_state_observation_certificate: None,
            deterministic_evaluator_table: None,
        }
    }
}

const EXECUTION_CONFIG_FIELDS: &[&str] = &[
    "max_evaluations",
    "annealer_kernel_profile",
    "cpu_affinity",
    "threads",
    "warmup_baseline_runs",
    "self_improvement_rounds",
    "stagnation_reset_evals",
    "log_path",
    "diagnostic_chunk_bytes",
    "rss_mode",
    "planner_deployable_model",
    "warmstart_trace_refresh",
    "theorem",
    // Theorem keys are accepted at the top level for sidecar/CLI parity.
    "claim_exact_finite_mdp",
    "claim_exact_observed_markov",
    "claim_planner_convergence",
    "timing_certification_tier",
    "determinism_deadline_certificate",
    "observation_adapter_spec_ref",
    "exact_state_encoder_spec_ref",
    "scalar_representation_ref",
    "finite_planner_state_certificate",
    "no_hidden_state_certificate",
    "exact_reward_encoding_certificate",
    "exact_state_observation_certificate",
    "deterministic_evaluator_table",
];

const THEOREM_CONFIG_FIELDS: &[&str] = &[
    "claim_exact_finite_mdp",
    "claim_exact_observed_markov",
    "claim_planner_convergence",
    "timing_certification_tier",
    "determinism_deadline_certificate",
    "observation_adapter_spec_ref",
    "exact_state_encoder_spec_ref",
    "scalar_representation_ref",
    "finite_planner_state_certificate",
    "no_hidden_state_certificate",
    "exact_reward_encoding_certificate",
    "exact_state_observation_certificate",
    "deterministic_evaluator_table",
];

const EXECUTION_AND_THEOREM_CONFIG_FIELDS: &[&str] = EXECUTION_CONFIG_FIELDS;

fn ensure_known_execution_config_fields(
    object: &serde_json::Map<String, Value>,
    allowed: &[&str],
) -> Result<(), String> {
    for key in object.keys() {
        if !allowed.contains(&key.as_str()) {
            return Err(format!("unknown execution config field '{key}'"));
        }
    }
    Ok(())
}

impl TuneExecutionConfig {
    /// Load executor controls from JSON. This is intentionally distinct from
    /// canonical `SpecDocument::Tune`.
    pub fn from_json_value(value: &Value) -> Result<Self, String> {
        let object = value
            .as_object()
            .ok_or_else(|| "execution config must be a JSON object".to_string())?;
        ensure_known_execution_config_fields(object, EXECUTION_CONFIG_FIELDS)?;
        let mut cfg = Self::default();
        apply_optional_usize(object.get("max_evaluations"), &mut cfg.max_evaluations)?;
        if let Some(raw) = object.get("annealer_kernel_profile") {
            cfg.annealer_kernel_profile =
                parse_annealer_kernel_profile(required_str(raw, "annealer_kernel_profile")?)?;
        }
        cfg.cpu_affinity = clean_optional_string(object.get("cpu_affinity"));
        apply_optional_usize(object.get("threads"), &mut cfg.threads)?;
        if let Some(raw) = object.get("warmup_baseline_runs") {
            cfg.warmup_baseline_runs = required_usize(raw, "warmup_baseline_runs")?;
        }
        if let Some(raw) = object.get("self_improvement_rounds") {
            cfg.self_improvement_rounds = required_usize(raw, "self_improvement_rounds")?;
        }
        apply_optional_usize(
            object.get("stagnation_reset_evals"),
            &mut cfg.stagnation_reset_evals,
        )?;
        cfg.log_path = clean_optional_string(object.get("log_path"));
        apply_optional_usize(
            object.get("diagnostic_chunk_bytes"),
            &mut cfg.diagnostic_chunk_bytes,
        )?;
        if let Some(raw) = object.get("rss_mode") {
            cfg.rss_mode = parse_peak_memory_mode(required_str(raw, "rss_mode")?)?;
        }
        if let Some(raw) = object.get("planner_deployable_model") {
            cfg.planner_deployable_model = required_bool(raw, "planner_deployable_model")?;
        }
        if let Some(raw) = object.get("warmstart_trace_refresh") {
            cfg.warmstart_trace_refresh = required_bool(raw, "warmstart_trace_refresh")?;
        }

        if let Some(theorem_obj) = object.get("theorem") {
            let theorem_map = theorem_obj
                .as_object()
                .ok_or_else(|| "execution config field 'theorem' must be an object".to_string())?;
            ensure_known_execution_config_fields(theorem_map, THEOREM_CONFIG_FIELDS)?;
            cfg.apply_theorem_object(theorem_map)?;
        }
        // Accept theorem keys at the top-level for CLI/sidecar ergonomics.
        cfg.apply_theorem_object(object)?;
        cfg.validate()?;
        Ok(cfg)
    }

    /// Load executor controls from a JSON file path.
    pub fn from_json_path(path: &str) -> Result<Self, String> {
        let raw = fs::read(path)
            .map_err(|err| format!("failed to read execution config '{}': {err}", path))?;
        let value: Value = serde_json::from_slice(&raw)
            .map_err(|err| format!("invalid execution config JSON '{}': {err}", path))?;
        Self::from_json_value(&value)
    }

    fn apply_theorem_object(
        &mut self,
        object: &serde_json::Map<String, Value>,
    ) -> Result<(), String> {
        ensure_known_execution_config_fields(object, EXECUTION_AND_THEOREM_CONFIG_FIELDS)?;
        if let Some(raw) = object.get("claim_exact_finite_mdp") {
            self.theorem.claim_exact_finite_mdp = required_bool(raw, "claim_exact_finite_mdp")?;
        }
        if let Some(raw) = object.get("claim_exact_observed_markov") {
            self.theorem.claim_exact_observed_markov =
                required_bool(raw, "claim_exact_observed_markov")?;
        }
        if let Some(raw) = object.get("claim_planner_convergence") {
            self.theorem.claim_planner_convergence =
                required_bool(raw, "claim_planner_convergence")?;
        }
        if let Some(raw) = object.get("timing_certification_tier") {
            self.theorem.timing_certification_tier =
                parse_timing_tier(required_str(raw, "timing_certification_tier")?)?;
        }
        if let Some(raw) = object.get("determinism_deadline_certificate") {
            self.theorem.determinism_deadline_certificate =
                parse_optional_non_empty_string(Some(raw), "determinism_deadline_certificate")?;
        }
        if let Some(raw) = object.get("observation_adapter_spec_ref") {
            self.theorem.observation_adapter_spec_ref =
                parse_optional_non_empty_string(Some(raw), "observation_adapter_spec_ref")?;
        }
        if let Some(raw) = object.get("exact_state_encoder_spec_ref") {
            self.theorem.exact_state_encoder_spec_ref =
                parse_optional_non_empty_string(Some(raw), "exact_state_encoder_spec_ref")?;
        }
        if let Some(raw) = object.get("scalar_representation_ref") {
            self.theorem.scalar_representation_ref =
                parse_optional_non_empty_string(Some(raw), "scalar_representation_ref")?;
        }
        if let Some(raw) = object.get("finite_planner_state_certificate") {
            self.theorem.finite_planner_state_certificate =
                parse_optional_non_empty_string(Some(raw), "finite_planner_state_certificate")?;
        }
        if let Some(raw) = object.get("no_hidden_state_certificate") {
            self.theorem.no_hidden_state_certificate =
                parse_optional_non_empty_string(Some(raw), "no_hidden_state_certificate")?;
        }
        if let Some(raw) = object.get("exact_reward_encoding_certificate") {
            self.theorem.exact_reward_encoding_certificate =
                parse_optional_non_empty_string(Some(raw), "exact_reward_encoding_certificate")?;
        }
        if let Some(raw) = object.get("exact_state_observation_certificate") {
            self.theorem.exact_state_observation_certificate =
                parse_optional_non_empty_string(Some(raw), "exact_state_observation_certificate")?;
        }
        if let Some(raw) = object.get("deterministic_evaluator_table") {
            self.theorem.deterministic_evaluator_table =
                parse_optional_non_empty_string(Some(raw), "deterministic_evaluator_table")?;
        }
        Ok(())
    }

    fn validate(&self) -> Result<(), String> {
        if let Some(threads) = self.threads
            && threads == 0
        {
            return Err("threads must be >= 1 when set".to_string());
        }
        if let Some(max_evaluations) = self.max_evaluations
            && max_evaluations == 0
        {
            return Err("max_evaluations must be >= 1 when set; the normative baseline evaluation counts as the first non-warmup admitted candidate result".to_string());
        }
        if self.self_improvement_rounds == 0 {
            return Err("self_improvement_rounds must be >= 1".to_string());
        }
        if let Some(bytes) = self.diagnostic_chunk_bytes
            && bytes == 0
        {
            return Err("diagnostic_chunk_bytes must be >= 1 when set".to_string());
        }
        Ok(())
    }

    /// Serialize this executor profile for report/provenance output.
    pub fn to_json_value(&self) -> Value {
        serde_json::json!({
            "max_evaluations": self.max_evaluations,
            "annealer_kernel_profile": annealer_kernel_profile_name(self.annealer_kernel_profile),
            "cpu_affinity": self.cpu_affinity,
            "threads": self.threads,
            "warmup_baseline_runs": self.warmup_baseline_runs,
            "self_improvement_rounds": self.self_improvement_rounds,
            "stagnation_reset_evals": self.stagnation_reset_evals,
            "log_path": self.log_path,
            "diagnostic_chunk_bytes": self.diagnostic_chunk_bytes,
            "rss_mode": peak_memory_mode_name(self.rss_mode),
            "planner_deployable_model": self.planner_deployable_model,
            "warmstart_trace_refresh": self.warmstart_trace_refresh,
            "theorem": {
                "claim_exact_finite_mdp": self.theorem.claim_exact_finite_mdp,
                "claim_exact_observed_markov": self.theorem.claim_exact_observed_markov,
                "claim_planner_convergence": self.theorem.claim_planner_convergence,
                "timing_certification_tier": timing_tier_name(self.theorem.timing_certification_tier),
                "determinism_deadline_certificate": self.theorem.determinism_deadline_certificate,
                "observation_adapter_spec_ref": self.theorem.observation_adapter_spec_ref,
                "exact_state_encoder_spec_ref": self.theorem.exact_state_encoder_spec_ref,
                "scalar_representation_ref": self.theorem.scalar_representation_ref,
                "finite_planner_state_certificate": self.theorem.finite_planner_state_certificate,
                "no_hidden_state_certificate": self.theorem.no_hidden_state_certificate,
                "exact_reward_encoding_certificate": self.theorem.exact_reward_encoding_certificate,
                "exact_state_observation_certificate": self.theorem.exact_state_observation_certificate,
                "deterministic_evaluator_table": self.theorem.deterministic_evaluator_table,
            }
        })
    }
}

/// Parse `infotheory tune ...` command arguments into canonical spec path and
/// executor config.
pub fn parse_tune_command_args(args: &[String]) -> Result<TuneCommandRequest, String> {
    if args.len() < 3 {
        return Err("Error: 'tune' requires <spec.json|spec.itsd>".to_string());
    }
    let mut request = TuneCommandRequest {
        spec_path: args[2].clone(),
        execution: TuneExecutionConfig::default(),
    };
    let mut exec_config_path = None::<String>;
    let mut i = 3usize;
    while i < args.len() {
        if args[i].as_str() == "--exec-config" {
            i += 1;
            exec_config_path = Some(
                args.get(i)
                    .ok_or_else(|| "Error: --exec-config requires a JSON path".to_string())?
                    .clone(),
            );
        }
        i += 1;
    }
    if let Some(path) = exec_config_path {
        request.execution = TuneExecutionConfig::from_json_path(&path)?;
    }
    i = 3usize;
    while i < args.len() {
        match args[i].as_str() {
            "--exec-config" => {
                i += 1;
                let _ = args
                    .get(i)
                    .ok_or_else(|| "Error: --exec-config requires a JSON path".to_string())?;
            }
            "--max-evaluations" => {
                i += 1;
                request.execution.max_evaluations =
                    Some(parse_cli_usize(args.get(i), "--max-evaluations")?);
            }
            "--annealer-kernel-profile" => {
                i += 1;
                request.execution.annealer_kernel_profile = parse_annealer_kernel_profile(
                    parse_cli_str(args.get(i), "--annealer-kernel-profile")?,
                )?;
            }
            "--cpu-affinity" => {
                i += 1;
                request.execution.cpu_affinity =
                    Some(parse_cli_str(args.get(i), "--cpu-affinity")?.to_string());
            }
            "--threads" => {
                i += 1;
                request.execution.threads = Some(parse_cli_usize(args.get(i), "--threads")?);
            }
            "--warmup-baseline-runs" => {
                i += 1;
                request.execution.warmup_baseline_runs =
                    parse_cli_usize(args.get(i), "--warmup-baseline-runs")?;
            }
            "--self-improvement-rounds" => {
                i += 1;
                request.execution.self_improvement_rounds =
                    parse_cli_usize(args.get(i), "--self-improvement-rounds")?;
            }
            "--stagnation-reset-evals" => {
                i += 1;
                request.execution.stagnation_reset_evals =
                    Some(parse_cli_usize(args.get(i), "--stagnation-reset-evals")?);
            }
            "--log-path" => {
                i += 1;
                request.execution.log_path =
                    Some(parse_cli_str(args.get(i), "--log-path")?.to_string());
            }
            "--diagnostic-chunk-bytes" => {
                i += 1;
                request.execution.diagnostic_chunk_bytes =
                    Some(parse_cli_usize(args.get(i), "--diagnostic-chunk-bytes")?);
            }
            "--rss-mode" => {
                i += 1;
                request.execution.rss_mode =
                    parse_peak_memory_mode(parse_cli_str(args.get(i), "--rss-mode")?)?;
            }
            "--planner-deployable-model" => {
                request.execution.planner_deployable_model = true;
            }
            "--warmstart-trace-refresh" => {
                request.execution.warmstart_trace_refresh = true;
            }
            "--timing-tier" => {
                i += 1;
                request.execution.theorem.timing_certification_tier =
                    parse_timing_tier(parse_cli_str(args.get(i), "--timing-tier")?)?;
            }
            "--determinism-deadline-certificate" => {
                i += 1;
                request.execution.theorem.determinism_deadline_certificate = Some(
                    parse_cli_non_empty_str(args.get(i), "--determinism-deadline-certificate")?
                        .to_string(),
                );
            }
            "--deterministic-evaluator-table" => {
                i += 1;
                request.execution.theorem.deterministic_evaluator_table = Some(
                    parse_cli_non_empty_str(args.get(i), "--deterministic-evaluator-table")?
                        .to_string(),
                );
            }
            "--finite-planner-state-certificate" => {
                i += 1;
                request.execution.theorem.finite_planner_state_certificate = Some(
                    parse_cli_non_empty_str(args.get(i), "--finite-planner-state-certificate")?
                        .to_string(),
                );
            }
            "--no-hidden-state-certificate" => {
                i += 1;
                request.execution.theorem.no_hidden_state_certificate = Some(
                    parse_cli_non_empty_str(args.get(i), "--no-hidden-state-certificate")?
                        .to_string(),
                );
            }
            "--exact-reward-encoding-certificate" => {
                i += 1;
                request.execution.theorem.exact_reward_encoding_certificate = Some(
                    parse_cli_non_empty_str(args.get(i), "--exact-reward-encoding-certificate")?
                        .to_string(),
                );
            }
            "--exact-state-observation-certificate" => {
                i += 1;
                request
                    .execution
                    .theorem
                    .exact_state_observation_certificate = Some(
                    parse_cli_non_empty_str(args.get(i), "--exact-state-observation-certificate")?
                        .to_string(),
                );
            }
            "--observation-adapter-spec-ref" => {
                i += 1;
                request.execution.theorem.observation_adapter_spec_ref = Some(
                    parse_cli_non_empty_str(args.get(i), "--observation-adapter-spec-ref")?
                        .to_string(),
                );
            }
            "--exact-state-encoder-spec-ref" => {
                i += 1;
                request.execution.theorem.exact_state_encoder_spec_ref = Some(
                    parse_cli_non_empty_str(args.get(i), "--exact-state-encoder-spec-ref")?
                        .to_string(),
                );
            }
            "--scalar-representation-ref" => {
                i += 1;
                request.execution.theorem.scalar_representation_ref = Some(
                    parse_cli_non_empty_str(args.get(i), "--scalar-representation-ref")?
                        .to_string(),
                );
            }
            "--claim-exact-finite-mdp" => {
                request.execution.theorem.claim_exact_finite_mdp = true;
            }
            "--claim-exact-observed-markov" => {
                request.execution.theorem.claim_exact_observed_markov = true;
            }
            "--claim-planner-convergence" => {
                request.execution.theorem.claim_planner_convergence = true;
            }
            other => {
                return Err(format!("Error: unknown tune option '{other}'"));
            }
        }
        i += 1;
    }
    request.execution.validate()?;
    Ok(request)
}

/// Execute tuning for one canonical tune document using executor-side controls.
///
/// The runtime enforces canonical/executor separation, baseline deployability
/// preconditions, and controller-specific bounded search semantics.
pub fn run_tune(request: &TuneCommandRequest) -> Result<(), String> {
    let config_path = Path::new(&request.spec_path);
    let config_dir = config_path.parent().unwrap_or(Path::new("."));
    apply_executor_controls(&request.execution)?;
    let document = load_spec_document(&request.spec_path).map_err(|err| err.to_string())?;
    let SpecDocument::Tune(spec) = document else {
        return Err(format!(
            "tune expects a tune document, found kind '{}'",
            document.kind_str()
        ));
    };
    let compiled = spec
        .compile_in(&SpecEnvironment::new(config_dir))
        .map_err(|err| err.to_string())?;

    validate_candidate_against_tune_bounds(
        compiled.baseline_candidate().canonical_spec(),
        &compiled.canonical_spec().bounds,
    )?;
    reject_candidate_local_external_artifacts(compiled.baseline_candidate().canonical_spec())?;

    let tune_started = Instant::now();
    let dataset = load_dataset(resolve_input_asset_path(
        &compiled,
        &compiled.canonical_spec().input_asset,
    )?)?;
    let initial_eval_limit = effective_eval_limit_seconds(
        &compiled,
        tune_started,
        Some(compiled.canonical_spec().eval_time_limit_seconds),
        None,
    );
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
        rss_mode: request.execution.rss_mode,
        timing_certification_tier: request.execution.theorem.timing_certification_tier,
        build_profile: option_env!("PROFILE").unwrap_or("unknown"),
        feature_set: compiled_feature_set(),
    };
    let verified_theorem = VerifiedTheoremInputs::load(
        &request.execution.theorem,
        &compiled,
        &dataset,
        &evaluator_profile,
        config_dir,
    )?;

    for _ in 0..request.execution.warmup_baseline_runs {
        let _ = evaluate_candidate(
            compiled.baseline_candidate(),
            &dataset,
            compiled.baseline_candidate_model_bytes(),
            compiled.canonical_spec().min_throughput_bytes_per_second,
            compiled.canonical_spec().max_memory_bytes,
            initial_eval_limit,
            request.execution.rss_mode,
            verified_theorem.deterministic_table.as_ref(),
        )?;
    }

    let baseline_eval = evaluate_candidate(
        compiled.baseline_candidate(),
        &dataset,
        compiled.baseline_candidate_model_bytes(),
        compiled.canonical_spec().min_throughput_bytes_per_second,
        compiled.canonical_spec().max_memory_bytes,
        initial_eval_limit,
        request.execution.rss_mode,
        verified_theorem.deterministic_table.as_ref(),
    )?;
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
            &mut cache,
        )?
    } else {
        SearchSummary {
            status: "baseline_not_deployable",
            warning: None,
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
            successful_non_deployable: if baseline_eval.deployable { 0 } else { 1 },
            final_best_move_reward: 0.0,
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
    );
    let bounds_hash = bounds_hash(&compiled.canonical_spec().bounds)?;
    let observation_adapter_hash = observation_adapter_content_hash()?;
    let evaluator_execution_model =
        evaluator_execution_model(verified_theorem.deterministic_table.as_ref());
    let theorem_timing_basis = theorem_timing_basis(&request.execution.theorem, &verified_theorem);
    let best_model_bytes = search_summary
        .best_candidate
        .compile_in(&SpecEnvironment::new(config_dir))
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
            },
            "stagnation_policy": {
                "stagnation_reset_evals": request.execution.stagnation_reset_evals,
            },
            "executor_controls": executor_controls_report(&request.execution),
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
            "proposals_attempted": search_summary.proposals_attempted,
            "proposals_invalid": search_summary.proposals_invalid,
            "self_loop_proposals": search_summary.self_loop_proposals,
            "successful_non_deployable": search_summary.successful_non_deployable,
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

    if search_summary.best_eval.deployable {
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
    if let Some(threads) = config.threads {
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build_global()
            .map_err(|err| format!("failed to apply threads execution control: {err}"))?;
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

fn executor_controls_report(config: &TuneExecutionConfig) -> Value {
    let cgroup_peak_available = cgroup_peak_memory_bytes().is_some();
    let effective_measurement = match config.rss_mode {
        PeakMemoryMode::ProcessRssPeak => "process_rss_peak",
        PeakMemoryMode::BackendReported if cgroup_peak_available => "cgroup_peak_memory",
        PeakMemoryMode::BackendReported => "process_rss_peak_fallback",
        PeakMemoryMode::HybridStrictMax if cgroup_peak_available => {
            "max_process_rss_peak_cgroup_peak"
        }
        PeakMemoryMode::HybridStrictMax => "process_rss_peak_fallback",
    };
    serde_json::json!({
        "cpu_affinity": {
            "requested": config.cpu_affinity,
            "applied_to_current_process": config.cpu_affinity.is_some(),
        },
        "threads": {
            "requested": config.threads,
            "rayon_global_pool_configured": config.threads.is_some(),
        },
        "log_path": config.log_path,
        "diagnostic_chunk_bytes": config.diagnostic_chunk_bytes,
        "rss_mode": {
            "requested": peak_memory_mode_name(config.rss_mode),
            "effective_measurement": effective_measurement,
            "cgroup_peak_memory_available": cgroup_peak_available,
            "backend_peak_memory_report_available": cgroup_peak_available,
        },
    })
}

fn diagnostic_chunking_report(dataset: &LoadedDataset, chunk_bytes: Option<usize>) -> Value {
    let charged_payload_bytes = dataset.raw_bytes.len();
    let Some(chunk_bytes) = chunk_bytes else {
        return serde_json::json!({
            "enabled": false,
            "chunk_bytes": null,
            "charged_payload_bytes": charged_payload_bytes,
            "chunk_count": 0usize,
            "last_chunk_bytes": 0usize,
            "affects_objective": false,
            "affects_canonical_candidate_identity": false,
        });
    };
    let chunk_count = if charged_payload_bytes == 0 {
        0usize
    } else {
        (charged_payload_bytes / chunk_bytes)
            + usize::from(charged_payload_bytes % chunk_bytes != 0)
    };
    let last_chunk_bytes = if charged_payload_bytes == 0 {
        0usize
    } else {
        let remainder = charged_payload_bytes % chunk_bytes;
        if remainder == 0 {
            chunk_bytes
        } else {
            remainder
        }
    };
    serde_json::json!({
        "enabled": true,
        "chunk_bytes": chunk_bytes,
        "charged_payload_bytes": charged_payload_bytes,
        "chunk_count": chunk_count,
        "last_chunk_bytes": last_chunk_bytes,
        "affects_objective": false,
        "affects_canonical_candidate_identity": false,
    })
}

fn causal_profile_report(dataset: &LoadedDataset) -> Value {
    let Some(profile) = &dataset.causal_profile else {
        return serde_json::json!(null);
    };
    let domains = profile
        .domains
        .iter()
        .map(|(domain, support)| match support {
            CausalTargetDomain::ByteAlphabet => serde_json::json!({
                "domain": domain,
                "kind": "byte_alphabet",
                "normalization": "per_byte_symbol",
                "symbol_width_bytes": profile.byte_alphabet_symbol_width,
                "symbols": 256usize,
            }),
            CausalTargetDomain::EnumeratedPayloads { payloads } => serde_json::json!({
                "domain": domain,
                "kind": "enumerated_payloads",
                "normalization": "finite_payload_set",
                "payloads": payloads.len(),
            }),
        })
        .collect::<Vec<_>>();
    serde_json::json!({
        "channel_set": profile.channel_set.iter().collect::<Vec<_>>(),
        "domain_support_crc32": profile.domain_support_hash,
        "header_profile_crc32": profile.header_profile_hash,
        "collection_policy": profile.collection_policy,
        "action_alphabet_size": profile.action_alphabet_size,
        "percept_schema_channels": profile
            .percept_channels
            .iter()
            .map(|pair| {
                serde_json::json!({"channel": pair.channel.as_str(), "domain": pair.domain.as_str()})
            })
            .collect::<Vec<_>>(),
        "reward_encoding": serde_json::json!({
            "channel": profile.reward_channel.channel.as_str(),
            "domain": profile.reward_channel.domain.as_str(),
        }),
        "terminal_encoding": serde_json::json!({
            "channel": profile.terminal_channel.channel.as_str(),
            "domain": profile.terminal_channel.domain.as_str(),
        }),
        "event_grammar": serde_json::json!({
            "context_channels": profile.event_grammar.context_channels.iter().collect::<Vec<_>>(),
            "observe_target_no_score": profile
                .event_grammar
                .observe_target_no_score
                .iter()
                .map(|pair| {
                    serde_json::json!({"channel": pair.channel.as_str(), "domain": pair.domain.as_str()})
                })
                .collect::<Vec<_>>(),
            "target": profile
                .event_grammar
                .target
                .iter()
                .map(|pair| {
                    serde_json::json!({"channel": pair.channel.as_str(), "domain": pair.domain.as_str()})
                })
                .collect::<Vec<_>>(),
        }),
        "domains": domains,
        "structural_conditioning": "zero_cost_event_descriptor_tags_v1",
        "byte_alphabet_expansion_policy": "multi_byte_targets_expand_to_single_byte_events",
    })
}

fn evaluator_execution_model(
    deterministic_table: Option<&VerifiedDeterministicEvaluatorTable>,
) -> &'static str {
    if deterministic_table.is_some() {
        "deterministic_table"
    } else if cfg!(unix) {
        "fork_process_isolated_operational"
    } else {
        "unsupported_non_unix"
    }
}

fn theorem_timing_basis(
    theorem: &TuneTheoremConfig,
    verified: &VerifiedTheoremInputs,
) -> &'static str {
    match theorem.timing_certification_tier {
        TimingCertificationTier::DeterministicTable if verified.deterministic_table.is_some() => {
            "verified_deterministic_evaluator_table"
        }
        TimingCertificationTier::RealTime if verified.determinism_deadline.is_some() => {
            "verified_real_time_deadline_certificate"
        }
        _ => "operational_only_uncertified",
    }
}

impl VerifiedTheoremInputs {
    fn load(
        theorem: &TuneTheoremConfig,
        compiled: &crate::spec::CompiledTuneSpec,
        dataset: &LoadedDataset,
        evaluator_profile: &EvaluatorProfile,
        config_dir: &Path,
    ) -> Result<Self, String> {
        let bounds_digest = bounds_hash(&compiled.canonical_spec().bounds)?;
        let controller_kind = controller_kind_name(compiled.controller());
        let scalar_representation = theorem
            .scalar_representation_ref
            .as_deref()
            .unwrap_or(SCALAR_REPRESENTATION_DECLARATION);
        let mut verified = Self::default();

        verified.finite_planner_state = load_generic_certificate(
            theorem.finite_planner_state_certificate.as_deref(),
            "finite_planner_state",
            compiled,
            dataset,
            evaluator_profile,
            config_dir,
            &bounds_digest,
            controller_kind,
        )?;
        verified.no_hidden_state = load_generic_certificate(
            theorem.no_hidden_state_certificate.as_deref(),
            "no_hidden_state",
            compiled,
            dataset,
            evaluator_profile,
            config_dir,
            &bounds_digest,
            controller_kind,
        )?;
        verified.determinism_deadline = load_generic_certificate(
            theorem.determinism_deadline_certificate.as_deref(),
            "determinism_deadline",
            compiled,
            dataset,
            evaluator_profile,
            config_dir,
            &bounds_digest,
            controller_kind,
        )?;
        verified.exact_state_observation = load_exact_state_observation_certificate(
            theorem,
            verified.finite_planner_state.as_ref(),
            compiled,
            dataset,
            evaluator_profile,
            config_dir,
            &bounds_digest,
            controller_kind,
        )?;
        verified.exact_reward_encoding = load_exact_reward_certificate(
            theorem.exact_reward_encoding_certificate.as_deref(),
            compiled,
            dataset,
            evaluator_profile,
            config_dir,
            &bounds_digest,
            controller_kind,
            scalar_representation,
        )?;
        verified.deterministic_table = load_deterministic_evaluator_table(
            theorem.deterministic_evaluator_table.as_deref(),
            compiled,
            dataset,
            evaluator_profile,
            config_dir,
            &bounds_digest,
            controller_kind,
        )?;
        Ok(verified)
    }

    fn timing_certified(&self, theorem: &TuneTheoremConfig) -> bool {
        match theorem.timing_certification_tier {
            TimingCertificationTier::RealTime => self.determinism_deadline.is_some(),
            TimingCertificationTier::DeterministicTable => self.deterministic_table.is_some(),
            TimingCertificationTier::BestEffort | TimingCertificationTier::Isolated => false,
        }
    }

    fn to_json_value(&self) -> Value {
        serde_json::json!({
            "finite_planner_state": self.finite_planner_state.as_ref().map(VerifiedCertificate::to_json_value),
            "no_hidden_state": self.no_hidden_state.as_ref().map(VerifiedCertificate::to_json_value),
            "exact_reward_encoding": self.exact_reward_encoding.as_ref().map(VerifiedExactRewardEncodingCertificate::to_json_value),
            "exact_state_observation": self.exact_state_observation.as_ref().map(VerifiedExactStateObservationCertificate::to_json_value),
            "determinism_deadline": self.determinism_deadline.as_ref().map(VerifiedCertificate::to_json_value),
            "deterministic_evaluator_table": self.deterministic_table.as_ref().map(VerifiedDeterministicEvaluatorTable::to_json_value),
        })
    }
}

impl VerifiedCertificate {
    fn to_json_value(&self) -> Value {
        serde_json::json!({
            "ref": self.ref_value,
            "content_crc32": self.content_hash,
            "verified": true,
        })
    }
}

impl VerifiedExactRewardEncodingCertificate {
    fn to_json_value(&self) -> Value {
        serde_json::json!({
            "ref": self.base.ref_value,
            "content_crc32": self.base.content_hash,
            "verified": true,
            "scalar_representation": self.scalar_representation,
            "reward_bits": self.reward_bits,
            "max_reward": self.max_reward,
            "encoding": match &self.mode {
                VerifiedRewardEncodingMode::IntegerObjectiveDifferenceInterval => "integer_objective_difference",
                VerifiedRewardEncodingMode::FiniteRewardMap { .. } => "finite_reward_map",
            },
            "finite_reward_values": match &self.mode {
                VerifiedRewardEncodingMode::IntegerObjectiveDifferenceInterval => None,
                VerifiedRewardEncodingMode::FiniteRewardMap { map } => Some(map.objective_difference_to_symbol.len()),
            },
            "complete_nonnegative_interval_max": match &self.mode {
                VerifiedRewardEncodingMode::IntegerObjectiveDifferenceInterval => None,
                VerifiedRewardEncodingMode::FiniteRewardMap { map } => map.complete_nonnegative_interval_max,
            },
        })
    }

    fn is_identity_or_interval_encoding(&self) -> bool {
        match &self.mode {
            VerifiedRewardEncodingMode::IntegerObjectiveDifferenceInterval => true,
            VerifiedRewardEncodingMode::FiniteRewardMap { map } => map
                .objective_difference_to_symbol
                .iter()
                .all(|(objective_difference, symbol)| objective_difference == symbol),
        }
    }
}

impl VerifiedExactStateObservationCertificate {
    fn to_json_value(&self) -> Value {
        serde_json::json!({
            "ref": self.base.ref_value,
            "content_crc32": self.base.content_hash,
            "verified": true,
            "observation_key_mode": self.observation_key_mode,
            "exact_state_encoder_spec_ref": self.exact_state_encoder_spec_ref,
            "observation_adapter_spec_ref": self.observation_adapter_spec_ref,
            "observation_adapter_content_crc32": self.observation_adapter_content_hash,
            "finite_planner_state_certificate_crc32": self.finite_planner_state_certificate_hash,
            "finite_state_count": self.finite_state_count,
        })
    }
}

impl VerifiedDeterministicEvaluatorTable {
    fn to_json_value(&self) -> Value {
        serde_json::json!({
            "ref": self.base.ref_value,
            "content_crc32": self.base.content_hash,
            "verified": true,
            "rows": self.rows.len(),
        })
    }

    fn evaluate(
        &self,
        candidate: &crate::spec::CompiledCompressionBackend,
        dataset: &LoadedDataset,
        model_bytes: usize,
        min_throughput_bytes_per_second: f64,
        max_memory_bytes: u64,
        effective_eval_time_limit_seconds: f64,
    ) -> Result<CandidateEvalResult, String> {
        let candidate_crc32 = crc32_hex(candidate.canonical_bytes().as_slice());
        let row = self.rows.get(&candidate_crc32).ok_or_else(|| {
            format!(
                "deterministic evaluator table missing row for candidate_crc32 '{candidate_crc32}'"
            )
        })?;
        if row.status == CandidateEvalStatus::Timeout {
            return Ok(timeout_eval_result(
                row.elapsed_seconds,
                row.peak_memory_bytes,
                effective_eval_time_limit_seconds,
            ));
        }
        if row.status != CandidateEvalStatus::Success {
            return Ok(CandidateEvalResult {
                status: row.status,
                compressed_bytes: row.compressed_bytes,
                elapsed_seconds: row.elapsed_seconds,
                effective_eval_time_limit_seconds,
                throughput_bytes_per_second: 0.0,
                peak_memory_bytes: row.peak_memory_bytes,
                target_loss_bits: f64::INFINITY,
                objective_bits: f64::INFINITY,
                deployable: false,
            });
        }
        let throughput_bytes_per_second = if row.elapsed_seconds <= 0.0 {
            f64::INFINITY
        } else {
            dataset.dataset_units / row.elapsed_seconds
        };
        let objective_bits = ((model_bytes as f64) * 8.0) + row.target_loss_bits;
        let deployable = throughput_bytes_per_second >= min_throughput_bytes_per_second
            && row.peak_memory_bytes <= max_memory_bytes;
        Ok(CandidateEvalResult {
            status: row.status,
            compressed_bytes: row.compressed_bytes,
            elapsed_seconds: row.elapsed_seconds,
            effective_eval_time_limit_seconds,
            throughput_bytes_per_second,
            peak_memory_bytes: row.peak_memory_bytes,
            target_loss_bits: row.target_loss_bits,
            objective_bits,
            deployable,
        })
    }
}

#[allow(clippy::too_many_arguments)]
fn load_generic_certificate(
    reference: Option<&str>,
    expected_kind: &str,
    compiled: &crate::spec::CompiledTuneSpec,
    dataset: &LoadedDataset,
    evaluator_profile: &EvaluatorProfile,
    config_dir: &Path,
    bounds_digest: &str,
    controller_kind: &str,
) -> Result<Option<VerifiedCertificate>, String> {
    let Some((ref_value, value, content_hash)) = load_certificate_value(reference, config_dir)?
    else {
        return Ok(None);
    };
    validate_certificate_common(
        &value,
        expected_kind,
        compiled,
        dataset,
        evaluator_profile,
        bounds_digest,
        controller_kind,
    )?;
    Ok(Some(VerifiedCertificate {
        ref_value,
        content_hash,
    }))
}

#[allow(clippy::too_many_arguments)]
fn load_exact_reward_certificate(
    reference: Option<&str>,
    compiled: &crate::spec::CompiledTuneSpec,
    dataset: &LoadedDataset,
    evaluator_profile: &EvaluatorProfile,
    config_dir: &Path,
    bounds_digest: &str,
    controller_kind: &str,
    scalar_representation: &str,
) -> Result<Option<VerifiedExactRewardEncodingCertificate>, String> {
    let Some((ref_value, value, content_hash)) = load_certificate_value(reference, config_dir)?
    else {
        return Ok(None);
    };
    validate_certificate_common(
        &value,
        "exact_reward_encoding",
        compiled,
        dataset,
        evaluator_profile,
        bounds_digest,
        controller_kind,
    )?;
    let object = certificate_object(&value, "exact_reward_encoding")?;
    let encoding = required_cert_str(object, "encoding", "exact_reward_encoding")?;
    if encoding != "integer_objective_difference" && encoding != "finite_reward_map" {
        return Err(format!(
            "exact_reward_encoding certificate uses unsupported encoding '{encoding}'"
        ));
    }
    let cert_scalar = required_cert_str(object, "scalar_representation", "exact_reward_encoding")?;
    if cert_scalar != scalar_representation {
        return Err(format!(
            "exact_reward_encoding certificate scalar_representation '{cert_scalar}' does not match requested '{scalar_representation}'"
        ));
    }
    let reward_bits = required_cert_u64(object, "reward_bits", "exact_reward_encoding")?;
    let reward_bits = usize::try_from(reward_bits)
        .map_err(|_| "exact_reward_encoding.reward_bits does not fit usize".to_string())?;
    let max_reward = required_cert_u64(object, "max_reward", "exact_reward_encoding")?;
    let max_reward = Reward::try_from(max_reward)
        .map_err(|_| "exact_reward_encoding.max_reward does not fit Reward".to_string())?;
    let max_encoded = max_nonnegative_reward_for_bits(reward_bits)?;
    if max_reward > max_encoded {
        return Err(format!(
            "exact_reward_encoding.max_reward {max_reward} exceeds reward_bits={reward_bits} maximum {max_encoded}"
        ));
    }
    let mode = if encoding == "integer_objective_difference" {
        VerifiedRewardEncodingMode::IntegerObjectiveDifferenceInterval
    } else {
        let map = parse_finite_reward_map(object, reward_bits, max_reward)?;
        if controller_requires_exact_objective_difference(controller_kind)
            && map.complete_nonnegative_interval_max.is_none()
        {
            return Err(
                "exact_reward_encoding finite_reward_map certificates for exact-objective controllers must declare complete_nonnegative_interval_max"
                    .to_string(),
            );
        }
        VerifiedRewardEncodingMode::FiniteRewardMap { map }
    };
    Ok(Some(VerifiedExactRewardEncodingCertificate {
        base: VerifiedCertificate {
            ref_value,
            content_hash,
        },
        max_reward,
        reward_bits,
        scalar_representation: cert_scalar.to_string(),
        mode,
    }))
}

fn controller_requires_exact_objective_difference(controller_kind: &str) -> bool {
    matches!(
        controller_kind,
        "mc_aixi_fac_ctw" | "aiqi_warmstart_exact_jh"
    )
}

fn parse_finite_reward_map(
    object: &serde_json::Map<String, Value>,
    reward_bits: usize,
    declared_max_reward: Reward,
) -> Result<VerifiedFiniteRewardMap, String> {
    let entries = object
        .get("values")
        .and_then(Value::as_array)
        .ok_or_else(|| {
            "exact_reward_encoding finite_reward_map requires a 'values' array".to_string()
        })?;
    if entries.is_empty() {
        return Err("exact_reward_encoding finite_reward_map values must be nonempty".to_string());
    }
    let max_encoded = max_nonnegative_reward_for_bits(reward_bits)?;
    let mut by_difference = BTreeMap::<Reward, Reward>::new();
    let mut seen_symbols = BTreeSet::<Reward>::new();
    for (index, entry) in entries.iter().enumerate() {
        let entry_object = entry
            .as_object()
            .ok_or_else(|| format!("exact_reward_encoding.values[{index}] must be an object"))?;
        let difference = required_cert_u64(
            entry_object,
            "objective_difference",
            &format!("exact_reward_encoding.values[{index}]"),
        )?;
        let difference = Reward::try_from(difference).map_err(|_| {
            format!(
                "exact_reward_encoding.values[{index}].objective_difference does not fit Reward"
            )
        })?;
        let symbol = required_cert_u64(
            entry_object,
            "symbol",
            &format!("exact_reward_encoding.values[{index}]"),
        )?;
        let symbol = Reward::try_from(symbol).map_err(|_| {
            format!("exact_reward_encoding.values[{index}].symbol does not fit Reward")
        })?;
        if symbol > max_encoded {
            return Err(format!(
                "exact_reward_encoding.values[{index}].symbol {symbol} exceeds reward_bits={reward_bits} maximum {max_encoded}"
            ));
        }
        if symbol > declared_max_reward {
            return Err(format!(
                "exact_reward_encoding.values[{index}].symbol {symbol} exceeds declared max_reward {declared_max_reward}"
            ));
        }
        if by_difference.insert(difference, symbol).is_some() {
            return Err(format!(
                "exact_reward_encoding finite_reward_map duplicates objective_difference {difference}"
            ));
        }
        if !seen_symbols.insert(symbol) {
            return Err(format!(
                "exact_reward_encoding finite_reward_map duplicates reward symbol {symbol}"
            ));
        }
    }
    let complete_nonnegative_interval_max = object
        .get("complete_nonnegative_interval_max")
        .map(|value| {
            value.as_u64().ok_or_else(|| {
                "exact_reward_encoding.complete_nonnegative_interval_max must be an unsigned integer"
                    .to_string()
            })
        })
        .transpose()?
        .map(|max| {
            Reward::try_from(max).map_err(|_| {
                "exact_reward_encoding.complete_nonnegative_interval_max does not fit Reward"
                    .to_string()
            })
        })
        .transpose()?;
    if let Some(complete_max) = complete_nonnegative_interval_max {
        if complete_max > declared_max_reward {
            return Err(format!(
                "exact_reward_encoding.complete_nonnegative_interval_max {complete_max} exceeds declared max_reward {declared_max_reward}"
            ));
        }
        validate_complete_finite_reward_interval(&by_difference, complete_max)?;
    }
    Ok(VerifiedFiniteRewardMap {
        objective_difference_to_symbol: by_difference,
        complete_nonnegative_interval_max,
    })
}

fn validate_complete_finite_reward_interval(
    objective_difference_to_symbol: &BTreeMap<Reward, Reward>,
    complete_max: Reward,
) -> Result<(), String> {
    for objective_difference in 0..=complete_max {
        if !objective_difference_to_symbol.contains_key(&objective_difference) {
            return Err(format!(
                "exact_reward_encoding finite_reward_map complete interval is missing objective_difference {objective_difference}"
            ));
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn load_exact_state_observation_certificate(
    theorem: &TuneTheoremConfig,
    finite_planner_state: Option<&VerifiedCertificate>,
    compiled: &crate::spec::CompiledTuneSpec,
    dataset: &LoadedDataset,
    evaluator_profile: &EvaluatorProfile,
    config_dir: &Path,
    bounds_digest: &str,
    controller_kind: &str,
) -> Result<Option<VerifiedExactStateObservationCertificate>, String> {
    let Some((ref_value, value, content_hash)) = load_certificate_value(
        theorem.exact_state_observation_certificate.as_deref(),
        config_dir,
    )?
    else {
        return Ok(None);
    };
    validate_certificate_common(
        &value,
        "exact_state_observation",
        compiled,
        dataset,
        evaluator_profile,
        bounds_digest,
        controller_kind,
    )?;
    let object = certificate_object(&value, "exact_state_observation")?;
    let observation_key_mode =
        required_cert_str(object, "observation_key_mode", "exact_state_observation")?;
    let requested_adapter = theorem
        .observation_adapter_spec_ref
        .as_deref()
        .unwrap_or(OBSERVATION_ADAPTER_DECLARATION);
    let observed_adapter = required_cert_str(
        object,
        "observation_adapter_spec_ref",
        "exact_state_observation",
    )?;
    if observed_adapter != requested_adapter {
        return Err(format!(
            "exact_state_observation certificate observation_adapter_spec_ref '{observed_adapter}' does not match requested '{requested_adapter}'"
        ));
    }
    require_cert_string_match(
        object,
        "observation_adapter_content_crc32",
        &observation_adapter_content_hash()?,
        "exact_state_observation",
    )?;
    let finite_planner_state = finite_planner_state.ok_or_else(|| {
        "exact_state_observation certificate requires a verified finite_planner_state_certificate"
            .to_string()
    })?;
    require_cert_string_match(
        object,
        "finite_planner_state_certificate_crc32",
        &finite_planner_state.content_hash,
        "exact_state_observation",
    )?;
    let observed_encoder = required_cert_str(
        object,
        "exact_state_encoder_spec_ref",
        "exact_state_observation",
    )?;
    if let Some(expected) = theorem.exact_state_encoder_spec_ref.as_deref()
        && observed_encoder != expected
    {
        return Err(format!(
            "exact_state_observation certificate encoder ref '{observed_encoder}' does not match requested '{expected}'"
        ));
    }
    let interface = planner_interface_for_controller(compiled.controller()).ok_or_else(|| {
        "exact_state_observation certificate can only be validated for planner-family controllers"
            .to_string()
    })?;
    let finite_state_count =
        validate_exact_state_observation_artifact(object, observation_key_mode, interface)?;
    Ok(Some(VerifiedExactStateObservationCertificate {
        base: VerifiedCertificate {
            ref_value,
            content_hash,
        },
        observation_key_mode: observation_key_mode.to_string(),
        exact_state_encoder_spec_ref: observed_encoder.to_string(),
        observation_adapter_spec_ref: observed_adapter.to_string(),
        observation_adapter_content_hash: observation_adapter_content_hash()?,
        finite_planner_state_certificate_hash: finite_planner_state.content_hash.clone(),
        finite_state_count,
    }))
}

fn planner_interface_for_controller(
    controller: &crate::spec::CompiledTuneController,
) -> Option<&crate::spec::TunePlannerInterfaceSpec> {
    match controller {
        crate::spec::CompiledTuneController::McAixiFacCtw(inner) => Some(&inner.interface),
        crate::spec::CompiledTuneController::AiqiDiscounted(inner) => Some(&inner.interface),
        crate::spec::CompiledTuneController::AiqiWarmstartExactJh(inner) => Some(&inner.interface),
        crate::spec::CompiledTuneController::AnnealedHillClimbing(_) => None,
    }
}

fn validate_exact_state_observation_artifact(
    object: &serde_json::Map<String, Value>,
    observation_key_mode: &str,
    interface: &crate::spec::TunePlannerInterfaceSpec,
) -> Result<usize, String> {
    let states = object
        .get("psi_h_outputs")
        .or_else(|| object.get("state_outputs"))
        .and_then(Value::as_array)
        .ok_or_else(|| {
            "exact_state_observation certificate requires a checkable psi_h_outputs array"
                .to_string()
        })?;
    if states.is_empty() {
        return Err("exact_state_observation psi_h_outputs must be nonempty".to_string());
    }
    let stream_len = interface.observation_stream_len.max(1);
    let max_symbol = max_observation_symbol_for_bits(interface.observation_bits)?;
    let mut seen_state_ids = BTreeSet::<String>::new();
    let mut seen_projected_outputs = BTreeSet::<Vec<PerceptVal>>::new();
    for (index, state_value) in states.iter().enumerate() {
        let state_object = state_value.as_object().ok_or_else(|| {
            format!("exact_state_observation.psi_h_outputs[{index}] must be an object")
        })?;
        let state_id = required_cert_str(
            state_object,
            "state_id",
            &format!("exact_state_observation.psi_h_outputs[{index}]"),
        )?;
        if !seen_state_ids.insert(state_id.to_string()) {
            return Err(format!(
                "exact_state_observation duplicate state_id '{state_id}'"
            ));
        }
        let observations = state_object
            .get("observations")
            .and_then(Value::as_array)
            .ok_or_else(|| {
                format!(
                    "exact_state_observation.psi_h_outputs[{index}].observations must be an array"
                )
            })?;
        if observations.len() != stream_len {
            return Err(format!(
                "exact_state_observation.psi_h_outputs[{index}].observations length {} does not match observation_stream_len {stream_len}",
                observations.len()
            ));
        }
        let mut output = Vec::<PerceptVal>::with_capacity(stream_len);
        for (symbol_index, symbol_value) in observations.iter().enumerate() {
            let symbol = symbol_value.as_u64().ok_or_else(|| {
                format!(
                    "exact_state_observation.psi_h_outputs[{index}].observations[{symbol_index}] must be an integer"
                )
            })?;
            if symbol > max_symbol {
                return Err(format!(
                    "exact_state_observation.psi_h_outputs[{index}].observations[{symbol_index}]={symbol} exceeds observation_bits={} maximum {max_symbol}",
                    interface.observation_bits
                ));
            }
            output.push(symbol);
        }
        let projected =
            project_observation_output(observation_key_mode, &output, interface.observation_bits)?;
        if !seen_projected_outputs.insert(projected) {
            return Err(format!(
                "exact_state_observation psi_h_outputs are not injective under observation_key_mode '{observation_key_mode}'"
            ));
        }
    }
    Ok(states.len())
}

fn project_observation_output(
    observation_key_mode: &str,
    output: &[PerceptVal],
    observation_bits: usize,
) -> Result<Vec<PerceptVal>, String> {
    match observation_key_mode {
        "full_stream" => Ok(output.to_vec()),
        "first_symbol" | "first" => output
            .first()
            .copied()
            .map(|value| vec![value])
            .ok_or_else(|| "observation stream cannot be empty".to_string()),
        "last_symbol" | "last" => output
            .last()
            .copied()
            .map(|value| vec![value])
            .ok_or_else(|| "observation stream cannot be empty".to_string()),
        "stream_hash" => Ok(vec![crate::aixi::common::observation_key_from_stream(
            ObservationKeyMode::StreamHash,
            output,
            observation_bits,
        )]),
        other => Err(format!(
            "exact_state_observation certificate uses unsupported observation_key_mode '{other}'"
        )),
    }
}

fn max_observation_symbol_for_bits(observation_bits: usize) -> Result<PerceptVal, String> {
    if observation_bits == 0 {
        Ok(0)
    } else if observation_bits >= 64 {
        Ok(PerceptVal::MAX)
    } else {
        Ok((1_u64 << observation_bits) - 1)
    }
}

#[allow(clippy::too_many_arguments)]
fn load_deterministic_evaluator_table(
    reference: Option<&str>,
    compiled: &crate::spec::CompiledTuneSpec,
    dataset: &LoadedDataset,
    evaluator_profile: &EvaluatorProfile,
    config_dir: &Path,
    bounds_digest: &str,
    controller_kind: &str,
) -> Result<Option<VerifiedDeterministicEvaluatorTable>, String> {
    let Some((ref_value, value, content_hash)) = load_certificate_value(reference, config_dir)?
    else {
        return Ok(None);
    };
    validate_certificate_common(
        &value,
        "deterministic_evaluator_table",
        compiled,
        dataset,
        evaluator_profile,
        bounds_digest,
        controller_kind,
    )?;
    let object = certificate_object(&value, "deterministic_evaluator_table")?;
    let rows_value = object
        .get("rows")
        .and_then(Value::as_array)
        .ok_or_else(|| "deterministic_evaluator_table.rows must be an array".to_string())?;
    let mut rows = HashMap::<String, DeterministicEvaluatorRow>::new();
    for (index, row_value) in rows_value.iter().enumerate() {
        let row_object = row_value.as_object().ok_or_else(|| {
            format!("deterministic_evaluator_table.rows[{index}] must be an object")
        })?;
        let candidate_crc32 = required_cert_str(
            row_object,
            "candidate_crc32",
            &format!("deterministic_evaluator_table.rows[{index}]"),
        )?;
        let status = match required_cert_str(
            row_object,
            "status",
            &format!("deterministic_evaluator_table.rows[{index}]"),
        )? {
            "success" => CandidateEvalStatus::Success,
            "timeout" => CandidateEvalStatus::Timeout,
            "invalid" => CandidateEvalStatus::Invalid,
            "error" => CandidateEvalStatus::Error,
            other => {
                return Err(format!(
                    "deterministic_evaluator_table.rows[{index}].status has unknown status '{other}'"
                ));
            }
        };
        let compressed_bytes = required_cert_u64(
            row_object,
            "compressed_bytes",
            &format!("deterministic_evaluator_table.rows[{index}]"),
        )?;
        let compressed_bytes = usize::try_from(compressed_bytes).map_err(|_| {
            format!(
                "deterministic_evaluator_table.rows[{index}].compressed_bytes does not fit usize"
            )
        })?;
        let target_loss_bits = required_cert_f64(
            row_object,
            "target_loss_bits",
            &format!("deterministic_evaluator_table.rows[{index}]"),
        )?;
        let elapsed_seconds = required_cert_f64(
            row_object,
            "elapsed_seconds",
            &format!("deterministic_evaluator_table.rows[{index}]"),
        )?;
        let peak_memory_bytes = required_cert_u64(
            row_object,
            "peak_memory_bytes",
            &format!("deterministic_evaluator_table.rows[{index}]"),
        )?;
        if rows
            .insert(
                candidate_crc32.to_string(),
                DeterministicEvaluatorRow {
                    status,
                    compressed_bytes,
                    target_loss_bits,
                    elapsed_seconds,
                    peak_memory_bytes,
                },
            )
            .is_some()
        {
            return Err(format!(
                "deterministic_evaluator_table duplicate candidate_crc32 '{candidate_crc32}'"
            ));
        }
    }
    if rows.is_empty() {
        return Err("deterministic_evaluator_table.rows must not be empty".to_string());
    }
    Ok(Some(VerifiedDeterministicEvaluatorTable {
        base: VerifiedCertificate {
            ref_value,
            content_hash,
        },
        rows,
    }))
}

fn load_certificate_value(
    reference: Option<&str>,
    config_dir: &Path,
) -> Result<Option<(String, Value, String)>, String> {
    let Some(raw_reference) = reference else {
        return Ok(None);
    };
    let raw_ref = raw_reference.trim();
    if raw_ref.is_empty() {
        return Err("theorem certificate reference must be non-empty when set".to_string());
    };
    if raw_ref.contains("://") && !raw_ref.starts_with("file://") {
        return Err(format!(
            "unsupported theorem certificate reference scheme in '{raw_ref}'; use a filesystem path or file:// URI"
        ));
    }
    let path_text = raw_ref.strip_prefix("file://").unwrap_or(raw_ref);
    let path = Path::new(path_text);
    let resolved = if path.is_absolute() {
        path.to_path_buf()
    } else {
        config_dir.join(path)
    };
    let raw = fs::read(&resolved).map_err(|err| {
        format!(
            "failed to read theorem certificate '{}': {err}",
            resolved.display()
        )
    })?;
    let value: Value = serde_json::from_slice(&raw).map_err(|err| {
        format!(
            "invalid theorem certificate JSON '{}': {err}",
            resolved.display()
        )
    })?;
    Ok(Some((raw_ref.to_string(), value, crc32_hex(&raw))))
}

#[allow(clippy::too_many_arguments)]
fn validate_certificate_common(
    value: &Value,
    expected_kind: &str,
    compiled: &crate::spec::CompiledTuneSpec,
    dataset: &LoadedDataset,
    evaluator_profile: &EvaluatorProfile,
    bounds_digest: &str,
    controller_kind: &str,
) -> Result<(), String> {
    let object = certificate_object(value, expected_kind)?;
    let schema_version = required_cert_u64(object, "schema_version", expected_kind)?;
    if schema_version != 1 {
        return Err(format!(
            "{expected_kind} certificate schema_version must be 1, got {schema_version}"
        ));
    }
    let kind = required_cert_str(object, "kind", expected_kind)?;
    if kind != expected_kind {
        return Err(format!("{expected_kind} certificate has kind '{kind}'"));
    }
    require_cert_string_match(
        object,
        "dataset_crc32",
        &dataset.canonical_content_hash,
        expected_kind,
    )?;
    require_cert_string_match(object, "bounds_crc32", bounds_digest, expected_kind)?;
    require_cert_string_match(
        object,
        "evaluator_profile_crc32",
        &evaluator_profile.hash()?,
        expected_kind,
    )?;
    require_cert_string_match(object, "controller_kind", controller_kind, expected_kind)?;
    if let Some(action_alphabet_size) = object.get("action_alphabet_size") {
        let expected = planner_action_count(compiled)?;
        let observed = action_alphabet_size
            .as_u64()
            .ok_or_else(|| format!("{expected_kind}.action_alphabet_size must be an integer"))?;
        if observed != expected as u64 {
            return Err(format!(
                "{expected_kind}.action_alphabet_size {observed} does not match compiled {expected}"
            ));
        }
    }
    Ok(())
}

fn certificate_object<'a>(
    value: &'a Value,
    label: &str,
) -> Result<&'a serde_json::Map<String, Value>, String> {
    value
        .as_object()
        .ok_or_else(|| format!("{label} certificate must be a JSON object"))
}

fn required_cert_str<'a>(
    object: &'a serde_json::Map<String, Value>,
    field: &str,
    label: &str,
) -> Result<&'a str, String> {
    object
        .get(field)
        .and_then(Value::as_str)
        .ok_or_else(|| format!("{label}.{field} must be a string"))
}

fn required_cert_u64(
    object: &serde_json::Map<String, Value>,
    field: &str,
    label: &str,
) -> Result<u64, String> {
    object
        .get(field)
        .and_then(Value::as_u64)
        .ok_or_else(|| format!("{label}.{field} must be an unsigned integer"))
}

fn required_cert_f64(
    object: &serde_json::Map<String, Value>,
    field: &str,
    label: &str,
) -> Result<f64, String> {
    let value = object
        .get(field)
        .and_then(Value::as_f64)
        .ok_or_else(|| format!("{label}.{field} must be a finite number"))?;
    if !value.is_finite() || value < 0.0 {
        return Err(format!("{label}.{field} must be finite and nonnegative"));
    }
    Ok(value)
}

fn require_cert_string_match(
    object: &serde_json::Map<String, Value>,
    field: &str,
    expected: &str,
    label: &str,
) -> Result<(), String> {
    let observed = required_cert_str(object, field, label)?;
    if observed != expected {
        return Err(format!(
            "{label}.{field} '{observed}' does not match expected '{expected}'"
        ));
    }
    Ok(())
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
    cache: &mut HashMap<CandidateCacheKey, CandidateEvalResult>,
    max_mutation_radius: usize,
) -> Result<SearchSummary, String> {
    let base_dir = Path::new(&request.spec_path)
        .parent()
        .unwrap_or(Path::new("."))
        .to_path_buf();
    let env = SpecEnvironment::new(base_dir);
    let mut rng = RandomGenerator::from_seed(compiled.canonical_spec().seed);

    let mut best_candidate = compiled.canonical_spec().baseline_candidate.clone();
    let mut best_eval = baseline_eval.clone();
    let mut best_hash = baseline_hash;
    let mut best_bytes = baseline_bytes;
    let mut best_key = baseline_key;

    let mut current_candidate = best_candidate.clone();
    let mut current_eval = baseline_eval;
    let mut cache_hits: usize = 0;
    let mut cache_misses: usize = 1;
    let mut candidate_evaluations_executed: usize = 1;
    let mut proposals_attempted: usize = 0;
    let mut proposals_invalid: usize = 0;
    let mut self_loop_proposals: usize = 0;
    let mut successful_non_deployable: usize = 0;
    let mut final_best_move_reward: f64 = 0.0;
    let mut evaluations_seen: usize = 1;
    let max_evaluations = request.execution.max_evaluations.unwrap_or(usize::MAX);

    let mut stagnation_counter: usize = 0;
    loop {
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

        if reject_candidate_local_external_artifacts(&proposed_candidate).is_err()
            || validate_candidate_against_tune_bounds(
                &proposed_candidate,
                &compiled.canonical_spec().bounds,
            )
            .is_err()
        {
            proposals_invalid = proposals_invalid.saturating_add(1);
            continue;
        }

        let compiled_candidate = match proposed_candidate.compile_in(&env) {
            Ok(value) => value,
            Err(_) => {
                proposals_invalid = proposals_invalid.saturating_add(1);
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
                request.execution.rss_mode,
                verified_theorem.deterministic_table.as_ref(),
            ) {
                Ok(value) => value,
                Err(_) => error_eval_result(0.0, 0, effective_limit),
            };
            cache.insert(cache_key.clone(), evaluated.clone());
            cache_misses = cache_misses.saturating_add(1);
            candidate_evaluations_executed = candidate_evaluations_executed.saturating_add(1);
            evaluated
        };
        evaluations_seen = evaluations_seen.saturating_add(1);

        if !candidate_eval.deployable {
            proposals_invalid = proposals_invalid.saturating_add(1);
            successful_non_deployable = successful_non_deployable.saturating_add(1);
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

        let better_objective = candidate_eval.objective_bits < best_eval.objective_bits;
        let tie_and_lexicographically_smaller =
            (candidate_eval.objective_bits - best_eval.objective_bits).abs() <= f64::EPSILON
                && candidate_bytes < best_bytes;
        if better_objective || tie_and_lexicographically_smaller {
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

    Ok(SearchSummary {
        status: "completed_annealed",
        warning: None,
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
        successful_non_deployable,
        final_best_move_reward,
        controller_report: serde_json::json!({
            "kind": "annealed_hill_climbing",
            "runtime_path": annealer_runtime_path_name(request.execution.annealer_kernel_profile),
            "max_mutation_radius": max_mutation_radius,
            "annealer_kernel_profile": annealer_kernel_profile_name(request.execution.annealer_kernel_profile),
            "proposal_mass_accounting": request.execution.annealer_kernel_profile == AnnealerKernelProfile::CompiledUniformMetropolisHastings,
            "proposal_action_distribution": "uniform_finite_integer_elementary_descriptors",
        }),
    })
}

#[allow(clippy::too_many_arguments)]
fn run_planner_family_controller(
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
    cache: &mut HashMap<CandidateCacheKey, CandidateEvalResult>,
) -> Result<SearchSummary, String> {
    let contract =
        planner_controller_contract(compiled.controller(), compiled, dataset, verified_theorem)?;
    let reward_encoder = contract.reward_encoder(dataset, &baseline_eval, verified_theorem)?;
    let actions = compile_planner_mutation_actions(
        compiled.canonical_spec().baseline_candidate.clone(),
        &compiled.canonical_spec().bounds,
    )?;
    validate_theorem_planner_mutation_domain(&actions, &request.execution.theorem)?;
    let declared_actions = contract.interface.agent_actions.get();
    if actions.len() != declared_actions {
        return Err(format!(
            "action_alphabet_mismatch: compiled mutation alphabet has {} actions but controller.interface.agent_actions declares {}",
            actions.len(),
            declared_actions
        ));
    }
    let planner_run = compile_tuner_planner_run_spec(
        compiled.controller(),
        &contract,
        &reward_encoder,
        compiled,
    )?;
    let mut agent_runtime = build_tuner_planner_agent_runtime(
        compiled.controller(),
        &planner_run,
        &contract,
        encode_tuner_planner_percept(
            &contract.interface,
            Some(&baseline_eval),
            dataset.dataset_units,
            0,
            "baseline_initial_state",
            Some(&baseline_eval),
            Some(&baseline_bytes),
            Some(compiled.canonical_spec().eval_time_limit_seconds),
            false,
        )?,
    )?;
    let base_dir = Path::new(&request.spec_path)
        .parent()
        .unwrap_or(Path::new("."))
        .to_path_buf();
    let env = SpecEnvironment::new(base_dir);

    let mut best_candidate = compiled.canonical_spec().baseline_candidate.clone();
    let mut best_eval = baseline_eval.clone();
    let mut best_hash = baseline_hash;
    let mut best_bytes = baseline_bytes;
    let mut best_key = baseline_key;
    let mut current_candidate = best_candidate.clone();
    let mut current_eval = baseline_eval;
    let mut current_bytes = best_bytes.clone();
    let mut cache_hits: usize = 0;
    let mut cache_misses: usize = 1;
    let mut candidate_evaluations_executed: usize = 1;
    let mut proposals_attempted: usize = 0;
    let mut proposals_invalid: usize = 0;
    let mut self_loop_proposals: usize = 0;
    let mut successful_non_deployable: usize = 0;
    let mut final_best_move_reward: f64 = 0.0;
    let mut evaluations_seen: usize = 1;
    let max_evaluations = request.execution.max_evaluations.unwrap_or(usize::MAX);
    let mut stagnation_counter: usize = 0;
    let mut decision_steps: usize = 0;
    let mut warmstart_trace_refresh_merges: usize = 0;
    let total_rounds = if contract.warmstart_self_improvement {
        request.execution.self_improvement_rounds.max(1)
    } else {
        1
    };
    let trace_refresh_enabled = contract.warmstart_self_improvement
        && request.execution.warmstart_trace_refresh
        && total_rounds > 1;
    let mut refresh_teacher = contract
        .teacher
        .as_ref()
        .map(|teacher| teacher.traces.clone());
    let planner_return_bins = planner_return_bins(&planner_run);

    for round in 0..total_rounds {
        if trace_refresh_enabled && round > 0 {
            if let (Some(live_trace), Some(teacher)) = (
                agent_runtime.same_task_live_trace(),
                refresh_teacher.as_mut(),
            ) {
                merge_warmstart_trace_deterministic(teacher, live_trace)?;
                agent_runtime.rebuild_warmstart_agent(&planner_run, teacher.clone())?;
                warmstart_trace_refresh_merges = warmstart_trace_refresh_merges.saturating_add(1);
            }
        }
        let round_deadline_seconds = if contract.warmstart_self_improvement && total_rounds > 1 {
            Some(
                ((round + 1) as f64 / total_rounds as f64)
                    * compiled.canonical_spec().time_budget_seconds,
            )
        } else {
            None
        };
        loop {
            let elapsed_seconds = tune_started.elapsed().as_secs_f64();
            if evaluations_seen >= max_evaluations
                || elapsed_seconds >= compiled.canonical_spec().time_budget_seconds
                || round_deadline_seconds
                    .map(|deadline| elapsed_seconds >= deadline)
                    .unwrap_or(false)
            {
                break;
            }
            let action = agent_runtime.select_action();
            proposals_attempted = proposals_attempted.saturating_add(1);
            let action_index = usize::try_from(action)
                .map_err(|_| format!("planner action {action} does not fit usize"))?;
            if action_index >= actions.len() {
                proposals_invalid = proposals_invalid.saturating_add(1);
                let percept = encode_tuner_planner_percept(
                    &contract.interface,
                    Some(&current_eval),
                    dataset.dataset_units,
                    0,
                    "invalid_action_index",
                    None,
                    None,
                    None,
                    false,
                )?;
                agent_runtime.observe_transition(action, percept)?;
                decision_steps = decision_steps.saturating_add(1);
                continue;
            }
            let Some(proposed_candidate) =
                apply_planner_mutation_action(&current_candidate, &actions[action_index])?
            else {
                self_loop_proposals = self_loop_proposals.saturating_add(1);
                let percept = encode_tuner_planner_percept(
                    &contract.interface,
                    Some(&current_eval),
                    dataset.dataset_units,
                    0,
                    "inapplicable_action",
                    None,
                    None,
                    None,
                    false,
                )?;
                agent_runtime.observe_transition(action, percept)?;
                decision_steps = decision_steps.saturating_add(1);
                continue;
            };
            if reject_candidate_local_external_artifacts(&proposed_candidate).is_err()
                || validate_candidate_against_tune_bounds(
                    &proposed_candidate,
                    &compiled.canonical_spec().bounds,
                )
                .is_err()
            {
                proposals_invalid = proposals_invalid.saturating_add(1);
                let percept = encode_tuner_planner_percept(
                    &contract.interface,
                    Some(&current_eval),
                    dataset.dataset_units,
                    0,
                    "candidate_out_of_bounds_or_external_artifact",
                    None,
                    None,
                    None,
                    false,
                )?;
                agent_runtime.observe_transition(action, percept)?;
                decision_steps = decision_steps.saturating_add(1);
                continue;
            }
            let compiled_candidate = match proposed_candidate.compile_in(&env) {
                Ok(value) => value,
                Err(_) => {
                    proposals_invalid = proposals_invalid.saturating_add(1);
                    let percept = encode_tuner_planner_percept(
                        &contract.interface,
                        Some(&current_eval),
                        dataset.dataset_units,
                        0,
                        "candidate_compile_error",
                        None,
                        None,
                        None,
                        false,
                    )?;
                    agent_runtime.observe_transition(action, percept)?;
                    decision_steps = decision_steps.saturating_add(1);
                    continue;
                }
            };
            let candidate_bytes = compiled_candidate.canonical_bytes().as_slice().to_vec();
            let candidate_hash = crc32_hex(&candidate_bytes);
            let effective_limit = effective_eval_limit_seconds(
                compiled,
                tune_started,
                Some(compiled.canonical_spec().eval_time_limit_seconds),
                round_deadline_seconds,
            );
            if effective_limit <= 0.0 {
                break;
            }
            let candidate_profile = evaluator_profile.with_eval_time_limit(effective_limit);
            if candidate_bytes == current_bytes {
                self_loop_proposals = self_loop_proposals.saturating_add(1);
                let percept = encode_tuner_planner_percept(
                    &contract.interface,
                    Some(&current_eval),
                    dataset.dataset_units,
                    0,
                    "self_loop_proposal",
                    None,
                    Some(&candidate_bytes),
                    None,
                    false,
                )?;
                agent_runtime.observe_transition(action, percept)?;
                decision_steps = decision_steps.saturating_add(1);
                continue;
            }
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
                    request.execution.rss_mode,
                    verified_theorem.deterministic_table.as_ref(),
                ) {
                    Ok(value) => value,
                    Err(_) => error_eval_result(0.0, 0, effective_limit),
                };
                cache.insert(cache_key.clone(), evaluated.clone());
                cache_misses = cache_misses.saturating_add(1);
                candidate_evaluations_executed = candidate_evaluations_executed.saturating_add(1);
                evaluated
            };
            evaluations_seen = evaluations_seen.saturating_add(1);
            let incumbent_eval_before_step = current_eval.clone();
            let mut raw_improvement = 0.0f64;
            if !candidate_eval.deployable {
                successful_non_deployable = successful_non_deployable.saturating_add(1);
            } else {
                if key_less(
                    &candidate_eval,
                    &candidate_bytes,
                    &current_eval,
                    &current_bytes,
                ) {
                    raw_improvement =
                        (current_eval.objective_bits - candidate_eval.objective_bits).max(0.0);
                    current_candidate = proposed_candidate.clone();
                    current_eval = candidate_eval.clone();
                    current_bytes = candidate_bytes.clone();
                }
                if key_less(&candidate_eval, &candidate_bytes, &best_eval, &best_bytes) {
                    final_best_move_reward = raw_improvement;
                    best_candidate = proposed_candidate;
                    best_eval = candidate_eval.clone();
                    best_hash = candidate_hash;
                    best_bytes = candidate_bytes.clone();
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
                    current_bytes = best_bytes.clone();
                    stagnation_counter = 0;
                }
            }
            let reward = reward_encoder.encode(raw_improvement)?;
            let diagnostic_token = if !candidate_eval.deployable {
                match candidate_eval.status {
                    CandidateEvalStatus::Timeout => "evaluator_timeout",
                    CandidateEvalStatus::Invalid => "evaluator_invalid",
                    CandidateEvalStatus::Error => "evaluator_error",
                    CandidateEvalStatus::Success => "nondeployable_candidate",
                }
            } else {
                "deployable_success"
            };
            let percept = encode_tuner_planner_percept(
                &contract.interface,
                Some(&incumbent_eval_before_step),
                dataset.dataset_units,
                reward,
                diagnostic_token,
                Some(&candidate_eval),
                Some(&candidate_bytes),
                Some(effective_limit),
                false,
            )?;
            agent_runtime.observe_transition(action, percept)?;
            decision_steps = decision_steps.saturating_add(1);
        }
    }

    Ok(SearchSummary {
        status: planner_completed_status(compiled.controller()),
        warning: None,
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
        successful_non_deployable,
        final_best_move_reward,
        controller_report: serde_json::json!({
            "kind": controller_kind_name(compiled.controller()),
            "runtime_path": planner_runtime_path_name(compiled.controller()),
            "planner_simulations_per_step": contract.planner_simulations_per_step,
            "simulations_per_decision_step": contract.planner_simulations_per_step,
            "decision_steps": decision_steps,
            "return_horizon": contract.return_horizon,
            "return_bins": planner_return_bins,
            "label_phase_period": contract.label_phase_period,
            "reward_semantics": contract.reward_semantics.name(),
            "reward_encoding": tuner_reward_encoding_name(&reward_encoder),
            "discount_factor": contract.discount_factor,
            "planner_run_controller_kind": planner_run.controller().kind_str(),
            "agent_runtime": planner_agent_runtime_name(compiled.controller()),
            "compiled_action_count": actions.len(),
            "compiled_action_paths": planner_action_paths(&actions),
            "declared_agent_actions": declared_actions,
            "observation_adapter": OBSERVATION_ADAPTER_DECLARATION,
            "scalar_representation": SCALAR_REPRESENTATION_DECLARATION,
            "warmstart_teacher_dataset": contract.teacher.as_ref().map(|value| serde_json::json!({
                "asset_id": value.asset_id.clone(),
                "resolved_path": value.resolved_path.clone(),
                "content_crc32": value.content_hash.clone(),
                "records": value.records,
            })),
            "warmstart_self_improvement_update": if contract.warmstart_self_improvement {
                Some("online_exact_h_step_delayed_label_update")
            } else {
                None::<&str>
            },
            "warmstart_trace_refresh_merges": warmstart_trace_refresh_merges,
        }),
    })
}

fn load_dataset(path: &Path) -> Result<LoadedDataset, String> {
    let bytes = fs::read(path)
        .map_err(|err| format!("failed to read input_asset '{}': {err}", path.display()))?;
    if let Ok(Value::Object(object)) = serde_json::from_slice::<Value>(&bytes) {
        if object.contains_key("events") {
            let value = Value::Object(object);
            return lower_interactive_trace_dataset(path, bytes.len(), &value);
        }
        if object.contains_key("examples") || object.contains_key("prefixes") {
            let value = Value::Object(object);
            return lower_causal_prefix_dataset(path, bytes.len(), &value);
        }
        return Err(
            "JSON object input_asset must match a canonical tuner causal dataset kind: provide 'events' or 'examples'/'prefixes'"
                .to_string(),
        );
    }
    let hash = crc32_hex(&bytes);
    Ok(LoadedDataset {
        kind: DatasetKind::PassiveBytes,
        objective_target: ObjectiveTarget::PassiveAc,
        lowering_version: PASSIVE_DATASET_LOWERING_VERSION,
        codec_hash: "passive-identity-bytes".to_string(),
        event_grammar_hash: "passive-target-only-byte-stream".to_string(),
        target_domain_support_hash: crc32_hex(b"passive-byte-alphabet"),
        causal_header_profile_hash: crc32_hex(b"passive-none"),
        target_size_function: "passive-bytes-len",
        canonical_content_hash: hash,
        lowered_skeleton_hash: crc32_hex(b"passive-bytes-target-only"),
        resolved_path: path.to_string_lossy().to_string(),
        source_size_bytes: bytes.len(),
        dataset_units: bytes.len() as f64,
        target_events: usize::from(!bytes.is_empty()),
        events: Vec::new(),
        causal_profile: None,
        raw_bytes: bytes,
    })
}

fn lower_interactive_trace_dataset(
    path: &Path,
    source_size_bytes: usize,
    value: &Value,
) -> Result<LoadedDataset, String> {
    let (header_profile, domain_supports) = parse_causal_header_profile(
        value,
        "interactive trace dataset",
        CausalPayloadKind::Events,
    )?;
    let events_value = value
        .get("events")
        .and_then(Value::as_array)
        .ok_or_else(|| "interactive trace dataset requires an array field 'events'".to_string())?;
    let events = events_value
        .iter()
        .enumerate()
        .map(|(index, event)| parse_lowered_event(event, &format!("events[{index}]")))
        .collect::<Result<Vec<_>, _>>()?;
    lowered_dataset_from_events(
        DatasetKind::InteractiveTrace,
        ObjectiveTarget::InteractiveCausalAc,
        INTERACTIVE_TRACE_LOWERING_VERSION,
        path,
        source_size_bytes,
        value,
        header_profile,
        domain_supports,
        events,
        "interactive-trace-target-bytes",
    )
}

fn lower_causal_prefix_dataset(
    path: &Path,
    source_size_bytes: usize,
    value: &Value,
) -> Result<LoadedDataset, String> {
    let (header_profile, domain_supports) = parse_causal_header_profile(
        value,
        "causal-prefix dataset",
        CausalPayloadKind::ExamplesOrPrefixes,
    )?;
    let examples_value = value
        .get("examples")
        .or_else(|| value.get("prefixes"))
        .and_then(Value::as_array)
        .ok_or_else(|| {
            "causal-prefix dataset requires an array field 'examples' or 'prefixes'".to_string()
        })?;
    let mut events = Vec::<LoweredCausalEvent>::new();
    for (index, example) in examples_value.iter().enumerate() {
        let object = example
            .as_object()
            .ok_or_else(|| format!("examples[{index}] must be an object"))?;
        events.push(LoweredCausalEvent::Reset);
        if let Some(history) = object.get("history").and_then(Value::as_array) {
            for (history_index, event) in history.iter().enumerate() {
                let replay = parse_lowered_event(
                    event,
                    &format!("examples[{index}].history[{history_index}]"),
                )?;
                if matches!(replay, LoweredCausalEvent::Target { .. }) {
                    return Err(format!(
                        "examples[{index}].history[{history_index}] must replay targets with observe_target_no_score, not charged target events"
                    ));
                }
                events.push(replay);
            }
        }
        if let Some(action) = object.get("action") {
            events.push(LoweredCausalEvent::Context {
                channel: "action".to_string(),
                bytes: payload_bytes(action, &format!("examples[{index}].action"))?,
            });
        }
        let target = object
            .get("target")
            .or_else(|| object.get("percept"))
            .ok_or_else(|| format!("examples[{index}] requires 'target' or 'percept'"))?;
        let weight = object
            .get("weight")
            .map(|raw| {
                raw.as_f64()
                    .filter(|value| value.is_finite() && *value > 0.0)
                    .ok_or_else(|| format!("examples[{index}].weight must be finite and > 0"))
            })
            .transpose()?
            .unwrap_or(1.0);
        events.push(LoweredCausalEvent::Target {
            channel: required_nonempty_string_field(
                object.get("channel"),
                &format!("examples[{index}].channel"),
            )?,
            domain: required_nonempty_string_field(
                object.get("domain"),
                &format!("examples[{index}].domain"),
            )?,
            bytes: payload_bytes(target, &format!("examples[{index}].target"))?,
            weight,
        });
    }
    lowered_dataset_from_events(
        DatasetKind::CausalPrefixDataset,
        ObjectiveTarget::InteractiveCausalAc,
        CAUSAL_PREFIX_LOWERING_VERSION,
        path,
        source_size_bytes,
        value,
        header_profile,
        domain_supports,
        events,
        "weighted-target-bytes-sum",
    )
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum CausalPayloadKind {
    Events,
    ExamplesOrPrefixes,
}

fn parse_causal_header_profile(
    value: &Value,
    label: &str,
    payload_kind: CausalPayloadKind,
) -> Result<(CausalHeaderProfile, BTreeMap<String, CausalTargetDomain>), String> {
    let object = value
        .as_object()
        .ok_or_else(|| format!("{label} must be a JSON object"))?;
    let allowed = match payload_kind {
        CausalPayloadKind::Events => [
            "schema_version",
            "environment_id",
            "environment_config_crc32",
            "codec_hash",
            "reset_convention",
            "action_alphabet",
            "percept_schema",
            "reward_encoding",
            "terminal_encoding",
            "collection_policy",
            "target_domains",
            "event_grammar",
            "events",
        ]
        .as_slice(),
        CausalPayloadKind::ExamplesOrPrefixes => [
            "schema_version",
            "environment_id",
            "environment_config_crc32",
            "codec_hash",
            "reset_convention",
            "action_alphabet",
            "percept_schema",
            "reward_encoding",
            "terminal_encoding",
            "collection_policy",
            "target_domains",
            "event_grammar",
            "examples",
            "prefixes",
        ]
        .as_slice(),
    };
    ensure_known_fields_in_object(object, allowed, label)?;

    let schema_version = object
        .get("schema_version")
        .and_then(Value::as_u64)
        .ok_or_else(|| format!("{label} requires schema_version: 1"))?;
    if schema_version != 1 {
        return Err(format!("{label} schema_version must be 1"));
    }
    let _environment_id =
        required_nonempty_string_field(object.get("environment_id"), "environment_id")?;
    let env_crc32 = required_nonempty_string_field(
        object.get("environment_config_crc32"),
        "environment_config_crc32",
    )?;
    if !is_lower_hex_crc32(&env_crc32) {
        return Err(format!(
            "{label}.environment_config_crc32 must be an 8-character lowercase hex CRC32"
        ));
    }
    let _codec_hash = required_nonempty_string_field(object.get("codec_hash"), "codec_hash")?;
    let reset_convention =
        required_nonempty_string_field(object.get("reset_convention"), "reset_convention")?;
    if reset_convention != "reset-before-episode" {
        return Err(format!(
            "{label}.reset_convention must be 'reset-before-episode'"
        ));
    }

    let domains = parse_causal_target_domains(
        object
            .get("target_domains")
            .ok_or_else(|| format!("{label} requires target_domains"))?,
    )?;
    let action_alphabet_size = parse_action_alphabet_header(
        object
            .get("action_alphabet")
            .ok_or_else(|| format!("{label} requires action_alphabet"))?,
    )?;
    let percept_channels = parse_percept_schema_header(
        object
            .get("percept_schema")
            .ok_or_else(|| format!("{label} requires percept_schema"))?,
    )?;
    let reward_channel = parse_single_encoding_channel_header(
        object
            .get("reward_encoding")
            .ok_or_else(|| format!("{label} requires reward_encoding"))?,
        "reward_encoding",
    )?;
    let terminal_channel = parse_single_encoding_channel_header(
        object
            .get("terminal_encoding")
            .ok_or_else(|| format!("{label} requires terminal_encoding"))?,
        "terminal_encoding",
    )?;
    let collection_policy =
        required_nonempty_string_field(object.get("collection_policy"), "collection_policy")?;
    let event_grammar = parse_event_grammar_header(
        object
            .get("event_grammar")
            .ok_or_else(|| format!("{label} requires event_grammar"))?,
    )?;

    validate_header_profile_consistency(
        &domains,
        &event_grammar,
        &percept_channels,
        &reward_channel,
        &terminal_channel,
    )?;

    if payload_kind == CausalPayloadKind::ExamplesOrPrefixes
        && object.get("examples").is_some()
        && object.get("prefixes").is_some()
    {
        return Err(format!(
            "{label} must declare exactly one of 'examples' or 'prefixes', not both"
        ));
    }

    let profile_hash = causal_header_profile_hash(
        action_alphabet_size,
        &collection_policy,
        &percept_channels,
        &reward_channel,
        &terminal_channel,
        &event_grammar,
    )?;
    Ok((
        CausalHeaderProfile {
            action_alphabet_size,
            collection_policy,
            percept_channels,
            reward_channel,
            terminal_channel,
            event_grammar,
            profile_hash,
        },
        domains,
    ))
}

fn ensure_known_fields_in_object(
    object: &serde_json::Map<String, Value>,
    allowed: &[&str],
    label: &str,
) -> Result<(), String> {
    for key in object.keys() {
        if !allowed.contains(&key.as_str()) {
            return Err(format!("{label} contains unknown field '{key}'"));
        }
    }
    Ok(())
}

fn parse_action_alphabet_header(value: &Value) -> Result<usize, String> {
    let object = value
        .as_object()
        .ok_or_else(|| "action_alphabet must be an object".to_string())?;
    ensure_known_fields_in_object(object, &["size"], "action_alphabet")?;
    let size = object
        .get("size")
        .and_then(Value::as_u64)
        .ok_or_else(|| "action_alphabet.size must be an integer".to_string())?;
    let size = usize::try_from(size).map_err(|_| "action_alphabet.size does not fit usize")?;
    if size == 0 {
        return Err("action_alphabet.size must be >= 1".to_string());
    }
    Ok(size)
}

fn parse_percept_schema_header(value: &Value) -> Result<BTreeSet<CausalChannelDomain>, String> {
    let object = value
        .as_object()
        .ok_or_else(|| "percept_schema must be an object".to_string())?;
    ensure_known_fields_in_object(object, &["encoding", "channels"], "percept_schema")?;
    let encoding = object
        .get("encoding")
        .and_then(Value::as_str)
        .ok_or_else(|| "percept_schema.encoding is required".to_string())?;
    if encoding != "bytes" {
        return Err("percept_schema.encoding must be 'bytes'".to_string());
    }
    let channels = object
        .get("channels")
        .and_then(Value::as_array)
        .ok_or_else(|| "percept_schema.channels must be an array".to_string())?;
    if channels.is_empty() {
        return Err("percept_schema.channels must contain at least one entry".to_string());
    }
    let mut out = BTreeSet::<CausalChannelDomain>::new();
    for (index, value) in channels.iter().enumerate() {
        let pair = parse_channel_domain_pair(value, &format!("percept_schema.channels[{index}]"))?;
        if !out.insert(pair.clone()) {
            return Err(format!(
                "percept_schema.channels[{index}] duplicates ({}, {})",
                pair.channel, pair.domain
            ));
        }
    }
    Ok(out)
}

fn parse_single_encoding_channel_header(
    value: &Value,
    label: &str,
) -> Result<CausalChannelDomain, String> {
    let object = value
        .as_object()
        .ok_or_else(|| format!("{label} must be an object"))?;
    ensure_known_fields_in_object(object, &["encoding", "channel", "domain"], label)?;
    let encoding = object
        .get("encoding")
        .and_then(Value::as_str)
        .ok_or_else(|| format!("{label}.encoding is required"))?;
    if encoding != "bytes" {
        return Err(format!("{label}.encoding must be 'bytes'"));
    }
    Ok(CausalChannelDomain {
        channel: required_nonempty_string_field(
            object.get("channel"),
            &format!("{label}.channel"),
        )?,
        domain: required_nonempty_string_field(object.get("domain"), &format!("{label}.domain"))?,
    })
}

fn parse_event_grammar_header(value: &Value) -> Result<CausalEventGrammar, String> {
    let object = value
        .as_object()
        .ok_or_else(|| "event_grammar must be an object".to_string())?;
    ensure_known_fields_in_object(
        object,
        &["context_channels", "observe_target_no_score", "target"],
        "event_grammar",
    )?;
    let context_channels = object
        .get("context_channels")
        .and_then(Value::as_array)
        .ok_or_else(|| "event_grammar.context_channels must be an array".to_string())?;
    let mut context = BTreeSet::<String>::new();
    for (index, raw) in context_channels.iter().enumerate() {
        let channel = raw
            .as_str()
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .ok_or_else(|| {
                format!(
                    "event_grammar.context_channels[{index}] must be a non-empty channel string"
                )
            })?
            .to_string();
        if !context.insert(channel.clone()) {
            return Err(format!(
                "event_grammar.context_channels[{index}] duplicates '{channel}'"
            ));
        }
    }
    let observe = parse_grammar_channel_domains(
        object
            .get("observe_target_no_score")
            .ok_or_else(|| "event_grammar.observe_target_no_score is required".to_string())?,
        "event_grammar.observe_target_no_score",
    )?;
    let target = parse_grammar_channel_domains(
        object
            .get("target")
            .ok_or_else(|| "event_grammar.target is required".to_string())?,
        "event_grammar.target",
    )?;
    Ok(CausalEventGrammar {
        context_channels: context,
        observe_target_no_score: observe,
        target,
    })
}

fn parse_grammar_channel_domains(
    value: &Value,
    label: &str,
) -> Result<BTreeSet<CausalChannelDomain>, String> {
    let array = value
        .as_array()
        .ok_or_else(|| format!("{label} must be an array"))?;
    let mut out = BTreeSet::<CausalChannelDomain>::new();
    for (index, entry) in array.iter().enumerate() {
        let pair = parse_channel_domain_pair(entry, &format!("{label}[{index}]"))?;
        if !out.insert(pair.clone()) {
            return Err(format!(
                "{label}[{index}] duplicates ({}, {})",
                pair.channel, pair.domain
            ));
        }
    }
    Ok(out)
}

fn parse_channel_domain_pair(value: &Value, label: &str) -> Result<CausalChannelDomain, String> {
    let object = value
        .as_object()
        .ok_or_else(|| format!("{label} must be an object"))?;
    ensure_known_fields_in_object(object, &["channel", "domain"], label)?;
    Ok(CausalChannelDomain {
        channel: required_nonempty_string_field(
            object.get("channel"),
            &format!("{label}.channel"),
        )?,
        domain: required_nonempty_string_field(object.get("domain"), &format!("{label}.domain"))?,
    })
}

fn validate_header_profile_consistency(
    domains: &BTreeMap<String, CausalTargetDomain>,
    event_grammar: &CausalEventGrammar,
    percept_channels: &BTreeSet<CausalChannelDomain>,
    reward_channel: &CausalChannelDomain,
    terminal_channel: &CausalChannelDomain,
) -> Result<(), String> {
    for pair in event_grammar
        .observe_target_no_score
        .iter()
        .chain(event_grammar.target.iter())
    {
        if !domains.contains_key(&pair.domain) {
            return Err(format!(
                "event_grammar references undeclared target domain '{}'",
                pair.domain
            ));
        }
    }
    for pair in percept_channels {
        if !event_grammar.target.contains(pair) {
            return Err(format!(
                "percept_schema channel/domain ({}, {}) must appear in event_grammar.target",
                pair.channel, pair.domain
            ));
        }
    }
    if !event_grammar.target.contains(reward_channel) {
        return Err(format!(
            "reward_encoding channel/domain ({}, {}) must appear in event_grammar.target",
            reward_channel.channel, reward_channel.domain
        ));
    }
    if !event_grammar.target.contains(terminal_channel) {
        return Err(format!(
            "terminal_encoding channel/domain ({}, {}) must appear in event_grammar.target",
            terminal_channel.channel, terminal_channel.domain
        ));
    }
    Ok(())
}

fn causal_header_profile_hash(
    action_alphabet_size: usize,
    collection_policy: &str,
    percept_channels: &BTreeSet<CausalChannelDomain>,
    reward_channel: &CausalChannelDomain,
    terminal_channel: &CausalChannelDomain,
    event_grammar: &CausalEventGrammar,
) -> Result<String, String> {
    let value = serde_json::json!({
        "action_alphabet_size": action_alphabet_size,
        "collection_policy": collection_policy,
        "percept_channels": percept_channels
            .iter()
            .map(|pair| {
                serde_json::json!({"channel": pair.channel.as_str(), "domain": pair.domain.as_str()})
            })
            .collect::<Vec<_>>(),
        "reward_channel": {"channel": reward_channel.channel.as_str(), "domain": reward_channel.domain.as_str()},
        "terminal_channel": {"channel": terminal_channel.channel.as_str(), "domain": terminal_channel.domain.as_str()},
        "event_grammar": {
            "context_channels": event_grammar.context_channels.iter().collect::<Vec<_>>(),
            "observe_target_no_score": event_grammar.observe_target_no_score
                .iter()
                .map(|pair| {
                    serde_json::json!({"channel": pair.channel.as_str(), "domain": pair.domain.as_str()})
                })
                .collect::<Vec<_>>(),
            "target": event_grammar.target
                .iter()
                .map(|pair| {
                    serde_json::json!({"channel": pair.channel.as_str(), "domain": pair.domain.as_str()})
                })
                .collect::<Vec<_>>(),
        }
    });
    let bytes = serde_json::to_vec(&value)
        .map_err(|err| format!("failed to encode causal header profile hash: {err}"))?;
    Ok(crc32_hex(&bytes))
}

fn parse_causal_target_domains(
    value: &Value,
) -> Result<BTreeMap<String, CausalTargetDomain>, String> {
    let object = value
        .as_object()
        .ok_or_else(|| "target_domains must be an object".to_string())?;
    if object.is_empty() {
        return Err("target_domains must declare at least one target domain".to_string());
    }
    object
        .iter()
        .map(|(domain, spec)| {
            if domain.trim().is_empty() {
                return Err("target_domains contains an empty domain tag".to_string());
            }
            let spec_object = spec
                .as_object()
                .ok_or_else(|| format!("target_domains.{domain} must be an object"))?;
            let kind = spec_object
                .get("kind")
                .and_then(Value::as_str)
                .ok_or_else(|| format!("target_domains.{domain}.kind is required"))?;
            match kind {
                "byte_alphabet" => {
                    ensure_known_causal_domain_fields(
                        spec_object,
                        &["kind"],
                        &format!("target_domains.{domain}"),
                    )?;
                    Ok((domain.clone(), CausalTargetDomain::ByteAlphabet))
                }
                "enumerated_payloads" => {
                    ensure_known_causal_domain_fields(
                        spec_object,
                        &["kind", "payloads"],
                        &format!("target_domains.{domain}"),
                    )?;
                    let payloads_value = spec_object
                        .get("payloads")
                        .and_then(Value::as_array)
                        .ok_or_else(|| {
                            format!("target_domains.{domain}.payloads must be an array")
                        })?;
                    if payloads_value.is_empty() {
                        return Err(format!(
                            "target_domains.{domain}.payloads must contain at least one payload"
                        ));
                    }
                    let mut seen = BTreeSet::<Vec<u8>>::new();
                    let mut payloads = Vec::<Vec<u8>>::with_capacity(payloads_value.len());
                    for (index, payload) in payloads_value.iter().enumerate() {
                        let bytes = payload_bytes(
                            payload,
                            &format!("target_domains.{domain}.payloads[{index}]"),
                        )?;
                        if !seen.insert(bytes.clone()) {
                            return Err(format!(
                                "target_domains.{domain}.payloads[{index}] duplicates an enumerated payload"
                            ));
                        }
                        payloads.push(bytes);
                    }
                    Ok((domain.clone(), CausalTargetDomain::EnumeratedPayloads { payloads }))
                }
                other => Err(format!(
                    "target_domains.{domain}.kind has unknown target-domain support kind '{other}'"
                )),
            }
        })
        .collect()
}

fn ensure_known_causal_domain_fields(
    object: &serde_json::Map<String, Value>,
    allowed: &[&str],
    label: &str,
) -> Result<(), String> {
    for key in object.keys() {
        if !allowed.contains(&key.as_str()) {
            return Err(format!(
                "{label} contains unknown target-domain field '{key}'"
            ));
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn lowered_dataset_from_events(
    kind: DatasetKind,
    objective_target: ObjectiveTarget,
    lowering_version: &'static str,
    path: &Path,
    source_size_bytes: usize,
    source_value: &Value,
    header_profile: CausalHeaderProfile,
    domain_supports: BTreeMap<String, CausalTargetDomain>,
    events: Vec<LoweredCausalEvent>,
    target_size_function: &'static str,
) -> Result<LoadedDataset, String> {
    let canonical_bytes = serde_json::to_vec(source_value)
        .map_err(|err| format!("failed to canonicalize causal dataset JSON: {err}"))?;
    let canonical_content_hash = crc32_hex(&canonical_bytes);
    let normalized_events = expand_byte_alphabet_events(events, &domain_supports)?;
    validate_events_against_causal_profile(&normalized_events, &domain_supports, &header_profile)?;
    let mut charged = Vec::<u8>::new();
    let mut target_events = 0usize;
    let mut dataset_units = 0.0f64;
    for event in &normalized_events {
        if let LoweredCausalEvent::Target { bytes, weight, .. } = event {
            charged.extend_from_slice(bytes);
            target_events = target_events.saturating_add(1);
            dataset_units += (*weight) * (bytes.len() as f64);
        }
    }
    let skeleton_bytes = lowered_event_skeleton_bytes(&normalized_events)?;
    let domain_support_bytes = causal_domain_support_bytes(&domain_supports)?;
    let domain_support_hash = crc32_hex(&domain_support_bytes);
    let channel_set = causal_channel_set(&normalized_events);
    Ok(LoadedDataset {
        kind,
        objective_target,
        lowering_version,
        codec_hash: causal_dataset_string_field(source_value, "codec_hash")
            .unwrap_or_else(|| "json-causal-byte-events-v1".to_string()),
        event_grammar_hash: crc32_hex(&skeleton_bytes),
        target_domain_support_hash: domain_support_hash.clone(),
        causal_header_profile_hash: header_profile.profile_hash.clone(),
        target_size_function,
        canonical_content_hash,
        lowered_skeleton_hash: crc32_hex(&skeleton_bytes),
        resolved_path: path.to_string_lossy().to_string(),
        source_size_bytes,
        raw_bytes: charged,
        events: normalized_events,
        causal_profile: Some(CausalEvaluationProfile {
            domains: domain_supports,
            channel_set,
            domain_support_hash,
            byte_alphabet_symbol_width: 1,
            header_profile_hash: header_profile.profile_hash,
            event_grammar: header_profile.event_grammar,
            action_alphabet_size: header_profile.action_alphabet_size,
            collection_policy: header_profile.collection_policy,
            percept_channels: header_profile.percept_channels,
            reward_channel: header_profile.reward_channel,
            terminal_channel: header_profile.terminal_channel,
        }),
        dataset_units,
        target_events,
    })
}

fn parse_lowered_event(value: &Value, label: &str) -> Result<LoweredCausalEvent, String> {
    let object = value
        .as_object()
        .ok_or_else(|| format!("{label} must be an object"))?;
    let kind = object
        .get("kind")
        .and_then(Value::as_str)
        .ok_or_else(|| format!("{label}.kind is required"))?;
    match kind {
        "reset" => {
            ensure_known_event_fields(object, &["kind"], label)?;
            Ok(LoweredCausalEvent::Reset)
        }
        "context" => {
            ensure_known_event_fields(object, &["kind", "channel", "bytes"], label)?;
            Ok(LoweredCausalEvent::Context {
                channel: required_nonempty_string_field(
                    object.get("channel"),
                    &format!("{label}.channel"),
                )?,
                bytes: event_payload_bytes(value, label)?,
            })
        }
        "observe_target_no_score" => {
            ensure_known_event_fields(object, &["kind", "channel", "domain", "bytes"], label)?;
            Ok(LoweredCausalEvent::ObserveTargetNoScore {
                channel: required_nonempty_string_field(
                    object.get("channel"),
                    &format!("{label}.channel"),
                )?,
                domain: required_nonempty_string_field(
                    object.get("domain"),
                    &format!("{label}.domain"),
                )?,
                bytes: event_payload_bytes(value, label)?,
            })
        }
        "target" => {
            ensure_known_event_fields(
                object,
                &["kind", "channel", "domain", "bytes", "weight"],
                label,
            )?;
            let channel =
                required_nonempty_string_field(object.get("channel"), &format!("{label}.channel"))?;
            let domain =
                required_nonempty_string_field(object.get("domain"), &format!("{label}.domain"))?;
            let weight = object
                .get("weight")
                .map(|raw| {
                    raw.as_f64()
                        .filter(|value| value.is_finite() && *value > 0.0)
                        .ok_or_else(|| format!("{label}.weight must be finite and > 0"))
                })
                .transpose()?
                .unwrap_or(1.0);
            Ok(LoweredCausalEvent::Target {
                channel,
                domain,
                bytes: event_payload_bytes(value, label)?,
                weight,
            })
        }
        other => Err(format!(
            "{label}.kind has unknown causal event kind '{other}'"
        )),
    }
}

fn event_payload_bytes(value: &Value, label: &str) -> Result<Vec<u8>, String> {
    let object = value
        .as_object()
        .ok_or_else(|| format!("{label} must be an object"))?;
    let payload = object
        .get("bytes")
        .ok_or_else(|| format!("{label}.bytes is required"))?;
    payload_bytes(payload, &format!("{label}.bytes"))
}

fn payload_bytes(value: &Value, label: &str) -> Result<Vec<u8>, String> {
    match value {
        Value::String(text) => Ok(text.as_bytes().to_vec()),
        Value::Array(items) => items
            .iter()
            .enumerate()
            .map(|(index, item)| {
                let byte = item
                    .as_u64()
                    .ok_or_else(|| format!("{label}[{index}] must be an integer byte"))?;
                u8::try_from(byte).map_err(|_| format!("{label}[{index}] must be in 0..=255"))
            })
            .collect(),
        _ => Err(format!(
            "{label} must be either a byte string or an array of integer bytes"
        )),
    }
}

fn required_nonempty_string_field(value: Option<&Value>, label: &str) -> Result<String, String> {
    let text = value
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|text| !text.is_empty())
        .ok_or_else(|| format!("{label} is required"))?;
    Ok(text.to_string())
}

fn ensure_known_event_fields(
    object: &serde_json::Map<String, Value>,
    allowed: &[&str],
    label: &str,
) -> Result<(), String> {
    for key in object.keys() {
        if !allowed.contains(&key.as_str()) {
            return Err(format!("{label} contains unknown event field '{key}'"));
        }
    }
    Ok(())
}

fn is_lower_hex_crc32(value: &str) -> bool {
    value.len() == 8
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn lowered_event_skeleton_bytes(events: &[LoweredCausalEvent]) -> Result<Vec<u8>, String> {
    let skeleton = events
        .iter()
        .map(|event| match event {
            LoweredCausalEvent::Reset => serde_json::json!({"kind": "reset"}),
            LoweredCausalEvent::Context { channel, bytes } => serde_json::json!({
                "kind": "context",
                "channel": channel,
                "bytes_len": bytes.len(),
            }),
            LoweredCausalEvent::ObserveTargetNoScore {
                channel,
                domain,
                bytes,
            } => serde_json::json!({
                "kind": "observe_target_no_score",
                "channel": channel,
                "domain": domain,
                "bytes_len": bytes.len(),
            }),
            LoweredCausalEvent::Target {
                channel,
                domain,
                bytes,
                weight,
            } => serde_json::json!({
                "kind": "target",
                "channel": channel,
                "domain": domain,
                "bytes_len": bytes.len(),
                "weight_bits": weight.to_bits(),
            }),
        })
        .collect::<Vec<_>>();
    serde_json::to_vec(&skeleton)
        .map_err(|err| format!("failed to serialize lowered event skeleton: {err}"))
}

fn causal_domain_support_bytes(
    domains: &BTreeMap<String, CausalTargetDomain>,
) -> Result<Vec<u8>, String> {
    let value = domains
        .iter()
        .map(|(domain, support)| match support {
            CausalTargetDomain::ByteAlphabet => serde_json::json!({
                "domain": domain,
                "kind": "byte_alphabet",
                "symbol_width_bytes": 1usize,
                "symbols": 256usize,
            }),
            CausalTargetDomain::EnumeratedPayloads { payloads } => serde_json::json!({
                "domain": domain,
                "kind": "enumerated_payloads",
                "payloads": payloads,
            }),
        })
        .collect::<Vec<_>>();
    serde_json::to_vec(&value)
        .map_err(|err| format!("failed to serialize causal target-domain supports: {err}"))
}

fn causal_channel_set(events: &[LoweredCausalEvent]) -> BTreeSet<String> {
    let mut channels = BTreeSet::<String>::new();
    for event in events {
        match event {
            LoweredCausalEvent::Reset => {}
            LoweredCausalEvent::Context { channel, .. }
            | LoweredCausalEvent::ObserveTargetNoScore { channel, .. }
            | LoweredCausalEvent::Target { channel, .. } => {
                channels.insert(channel.clone());
            }
        }
    }
    channels
}

fn validate_events_against_causal_profile(
    events: &[LoweredCausalEvent],
    domains: &BTreeMap<String, CausalTargetDomain>,
    header: &CausalHeaderProfile,
) -> Result<(), String> {
    if header.action_alphabet_size > 256 && header.event_grammar.context_channels.contains("action")
    {
        return Err(
            "event_grammar contains action context but action_alphabet.size exceeds byte encoding capacity (must be <= 256)"
                .to_string(),
        );
    }
    for event in events {
        match event {
            LoweredCausalEvent::Reset => {}
            LoweredCausalEvent::Context { channel, bytes } => {
                if !header.event_grammar.context_channels.contains(channel) {
                    return Err(format!(
                        "causal context event channel '{channel}' is not declared in event_grammar.context_channels"
                    ));
                }
                if channel == "action" {
                    if bytes.len() != 1 {
                        return Err(
                            "action context payload must encode exactly one byte".to_string()
                        );
                    }
                    if bytes[0] as usize >= header.action_alphabet_size {
                        return Err(format!(
                            "action context value {} is outside action_alphabet.size={}",
                            bytes[0], header.action_alphabet_size
                        ));
                    }
                }
            }
            LoweredCausalEvent::ObserveTargetNoScore { domain, bytes, .. }
            | LoweredCausalEvent::Target { domain, bytes, .. } => {
                let support = domains.get(domain).ok_or_else(|| {
                    format!("causal event references undeclared target domain '{domain}'")
                })?;
                let descriptor = match event {
                    LoweredCausalEvent::ObserveTargetNoScore {
                        channel, domain, ..
                    } => CausalChannelDomain {
                        channel: channel.clone(),
                        domain: domain.clone(),
                    },
                    LoweredCausalEvent::Target {
                        channel, domain, ..
                    } => CausalChannelDomain {
                        channel: channel.clone(),
                        domain: domain.clone(),
                    },
                    _ => unreachable!(),
                };
                match event {
                    LoweredCausalEvent::ObserveTargetNoScore { .. } => {
                        if !header
                            .event_grammar
                            .observe_target_no_score
                            .contains(&descriptor)
                        {
                            return Err(format!(
                                "observe_target_no_score ({}, {}) is not declared in event_grammar.observe_target_no_score",
                                descriptor.channel, descriptor.domain
                            ));
                        }
                    }
                    LoweredCausalEvent::Target { .. } => {
                        if !header.event_grammar.target.contains(&descriptor) {
                            return Err(format!(
                                "target ({}, {}) is not declared in event_grammar.target",
                                descriptor.channel, descriptor.domain
                            ));
                        }
                    }
                    _ => {}
                }
                if !causal_support_contains(support, bytes) {
                    return Err(format!(
                        "causal event payload is outside target-domain support '{domain}'"
                    ));
                }
            }
        }
    }
    Ok(())
}

fn causal_support_contains(support: &CausalTargetDomain, bytes: &[u8]) -> bool {
    match support {
        CausalTargetDomain::ByteAlphabet => bytes.len() == 1,
        CausalTargetDomain::EnumeratedPayloads { payloads } => {
            payloads.iter().any(|payload| payload == bytes)
        }
    }
}

fn expand_byte_alphabet_events(
    events: Vec<LoweredCausalEvent>,
    domains: &BTreeMap<String, CausalTargetDomain>,
) -> Result<Vec<LoweredCausalEvent>, String> {
    let mut expanded = Vec::<LoweredCausalEvent>::new();
    for event in events {
        match event {
            LoweredCausalEvent::Reset | LoweredCausalEvent::Context { .. } => expanded.push(event),
            LoweredCausalEvent::ObserveTargetNoScore {
                channel,
                domain,
                bytes,
            } => {
                let Some(support) = domains.get(&domain) else {
                    return Err(format!(
                        "causal event references undeclared target domain '{domain}'"
                    ));
                };
                match support {
                    CausalTargetDomain::ByteAlphabet => {
                        if bytes.is_empty() {
                            return Err(
                                "byte_alphabet payloads must contain at least one byte".to_string()
                            );
                        }
                        for byte in bytes {
                            expanded.push(LoweredCausalEvent::ObserveTargetNoScore {
                                channel: channel.clone(),
                                domain: domain.clone(),
                                bytes: vec![byte],
                            });
                        }
                    }
                    CausalTargetDomain::EnumeratedPayloads { .. } => {
                        expanded.push(LoweredCausalEvent::ObserveTargetNoScore {
                            channel,
                            domain,
                            bytes,
                        });
                    }
                }
            }
            LoweredCausalEvent::Target {
                channel,
                domain,
                bytes,
                weight,
            } => {
                let Some(support) = domains.get(&domain) else {
                    return Err(format!(
                        "causal event references undeclared target domain '{domain}'"
                    ));
                };
                match support {
                    CausalTargetDomain::ByteAlphabet => {
                        if bytes.is_empty() {
                            return Err(
                                "byte_alphabet payloads must contain at least one byte".to_string()
                            );
                        }
                        for byte in bytes {
                            expanded.push(LoweredCausalEvent::Target {
                                channel: channel.clone(),
                                domain: domain.clone(),
                                bytes: vec![byte],
                                weight,
                            });
                        }
                    }
                    CausalTargetDomain::EnumeratedPayloads { .. } => {
                        expanded.push(LoweredCausalEvent::Target {
                            channel,
                            domain,
                            bytes,
                            weight,
                        });
                    }
                }
            }
        }
    }
    Ok(expanded)
}

fn causal_dataset_string_field(value: &Value, field: &str) -> Option<String> {
    value
        .get(field)
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|text| !text.is_empty())
        .map(ToOwned::to_owned)
}

fn planner_controller_contract(
    controller: &crate::spec::CompiledTuneController,
    compiled: &crate::spec::CompiledTuneSpec,
    dataset: &LoadedDataset,
    verified_theorem: &VerifiedTheoremInputs,
) -> Result<PlannerControllerContract, String> {
    match controller {
        crate::spec::CompiledTuneController::McAixiFacCtw(inner) => Ok(PlannerControllerContract {
            interface: inner.interface.clone(),
            planner_simulations_per_step: inner.planner_simulations_per_step,
            return_horizon: None,
            label_phase_period: None,
            discount_factor: 1.0,
            reward_semantics: PlannerRewardSemantics::ExactObjectiveDifference,
            clipping_interval: None,
            teacher: None,
            warmstart_self_improvement: false,
        }),
        crate::spec::CompiledTuneController::AiqiDiscounted(inner) => {
            if inner.max_improvement <= inner.min_improvement {
                return Err(
                    "normalized clipped reward contract requires max_improvement > min_improvement"
                        .to_string(),
                );
            }
            Ok(PlannerControllerContract {
                interface: inner.interface.clone(),
                planner_simulations_per_step: inner.planner_simulations_per_step,
                return_horizon: Some(inner.return_horizon),
                label_phase_period: None,
                discount_factor: inner.discount_factor,
                reward_semantics: PlannerRewardSemantics::NormalizedClippedImprovement,
                clipping_interval: Some((inner.min_improvement, inner.max_improvement)),
                teacher: None,
                warmstart_self_improvement: false,
            })
        }
        crate::spec::CompiledTuneController::AiqiWarmstartExactJh(inner) => {
            let reward_certificate = verified_theorem.exact_reward_encoding.as_ref().ok_or_else(|| {
                "reward_encoding_unsafe: warm-start exact-J_H requires a verified exact_reward_encoding_certificate"
                    .to_string()
            })?;
            if !reward_certificate.is_identity_or_interval_encoding() {
                return Err(
                    "reward_encoding_unsafe: warm-start exact-J_H cannot use a non-identity finite_reward_map until teacher/live traces carry objective-difference labels or a verified decoder-based return encoder"
                        .to_string(),
                );
            }
            let teacher = load_warmstart_teacher_dataset(
                compiled,
                dataset,
                verified_theorem,
                &inner.warmstart_teacher_dataset_asset,
                &inner.interface,
                inner.return_horizon,
                inner.label_phase_period,
            )?;
            Ok(PlannerControllerContract {
                interface: inner.interface.clone(),
                planner_simulations_per_step: inner.planner_simulations_per_step,
                return_horizon: Some(inner.return_horizon),
                label_phase_period: Some(inner.label_phase_period),
                discount_factor: 1.0,
                reward_semantics: PlannerRewardSemantics::ExactObjectiveDifference,
                clipping_interval: None,
                teacher: Some(teacher),
                warmstart_self_improvement: true,
            })
        }
        crate::spec::CompiledTuneController::AnnealedHillClimbing(_) => {
            Err("annealed controller does not use planner-family contract".to_string())
        }
    }
}

fn load_warmstart_teacher_dataset(
    compiled: &crate::spec::CompiledTuneSpec,
    dataset: &LoadedDataset,
    verified_theorem: &VerifiedTheoremInputs,
    asset_id: &str,
    interface: &crate::spec::TunePlannerInterfaceSpec,
    return_horizon: usize,
    label_phase_period: usize,
) -> Result<WarmstartTeacherDataset, String> {
    if asset_id == compiled.canonical_spec().input_asset {
        return Err(
            "warmstart_teacher_dataset_asset must be distinct from input_asset".to_string(),
        );
    }
    let binding = compiled
        .resolved_assets()
        .iter()
        .find(|entry| entry.id == asset_id)
        .ok_or_else(|| format!("unknown warmstart teacher dataset asset '{asset_id}'"))?;
    let AssetRef::Filesystem(path) = &binding.asset;
    let bytes = fs::read(path).map_err(|err| {
        format!(
            "failed to read warmstart_teacher_dataset_asset '{}': {err}",
            path.display()
        )
    })?;
    let hash = crc32_hex(&bytes);
    let traces = WarmStartExactJhTeacherDataset::from_json_slice(&bytes)
        .map_err(|err| format!("invalid warmstart_teacher_dataset_asset: {err}"))?;
    validate_warmstart_teacher_contract(
        compiled,
        dataset,
        verified_theorem,
        interface,
        return_horizon,
        label_phase_period,
        &traces,
    )?;
    let records = traces
        .traces
        .iter()
        .map(|trace| trace.transitions.len())
        .sum();
    Ok(WarmstartTeacherDataset {
        asset_id: asset_id.to_string(),
        resolved_path: path.to_string_lossy().to_string(),
        content_hash: hash,
        records,
        traces,
    })
}

fn validate_warmstart_teacher_contract(
    compiled: &crate::spec::CompiledTuneSpec,
    dataset: &LoadedDataset,
    verified_theorem: &VerifiedTheoremInputs,
    interface: &crate::spec::TunePlannerInterfaceSpec,
    return_horizon: usize,
    label_phase_period: usize,
    teacher: &WarmStartExactJhTeacherDataset,
) -> Result<(), String> {
    let contract = &teacher.contract;
    if contract.schema_version != 1 {
        return Err("warmstart teacher schema_version must be 1".to_string());
    }
    let task_fingerprint = warmstart_task_fingerprint(compiled, dataset, verified_theorem)?;
    if contract.task_fingerprint != task_fingerprint {
        return Err(format!(
            "warmstart teacher task_fingerprint '{}' does not match current task '{}'",
            contract.task_fingerprint, task_fingerprint
        ));
    }
    if contract.action_alphabet_size != interface.agent_actions.get()
        || contract.observation_bits != interface.observation_bits
        || contract.observation_stream_len != interface.observation_stream_len.max(1)
        || contract.observation_key_mode
            != observation_key_mode_name(interface.observation_key_mode)
        || contract.reward_bits != interface.reward_bits
        || contract.return_horizon != return_horizon
        || contract.label_phase_period != label_phase_period
    {
        return Err(
            "warmstart teacher planner interface fingerprint does not match current controller"
                .to_string(),
        );
    }
    let expected_adapter_ref = OBSERVATION_ADAPTER_DECLARATION;
    let expected_adapter_hash = observation_adapter_content_hash()?;
    if contract.observation_adapter_spec_ref != expected_adapter_ref
        || contract.observation_adapter_content_crc32 != expected_adapter_hash
    {
        return Err(
            "warmstart teacher observation adapter fingerprint does not match current tuner observation adapter"
                .to_string(),
        );
    }
    let reward_cert = verified_theorem
        .exact_reward_encoding
        .as_ref()
        .ok_or_else(|| {
            "warmstart exact-J_H requires a verified exact_reward_encoding_certificate".to_string()
        })?;
    if contract.min_reward != 0
        || contract.max_reward != reward_cert.max_reward
        || contract.scalar_representation != reward_cert.scalar_representation
        || contract.exact_reward_encoding_certificate != reward_cert.base.content_hash
    {
        return Err("warmstart teacher reward/scalar fingerprint does not match verified exact reward encoder".to_string());
    }
    Ok(())
}

fn warmstart_task_fingerprint(
    compiled: &crate::spec::CompiledTuneSpec,
    dataset: &LoadedDataset,
    verified_theorem: &VerifiedTheoremInputs,
) -> Result<String, String> {
    let reward_hash = verified_theorem
        .exact_reward_encoding
        .as_ref()
        .map(|cert| cert.base.content_hash.as_str())
        .unwrap_or("unverified");
    let payload = serde_json::json!({
        "tune_canonical_crc32": crc32_hex(compiled.canonical_bytes().as_slice()),
        "dataset_crc32": dataset.canonical_content_hash,
        "dataset_kind": dataset_kind_name(dataset.kind),
        "bounds_crc32": bounds_hash(&compiled.canonical_spec().bounds)?,
        "controller_kind": controller_kind_name(compiled.controller()),
        "reward_certificate_crc32": reward_hash,
    });
    serde_json::to_vec(&payload)
        .map(|bytes| crc32_hex(&bytes))
        .map_err(|err| format!("failed to encode warmstart task fingerprint: {err}"))
}

fn compile_planner_mutation_actions(
    baseline: crate::api::CompressionBackend,
    bounds: &crate::spec::TuneBoundsSpec,
) -> Result<Vec<PlannerMutationAction>, String> {
    let json = crate::spec::compression_backend_to_json_value(&baseline)
        .map_err(|err| format!("failed to serialize baseline for action compilation: {err}"))?;
    let range_map = bounds
        .parameter_ranges
        .iter()
        .map(|range| (range.parameter.clone(), (range.min, range.max)))
        .collect::<BTreeMap<_, _>>();
    let mut leaves = collect_numeric_leaves(&json);
    if !range_map.is_empty() {
        leaves.retain(|leaf| range_map.contains_key(&leaf.path));
    }
    let mut actions = Vec::<PlannerMutationAction>::new();
    for leaf in leaves {
        let deltas = match leaf.kind {
            NumericKind::Unsigned | NumericKind::Signed => [-1.0, 1.0],
            NumericKind::Float => [-0.05, 0.05],
        };
        for delta in deltas {
            actions.push(PlannerMutationAction::NumericStep {
                path: leaf.path.clone(),
                pointer: leaf.pointer.clone(),
                kind: leaf.kind,
                delta,
            });
        }
    }
    if actions.is_empty() {
        actions.push(PlannerMutationAction::Noop);
    }
    Ok(actions)
}

fn apply_planner_mutation_action(
    candidate: &crate::api::CompressionBackend,
    action: &PlannerMutationAction,
) -> Result<Option<crate::api::CompressionBackend>, String> {
    let PlannerMutationAction::NumericStep {
        path: _,
        pointer,
        kind,
        delta,
    } = action
    else {
        return Ok(None);
    };
    let mut json = crate::spec::compression_backend_to_json_value(candidate)
        .map_err(|err| format!("failed to serialize candidate for planner action: {err}"))?;
    let Some(slot) = json.pointer_mut(pointer) else {
        return Ok(None);
    };
    if !apply_numeric_delta(slot, *kind, *delta) {
        return Ok(None);
    }
    let mutated = crate::spec::parse_compression_backend_json(
        &json,
        Path::new("."),
        None,
        crate::compression::FramingMode::Framed,
    )
    .map_err(|err| format!("planner action produced unparsable candidate: {err}"))?;
    Ok(Some(mutated))
}

fn apply_numeric_delta(slot: &mut Value, kind: NumericKind, delta: f64) -> bool {
    match kind {
        NumericKind::Unsigned => {
            let Some(current) = slot.as_u64() else {
                return false;
            };
            let next = if delta >= 0.0 {
                current.saturating_add(delta.abs().ceil() as u64)
            } else {
                current.saturating_sub(delta.abs().ceil() as u64)
            };
            if next == current {
                return false;
            }
            *slot = Value::Number(serde_json::Number::from(next));
            true
        }
        NumericKind::Signed => {
            let Some(current) = slot.as_i64() else {
                return false;
            };
            let step = delta.abs().ceil() as i64;
            let next = if delta >= 0.0 {
                current.saturating_add(step)
            } else {
                current.saturating_sub(step)
            };
            if next == current {
                return false;
            }
            *slot = Value::Number(serde_json::Number::from(next));
            true
        }
        NumericKind::Float => {
            let Some(current) = slot.as_f64() else {
                return false;
            };
            let next = current + current.abs().max(1.0) * delta;
            if !next.is_finite() || (next - current).abs() <= f64::EPSILON {
                return false;
            }
            if let Some(number) = serde_json::Number::from_f64(next) {
                *slot = Value::Number(number);
                true
            } else {
                false
            }
        }
    }
}

fn planner_action_paths(actions: &[PlannerMutationAction]) -> Vec<String> {
    actions
        .iter()
        .map(|action| match action {
            PlannerMutationAction::NumericStep { path, delta, .. } => {
                format!("{path}:{delta:+}")
            }
            PlannerMutationAction::Noop => "noop".to_string(),
        })
        .collect()
}

fn validate_theorem_planner_mutation_domain(
    actions: &[PlannerMutationAction],
    theorem: &TuneTheoremConfig,
) -> Result<(), String> {
    if !theorem_requests_exact_claims(theorem) {
        return Ok(());
    }
    if let Some(path) = actions.iter().find_map(|action| match action {
        PlannerMutationAction::NumericStep {
            path,
            kind: NumericKind::Float,
            ..
        } => Some(path.as_str()),
        _ => None,
    }) {
        return Err(format!(
            "theorem_finite_state_unsafe: planner mutation action '{path}' targets a floating-point leaf; exact theorem claims require an integer finite mutation grammar or a future certificate-enumerated finite float domain"
        ));
    }
    Ok(())
}

fn theorem_requests_exact_claims(theorem: &TuneTheoremConfig) -> bool {
    theorem.claim_exact_finite_mdp
        || theorem.claim_exact_observed_markov
        || theorem.claim_planner_convergence
}

fn compile_tuner_planner_run_spec(
    controller: &crate::spec::CompiledTuneController,
    contract: &PlannerControllerContract,
    reward_encoder: &TunerRewardEncoder,
    compiled: &crate::spec::CompiledTuneSpec,
) -> Result<CompiledPlannerRunSpec, String> {
    let interface = PlannerInterfaceSpec {
        observation_bits: contract.interface.observation_bits,
        observation_stream_len: contract.interface.observation_stream_len,
        observation_key_mode: contract.interface.observation_key_mode,
        reward_bits: contract.interface.reward_bits,
        agent_actions: contract.interface.agent_actions,
        min_reward: reward_encoder.min_reward(),
        max_reward: reward_encoder.max_reward(),
        reward_offset: reward_encoder.reward_offset(),
    };
    let percept_bits = interface
        .observation_bits
        .saturating_mul(interface.observation_stream_len.max(1))
        .saturating_add(interface.reward_bits)
        .max(1);
    let controller_spec = match controller {
        crate::spec::CompiledTuneController::McAixiFacCtw(inner) => {
            ControllerSpec::McAixi(McAixiControllerSpec {
                predictor: RateBackend::FacCtw {
                    base_depth: TUNER_MCAIXI_FAC_CTW_BASE_DEPTH,
                    num_percept_bits: percept_bits,
                    encoding_bits: 1,
                },
                agent_horizon: TUNER_MCAIXI_HORIZON,
                num_simulations: inner.planner_simulations_per_step,
                mcts_strategy: MctsStrategy::RhoUct,
                exploration_exploitation_ratio: 1.4,
                discount_gamma: 1.0,
            })
        }
        crate::spec::CompiledTuneController::AiqiDiscounted(inner) => {
            ControllerSpec::AiqiDiscounted(AiqiDiscountedControllerSpec {
                predictor: RateBackend::Ctw { depth: 8 },
                discount_gamma: inner.discount_factor,
                return_horizon: inner.return_horizon,
                return_bins: inner.return_bins,
                augmentation_period: inner.return_horizon,
                history_prune_keep_steps: None,
                baseline_exploration: 1.0e-12,
            })
        }
        crate::spec::CompiledTuneController::AiqiWarmstartExactJh(inner) => {
            let return_bins =
                warmstart_return_bins(reward_encoder.max_reward(), inner.return_horizon)?;
            ControllerSpec::AiqiWarmstartExactJh(WarmStartExactJhControllerSpec {
                predictor: RateBackend::Ctw { depth: 8 },
                return_horizon: inner.return_horizon,
                return_bins,
                label_phase_period: inner.label_phase_period,
                teacher_dataset_asset: inner.warmstart_teacher_dataset_asset.clone(),
                planner_simulations_per_step: inner.planner_simulations_per_step,
            })
        }
        crate::spec::CompiledTuneController::AnnealedHillClimbing(_) => {
            return Err("annealed controller does not compile to planner-run agent".to_string());
        }
    };
    PlannerRunSpec {
        assets: compiled.canonical_spec().assets.clone(),
        environment: EnvironmentSpec::Builtin {
            builtin: BuiltinEnvironmentSpec::TunerBridge,
        },
        interface,
        controller: controller_spec,
        runtime: PlannerRuntimeSpec {
            random_seed: Some(compiled.canonical_spec().seed),
            learn_cycles: None,
            eval_cycles: None,
            terminate_lifetime: 1,
            log_every: 1,
            perf: false,
            vm_perf_only: false,
            explore_epsilon: 0.0,
            explore_gamma: 1.0,
        },
    }
    .compile_in(&SpecEnvironment::default())
    .map_err(|err| format!("failed to compile tuner planner-run bridge: {err}"))
}

fn build_tuner_planner_agent_runtime(
    controller: &crate::spec::CompiledTuneController,
    planner_run: &CompiledPlannerRunSpec,
    contract: &PlannerControllerContract,
    initial_percept: PlannerEncodedPercept,
) -> Result<TunerPlannerAgentRuntime, String> {
    match controller {
        crate::spec::CompiledTuneController::McAixiFacCtw(_) => {
            Ok(TunerPlannerAgentRuntime::McAixi {
                agent: Agent::from_compiled_planner_run(planner_run)
                    .map_err(|err| err.to_string())?,
                prev_action: 0,
                prev_percept: initial_percept,
            })
        }
        crate::spec::CompiledTuneController::AiqiDiscounted(_) => {
            Ok(TunerPlannerAgentRuntime::AiqiDiscounted {
                agent: AiqiAgent::from_compiled_planner_run(planner_run)
                    .map_err(|err| err.to_string())?,
            })
        }
        crate::spec::CompiledTuneController::AiqiWarmstartExactJh(_) => {
            let teacher = contract
                .teacher
                .as_ref()
                .ok_or_else(|| "warm-start controller missing teacher dataset".to_string())?;
            Ok(TunerPlannerAgentRuntime::WarmStartExactJh {
                agent: WarmStartExactJhAgent::from_compiled_planner_run(
                    planner_run,
                    teacher.traces.clone(),
                )
                .map_err(|err| err.to_string())?,
            })
        }
        crate::spec::CompiledTuneController::AnnealedHillClimbing(_) => {
            Err("annealed controller does not instantiate planner-family agents".to_string())
        }
    }
}

fn merge_warmstart_trace_deterministic(
    teacher: &mut WarmStartExactJhTeacherDataset,
    trace: WarmStartExactJhTeacherTrace,
) -> Result<(), String> {
    let key = warmstart_trace_key(&trace)?;
    let already_present = teacher
        .traces
        .iter()
        .map(warmstart_trace_key)
        .collect::<Result<BTreeSet<String>, String>>()?
        .contains(&key);
    if !already_present {
        teacher.traces.push(trace);
        teacher
            .traces
            .sort_by_key(|trace| warmstart_trace_key(trace).unwrap_or_default());
    }
    Ok(())
}

fn warmstart_trace_key(trace: &WarmStartExactJhTeacherTrace) -> Result<String, String> {
    let value = serde_json::json!({
        "transitions": trace.transitions.iter().map(|transition| {
            serde_json::json!({
                "action": transition.action,
                "observations": transition.observations,
                "reward": transition.reward,
            })
        }).collect::<Vec<Value>>(),
    });
    serde_json::to_vec(&value)
        .map(|bytes| crc32_hex(&bytes))
        .map_err(|err| format!("failed to encode warm-start trace key: {err}"))
}

fn encode_tuner_planner_percept(
    interface: &crate::spec::TunePlannerInterfaceSpec,
    incumbent_eval: Option<&CandidateEvalResult>,
    dataset_units: f64,
    reward: Reward,
    diagnostic_token: &str,
    candidate_eval: Option<&CandidateEvalResult>,
    candidate_bytes: Option<&[u8]>,
    effective_eval_limit_seconds: Option<f64>,
    terminal: bool,
) -> Result<PlannerEncodedPercept, String> {
    let raw = TunerRawObservation::from_runtime_step(
        incumbent_eval,
        dataset_units,
        candidate_eval,
        candidate_bytes,
        effective_eval_limit_seconds,
        diagnostic_token,
        terminal,
    );
    let stream_len = interface.observation_stream_len.max(1);
    let mut observations = Vec::with_capacity(stream_len);
    for index in 0..stream_len {
        observations.push(packed_observation_symbol(
            interface.observation_bits,
            index,
            raw.encoded_bytes(),
        ));
    }
    Ok(PlannerEncodedPercept {
        observations,
        reward,
    })
}

struct TunerRawObservation {
    encoded: Vec<u8>,
}

impl TunerRawObservation {
    #[allow(clippy::too_many_arguments)]
    fn from_runtime_step(
        incumbent_eval: Option<&CandidateEvalResult>,
        dataset_units: f64,
        candidate_eval: Option<&CandidateEvalResult>,
        candidate_bytes: Option<&[u8]>,
        effective_eval_limit_seconds: Option<f64>,
        diagnostic_token: &str,
        terminal: bool,
    ) -> Self {
        let units = if dataset_units.is_finite() && dataset_units > 0.0 {
            dataset_units
        } else {
            1.0
        };
        let fail_flag = candidate_eval
            .map(|value| value.status != CandidateEvalStatus::Success || !value.deployable)
            .unwrap_or(true);
        let successful_eval =
            candidate_eval.filter(|value| value.status == CandidateEvalStatus::Success);
        let normalized_physical_size =
            successful_eval.map(|value| (value.compressed_bytes as f64) / units);
        let normalized_target_loss = successful_eval.and_then(|value| {
            if value.target_loss_bits.is_finite() {
                Some((value.target_loss_bits / units).max(0.0))
            } else {
                None
            }
        });
        let normalized_eval_time = match candidate_eval {
            Some(value) if value.status == CandidateEvalStatus::Timeout => Some(1.0),
            Some(value) if value.status == CandidateEvalStatus::Success => Some(
                normalize_eval_time(value.elapsed_seconds, effective_eval_limit_seconds),
            ),
            _ => None,
        };
        let physical_size_delta = match (incumbent_eval, successful_eval) {
            (Some(incumbent), Some(value))
                if incumbent.status == CandidateEvalStatus::Success
                    && incumbent.compressed_bytes > 0 =>
            {
                Some(
                    (incumbent.compressed_bytes as f64 - value.compressed_bytes as f64)
                        / incumbent.compressed_bytes as f64,
                )
            }
            _ => None,
        };
        let eval_time_delta = match (incumbent_eval, successful_eval) {
            (Some(incumbent), Some(value))
                if incumbent.status == CandidateEvalStatus::Success
                    && incumbent.elapsed_seconds.is_finite()
                    && value.elapsed_seconds.is_finite() =>
            {
                let denom = incumbent.elapsed_seconds.max(1.0e-9);
                Some((incumbent.elapsed_seconds - value.elapsed_seconds) / denom)
            }
            _ => None,
        };
        let signature_source = candidate_bytes.unwrap_or_else(|| diagnostic_token.as_bytes());
        let candidate_signature_crc32 = crc32_u32(signature_source);
        let mut encoded = Vec::<u8>::with_capacity(64);
        encoded.push(u8::from(fail_flag));
        push_optional_f64(&mut encoded, normalized_physical_size);
        push_optional_f64(&mut encoded, normalized_target_loss);
        push_optional_f64(&mut encoded, normalized_eval_time);
        push_optional_f64(&mut encoded, physical_size_delta);
        push_optional_f64(&mut encoded, eval_time_delta);
        encoded.extend_from_slice(&candidate_signature_crc32.to_le_bytes());
        encoded.push(u8::from(terminal));
        Self { encoded }
    }

    fn encoded_bytes(&self) -> &[u8] {
        &self.encoded
    }
}

fn normalize_eval_time(elapsed_seconds: f64, effective_eval_limit_seconds: Option<f64>) -> f64 {
    if let Some(limit) = effective_eval_limit_seconds
        && limit > 0.0
    {
        return (elapsed_seconds / limit).clamp(0.0, 1.0);
    }
    if elapsed_seconds.is_finite() {
        elapsed_seconds.max(0.0)
    } else {
        1.0
    }
}

fn push_optional_f64(out: &mut Vec<u8>, value: Option<f64>) {
    match value {
        Some(number) => {
            out.push(1);
            out.extend_from_slice(&number.to_bits().to_le_bytes());
        }
        None => out.push(0),
    }
}

fn packed_observation_symbol(
    observation_bits: usize,
    index: usize,
    raw_payload: &[u8],
) -> PerceptVal {
    if observation_bits == 0 {
        return 0;
    }
    let offset = index.saturating_mul(std::mem::size_of::<u64>());
    let mut bytes = [0_u8; 8];
    if offset < raw_payload.len() {
        let available = (raw_payload.len() - offset).min(bytes.len());
        bytes[..available].copy_from_slice(&raw_payload[offset..offset + available]);
    } else {
        bytes[0] = 0xff;
    }
    let value = u64::from_le_bytes(bytes);
    if observation_bits >= 64 {
        value
    } else {
        value & ((1u64 << observation_bits) - 1)
    }
}

fn crc32_u32(bytes: &[u8]) -> u32 {
    let mut hasher = Hasher::new();
    hasher.update(bytes);
    hasher.finalize()
}

fn max_nonnegative_reward_for_bits(reward_bits: usize) -> Result<Reward, String> {
    if reward_bits == 0 {
        return Err("reward_bits must be >= 1".to_string());
    }
    if reward_bits >= 63 {
        Ok(Reward::MAX)
    } else {
        Ok(((1u64 << reward_bits) - 1) as Reward)
    }
}

fn exact_nonnegative_i64_from_f64(value: f64, label: &str) -> Result<Reward, String> {
    if !value.is_finite() {
        return Err(format!("{label} must be finite"));
    }
    if value < 0.0 {
        return Err(format!("{label} must be nonnegative"));
    }
    let rounded = value.round();
    if (rounded - value).abs() > f64::EPSILON {
        return Err(format!(
            "reward_encoding_unsafe: {label} must be exactly representable as a finite nonnegative integer"
        ));
    }
    if rounded > (Reward::MAX as f64) {
        return Err(format!(
            "reward_encoding_unsafe: {label} exceeds maximum representable reward"
        ));
    }
    Ok(rounded as Reward)
}

fn warmstart_return_bins(max_reward: Reward, return_horizon: usize) -> Result<usize, String> {
    let max_total = (max_reward as u128)
        .checked_mul(return_horizon as u128)
        .and_then(|value| value.checked_add(1))
        .ok_or_else(|| "warm-start exact-J_H return label range overflowed".to_string())?;
    usize::try_from(max_total)
        .map_err(|_| "warm-start exact-J_H return label range does not fit usize".to_string())
}

fn planner_return_bins(planner_run: &CompiledPlannerRunSpec) -> Option<usize> {
    match planner_run.controller() {
        crate::spec::CompiledPlannerController::AiqiDiscounted { return_bins, .. }
        | crate::spec::CompiledPlannerController::AiqiWarmstartExactJh { return_bins, .. } => {
            Some(*return_bins)
        }
        crate::spec::CompiledPlannerController::McAixi { .. } => None,
    }
}

fn planner_agent_runtime_name(controller: &crate::spec::CompiledTuneController) -> &'static str {
    match controller {
        crate::spec::CompiledTuneController::McAixiFacCtw(_) => "aixi::agent::Agent",
        crate::spec::CompiledTuneController::AiqiDiscounted(_) => "aixi::aiqi::AiqiAgent",
        crate::spec::CompiledTuneController::AiqiWarmstartExactJh(_) => {
            "aixi::warmstart::WarmStartExactJhAgent"
        }
        crate::spec::CompiledTuneController::AnnealedHillClimbing(_) => "annealer",
    }
}

fn tuner_reward_encoding_name(encoder: &TunerRewardEncoder) -> &'static str {
    match encoder {
        TunerRewardEncoder::ExactIntegerObjectiveDifference {
            objective_difference_to_symbol: Some(_),
            ..
        } => "exact_finite_reward_symbol_map",
        TunerRewardEncoder::ExactIntegerObjectiveDifference { .. } => {
            "exact_integer_objective_difference_interval"
        }
        TunerRewardEncoder::NormalizedClipped { .. } => "rounded_normalized_clipped_scalar",
    }
}

fn normalized_clipped_improvement(
    raw_improvement: f64,
    min_improvement: f64,
    max_improvement: f64,
) -> Result<f64, String> {
    if !(min_improvement.is_finite()
        && max_improvement.is_finite()
        && max_improvement > min_improvement)
    {
        return Err(
            "normalized clipped improvement requires finite max_improvement > min_improvement"
                .to_string(),
        );
    }
    Ok(((raw_improvement - min_improvement) / (max_improvement - min_improvement)).clamp(0.0, 1.0))
}

fn key_less(
    candidate_eval: &CandidateEvalResult,
    candidate_bytes: &[u8],
    incumbent_eval: &CandidateEvalResult,
    incumbent_bytes: &[u8],
) -> bool {
    candidate_eval.objective_bits < incumbent_eval.objective_bits
        || ((candidate_eval.objective_bits - incumbent_eval.objective_bits).abs() <= f64::EPSILON
            && candidate_bytes < incumbent_bytes)
}

fn annealer_progress(tune_started: Instant, time_budget_seconds: f64) -> f64 {
    annealer_progress_from_elapsed(tune_started.elapsed().as_secs_f64(), time_budget_seconds)
}

fn annealer_progress_from_elapsed(elapsed_seconds: f64, time_budget_seconds: f64) -> f64 {
    if time_budget_seconds <= 0.0 {
        return 1.0;
    }
    (elapsed_seconds / time_budget_seconds).clamp(0.0, 1.0)
}

fn annealer_temperature(progress: f64) -> f64 {
    let u = progress.clamp(0.0, 1.0);
    let temperature = ANNEALER_T_MIN_BITS * (ANNEALER_T0_BITS / ANNEALER_T_MIN_BITS).powf(1.0 - u);
    temperature.clamp(ANNEALER_T_MIN_BITS, ANNEALER_T0_BITS)
}

fn annealer_active_radius(max_mutation_radius: usize, temperature: f64) -> usize {
    ((max_mutation_radius as f64) * temperature)
        .floor()
        .max(1.0) as usize
}

fn annealer_runtime_path_name(profile: AnnealerKernelProfile) -> &'static str {
    match profile {
        AnnealerKernelProfile::ReversibleElementaryMetropolis => "reversible_elementary_metropolis",
        AnnealerKernelProfile::CompiledUniformMetropolisHastings => {
            "compiled_uniform_metropolis_hastings"
        }
    }
}

fn annealer_acceptance_probability(
    profile: AnnealerKernelProfile,
    delta: f64,
    temperature: f64,
    proposal: &AnnealedProposal,
) -> Result<f64, String> {
    if proposal.forward_raw_action_count == 0 || proposal.forward_total_raw_actions == 0 {
        return Ok(0.0);
    }
    let metropolis = (-delta / temperature).exp();
    match profile {
        AnnealerKernelProfile::ReversibleElementaryMetropolis => {
            let forward = (proposal.forward_raw_action_count as u128)
                * (proposal.reverse_total_raw_actions as u128);
            let reverse = (proposal.reverse_raw_action_count as u128)
                * (proposal.forward_total_raw_actions as u128);
            if forward != reverse {
                return Err(
                    "compiled elementary proposal kernel failed reversibility check".to_string(),
                );
            }
            Ok(metropolis.clamp(0.0, 1.0))
        }
        AnnealerKernelProfile::CompiledUniformMetropolisHastings => {
            if proposal.reverse_raw_action_count == 0 || proposal.reverse_total_raw_actions == 0 {
                return Ok(0.0);
            }
            let hastings_ratio = ((proposal.reverse_raw_action_count as f64)
                * (proposal.forward_total_raw_actions as f64))
                / ((proposal.forward_raw_action_count as f64)
                    * (proposal.reverse_total_raw_actions as f64));
            Ok((metropolis * hastings_ratio).clamp(0.0, 1.0))
        }
    }
}

fn sample_annealed_proposal(
    candidate: &crate::api::CompressionBackend,
    bounds: &crate::spec::TuneBoundsSpec,
    max_mutation_radius: usize,
    active_radius: usize,
    env: &SpecEnvironment,
    rng: &mut RandomGenerator,
) -> Result<AnnealedProposalDraw, String> {
    let current_compiled = candidate
        .compile_in(env)
        .map_err(|err| format!("failed to compile current annealer candidate: {err}"))?;
    let current_canonical_bytes = current_compiled.canonical_bytes().as_slice().to_vec();
    let forward = compile_canonical_proposal_kernel(
        candidate,
        bounds,
        max_mutation_radius,
        active_radius,
        env,
        &current_canonical_bytes,
    )?;
    if forward.transitions.is_empty() {
        return Ok(AnnealedProposalDraw::Exhausted);
    }
    let Some(proposed) = forward.sample(rng) else {
        return Ok(AnnealedProposalDraw::SelfLoop);
    };
    let reverse = compile_canonical_proposal_kernel(
        &proposed.candidate,
        bounds,
        max_mutation_radius,
        active_radius,
        env,
        &proposed.candidate_canonical_bytes,
    )?;
    Ok(AnnealedProposalDraw::Proposal(AnnealedProposal {
        candidate: proposed.candidate.clone(),
        forward_raw_action_count: proposed.raw_action_count,
        forward_total_raw_actions: forward.total_raw_actions,
        reverse_raw_action_count: reverse
            .proposal_mass_to_canonical_bytes(&current_canonical_bytes),
        reverse_total_raw_actions: reverse.total_raw_actions,
    }))
}

fn compile_canonical_proposal_kernel(
    candidate: &crate::api::CompressionBackend,
    bounds: &crate::spec::TuneBoundsSpec,
    max_mutation_radius: usize,
    active_radius: usize,
    env: &SpecEnvironment,
    current_canonical_bytes: &[u8],
) -> Result<CanonicalProposalKernel, String> {
    let json = crate::spec::compression_backend_to_json_value(candidate)
        .map_err(|err| format!("failed to serialize candidate for proposal kernel: {err}"))?;
    let range_map = bounds
        .parameter_ranges
        .iter()
        .map(|range| (range.parameter.clone(), (range.min, range.max)))
        .collect::<BTreeMap<_, _>>();
    let mut leaves = collect_numeric_leaves(&json);
    if !range_map.is_empty() {
        leaves.retain(|leaf| range_map.contains_key(&leaf.path));
    }
    leaves.retain(|leaf| matches!(leaf.kind, NumericKind::Unsigned | NumericKind::Signed));
    let total_raw_actions = (leaves.len() as u64)
        .saturating_mul(2)
        .saturating_mul(max_mutation_radius.max(1) as u64);
    let mut transitions = BTreeMap::<Vec<u8>, CanonicalProposal>::new();
    for leaf in &leaves {
        let Some((min_bound, max_bound)) =
            integer_leaf_bounds(leaf.kind, range_map.get(&leaf.path).copied())
        else {
            continue;
        };
        for magnitude in 1..=max_mutation_radius.max(1) {
            for sign in [-1i128, 1i128] {
                if magnitude > active_radius {
                    continue;
                }
                let mut next_json = json.clone();
                let delta = sign * (magnitude as i128);
                if !apply_integer_descriptor(&mut next_json, leaf, min_bound, max_bound, delta) {
                    continue;
                }
                let parsed = match crate::spec::parse_compression_backend_json(
                    &next_json,
                    Path::new("."),
                    None,
                    crate::compression::FramingMode::Framed,
                ) {
                    Ok(value) => value,
                    Err(_) => continue,
                };
                if reject_candidate_local_external_artifacts(&parsed).is_err()
                    || validate_candidate_against_tune_bounds(&parsed, bounds).is_err()
                {
                    continue;
                }
                let compiled = match parsed.compile_in(env) {
                    Ok(value) => value,
                    Err(_) => continue,
                };
                let candidate_canonical_bytes = compiled.canonical_bytes().as_slice().to_vec();
                if candidate_canonical_bytes == current_canonical_bytes {
                    continue;
                }
                transitions
                    .entry(candidate_canonical_bytes.clone())
                    .and_modify(|proposal| {
                        proposal.raw_action_count = proposal.raw_action_count.saturating_add(1);
                    })
                    .or_insert(CanonicalProposal {
                        candidate: parsed,
                        candidate_canonical_bytes,
                        raw_action_count: 1,
                    });
            }
        }
    }
    Ok(CanonicalProposalKernel {
        transitions: transitions.into_values().collect(),
        total_raw_actions,
    })
}

fn integer_leaf_bounds(kind: NumericKind, range: Option<(f64, f64)>) -> Option<(i128, i128)> {
    let (type_min, type_max) = match kind {
        NumericKind::Unsigned => (0i128, u64::MAX as i128),
        NumericKind::Signed => (i64::MIN as i128, i64::MAX as i128),
        NumericKind::Float => return None,
    };
    let (min, max) = match range {
        Some((min, max)) => (
            (min.ceil() as i128).clamp(type_min, type_max),
            (max.floor() as i128).clamp(type_min, type_max),
        ),
        None => (type_min, type_max),
    };
    (min <= max).then_some((min, max))
}

fn apply_integer_descriptor(
    json: &mut Value,
    leaf: &NumericLeaf,
    min_bound: i128,
    max_bound: i128,
    delta: i128,
) -> bool {
    let Some(slot) = json.pointer_mut(&leaf.pointer) else {
        return false;
    };
    let current = match leaf.kind {
        NumericKind::Unsigned => slot.as_u64().map(i128::from),
        NumericKind::Signed => slot.as_i64().map(i128::from),
        NumericKind::Float => None,
    };
    let Some(current) = current else {
        return false;
    };
    let Some(next) = current.checked_add(delta) else {
        return false;
    };
    if next < min_bound || next > max_bound || next == current {
        return false;
    }
    match leaf.kind {
        NumericKind::Unsigned => {
            let Ok(value) = u64::try_from(next) else {
                return false;
            };
            *slot = Value::Number(serde_json::Number::from(value));
            true
        }
        NumericKind::Signed => {
            let Ok(value) = i64::try_from(next) else {
                return false;
            };
            *slot = Value::Number(serde_json::Number::from(value));
            true
        }
        NumericKind::Float => false,
    }
}

fn collect_numeric_leaves(root: &Value) -> Vec<NumericLeaf> {
    let mut out = Vec::<NumericLeaf>::new();
    collect_numeric_leaves_inner(root, "", "", &mut out);
    out
}

fn collect_numeric_leaves_inner(
    value: &Value,
    path_prefix: &str,
    pointer_prefix: &str,
    out: &mut Vec<NumericLeaf>,
) {
    match value {
        Value::Object(object) => {
            for (key, child) in object {
                let next_path = if path_prefix.is_empty() {
                    key.to_string()
                } else {
                    format!("{path_prefix}.{key}")
                };
                let escaped = key.replace('~', "~0").replace('/', "~1");
                let next_pointer = if pointer_prefix.is_empty() {
                    format!("/{escaped}")
                } else {
                    format!("{pointer_prefix}/{escaped}")
                };
                collect_numeric_leaves_inner(child, &next_path, &next_pointer, out);
            }
        }
        Value::Array(items) => {
            for (index, child) in items.iter().enumerate() {
                let next_path = format!("{path_prefix}[{index}]");
                let next_pointer = if pointer_prefix.is_empty() {
                    format!("/{index}")
                } else {
                    format!("{pointer_prefix}/{index}")
                };
                collect_numeric_leaves_inner(child, &next_path, &next_pointer, out);
            }
        }
        Value::Number(number) => {
            let kind = if number.as_u64().is_some() {
                Some(NumericKind::Unsigned)
            } else if number.as_i64().is_some() {
                Some(NumericKind::Signed)
            } else if number.as_f64().is_some() {
                Some(NumericKind::Float)
            } else {
                None
            };
            if let Some(kind) = kind {
                out.push(NumericLeaf {
                    path: path_prefix.to_string(),
                    pointer: pointer_prefix.to_string(),
                    kind,
                });
            }
        }
        _ => {}
    }
}

fn evaluate_candidate(
    candidate: &crate::spec::CompiledCompressionBackend,
    dataset: &LoadedDataset,
    model_bytes: usize,
    min_throughput_bytes_per_second: f64,
    max_memory_bytes: u64,
    effective_eval_time_limit_seconds: f64,
    rss_mode: PeakMemoryMode,
    deterministic_table: Option<&VerifiedDeterministicEvaluatorTable>,
) -> Result<CandidateEvalResult, String> {
    if let Some(table) = deterministic_table {
        return table.evaluate(
            candidate,
            dataset,
            model_bytes,
            min_throughput_bytes_per_second,
            max_memory_bytes,
            effective_eval_time_limit_seconds,
        );
    }
    #[cfg(not(unix))]
    {
        let _ = candidate;
        let _ = dataset;
        let _ = model_bytes;
        let _ = min_throughput_bytes_per_second;
        let _ = max_memory_bytes;
        let _ = effective_eval_time_limit_seconds;
        let _ = rss_mode;
        let _ = deterministic_table;
        return Err(
            "tuner requires a Unix target for process-isolated candidate evaluation".to_string(),
        );
    }

    #[cfg(unix)]
    {
        return evaluate_candidate_unix_isolated(
            candidate,
            dataset,
            model_bytes,
            min_throughput_bytes_per_second,
            max_memory_bytes,
            effective_eval_time_limit_seconds,
            rss_mode,
        );
    }
}

#[cfg(unix)]
fn evaluate_candidate_unix_isolated(
    candidate: &crate::spec::CompiledCompressionBackend,
    dataset: &LoadedDataset,
    model_bytes: usize,
    min_throughput_bytes_per_second: f64,
    max_memory_bytes: u64,
    effective_eval_time_limit_seconds: f64,
    rss_mode: PeakMemoryMode,
) -> Result<CandidateEvalResult, String> {
    if effective_eval_time_limit_seconds <= 0.0 {
        return Ok(timeout_eval_result(
            0.0,
            peak_memory_bytes(rss_mode),
            effective_eval_time_limit_seconds,
        ));
    }
    let mut pipefds: [i32; 2] = [0, 0];
    // SAFETY: `pipefds` points to two writable `i32` slots as required by `pipe(2)`.
    let pipe_result = unsafe { libc::pipe(pipefds.as_mut_ptr()) };
    if pipe_result != 0 {
        return Err(format!(
            "failed to create evaluator IPC pipe: {}",
            std::io::Error::last_os_error()
        ));
    }
    let read_fd = pipefds[0];
    let write_fd = pipefds[1];

    // SAFETY: `fork(2)` duplicates the current process. We perform only
    // async-signal-safe operations before branching, and in the child we avoid
    // touching shared parent state except pure evaluation + pipe write + `_exit`.
    let pid = unsafe { libc::fork() };
    if pid < 0 {
        close_fd(read_fd);
        close_fd(write_fd);
        return Err(format!(
            "failed to fork evaluator process: {}",
            std::io::Error::last_os_error()
        ));
    }
    if pid == 0 {
        close_fd(read_fd);
        let result = evaluate_candidate_unbounded(
            candidate,
            dataset,
            model_bytes,
            min_throughput_bytes_per_second,
            max_memory_bytes,
            effective_eval_time_limit_seconds,
            rss_mode,
        );
        let payload = match result {
            Ok(value) => serde_json::json!({
                "ok": true,
                "status": value.status.name(),
                "compressed_bytes": value.compressed_bytes,
                "elapsed_seconds": value.elapsed_seconds,
                "effective_eval_time_limit_seconds": value.effective_eval_time_limit_seconds,
                "throughput_bytes_per_second": value.throughput_bytes_per_second,
                "peak_memory_bytes": value.peak_memory_bytes,
                "target_loss_bits": value.target_loss_bits,
                "objective_bits": value.objective_bits,
                "deployable": value.deployable,
            }),
            Err(err) => serde_json::json!({
                "ok": false,
                "error": err,
            }),
        };
        if let Ok(bytes) = serde_json::to_vec(&payload) {
            let _ = write_all_fd(write_fd, &bytes);
        }
        close_fd(write_fd);
        // SAFETY: `_exit` terminates the child immediately without running
        // parent-owned destructors after `fork`.
        unsafe { libc::_exit(0) }
    } else {
        close_fd(write_fd);
        let started = Instant::now();
        let mut status: i32 = 0;
        let timeout = Duration::from_secs_f64(effective_eval_time_limit_seconds);
        loop {
            // SAFETY: `waitpid` is called for the specific child pid, with a valid
            // status pointer and `WNOHANG` for polling.
            let waited = unsafe { libc::waitpid(pid, &mut status as *mut i32, libc::WNOHANG) };
            if waited == pid {
                break;
            }
            if waited < 0 {
                close_fd(read_fd);
                return Err(format!(
                    "failed while waiting for evaluator process: {}",
                    std::io::Error::last_os_error()
                ));
            }
            if started.elapsed() >= timeout {
                let peak_before_kill =
                    peak_memory_bytes_for_pid(pid as libc::pid_t, rss_mode).unwrap_or(0);
                // SAFETY: `pid` is the live evaluator child process id.
                unsafe {
                    libc::kill(pid, libc::SIGKILL);
                }
                // SAFETY: reap child after kill to avoid zombies.
                unsafe {
                    libc::waitpid(pid, &mut status as *mut i32, 0);
                }
                close_fd(read_fd);
                return Ok(timeout_eval_result(
                    effective_eval_time_limit_seconds,
                    peak_before_kill,
                    effective_eval_time_limit_seconds,
                ));
            }
            std::thread::sleep(Duration::from_millis(1));
        }

        let payload_bytes = read_all_fd(read_fd)?;
        if payload_bytes.is_empty() {
            return Err("candidate evaluation worker returned no payload".to_string());
        }
        let payload: Value = serde_json::from_slice(&payload_bytes)
            .map_err(|err| format!("invalid evaluator payload from child process: {err}"))?;
        parse_candidate_eval_payload(&payload)
    }
}

#[cfg(unix)]
fn parse_candidate_eval_payload(payload: &Value) -> Result<CandidateEvalResult, String> {
    let object = payload
        .as_object()
        .ok_or_else(|| "invalid evaluator payload shape".to_string())?;
    let ok = object
        .get("ok")
        .and_then(Value::as_bool)
        .ok_or_else(|| "evaluator payload missing boolean 'ok' field".to_string())?;
    if !ok {
        let err = object
            .get("error")
            .and_then(Value::as_str)
            .unwrap_or("candidate evaluator failed without error message");
        return Err(err.to_string());
    }
    let status = match object
        .get("status")
        .and_then(Value::as_str)
        .ok_or_else(|| "evaluator payload missing status".to_string())?
    {
        "success" => CandidateEvalStatus::Success,
        "timeout" => CandidateEvalStatus::Timeout,
        "invalid" => CandidateEvalStatus::Invalid,
        "error" => CandidateEvalStatus::Error,
        other => return Err(format!("unknown evaluator status '{other}'")),
    };
    let compressed_bytes = object
        .get("compressed_bytes")
        .and_then(Value::as_u64)
        .ok_or_else(|| "evaluator payload missing compressed_bytes".to_string())?
        as usize;
    let elapsed_seconds = object
        .get("elapsed_seconds")
        .and_then(Value::as_f64)
        .ok_or_else(|| "evaluator payload missing elapsed_seconds".to_string())?;
    let effective_eval_time_limit_seconds = object
        .get("effective_eval_time_limit_seconds")
        .and_then(Value::as_f64)
        .ok_or_else(|| "evaluator payload missing effective_eval_time_limit_seconds".to_string())?;
    if !effective_eval_time_limit_seconds.is_finite() || effective_eval_time_limit_seconds < 0.0 {
        return Err("evaluator payload has invalid effective_eval_time_limit_seconds".to_string());
    }
    let throughput_bytes_per_second = object
        .get("throughput_bytes_per_second")
        .and_then(Value::as_f64)
        .ok_or_else(|| "evaluator payload missing throughput_bytes_per_second".to_string())?;
    let peak_memory_bytes = object
        .get("peak_memory_bytes")
        .and_then(Value::as_u64)
        .ok_or_else(|| "evaluator payload missing peak_memory_bytes".to_string())?;
    let target_loss_bits = object
        .get("target_loss_bits")
        .and_then(Value::as_f64)
        .ok_or_else(|| "evaluator payload missing target_loss_bits".to_string())?;
    let objective_bits = object
        .get("objective_bits")
        .and_then(Value::as_f64)
        .ok_or_else(|| "evaluator payload missing objective_bits".to_string())?;
    let deployable = object
        .get("deployable")
        .and_then(Value::as_bool)
        .ok_or_else(|| "evaluator payload missing deployable".to_string())?;
    Ok(CandidateEvalResult {
        status,
        compressed_bytes,
        elapsed_seconds,
        effective_eval_time_limit_seconds,
        throughput_bytes_per_second,
        peak_memory_bytes,
        target_loss_bits,
        objective_bits,
        deployable,
    })
}

#[cfg(unix)]
fn read_all_fd(fd: RawFd) -> Result<Vec<u8>, String> {
    // SAFETY: `fd` is a valid read end of a pipe owned by this process; we
    // transfer ownership into `File` exactly once.
    let mut file = unsafe { std::fs::File::from_raw_fd(fd) };
    let mut bytes = Vec::new();
    file.read_to_end(&mut bytes)
        .map_err(|err| format!("failed to read evaluator payload: {err}"))?;
    Ok(bytes)
}

#[cfg(unix)]
fn write_all_fd(fd: RawFd, bytes: &[u8]) -> Result<(), String> {
    let mut offset: usize = 0;
    while offset < bytes.len() {
        // SAFETY: `fd` is the writable end of a pipe and pointer/length are
        // valid for the remaining byte slice.
        let written = unsafe {
            libc::write(
                fd,
                bytes[offset..].as_ptr().cast::<libc::c_void>(),
                bytes.len() - offset,
            )
        };
        if written < 0 {
            return Err(format!(
                "failed to write evaluator payload: {}",
                std::io::Error::last_os_error()
            ));
        }
        if written == 0 {
            return Err("evaluator payload write returned zero bytes".to_string());
        }
        offset = offset.saturating_add(written as usize);
    }
    Ok(())
}

#[cfg(unix)]
fn close_fd(fd: RawFd) {
    // SAFETY: Closing a raw file descriptor is safe when ownership is local.
    let _ = unsafe { libc::close(fd) };
}

fn evaluate_candidate_unbounded(
    candidate: &crate::spec::CompiledCompressionBackend,
    dataset: &LoadedDataset,
    model_bytes: usize,
    min_throughput_bytes_per_second: f64,
    max_memory_bytes: u64,
    effective_eval_time_limit_seconds: f64,
    rss_mode: PeakMemoryMode,
) -> Result<CandidateEvalResult, String> {
    let before_peak = peak_memory_bytes(rss_mode);
    let start = Instant::now();
    let deadline = start + Duration::from_secs_f64(effective_eval_time_limit_seconds);
    let (compressed_bytes, target_loss_bits) = match dataset.kind {
        DatasetKind::PassiveBytes => {
            let mut runtime = crate::runtime::build_compression_runtime(candidate)
                .map_err(|err| format!("failed to build candidate runtime: {err}"))?;
            let compressed_bytes_u64 = runtime
                .compress_size(&dataset.raw_bytes)
                .map_err(|err| format!("candidate evaluation failed: {err}"))?;
            let compressed_bytes = usize::try_from(compressed_bytes_u64)
                .map_err(|_| "compressed size does not fit usize on this platform".to_string())?;
            (compressed_bytes, (compressed_bytes as f64) * 8.0)
        }
        DatasetKind::InteractiveTrace | DatasetKind::CausalPrefixDataset => {
            evaluate_candidate_causal_loss(candidate, dataset, deadline)?
        }
    };
    let elapsed_seconds = start.elapsed().as_secs_f64();
    let after_peak = peak_memory_bytes(rss_mode);
    let peak_memory_bytes = after_peak.max(before_peak);
    if elapsed_seconds >= effective_eval_time_limit_seconds || target_loss_bits.is_infinite() {
        return Ok(timeout_eval_result(
            elapsed_seconds,
            peak_memory_bytes,
            effective_eval_time_limit_seconds,
        ));
    }

    let throughput_bytes_per_second = if elapsed_seconds <= 0.0 {
        f64::INFINITY
    } else {
        dataset.dataset_units / elapsed_seconds
    };
    let objective_bits = ((model_bytes as f64) * 8.0) + target_loss_bits;
    let deployable = throughput_bytes_per_second >= min_throughput_bytes_per_second
        && peak_memory_bytes <= max_memory_bytes;

    Ok(CandidateEvalResult {
        status: CandidateEvalStatus::Success,
        compressed_bytes,
        elapsed_seconds,
        effective_eval_time_limit_seconds,
        throughput_bytes_per_second,
        peak_memory_bytes,
        target_loss_bits,
        objective_bits,
        deployable,
    })
}

fn timeout_eval_result(
    elapsed_seconds: f64,
    peak_memory_bytes: u64,
    effective_eval_time_limit_seconds: f64,
) -> CandidateEvalResult {
    CandidateEvalResult {
        status: CandidateEvalStatus::Timeout,
        compressed_bytes: 0,
        elapsed_seconds,
        effective_eval_time_limit_seconds,
        throughput_bytes_per_second: 0.0,
        peak_memory_bytes,
        target_loss_bits: f64::INFINITY,
        objective_bits: f64::INFINITY,
        deployable: false,
    }
}

fn error_eval_result(
    elapsed_seconds: f64,
    peak_memory_bytes: u64,
    effective_eval_time_limit_seconds: f64,
) -> CandidateEvalResult {
    CandidateEvalResult {
        status: CandidateEvalStatus::Error,
        compressed_bytes: 0,
        elapsed_seconds,
        effective_eval_time_limit_seconds,
        throughput_bytes_per_second: 0.0,
        peak_memory_bytes,
        target_loss_bits: f64::INFINITY,
        objective_bits: f64::INFINITY,
        deployable: false,
    }
}

fn evaluate_candidate_causal_loss(
    candidate: &crate::spec::CompiledCompressionBackend,
    dataset: &LoadedDataset,
    deadline: Instant,
) -> Result<(usize, f64), String> {
    if let crate::spec::core::CompressionBackendPlan::Rate { rate_backend, .. } = candidate.plan() {
        let causal_profile = dataset.causal_profile.as_ref().ok_or_else(|| {
            "causal dataset evaluation requires a typed causal profile".to_string()
        })?;
        let compiled_rate =
            crate::spec::core::compiled_rate_backend_from_plan(rate_backend.clone())
                .map_err(|err| format!("failed to compile causal evaluator rate backend: {err}"))?;
        let mut prefix_parts = Vec::<Vec<u8>>::new();
        let mut target_loss_bits = 0.0f64;
        for event in &dataset.events {
            if Instant::now() >= deadline {
                return Ok((0, f64::INFINITY));
            }
            match event {
                LoweredCausalEvent::Reset => prefix_parts.clear(),
                LoweredCausalEvent::Context { channel, bytes } => {
                    prefix_parts.push(causal_event_conditioning_bytes(
                        "context", channel, None, bytes,
                    ));
                }
                LoweredCausalEvent::ObserveTargetNoScore {
                    channel,
                    domain,
                    bytes,
                } => {
                    prefix_parts.push(causal_event_conditioning_bytes(
                        "observe_target_no_score",
                        channel,
                        Some(domain),
                        bytes,
                    ));
                }
                LoweredCausalEvent::Target {
                    channel,
                    domain,
                    bytes,
                    weight,
                } => {
                    let support = causal_profile.domains.get(domain).ok_or_else(|| {
                        format!("target event references undeclared domain '{domain}'")
                    })?;
                    let descriptor = causal_event_descriptor_bytes("target", channel, Some(domain));
                    let refs = prefix_parts
                        .iter()
                        .map(Vec::as_slice)
                        .collect::<Vec<&[u8]>>();
                    let loss = causal_target_loss_bits(
                        &refs,
                        &descriptor,
                        bytes,
                        support,
                        &compiled_rate,
                    )?;
                    target_loss_bits += (*weight) * loss;
                    prefix_parts.push(causal_event_conditioning_bytes(
                        "target",
                        channel,
                        Some(domain),
                        bytes,
                    ));
                }
            }
        }
        let compressed_bytes = (target_loss_bits / 8.0).ceil().max(0.0) as usize;
        Ok((compressed_bytes, target_loss_bits))
    } else {
        Err(
            "causal dataset evaluation requires a rate backend with conditional target-loss semantics"
                .to_string(),
        )
    }
}

fn causal_target_loss_bits(
    prefix_parts: &[&[u8]],
    descriptor: &[u8],
    target: &[u8],
    support: &CausalTargetDomain,
    compiled_rate: &crate::spec::CompiledRateBackend,
) -> Result<f64, String> {
    let mut descriptor_conditioned = Vec::<&[u8]>::with_capacity(prefix_parts.len() + 1);
    descriptor_conditioned.extend_from_slice(prefix_parts);
    descriptor_conditioned.push(descriptor);
    match support {
        CausalTargetDomain::ByteAlphabet => {
            if target.len() != 1 {
                return Err(
                    "byte_alphabet target payloads must be exactly one byte after lowering"
                        .to_string(),
                );
            }
            crate::runtime::try_cross_entropy_conditional_chain_backend(
                &descriptor_conditioned,
                target,
                compiled_rate,
            )
            .map_err(|err| format!("causal byte-domain target evaluation failed: {err}"))
        }
        CausalTargetDomain::EnumeratedPayloads { payloads } => {
            if !payloads.iter().any(|payload| payload == target) {
                return Err(
                    "target payload is outside enumerated target-domain support".to_string()
                );
            }
            let mut target_loss = None::<f64>;
            let mut log2_terms = Vec::<f64>::with_capacity(payloads.len());
            for payload in payloads {
                let loss = crate::runtime::try_cross_entropy_conditional_chain_backend(
                    &descriptor_conditioned,
                    payload,
                    compiled_rate,
                )
                .map_err(|err| {
                    format!("causal enumerated-domain target evaluation failed: {err}")
                })?;
                if payload == target {
                    target_loss = Some(loss);
                }
                log2_terms.push(-loss);
            }
            let log2_z = log2_sum_exp(&log2_terms);
            let loss = target_loss.ok_or_else(|| {
                "target payload is outside enumerated target-domain support".to_string()
            })?;
            Ok(loss + log2_z)
        }
    }
}

fn log2_sum_exp(log2_terms: &[f64]) -> f64 {
    let max_term = log2_terms.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    if !max_term.is_finite() {
        return max_term;
    }
    let sum = log2_terms
        .iter()
        .map(|term| 2.0f64.powf(*term - max_term))
        .sum::<f64>();
    max_term + sum.log2()
}

fn causal_event_conditioning_bytes(
    kind: &str,
    channel: &str,
    domain: Option<&str>,
    payload: &[u8],
) -> Vec<u8> {
    let mut out = causal_event_descriptor_bytes(kind, channel, domain);
    out.extend_from_slice(&(payload.len() as u64).to_le_bytes());
    out.extend_from_slice(payload);
    out
}

fn causal_event_descriptor_bytes(kind: &str, channel: &str, domain: Option<&str>) -> Vec<u8> {
    let mut out = Vec::<u8>::new();
    out.extend_from_slice(b"infotheory:tuner:causal-event:v1\0");
    push_tag_component(&mut out, kind.as_bytes());
    push_tag_component(&mut out, channel.as_bytes());
    push_tag_component(&mut out, domain.unwrap_or("").as_bytes());
    out
}

fn push_tag_component(out: &mut Vec<u8>, component: &[u8]) {
    out.extend_from_slice(&(component.len() as u64).to_le_bytes());
    out.extend_from_slice(component);
}

fn cache_key_for_candidate(
    candidate_bytes: &[u8],
    evaluator_profile: &EvaluatorProfile,
    dataset_hash: &str,
) -> Result<CandidateCacheKey, String> {
    let evaluator_profile_bytes = evaluator_profile.cache_identity_bytes()?;
    Ok(CandidateCacheKey {
        candidate_canonical_bytes: candidate_bytes.to_vec(),
        evaluator_profile_bytes,
        dataset_identity: dataset_hash.to_string(),
    })
}

fn dataset_kind_name(value: DatasetKind) -> &'static str {
    match value {
        DatasetKind::PassiveBytes => "passive_bytes",
        DatasetKind::InteractiveTrace => "interactive_trace",
        DatasetKind::CausalPrefixDataset => "causal_prefix_dataset",
    }
}

fn objective_target_name(value: ObjectiveTarget) -> &'static str {
    match value {
        ObjectiveTarget::PassiveAc => "passive_ac",
        ObjectiveTarget::InteractiveCausalAc => "interactive_causal_ac",
        ObjectiveTarget::PlannerDeployableModel => "planner_deployable_model",
    }
}

fn planner_deployability_report(
    enabled: bool,
    model_state_bytes: usize,
    eval_latency_seconds: f64,
    deployable_under_executor_limits: bool,
) -> Value {
    serde_json::json!({
        "enabled": enabled,
        "primary_score": "8L_B(z)+ell_D(z)",
        "diagnostics_are_secondary": true,
        "model_state_bytes": model_state_bytes,
        "snapshot_bytes": model_state_bytes,
        "clone_latency_seconds": 0.0,
        "update_latency_seconds": eval_latency_seconds.max(0.0),
        "restore_latency_seconds": 0.0,
        "sampling_support": true,
        "exact_log_probability_support": true,
        "deployable_under_executor_limits": deployable_under_executor_limits,
    })
}

fn observation_key_mode_name(mode: crate::aixi::common::ObservationKeyMode) -> &'static str {
    match mode {
        crate::aixi::common::ObservationKeyMode::First => "first",
        crate::aixi::common::ObservationKeyMode::Last => "last",
        crate::aixi::common::ObservationKeyMode::StreamHash => "stream_hash",
        crate::aixi::common::ObservationKeyMode::FullStream => "full_stream",
    }
}

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
) -> Result<(), String> {
    if candidate_contains_external_artifact(candidate) {
        Err("candidate-local external filesystem/model path references are not allowed in tune candidates".to_string())
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
        crate::api::CompressionBackend::Zpaq { method } => {
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

fn required_str<'a>(value: &'a Value, label: &str) -> Result<&'a str, String> {
    value
        .as_str()
        .ok_or_else(|| format!("{label} must be a string"))
}

fn required_bool(value: &Value, label: &str) -> Result<bool, String> {
    value
        .as_bool()
        .ok_or_else(|| format!("{label} must be a boolean"))
}

fn required_usize(value: &Value, label: &str) -> Result<usize, String> {
    let raw = value
        .as_u64()
        .ok_or_else(|| format!("{label} must be an unsigned integer"))?;
    usize::try_from(raw).map_err(|_| format!("{label} is too large"))
}

fn apply_optional_usize(value: Option<&Value>, output: &mut Option<usize>) -> Result<(), String> {
    if let Some(raw) = value {
        *output = Some(required_usize(raw, "value")?);
    }
    Ok(())
}

fn parse_annealer_kernel_profile(raw: &str) -> Result<AnnealerKernelProfile, String> {
    match raw {
        "reversible_elementary_metropolis" => {
            Ok(AnnealerKernelProfile::ReversibleElementaryMetropolis)
        }
        "compiled_uniform_metropolis_hastings" => {
            Ok(AnnealerKernelProfile::CompiledUniformMetropolisHastings)
        }
        other => Err(format!(
            "unknown annealer kernel profile '{other}', expected 'reversible_elementary_metropolis' or 'compiled_uniform_metropolis_hastings'"
        )),
    }
}

fn parse_peak_memory_mode(raw: &str) -> Result<PeakMemoryMode, String> {
    match raw {
        "process_rss_peak" => Ok(PeakMemoryMode::ProcessRssPeak),
        "backend_reported" => Ok(PeakMemoryMode::BackendReported),
        "hybrid_strict_max" => Ok(PeakMemoryMode::HybridStrictMax),
        other => Err(format!(
            "unknown rss mode '{other}', expected 'process_rss_peak', 'backend_reported', or 'hybrid_strict_max'"
        )),
    }
}

fn parse_timing_tier(raw: &str) -> Result<TimingCertificationTier, String> {
    match raw {
        "best_effort" => Ok(TimingCertificationTier::BestEffort),
        "isolated" => Ok(TimingCertificationTier::Isolated),
        "real_time" => Ok(TimingCertificationTier::RealTime),
        "deterministic_table" => Ok(TimingCertificationTier::DeterministicTable),
        other => Err(format!(
            "unknown timing tier '{other}', expected 'best_effort', 'isolated', 'real_time', or 'deterministic_table'"
        )),
    }
}

fn clean_optional_string(value: Option<&Value>) -> Option<String> {
    value
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|text| !text.is_empty())
        .map(ToOwned::to_owned)
}

fn parse_optional_non_empty_string(
    value: Option<&Value>,
    label: &str,
) -> Result<Option<String>, String> {
    match value {
        None | Some(Value::Null) => Ok(None),
        Some(Value::String(raw)) => {
            let trimmed = raw.trim();
            if trimmed.is_empty() {
                Err(format!("{label} must be a non-empty string when set"))
            } else {
                Ok(Some(trimmed.to_string()))
            }
        }
        Some(_) => Err(format!("{label} must be a string or null")),
    }
}

fn parse_cli_str<'a>(value: Option<&'a String>, label: &str) -> Result<&'a str, String> {
    value
        .map(String::as_str)
        .ok_or_else(|| format!("Error: {label} requires a value"))
}

fn parse_cli_non_empty_str<'a>(value: Option<&'a String>, label: &str) -> Result<&'a str, String> {
    let raw = parse_cli_str(value, label)?;
    if raw.trim().is_empty() {
        Err(format!("Error: {label} requires a non-empty value"))
    } else {
        Ok(raw)
    }
}

fn parse_cli_usize(value: Option<&String>, label: &str) -> Result<usize, String> {
    let raw = parse_cli_str(value, label)?;
    raw.parse::<usize>()
        .map_err(|_| format!("Error: {label} expects an unsigned integer"))
}

fn compiled_feature_set() -> Vec<&'static str> {
    let mut features = Vec::<&'static str>::new();
    if cfg!(feature = "default-backends") {
        features.push("default-backends");
    }
    if cfg!(feature = "capability-default") {
        features.push("capability-default");
    }
    if cfg!(feature = "capability-statistical") {
        features.push("capability-statistical");
    }
    if cfg!(feature = "capability-neural") {
        features.push("capability-neural");
    }
    if cfg!(feature = "capability-archive") {
        features.push("capability-archive");
    }
    if cfg!(feature = "capability-vm") {
        features.push("capability-vm");
    }
    if cfg!(feature = "aixi") {
        features.push("aixi");
    }
    if cfg!(feature = "tuner") {
        features.push("tuner");
    }
    if cfg!(feature = "aixi-gameengine") {
        features.push("aixi-gameengine");
    }
    if cfg!(feature = "aixi-gameengine-physics") {
        features.push("aixi-gameengine-physics");
    }
    if cfg!(feature = "aixi-vm") {
        features.push("aixi-vm");
    }
    if cfg!(feature = "all-backends") {
        features.push("all-backends");
    }
    if cfg!(feature = "backend-rosa") {
        features.push("backend-rosa");
    }
    if cfg!(feature = "backend-ctw") {
        features.push("backend-ctw");
    }
    if cfg!(feature = "backend-match") {
        features.push("backend-match");
    }
    if cfg!(feature = "backend-ppmd") {
        features.push("backend-ppmd");
    }
    if cfg!(feature = "backend-sequitur") {
        features.push("backend-sequitur");
    }
    if cfg!(feature = "backend-mixture") {
        features.push("backend-mixture");
    }
    if cfg!(feature = "backend-particle") {
        features.push("backend-particle");
    }
    if cfg!(feature = "backend-calibrated") {
        features.push("backend-calibrated");
    }
    if cfg!(feature = "backend-mamba") {
        features.push("backend-mamba");
    }
    if cfg!(feature = "backend-rwkv") {
        features.push("backend-rwkv");
    }
    if cfg!(feature = "backend-zpaq") {
        features.push("backend-zpaq");
    }
    if cfg!(feature = "cli") {
        features.push("cli");
    }
    if cfg!(feature = "vm") {
        features.push("vm");
    }
    features
}

fn annealer_kernel_profile_name(value: AnnealerKernelProfile) -> &'static str {
    match value {
        AnnealerKernelProfile::ReversibleElementaryMetropolis => "reversible_elementary_metropolis",
        AnnealerKernelProfile::CompiledUniformMetropolisHastings => {
            "compiled_uniform_metropolis_hastings"
        }
    }
}

fn peak_memory_mode_name(value: PeakMemoryMode) -> &'static str {
    match value {
        PeakMemoryMode::ProcessRssPeak => "process_rss_peak",
        PeakMemoryMode::BackendReported => "backend_reported",
        PeakMemoryMode::HybridStrictMax => "hybrid_strict_max",
    }
}

fn timing_tier_name(value: TimingCertificationTier) -> &'static str {
    match value {
        TimingCertificationTier::BestEffort => "best_effort",
        TimingCertificationTier::Isolated => "isolated",
        TimingCertificationTier::RealTime => "real_time",
        TimingCertificationTier::DeterministicTable => "deterministic_table",
    }
}

fn controller_kind_name(controller: &crate::spec::CompiledTuneController) -> &'static str {
    match controller {
        crate::spec::CompiledTuneController::AnnealedHillClimbing(_) => "annealed_hill_climbing",
        crate::spec::CompiledTuneController::McAixiFacCtw(_) => "mc_aixi_fac_ctw",
        crate::spec::CompiledTuneController::AiqiDiscounted(_) => "aiqi_discounted",
        crate::spec::CompiledTuneController::AiqiWarmstartExactJh(_) => "aiqi_warmstart_exact_jh",
    }
}

fn planner_completed_status(controller: &crate::spec::CompiledTuneController) -> &'static str {
    match controller {
        crate::spec::CompiledTuneController::McAixiFacCtw(_) => "completed_mc_aixi_fac_ctw",
        crate::spec::CompiledTuneController::AiqiDiscounted(_) => "completed_aiqi_discounted",
        crate::spec::CompiledTuneController::AiqiWarmstartExactJh(_) => {
            "completed_aiqi_warmstart_exact_jh"
        }
        crate::spec::CompiledTuneController::AnnealedHillClimbing(_) => "completed_annealed",
    }
}

fn planner_runtime_path_name(controller: &crate::spec::CompiledTuneController) -> &'static str {
    match controller {
        crate::spec::CompiledTuneController::McAixiFacCtw(_) => {
            "finite_mutation_agent_bridge_mcaixi_fac_ctw"
        }
        crate::spec::CompiledTuneController::AiqiDiscounted(_) => {
            "finite_mutation_agent_bridge_aiqi_discounted"
        }
        crate::spec::CompiledTuneController::AiqiWarmstartExactJh(_) => {
            "finite_mutation_agent_bridge_aiqi_warmstart_exact_jh"
        }
        crate::spec::CompiledTuneController::AnnealedHillClimbing(_) => {
            "reversible_elementary_metropolis"
        }
    }
}

fn theorem_claims_report(
    theorem: &TuneTheoremConfig,
    verified: &VerifiedTheoremInputs,
    controller: &crate::spec::CompiledTuneController,
    dataset: &LoadedDataset,
    search: &SearchSummary,
) -> Value {
    serde_json::json!({
        "exact_finite_mdp": theorem_claim_status(
            theorem.claim_exact_finite_mdp,
            exact_finite_mdp_missing_prereqs(theorem, verified, controller, search),
            &[
                "Assumptions finite-Z/no-hidden-state are represented by finite compiled mutation alphabet",
                "Timing tier is theorem-admissible",
                "Verified determinism/deadline certificate is present unless verified deterministic_table is used",
            ],
        ),
        "exact_observed_markov": theorem_claim_status(
            theorem.claim_exact_observed_markov,
            exact_observed_markov_missing_prereqs(theorem, verified, controller, search),
            &[
                "All exact finite-MDP prerequisites hold",
                "Exact-state observation encoder reference is present",
                "Verified exact-state observation certificate is present",
            ],
        ),
        "planner_convergence": theorem_claim_status(
            theorem.claim_planner_convergence,
            planner_convergence_missing_prereqs(theorem, verified, controller, search),
            &[
                "Controller is MC-AIXI(FAC-CTW)",
                "Exact finite-MDP prerequisites hold",
                "Exact objective-difference reward semantics are active",
            ],
        ),
        "refs": {
            "proof_boundary": theorem_proof_boundary_report(verified),
            "timing_certification_tier": timing_tier_name(theorem.timing_certification_tier),
            "determinism_deadline_certificate": theorem.determinism_deadline_certificate,
            "observation_adapter_spec_ref": theorem.observation_adapter_spec_ref.as_deref().unwrap_or(OBSERVATION_ADAPTER_DECLARATION),
            "exact_state_encoder_spec_ref": theorem.exact_state_encoder_spec_ref,
            "exact_state_observation_basis": {
                "verified_certificate": verified.exact_state_observation.is_some(),
                "certificate": verified.exact_state_observation.as_ref().map(VerifiedExactStateObservationCertificate::to_json_value),
            },
            "scalar_representation_ref": theorem.scalar_representation_ref.as_deref().unwrap_or(SCALAR_REPRESENTATION_DECLARATION),
            "dataset_kind": dataset_kind_name(dataset.kind),
            "dataset_lowering_version": dataset.lowering_version,
            "target_domain_support_hash": dataset.target_domain_support_hash,
            "causal_header_profile_hash": dataset.causal_header_profile_hash,
            "verified": verified.to_json_value(),
        }
    })
}

fn theorem_proof_boundary_report(verified: &VerifiedTheoremInputs) -> Value {
    serde_json::json!({
        "finite_planner_state_certificate": certificate_boundary_kind(verified.finite_planner_state.as_ref()),
        "no_hidden_state_certificate": certificate_boundary_kind(verified.no_hidden_state.as_ref()),
        "determinism_deadline_certificate": certificate_boundary_kind(verified.determinism_deadline.as_ref()),
        "exact_reward_encoding_certificate": if verified.exact_reward_encoding.is_some() {
            "certified_by_checked_artifact"
        } else {
            "operational_only_uncertified"
        },
        "exact_state_observation_certificate": if verified.exact_state_observation.is_some() {
            "certified_by_checked_artifact"
        } else {
            "operational_only_uncertified"
        },
        "deterministic_evaluator_table": if verified.deterministic_table.is_some() {
            "certified_by_checked_artifact"
        } else {
            "operational_only_uncertified"
        },
        "generic_certificate_semantics": "content_hash_and_domain_context_checked_external_certificate",
    })
}

fn certificate_boundary_kind(value: Option<&VerifiedCertificate>) -> &'static str {
    if value.is_some() {
        "certified_by_external_certificate"
    } else {
        "operational_only_uncertified"
    }
}

fn theorem_claim_status(
    requested: bool,
    missing_prereqs: Vec<&'static str>,
    certified_basis: &[&'static str],
) -> Value {
    if !requested {
        serde_json::json!({
            "requested": false,
            "status": "disabled",
            "missing_prerequisites": [],
            "certified_basis": [],
        })
    } else if missing_prereqs.is_empty() {
        serde_json::json!({
            "requested": true,
            "status": "certified",
            "missing_prerequisites": [],
            "certified_basis": certified_basis,
        })
    } else {
        serde_json::json!({
            "requested": true,
            "status": "uncertified",
            "missing_prerequisites": missing_prereqs,
            "certified_basis": [],
        })
    }
}

fn exact_finite_mdp_missing_prereqs(
    theorem: &TuneTheoremConfig,
    verified: &VerifiedTheoremInputs,
    controller: &crate::spec::CompiledTuneController,
    search: &SearchSummary,
) -> Vec<&'static str> {
    let mut missing = Vec::new();
    match controller {
        crate::spec::CompiledTuneController::AnnealedHillClimbing(_) => {
            missing.push("planner_family_controller");
        }
        crate::spec::CompiledTuneController::AiqiDiscounted(_) => {
            missing.push("exact_objective_difference_controller");
        }
        crate::spec::CompiledTuneController::McAixiFacCtw(_)
        | crate::spec::CompiledTuneController::AiqiWarmstartExactJh(_) => {}
    }
    if verified.finite_planner_state.is_none() {
        missing.push("verified_finite_planner_state_certificate");
    }
    if verified.no_hidden_state.is_none() {
        missing.push("verified_no_hidden_state_or_inert_state_certificate");
    }
    if verified.exact_reward_encoding.is_none() {
        missing.push("verified_exact_reward_encoding_certificate");
    }
    if !verified.timing_certified(theorem) {
        missing.push("theorem_certified_timing_or_deterministic_table");
    }
    if theorem.scalar_representation_ref.is_none() {
        missing.push("scalar_representation_ref");
    }
    if search.best_eval.objective_bits.is_finite() {
        missing
    } else {
        missing.push("finite_deployable_objective");
        missing
    }
}

fn exact_observed_markov_missing_prereqs(
    theorem: &TuneTheoremConfig,
    verified: &VerifiedTheoremInputs,
    controller: &crate::spec::CompiledTuneController,
    search: &SearchSummary,
) -> Vec<&'static str> {
    let mut missing = exact_finite_mdp_missing_prereqs(theorem, verified, controller, search);
    if theorem.exact_state_encoder_spec_ref.is_none() {
        missing.push("exact_state_encoder_spec_ref");
    }
    if verified.exact_state_observation.is_none() {
        missing.push("verified_exact_state_observation_certificate");
    }
    missing
}

fn planner_convergence_missing_prereqs(
    theorem: &TuneTheoremConfig,
    verified: &VerifiedTheoremInputs,
    controller: &crate::spec::CompiledTuneController,
    search: &SearchSummary,
) -> Vec<&'static str> {
    let mut missing = exact_finite_mdp_missing_prereqs(theorem, verified, controller, search);
    if !matches!(
        controller,
        crate::spec::CompiledTuneController::McAixiFacCtw(_)
    ) {
        missing.push("mc_aixi_fac_ctw_controller");
    }
    missing
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
    match mode {
        PeakMemoryMode::ProcessRssPeak => process,
        PeakMemoryMode::BackendReported => cgroup_peak_memory_bytes().unwrap_or(process),
        PeakMemoryMode::HybridStrictMax => cgroup_peak_memory_bytes().unwrap_or(0).max(process),
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
fn peak_rss_bytes_for_pid(pid: libc::pid_t) -> Option<u64> {
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

#[cfg(target_os = "linux")]
fn peak_memory_bytes_for_pid(pid: libc::pid_t, mode: PeakMemoryMode) -> Option<u64> {
    let process = peak_rss_bytes_for_pid(pid).unwrap_or(0);
    match mode {
        PeakMemoryMode::ProcessRssPeak => Some(process),
        PeakMemoryMode::BackendReported => cgroup_peak_memory_bytes().or(Some(process)),
        PeakMemoryMode::HybridStrictMax => {
            Some(cgroup_peak_memory_bytes().unwrap_or(0).max(process))
        }
    }
}

#[cfg(not(target_os = "linux"))]
fn peak_rss_bytes_for_pid(_pid: libc::pid_t) -> Option<u64> {
    None
}

#[cfg(not(target_os = "linux"))]
fn peak_memory_bytes_for_pid(_pid: libc::pid_t, _mode: PeakMemoryMode) -> Option<u64> {
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(feature = "backend-ctw")]
    use crate::aixi::common::{ActionAlphabet, ObservationKeyMode};
    use crate::aixi::warmstart::WarmStartExactJhTransition;
    use crate::api::CompressionBackend;
    #[cfg(feature = "backend-ctw")]
    use crate::api::RateBackend;
    #[cfg(feature = "backend-ctw")]
    use crate::compression::FramingMode;
    #[cfg(feature = "backend-ctw")]
    use crate::spec::{
        AiqiDiscountedTuneControllerSpec, AnnealedHillClimbingTuneControllerSpec, AssetBinding,
        McAixiFacCtwTuneControllerSpec, SpecDocument, TuneBoundsSpec, TuneControllerSpec,
        TunePlannerInterfaceSpec, TuneSpec, WarmStartExactJhTuneControllerSpec,
    };
    #[cfg(feature = "backend-ctw")]
    use std::time::{SystemTime, UNIX_EPOCH};

    #[cfg(feature = "backend-ctw")]
    fn temp_path(prefix: &str, suffix: &str) -> std::path::PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock")
            .as_nanos();
        std::env::temp_dir().join(format!("infotheory-tuner-{prefix}-{nanos}{suffix}"))
    }

    #[cfg(feature = "backend-ctw")]
    fn sample_tune_spec(dataset_path: &str, output_path: &str, report_path: &str) -> TuneSpec {
        TuneSpec {
            assets: vec![AssetBinding {
                id: "dataset".to_string(),
                path: dataset_path.to_string(),
            }],
            input_asset: "dataset".to_string(),
            baseline_candidate: CompressionBackend::Rate {
                rate_backend: RateBackend::Ctw { depth: 8 },
                coder: crate::coders::CoderType::AC,
                framing: FramingMode::Framed,
            },
            controller: TuneControllerSpec::AnnealedHillClimbing(
                AnnealedHillClimbingTuneControllerSpec {
                    max_mutation_radius: 1,
                },
            ),
            bounds: TuneBoundsSpec {
                allowed_backends: vec!["ctw".to_string()],
                forbidden_backends: Vec::new(),
                parameter_ranges: Vec::new(),
                max_experts: 2,
                max_mixture_nesting_depth: 1,
                min_experts: Some(1),
                allow_duplicate_experts: Some(false),
                required_experts: Vec::new(),
                forbidden_expert_pairs: Vec::new(),
            },
            eval_time_limit_seconds: 1.0,
            time_budget_seconds: 2.0,
            min_throughput_bytes_per_second: 1.0,
            max_memory_bytes: u64::MAX,
            output_config_path: output_path.to_string(),
            seed: 7,
            report_path: Some(report_path.to_string()),
        }
    }

    #[cfg(feature = "backend-ctw")]
    fn action_alphabet(n: usize) -> ActionAlphabet {
        ActionAlphabet::try_from_usize(n).expect("test action alphabet must be non-zero")
    }

    #[cfg(feature = "backend-ctw")]
    fn causal_dataset_value(codec_hash: &str, payload_key: &str, payload: Value) -> Value {
        let mut object = serde_json::Map::new();
        object.insert("schema_version".to_string(), serde_json::json!(1));
        object.insert("environment_id".to_string(), serde_json::json!("test-env"));
        object.insert(
            "environment_config_crc32".to_string(),
            serde_json::json!("00000000"),
        );
        object.insert("codec_hash".to_string(), serde_json::json!(codec_hash));
        object.insert(
            "reset_convention".to_string(),
            serde_json::json!("reset-before-episode"),
        );
        object.insert(
            "action_alphabet".to_string(),
            serde_json::json!({"size": 2}),
        );
        object.insert(
            "percept_schema".to_string(),
            serde_json::json!({
                "encoding": "bytes",
                "channels": [{"channel": "percept", "domain": "bytes"}],
            }),
        );
        object.insert(
            "reward_encoding".to_string(),
            serde_json::json!({
                "encoding": "bytes",
                "channel": "reward",
                "domain": "binary",
            }),
        );
        object.insert(
            "terminal_encoding".to_string(),
            serde_json::json!({
                "encoding": "bytes",
                "channel": "terminal",
                "domain": "binary",
            }),
        );
        object.insert("collection_policy".to_string(), serde_json::json!("test"));
        object.insert(
            "target_domains".to_string(),
            serde_json::json!({
                "bytes": {"kind": "byte_alphabet"},
                "binary": {"kind": "enumerated_payloads", "payloads": [[0], [1]]}
            }),
        );
        object.insert(
            "event_grammar".to_string(),
            serde_json::json!({
                "context_channels": ["action"],
                "observe_target_no_score": [
                    {"channel": "percept", "domain": "bytes"},
                    {"channel": "percept", "domain": "binary"},
                    {"channel": "reward", "domain": "binary"},
                    {"channel": "terminal", "domain": "binary"}
                ],
                "target": [
                    {"channel": "percept", "domain": "bytes"},
                    {"channel": "percept", "domain": "binary"},
                    {"channel": "reward", "domain": "binary"},
                    {"channel": "terminal", "domain": "binary"}
                ]
            }),
        );
        object.insert(payload_key.to_string(), payload);
        Value::Object(object)
    }

    #[cfg(feature = "backend-ctw")]
    fn planner_interface_for_baseline(candidate: &CompressionBackend) -> TunePlannerInterfaceSpec {
        let json = crate::spec::compression_backend_to_json_value(candidate)
            .expect("baseline candidate must serialize");
        let actions = (collect_numeric_leaves(&json).len() * 2).max(1);
        TunePlannerInterfaceSpec {
            observation_bits: 8,
            observation_stream_len: 1,
            observation_key_mode: ObservationKeyMode::FullStream,
            reward_bits: 16,
            agent_actions: action_alphabet(actions),
        }
    }

    #[cfg(feature = "backend-ctw")]
    fn write_test_exact_reward_certificate(
        path: &std::path::Path,
        dataset_path: &std::path::Path,
        bounds: &TuneBoundsSpec,
        controller_kind: &str,
    ) {
        let dataset = load_dataset(dataset_path).expect("load dataset for certificate");
        let evaluator_profile = EvaluatorProfile {
            dataset_kind: dataset.kind,
            objective_target: dataset.objective_target,
            dataset_lowering_version: dataset.lowering_version,
            dataset_codec_hash: dataset.codec_hash.clone(),
            event_grammar_hash: dataset.event_grammar_hash.clone(),
            target_domain_support_hash: dataset.target_domain_support_hash.clone(),
            causal_header_profile_hash: dataset.causal_header_profile_hash.clone(),
            target_size_function: dataset.target_size_function,
            evaluator_interface_version: TUNER_EVALUATOR_INTERFACE_VERSION,
            candidate_canonicalization_version: "bounds-v1".to_string(),
            warmup_baseline_runs: 0,
            diagnostic_chunk_bytes: None,
            eval_time_limit_seconds: 1.0,
            rss_mode: PeakMemoryMode::ProcessRssPeak,
            timing_certification_tier: TimingCertificationTier::BestEffort,
            build_profile: option_env!("PROFILE").unwrap_or("unknown"),
            feature_set: compiled_feature_set(),
        };
        let reward_cert = serde_json::json!({
            "schema_version": 1,
            "kind": "exact_reward_encoding",
            "dataset_crc32": dataset.canonical_content_hash,
            "bounds_crc32": bounds_hash(bounds).expect("bounds hash"),
            "evaluator_profile_crc32": evaluator_profile.hash().expect("profile hash"),
            "controller_kind": controller_kind,
            "action_alphabet_size": 2,
            "encoding": "integer_objective_difference",
            "scalar_representation": SCALAR_REPRESENTATION_DECLARATION,
            "reward_bits": 16,
            "max_reward": 65_535u64,
        });
        std::fs::write(
            path,
            serde_json::to_vec(&reward_cert).expect("reward cert json"),
        )
        .expect("write reward cert");
    }

    #[test]
    fn parse_tune_cli_args_and_theorem_flags() {
        let args = vec![
            "infotheory".to_string(),
            "tune".to_string(),
            "spec.json".to_string(),
            "--max-evaluations".to_string(),
            "12".to_string(),
            "--timing-tier".to_string(),
            "real_time".to_string(),
            "--claim-exact-finite-mdp".to_string(),
        ];
        let parsed = parse_tune_command_args(&args).expect("parse tune args");
        assert_eq!(parsed.spec_path, "spec.json");
        assert_eq!(parsed.execution.max_evaluations, Some(12));
        assert_eq!(
            parsed.execution.theorem.timing_certification_tier,
            TimingCertificationTier::RealTime
        );
        assert!(parsed.execution.theorem.claim_exact_finite_mdp);
    }

    #[test]
    fn tune_execution_config_accepts_nested_theorem_json() {
        let value = serde_json::json!({
            "warmup_baseline_runs": 2,
            "planner_deployable_model": true,
            "theorem": {
                "claim_exact_observed_markov": true,
                "timing_certification_tier": "isolated"
            }
        });
        let cfg = TuneExecutionConfig::from_json_value(&value).expect("config parse");
        assert_eq!(cfg.warmup_baseline_runs, 2);
        assert!(cfg.planner_deployable_model);
        assert!(cfg.theorem.claim_exact_observed_markov);
        assert_eq!(
            cfg.theorem.timing_certification_tier,
            TimingCertificationTier::Isolated
        );
    }

    #[cfg(feature = "backend-ctw")]
    #[test]
    fn tune_planner_interface_requires_explicit_observation_key_mode() {
        let mut value =
            SpecDocument::Tune(sample_tune_spec("dataset.bin", "out.json", "report.json"))
                .to_canonical_json_value()
                .expect("canonical tune json");
        value["controller"] = serde_json::json!({
            "kind": "mc_aixi_fac_ctw",
            "interface": {
                "observation_bits": 8,
                "observation_stream_len": 1,
                "reward_bits": 8,
                "agent_actions": 1
            },
            "planner_simulations_per_step": 1
        });
        let err = match SpecDocument::parse_json_value(&value, Path::new(".")) {
            Ok(_) => panic!("missing tune observation_key_mode must fail"),
            Err(err) => err,
        };
        assert!(
            err.to_string()
                .contains("controller.interface.observation_key_mode is required"),
            "{err}"
        );
    }

    #[test]
    fn tune_execution_config_rejects_observation_certified_boolean() {
        let value = serde_json::json!({
            "theorem": {
                "exact_state_observation_certified": true
            }
        });
        let err = TuneExecutionConfig::from_json_value(&value)
            .expect_err("unchecked observation proof boolean must be rejected");
        assert!(err.contains("unknown execution config field"), "{err}");
    }

    #[test]
    fn finite_reward_map_accepts_non_contiguous_injective_symbols() {
        let value = serde_json::json!({
            "values": [
                {"objective_difference": 0, "symbol": 0},
                {"objective_difference": 3, "symbol": 7}
            ]
        });
        let map = parse_finite_reward_map(value.as_object().expect("object"), 4, 15)
            .expect("finite reward map");
        assert_eq!(map.objective_difference_to_symbol.get(&0), Some(&0));
        assert_eq!(map.objective_difference_to_symbol.get(&3), Some(&7));
        assert_eq!(map.complete_nonnegative_interval_max, None);
    }

    #[test]
    fn finite_reward_map_rejects_reachable_rewards_alias() {
        let value = serde_json::json!({
            "reachable_rewards": [
                {"objective_difference": 0, "symbol": 0},
                {"objective_difference": 1, "symbol": 1}
            ]
        });
        let err = parse_finite_reward_map(value.as_object().expect("object"), 4, 15)
            .expect_err("reachable_rewards alias must be rejected");
        assert!(err.contains("requires a 'values' array"), "{err}");
    }

    #[test]
    fn finite_reward_map_rejects_duplicate_symbols() {
        let value = serde_json::json!({
            "values": [
                {"objective_difference": 0, "symbol": 1},
                {"objective_difference": 2, "symbol": 1}
            ]
        });
        let err = parse_finite_reward_map(value.as_object().expect("object"), 4, 15)
            .expect_err("duplicate symbol must fail");
        assert!(err.contains("duplicates reward symbol"), "{err}");
    }

    #[test]
    fn finite_reward_map_rejects_incomplete_declared_interval() {
        let value = serde_json::json!({
            "complete_nonnegative_interval_max": 3,
            "values": [
                {"objective_difference": 0, "symbol": 0},
                {"objective_difference": 1, "symbol": 1},
                {"objective_difference": 3, "symbol": 3}
            ]
        });
        let err = parse_finite_reward_map(value.as_object().expect("object"), 4, 15)
            .expect_err("declared complete interval must contain every difference");
        assert!(err.contains("missing objective_difference 2"), "{err}");
    }

    #[test]
    fn exact_finite_reward_map_encodes_objective_difference_not_symbol_arithmetic() {
        let encoder = TunerRewardEncoder::ExactIntegerObjectiveDifference {
            max_reward: 2,
            objective_difference_to_symbol: Some(BTreeMap::from([(0, 0), (1, 2), (2, 1)])),
        };
        assert_eq!(encoder.encode(1.0).expect("mapped reward"), 2);
        assert_eq!(encoder.encode(2.0).expect("mapped reward"), 1);
    }

    #[cfg(all(feature = "backend-ctw", target_os = "linux"))]
    #[test]
    fn cgroup_peak_reader_parses_fixture_file() {
        let path = temp_path("cgroup-memory-peak", ".txt");
        fs::write(&path, b"12345\n").expect("write cgroup fixture");
        assert_eq!(read_u64_from_file(&path).expect("parse cgroup peak"), 12345);
        fs::write(&path, b"max\n").expect("write cgroup sentinel fixture");
        let err = read_u64_from_file(&path).expect_err("max sentinel is not a measurement");
        assert!(err.contains("unbounded sentinel"), "{err}");
        let _ = fs::remove_file(path);
    }

    #[test]
    fn executor_controls_report_reflects_requested_rss_mode() {
        let mut config = TuneExecutionConfig::default();
        config.rss_mode = PeakMemoryMode::HybridStrictMax;
        let report = executor_controls_report(&config);
        assert_eq!(report["rss_mode"]["requested"], "hybrid_strict_max");
        let effective = report["rss_mode"]["effective_measurement"]
            .as_str()
            .expect("effective measurement");
        assert!(
            effective == "max_process_rss_peak_cgroup_peak"
                || effective == "process_rss_peak_fallback",
            "{effective}"
        );
    }

    #[cfg(feature = "backend-ctw")]
    #[test]
    fn evaluator_profile_cache_key_changes_with_execution_profile_only() {
        let profile_a = EvaluatorProfile {
            dataset_kind: DatasetKind::PassiveBytes,
            objective_target: ObjectiveTarget::PassiveAc,
            dataset_lowering_version: PASSIVE_DATASET_LOWERING_VERSION,
            dataset_codec_hash: "passive-identity-bytes".to_string(),
            event_grammar_hash: "passive-target-only-byte-stream".to_string(),
            target_domain_support_hash: crc32_hex(b"passive-byte-alphabet"),
            causal_header_profile_hash: crc32_hex(b"passive-none"),
            target_size_function: "passive-bytes-len",
            evaluator_interface_version: TUNER_EVALUATOR_INTERFACE_VERSION,
            candidate_canonicalization_version: "bounds-v1".to_string(),
            warmup_baseline_runs: 0,
            diagnostic_chunk_bytes: None,
            eval_time_limit_seconds: 1.0,
            rss_mode: PeakMemoryMode::ProcessRssPeak,
            timing_certification_tier: TimingCertificationTier::BestEffort,
            build_profile: "test",
            feature_set: vec!["test"],
        };
        let mut profile_b = profile_a.clone();
        profile_b.warmup_baseline_runs = 3;
        let mut profile_c = profile_a.clone();
        profile_c.eval_time_limit_seconds = 0.5;
        let mut profile_d = profile_a.clone();
        profile_d.diagnostic_chunk_bytes = Some(4096);

        let candidate = CompressionBackend::Rate {
            rate_backend: RateBackend::Ctw { depth: 8 },
            coder: crate::coders::CoderType::AC,
            framing: FramingMode::Framed,
        };
        let candidate_bytes = candidate
            .compile()
            .expect("compile candidate")
            .canonical_bytes()
            .as_slice()
            .to_vec();
        let dataset_hash = crc32_hex(b"same-dataset");
        let key_a = cache_key_for_candidate(&candidate_bytes, &profile_a, &dataset_hash)
            .expect("cache key a");
        let key_b = cache_key_for_candidate(&candidate_bytes, &profile_b, &dataset_hash)
            .expect("cache key b");
        let key_c = cache_key_for_candidate(&candidate_bytes, &profile_c, &dataset_hash)
            .expect("cache key c");
        let key_d = cache_key_for_candidate(&candidate_bytes, &profile_d, &dataset_hash)
            .expect("cache key d");
        assert_ne!(key_a, key_b);
        assert_ne!(key_a, key_c);
        assert_ne!(key_a, key_d);
        assert_eq!(key_a.candidate_canonical_bytes, candidate_bytes);
        assert_eq!(
            key_b.candidate_canonical_bytes,
            key_a.candidate_canonical_bytes
        );
        assert_eq!(
            key_c.candidate_canonical_bytes,
            key_a.candidate_canonical_bytes
        );
        assert_eq!(key_b.dataset_identity, key_a.dataset_identity);
        assert_ne!(key_b.evaluator_profile_bytes, key_a.evaluator_profile_bytes);
        assert_ne!(key_c.evaluator_profile_bytes, key_a.evaluator_profile_bytes);
    }

    #[cfg(feature = "backend-ctw")]
    #[test]
    fn deterministic_table_evaluation_enforces_exact_objective_formula() {
        let candidate = CompressionBackend::Rate {
            rate_backend: RateBackend::Ctw { depth: 4 },
            coder: crate::coders::CoderType::AC,
            framing: FramingMode::Framed,
        }
        .compile()
        .expect("compile candidate");
        let candidate_crc32 = crc32_hex(candidate.canonical_bytes().as_slice());
        let model_bytes: usize = 17;
        let target_loss_bits = 23.5;
        let table = VerifiedDeterministicEvaluatorTable {
            base: VerifiedCertificate {
                ref_value: "test://deterministic-table".to_string(),
                content_hash: "00000000".to_string(),
            },
            rows: HashMap::from([(
                candidate_crc32,
                DeterministicEvaluatorRow {
                    status: CandidateEvalStatus::Success,
                    compressed_bytes: 3,
                    target_loss_bits,
                    elapsed_seconds: 0.25,
                    peak_memory_bytes: 16,
                },
            )]),
        };
        let dataset = LoadedDataset {
            kind: DatasetKind::PassiveBytes,
            objective_target: ObjectiveTarget::PassiveAc,
            lowering_version: PASSIVE_DATASET_LOWERING_VERSION,
            codec_hash: "passive-identity-bytes".to_string(),
            event_grammar_hash: "passive-target-only-byte-stream".to_string(),
            target_domain_support_hash: crc32_hex(b"passive-byte-alphabet"),
            causal_header_profile_hash: crc32_hex(b"passive-none"),
            target_size_function: "passive-bytes-len",
            canonical_content_hash: crc32_hex(b"dataset"),
            lowered_skeleton_hash: crc32_hex(b"passive-bytes-target-only"),
            resolved_path: "test://dataset".to_string(),
            source_size_bytes: 11,
            raw_bytes: b"hello world".to_vec(),
            events: Vec::new(),
            causal_profile: None,
            dataset_units: 11.0,
            target_events: 1,
        };

        let result = table
            .evaluate(&candidate, &dataset, model_bytes, 1.0, 1024, 1.0)
            .expect("deterministic table evaluation");
        assert_eq!(result.status, CandidateEvalStatus::Success);
        assert_eq!(result.target_loss_bits, target_loss_bits);
        assert_eq!(
            result.objective_bits,
            (model_bytes as f64 * 8.0) + target_loss_bits
        );
        assert_eq!(result.throughput_bytes_per_second, 44.0);
        assert!(result.deployable);
    }

    #[cfg(feature = "backend-ctw")]
    #[test]
    fn candidate_bounds_validation_enforces_parameter_ranges() {
        let candidate = CompressionBackend::Rate {
            rate_backend: RateBackend::Ctw { depth: 8 },
            coder: crate::coders::CoderType::AC,
            framing: FramingMode::Framed,
        };
        let bounds = TuneBoundsSpec {
            allowed_backends: vec!["ctw".to_string()],
            forbidden_backends: Vec::new(),
            parameter_ranges: vec![crate::spec::TuneParameterRangeSpec {
                parameter: "rate_backend.depth".to_string(),
                min: 4.0,
                max: 6.0,
            }],
            max_experts: 2,
            max_mixture_nesting_depth: 1,
            min_experts: Some(1),
            allow_duplicate_experts: Some(false),
            required_experts: Vec::new(),
            forbidden_expert_pairs: Vec::new(),
        };
        let err = validate_candidate_against_tune_bounds(&candidate, &bounds)
            .expect_err("depth out of range must fail");
        assert!(err.contains("rate_backend.depth"));
    }

    #[cfg(feature = "backend-ctw")]
    #[test]
    fn canonical_proposal_kernel_accounts_exact_integer_masses() {
        let candidate = CompressionBackend::Rate {
            rate_backend: RateBackend::Ctw { depth: 2 },
            coder: crate::coders::CoderType::AC,
            framing: FramingMode::Framed,
        };
        let bounds = TuneBoundsSpec {
            allowed_backends: vec!["ctw".to_string()],
            forbidden_backends: Vec::new(),
            parameter_ranges: vec![crate::spec::TuneParameterRangeSpec {
                parameter: "rate_backend.depth".to_string(),
                min: 1.0,
                max: 3.0,
            }],
            max_experts: 2,
            max_mixture_nesting_depth: 1,
            min_experts: Some(1),
            allow_duplicate_experts: Some(false),
            required_experts: Vec::new(),
            forbidden_expert_pairs: Vec::new(),
        };
        let env = SpecEnvironment::new(".");
        let current = candidate.compile_in(&env).expect("compile current");
        let current_bytes = current.canonical_bytes().as_slice().to_vec();
        let kernel =
            compile_canonical_proposal_kernel(&candidate, &bounds, 1, 1, &env, &current_bytes)
                .expect("compile proposal kernel");
        assert_eq!(kernel.total_raw_actions, 2);
        assert_eq!(kernel.transitions.len(), 2);
        assert!(
            kernel
                .transitions
                .iter()
                .all(|proposal| proposal.raw_action_count == 1)
        );

        let lower = CompressionBackend::Rate {
            rate_backend: RateBackend::Ctw { depth: 1 },
            coder: crate::coders::CoderType::AC,
            framing: FramingMode::Framed,
        };
        let lower_bytes = lower
            .compile_in(&env)
            .expect("compile lower")
            .canonical_bytes()
            .as_slice()
            .to_vec();
        assert_eq!(kernel.proposal_mass_to_canonical_bytes(&lower_bytes), 1);

        let reverse = compile_canonical_proposal_kernel(&lower, &bounds, 1, 1, &env, &lower_bytes)
            .expect("compile reverse kernel");
        assert_eq!(reverse.total_raw_actions, 2);
        assert_eq!(reverse.proposal_mass_to_canonical_bytes(&current_bytes), 1);
        assert_eq!(reverse.transitions.len(), 1);
    }

    #[cfg(feature = "backend-ctw")]
    #[test]
    fn reversible_metropolis_acceptance_uses_objective_bits_temperature() {
        let proposal = AnnealedProposal {
            candidate: CompressionBackend::Rate {
                rate_backend: RateBackend::Ctw { depth: 1 },
                coder: crate::coders::CoderType::AC,
                framing: FramingMode::Framed,
            },
            forward_raw_action_count: 1,
            forward_total_raw_actions: 2,
            reverse_raw_action_count: 1,
            reverse_total_raw_actions: 2,
        };
        let uphill = annealer_acceptance_probability(
            AnnealerKernelProfile::ReversibleElementaryMetropolis,
            3.0,
            2.0,
            &proposal,
        )
        .expect("reversible metropolis probability");
        assert!((uphill - (-1.5f64).exp()).abs() <= f64::EPSILON);
        let downhill = annealer_acceptance_probability(
            AnnealerKernelProfile::ReversibleElementaryMetropolis,
            -3.0,
            2.0,
            &proposal,
        )
        .expect("reversible metropolis probability");
        assert_eq!(downhill, 1.0);
    }

    #[cfg(feature = "backend-ctw")]
    #[test]
    fn compiled_uniform_mh_uses_hastings_ratio_for_asymmetric_boundary_mass() {
        let proposal = AnnealedProposal {
            candidate: CompressionBackend::Rate {
                rate_backend: RateBackend::Ctw { depth: 1 },
                coder: crate::coders::CoderType::AC,
                framing: FramingMode::Framed,
            },
            forward_raw_action_count: 1,
            forward_total_raw_actions: 2,
            reverse_raw_action_count: 1,
            reverse_total_raw_actions: 4,
        };
        let probability = annealer_acceptance_probability(
            AnnealerKernelProfile::CompiledUniformMetropolisHastings,
            1.0,
            1.0,
            &proposal,
        )
        .expect("mh probability");
        let expected = (-1.0f64).exp() * 0.5;
        assert!((probability - expected).abs() <= f64::EPSILON);
        let err = annealer_acceptance_probability(
            AnnealerKernelProfile::ReversibleElementaryMetropolis,
            1.0,
            1.0,
            &proposal,
        )
        .expect_err("default profile must reject asymmetric masses");
        assert!(err.contains("reversibility check"));
    }

    #[test]
    fn key_less_uses_canonical_bytes_on_objective_ties() {
        let eval = CandidateEvalResult {
            status: CandidateEvalStatus::Success,
            compressed_bytes: 0,
            elapsed_seconds: 1.0,
            effective_eval_time_limit_seconds: 1.0,
            throughput_bytes_per_second: 1.0,
            peak_memory_bytes: 1,
            target_loss_bits: 1.0,
            objective_bits: 42.0,
            deployable: true,
        };
        let smaller = vec![0x01_u8, 0x02_u8];
        let larger = vec![0x01_u8, 0x03_u8];
        assert!(key_less(&eval, &smaller, &eval, &larger));
        assert!(!key_less(&eval, &larger, &eval, &smaller));
    }

    fn decode_observation_optional_f64(bytes: &[u8], field_index: usize) -> Option<f64> {
        let mut offset = 1usize;
        for index in 0..5 {
            let present = bytes[offset];
            offset += 1;
            if present == 1 {
                let mut raw = [0_u8; 8];
                raw.copy_from_slice(&bytes[offset..offset + 8]);
                let value = f64::from_bits(u64::from_le_bytes(raw));
                offset += 8;
                if index == field_index {
                    return Some(value);
                }
            } else if index == field_index {
                return None;
            }
        }
        None
    }

    #[test]
    fn raw_observation_timeout_sets_tau_one_and_invalid_uses_sentinel() {
        let incumbent = CandidateEvalResult {
            status: CandidateEvalStatus::Success,
            compressed_bytes: 10,
            elapsed_seconds: 0.5,
            effective_eval_time_limit_seconds: 1.0,
            throughput_bytes_per_second: 2.0,
            peak_memory_bytes: 1,
            target_loss_bits: 80.0,
            objective_bits: 100.0,
            deployable: true,
        };
        let timeout = timeout_eval_result(0.25, 1, 0.25);
        let timeout_observation = TunerRawObservation::from_runtime_step(
            Some(&incumbent),
            10.0,
            Some(&timeout),
            Some(b"candidate-timeout"),
            Some(0.25),
            "evaluator_timeout",
            false,
        );
        assert_eq!(
            decode_observation_optional_f64(timeout_observation.encoded_bytes(), 2),
            Some(1.0)
        );
        let invalid = CandidateEvalResult {
            status: CandidateEvalStatus::Invalid,
            compressed_bytes: 0,
            elapsed_seconds: 0.0,
            effective_eval_time_limit_seconds: 1.0,
            throughput_bytes_per_second: 0.0,
            peak_memory_bytes: 0,
            target_loss_bits: f64::INFINITY,
            objective_bits: f64::INFINITY,
            deployable: false,
        };
        let invalid_observation = TunerRawObservation::from_runtime_step(
            Some(&incumbent),
            10.0,
            Some(&invalid),
            Some(b"candidate-invalid"),
            Some(1.0),
            "evaluator_invalid",
            false,
        );
        assert_eq!(
            decode_observation_optional_f64(invalid_observation.encoded_bytes(), 2),
            None
        );
    }

    #[cfg(feature = "backend-ctw")]
    #[test]
    fn planner_percept_encoding_distinguishes_diagnostic_tokens() {
        let interface = TunePlannerInterfaceSpec {
            observation_bits: 16,
            observation_stream_len: 2,
            observation_key_mode: ObservationKeyMode::FullStream,
            reward_bits: 8,
            agent_actions: action_alphabet(2),
        };
        let current = vec![1_u8, 2, 3];
        let incumbent_eval = CandidateEvalResult {
            status: CandidateEvalStatus::Success,
            compressed_bytes: 12,
            elapsed_seconds: 0.25,
            effective_eval_time_limit_seconds: 1.0,
            throughput_bytes_per_second: 4.0,
            peak_memory_bytes: 1,
            target_loss_bits: 96.0,
            objective_bits: 128.0,
            deployable: true,
        };
        let inapplicable = encode_tuner_planner_percept(
            &interface,
            Some(&incumbent_eval),
            16.0,
            0,
            "inapplicable_action",
            None,
            None,
            None,
            false,
        )
        .expect("inapplicable percept");
        let invalid = encode_tuner_planner_percept(
            &interface,
            Some(&incumbent_eval),
            16.0,
            0,
            "invalid_action_index",
            None,
            None,
            None,
            false,
        )
        .expect("invalid percept");
        let nondeployable = encode_tuner_planner_percept(
            &interface,
            Some(&incumbent_eval),
            16.0,
            0,
            "nondeployable_candidate",
            Some(&CandidateEvalResult {
                status: CandidateEvalStatus::Invalid,
                compressed_bytes: 0,
                elapsed_seconds: 0.0,
                effective_eval_time_limit_seconds: 1.0,
                throughput_bytes_per_second: 0.0,
                peak_memory_bytes: 0,
                target_loss_bits: f64::INFINITY,
                objective_bits: f64::INFINITY,
                deployable: false,
            }),
            Some(&current),
            Some(1.0),
            false,
        )
        .expect("nondeployable percept");
        assert_ne!(inapplicable.observations, invalid.observations);
        assert_ne!(inapplicable.observations, nondeployable.observations);
        assert_ne!(invalid.observations, nondeployable.observations);
    }

    #[test]
    fn theorem_claims_reject_float_planner_mutation_domains() {
        let actions = vec![
            PlannerMutationAction::NumericStep {
                path: "rate_backend.temperature".to_string(),
                pointer: "/rate_backend/temperature".to_string(),
                kind: NumericKind::Float,
                delta: 0.05,
            },
            PlannerMutationAction::Noop,
        ];
        let mut theorem = TuneTheoremConfig::default();
        validate_theorem_planner_mutation_domain(&actions, &theorem)
            .expect("operational run may use float mutation leaves");
        theorem.claim_exact_finite_mdp = true;
        let err = validate_theorem_planner_mutation_domain(&actions, &theorem)
            .expect_err("exact theorem claim must reject float mutation leaves");
        assert!(err.contains("theorem_finite_state_unsafe"), "{err}");
    }

    #[cfg(feature = "backend-ctw")]
    #[test]
    fn theorem_claims_continue_as_uncertified_when_requested_prereqs_are_missing() {
        let dataset_path = temp_path("dataset-theorem-policy", ".bin");
        let output_path = temp_path("output-theorem-policy", ".json");
        let report_path = temp_path("report-theorem-policy", ".json");
        std::fs::write(&dataset_path, b"theorem policy dataset").expect("write dataset");
        let mut spec = sample_tune_spec(
            dataset_path.to_str().expect("dataset path"),
            output_path.to_str().expect("output path"),
            report_path.to_str().expect("report path"),
        );
        let interface = planner_interface_for_baseline(&spec.baseline_candidate);
        spec.controller = TuneControllerSpec::McAixiFacCtw(McAixiFacCtwTuneControllerSpec {
            interface,
            planner_simulations_per_step: 2,
        });
        let best_candidate = spec.baseline_candidate.clone();
        let compiled = spec.compile().expect("compile tune spec");
        let dataset = load_dataset(&dataset_path).expect("load dataset");
        let search = SearchSummary {
            status: "completed_mc_aixi_fac_ctw",
            warning: None,
            best_candidate,
            best_candidate_crc32: "00000000".to_string(),
            best_eval: CandidateEvalResult {
                status: CandidateEvalStatus::Success,
                compressed_bytes: 8,
                elapsed_seconds: 0.1,
                effective_eval_time_limit_seconds: 1.0,
                throughput_bytes_per_second: 10.0,
                peak_memory_bytes: 1,
                target_loss_bits: 64.0,
                objective_bits: 128.0,
                deployable: true,
            },
            cache_key_digest: "00000000".to_string(),
            cache_hits: 0,
            cache_misses: 1,
            candidate_evaluations_executed: 1,
            non_warmup_candidate_results_seen: 1,
            post_baseline_candidate_results_seen: 0,
            proposals_attempted: 0,
            proposals_invalid: 0,
            self_loop_proposals: 0,
            successful_non_deployable: 0,
            final_best_move_reward: 0.0,
            controller_report: Value::Null,
        };
        let theorem = TuneTheoremConfig {
            claim_exact_finite_mdp: true,
            claim_exact_observed_markov: true,
            claim_planner_convergence: true,
            ..TuneTheoremConfig::default()
        };

        let report = theorem_claims_report(
            &theorem,
            &VerifiedTheoremInputs::default(),
            compiled.controller(),
            &dataset,
            &search,
        );
        for pointer in [
            "/exact_finite_mdp/status",
            "/exact_observed_markov/status",
            "/planner_convergence/status",
        ] {
            assert_eq!(
                report.pointer(pointer).and_then(Value::as_str),
                Some("uncertified")
            );
        }
        assert!(
            report["exact_observed_markov"]["missing_prerequisites"]
                .as_array()
                .expect("missing prerequisites")
                .iter()
                .any(|item| item.as_str() == Some("verified_exact_state_observation_certificate"))
        );

        let _ = std::fs::remove_file(dataset_path);
        let _ = std::fs::remove_file(output_path);
        let _ = std::fs::remove_file(report_path);
    }

    #[test]
    fn candidate_external_asset_references_are_rejected() {
        let candidate = CompressionBackend::Zpaq {
            method: crate::api::ZpaqMethodSpec::literal("file:./candidate-model.zpaq"),
        };
        let err = reject_candidate_local_external_artifacts(&candidate)
            .expect_err("external file reference must fail");
        assert!(err.contains("candidate-local external filesystem/model path"));
    }

    #[cfg(feature = "backend-ctw")]
    #[test]
    fn run_tune_writes_output_and_report_for_baseline_pass() {
        let dataset_path = temp_path("dataset", ".bin");
        let spec_path = temp_path("spec", ".json");
        let output_path = temp_path("output", ".json");
        let report_path = temp_path("report", ".json");
        std::fs::write(&dataset_path, b"hello baseline").expect("write dataset");

        let spec = sample_tune_spec(
            dataset_path.to_str().expect("dataset path"),
            output_path.to_str().expect("output path"),
            report_path.to_str().expect("report path"),
        );
        let spec_json = SpecDocument::Tune(spec)
            .to_canonical_json()
            .expect("spec json");
        std::fs::write(&spec_path, spec_json).expect("write spec");

        let request = TuneCommandRequest {
            spec_path: spec_path.to_string_lossy().to_string(),
            execution: TuneExecutionConfig::default(),
        };
        run_tune(&request).expect("run tune");

        let output = std::fs::read_to_string(&output_path).expect("output exists");
        assert!(output.contains("\"kind\": \"rate-ac\""));
        let report = std::fs::read_to_string(&report_path).expect("report exists");
        assert!(report.contains("\"kind\": \"tune_report\""));

        let _ = std::fs::remove_file(dataset_path);
        let _ = std::fs::remove_file(spec_path);
        let _ = std::fs::remove_file(output_path);
        let _ = std::fs::remove_file(report_path);
    }

    #[cfg(feature = "backend-ctw")]
    #[test]
    fn run_tune_fails_when_baseline_not_deployable() {
        let dataset_path = temp_path("dataset", ".bin");
        let spec_path = temp_path("spec", ".json");
        let output_path = temp_path("output", ".json");
        let report_path = temp_path("report", ".json");
        std::fs::write(&dataset_path, vec![0u8; 4096]).expect("write dataset");

        let mut spec = sample_tune_spec(
            dataset_path.to_str().expect("dataset path"),
            output_path.to_str().expect("output path"),
            report_path.to_str().expect("report path"),
        );
        spec.min_throughput_bytes_per_second = f64::MAX;
        let spec_json = SpecDocument::Tune(spec)
            .to_canonical_json()
            .expect("spec json");
        std::fs::write(&spec_path, spec_json).expect("write spec");

        let request = TuneCommandRequest {
            spec_path: spec_path.to_string_lossy().to_string(),
            execution: TuneExecutionConfig::default(),
        };
        let err = run_tune(&request).expect_err("non-deployable baseline must fail");
        assert!(err.contains("not deployable"));

        let report = std::fs::read_to_string(&report_path).expect("report exists");
        assert!(report.contains("\"status\": \"baseline_not_deployable\""));
        assert!(!output_path.exists());

        let _ = std::fs::remove_file(dataset_path);
        let _ = std::fs::remove_file(spec_path);
        let _ = std::fs::remove_file(report_path);
    }

    #[cfg(feature = "backend-ctw")]
    #[test]
    fn structured_causal_dataset_objects_lower_into_charged_targets() {
        let dataset_path = temp_path("dataset-causal", ".json");
        std::fs::write(
            &dataset_path,
            causal_dataset_value(
                "test-codec",
                "events",
                serde_json::json!([
                    {"kind": "context", "channel": "action", "bytes": [1]},
                    {"kind": "observe_target_no_score", "channel": "percept", "domain": "bytes", "bytes": [2]},
                    {"kind": "target", "channel": "percept", "domain": "bytes", "bytes": [3, 4]}
                ]),
            )
            .to_string(),
        )
        .expect("write dataset");

        let dataset = load_dataset(&dataset_path).expect("causal dataset lowers");
        assert_eq!(dataset.kind, DatasetKind::InteractiveTrace);
        assert_eq!(
            dataset.objective_target,
            ObjectiveTarget::InteractiveCausalAc
        );
        assert_eq!(dataset.lowering_version, INTERACTIVE_TRACE_LOWERING_VERSION);
        assert_eq!(dataset.codec_hash, "test-codec");
        assert_eq!(dataset.raw_bytes, vec![3, 4]);
        assert_eq!(dataset.target_events, 2);
        assert_eq!(dataset.dataset_units, 2.0);
        assert!(matches!(
            &dataset.events[0],
            LoweredCausalEvent::Context { channel, bytes }
                if channel == "action" && bytes == &[1]
        ));
        assert!(matches!(
            &dataset.events[1],
            LoweredCausalEvent::ObserveTargetNoScore {
                channel,
                domain,
                bytes,
            } if channel == "percept" && domain == "bytes" && bytes == &[2]
        ));
        assert!(matches!(
            &dataset.events[2],
            LoweredCausalEvent::Target {
                channel,
                domain,
                bytes,
                weight,
            } if channel == "percept" && domain == "bytes" && bytes == &[3] && *weight == 1.0
        ));
        assert!(matches!(
            &dataset.events[3],
            LoweredCausalEvent::Target {
                channel,
                domain,
                bytes,
                weight,
            } if channel == "percept" && domain == "bytes" && bytes == &[4] && *weight == 1.0
        ));

        let _ = std::fs::remove_file(dataset_path);
    }

    #[cfg(feature = "backend-ctw")]
    #[test]
    fn causal_prefix_lowering_resets_examples_and_replays_targets_without_score() {
        let dataset_path = temp_path("dataset-prefix-semantics", ".json");
        std::fs::write(
            &dataset_path,
            causal_dataset_value(
                "test-prefix-codec",
                "examples",
                serde_json::json!([
                    {
                        "history": [
                            {"kind": "observe_target_no_score", "channel": "percept", "domain": "bytes", "bytes": [7]}
                        ],
                        "action": [1],
                        "channel": "percept",
                        "domain": "bytes",
                        "target": [8],
                        "weight": 2.0
                    },
                    {
                        "action": [0],
                        "channel": "percept",
                        "domain": "bytes",
                        "target": [9]
                    }
                ]),
            )
            .to_string(),
        )
        .expect("write causal-prefix dataset");

        let dataset = load_dataset(&dataset_path).expect("causal-prefix dataset lowers");
        assert_eq!(dataset.kind, DatasetKind::CausalPrefixDataset);
        assert_eq!(dataset.raw_bytes, vec![8, 9]);
        assert_eq!(dataset.target_events, 2);
        assert_eq!(dataset.dataset_units, 3.0);
        assert_eq!(
            dataset
                .events
                .iter()
                .filter(|event| matches!(event, LoweredCausalEvent::Reset))
                .count(),
            2
        );
        assert!(matches!(&dataset.events[0], LoweredCausalEvent::Reset));
        assert!(matches!(
            &dataset.events[1],
            LoweredCausalEvent::ObserveTargetNoScore { bytes, .. } if bytes == &[7]
        ));
        assert!(matches!(
            &dataset.events[2],
            LoweredCausalEvent::Context { channel, bytes }
                if channel == "action" && bytes == &[1]
        ));
        assert!(matches!(
            &dataset.events[3],
            LoweredCausalEvent::Target { bytes, weight, .. }
                if bytes == &[8] && *weight == 2.0
        ));
        assert!(matches!(&dataset.events[4], LoweredCausalEvent::Reset));
        assert!(matches!(
            &dataset.events[5],
            LoweredCausalEvent::Context { channel, bytes }
                if channel == "action" && bytes == &[0]
        ));
        assert!(matches!(
            &dataset.events[6],
            LoweredCausalEvent::Target { bytes, weight, .. }
                if bytes == &[9] && *weight == 1.0
        ));

        let _ = std::fs::remove_file(dataset_path);
    }

    #[cfg(feature = "backend-ctw")]
    #[test]
    fn structured_causal_dataset_rejects_missing_header_and_charged_history() {
        let missing_header_path = temp_path("dataset-missing-causal-header", ".json");
        std::fs::write(
            &missing_header_path,
            serde_json::json!({
                "schema_version": 1,
                "events": [{"kind": "target", "bytes": [1]}]
            })
            .to_string(),
        )
        .expect("write missing-header dataset");
        let err = load_dataset(&missing_header_path).expect_err("header must be required");
        assert!(err.contains("environment_id is required"), "{err}");

        let malformed_structured_path = temp_path("dataset-malformed-structured", ".json");
        std::fs::write(
            &malformed_structured_path,
            serde_json::json!({
                "schema_version": 1,
                "codec_hash": "looks-structured"
            })
            .to_string(),
        )
        .expect("write malformed structured dataset");
        let err = load_dataset(&malformed_structured_path)
            .expect_err("structured object must not be passive");
        assert!(
            err.contains("must match a canonical tuner causal dataset kind"),
            "{err}"
        );

        let charged_history_path = temp_path("dataset-charged-history", ".json");
        std::fs::write(
            &charged_history_path,
            causal_dataset_value(
                "charged-history",
                "examples",
                serde_json::json!([{
                    "history": [{"kind": "target", "channel": "percept", "domain": "bytes", "bytes": [7]}],
                    "action": [1],
                    "channel": "percept",
                    "domain": "bytes",
                    "target": [8]
                }]),
            )
            .to_string(),
        )
        .expect("write charged-history dataset");
        let err = load_dataset(&charged_history_path).expect_err("charged history must fail");
        assert!(err.contains("observe_target_no_score"), "{err}");

        let _ = std::fs::remove_file(missing_header_path);
        let _ = std::fs::remove_file(malformed_structured_path);
        let _ = std::fs::remove_file(charged_history_path);
    }

    #[cfg(feature = "backend-ctw")]
    #[test]
    fn causal_dataset_header_and_event_grammar_are_strict() {
        let invalid_header_path = temp_path("dataset-invalid-header-types", ".json");
        std::fs::write(
            &invalid_header_path,
            serde_json::json!({
                "schema_version": 1,
                "environment_id": 7,
                "environment_config_crc32": "00000000",
                "codec_hash": "codec",
                "reset_convention": "reset-before-episode",
                "action_alphabet": {"size": 2},
                "percept_schema": {"encoding": "bytes"},
                "reward_encoding": {"encoding": "bytes"},
                "terminal_encoding": {"encoding": "bytes"},
                "collection_policy": "test",
                "events": [{"kind": "target", "channel": "percept", "domain": "bytes", "bytes": [1]}]
            })
            .to_string(),
        )
        .expect("write invalid-header dataset");
        let err = load_dataset(&invalid_header_path).expect_err("invalid header type must fail");
        assert!(err.contains("environment_id is required"), "{err}");

        let alias_event_path = temp_path("dataset-alias-event-kind", ".json");
        std::fs::write(
            &alias_event_path,
            causal_dataset_value(
                "test-codec",
                "events",
                serde_json::json!([
                    {"kind": "context", "channel": "action", "bytes": [1]},
                    {"kind": "observe", "channel": "percept", "domain": "bytes", "bytes": [2]},
                ]),
            )
            .to_string(),
        )
        .expect("write alias-event dataset");
        let err = load_dataset(&alias_event_path).expect_err("alias event kind must fail");
        assert!(err.contains("unknown causal event kind"), "{err}");

        let missing_event_grammar_path = temp_path("dataset-missing-event-grammar", ".json");
        std::fs::write(
            &missing_event_grammar_path,
            serde_json::json!({
                "schema_version": 1,
                "environment_id": "env",
                "environment_config_crc32": "00000000",
                "codec_hash": "codec",
                "reset_convention": "reset-before-episode",
                "action_alphabet": {"size": 2},
                "percept_schema": {"encoding": "bytes", "channels": [{"channel": "percept", "domain": "bytes"}]},
                "reward_encoding": {"encoding": "bytes", "channel": "reward", "domain": "binary"},
                "terminal_encoding": {"encoding": "bytes", "channel": "terminal", "domain": "binary"},
                "collection_policy": "test",
                "target_domains": {
                    "bytes": {"kind": "byte_alphabet"},
                    "binary": {"kind": "enumerated_payloads", "payloads": [[0], [1]]}
                },
                "events": [{"kind": "target", "channel": "percept", "domain": "bytes", "bytes": [1]}]
            })
            .to_string(),
        )
        .expect("write missing-event-grammar dataset");
        let err =
            load_dataset(&missing_event_grammar_path).expect_err("missing event_grammar must fail");
        assert!(err.contains("requires event_grammar"), "{err}");

        let missing_domain_path = temp_path("dataset-missing-domain", ".json");
        std::fs::write(
            &missing_domain_path,
            causal_dataset_value(
                "test-codec",
                "events",
                serde_json::json!([
                    {"kind": "context", "channel": "action", "bytes": [1]},
                    {"kind": "target", "channel": "percept", "bytes": [2]},
                ]),
            )
            .to_string(),
        )
        .expect("write missing-domain dataset");
        let err = load_dataset(&missing_domain_path).expect_err("missing domain must fail");
        assert!(err.contains(".domain is required"), "{err}");

        let _ = std::fs::remove_file(invalid_header_path);
        let _ = std::fs::remove_file(alias_event_path);
        let _ = std::fs::remove_file(missing_event_grammar_path);
        let _ = std::fs::remove_file(missing_domain_path);
    }

    #[cfg(feature = "backend-ctw")]
    #[test]
    fn byte_alphabet_payloads_expand_to_single_byte_events() {
        let dataset_path = temp_path("dataset-byte-alphabet-expand", ".json");
        std::fs::write(
            &dataset_path,
            causal_dataset_value(
                "byte-expand-codec",
                "events",
                serde_json::json!([
                    {"kind": "observe_target_no_score", "channel": "percept", "domain": "bytes", "bytes": [3, 4]},
                    {"kind": "target", "channel": "percept", "domain": "bytes", "bytes": [7, 8], "weight": 2.0}
                ]),
            )
            .to_string(),
        )
        .expect("write byte-alphabet expansion dataset");
        let dataset = load_dataset(&dataset_path).expect("dataset lowers");
        assert!(matches!(
            &dataset.events[0],
            LoweredCausalEvent::ObserveTargetNoScore { bytes, .. } if bytes == &[3]
        ));
        assert!(matches!(
            &dataset.events[1],
            LoweredCausalEvent::ObserveTargetNoScore { bytes, .. } if bytes == &[4]
        ));
        assert!(matches!(
            &dataset.events[2],
            LoweredCausalEvent::Target { bytes, weight, .. } if bytes == &[7] && *weight == 2.0
        ));
        assert!(matches!(
            &dataset.events[3],
            LoweredCausalEvent::Target { bytes, weight, .. } if bytes == &[8] && *weight == 2.0
        ));
        let _ = std::fs::remove_file(dataset_path);
    }

    #[cfg(feature = "backend-ctw")]
    #[test]
    fn byte_alphabet_empty_payload_is_rejected() {
        let dataset_path = temp_path("dataset-byte-alphabet-empty", ".json");
        std::fs::write(
            &dataset_path,
            causal_dataset_value(
                "byte-empty-codec",
                "events",
                serde_json::json!([
                    {"kind": "target", "channel": "percept", "domain": "bytes", "bytes": []}
                ]),
            )
            .to_string(),
        )
        .expect("write byte-alphabet empty payload dataset");
        let err = load_dataset(&dataset_path).expect_err("empty byte-alphabet payload must fail");
        assert!(err.contains("must contain at least one byte"), "{err}");
        let _ = std::fs::remove_file(dataset_path);
    }

    #[cfg(feature = "backend-ctw")]
    #[test]
    fn causal_header_cross_checks_enforce_grammar_and_action_contracts() {
        let undeclared_descriptor_path = temp_path("dataset-undeclared-event-descriptor", ".json");
        std::fs::write(
            &undeclared_descriptor_path,
            causal_dataset_value(
                "descriptor-codec",
                "events",
                serde_json::json!([
                    {"kind": "target", "channel": "other", "domain": "bytes", "bytes": [1]}
                ]),
            )
            .to_string(),
        )
        .expect("write undeclared descriptor dataset");
        let err =
            load_dataset(&undeclared_descriptor_path).expect_err("undeclared descriptor must fail");
        assert!(
            err.contains("not declared in event_grammar.target"),
            "{err}"
        );

        let invalid_action_path = temp_path("dataset-invalid-action-context", ".json");
        std::fs::write(
            &invalid_action_path,
            causal_dataset_value(
                "invalid-action-codec",
                "events",
                serde_json::json!([
                    {"kind": "context", "channel": "action", "bytes": [2]},
                    {"kind": "target", "channel": "percept", "domain": "bytes", "bytes": [1]}
                ]),
            )
            .to_string(),
        )
        .expect("write invalid action dataset");
        let err = load_dataset(&invalid_action_path).expect_err("invalid action context must fail");
        assert!(err.contains("outside action_alphabet.size"), "{err}");

        let mut invalid_grammar = causal_dataset_value(
            "invalid-grammar-codec",
            "events",
            serde_json::json!([
                {"kind": "target", "channel": "percept", "domain": "bytes", "bytes": [1]}
            ]),
        );
        let grammar = invalid_grammar
            .get_mut("event_grammar")
            .and_then(Value::as_object_mut)
            .expect("event_grammar object");
        let targets = grammar
            .get_mut("target")
            .and_then(Value::as_array_mut)
            .expect("target grammar array");
        targets.push(serde_json::json!({"channel": "ghost", "domain": "ghost"}));
        let invalid_grammar_path = temp_path("dataset-invalid-grammar-domain", ".json");
        std::fs::write(
            &invalid_grammar_path,
            serde_json::to_string(&invalid_grammar).expect("invalid grammar json"),
        )
        .expect("write invalid grammar dataset");
        let err = load_dataset(&invalid_grammar_path)
            .expect_err("grammar with undeclared domain must fail");
        assert!(
            err.contains("event_grammar references undeclared target domain"),
            "{err}"
        );

        let _ = std::fs::remove_file(undeclared_descriptor_path);
        let _ = std::fs::remove_file(invalid_action_path);
        let _ = std::fs::remove_file(invalid_grammar_path);
    }

    #[cfg(feature = "backend-ctw")]
    #[test]
    fn byte_alphabet_expansion_matches_chain_rule_loss() {
        let dataset_expanded_from_multibyte = temp_path("dataset-byte-chain-multibyte", ".json");
        std::fs::write(
            &dataset_expanded_from_multibyte,
            causal_dataset_value(
                "chain-rule-codec",
                "events",
                serde_json::json!([
                    {"kind": "target", "channel": "percept", "domain": "bytes", "bytes": [65, 66]}
                ]),
            )
            .to_string(),
        )
        .expect("write multi-byte dataset");
        let dataset_explicit_singletons = temp_path("dataset-byte-chain-singletons", ".json");
        std::fs::write(
            &dataset_explicit_singletons,
            causal_dataset_value(
                "chain-rule-codec",
                "events",
                serde_json::json!([
                    {"kind": "target", "channel": "percept", "domain": "bytes", "bytes": [65]},
                    {"kind": "target", "channel": "percept", "domain": "bytes", "bytes": [66]}
                ]),
            )
            .to_string(),
        )
        .expect("write singleton dataset");

        let dataset_a = load_dataset(&dataset_expanded_from_multibyte).expect("load dataset a");
        let dataset_b = load_dataset(&dataset_explicit_singletons).expect("load dataset b");
        let candidate = CompressionBackend::Rate {
            rate_backend: RateBackend::Ctw { depth: 8 },
            coder: crate::coders::CoderType::AC,
            framing: FramingMode::Framed,
        }
        .compile()
        .expect("compile ctw candidate");
        let deadline = Instant::now() + Duration::from_secs(2);
        let (compressed_a, loss_a) =
            evaluate_candidate_causal_loss(&candidate, &dataset_a, deadline).expect("eval a");
        let (compressed_b, loss_b) =
            evaluate_candidate_causal_loss(&candidate, &dataset_b, deadline).expect("eval b");
        assert_eq!(compressed_a, compressed_b);
        assert!(
            (loss_a - loss_b).abs() < 1.0e-10,
            "loss_a={loss_a}, loss_b={loss_b}"
        );

        let _ = std::fs::remove_file(dataset_expanded_from_multibyte);
        let _ = std::fs::remove_file(dataset_explicit_singletons);
    }

    #[cfg(feature = "backend-ctw")]
    #[test]
    fn causal_dataset_domains_are_profile_fixed_and_support_checked() {
        let dataset_path = temp_path("dataset-enumerated-domain", ".json");
        std::fs::write(
            &dataset_path,
            causal_dataset_value(
                "test-enumerated-codec",
                "events",
                serde_json::json!([
                    {"kind": "context", "channel": "action", "bytes": [1]},
                    {"kind": "target", "channel": "percept", "domain": "binary", "bytes": [1]}
                ]),
            )
            .to_string(),
        )
        .expect("write enumerated-domain dataset");
        let dataset = load_dataset(&dataset_path).expect("enumerated domain dataset lowers");
        let causal_profile = dataset.causal_profile.as_ref().expect("causal profile");
        assert!(causal_profile.domains.contains_key("binary"));
        assert_eq!(
            dataset.target_domain_support_hash,
            causal_profile.domain_support_hash
        );
        assert_ne!(
            dataset.target_domain_support_hash,
            crc32_hex(b"passive-byte-alphabet")
        );

        let out_of_support_path = temp_path("dataset-enumerated-domain-out", ".json");
        std::fs::write(
            &out_of_support_path,
            causal_dataset_value(
                "test-enumerated-codec",
                "events",
                serde_json::json!([
                    {"kind": "target", "channel": "percept", "domain": "binary", "bytes": [2]}
                ]),
            )
            .to_string(),
        )
        .expect("write out-of-support dataset");
        let err = load_dataset(&out_of_support_path).expect_err("out-of-support target fails");
        assert!(err.contains("outside target-domain support"), "{err}");

        let _ = std::fs::remove_file(dataset_path);
        let _ = std::fs::remove_file(out_of_support_path);
    }

    #[cfg(feature = "backend-ctw")]
    #[test]
    fn causal_event_channel_and_domain_affect_skeleton_identity() {
        let path_a = temp_path("dataset-channel-a", ".json");
        let path_b = temp_path("dataset-channel-b", ".json");
        std::fs::write(
            &path_a,
            causal_dataset_value(
                "same-codec",
                "events",
                serde_json::json!([
                    {"kind": "target", "channel": "percept", "domain": "bytes", "bytes": [7]}
                ]),
            )
            .to_string(),
        )
        .expect("write channel a");
        std::fs::write(
            &path_b,
            causal_dataset_value(
                "same-codec",
                "events",
                serde_json::json!([
                    {"kind": "target", "channel": "reward", "domain": "binary", "bytes": [1]}
                ]),
            )
            .to_string(),
        )
        .expect("write channel b");

        let dataset_a = load_dataset(&path_a).expect("load channel a");
        let dataset_b = load_dataset(&path_b).expect("load channel b");
        assert_ne!(dataset_a.event_grammar_hash, dataset_b.event_grammar_hash);
        assert_eq!(dataset_a.codec_hash, dataset_b.codec_hash);

        let _ = std::fs::remove_file(path_a);
        let _ = std::fs::remove_file(path_b);
    }

    #[test]
    fn annealer_schedule_matches_normative_log_linear_law() {
        let mid = annealer_temperature(0.5);
        let expected_mid = ANNEALER_T_MIN_BITS * (ANNEALER_T0_BITS / ANNEALER_T_MIN_BITS).powf(0.5);
        assert_eq!(annealer_progress_from_elapsed(0.0, 10.0), 0.0);
        assert_eq!(annealer_progress_from_elapsed(5.0, 10.0), 0.5);
        assert_eq!(annealer_progress_from_elapsed(20.0, 10.0), 1.0);
        assert!((annealer_temperature(0.0) - ANNEALER_T0_BITS).abs() < 1.0e-12);
        assert!((annealer_temperature(1.0) - ANNEALER_T_MIN_BITS).abs() < 1.0e-12);
        assert!((mid - expected_mid).abs() < 1.0e-12);
    }

    #[cfg(feature = "backend-ctw")]
    #[test]
    fn exact_state_observation_projection_supports_stream_hash() {
        let interface = PlannerInterfaceSpec {
            observation_bits: 3,
            observation_stream_len: 2,
            observation_key_mode: ObservationKeyMode::StreamHash,
            reward_bits: 16,
            min_reward: 0,
            max_reward: 10,
            reward_offset: 0,
            agent_actions: action_alphabet(2),
        };
        let projected =
            project_observation_output("stream_hash", &[9, 2], interface.observation_bits)
                .expect("stream_hash projection");
        assert_eq!(projected, vec![130]);
    }

    #[cfg(feature = "backend-ctw")]
    #[test]
    fn discounted_aiqi_exact_theorem_claims_remain_uncertified_by_family() {
        let controller =
            crate::spec::CompiledTuneController::AiqiDiscounted(AiqiDiscountedTuneControllerSpec {
                interface: TunePlannerInterfaceSpec {
                    observation_bits: 8,
                    observation_stream_len: 1,
                    observation_key_mode: ObservationKeyMode::FullStream,
                    reward_bits: 16,
                    agent_actions: action_alphabet(2),
                },
                planner_simulations_per_step: 1,
                return_horizon: 1,
                return_bins: 2,
                discount_factor: 0.0,
                min_improvement: 0.0,
                max_improvement: 1.0,
            });
        let candidate = CompressionBackend::Rate {
            rate_backend: RateBackend::Ctw { depth: 8 },
            coder: crate::coders::CoderType::AC,
            framing: FramingMode::Framed,
        };
        let search = SearchSummary {
            status: "test",
            warning: None,
            best_candidate: candidate,
            best_candidate_crc32: "00000000".to_string(),
            best_eval: CandidateEvalResult {
                status: CandidateEvalStatus::Success,
                compressed_bytes: 1,
                elapsed_seconds: 0.1,
                effective_eval_time_limit_seconds: 1.0,
                throughput_bytes_per_second: 10.0,
                peak_memory_bytes: 1,
                target_loss_bits: 8.0,
                objective_bits: 16.0,
                deployable: true,
            },
            cache_key_digest: "00000000".to_string(),
            cache_hits: 0,
            cache_misses: 0,
            candidate_evaluations_executed: 1,
            non_warmup_candidate_results_seen: 1,
            post_baseline_candidate_results_seen: 0,
            proposals_attempted: 0,
            proposals_invalid: 0,
            self_loop_proposals: 0,
            successful_non_deployable: 0,
            final_best_move_reward: 0.0,
            controller_report: Value::Null,
        };
        let theorem = TuneTheoremConfig {
            claim_exact_finite_mdp: true,
            scalar_representation_ref: Some(SCALAR_REPRESENTATION_DECLARATION.to_string()),
            ..TuneTheoremConfig::default()
        };
        let verified = VerifiedTheoremInputs {
            finite_planner_state: Some(VerifiedCertificate {
                ref_value: "finite.json".to_string(),
                content_hash: "00000000".to_string(),
            }),
            no_hidden_state: Some(VerifiedCertificate {
                ref_value: "hidden.json".to_string(),
                content_hash: "00000000".to_string(),
            }),
            exact_reward_encoding: Some(VerifiedExactRewardEncodingCertificate {
                base: VerifiedCertificate {
                    ref_value: "reward.json".to_string(),
                    content_hash: "00000000".to_string(),
                },
                max_reward: 65_535,
                reward_bits: 16,
                scalar_representation: SCALAR_REPRESENTATION_DECLARATION.to_string(),
                mode: VerifiedRewardEncodingMode::IntegerObjectiveDifferenceInterval,
            }),
            exact_state_observation: None,
            determinism_deadline: None,
            deterministic_table: None,
        };
        let missing = exact_finite_mdp_missing_prereqs(&theorem, &verified, &controller, &search);
        assert!(missing.contains(&"exact_objective_difference_controller"));
    }

    #[test]
    fn warmstart_trace_merge_is_content_deduplicated_and_deterministically_ordered() {
        let mut teacher = WarmStartExactJhTeacherDataset::default();
        let high_key_trace = WarmStartExactJhTeacherTrace {
            transitions: vec![WarmStartExactJhTransition {
                action: 1_u64,
                observations: vec![2],
                reward: 3,
            }],
        };
        let low_key_trace = WarmStartExactJhTeacherTrace {
            transitions: vec![WarmStartExactJhTransition {
                action: 0_u64,
                observations: vec![1],
                reward: 1,
            }],
        };
        teacher.traces.push(high_key_trace.clone());

        merge_warmstart_trace_deterministic(&mut teacher, low_key_trace.clone())
            .expect("merge distinct trace");
        let ordered_keys = teacher
            .traces
            .iter()
            .map(warmstart_trace_key)
            .collect::<Result<Vec<_>, _>>()
            .expect("trace keys");
        let mut sorted_keys = ordered_keys.clone();
        sorted_keys.sort();
        assert_eq!(ordered_keys, sorted_keys);
        assert_eq!(teacher.traces.len(), 2);

        merge_warmstart_trace_deterministic(&mut teacher, low_key_trace)
            .expect("duplicate merge remains idempotent");
        assert_eq!(teacher.traces.len(), 2);
    }

    #[cfg(feature = "backend-ctw")]
    #[test]
    fn run_tune_annealed_reports_search_activity() {
        let dataset_path = temp_path("dataset-annealed", ".bin");
        let spec_path = temp_path("spec-annealed", ".json");
        let output_path = temp_path("output-annealed", ".json");
        let report_path = temp_path("report-annealed", ".json");
        std::fs::write(&dataset_path, b"annealed-search-dataset").expect("write dataset");

        let mut spec = sample_tune_spec(
            dataset_path.to_str().expect("dataset path"),
            output_path.to_str().expect("output path"),
            report_path.to_str().expect("report path"),
        );
        spec.bounds.parameter_ranges = vec![crate::spec::TuneParameterRangeSpec {
            parameter: "rate_backend.depth".to_string(),
            min: 1.0,
            max: 16.0,
        }];
        let spec_json = SpecDocument::Tune(spec)
            .to_canonical_json()
            .expect("spec json");
        std::fs::write(&spec_path, spec_json).expect("write spec");

        let request = TuneCommandRequest {
            spec_path: spec_path.to_string_lossy().to_string(),
            execution: TuneExecutionConfig {
                max_evaluations: Some(3),
                ..TuneExecutionConfig::default()
            },
        };
        run_tune(&request).expect("run tune");

        let report = std::fs::read_to_string(&report_path).expect("report exists");
        assert!(report.contains("\"status\": \"completed_annealed\""));
        assert!(report.contains("\"proposals_attempted\":"));
        let report_json: Value = serde_json::from_str(&report).expect("report json");
        assert_eq!(
            report_json
                .pointer("/search/baseline_counts_toward_max_evaluations")
                .and_then(Value::as_bool),
            Some(true)
        );
        let non_warmup_results = report_json
            .pointer("/search/non_warmup_candidate_results_seen")
            .and_then(Value::as_u64)
            .expect("non_warmup_candidate_results_seen");
        let post_baseline_results = report_json
            .pointer("/search/post_baseline_candidate_results_seen")
            .and_then(Value::as_u64)
            .expect("post_baseline_candidate_results_seen");
        assert!((1..=3).contains(&non_warmup_results));
        assert_eq!(post_baseline_results + 1, non_warmup_results);
        let cache_calls = report_json
            .pointer("/cache/actual_evaluator_calls_excluding_warmups")
            .and_then(Value::as_u64)
            .expect("actual_evaluator_calls_excluding_warmups");
        assert_eq!(
            report_json
                .pointer("/cache/candidate_evaluations_executed")
                .and_then(Value::as_u64),
            Some(cache_calls)
        );

        let _ = std::fs::remove_file(dataset_path);
        let _ = std::fs::remove_file(spec_path);
        let _ = std::fs::remove_file(output_path);
        let _ = std::fs::remove_file(report_path);
    }

    #[cfg(feature = "backend-ctw")]
    #[test]
    fn planner_family_controller_executes_runtime_path() {
        let dataset_path = temp_path("dataset-planner", ".bin");
        let spec_path = temp_path("spec-planner", ".json");
        let output_path = temp_path("output-planner", ".json");
        let report_path = temp_path("report-planner", ".json");
        std::fs::write(&dataset_path, b"planner-controller-dataset").expect("write dataset");

        let mut spec = sample_tune_spec(
            dataset_path.to_str().expect("dataset path"),
            output_path.to_str().expect("output path"),
            report_path.to_str().expect("report path"),
        );
        spec.controller = TuneControllerSpec::McAixiFacCtw(McAixiFacCtwTuneControllerSpec {
            interface: planner_interface_for_baseline(&spec.baseline_candidate),
            planner_simulations_per_step: 8,
        });
        let bounds = spec.bounds.clone();
        let spec_json = SpecDocument::Tune(spec)
            .to_canonical_json()
            .expect("spec json");
        std::fs::write(&spec_path, spec_json).expect("write spec");
        let reward_cert_path = temp_path("reward-cert-planner", ".json");
        write_test_exact_reward_certificate(
            &reward_cert_path,
            &dataset_path,
            &bounds,
            "mc_aixi_fac_ctw",
        );

        let request = TuneCommandRequest {
            spec_path: spec_path.to_string_lossy().to_string(),
            execution: TuneExecutionConfig {
                theorem: TuneTheoremConfig {
                    exact_reward_encoding_certificate: Some(
                        reward_cert_path.to_string_lossy().to_string(),
                    ),
                    ..TuneTheoremConfig::default()
                },
                ..TuneExecutionConfig::default()
            },
        };
        run_tune(&request).expect("run tune");

        let report = std::fs::read_to_string(&report_path).expect("report exists");
        assert!(report.contains("\"status\": \"completed_mc_aixi_fac_ctw\""));
        assert!(
            report.contains("\"runtime_path\": \"finite_mutation_agent_bridge_mcaixi_fac_ctw\"")
        );

        let _ = std::fs::remove_file(dataset_path);
        let _ = std::fs::remove_file(spec_path);
        let _ = std::fs::remove_file(output_path);
        let _ = std::fs::remove_file(report_path);
        let _ = std::fs::remove_file(reward_cert_path);
    }

    #[cfg(feature = "backend-ctw")]
    #[test]
    fn executor_profile_and_dataset_mode_do_not_change_canonical_tune_identity() {
        let passive_dataset_path = temp_path("dataset-passive", ".bin");
        let trace_dataset_path = temp_path("dataset-trace", ".json");
        let prefix_dataset_path = temp_path("dataset-prefix", ".json");
        let output_path = temp_path("output-identity", ".json");
        let report_path = temp_path("report-identity", ".json");
        std::fs::write(&passive_dataset_path, b"identity-passive").expect("write passive");
        std::fs::write(
            &trace_dataset_path,
            causal_dataset_value(
                "identity-trace-codec",
                "events",
                serde_json::json!([
                    {"kind": "context", "channel": "action", "bytes": [1]},
                    {"kind": "target", "channel": "percept", "domain": "bytes", "bytes": [2, 3]}
                ]),
            )
            .to_string(),
        )
        .expect("write trace");
        std::fs::write(
            &prefix_dataset_path,
            causal_dataset_value(
                "identity-prefix-codec",
                "examples",
                serde_json::json!([
                    {
                        "history": [{"kind": "observe_target_no_score", "channel": "percept", "domain": "bytes", "bytes": [7]}],
                        "action": [1],
                        "channel": "percept",
                        "domain": "bytes",
                        "target": [8],
                        "weight": 2.0
                    }
                ]),
            )
            .to_string(),
        )
        .expect("write prefix");

        let dataset_paths = [
            passive_dataset_path.as_path(),
            trace_dataset_path.as_path(),
            prefix_dataset_path.as_path(),
        ];
        for dataset_path in dataset_paths {
            let mut base = sample_tune_spec(
                dataset_path.to_str().expect("dataset path"),
                output_path.to_str().expect("output path"),
                report_path.to_str().expect("report path"),
            );
            let interface = planner_interface_for_baseline(&base.baseline_candidate);
            let controllers = vec![
                TuneControllerSpec::AnnealedHillClimbing(AnnealedHillClimbingTuneControllerSpec {
                    max_mutation_radius: 1,
                }),
                TuneControllerSpec::McAixiFacCtw(McAixiFacCtwTuneControllerSpec {
                    interface: interface.clone(),
                    planner_simulations_per_step: 2,
                }),
                TuneControllerSpec::AiqiDiscounted(AiqiDiscountedTuneControllerSpec {
                    interface: interface.clone(),
                    planner_simulations_per_step: 2,
                    return_horizon: 1,
                    return_bins: 2,
                    discount_factor: 0.5,
                    min_improvement: 0.0,
                    max_improvement: 1.0,
                }),
                TuneControllerSpec::AiqiWarmstartExactJh(WarmStartExactJhTuneControllerSpec {
                    interface: interface.clone(),
                    planner_simulations_per_step: 2,
                    return_horizon: 1,
                    warmstart_teacher_dataset_asset: "teacher".to_string(),
                    label_phase_period: 1,
                }),
            ];
            for controller in controllers {
                base.controller = controller;
                base.assets.retain(|asset| asset.id == "dataset");
                if matches!(base.controller, TuneControllerSpec::AiqiWarmstartExactJh(_)) {
                    base.assets.push(AssetBinding {
                        id: "teacher".to_string(),
                        path: passive_dataset_path.to_string_lossy().to_string(),
                    });
                }
                let canonical_a = SpecDocument::Tune(base.clone())
                    .to_canonical_json()
                    .expect("canonical tune a");
                let mut execution = TuneExecutionConfig {
                    max_evaluations: Some(1),
                    warmup_baseline_runs: 3,
                    ..TuneExecutionConfig::default()
                };
                execution.theorem.timing_certification_tier = TimingCertificationTier::RealTime;
                execution.theorem.determinism_deadline_certificate =
                    Some("cert://deadline".to_string());
                let canonical_b = SpecDocument::Tune(base.clone())
                    .to_canonical_json()
                    .expect("canonical tune b");
                assert_eq!(canonical_a, canonical_b);
                let loaded = load_dataset(dataset_path).expect("dataset mode loads");
                let profile_a = EvaluatorProfile {
                    dataset_kind: loaded.kind,
                    objective_target: loaded.objective_target,
                    dataset_lowering_version: loaded.lowering_version,
                    dataset_codec_hash: loaded.codec_hash.clone(),
                    event_grammar_hash: loaded.event_grammar_hash.clone(),
                    target_domain_support_hash: loaded.target_domain_support_hash.clone(),
                    causal_header_profile_hash: loaded.causal_header_profile_hash.clone(),
                    target_size_function: loaded.target_size_function,
                    evaluator_interface_version: TUNER_EVALUATOR_INTERFACE_VERSION,
                    candidate_canonicalization_version: "bounds-v1".to_string(),
                    warmup_baseline_runs: 0,
                    diagnostic_chunk_bytes: None,
                    eval_time_limit_seconds: base.eval_time_limit_seconds,
                    rss_mode: PeakMemoryMode::ProcessRssPeak,
                    timing_certification_tier: TimingCertificationTier::BestEffort,
                    build_profile: "test",
                    feature_set: vec!["test"],
                };
                let mut profile_b = profile_a.clone();
                profile_b.warmup_baseline_runs = execution.warmup_baseline_runs;
                profile_b.timing_certification_tier = execution.theorem.timing_certification_tier;
                assert_ne!(
                    profile_a.hash().expect("profile a"),
                    profile_b.hash().expect("profile b")
                );
            }
        }

        let _ = std::fs::remove_file(passive_dataset_path);
        let _ = std::fs::remove_file(trace_dataset_path);
        let _ = std::fs::remove_file(prefix_dataset_path);
    }
}
