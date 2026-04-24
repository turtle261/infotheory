//! Canonical top-level specification document schema types.

use crate::aixi::common::ObservationKeyMode;
use crate::api::{CompressionBackend, RateBackend};
use crate::spec::core::{
    AssetRef, CanonicalBytes, CompiledCompressionBackend, CompiledRateBackend,
    ValidatedCompressionBackend, ValidatedRateBackend,
};
use std::path::PathBuf;
use std::sync::Arc;

/// Stable identifier for an external asset binding.
pub type AssetId = String;

/// Filesystem binding for a named external asset referenced by a spec document.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AssetBinding {
    /// Stable asset identifier used inside canonical specs.
    pub id: AssetId,
    /// Filesystem path used to resolve the asset at runtime.
    pub path: String,
}

/// Resolved runtime asset binding derived from a canonical asset identifier.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ResolvedAssetBinding {
    /// Stable asset identifier used inside canonical specs.
    pub id: AssetId,
    /// Resolved asset handle in the current compilation environment.
    pub asset: AssetRef,
}

/// Built-in non-VM environment choices available to planner runs.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BuiltinEnvironmentSpec {
    /// GameEngine biased coin-flip environment.
    CoinFlip,
    /// GameEngine biased rock-paper-scissor environment.
    BiasedRockPaperScissor,
    /// GameEngine Kuhn poker environment.
    KuhnPoker,
    /// GameEngine extended tiger environment.
    ExtendedTiger,
    /// GameEngine tic-tac-toe environment.
    TicTacToe,
    /// GameEngine blackjack environment.
    Blackjack,
    /// GameEngine platformer environment.
    Platformer,
}

/// Shared-memory persistence policy for Nyx VM environments.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SharedMemoryPolicySpec {
    /// Preserve the region across resets.
    Preserve,
    /// Reset from the snapshot baseline each iteration.
    Snapshot,
}

/// Reward shaping configuration for VM environments.
#[derive(Clone, Debug, PartialEq)]
pub enum VmRewardShapingSpec {
    /// Entropy reduction relative to a baseline asset.
    EntropyReduction {
        /// Asset identifier providing baseline bytes.
        baseline_asset: AssetId,
        /// Max order hint for the entropy estimator.
        max_order: i64,
        /// Linear scale applied to the shaping reward.
        scale: f64,
        /// Optional bonus applied on crash exits.
        crash_bonus: Option<i64>,
        /// Optional bonus applied on timeout exits.
        timeout_bonus: Option<i64>,
    },
    /// Trace entropy shaping using online trace bytes.
    TraceEntropy {
        /// Max order hint for the entropy estimator.
        max_order: i64,
        /// Linear scale applied to the shaping reward.
        scale: f64,
        /// Whether to normalize by trace length.
        normalize: bool,
    },
}

/// Reward policy for canonical VM environment specs.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum VmRewardPolicySpec {
    /// Parse reward directly from the guest protocol.
    FromGuest,
    /// Pattern-based reward shaping against guest output.
    Pattern {
        /// Pattern searched in guest output.
        pattern: String,
        /// Reward applied when the pattern does not match.
        base_reward: i64,
        /// Additional reward applied on match.
        bonus_reward: i64,
    },
}

/// Optional information-theoretic action filtering for VM runs.
#[derive(Clone, Debug, PartialEq)]
pub struct VmActionFilterSpec {
    /// Minimum entropy threshold.
    pub min_entropy: Option<f64>,
    /// Maximum entropy threshold.
    pub max_entropy: Option<f64>,
    /// Minimum intrinsic dependence threshold.
    pub min_intrinsic_dependence: Option<f64>,
    /// Minimum novelty threshold.
    pub min_novelty: Option<f64>,
    /// Optional prior asset used for novelty scoring.
    pub novelty_prior_asset: Option<AssetId>,
    /// Max-order hint for entropy estimators.
    pub max_order: i64,
    /// Reward assigned when an action is rejected.
    pub reject_reward: Option<i64>,
}

/// Optional VM trace collection settings.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct VmTraceSpec {
    /// Shared-memory region name carrying trace bytes.
    pub shared_region_name: Option<String>,
    /// Maximum trace bytes collected per step.
    pub max_bytes: usize,
    /// Whether the trace model resets on episode boundaries.
    pub reset_on_episode: bool,
}

/// Canonical observation derivation modes for VM environments.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum VmObservationPolicySpec {
    /// Parse observations from the guest protocol.
    FromGuest,
    /// Hash guest output into the observation stream.
    OutputHash,
    /// Use raw guest output bytes directly.
    RawOutput,
    /// Read observations from shared memory.
    SharedMemory,
}

/// Canonical normalization modes for VM observation streams.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum VmObservationStreamModeSpec {
    /// Pad short streams and truncate long streams.
    PadTruncate,
    /// Only pad short streams.
    Pad,
    /// Only truncate long streams.
    Truncate,
}

/// Canonical payload encodings for VM action and protocol payloads.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum VmPayloadEncodingSpec {
    /// Interpret payload strings as UTF-8 text.
    Utf8,
    /// Interpret payload strings as hexadecimal bytes.
    Hex,
}

/// Canonical fuzz mutator choices for VM action generation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum VmFuzzMutatorSpec {
    /// Flip one random bit.
    FlipBit,
    /// Flip one random byte.
    FlipByte,
    /// Insert one random byte.
    InsertByte,
    /// Delete one random byte.
    DeleteByte,
    /// Splice in bytes from another seed.
    SpliceSeed,
    /// Reset to a seed input.
    ResetSeed,
    /// Apply a short random mutation sequence.
    Havoc,
}

/// Canonical runtime action source for VM environments.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum VmRuntimeActionSourceSpec {
    /// Inline literal action payloads.
    Literal {
        /// Human-readable action names.
        names: Vec<Option<String>>,
        /// Action payloads encoded with the selected payload encoding.
        payloads: Vec<String>,
        /// Payload encoding applied to all literals.
        encoding: VmPayloadEncodingSpec,
    },
    /// Mutation-based fuzzing configuration.
    Fuzz {
        /// Seed inputs encoded with the selected payload encoding.
        seeds: Vec<String>,
        /// Payload encoding applied to seeds and dictionary entries.
        encoding: VmPayloadEncodingSpec,
        /// Enabled mutators by canonical name.
        mutators: Vec<VmFuzzMutatorSpec>,
        /// Minimum generated payload length.
        min_len: usize,
        /// Maximum generated payload length.
        max_len: usize,
        /// Optional dictionary entries.
        dictionary: Vec<String>,
        /// Deterministic RNG seed for mutation sampling.
        rng_seed: u64,
    },
}

/// Canonical Nyx/Firecracker environment configuration.
#[derive(Clone)]
pub struct VmEnvironmentSpec {
    /// Asset identifier pointing at the Firecracker JSON config.
    pub firecracker_config_asset: AssetId,
    /// VM instance identifier.
    pub instance_id: String,
    /// Shared-memory region name used for guest communication.
    pub shared_region_name: String,
    /// Shared-memory region size in bytes.
    pub shared_region_size: usize,
    /// Snapshot-vs-preserve policy for shared memory.
    pub shared_memory_policy: SharedMemoryPolicySpec,
    /// Per-step timeout in milliseconds.
    pub step_timeout_ms: u64,
    /// Initial boot timeout in milliseconds.
    pub boot_timeout_ms: u64,
    /// Episode length in steps.
    pub episode_steps: usize,
    /// Per-step cost subtracted from rewards.
    pub step_cost: i64,
    /// Observation derivation mode.
    pub observation_policy: VmObservationPolicySpec,
    /// Observation bit width.
    pub observation_bits: usize,
    /// Observation stream length.
    pub observation_stream_len: usize,
    /// Observation stream normalization mode.
    pub observation_stream_mode: VmObservationStreamModeSpec,
    /// Padding byte for short observation streams.
    pub observation_pad_byte: u8,
    /// Reward bit width.
    pub reward_bits: usize,
    /// Reward policy.
    pub reward_policy: VmRewardPolicySpec,
    /// Optional reward shaping policy.
    pub reward_shaping: Option<VmRewardShapingSpec>,
    /// Runtime action source.
    pub action_source: VmRuntimeActionSourceSpec,
    /// Optional information-theoretic filter.
    pub action_filter: Option<VmActionFilterSpec>,
    /// Protocol action prefix.
    pub action_prefix: String,
    /// Protocol action suffix.
    pub action_suffix: String,
    /// Protocol observation prefix.
    pub obs_prefix: String,
    /// Protocol reward prefix.
    pub rew_prefix: String,
    /// Protocol done prefix.
    pub done_prefix: String,
    /// Protocol data prefix.
    pub data_prefix: String,
    /// Payload encoding label (`utf8` or `hex`).
    pub wire_encoding: VmPayloadEncodingSpec,
    /// Rate backend used for entropy/statistics estimation.
    pub stats_backend: RateBackend,
    /// Optional trace configuration.
    pub trace: Option<VmTraceSpec>,
    /// Whether to enable verbose VM diagnostics.
    pub debug_mode: bool,
    /// Optional crash log path for VM exits.
    pub crash_log: Option<String>,
}

/// Canonical planner-visible environment specification.
#[derive(Clone)]
pub enum EnvironmentSpec {
    /// Built-in Rust environment.
    Builtin {
        /// Built-in environment kind.
        builtin: BuiltinEnvironmentSpec,
    },
    /// Nyx/Firecracker VM environment.
    #[cfg(feature = "vm")]
    NyxVm(VmEnvironmentSpec),
}

/// Planner observation/reward/action interface contract.
#[derive(Clone, Debug, PartialEq)]
pub struct PlannerInterfaceSpec {
    /// Observation bit width.
    pub observation_bits: usize,
    /// Observation stream length.
    pub observation_stream_len: usize,
    /// Observation key projection used by MC-AIXI.
    pub observation_key_mode: ObservationKeyMode,
    /// Reward bit width.
    pub reward_bits: usize,
    /// Action alphabet cardinality.
    pub agent_actions: usize,
    /// Minimum instantaneous reward.
    pub min_reward: i64,
    /// Maximum instantaneous reward.
    pub max_reward: i64,
    /// Reward offset used for unsigned encoding.
    pub reward_offset: i64,
}

/// MC-AIXI controller configuration using a unified rate-backend predictor.
#[derive(Clone)]
pub struct McAixiControllerSpec {
    /// Predictive backend used by the planner model.
    pub predictor: RateBackend,
    /// Max-order hint for backends that use it.
    pub predictor_max_order: i64,
    /// Planning horizon.
    pub agent_horizon: usize,
    /// Number of simulations per planning step.
    pub num_simulations: usize,
    /// UCT exploration constant.
    pub exploration_exploitation_ratio: f64,
    /// Reward discount factor.
    pub discount_gamma: f64,
}

/// Discounted AIQI controller configuration.
#[derive(Clone)]
pub struct AiqiDiscountedControllerSpec {
    /// Predictive backend used by the return model.
    pub predictor: RateBackend,
    /// Max-order hint for backends that use it.
    pub predictor_max_order: i64,
    /// Discount factor used for return construction.
    pub discount_gamma: f64,
    /// Return horizon.
    pub return_horizon: usize,
    /// Number of discrete return bins.
    pub return_bins: usize,
    /// Label augmentation period.
    pub augmentation_period: usize,
    /// Optional bounded-history retention hint.
    pub history_prune_keep_steps: Option<usize>,
    /// Baseline exploration probability.
    pub baseline_exploration: f64,
}

/// Warm-start exact-\u{1d4a5}_H controller configuration.
#[derive(Clone)]
pub struct WarmStartExactJhControllerSpec {
    /// Predictive backend used by the return model.
    pub predictor: RateBackend,
    /// Max-order hint for backends that use it.
    pub predictor_max_order: i64,
    /// Return horizon in planner steps.
    pub return_horizon: usize,
    /// Exact return-label alphabet size.
    pub return_bins: usize,
    /// Delayed-label phase period.
    pub label_phase_period: usize,
    /// Asset identifier for the warm-start teacher dataset.
    pub teacher_dataset_asset: AssetId,
    /// Simulation budget per planner step.
    pub planner_simulations_per_step: usize,
}

/// Canonical planner controller selection.
#[derive(Clone)]
pub enum ControllerSpec {
    /// Monte Carlo AIXI.
    McAixi(McAixiControllerSpec),
    /// Discounted AIQI.
    AiqiDiscounted(AiqiDiscountedControllerSpec),
    /// Warm-start exact-\u{1d4a5}_H AIQI-style controller.
    AiqiWarmstartExactJh(WarmStartExactJhControllerSpec),
}

/// Operational planner-run controls that do not change predictor semantics.
#[derive(Clone, Debug, PartialEq)]
pub struct PlannerRuntimeSpec {
    /// Seed used for planner/environment stochasticity.
    ///
    /// `None` canonicalizes to `Some(0)` during validation/compilation.
    pub random_seed: Option<u64>,
    /// Number of learning cycles.
    pub learn_cycles: Option<usize>,
    /// Number of evaluation cycles.
    pub eval_cycles: Option<usize>,
    /// Default cycle count when learn/eval are omitted.
    pub terminate_lifetime: usize,
    /// Logging interval in steps.
    pub log_every: usize,
    /// Whether to print throughput diagnostics.
    pub perf: bool,
    /// Whether to run VM perf-only mode.
    pub vm_perf_only: bool,
    /// Extra epsilon exploration used during execution.
    pub explore_epsilon: f64,
    /// Exponential decay for extra exploration.
    pub explore_gamma: f64,
}

/// Canonical planner-run specification.
#[derive(Clone)]
pub struct PlannerRunSpec {
    /// External asset bindings referenced by this run.
    pub assets: Vec<AssetBinding>,
    /// Planner-facing environment.
    pub environment: EnvironmentSpec,
    /// Planner observation/reward/action contract.
    pub interface: PlannerInterfaceSpec,
    /// Controller configuration.
    pub controller: ControllerSpec,
    /// Operational run controls.
    pub runtime: PlannerRuntimeSpec,
}

/// Tuning controller kind specified by the formal tuner document.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TuneControllerKind {
    /// Annealed hill climbing.
    AnnealedHillClimbing,
    /// MC-AIXI(FAC-CTW).
    McAixiFacCtw,
    /// Discounted AIQI.
    AiqiDiscounted,
    /// Warm-start exact-\u{1d4a5}_H controller.
    AiqiWarmstartExactJh,
}

/// Annealed hill-climbing controller settings for tuning.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AnnealedHillClimbingTuneControllerSpec {
    /// Maximum mutation radius applied to a candidate step.
    pub max_mutation_radius: usize,
}

/// MC-AIXI(FAC-CTW) controller settings for tuning.
#[derive(Clone, Debug, PartialEq)]
pub struct McAixiFacCtwTuneControllerSpec {
    /// Planner/environment observation/reward/action contract.
    pub interface: PlannerInterfaceSpec,
    /// Simulation budget per planner step.
    pub planner_simulations_per_step: usize,
}

/// Discounted AIQI controller settings for tuning.
#[derive(Clone, Debug, PartialEq)]
pub struct AiqiDiscountedTuneControllerSpec {
    /// Planner/environment observation/reward/action contract.
    pub interface: PlannerInterfaceSpec,
    /// Simulation budget per planner step.
    pub planner_simulations_per_step: usize,
    /// Return horizon.
    pub return_horizon: usize,
    /// Number of return bins.
    pub return_bins: usize,
    /// Discount factor used to construct returns.
    pub discount_factor: f64,
}

/// Warm-start exact-J_H controller settings for tuning.
#[derive(Clone, Debug, PartialEq)]
pub struct WarmStartExactJhTuneControllerSpec {
    /// Planner/environment observation/reward/action contract.
    pub interface: PlannerInterfaceSpec,
    /// Simulation budget per planner step.
    pub planner_simulations_per_step: usize,
    /// Return horizon.
    pub return_horizon: usize,
    /// Teacher dataset asset for warm-start labels.
    pub warmstart_teacher_dataset_asset: AssetId,
    /// Label phase period.
    pub label_phase_period: usize,
}

/// Runtime-selectable controller configuration for the tuning runtime.
#[derive(Clone, Debug, PartialEq)]
pub enum TuneControllerSpec {
    /// Annealed hill climbing.
    AnnealedHillClimbing(AnnealedHillClimbingTuneControllerSpec),
    /// MC-AIXI(FAC-CTW).
    McAixiFacCtw(McAixiFacCtwTuneControllerSpec),
    /// Discounted AIQI.
    AiqiDiscounted(AiqiDiscountedTuneControllerSpec),
    /// Warm-start exact-J_H.
    AiqiWarmstartExactJh(WarmStartExactJhTuneControllerSpec),
}

impl TuneControllerSpec {
    /// Controller family tag for this tuning controller configuration.
    pub fn kind(&self) -> TuneControllerKind {
        match self {
            Self::AnnealedHillClimbing(_) => TuneControllerKind::AnnealedHillClimbing,
            Self::McAixiFacCtw(_) => TuneControllerKind::McAixiFacCtw,
            Self::AiqiDiscounted(_) => TuneControllerKind::AiqiDiscounted,
            Self::AiqiWarmstartExactJh(_) => TuneControllerKind::AiqiWarmstartExactJh,
        }
    }
}

/// Bounded numeric range for a named canonical tuning parameter.
#[derive(Clone, Debug, PartialEq)]
pub struct TuneParameterRangeSpec {
    /// Canonical parameter path or name.
    pub parameter: String,
    /// Inclusive lower bound.
    pub min: f64,
    /// Inclusive upper bound.
    pub max: f64,
}

/// Canonical bounds specification for the future tuning runtime.
#[derive(Clone, Debug, PartialEq)]
pub struct TuneBoundsSpec {
    /// Allowed canonical backend names.
    pub allowed_backends: Vec<String>,
    /// Forbidden canonical backend names.
    pub forbidden_backends: Vec<String>,
    /// Inclusive ranges for canonical numeric parameters.
    pub parameter_ranges: Vec<TuneParameterRangeSpec>,
    /// Maximum experts per mixture node.
    pub max_experts: usize,
    /// Maximum recursive mixture nesting depth.
    pub max_mixture_nesting_depth: usize,
    /// Optional minimum experts per mixture node.
    pub min_experts: Option<usize>,
    /// Whether duplicate experts are permitted.
    pub allow_duplicate_experts: Option<bool>,
    /// Expert names that must appear in the candidate set.
    pub required_experts: Vec<String>,
    /// Canonical expert-pair combinations that are forbidden together.
    pub forbidden_expert_pairs: Vec<(String, String)>,
}

/// Canonical future-facing tune request document.
#[derive(Clone)]
pub struct TuneSpec {
    /// External asset bindings referenced by this request.
    pub assets: Vec<AssetBinding>,
    /// Asset identifier for the input dataset.
    pub input_asset: AssetId,
    /// Baseline candidate configuration.
    pub baseline_candidate: CompressionBackend,
    /// Runtime-selectable controller used by the tuning runtime.
    pub controller: TuneControllerSpec,
    /// Candidate bounds specification.
    pub bounds: TuneBoundsSpec,
    /// Per-candidate evaluation time limit in seconds.
    pub eval_time_limit_seconds: f64,
    /// Total search budget in seconds.
    pub time_budget_seconds: f64,
    /// Minimum required throughput in bytes/second.
    pub min_throughput_bytes_per_second: f64,
    /// Maximum allowed memory use in bytes.
    pub max_memory_bytes: u64,
    /// Output path for the best canonical candidate.
    pub output_config_path: String,
    /// Deterministic search seed.
    pub seed: u64,
    /// Optional report output path.
    pub report_path: Option<String>,
}

/// Compiled planner controller with precompiled predictor backends.
#[derive(Clone)]
pub enum CompiledPlannerController {
    /// MC-AIXI controller.
    McAixi {
        /// Compiled predictor backend.
        predictor: CompiledRateBackend,
        /// Max-order hint used by the predictor adapter.
        predictor_max_order: i64,
        /// Planning horizon.
        agent_horizon: usize,
        /// Number of simulations per planning step.
        num_simulations: usize,
        /// UCT exploration constant.
        exploration_exploitation_ratio: f64,
        /// Reward discount factor.
        discount_gamma: f64,
    },
    /// Discounted AIQI controller.
    AiqiDiscounted {
        /// Compiled predictor backend.
        predictor: CompiledRateBackend,
        /// Max-order hint used by the predictor adapter.
        predictor_max_order: i64,
        /// Discount factor used for return construction.
        discount_gamma: f64,
        /// Return horizon.
        return_horizon: usize,
        /// Number of return bins.
        return_bins: usize,
        /// Label augmentation period.
        augmentation_period: usize,
        /// Optional bounded-history retention hint.
        history_prune_keep_steps: Option<usize>,
        /// Baseline exploration probability.
        baseline_exploration: f64,
    },
    /// Warm-start exact-J_H controller.
    AiqiWarmstartExactJh {
        /// Compiled predictor backend.
        predictor: CompiledRateBackend,
        /// Max-order hint used by the predictor adapter.
        predictor_max_order: i64,
        /// Return horizon.
        return_horizon: usize,
        /// Number of return bins.
        return_bins: usize,
        /// Label phase period.
        label_phase_period: usize,
        /// Teacher dataset asset id.
        teacher_dataset_asset: AssetId,
        /// Simulation budget per planner step.
        planner_simulations_per_step: usize,
    },
}

/// Compiled planner-run specification with resolved assets and compiled backends.
#[derive(Clone)]
pub struct CompiledPlannerRunSpec {
    pub(super) canonical_spec: Arc<PlannerRunSpec>,
    pub(super) canonical_bytes: CanonicalBytes,
    pub(super) resolved_assets: Arc<[ResolvedAssetBinding]>,
    pub(super) interface: PlannerInterfaceSpec,
    pub(super) runtime: PlannerRuntimeSpec,
    pub(super) controller: CompiledPlannerController,
    pub(super) action_bits: usize,
}

/// Compiled tuning controller configuration.
#[derive(Clone)]
pub enum CompiledTuneController {
    /// Annealed hill climbing.
    AnnealedHillClimbing(AnnealedHillClimbingTuneControllerSpec),
    /// MC-AIXI(FAC-CTW).
    McAixiFacCtw(McAixiFacCtwTuneControllerSpec),
    /// Discounted AIQI.
    AiqiDiscounted(AiqiDiscountedTuneControllerSpec),
    /// Warm-start exact-J_H.
    AiqiWarmstartExactJh(WarmStartExactJhTuneControllerSpec),
}

/// Compiled tune request with resolved assets and compiled baseline candidate.
#[derive(Clone)]
pub struct CompiledTuneSpec {
    pub(super) canonical_spec: Arc<TuneSpec>,
    pub(super) canonical_bytes: CanonicalBytes,
    pub(super) resolved_assets: Arc<[ResolvedAssetBinding]>,
    pub(super) baseline_candidate: CompiledCompressionBackend,
    pub(super) controller: CompiledTuneController,
    pub(super) candidate_canonicalization_version: &'static str,
}

/// Universal top-level spec document.
#[derive(Clone)]
pub enum SpecDocument {
    /// Planner-run configuration document.
    PlannerRun(PlannerRunSpec),
    /// Tune request document.
    Tune(TuneSpec),
    /// Standalone rate-backend document.
    RateBackend(RateBackend),
    /// Standalone compression-backend document.
    CompressionBackend(CompressionBackend),
}

/// Parsed top-level spec document paired with its parse base directory.
///
/// This is the first stage of the canonical pipeline:
/// parse -> validate -> compile.
#[derive(Clone)]
pub struct ParsedSpecDocument {
    pub(super) document: SpecDocument,
    pub(super) base_dir: PathBuf,
}

/// Canonicalized and validated planner-run document.
#[derive(Clone)]
pub struct ValidatedPlannerRunSpec {
    pub(super) canonical_spec: Arc<PlannerRunSpec>,
    pub(super) canonical_bytes: CanonicalBytes,
    pub(super) base_dir: PathBuf,
}

/// Canonicalized and validated tune request document.
#[derive(Clone)]
pub struct ValidatedTuneSpec {
    pub(super) canonical_spec: Arc<TuneSpec>,
    pub(super) canonical_bytes: CanonicalBytes,
    pub(super) base_dir: PathBuf,
}

/// Validated top-level spec document.
///
/// This is the second stage of the canonical pipeline and can be compiled into
/// runtime-ready plans/backends.
#[derive(Clone)]
pub enum ValidatedSpecDocument {
    /// Validated planner-run document.
    PlannerRun(ValidatedPlannerRunSpec),
    /// Validated tune document.
    Tune(ValidatedTuneSpec),
    /// Validated standalone rate-backend document.
    RateBackend(ValidatedRateBackend),
    /// Validated standalone compression-backend document.
    CompressionBackend(ValidatedCompressionBackend),
}

/// Compiled top-level spec document.
///
/// This is the final stage of the canonical pipeline and is executable by
/// runtime adapters.
#[derive(Clone)]
pub enum CompiledSpecDocument {
    /// Compiled planner-run document.
    PlannerRun(CompiledPlannerRunSpec),
    /// Compiled tune document.
    Tune(CompiledTuneSpec),
    /// Compiled standalone rate-backend document.
    RateBackend(CompiledRateBackend),
    /// Compiled standalone compression-backend document.
    CompressionBackend(CompiledCompressionBackend),
}
