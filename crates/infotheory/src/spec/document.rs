//! Canonical top-level specification documents for planner runs and tuning.

use super::core::{AssetRef, CanonicalBytes, CompiledCompressionBackend, CompiledRateBackend};
use super::{
    SpecEnvironment, SpecError, SpecResult, compression_backend_to_canonical_json,
    compression_backend_to_json_value, parse_compression_backend_json, parse_rate_backend_json,
    rate_backend_to_canonical_json, rate_backend_to_json_value,
};
use crate::aixi::common::ObservationKeyMode;
use crate::api::{CompressionBackend, RateBackend};
use std::collections::HashMap;
use std::fmt;
use std::path::{Path, PathBuf};
use std::sync::Arc;

/// Schema version for canonical top-level spec documents.
pub const SPEC_DOCUMENT_SCHEMA_VERSION: u32 = 1;

const DOCUMENT_MAGIC: &[u8; 4] = b"itsd";
const DOCUMENT_BINARY_VERSION: u8 = 1;
const TUNE_CANONICALIZATION_CLASSIFICATION_VERSION: &str = "bounds-v1";

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
    /// Bernoulli coin-flip environment.
    CoinFlip,
    /// Action-conditional CTW test environment.
    CtwTest,
    /// Extended Tiger POMDP.
    ExtendedTiger,
    /// Tic-tac-toe environment.
    TicTacToe,
    /// Biased rock-paper-scissor environment.
    BiasedRockPaperScissor,
    /// Kuhn poker environment.
    KuhnPoker,
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
        encoding: String,
    },
    /// Mutation-based fuzzing configuration.
    Fuzz {
        /// Seed inputs encoded with the selected payload encoding.
        seeds: Vec<String>,
        /// Payload encoding applied to seeds and dictionary entries.
        encoding: String,
        /// Enabled mutators by canonical name.
        mutators: Vec<String>,
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
    pub observation_policy: String,
    /// Observation bit width.
    pub observation_bits: usize,
    /// Observation stream length.
    pub observation_stream_len: usize,
    /// Observation stream normalization mode.
    pub observation_stream_mode: String,
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
    pub wire_encoding: String,
    /// Rate backend used for entropy/statistics estimation.
    pub stats_backend: RateBackend,
    /// Optional trace configuration.
    pub trace: Option<VmTraceSpec>,
    /// Whether to enable verbose VM diagnostics.
    pub debug_mode: bool,
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
    canonical_spec: Arc<PlannerRunSpec>,
    canonical_bytes: CanonicalBytes,
    resolved_assets: Arc<[ResolvedAssetBinding]>,
    interface: PlannerInterfaceSpec,
    runtime: PlannerRuntimeSpec,
    controller: CompiledPlannerController,
    action_bits: usize,
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
    canonical_spec: Arc<TuneSpec>,
    canonical_bytes: CanonicalBytes,
    resolved_assets: Arc<[ResolvedAssetBinding]>,
    baseline_candidate: CompiledCompressionBackend,
    controller: CompiledTuneController,
    candidate_canonicalization_version: &'static str,
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

/// Canonicalized and validated planner-run document.
#[derive(Clone)]
pub struct ValidatedPlannerRunSpec {
    canonical_spec: Arc<PlannerRunSpec>,
    canonical_bytes: CanonicalBytes,
    base_dir: PathBuf,
}

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
        compile_validated_planner_run_spec(self)
    }
}

/// Canonicalized and validated tune request document.
#[derive(Clone)]
pub struct ValidatedTuneSpec {
    canonical_spec: Arc<TuneSpec>,
    canonical_bytes: CanonicalBytes,
    base_dir: PathBuf,
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
        compile_validated_tune_spec(self)
    }
}

/// Compatibility alias for a planner-run top-level spec document.
pub type PlannerRunDocument = PlannerRunSpec;

/// Compatibility alias for a tune top-level spec document.
pub type TuneDocument = TuneSpec;

impl PlannerRunSpec {
    /// Validate this planner-run spec and return its canonical binary encoding.
    pub fn validate_in(&self, env: &SpecEnvironment) -> SpecResult<ValidatedPlannerRunSpec> {
        let canonical = canonicalize_planner_run(self, env)?;
        Ok(ValidatedPlannerRunSpec {
            canonical_bytes: CanonicalBytes::from(encode_spec_document_payload(
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
        compile_planner_run_spec(self, env.base_dir())
    }

    /// Validate and compile this planner-run spec using the default environment.
    pub fn compile(&self) -> SpecResult<CompiledPlannerRunSpec> {
        self.compile_in(&SpecEnvironment::default())
    }

    /// Serialize this planner-run spec to deterministic canonical JSON.
    pub fn to_canonical_json(&self) -> SpecResult<String> {
        serde_json::to_string_pretty(&planner_run_to_json_value(self)?).map_err(SpecError::from)
    }

    /// Serialize this planner-run spec to canonical JSON value form.
    pub fn to_canonical_json_value(&self) -> SpecResult<serde_json::Value> {
        planner_run_to_json_value(self)
    }
}

impl EnvironmentSpec {
    /// Validate and canonicalize this environment spec against the supplied asset bindings.
    pub fn validate_in(
        &self,
        assets: &[AssetBinding],
        env: &SpecEnvironment,
    ) -> SpecResult<EnvironmentSpec> {
        canonicalize_environment_spec(self, assets, env)
    }

    /// Validate and canonicalize this environment spec using the default environment.
    pub fn validate(&self, assets: &[AssetBinding]) -> SpecResult<EnvironmentSpec> {
        self.validate_in(assets, &SpecEnvironment::default())
    }
}

impl TuneSpec {
    /// Validate this tune request and return its canonical binary encoding.
    pub fn validate_in(&self, env: &SpecEnvironment) -> SpecResult<ValidatedTuneSpec> {
        let canonical = canonicalize_tune_spec(self, env)?;
        Ok(ValidatedTuneSpec {
            canonical_bytes: CanonicalBytes::from(encode_spec_document_payload(
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
        compile_tune_spec(self, env.base_dir())
    }

    /// Validate and compile this tune request using the default environment.
    pub fn compile(&self) -> SpecResult<CompiledTuneSpec> {
        self.compile_in(&SpecEnvironment::default())
    }

    /// Serialize this tune request to deterministic canonical JSON.
    pub fn to_canonical_json(&self) -> SpecResult<String> {
        serde_json::to_string_pretty(&tune_spec_to_json_value(self)?).map_err(SpecError::from)
    }

    /// Serialize this tune request to canonical JSON value form.
    pub fn to_canonical_json_value(&self) -> SpecResult<serde_json::Value> {
        tune_spec_to_json_value(self)
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
        match self {
            Self::PlannerRun(spec) => planner_run_to_json_value(spec),
            Self::Tune(spec) => tune_spec_to_json_value(spec),
            Self::RateBackend(backend) => Ok(serde_json::json!({
                "schema_version": SPEC_DOCUMENT_SCHEMA_VERSION,
                "kind": "rate_backend",
                "backend": rate_backend_to_json_value(backend)?,
            })),
            Self::CompressionBackend(backend) => Ok(serde_json::json!({
                "schema_version": SPEC_DOCUMENT_SCHEMA_VERSION,
                "kind": "compression_backend",
                "backend": compression_backend_to_json_value(backend)?,
            })),
        }
    }

    /// Encode this document in the versioned binary document envelope.
    pub fn to_binary(&self) -> Vec<u8> {
        encode_spec_document_payload(self)
    }

    /// Parse a canonical JSON document from a raw JSON value.
    pub fn parse_json_value(value: &serde_json::Value, base_dir: &Path) -> SpecResult<Self> {
        parse_spec_document_json_value(value, base_dir)
    }

    /// Decode a binary spec document.
    pub fn from_binary(bytes: &[u8], base_dir: &Path) -> SpecResult<Self> {
        decode_spec_document(bytes, base_dir)
    }
}

/// Load a spec document from a JSON file on disk.
pub fn load_spec_document(path: &str) -> SpecResult<SpecDocument> {
    let full = Path::new(path);
    let base_dir = full.parent().unwrap_or_else(|| Path::new("."));
    let raw = std::fs::read(full)?;
    match serde_json::from_slice::<serde_json::Value>(&raw) {
        Ok(value) => SpecDocument::parse_json_value(&value, base_dir),
        Err(_) => SpecDocument::from_binary(&raw, base_dir),
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

fn resolve_asset_bindings(
    bindings: &[AssetBinding],
    base_dir: &Path,
) -> Arc<[ResolvedAssetBinding]> {
    bindings
        .iter()
        .map(|binding| ResolvedAssetBinding {
            id: binding.id.clone(),
            asset: AssetRef::Filesystem(super::resolve_spec_path(base_dir, &binding.path)),
        })
        .collect::<Vec<_>>()
        .into()
}

fn bits_for_cardinality(cardinality: usize) -> usize {
    if cardinality <= 1 {
        return 1;
    }
    let mut bits = 0usize;
    let mut value = cardinality - 1;
    while value > 0 {
        bits += 1;
        value >>= 1;
    }
    bits
}

fn compile_planner_controller(
    spec: &ControllerSpec,
    env: &SpecEnvironment,
) -> SpecResult<CompiledPlannerController> {
    match spec {
        ControllerSpec::McAixi(inner) => Ok(CompiledPlannerController::McAixi {
            predictor: inner.predictor.validate_in(env)?.compile()?,
            predictor_max_order: inner.predictor_max_order,
            agent_horizon: inner.agent_horizon,
            num_simulations: inner.num_simulations,
            exploration_exploitation_ratio: inner.exploration_exploitation_ratio,
            discount_gamma: inner.discount_gamma,
        }),
        ControllerSpec::AiqiDiscounted(inner) => Ok(CompiledPlannerController::AiqiDiscounted {
            predictor: inner.predictor.validate_in(env)?.compile()?,
            predictor_max_order: inner.predictor_max_order,
            discount_gamma: inner.discount_gamma,
            return_horizon: inner.return_horizon,
            return_bins: inner.return_bins,
            augmentation_period: inner.augmentation_period,
            history_prune_keep_steps: inner.history_prune_keep_steps,
            baseline_exploration: inner.baseline_exploration,
        }),
        ControllerSpec::AiqiWarmstartExactJh(inner) => {
            Ok(CompiledPlannerController::AiqiWarmstartExactJh {
                predictor: inner.predictor.validate_in(env)?.compile()?,
                predictor_max_order: inner.predictor_max_order,
                return_horizon: inner.return_horizon,
                return_bins: inner.return_bins,
                label_phase_period: inner.label_phase_period,
                teacher_dataset_asset: inner.teacher_dataset_asset.clone(),
                planner_simulations_per_step: inner.planner_simulations_per_step,
            })
        }
    }
}

fn compile_tune_controller(spec: &TuneControllerSpec) -> CompiledTuneController {
    match spec {
        TuneControllerSpec::AnnealedHillClimbing(inner) => {
            CompiledTuneController::AnnealedHillClimbing(inner.clone())
        }
        TuneControllerSpec::McAixiFacCtw(inner) => {
            CompiledTuneController::McAixiFacCtw(inner.clone())
        }
        TuneControllerSpec::AiqiDiscounted(inner) => {
            CompiledTuneController::AiqiDiscounted(inner.clone())
        }
        TuneControllerSpec::AiqiWarmstartExactJh(inner) => {
            CompiledTuneController::AiqiWarmstartExactJh(inner.clone())
        }
    }
}

fn compile_planner_run_spec(
    spec: &PlannerRunSpec,
    base_dir: &Path,
) -> SpecResult<CompiledPlannerRunSpec> {
    let validated = spec.validate_in(&SpecEnvironment::new(base_dir))?;
    compile_validated_planner_run_spec(&validated)
}

fn compile_validated_planner_run_spec(
    validated: &ValidatedPlannerRunSpec,
) -> SpecResult<CompiledPlannerRunSpec> {
    let env = SpecEnvironment::new(&validated.base_dir);
    Ok(CompiledPlannerRunSpec {
        canonical_spec: validated.canonical_spec.clone(),
        canonical_bytes: validated.canonical_bytes().clone(),
        resolved_assets: resolve_asset_bindings(
            &validated.canonical_spec().assets,
            &validated.base_dir,
        ),
        interface: validated.canonical_spec().interface.clone(),
        runtime: validated.canonical_spec().runtime.clone(),
        controller: compile_planner_controller(&validated.canonical_spec().controller, &env)?,
        action_bits: bits_for_cardinality(validated.canonical_spec().interface.agent_actions),
    })
}

fn compile_tune_spec(spec: &TuneSpec, base_dir: &Path) -> SpecResult<CompiledTuneSpec> {
    let validated = spec.validate_in(&SpecEnvironment::new(base_dir))?;
    compile_validated_tune_spec(&validated)
}

fn compile_validated_tune_spec(validated: &ValidatedTuneSpec) -> SpecResult<CompiledTuneSpec> {
    let env = SpecEnvironment::new(&validated.base_dir);
    Ok(CompiledTuneSpec {
        canonical_spec: validated.canonical_spec.clone(),
        canonical_bytes: validated.canonical_bytes().clone(),
        resolved_assets: resolve_asset_bindings(
            &validated.canonical_spec().assets,
            &validated.base_dir,
        ),
        baseline_candidate: validated
            .canonical_spec()
            .baseline_candidate
            .validate_in(&env)?
            .compile()?,
        controller: compile_tune_controller(&validated.canonical_spec().controller),
        candidate_canonicalization_version: TUNE_CANONICALIZATION_CLASSIFICATION_VERSION,
    })
}

fn canonicalize_planner_run(
    spec: &PlannerRunSpec,
    env: &SpecEnvironment,
) -> SpecResult<PlannerRunSpec> {
    validate_asset_bindings(&spec.assets)?;
    let environment = canonicalize_environment_spec(&spec.environment, &spec.assets, env)?;
    let interface = canonicalize_interface_spec(&spec.interface)?;
    let controller = canonicalize_controller_spec(&spec.controller, env)?;
    let runtime = canonicalize_runtime_spec(&spec.runtime)?;
    Ok(PlannerRunSpec {
        assets: canonicalize_assets(&spec.assets),
        environment,
        interface,
        controller,
        runtime,
    })
}

fn canonicalize_tune_spec(spec: &TuneSpec, env: &SpecEnvironment) -> SpecResult<TuneSpec> {
    validate_asset_bindings(&spec.assets)?;
    ensure_asset_exists(&spec.assets, &spec.input_asset)?;
    let baseline = parse_compression_backend_json(
        &compression_backend_to_json_value(&spec.baseline_candidate)?,
        env.base_dir(),
        None,
        crate::compression::FramingMode::Framed,
    )?;
    let controller = canonicalize_tune_controller(&spec.controller, &spec.assets, env)?;
    validate_tune_bounds(&spec.bounds)?;
    Ok(TuneSpec {
        assets: canonicalize_assets(&spec.assets),
        input_asset: spec.input_asset.trim().to_string(),
        baseline_candidate: baseline,
        controller,
        bounds: canonicalize_tune_bounds(&spec.bounds),
        eval_time_limit_seconds: finite_positive(
            spec.eval_time_limit_seconds,
            "eval_time_limit_seconds",
        )?,
        time_budget_seconds: finite_positive(spec.time_budget_seconds, "time_budget_seconds")?,
        min_throughput_bytes_per_second: finite_positive(
            spec.min_throughput_bytes_per_second,
            "min_throughput_bytes_per_second",
        )?,
        max_memory_bytes: nonzero_u64(spec.max_memory_bytes, "max_memory_bytes")?,
        output_config_path: spec.output_config_path.trim().to_string(),
        seed: spec.seed,
        report_path: clean_optional_string(spec.report_path.as_deref()),
    })
}

fn canonicalize_tune_controller(
    controller: &TuneControllerSpec,
    assets: &[AssetBinding],
    env: &SpecEnvironment,
) -> SpecResult<TuneControllerSpec> {
    match controller {
        TuneControllerSpec::AnnealedHillClimbing(inner) => {
            if inner.max_mutation_radius == 0 {
                return Err(SpecError::new("max_mutation_radius must be >= 1"));
            }
            Ok(TuneControllerSpec::AnnealedHillClimbing(inner.clone()))
        }
        TuneControllerSpec::McAixiFacCtw(inner) => {
            validate_planner_interface_for_tuning(&inner.interface)?;
            if inner.planner_simulations_per_step == 0 {
                return Err(SpecError::new("planner_simulations_per_step must be >= 1"));
            }
            Ok(TuneControllerSpec::McAixiFacCtw(inner.clone()))
        }
        TuneControllerSpec::AiqiDiscounted(inner) => {
            validate_planner_interface_for_tuning(&inner.interface)?;
            if inner.planner_simulations_per_step == 0 {
                return Err(SpecError::new("planner_simulations_per_step must be >= 1"));
            }
            if inner.return_horizon == 0 {
                return Err(SpecError::new("return_horizon must be >= 1"));
            }
            if inner.return_bins == 0 || !inner.return_bins.is_power_of_two() {
                return Err(SpecError::new("return_bins must be a power of two"));
            }
            if !(0.0..1.0).contains(&inner.discount_factor) {
                return Err(SpecError::new("discount_factor must be in [0, 1)"));
            }
            Ok(TuneControllerSpec::AiqiDiscounted(inner.clone()))
        }
        TuneControllerSpec::AiqiWarmstartExactJh(inner) => {
            validate_planner_interface_for_tuning(&inner.interface)?;
            if inner.planner_simulations_per_step == 0 {
                return Err(SpecError::new("planner_simulations_per_step must be >= 1"));
            }
            if inner.return_horizon == 0 {
                return Err(SpecError::new("return_horizon must be >= 1"));
            }
            if inner.label_phase_period < inner.return_horizon {
                return Err(SpecError::new(
                    "label_phase_period must be >= return_horizon",
                ));
            }
            ensure_asset_exists(assets, &inner.warmstart_teacher_dataset_asset)?;
            let _ = env;
            Ok(TuneControllerSpec::AiqiWarmstartExactJh(inner.clone()))
        }
    }
}

fn validate_planner_interface_for_tuning(spec: &PlannerInterfaceSpec) -> SpecResult<()> {
    canonicalize_interface_spec(spec).map(|_| ())
}

fn canonicalize_assets(bindings: &[AssetBinding]) -> Vec<AssetBinding> {
    let mut out = bindings.to_vec();
    out.sort_by(|a, b| a.id.cmp(&b.id).then_with(|| a.path.cmp(&b.path)));
    out
}

fn validate_asset_bindings(bindings: &[AssetBinding]) -> SpecResult<()> {
    let mut seen = HashMap::<&str, &str>::new();
    for binding in bindings {
        let id = binding.id.trim();
        let path = binding.path.trim();
        if id.is_empty() {
            return Err(SpecError::new("asset id cannot be empty"));
        }
        if path.is_empty() {
            return Err(SpecError::new(format!(
                "asset '{}' path cannot be empty",
                binding.id
            )));
        }
        if let Some(previous) = seen.insert(id, path)
            && previous != path
        {
            return Err(SpecError::new(format!(
                "asset '{}' is bound to more than one path",
                binding.id
            )));
        }
    }
    Ok(())
}

fn ensure_asset_exists(bindings: &[AssetBinding], id: &str) -> SpecResult<()> {
    if bindings.iter().any(|binding| binding.id == id) {
        Ok(())
    } else {
        Err(SpecError::new(format!("unknown asset id '{id}'")))
    }
}

fn canonicalize_interface_spec(spec: &PlannerInterfaceSpec) -> SpecResult<PlannerInterfaceSpec> {
    if spec.agent_actions == 0 {
        return Err(SpecError::new("agent_actions must be >= 1"));
    }
    if spec.observation_stream_len == 0 {
        return Err(SpecError::new("observation_stream_len must be >= 1"));
    }
    if spec.reward_bits == 0 {
        return Err(SpecError::new("reward_bits must be >= 1"));
    }
    if spec.max_reward < spec.min_reward {
        return Err(SpecError::new("max_reward must be >= min_reward"));
    }
    Ok(spec.clone())
}

fn canonicalize_controller_spec(
    spec: &ControllerSpec,
    env: &SpecEnvironment,
) -> SpecResult<ControllerSpec> {
    match spec {
        ControllerSpec::McAixi(inner) => {
            if inner.agent_horizon == 0 {
                return Err(SpecError::new("agent_horizon must be >= 1"));
            }
            if inner.num_simulations == 0 {
                return Err(SpecError::new("num_simulations must be >= 1"));
            }
            if inner.exploration_exploitation_ratio <= 0.0 {
                return Err(SpecError::new("exploration_exploitation_ratio must be > 0"));
            }
            if !(0.0..=1.0).contains(&inner.discount_gamma) {
                return Err(SpecError::new("discount_gamma must be in [0, 1]"));
            }
            let predictor = parse_rate_backend_json(
                &rate_backend_to_json_value(&inner.predictor)?,
                env.base_dir(),
                crate::api::MAX_MIXTURE_NESTING,
            )?;
            predictor.validate_in(env)?;
            Ok(ControllerSpec::McAixi(McAixiControllerSpec {
                predictor,
                predictor_max_order: inner.predictor_max_order,
                agent_horizon: inner.agent_horizon,
                num_simulations: inner.num_simulations,
                exploration_exploitation_ratio: inner.exploration_exploitation_ratio,
                discount_gamma: inner.discount_gamma,
            }))
        }
        ControllerSpec::AiqiDiscounted(inner) => {
            if inner.return_horizon == 0 {
                return Err(SpecError::new("return_horizon must be >= 1"));
            }
            if inner.return_bins == 0 || !inner.return_bins.is_power_of_two() {
                return Err(SpecError::new("return_bins must be a power of two"));
            }
            if inner.augmentation_period < inner.return_horizon {
                return Err(SpecError::new(
                    "augmentation_period must be >= return_horizon",
                ));
            }
            if !(0.0 < inner.discount_gamma && inner.discount_gamma < 1.0) {
                return Err(SpecError::new("discount_gamma must be in (0, 1)"));
            }
            if !(0.0 < inner.baseline_exploration && inner.baseline_exploration <= 1.0) {
                return Err(SpecError::new("baseline_exploration must be in (0, 1]"));
            }
            let predictor = parse_rate_backend_json(
                &rate_backend_to_json_value(&inner.predictor)?,
                env.base_dir(),
                crate::api::MAX_MIXTURE_NESTING,
            )?;
            predictor.validate_in(env)?;
            Ok(ControllerSpec::AiqiDiscounted(
                AiqiDiscountedControllerSpec {
                    predictor,
                    predictor_max_order: inner.predictor_max_order,
                    discount_gamma: inner.discount_gamma,
                    return_horizon: inner.return_horizon,
                    return_bins: inner.return_bins,
                    augmentation_period: inner.augmentation_period,
                    history_prune_keep_steps: inner.history_prune_keep_steps,
                    baseline_exploration: inner.baseline_exploration,
                },
            ))
        }
        ControllerSpec::AiqiWarmstartExactJh(inner) => {
            if inner.return_horizon == 0 {
                return Err(SpecError::new("return_horizon must be >= 1"));
            }
            if inner.return_bins == 0 {
                return Err(SpecError::new("return_bins must be >= 1"));
            }
            if inner.label_phase_period < inner.return_horizon {
                return Err(SpecError::new(
                    "label_phase_period must be >= return_horizon",
                ));
            }
            let predictor = parse_rate_backend_json(
                &rate_backend_to_json_value(&inner.predictor)?,
                env.base_dir(),
                crate::api::MAX_MIXTURE_NESTING,
            )?;
            predictor.validate_in(env)?;
            Ok(ControllerSpec::AiqiWarmstartExactJh(
                WarmStartExactJhControllerSpec {
                    predictor,
                    predictor_max_order: inner.predictor_max_order,
                    return_horizon: inner.return_horizon,
                    return_bins: inner.return_bins,
                    label_phase_period: inner.label_phase_period,
                    teacher_dataset_asset: inner.teacher_dataset_asset.trim().to_string(),
                    planner_simulations_per_step: inner.planner_simulations_per_step,
                },
            ))
        }
    }
}

fn canonicalize_environment_spec(
    spec: &EnvironmentSpec,
    _assets: &[AssetBinding],
    _env: &SpecEnvironment,
) -> SpecResult<EnvironmentSpec> {
    match spec {
        EnvironmentSpec::Builtin { builtin } => Ok(EnvironmentSpec::Builtin { builtin: *builtin }),
        #[cfg(feature = "vm")]
        EnvironmentSpec::NyxVm(vm) => {
            ensure_asset_exists(_assets, &vm.firecracker_config_asset)?;
            vm.stats_backend.validate_in(_env)?;
            if let Some(shape) = &vm.reward_shaping
                && let VmRewardShapingSpec::EntropyReduction { baseline_asset, .. } = shape
            {
                ensure_asset_exists(_assets, baseline_asset)?;
            }
            if let Some(filter) = &vm.action_filter
                && let Some(asset) = &filter.novelty_prior_asset
            {
                ensure_asset_exists(_assets, asset)?;
            }
            Ok(EnvironmentSpec::NyxVm(vm.clone()))
        }
    }
}

fn canonicalize_runtime_spec(spec: &PlannerRuntimeSpec) -> SpecResult<PlannerRuntimeSpec> {
    if spec.terminate_lifetime == 0 {
        return Err(SpecError::new("terminate_lifetime must be >= 1"));
    }
    if spec.log_every == 0 {
        return Err(SpecError::new("log_every must be >= 1"));
    }
    if spec.explore_epsilon < 0.0 {
        return Err(SpecError::new("explore_epsilon must be >= 0"));
    }
    if spec.explore_gamma <= 0.0 {
        return Err(SpecError::new("explore_gamma must be > 0"));
    }
    Ok(spec.clone())
}

fn validate_tune_bounds(bounds: &TuneBoundsSpec) -> SpecResult<()> {
    if bounds.max_experts == 0 {
        return Err(SpecError::new("max_experts must be >= 1"));
    }
    if bounds.max_mixture_nesting_depth == 0 {
        return Err(SpecError::new("max_mixture_nesting_depth must be >= 1"));
    }
    if let Some(min_experts) = bounds.min_experts
        && min_experts > bounds.max_experts
    {
        return Err(SpecError::new("min_experts cannot exceed max_experts"));
    }
    for range in &bounds.parameter_ranges {
        if range.parameter.trim().is_empty() {
            return Err(SpecError::new(
                "bounds.parameter_ranges[].parameter cannot be empty",
            ));
        }
        if !range.min.is_finite() || !range.max.is_finite() {
            return Err(SpecError::new(
                "bounds.parameter_ranges must use finite min/max values",
            ));
        }
        if range.min > range.max {
            return Err(SpecError::new(
                "bounds.parameter_ranges min cannot exceed max",
            ));
        }
    }
    for name in bounds
        .allowed_backends
        .iter()
        .chain(bounds.forbidden_backends.iter())
        .chain(bounds.required_experts.iter())
    {
        if name.trim().is_empty() {
            return Err(SpecError::new(
                "bounds backend/expert names cannot be empty",
            ));
        }
    }
    for name in &bounds.allowed_backends {
        if bounds
            .forbidden_backends
            .iter()
            .any(|forbidden| forbidden == name)
        {
            return Err(SpecError::new(
                "allowed_backends and forbidden_backends cannot overlap",
            ));
        }
    }
    Ok(())
}

fn canonicalize_tune_bounds(bounds: &TuneBoundsSpec) -> TuneBoundsSpec {
    let mut allowed = bounds.allowed_backends.clone();
    allowed.sort();
    allowed.dedup();
    let mut forbidden_backends = bounds.forbidden_backends.clone();
    forbidden_backends.sort();
    forbidden_backends.dedup();
    let mut required = bounds.required_experts.clone();
    required.sort();
    required.dedup();
    let mut parameter_ranges = bounds.parameter_ranges.clone();
    parameter_ranges.sort_by(|a, b| a.parameter.cmp(&b.parameter));
    parameter_ranges
        .dedup_by(|a, b| a.parameter == b.parameter && a.min == b.min && a.max == b.max);
    let mut forbidden_pairs = bounds
        .forbidden_expert_pairs
        .iter()
        .map(|(a, b)| {
            if a <= b {
                (a.clone(), b.clone())
            } else {
                (b.clone(), a.clone())
            }
        })
        .collect::<Vec<_>>();
    forbidden_pairs.sort();
    forbidden_pairs.dedup();
    TuneBoundsSpec {
        allowed_backends: allowed,
        forbidden_backends,
        parameter_ranges,
        max_experts: bounds.max_experts,
        max_mixture_nesting_depth: bounds.max_mixture_nesting_depth,
        min_experts: bounds.min_experts,
        allow_duplicate_experts: bounds.allow_duplicate_experts,
        required_experts: required,
        forbidden_expert_pairs: forbidden_pairs,
    }
}

fn finite_positive(value: f64, label: &str) -> SpecResult<f64> {
    if value.is_finite() && value > 0.0 {
        Ok(value)
    } else {
        Err(SpecError::new(format!("{label} must be > 0")))
    }
}

fn nonzero_u64(value: u64, label: &str) -> SpecResult<u64> {
    if value > 0 {
        Ok(value)
    } else {
        Err(SpecError::new(format!("{label} must be > 0")))
    }
}

fn clean_optional_string(value: Option<&str>) -> Option<String> {
    value.and_then(|text| {
        let trimmed = text.trim();
        (!trimmed.is_empty()).then(|| trimmed.to_string())
    })
}

fn planner_run_to_json_value(spec: &PlannerRunSpec) -> SpecResult<serde_json::Value> {
    Ok(serde_json::json!({
        "schema_version": SPEC_DOCUMENT_SCHEMA_VERSION,
        "kind": "planner_run",
        "assets": spec.assets.iter().map(asset_binding_to_json_value).collect::<Vec<_>>(),
        "environment": environment_spec_to_json_value(&spec.environment)?,
        "interface": interface_spec_to_json_value(&spec.interface),
        "controller": controller_spec_to_json_value(&spec.controller)?,
        "runtime": runtime_spec_to_json_value(&spec.runtime),
    }))
}

fn tune_spec_to_json_value(spec: &TuneSpec) -> SpecResult<serde_json::Value> {
    Ok(serde_json::json!({
        "schema_version": SPEC_DOCUMENT_SCHEMA_VERSION,
        "kind": "tune",
        "assets": spec.assets.iter().map(asset_binding_to_json_value).collect::<Vec<_>>(),
        "input_asset": spec.input_asset,
        "baseline_candidate": compression_backend_to_json_value(&spec.baseline_candidate)?,
        "controller": tune_controller_to_json_value(&spec.controller),
        "bounds": tune_bounds_to_json_value(&spec.bounds),
        "eval_time_limit_seconds": spec.eval_time_limit_seconds,
        "time_budget_seconds": spec.time_budget_seconds,
        "min_throughput_bytes_per_second": spec.min_throughput_bytes_per_second,
        "max_memory_bytes": spec.max_memory_bytes,
        "output_config_path": spec.output_config_path,
        "seed": spec.seed,
        "report_path": spec.report_path,
    }))
}

fn asset_binding_to_json_value(binding: &AssetBinding) -> serde_json::Value {
    serde_json::json!({
        "id": binding.id,
        "path": binding.path,
    })
}

fn environment_spec_to_json_value(spec: &EnvironmentSpec) -> SpecResult<serde_json::Value> {
    match spec {
        EnvironmentSpec::Builtin { builtin } => Ok(serde_json::json!({
            "kind": "builtin",
            "name": builtin_environment_name(*builtin),
        })),
        #[cfg(feature = "vm")]
        EnvironmentSpec::NyxVm(vm) => Ok(serde_json::json!({
            "kind": "nyx_vm",
            "firecracker_config_asset": vm.firecracker_config_asset,
            "instance_id": vm.instance_id,
            "shared_region_name": vm.shared_region_name,
            "shared_region_size": vm.shared_region_size,
            "shared_memory_policy": shared_memory_policy_name(vm.shared_memory_policy),
            "step_timeout_ms": vm.step_timeout_ms,
            "boot_timeout_ms": vm.boot_timeout_ms,
            "episode_steps": vm.episode_steps,
            "step_cost": vm.step_cost,
            "observation_policy": vm.observation_policy,
            "observation_bits": vm.observation_bits,
            "observation_stream_len": vm.observation_stream_len,
            "observation_stream_mode": vm.observation_stream_mode,
            "observation_pad_byte": vm.observation_pad_byte,
            "reward_bits": vm.reward_bits,
            "reward_policy": vm_reward_policy_to_json_value(&vm.reward_policy),
            "reward_shaping": vm.reward_shaping.as_ref().map(vm_reward_shaping_to_json_value),
            "action_source": vm_action_source_to_json_value(&vm.action_source),
            "action_filter": vm.action_filter.as_ref().map(vm_action_filter_to_json_value),
            "protocol": {
                "action_prefix": vm.action_prefix,
                "action_suffix": vm.action_suffix,
                "obs_prefix": vm.obs_prefix,
                "rew_prefix": vm.rew_prefix,
                "done_prefix": vm.done_prefix,
                "data_prefix": vm.data_prefix,
                "wire_encoding": vm.wire_encoding,
            },
            "stats_backend": rate_backend_to_json_value(&vm.stats_backend)?,
            "trace": vm.trace.as_ref().map(vm_trace_to_json_value),
            "debug_mode": vm.debug_mode,
        })),
    }
}

fn interface_spec_to_json_value(spec: &PlannerInterfaceSpec) -> serde_json::Value {
    serde_json::json!({
        "observation_bits": spec.observation_bits,
        "observation_stream_len": spec.observation_stream_len,
        "observation_key_mode": observation_key_mode_name(spec.observation_key_mode),
        "reward_bits": spec.reward_bits,
        "agent_actions": spec.agent_actions,
        "min_reward": spec.min_reward,
        "max_reward": spec.max_reward,
        "reward_offset": spec.reward_offset,
    })
}

fn controller_spec_to_json_value(spec: &ControllerSpec) -> SpecResult<serde_json::Value> {
    match spec {
        ControllerSpec::McAixi(inner) => Ok(serde_json::json!({
            "kind": "mc_aixi",
            "predictor": rate_backend_to_json_value(&inner.predictor)?,
            "predictor_max_order": inner.predictor_max_order,
            "agent_horizon": inner.agent_horizon,
            "num_simulations": inner.num_simulations,
            "exploration_exploitation_ratio": inner.exploration_exploitation_ratio,
            "discount_gamma": inner.discount_gamma,
        })),
        ControllerSpec::AiqiDiscounted(inner) => Ok(serde_json::json!({
            "kind": "aiqi_discounted",
            "predictor": rate_backend_to_json_value(&inner.predictor)?,
            "predictor_max_order": inner.predictor_max_order,
            "discount_gamma": inner.discount_gamma,
            "return_horizon": inner.return_horizon,
            "return_bins": inner.return_bins,
            "augmentation_period": inner.augmentation_period,
            "history_prune_keep_steps": inner.history_prune_keep_steps,
            "baseline_exploration": inner.baseline_exploration,
        })),
        ControllerSpec::AiqiWarmstartExactJh(inner) => Ok(serde_json::json!({
            "kind": "aiqi_warmstart_exact_jh",
            "predictor": rate_backend_to_json_value(&inner.predictor)?,
            "predictor_max_order": inner.predictor_max_order,
            "return_horizon": inner.return_horizon,
            "return_bins": inner.return_bins,
            "label_phase_period": inner.label_phase_period,
            "teacher_dataset_asset": inner.teacher_dataset_asset,
            "planner_simulations_per_step": inner.planner_simulations_per_step,
        })),
    }
}

fn runtime_spec_to_json_value(spec: &PlannerRuntimeSpec) -> serde_json::Value {
    serde_json::json!({
        "random_seed": spec.random_seed,
        "learn_cycles": spec.learn_cycles,
        "eval_cycles": spec.eval_cycles,
        "terminate_lifetime": spec.terminate_lifetime,
        "log_every": spec.log_every,
        "perf": spec.perf,
        "vm_perf_only": spec.vm_perf_only,
        "explore_epsilon": spec.explore_epsilon,
        "explore_gamma": spec.explore_gamma,
    })
}

fn tune_bounds_to_json_value(bounds: &TuneBoundsSpec) -> serde_json::Value {
    serde_json::json!({
        "allowed_backends": bounds.allowed_backends,
        "forbidden_backends": bounds.forbidden_backends,
        "parameter_ranges": bounds.parameter_ranges.iter().map(tune_parameter_range_to_json_value).collect::<Vec<_>>(),
        "max_experts": bounds.max_experts,
        "max_mixture_nesting_depth": bounds.max_mixture_nesting_depth,
        "min_experts": bounds.min_experts,
        "allow_duplicate_experts": bounds.allow_duplicate_experts,
        "required_experts": bounds.required_experts,
        "forbidden_expert_pairs": bounds.forbidden_expert_pairs.iter().map(|(a, b)| vec![a, b]).collect::<Vec<_>>(),
    })
}

fn tune_controller_to_json_value(spec: &TuneControllerSpec) -> serde_json::Value {
    match spec {
        TuneControllerSpec::AnnealedHillClimbing(inner) => serde_json::json!({
            "kind": "annealed_hill_climbing",
            "max_mutation_radius": inner.max_mutation_radius,
        }),
        TuneControllerSpec::McAixiFacCtw(inner) => serde_json::json!({
            "kind": "mc_aixi_fac_ctw",
            "interface": interface_spec_to_json_value(&inner.interface),
            "planner_simulations_per_step": inner.planner_simulations_per_step,
        }),
        TuneControllerSpec::AiqiDiscounted(inner) => serde_json::json!({
            "kind": "aiqi_discounted",
            "interface": interface_spec_to_json_value(&inner.interface),
            "planner_simulations_per_step": inner.planner_simulations_per_step,
            "return_horizon": inner.return_horizon,
            "return_bins": inner.return_bins,
            "discount_factor": inner.discount_factor,
        }),
        TuneControllerSpec::AiqiWarmstartExactJh(inner) => serde_json::json!({
            "kind": "aiqi_warmstart_exact_jh",
            "interface": interface_spec_to_json_value(&inner.interface),
            "planner_simulations_per_step": inner.planner_simulations_per_step,
            "return_horizon": inner.return_horizon,
            "warmstart_teacher_dataset_asset": inner.warmstart_teacher_dataset_asset,
            "label_phase_period": inner.label_phase_period,
        }),
    }
}

fn tune_parameter_range_to_json_value(range: &TuneParameterRangeSpec) -> serde_json::Value {
    serde_json::json!({
        "parameter": range.parameter,
        "min": range.min,
        "max": range.max,
    })
}

#[cfg(feature = "vm")]
fn vm_reward_policy_to_json_value(policy: &VmRewardPolicySpec) -> serde_json::Value {
    match policy {
        VmRewardPolicySpec::FromGuest => serde_json::json!({ "kind": "from_guest" }),
        VmRewardPolicySpec::Pattern {
            pattern,
            base_reward,
            bonus_reward,
        } => serde_json::json!({
            "kind": "pattern",
            "pattern": pattern,
            "base_reward": base_reward,
            "bonus_reward": bonus_reward,
        }),
    }
}

#[cfg(feature = "vm")]
fn vm_reward_shaping_to_json_value(spec: &VmRewardShapingSpec) -> serde_json::Value {
    match spec {
        VmRewardShapingSpec::EntropyReduction {
            baseline_asset,
            max_order,
            scale,
            crash_bonus,
            timeout_bonus,
        } => serde_json::json!({
            "kind": "entropy_reduction",
            "baseline_asset": baseline_asset,
            "max_order": max_order,
            "scale": scale,
            "crash_bonus": crash_bonus,
            "timeout_bonus": timeout_bonus,
        }),
        VmRewardShapingSpec::TraceEntropy {
            max_order,
            scale,
            normalize,
        } => serde_json::json!({
            "kind": "trace_entropy",
            "max_order": max_order,
            "scale": scale,
            "normalize": normalize,
        }),
    }
}

#[cfg(feature = "vm")]
fn vm_action_source_to_json_value(spec: &VmRuntimeActionSourceSpec) -> serde_json::Value {
    match spec {
        VmRuntimeActionSourceSpec::Literal {
            names,
            payloads,
            encoding,
        } => serde_json::json!({
            "kind": "literal",
            "encoding": encoding,
            "actions": payloads.iter().enumerate().map(|(idx, payload)| serde_json::json!({
                "name": names.get(idx).cloned().flatten(),
                "payload": payload,
            })).collect::<Vec<_>>(),
        }),
        VmRuntimeActionSourceSpec::Fuzz {
            seeds,
            encoding,
            mutators,
            min_len,
            max_len,
            dictionary,
            rng_seed,
        } => serde_json::json!({
            "kind": "fuzz",
            "encoding": encoding,
            "seeds": seeds,
            "mutators": mutators,
            "min_len": min_len,
            "max_len": max_len,
            "dictionary": dictionary,
            "rng_seed": rng_seed,
        }),
    }
}

#[cfg(feature = "vm")]
fn vm_action_filter_to_json_value(spec: &VmActionFilterSpec) -> serde_json::Value {
    serde_json::json!({
        "min_entropy": spec.min_entropy,
        "max_entropy": spec.max_entropy,
        "min_intrinsic_dependence": spec.min_intrinsic_dependence,
        "min_novelty": spec.min_novelty,
        "novelty_prior_asset": spec.novelty_prior_asset,
        "max_order": spec.max_order,
        "reject_reward": spec.reject_reward,
    })
}

#[cfg(feature = "vm")]
fn vm_trace_to_json_value(spec: &VmTraceSpec) -> serde_json::Value {
    serde_json::json!({
        "shared_region_name": spec.shared_region_name,
        "max_bytes": spec.max_bytes,
        "reset_on_episode": spec.reset_on_episode,
    })
}

fn parse_spec_document_json_value(
    value: &serde_json::Value,
    base_dir: &Path,
) -> SpecResult<SpecDocument> {
    let version = value["schema_version"].as_u64().unwrap_or(0);
    if version != SPEC_DOCUMENT_SCHEMA_VERSION as u64 {
        return Err(SpecError::new(format!(
            "unsupported spec document schema_version '{version}'"
        )));
    }
    let kind = value["kind"]
        .as_str()
        .ok_or_else(|| SpecError::new("spec document kind is required"))?;
    match kind {
        "planner_run" => Ok(SpecDocument::PlannerRun(parse_planner_run_json_value(
            value, base_dir,
        )?)),
        "tune" => Ok(SpecDocument::Tune(parse_tune_spec_json_value(
            value, base_dir,
        )?)),
        "rate_backend" => Ok(SpecDocument::RateBackend(parse_rate_backend_json(
            &value["backend"],
            base_dir,
            crate::api::MAX_MIXTURE_NESTING,
        )?)),
        "compression_backend" => Ok(SpecDocument::CompressionBackend(
            parse_compression_backend_json(
                &value["backend"],
                base_dir,
                None,
                crate::compression::FramingMode::Framed,
            )?,
        )),
        other => Err(SpecError::new(format!(
            "unknown spec document kind '{other}'"
        ))),
    }
}

fn parse_planner_run_json_value(
    value: &serde_json::Value,
    base_dir: &Path,
) -> SpecResult<PlannerRunSpec> {
    Ok(PlannerRunSpec {
        assets: parse_asset_bindings(&value["assets"])?,
        environment: parse_environment_spec(&value["environment"], base_dir)?,
        interface: parse_interface_spec(&value["interface"])?,
        controller: parse_controller_spec(&value["controller"], base_dir)?,
        runtime: parse_runtime_spec(&value["runtime"])?,
    })
}

fn parse_tune_spec_json_value(value: &serde_json::Value, base_dir: &Path) -> SpecResult<TuneSpec> {
    Ok(TuneSpec {
        assets: parse_asset_bindings(&value["assets"])?,
        input_asset: required_string(&value["input_asset"], "input_asset")?,
        baseline_candidate: parse_compression_backend_json(
            &value["baseline_candidate"],
            base_dir,
            None,
            crate::compression::FramingMode::Framed,
        )?,
        controller: parse_tune_controller_spec(&value["controller"])?,
        bounds: parse_tune_bounds_spec(&value["bounds"])?,
        eval_time_limit_seconds: required_f64(
            &value["eval_time_limit_seconds"],
            "eval_time_limit_seconds",
        )?,
        time_budget_seconds: required_f64(&value["time_budget_seconds"], "time_budget_seconds")?,
        min_throughput_bytes_per_second: required_f64(
            &value["min_throughput_bytes_per_second"],
            "min_throughput_bytes_per_second",
        )?,
        max_memory_bytes: required_u64(&value["max_memory_bytes"], "max_memory_bytes")?,
        output_config_path: required_string(&value["output_config_path"], "output_config_path")?,
        seed: required_u64(&value["seed"], "seed")?,
        report_path: optional_string(&value["report_path"]),
    })
}

fn parse_asset_bindings(value: &serde_json::Value) -> SpecResult<Vec<AssetBinding>> {
    let Some(items) = value.as_array() else {
        return Ok(Vec::new());
    };
    items
        .iter()
        .map(|item| {
            Ok(AssetBinding {
                id: required_string(&item["id"], "assets[].id")?,
                path: required_string(&item["path"], "assets[].path")?,
            })
        })
        .collect()
}

fn parse_environment_spec(
    value: &serde_json::Value,
    base_dir: &Path,
) -> SpecResult<EnvironmentSpec> {
    #[cfg(not(feature = "vm"))]
    let _ = base_dir;
    let kind = value["kind"]
        .as_str()
        .ok_or_else(|| SpecError::new("environment.kind is required"))?;
    match kind {
        "builtin" => Ok(EnvironmentSpec::Builtin {
            builtin: parse_builtin_environment(
                value["name"]
                    .as_str()
                    .ok_or_else(|| SpecError::new("environment.name is required"))?,
            )?,
        }),
        #[cfg(feature = "vm")]
        "nyx_vm" => Ok(EnvironmentSpec::NyxVm(VmEnvironmentSpec {
            firecracker_config_asset: required_string(
                &value["firecracker_config_asset"],
                "environment.firecracker_config_asset",
            )?,
            instance_id: optional_string(&value["instance_id"])
                .unwrap_or_else(|| "aixi-nyx".to_string()),
            shared_region_name: optional_string(&value["shared_region_name"])
                .unwrap_or_else(|| "shared".to_string()),
            shared_region_size: value["shared_region_size"].as_u64().unwrap_or(4096) as usize,
            shared_memory_policy: parse_shared_memory_policy(
                value["shared_memory_policy"].as_str().unwrap_or("snapshot"),
            )?,
            step_timeout_ms: value["step_timeout_ms"].as_u64().unwrap_or(100),
            boot_timeout_ms: value["boot_timeout_ms"].as_u64().unwrap_or(30_000),
            episode_steps: value["episode_steps"].as_u64().unwrap_or(100) as usize,
            step_cost: value["step_cost"].as_i64().unwrap_or(0),
            observation_policy: optional_string(&value["observation_policy"])
                .unwrap_or_else(|| "shared_memory".to_string()),
            observation_bits: value["observation_bits"].as_u64().unwrap_or(8) as usize,
            observation_stream_len: value["observation_stream_len"].as_u64().unwrap_or(64) as usize,
            observation_stream_mode: optional_string(&value["observation_stream_mode"])
                .unwrap_or_else(|| "pad_truncate".to_string()),
            observation_pad_byte: value["observation_pad_byte"].as_u64().unwrap_or(0) as u8,
            reward_bits: value["reward_bits"].as_u64().unwrap_or(8) as usize,
            reward_policy: parse_vm_reward_policy(&value["reward_policy"])?,
            reward_shaping: parse_optional_vm_reward_shaping(&value["reward_shaping"])?,
            action_source: parse_vm_action_source(&value["action_source"])?,
            action_filter: parse_optional_vm_action_filter(&value["action_filter"])?,
            action_prefix: value["protocol"]["action_prefix"]
                .as_str()
                .unwrap_or("ACT ")
                .to_string(),
            action_suffix: value["protocol"]["action_suffix"]
                .as_str()
                .unwrap_or("\n")
                .to_string(),
            obs_prefix: value["protocol"]["obs_prefix"]
                .as_str()
                .unwrap_or("OBS ")
                .to_string(),
            rew_prefix: value["protocol"]["rew_prefix"]
                .as_str()
                .unwrap_or("REW ")
                .to_string(),
            done_prefix: value["protocol"]["done_prefix"]
                .as_str()
                .unwrap_or("DONE ")
                .to_string(),
            data_prefix: value["protocol"]["data_prefix"]
                .as_str()
                .unwrap_or("DATA ")
                .to_string(),
            wire_encoding: value["protocol"]["wire_encoding"]
                .as_str()
                .unwrap_or("hex")
                .to_string(),
            stats_backend: parse_rate_backend_json(
                &value["stats_backend"],
                base_dir,
                crate::api::MAX_MIXTURE_NESTING,
            )?,
            trace: parse_optional_vm_trace(&value["trace"])?,
            debug_mode: value["debug_mode"].as_bool().unwrap_or(false),
        })),
        #[cfg(not(feature = "vm"))]
        "nyx_vm" => Err(SpecError::new(
            "nyx_vm environment requires the 'vm' feature",
        )),
        other => Err(SpecError::new(format!(
            "unknown environment kind '{other}'"
        ))),
    }
}

fn parse_interface_spec(value: &serde_json::Value) -> SpecResult<PlannerInterfaceSpec> {
    Ok(PlannerInterfaceSpec {
        observation_bits: required_u64(&value["observation_bits"], "interface.observation_bits")?
            as usize,
        observation_stream_len: required_u64(
            &value["observation_stream_len"],
            "interface.observation_stream_len",
        )? as usize,
        observation_key_mode: parse_observation_key_mode(
            value["observation_key_mode"]
                .as_str()
                .unwrap_or("full_stream"),
        )?,
        reward_bits: required_u64(&value["reward_bits"], "interface.reward_bits")? as usize,
        agent_actions: required_u64(&value["agent_actions"], "interface.agent_actions")? as usize,
        min_reward: required_i64(&value["min_reward"], "interface.min_reward")?,
        max_reward: required_i64(&value["max_reward"], "interface.max_reward")?,
        reward_offset: required_i64(&value["reward_offset"], "interface.reward_offset")?,
    })
}

fn parse_controller_spec(value: &serde_json::Value, base_dir: &Path) -> SpecResult<ControllerSpec> {
    let kind = value["kind"]
        .as_str()
        .ok_or_else(|| SpecError::new("controller.kind is required"))?;
    match kind {
        "mc_aixi" => Ok(ControllerSpec::McAixi(McAixiControllerSpec {
            predictor: parse_rate_backend_json(
                &value["predictor"],
                base_dir,
                crate::api::MAX_MIXTURE_NESTING,
            )?,
            predictor_max_order: value["predictor_max_order"].as_i64().unwrap_or(20),
            agent_horizon: required_u64(&value["agent_horizon"], "controller.agent_horizon")?
                as usize,
            num_simulations: required_u64(&value["num_simulations"], "controller.num_simulations")?
                as usize,
            exploration_exploitation_ratio: required_f64(
                &value["exploration_exploitation_ratio"],
                "controller.exploration_exploitation_ratio",
            )?,
            discount_gamma: required_f64(&value["discount_gamma"], "controller.discount_gamma")?,
        })),
        "aiqi_discounted" => Ok(ControllerSpec::AiqiDiscounted(
            AiqiDiscountedControllerSpec {
                predictor: parse_rate_backend_json(
                    &value["predictor"],
                    base_dir,
                    crate::api::MAX_MIXTURE_NESTING,
                )?,
                predictor_max_order: value["predictor_max_order"].as_i64().unwrap_or(20),
                discount_gamma: required_f64(
                    &value["discount_gamma"],
                    "controller.discount_gamma",
                )?,
                return_horizon: required_u64(&value["return_horizon"], "controller.return_horizon")?
                    as usize,
                return_bins: required_u64(&value["return_bins"], "controller.return_bins")?
                    as usize,
                augmentation_period: required_u64(
                    &value["augmentation_period"],
                    "controller.augmentation_period",
                )? as usize,
                history_prune_keep_steps: value["history_prune_keep_steps"]
                    .as_u64()
                    .map(|n| n as usize),
                baseline_exploration: required_f64(
                    &value["baseline_exploration"],
                    "controller.baseline_exploration",
                )?,
            },
        )),
        "aiqi_warmstart_exact_jh" => Ok(ControllerSpec::AiqiWarmstartExactJh(
            WarmStartExactJhControllerSpec {
                predictor: parse_rate_backend_json(
                    &value["predictor"],
                    base_dir,
                    crate::api::MAX_MIXTURE_NESTING,
                )?,
                predictor_max_order: value["predictor_max_order"].as_i64().unwrap_or(20),
                return_horizon: required_u64(&value["return_horizon"], "controller.return_horizon")?
                    as usize,
                return_bins: required_u64(&value["return_bins"], "controller.return_bins")?
                    as usize,
                label_phase_period: required_u64(
                    &value["label_phase_period"],
                    "controller.label_phase_period",
                )? as usize,
                teacher_dataset_asset: required_string(
                    &value["teacher_dataset_asset"],
                    "controller.teacher_dataset_asset",
                )?,
                planner_simulations_per_step: required_u64(
                    &value["planner_simulations_per_step"],
                    "controller.planner_simulations_per_step",
                )? as usize,
            },
        )),
        other => Err(SpecError::new(format!("unknown controller kind '{other}'"))),
    }
}

fn parse_runtime_spec(value: &serde_json::Value) -> SpecResult<PlannerRuntimeSpec> {
    Ok(PlannerRuntimeSpec {
        random_seed: value["random_seed"].as_u64(),
        learn_cycles: value["learn_cycles"].as_u64().map(|n| n as usize),
        eval_cycles: value["eval_cycles"].as_u64().map(|n| n as usize),
        terminate_lifetime: value["terminate_lifetime"].as_u64().unwrap_or(20) as usize,
        log_every: value["log_every"].as_u64().unwrap_or(1) as usize,
        perf: value["perf"].as_bool().unwrap_or(false),
        vm_perf_only: value["vm_perf_only"].as_bool().unwrap_or(false),
        explore_epsilon: value["explore_epsilon"].as_f64().unwrap_or(0.0),
        explore_gamma: value["explore_gamma"].as_f64().unwrap_or(1.0),
    })
}

fn parse_tune_bounds_spec(value: &serde_json::Value) -> SpecResult<TuneBoundsSpec> {
    Ok(TuneBoundsSpec {
        allowed_backends: string_list(&value["allowed_backends"])?,
        forbidden_backends: string_list(&value["forbidden_backends"])?,
        parameter_ranges: parse_tune_parameter_ranges(&value["parameter_ranges"])?,
        max_experts: required_u64(&value["max_experts"], "bounds.max_experts")? as usize,
        max_mixture_nesting_depth: required_u64(
            &value["max_mixture_nesting_depth"],
            "bounds.max_mixture_nesting_depth",
        )? as usize,
        min_experts: value["min_experts"].as_u64().map(|n| n as usize),
        allow_duplicate_experts: value["allow_duplicate_experts"].as_bool(),
        required_experts: string_list(&value["required_experts"])?,
        forbidden_expert_pairs: pair_list(&value["forbidden_expert_pairs"])?,
    })
}

fn parse_tune_controller_spec(value: &serde_json::Value) -> SpecResult<TuneControllerSpec> {
    let kind = value["kind"]
        .as_str()
        .ok_or_else(|| SpecError::new("controller.kind is required"))?;
    match kind {
        "annealed_hill_climbing" => Ok(TuneControllerSpec::AnnealedHillClimbing(
            AnnealedHillClimbingTuneControllerSpec {
                max_mutation_radius: required_u64(
                    &value["max_mutation_radius"],
                    "controller.max_mutation_radius",
                )? as usize,
            },
        )),
        "mc_aixi_fac_ctw" => Ok(TuneControllerSpec::McAixiFacCtw(
            McAixiFacCtwTuneControllerSpec {
                interface: parse_interface_spec(&value["interface"])?,
                planner_simulations_per_step: required_u64(
                    &value["planner_simulations_per_step"],
                    "controller.planner_simulations_per_step",
                )? as usize,
            },
        )),
        "aiqi_discounted" => Ok(TuneControllerSpec::AiqiDiscounted(
            AiqiDiscountedTuneControllerSpec {
                interface: parse_interface_spec(&value["interface"])?,
                planner_simulations_per_step: required_u64(
                    &value["planner_simulations_per_step"],
                    "controller.planner_simulations_per_step",
                )? as usize,
                return_horizon: required_u64(&value["return_horizon"], "controller.return_horizon")?
                    as usize,
                return_bins: required_u64(&value["return_bins"], "controller.return_bins")?
                    as usize,
                discount_factor: required_f64(
                    &value["discount_factor"],
                    "controller.discount_factor",
                )?,
            },
        )),
        "aiqi_warmstart_exact_jh" => Ok(TuneControllerSpec::AiqiWarmstartExactJh(
            WarmStartExactJhTuneControllerSpec {
                interface: parse_interface_spec(&value["interface"])?,
                planner_simulations_per_step: required_u64(
                    &value["planner_simulations_per_step"],
                    "controller.planner_simulations_per_step",
                )? as usize,
                return_horizon: required_u64(&value["return_horizon"], "controller.return_horizon")?
                    as usize,
                warmstart_teacher_dataset_asset: required_string(
                    &value["warmstart_teacher_dataset_asset"],
                    "controller.warmstart_teacher_dataset_asset",
                )?,
                label_phase_period: required_u64(
                    &value["label_phase_period"],
                    "controller.label_phase_period",
                )? as usize,
            },
        )),
        other => Err(SpecError::new(format!(
            "unknown tune controller kind '{other}'"
        ))),
    }
}

fn parse_tune_parameter_ranges(
    value: &serde_json::Value,
) -> SpecResult<Vec<TuneParameterRangeSpec>> {
    let Some(items) = value.as_array() else {
        return Ok(Vec::new());
    };
    items
        .iter()
        .map(|item| {
            Ok(TuneParameterRangeSpec {
                parameter: required_string(
                    &item["parameter"],
                    "bounds.parameter_ranges[].parameter",
                )?,
                min: required_f64(&item["min"], "bounds.parameter_ranges[].min")?,
                max: required_f64(&item["max"], "bounds.parameter_ranges[].max")?,
            })
        })
        .collect()
}

#[cfg(feature = "vm")]
fn parse_vm_reward_policy(value: &serde_json::Value) -> SpecResult<VmRewardPolicySpec> {
    match value["kind"].as_str().unwrap_or("from_guest") {
        "from_guest" => Ok(VmRewardPolicySpec::FromGuest),
        "pattern" => Ok(VmRewardPolicySpec::Pattern {
            pattern: required_string(&value["pattern"], "environment.reward_policy.pattern")?,
            base_reward: value["base_reward"].as_i64().unwrap_or(0),
            bonus_reward: value["bonus_reward"].as_i64().unwrap_or(10),
        }),
        other => Err(SpecError::new(format!(
            "unknown vm reward policy '{other}'"
        ))),
    }
}

#[cfg(feature = "vm")]
fn parse_optional_vm_reward_shaping(
    value: &serde_json::Value,
) -> SpecResult<Option<VmRewardShapingSpec>> {
    if value.is_null() {
        return Ok(None);
    }
    let spec = match value["kind"].as_str().unwrap_or("trace_entropy") {
        "entropy_reduction" => VmRewardShapingSpec::EntropyReduction {
            baseline_asset: required_string(
                &value["baseline_asset"],
                "environment.reward_shaping.baseline_asset",
            )?,
            max_order: value["max_order"].as_i64().unwrap_or(8),
            scale: value["scale"].as_f64().unwrap_or(1.0),
            crash_bonus: value["crash_bonus"].as_i64(),
            timeout_bonus: value["timeout_bonus"].as_i64(),
        },
        "trace_entropy" => VmRewardShapingSpec::TraceEntropy {
            max_order: value["max_order"].as_i64().unwrap_or(8),
            scale: value["scale"].as_f64().unwrap_or(1.0),
            normalize: value["normalize"].as_bool().unwrap_or(false),
        },
        other => {
            return Err(SpecError::new(format!(
                "unknown vm reward shaping '{other}'"
            )));
        }
    };
    Ok(Some(spec))
}

#[cfg(feature = "vm")]
fn parse_vm_action_source(value: &serde_json::Value) -> SpecResult<VmRuntimeActionSourceSpec> {
    match value["kind"].as_str().unwrap_or("literal") {
        "literal" => {
            let encoding = value["encoding"].as_str().unwrap_or("utf8").to_string();
            let actions = value["actions"]
                .as_array()
                .ok_or_else(|| SpecError::new("environment.action_source.actions is required"))?;
            let mut names = Vec::with_capacity(actions.len());
            let mut payloads = Vec::with_capacity(actions.len());
            for action in actions {
                names.push(optional_string(&action["name"]));
                payloads.push(required_string(
                    &action["payload"],
                    "environment.action_source.actions[].payload",
                )?);
            }
            Ok(VmRuntimeActionSourceSpec::Literal {
                names,
                payloads,
                encoding,
            })
        }
        "fuzz" => Ok(VmRuntimeActionSourceSpec::Fuzz {
            seeds: string_list(&value["seeds"])?,
            encoding: value["encoding"].as_str().unwrap_or("utf8").to_string(),
            mutators: string_list(&value["mutators"])?,
            min_len: value["min_len"].as_u64().unwrap_or(1) as usize,
            max_len: value["max_len"].as_u64().unwrap_or(4096) as usize,
            dictionary: string_list(&value["dictionary"])?,
            rng_seed: value["rng_seed"].as_u64().unwrap_or(0),
        }),
        other => Err(SpecError::new(format!(
            "unknown vm action source '{other}'"
        ))),
    }
}

#[cfg(feature = "vm")]
fn parse_optional_vm_action_filter(
    value: &serde_json::Value,
) -> SpecResult<Option<VmActionFilterSpec>> {
    if value.is_null() {
        return Ok(None);
    }
    Ok(Some(VmActionFilterSpec {
        min_entropy: value["min_entropy"].as_f64(),
        max_entropy: value["max_entropy"].as_f64(),
        min_intrinsic_dependence: value["min_intrinsic_dependence"].as_f64(),
        min_novelty: value["min_novelty"].as_f64(),
        novelty_prior_asset: optional_string(&value["novelty_prior_asset"]),
        max_order: value["max_order"].as_i64().unwrap_or(8),
        reject_reward: value["reject_reward"].as_i64(),
    }))
}

#[cfg(feature = "vm")]
fn parse_optional_vm_trace(value: &serde_json::Value) -> SpecResult<Option<VmTraceSpec>> {
    if value.is_null() {
        return Ok(None);
    }
    Ok(Some(VmTraceSpec {
        shared_region_name: optional_string(&value["shared_region_name"]),
        max_bytes: value["max_bytes"].as_u64().unwrap_or(1_000_000) as usize,
        reset_on_episode: value["reset_on_episode"].as_bool().unwrap_or(false),
    }))
}

fn encode_spec_document_payload(doc: &SpecDocument) -> Vec<u8> {
    let mut out = Vec::new();
    out.extend_from_slice(DOCUMENT_MAGIC);
    out.push(DOCUMENT_BINARY_VERSION);
    match doc {
        SpecDocument::PlannerRun(spec) => {
            out.push(0);
            encode_planner_run(spec, &mut out);
        }
        SpecDocument::Tune(spec) => {
            out.push(1);
            encode_tune_spec(spec, &mut out);
        }
        SpecDocument::RateBackend(backend) => {
            out.push(2);
            push_string(&mut out, &backend.to_canonical_json().unwrap_or_default());
        }
        SpecDocument::CompressionBackend(backend) => {
            out.push(3);
            push_string(&mut out, &backend.to_canonical_json().unwrap_or_default());
        }
    }
    out
}

fn decode_spec_document(bytes: &[u8], base_dir: &Path) -> SpecResult<SpecDocument> {
    let mut cursor = Cursor::new(bytes);
    let magic = cursor.read_exact(4)?;
    if magic != DOCUMENT_MAGIC {
        return Err(SpecError::new("invalid spec document magic"));
    }
    let version = cursor.read_u8()?;
    if version != DOCUMENT_BINARY_VERSION {
        return Err(SpecError::new(format!(
            "unsupported spec document binary version '{version}'"
        )));
    }
    match cursor.read_u8()? {
        0 => Ok(SpecDocument::PlannerRun(decode_planner_run(
            &mut cursor,
            base_dir,
        )?)),
        1 => Ok(SpecDocument::Tune(decode_tune_spec(&mut cursor, base_dir)?)),
        2 => {
            let json = cursor.read_string()?;
            let value: serde_json::Value = serde_json::from_str(&json)?;
            Ok(SpecDocument::RateBackend(parse_rate_backend_json(
                &value,
                base_dir,
                crate::api::MAX_MIXTURE_NESTING,
            )?))
        }
        3 => {
            let json = cursor.read_string()?;
            let value: serde_json::Value = serde_json::from_str(&json)?;
            Ok(SpecDocument::CompressionBackend(
                parse_compression_backend_json(
                    &value,
                    base_dir,
                    None,
                    crate::compression::FramingMode::Framed,
                )?,
            ))
        }
        tag => Err(SpecError::new(format!("unknown spec document tag '{tag}'"))),
    }
}

fn encode_planner_run(spec: &PlannerRunSpec, out: &mut Vec<u8>) {
    encode_assets(&spec.assets, out);
    encode_environment_spec(&spec.environment, out);
    encode_interface_spec(&spec.interface, out);
    encode_controller_spec(&spec.controller, out);
    encode_runtime_spec(&spec.runtime, out);
}

fn decode_planner_run(cursor: &mut Cursor<'_>, base_dir: &Path) -> SpecResult<PlannerRunSpec> {
    Ok(PlannerRunSpec {
        assets: decode_assets(cursor)?,
        environment: decode_environment_spec(cursor, base_dir)?,
        interface: decode_interface_spec(cursor)?,
        controller: decode_controller_spec(cursor, base_dir)?,
        runtime: decode_runtime_spec(cursor)?,
    })
}

fn encode_tune_spec(spec: &TuneSpec, out: &mut Vec<u8>) {
    encode_assets(&spec.assets, out);
    push_string(out, &spec.input_asset);
    push_string(
        out,
        &compression_backend_to_canonical_json(&spec.baseline_candidate).unwrap_or_default(),
    );
    encode_tune_controller(&spec.controller, out);
    encode_tune_bounds(&spec.bounds, out);
    push_f64(out, spec.eval_time_limit_seconds);
    push_f64(out, spec.time_budget_seconds);
    push_f64(out, spec.min_throughput_bytes_per_second);
    push_u64(out, spec.max_memory_bytes);
    push_string(out, &spec.output_config_path);
    push_u64(out, spec.seed);
    push_option_string(out, spec.report_path.as_deref());
}

fn decode_tune_spec(cursor: &mut Cursor<'_>, base_dir: &Path) -> SpecResult<TuneSpec> {
    let assets = decode_assets(cursor)?;
    let input_asset = cursor.read_string()?;
    let baseline_json = cursor.read_string()?;
    let baseline_value: serde_json::Value = serde_json::from_str(&baseline_json)?;
    Ok(TuneSpec {
        assets,
        input_asset,
        baseline_candidate: parse_compression_backend_json(
            &baseline_value,
            base_dir,
            None,
            crate::compression::FramingMode::Framed,
        )?,
        controller: decode_tune_controller(cursor)?,
        bounds: decode_tune_bounds(cursor)?,
        eval_time_limit_seconds: cursor.read_f64()?,
        time_budget_seconds: cursor.read_f64()?,
        min_throughput_bytes_per_second: cursor.read_f64()?,
        max_memory_bytes: cursor.read_u64()?,
        output_config_path: cursor.read_string()?,
        seed: cursor.read_u64()?,
        report_path: cursor.read_option_string()?,
    })
}

fn encode_assets(assets: &[AssetBinding], out: &mut Vec<u8>) {
    push_u64(out, assets.len() as u64);
    for asset in assets {
        push_string(out, &asset.id);
        push_string(out, &asset.path);
    }
}

fn decode_assets(cursor: &mut Cursor<'_>) -> SpecResult<Vec<AssetBinding>> {
    let len = cursor.read_u64()? as usize;
    let mut assets = Vec::with_capacity(len);
    for _ in 0..len {
        assets.push(AssetBinding {
            id: cursor.read_string()?,
            path: cursor.read_string()?,
        });
    }
    Ok(assets)
}

fn encode_environment_spec(spec: &EnvironmentSpec, out: &mut Vec<u8>) {
    match spec {
        EnvironmentSpec::Builtin { builtin } => {
            out.push(0);
            out.push(builtin_environment_tag(*builtin));
        }
        #[cfg(feature = "vm")]
        EnvironmentSpec::NyxVm(vm) => {
            out.push(1);
            push_string(out, &vm.firecracker_config_asset);
            push_string(out, &vm.instance_id);
            push_string(out, &vm.shared_region_name);
            push_u64(out, vm.shared_region_size as u64);
            out.push(shared_memory_policy_tag(vm.shared_memory_policy));
            push_u64(out, vm.step_timeout_ms);
            push_u64(out, vm.boot_timeout_ms);
            push_u64(out, vm.episode_steps as u64);
            push_i64(out, vm.step_cost);
            push_string(out, &vm.observation_policy);
            push_u64(out, vm.observation_bits as u64);
            push_u64(out, vm.observation_stream_len as u64);
            push_string(out, &vm.observation_stream_mode);
            out.push(vm.observation_pad_byte);
            push_u64(out, vm.reward_bits as u64);
            encode_vm_reward_policy(&vm.reward_policy, out);
            match &vm.reward_shaping {
                Some(shape) => {
                    out.push(1);
                    encode_vm_reward_shaping(shape, out);
                }
                None => out.push(0),
            }
            encode_vm_action_source(&vm.action_source, out);
            match &vm.action_filter {
                Some(filter) => {
                    out.push(1);
                    encode_vm_action_filter(filter, out);
                }
                None => out.push(0),
            }
            push_string(out, &vm.action_prefix);
            push_string(out, &vm.action_suffix);
            push_string(out, &vm.obs_prefix);
            push_string(out, &vm.rew_prefix);
            push_string(out, &vm.done_prefix);
            push_string(out, &vm.data_prefix);
            push_string(out, &vm.wire_encoding);
            push_string(
                out,
                &rate_backend_to_canonical_json(&vm.stats_backend).unwrap_or_default(),
            );
            match &vm.trace {
                Some(trace) => {
                    out.push(1);
                    encode_vm_trace(trace, out);
                }
                None => out.push(0),
            }
            push_bool(out, vm.debug_mode);
        }
    }
}

fn decode_environment_spec(
    cursor: &mut Cursor<'_>,
    base_dir: &Path,
) -> SpecResult<EnvironmentSpec> {
    #[cfg(not(feature = "vm"))]
    let _ = base_dir;
    match cursor.read_u8()? {
        0 => Ok(EnvironmentSpec::Builtin {
            builtin: decode_builtin_environment(cursor.read_u8()?)?,
        }),
        #[cfg(feature = "vm")]
        1 => {
            let baseline = cursor.read_string()?;
            let instance_id = cursor.read_string()?;
            let shared_region_name = cursor.read_string()?;
            let shared_region_size = cursor.read_u64()? as usize;
            let shared_memory_policy = decode_shared_memory_policy(cursor.read_u8()?)?;
            let step_timeout_ms = cursor.read_u64()?;
            let boot_timeout_ms = cursor.read_u64()?;
            let episode_steps = cursor.read_u64()? as usize;
            let step_cost = cursor.read_i64()?;
            let observation_policy = cursor.read_string()?;
            let observation_bits = cursor.read_u64()? as usize;
            let observation_stream_len = cursor.read_u64()? as usize;
            let observation_stream_mode = cursor.read_string()?;
            let observation_pad_byte = cursor.read_u8()?;
            let reward_bits = cursor.read_u64()? as usize;
            let reward_policy = decode_vm_reward_policy(cursor)?;
            let reward_shaping = if cursor.read_u8()? == 1 {
                Some(decode_vm_reward_shaping(cursor)?)
            } else {
                None
            };
            let action_source = decode_vm_action_source(cursor)?;
            let action_filter = if cursor.read_u8()? == 1 {
                Some(decode_vm_action_filter(cursor)?)
            } else {
                None
            };
            let action_prefix = cursor.read_string()?;
            let action_suffix = cursor.read_string()?;
            let obs_prefix = cursor.read_string()?;
            let rew_prefix = cursor.read_string()?;
            let done_prefix = cursor.read_string()?;
            let data_prefix = cursor.read_string()?;
            let wire_encoding = cursor.read_string()?;
            let backend_json = cursor.read_string()?;
            let backend_value: serde_json::Value = serde_json::from_str(&backend_json)?;
            let stats_backend =
                parse_rate_backend_json(&backend_value, base_dir, crate::api::MAX_MIXTURE_NESTING)?;
            let trace = if cursor.read_u8()? == 1 {
                Some(decode_vm_trace(cursor)?)
            } else {
                None
            };
            let debug_mode = cursor.read_bool()?;
            Ok(EnvironmentSpec::NyxVm(VmEnvironmentSpec {
                firecracker_config_asset: baseline,
                instance_id,
                shared_region_name,
                shared_region_size,
                shared_memory_policy,
                step_timeout_ms,
                boot_timeout_ms,
                episode_steps,
                step_cost,
                observation_policy,
                observation_bits,
                observation_stream_len,
                observation_stream_mode,
                observation_pad_byte,
                reward_bits,
                reward_policy,
                reward_shaping,
                action_source,
                action_filter,
                action_prefix,
                action_suffix,
                obs_prefix,
                rew_prefix,
                done_prefix,
                data_prefix,
                wire_encoding,
                stats_backend,
                trace,
                debug_mode,
            }))
        }
        #[cfg(not(feature = "vm"))]
        1 => Err(SpecError::new(
            "binary nyx_vm environment requires the 'vm' feature",
        )),
        tag => Err(SpecError::new(format!("unknown environment tag '{tag}'"))),
    }
}

fn encode_interface_spec(spec: &PlannerInterfaceSpec, out: &mut Vec<u8>) {
    push_u64(out, spec.observation_bits as u64);
    push_u64(out, spec.observation_stream_len as u64);
    out.push(observation_key_mode_tag(spec.observation_key_mode));
    push_u64(out, spec.reward_bits as u64);
    push_u64(out, spec.agent_actions as u64);
    push_i64(out, spec.min_reward);
    push_i64(out, spec.max_reward);
    push_i64(out, spec.reward_offset);
}

fn decode_interface_spec(cursor: &mut Cursor<'_>) -> SpecResult<PlannerInterfaceSpec> {
    Ok(PlannerInterfaceSpec {
        observation_bits: cursor.read_u64()? as usize,
        observation_stream_len: cursor.read_u64()? as usize,
        observation_key_mode: decode_observation_key_mode(cursor.read_u8()?)?,
        reward_bits: cursor.read_u64()? as usize,
        agent_actions: cursor.read_u64()? as usize,
        min_reward: cursor.read_i64()?,
        max_reward: cursor.read_i64()?,
        reward_offset: cursor.read_i64()?,
    })
}

fn encode_controller_spec(spec: &ControllerSpec, out: &mut Vec<u8>) {
    match spec {
        ControllerSpec::McAixi(inner) => {
            out.push(0);
            push_string(
                out,
                &rate_backend_to_canonical_json(&inner.predictor).unwrap_or_default(),
            );
            push_i64(out, inner.predictor_max_order);
            push_u64(out, inner.agent_horizon as u64);
            push_u64(out, inner.num_simulations as u64);
            push_f64(out, inner.exploration_exploitation_ratio);
            push_f64(out, inner.discount_gamma);
        }
        ControllerSpec::AiqiDiscounted(inner) => {
            out.push(1);
            push_string(
                out,
                &rate_backend_to_canonical_json(&inner.predictor).unwrap_or_default(),
            );
            push_i64(out, inner.predictor_max_order);
            push_f64(out, inner.discount_gamma);
            push_u64(out, inner.return_horizon as u64);
            push_u64(out, inner.return_bins as u64);
            push_u64(out, inner.augmentation_period as u64);
            push_option_u64(out, inner.history_prune_keep_steps.map(|n| n as u64));
            push_f64(out, inner.baseline_exploration);
        }
        ControllerSpec::AiqiWarmstartExactJh(inner) => {
            out.push(2);
            push_string(
                out,
                &rate_backend_to_canonical_json(&inner.predictor).unwrap_or_default(),
            );
            push_i64(out, inner.predictor_max_order);
            push_u64(out, inner.return_horizon as u64);
            push_u64(out, inner.return_bins as u64);
            push_u64(out, inner.label_phase_period as u64);
            push_string(out, &inner.teacher_dataset_asset);
            push_u64(out, inner.planner_simulations_per_step as u64);
        }
    }
}

fn decode_controller_spec(cursor: &mut Cursor<'_>, base_dir: &Path) -> SpecResult<ControllerSpec> {
    match cursor.read_u8()? {
        0 => {
            let json = cursor.read_string()?;
            let value: serde_json::Value = serde_json::from_str(&json)?;
            Ok(ControllerSpec::McAixi(McAixiControllerSpec {
                predictor: parse_rate_backend_json(
                    &value,
                    base_dir,
                    crate::api::MAX_MIXTURE_NESTING,
                )?,
                predictor_max_order: cursor.read_i64()?,
                agent_horizon: cursor.read_u64()? as usize,
                num_simulations: cursor.read_u64()? as usize,
                exploration_exploitation_ratio: cursor.read_f64()?,
                discount_gamma: cursor.read_f64()?,
            }))
        }
        1 => {
            let json = cursor.read_string()?;
            let value: serde_json::Value = serde_json::from_str(&json)?;
            Ok(ControllerSpec::AiqiDiscounted(
                AiqiDiscountedControllerSpec {
                    predictor: parse_rate_backend_json(
                        &value,
                        base_dir,
                        crate::api::MAX_MIXTURE_NESTING,
                    )?,
                    predictor_max_order: cursor.read_i64()?,
                    discount_gamma: cursor.read_f64()?,
                    return_horizon: cursor.read_u64()? as usize,
                    return_bins: cursor.read_u64()? as usize,
                    augmentation_period: cursor.read_u64()? as usize,
                    history_prune_keep_steps: cursor.read_option_u64()?.map(|n| n as usize),
                    baseline_exploration: cursor.read_f64()?,
                },
            ))
        }
        2 => {
            let json = cursor.read_string()?;
            let value: serde_json::Value = serde_json::from_str(&json)?;
            Ok(ControllerSpec::AiqiWarmstartExactJh(
                WarmStartExactJhControllerSpec {
                    predictor: parse_rate_backend_json(
                        &value,
                        base_dir,
                        crate::api::MAX_MIXTURE_NESTING,
                    )?,
                    predictor_max_order: cursor.read_i64()?,
                    return_horizon: cursor.read_u64()? as usize,
                    return_bins: cursor.read_u64()? as usize,
                    label_phase_period: cursor.read_u64()? as usize,
                    teacher_dataset_asset: cursor.read_string()?,
                    planner_simulations_per_step: cursor.read_u64()? as usize,
                },
            ))
        }
        tag => Err(SpecError::new(format!("unknown controller tag '{tag}'"))),
    }
}

fn encode_runtime_spec(spec: &PlannerRuntimeSpec, out: &mut Vec<u8>) {
    push_option_u64(out, spec.random_seed);
    push_option_u64(out, spec.learn_cycles.map(|n| n as u64));
    push_option_u64(out, spec.eval_cycles.map(|n| n as u64));
    push_u64(out, spec.terminate_lifetime as u64);
    push_u64(out, spec.log_every as u64);
    push_bool(out, spec.perf);
    push_bool(out, spec.vm_perf_only);
    push_f64(out, spec.explore_epsilon);
    push_f64(out, spec.explore_gamma);
}

fn decode_runtime_spec(cursor: &mut Cursor<'_>) -> SpecResult<PlannerRuntimeSpec> {
    Ok(PlannerRuntimeSpec {
        random_seed: cursor.read_option_u64()?,
        learn_cycles: cursor.read_option_u64()?.map(|n| n as usize),
        eval_cycles: cursor.read_option_u64()?.map(|n| n as usize),
        terminate_lifetime: cursor.read_u64()? as usize,
        log_every: cursor.read_u64()? as usize,
        perf: cursor.read_bool()?,
        vm_perf_only: cursor.read_bool()?,
        explore_epsilon: cursor.read_f64()?,
        explore_gamma: cursor.read_f64()?,
    })
}

fn encode_tune_bounds(bounds: &TuneBoundsSpec, out: &mut Vec<u8>) {
    push_string_list(out, &bounds.allowed_backends);
    push_string_list(out, &bounds.forbidden_backends);
    push_u64(out, bounds.parameter_ranges.len() as u64);
    for range in &bounds.parameter_ranges {
        push_string(out, &range.parameter);
        push_f64(out, range.min);
        push_f64(out, range.max);
    }
    push_u64(out, bounds.max_experts as u64);
    push_u64(out, bounds.max_mixture_nesting_depth as u64);
    push_option_u64(out, bounds.min_experts.map(|n| n as u64));
    match bounds.allow_duplicate_experts {
        Some(value) => {
            out.push(1);
            push_bool(out, value);
        }
        None => out.push(0),
    }
    push_string_list(out, &bounds.required_experts);
    push_u64(out, bounds.forbidden_expert_pairs.len() as u64);
    for (left, right) in &bounds.forbidden_expert_pairs {
        push_string(out, left);
        push_string(out, right);
    }
}

fn decode_tune_bounds(cursor: &mut Cursor<'_>) -> SpecResult<TuneBoundsSpec> {
    let allowed_backends = cursor.read_string_list()?;
    let forbidden_backends = cursor.read_string_list()?;
    let range_len = cursor.read_u64()? as usize;
    let mut parameter_ranges = Vec::with_capacity(range_len);
    for _ in 0..range_len {
        parameter_ranges.push(TuneParameterRangeSpec {
            parameter: cursor.read_string()?,
            min: cursor.read_f64()?,
            max: cursor.read_f64()?,
        });
    }
    let max_experts = cursor.read_u64()? as usize;
    let max_mixture_nesting_depth = cursor.read_u64()? as usize;
    let min_experts = cursor.read_option_u64()?.map(|n| n as usize);
    let allow_duplicate_experts = if cursor.read_u8()? == 1 {
        Some(cursor.read_bool()?)
    } else {
        None
    };
    let required_experts = cursor.read_string_list()?;
    let pair_len = cursor.read_u64()? as usize;
    let mut forbidden_expert_pairs = Vec::with_capacity(pair_len);
    for _ in 0..pair_len {
        forbidden_expert_pairs.push((cursor.read_string()?, cursor.read_string()?));
    }
    Ok(TuneBoundsSpec {
        allowed_backends,
        forbidden_backends,
        parameter_ranges,
        max_experts,
        max_mixture_nesting_depth,
        min_experts,
        allow_duplicate_experts,
        required_experts,
        forbidden_expert_pairs,
    })
}

fn encode_tune_controller(spec: &TuneControllerSpec, out: &mut Vec<u8>) {
    match spec {
        TuneControllerSpec::AnnealedHillClimbing(inner) => {
            out.push(tune_controller_kind_tag(
                TuneControllerKind::AnnealedHillClimbing,
            ));
            push_u64(out, inner.max_mutation_radius as u64);
        }
        TuneControllerSpec::McAixiFacCtw(inner) => {
            out.push(tune_controller_kind_tag(TuneControllerKind::McAixiFacCtw));
            encode_interface_spec(&inner.interface, out);
            push_u64(out, inner.planner_simulations_per_step as u64);
        }
        TuneControllerSpec::AiqiDiscounted(inner) => {
            out.push(tune_controller_kind_tag(TuneControllerKind::AiqiDiscounted));
            encode_interface_spec(&inner.interface, out);
            push_u64(out, inner.planner_simulations_per_step as u64);
            push_u64(out, inner.return_horizon as u64);
            push_u64(out, inner.return_bins as u64);
            push_f64(out, inner.discount_factor);
        }
        TuneControllerSpec::AiqiWarmstartExactJh(inner) => {
            out.push(tune_controller_kind_tag(
                TuneControllerKind::AiqiWarmstartExactJh,
            ));
            encode_interface_spec(&inner.interface, out);
            push_u64(out, inner.planner_simulations_per_step as u64);
            push_u64(out, inner.return_horizon as u64);
            push_string(out, &inner.warmstart_teacher_dataset_asset);
            push_u64(out, inner.label_phase_period as u64);
        }
    }
}

fn decode_tune_controller(cursor: &mut Cursor<'_>) -> SpecResult<TuneControllerSpec> {
    match decode_tune_controller_kind(cursor.read_u8()?)? {
        TuneControllerKind::AnnealedHillClimbing => Ok(TuneControllerSpec::AnnealedHillClimbing(
            AnnealedHillClimbingTuneControllerSpec {
                max_mutation_radius: cursor.read_u64()? as usize,
            },
        )),
        TuneControllerKind::McAixiFacCtw => Ok(TuneControllerSpec::McAixiFacCtw(
            McAixiFacCtwTuneControllerSpec {
                interface: decode_interface_spec(cursor)?,
                planner_simulations_per_step: cursor.read_u64()? as usize,
            },
        )),
        TuneControllerKind::AiqiDiscounted => Ok(TuneControllerSpec::AiqiDiscounted(
            AiqiDiscountedTuneControllerSpec {
                interface: decode_interface_spec(cursor)?,
                planner_simulations_per_step: cursor.read_u64()? as usize,
                return_horizon: cursor.read_u64()? as usize,
                return_bins: cursor.read_u64()? as usize,
                discount_factor: cursor.read_f64()?,
            },
        )),
        TuneControllerKind::AiqiWarmstartExactJh => Ok(TuneControllerSpec::AiqiWarmstartExactJh(
            WarmStartExactJhTuneControllerSpec {
                interface: decode_interface_spec(cursor)?,
                planner_simulations_per_step: cursor.read_u64()? as usize,
                return_horizon: cursor.read_u64()? as usize,
                warmstart_teacher_dataset_asset: cursor.read_string()?,
                label_phase_period: cursor.read_u64()? as usize,
            },
        )),
    }
}

#[cfg(feature = "vm")]
fn encode_vm_reward_policy(policy: &VmRewardPolicySpec, out: &mut Vec<u8>) {
    match policy {
        VmRewardPolicySpec::FromGuest => out.push(0),
        VmRewardPolicySpec::Pattern {
            pattern,
            base_reward,
            bonus_reward,
        } => {
            out.push(1);
            push_string(out, pattern);
            push_i64(out, *base_reward);
            push_i64(out, *bonus_reward);
        }
    }
}

#[cfg(feature = "vm")]
fn decode_vm_reward_policy(cursor: &mut Cursor<'_>) -> SpecResult<VmRewardPolicySpec> {
    match cursor.read_u8()? {
        0 => Ok(VmRewardPolicySpec::FromGuest),
        1 => Ok(VmRewardPolicySpec::Pattern {
            pattern: cursor.read_string()?,
            base_reward: cursor.read_i64()?,
            bonus_reward: cursor.read_i64()?,
        }),
        tag => Err(SpecError::new(format!(
            "unknown vm reward policy tag '{tag}'"
        ))),
    }
}

#[cfg(feature = "vm")]
fn encode_vm_reward_shaping(spec: &VmRewardShapingSpec, out: &mut Vec<u8>) {
    match spec {
        VmRewardShapingSpec::EntropyReduction {
            baseline_asset,
            max_order,
            scale,
            crash_bonus,
            timeout_bonus,
        } => {
            out.push(0);
            push_string(out, baseline_asset);
            push_i64(out, *max_order);
            push_f64(out, *scale);
            push_option_i64(out, *crash_bonus);
            push_option_i64(out, *timeout_bonus);
        }
        VmRewardShapingSpec::TraceEntropy {
            max_order,
            scale,
            normalize,
        } => {
            out.push(1);
            push_i64(out, *max_order);
            push_f64(out, *scale);
            push_bool(out, *normalize);
        }
    }
}

#[cfg(feature = "vm")]
fn decode_vm_reward_shaping(cursor: &mut Cursor<'_>) -> SpecResult<VmRewardShapingSpec> {
    match cursor.read_u8()? {
        0 => Ok(VmRewardShapingSpec::EntropyReduction {
            baseline_asset: cursor.read_string()?,
            max_order: cursor.read_i64()?,
            scale: cursor.read_f64()?,
            crash_bonus: cursor.read_option_i64()?,
            timeout_bonus: cursor.read_option_i64()?,
        }),
        1 => Ok(VmRewardShapingSpec::TraceEntropy {
            max_order: cursor.read_i64()?,
            scale: cursor.read_f64()?,
            normalize: cursor.read_bool()?,
        }),
        tag => Err(SpecError::new(format!(
            "unknown vm reward shaping tag '{tag}'"
        ))),
    }
}

#[cfg(feature = "vm")]
fn encode_vm_action_source(spec: &VmRuntimeActionSourceSpec, out: &mut Vec<u8>) {
    match spec {
        VmRuntimeActionSourceSpec::Literal {
            names,
            payloads,
            encoding,
        } => {
            out.push(0);
            push_string(out, encoding);
            push_u64(out, payloads.len() as u64);
            for (idx, payload) in payloads.iter().enumerate() {
                push_option_string(out, names.get(idx).cloned().flatten().as_deref());
                push_string(out, payload);
            }
        }
        VmRuntimeActionSourceSpec::Fuzz {
            seeds,
            encoding,
            mutators,
            min_len,
            max_len,
            dictionary,
            rng_seed,
        } => {
            out.push(1);
            push_string(out, encoding);
            push_string_list(out, seeds);
            push_string_list(out, mutators);
            push_u64(out, *min_len as u64);
            push_u64(out, *max_len as u64);
            push_string_list(out, dictionary);
            push_u64(out, *rng_seed);
        }
    }
}

#[cfg(feature = "vm")]
fn decode_vm_action_source(cursor: &mut Cursor<'_>) -> SpecResult<VmRuntimeActionSourceSpec> {
    match cursor.read_u8()? {
        0 => {
            let encoding = cursor.read_string()?;
            let len = cursor.read_u64()? as usize;
            let mut names = Vec::with_capacity(len);
            let mut payloads = Vec::with_capacity(len);
            for _ in 0..len {
                names.push(cursor.read_option_string()?);
                payloads.push(cursor.read_string()?);
            }
            Ok(VmRuntimeActionSourceSpec::Literal {
                names,
                payloads,
                encoding,
            })
        }
        1 => Ok(VmRuntimeActionSourceSpec::Fuzz {
            seeds: cursor.read_string_list()?,
            encoding: cursor.read_string()?,
            mutators: cursor.read_string_list()?,
            min_len: cursor.read_u64()? as usize,
            max_len: cursor.read_u64()? as usize,
            dictionary: cursor.read_string_list()?,
            rng_seed: cursor.read_u64()?,
        }),
        tag => Err(SpecError::new(format!(
            "unknown vm action source tag '{tag}'"
        ))),
    }
}

#[cfg(feature = "vm")]
fn encode_vm_action_filter(spec: &VmActionFilterSpec, out: &mut Vec<u8>) {
    push_option_f64(out, spec.min_entropy);
    push_option_f64(out, spec.max_entropy);
    push_option_f64(out, spec.min_intrinsic_dependence);
    push_option_f64(out, spec.min_novelty);
    push_option_string(out, spec.novelty_prior_asset.as_deref());
    push_i64(out, spec.max_order);
    push_option_i64(out, spec.reject_reward);
}

#[cfg(feature = "vm")]
fn decode_vm_action_filter(cursor: &mut Cursor<'_>) -> SpecResult<VmActionFilterSpec> {
    Ok(VmActionFilterSpec {
        min_entropy: cursor.read_option_f64()?,
        max_entropy: cursor.read_option_f64()?,
        min_intrinsic_dependence: cursor.read_option_f64()?,
        min_novelty: cursor.read_option_f64()?,
        novelty_prior_asset: cursor.read_option_string()?,
        max_order: cursor.read_i64()?,
        reject_reward: cursor.read_option_i64()?,
    })
}

#[cfg(feature = "vm")]
fn encode_vm_trace(spec: &VmTraceSpec, out: &mut Vec<u8>) {
    push_option_string(out, spec.shared_region_name.as_deref());
    push_u64(out, spec.max_bytes as u64);
    push_bool(out, spec.reset_on_episode);
}

#[cfg(feature = "vm")]
fn decode_vm_trace(cursor: &mut Cursor<'_>) -> SpecResult<VmTraceSpec> {
    Ok(VmTraceSpec {
        shared_region_name: cursor.read_option_string()?,
        max_bytes: cursor.read_u64()? as usize,
        reset_on_episode: cursor.read_bool()?,
    })
}

fn string_list(value: &serde_json::Value) -> SpecResult<Vec<String>> {
    let Some(items) = value.as_array() else {
        return Ok(Vec::new());
    };
    items
        .iter()
        .map(|item| {
            item.as_str()
                .map(|text| text.to_string())
                .ok_or_else(|| SpecError::new("expected string list item"))
        })
        .collect()
}

fn pair_list(value: &serde_json::Value) -> SpecResult<Vec<(String, String)>> {
    let Some(items) = value.as_array() else {
        return Ok(Vec::new());
    };
    let mut pairs = Vec::with_capacity(items.len());
    for item in items {
        let Some(pair) = item.as_array() else {
            return Err(SpecError::new(
                "forbidden_expert_pairs entries must be arrays",
            ));
        };
        if pair.len() != 2 {
            return Err(SpecError::new(
                "forbidden_expert_pairs entries must have length 2",
            ));
        }
        pairs.push((
            required_string(&pair[0], "forbidden_expert_pairs[][0]")?,
            required_string(&pair[1], "forbidden_expert_pairs[][1]")?,
        ));
    }
    Ok(pairs)
}

fn optional_string(value: &serde_json::Value) -> Option<String> {
    value.as_str().map(|text| text.to_string())
}

fn required_string(value: &serde_json::Value, label: &str) -> SpecResult<String> {
    value
        .as_str()
        .map(|text| text.to_string())
        .ok_or_else(|| SpecError::new(format!("{label} is required")))
}

fn required_f64(value: &serde_json::Value, label: &str) -> SpecResult<f64> {
    value
        .as_f64()
        .ok_or_else(|| SpecError::new(format!("{label} is required")))
}

fn required_u64(value: &serde_json::Value, label: &str) -> SpecResult<u64> {
    value
        .as_u64()
        .ok_or_else(|| SpecError::new(format!("{label} is required")))
}

fn required_i64(value: &serde_json::Value, label: &str) -> SpecResult<i64> {
    value
        .as_i64()
        .ok_or_else(|| SpecError::new(format!("{label} is required")))
}

fn parse_builtin_environment(name: &str) -> SpecResult<BuiltinEnvironmentSpec> {
    match name {
        "coin_flip" | "coin-flip" => Ok(BuiltinEnvironmentSpec::CoinFlip),
        "ctw_test" | "ctw-test" => Ok(BuiltinEnvironmentSpec::CtwTest),
        "extended_tiger" | "extended-tiger" => Ok(BuiltinEnvironmentSpec::ExtendedTiger),
        "tic_tac_toe" | "tictactoe" => Ok(BuiltinEnvironmentSpec::TicTacToe),
        "biased_rock_paper_scissor" | "biased-rock-paper-scissor" => {
            Ok(BuiltinEnvironmentSpec::BiasedRockPaperScissor)
        }
        "kuhn_poker" | "kuhn-poker" => Ok(BuiltinEnvironmentSpec::KuhnPoker),
        other => Err(SpecError::new(format!(
            "unknown builtin environment '{other}'"
        ))),
    }
}

fn builtin_environment_name(env: BuiltinEnvironmentSpec) -> &'static str {
    match env {
        BuiltinEnvironmentSpec::CoinFlip => "coin_flip",
        BuiltinEnvironmentSpec::CtwTest => "ctw_test",
        BuiltinEnvironmentSpec::ExtendedTiger => "extended_tiger",
        BuiltinEnvironmentSpec::TicTacToe => "tic_tac_toe",
        BuiltinEnvironmentSpec::BiasedRockPaperScissor => "biased_rock_paper_scissor",
        BuiltinEnvironmentSpec::KuhnPoker => "kuhn_poker",
    }
}

fn builtin_environment_tag(env: BuiltinEnvironmentSpec) -> u8 {
    match env {
        BuiltinEnvironmentSpec::CoinFlip => 0,
        BuiltinEnvironmentSpec::CtwTest => 1,
        BuiltinEnvironmentSpec::ExtendedTiger => 2,
        BuiltinEnvironmentSpec::TicTacToe => 3,
        BuiltinEnvironmentSpec::BiasedRockPaperScissor => 4,
        BuiltinEnvironmentSpec::KuhnPoker => 5,
    }
}

fn decode_builtin_environment(tag: u8) -> SpecResult<BuiltinEnvironmentSpec> {
    match tag {
        0 => Ok(BuiltinEnvironmentSpec::CoinFlip),
        1 => Ok(BuiltinEnvironmentSpec::CtwTest),
        2 => Ok(BuiltinEnvironmentSpec::ExtendedTiger),
        3 => Ok(BuiltinEnvironmentSpec::TicTacToe),
        4 => Ok(BuiltinEnvironmentSpec::BiasedRockPaperScissor),
        5 => Ok(BuiltinEnvironmentSpec::KuhnPoker),
        _ => Err(SpecError::new(format!(
            "unknown builtin environment tag '{tag}'"
        ))),
    }
}

#[cfg(feature = "vm")]
fn parse_shared_memory_policy(name: &str) -> SpecResult<SharedMemoryPolicySpec> {
    match name {
        "snapshot" => Ok(SharedMemoryPolicySpec::Snapshot),
        "preserve" => Ok(SharedMemoryPolicySpec::Preserve),
        other => Err(SpecError::new(format!(
            "unknown shared memory policy '{other}'"
        ))),
    }
}

#[cfg(feature = "vm")]
fn shared_memory_policy_name(policy: SharedMemoryPolicySpec) -> &'static str {
    match policy {
        SharedMemoryPolicySpec::Preserve => "preserve",
        SharedMemoryPolicySpec::Snapshot => "snapshot",
    }
}

#[cfg(feature = "vm")]
fn shared_memory_policy_tag(policy: SharedMemoryPolicySpec) -> u8 {
    match policy {
        SharedMemoryPolicySpec::Preserve => 0,
        SharedMemoryPolicySpec::Snapshot => 1,
    }
}

#[cfg(feature = "vm")]
fn decode_shared_memory_policy(tag: u8) -> SpecResult<SharedMemoryPolicySpec> {
    match tag {
        0 => Ok(SharedMemoryPolicySpec::Preserve),
        1 => Ok(SharedMemoryPolicySpec::Snapshot),
        _ => Err(SpecError::new(format!(
            "unknown shared memory policy tag '{tag}'"
        ))),
    }
}

fn parse_observation_key_mode(name: &str) -> SpecResult<ObservationKeyMode> {
    match name {
        "first" => Ok(ObservationKeyMode::First),
        "last" => Ok(ObservationKeyMode::Last),
        "stream_hash" | "stream-hash" => Ok(ObservationKeyMode::StreamHash),
        "full_stream" | "full-stream" | "full" => Ok(ObservationKeyMode::FullStream),
        other => Err(SpecError::new(format!(
            "unknown observation key mode '{other}'"
        ))),
    }
}

fn observation_key_mode_name(mode: ObservationKeyMode) -> &'static str {
    match mode {
        ObservationKeyMode::First => "first",
        ObservationKeyMode::Last => "last",
        ObservationKeyMode::StreamHash => "stream_hash",
        ObservationKeyMode::FullStream => "full_stream",
    }
}

fn observation_key_mode_tag(mode: ObservationKeyMode) -> u8 {
    match mode {
        ObservationKeyMode::First => 0,
        ObservationKeyMode::Last => 1,
        ObservationKeyMode::StreamHash => 2,
        ObservationKeyMode::FullStream => 3,
    }
}

fn decode_observation_key_mode(tag: u8) -> SpecResult<ObservationKeyMode> {
    match tag {
        0 => Ok(ObservationKeyMode::First),
        1 => Ok(ObservationKeyMode::Last),
        2 => Ok(ObservationKeyMode::StreamHash),
        3 => Ok(ObservationKeyMode::FullStream),
        _ => Err(SpecError::new(format!(
            "unknown observation key mode tag '{tag}'"
        ))),
    }
}

fn tune_controller_kind_tag(kind: TuneControllerKind) -> u8 {
    match kind {
        TuneControllerKind::AnnealedHillClimbing => 0,
        TuneControllerKind::McAixiFacCtw => 1,
        TuneControllerKind::AiqiDiscounted => 2,
        TuneControllerKind::AiqiWarmstartExactJh => 3,
    }
}

fn decode_tune_controller_kind(tag: u8) -> SpecResult<TuneControllerKind> {
    match tag {
        0 => Ok(TuneControllerKind::AnnealedHillClimbing),
        1 => Ok(TuneControllerKind::McAixiFacCtw),
        2 => Ok(TuneControllerKind::AiqiDiscounted),
        3 => Ok(TuneControllerKind::AiqiWarmstartExactJh),
        _ => Err(SpecError::new(format!(
            "unknown tune controller tag '{tag}'"
        ))),
    }
}

fn push_bool(out: &mut Vec<u8>, value: bool) {
    out.push(u8::from(value));
}

fn push_u64(out: &mut Vec<u8>, mut value: u64) {
    while value >= 0x80 {
        out.push((value as u8) | 0x80);
        value >>= 7;
    }
    out.push(value as u8);
}

fn push_i64(out: &mut Vec<u8>, value: i64) {
    push_u64(out, ((value << 1) ^ (value >> 63)) as u64);
}

fn push_f64(out: &mut Vec<u8>, value: f64) {
    out.extend_from_slice(&value.to_le_bytes());
}

fn push_string(out: &mut Vec<u8>, value: &str) {
    push_u64(out, value.len() as u64);
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

fn push_option_u64(out: &mut Vec<u8>, value: Option<u64>) {
    match value {
        Some(value) => {
            out.push(1);
            push_u64(out, value);
        }
        None => out.push(0),
    }
}

#[cfg(feature = "vm")]
fn push_option_i64(out: &mut Vec<u8>, value: Option<i64>) {
    match value {
        Some(value) => {
            out.push(1);
            push_i64(out, value);
        }
        None => out.push(0),
    }
}

#[cfg(feature = "vm")]
fn push_option_f64(out: &mut Vec<u8>, value: Option<f64>) {
    match value {
        Some(value) => {
            out.push(1);
            push_f64(out, value);
        }
        None => out.push(0),
    }
}

fn push_string_list(out: &mut Vec<u8>, items: &[String]) {
    push_u64(out, items.len() as u64);
    for item in items {
        push_string(out, item);
    }
}

struct Cursor<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> Cursor<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, pos: 0 }
    }

    fn read_exact(&mut self, len: usize) -> SpecResult<&'a [u8]> {
        if self.pos + len > self.bytes.len() {
            return Err(SpecError::new("unexpected end of spec document"));
        }
        let start = self.pos;
        self.pos += len;
        Ok(&self.bytes[start..self.pos])
    }

    fn read_u8(&mut self) -> SpecResult<u8> {
        Ok(self.read_exact(1)?[0])
    }

    fn read_bool(&mut self) -> SpecResult<bool> {
        Ok(self.read_u8()? != 0)
    }

    fn read_u64(&mut self) -> SpecResult<u64> {
        let mut shift = 0u32;
        let mut out = 0u64;
        loop {
            let byte = self.read_u8()?;
            out |= ((byte & 0x7f) as u64) << shift;
            if byte & 0x80 == 0 {
                return Ok(out);
            }
            shift += 7;
            if shift > 63 {
                return Err(SpecError::new("invalid varint in spec document"));
            }
        }
    }

    fn read_i64(&mut self) -> SpecResult<i64> {
        let value = self.read_u64()?;
        Ok(((value >> 1) as i64) ^ (-((value & 1) as i64)))
    }

    fn read_f64(&mut self) -> SpecResult<f64> {
        let mut bytes = [0u8; 8];
        bytes.copy_from_slice(self.read_exact(8)?);
        Ok(f64::from_le_bytes(bytes))
    }

    fn read_string(&mut self) -> SpecResult<String> {
        let len = self.read_u64()? as usize;
        let bytes = self.read_exact(len)?;
        String::from_utf8(bytes.to_vec()).map_err(|err| SpecError::new(err.to_string()))
    }

    fn read_option_string(&mut self) -> SpecResult<Option<String>> {
        if self.read_u8()? == 1 {
            Ok(Some(self.read_string()?))
        } else {
            Ok(None)
        }
    }

    fn read_option_u64(&mut self) -> SpecResult<Option<u64>> {
        if self.read_u8()? == 1 {
            Ok(Some(self.read_u64()?))
        } else {
            Ok(None)
        }
    }

    #[cfg(feature = "vm")]
    fn read_option_i64(&mut self) -> SpecResult<Option<i64>> {
        if self.read_u8()? == 1 {
            Ok(Some(self.read_i64()?))
        } else {
            Ok(None)
        }
    }

    #[cfg(feature = "vm")]
    fn read_option_f64(&mut self) -> SpecResult<Option<f64>> {
        if self.read_u8()? == 1 {
            Ok(Some(self.read_f64()?))
        } else {
            Ok(None)
        }
    }

    fn read_string_list(&mut self) -> SpecResult<Vec<String>> {
        let len = self.read_u64()? as usize;
        let mut items = Vec::with_capacity(len);
        for _ in 0..len {
            items.push(self.read_string()?);
        }
        Ok(items)
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
mod tests {
    use super::*;

    #[cfg(feature = "backend-ctw")]
    fn sample_planner_run() -> PlannerRunSpec {
        PlannerRunSpec {
            assets: Vec::new(),
            environment: EnvironmentSpec::Builtin {
                builtin: BuiltinEnvironmentSpec::CoinFlip,
            },
            interface: PlannerInterfaceSpec {
                observation_bits: 1,
                observation_stream_len: 1,
                observation_key_mode: ObservationKeyMode::FullStream,
                reward_bits: 1,
                agent_actions: 2,
                min_reward: 0,
                max_reward: 1,
                reward_offset: 0,
            },
            controller: ControllerSpec::AiqiDiscounted(AiqiDiscountedControllerSpec {
                predictor: RateBackend::Ctw { depth: 8 },
                predictor_max_order: 8,
                discount_gamma: 0.99,
                return_horizon: 2,
                return_bins: 8,
                augmentation_period: 2,
                history_prune_keep_steps: None,
                baseline_exploration: 0.01,
            }),
            runtime: PlannerRuntimeSpec {
                random_seed: Some(7),
                learn_cycles: Some(4),
                eval_cycles: Some(2),
                terminate_lifetime: 4,
                log_every: 1,
                perf: false,
                vm_perf_only: false,
                explore_epsilon: 0.0,
                explore_gamma: 1.0,
            },
        }
    }

    #[cfg(feature = "backend-ctw")]
    #[test]
    fn planner_run_json_roundtrip_is_stable() {
        let spec = sample_planner_run();
        let expected = spec.to_canonical_json().expect("json");
        let value = spec.to_canonical_json_value().expect("json value");
        let reparsed = SpecDocument::parse_json_value(&value, Path::new(".")).expect("parse");
        match reparsed {
            SpecDocument::PlannerRun(parsed) => {
                assert_eq!(parsed.to_canonical_json().expect("parsed json"), expected)
            }
            _ => panic!("expected planner run document"),
        }
    }

    #[cfg(feature = "backend-ctw")]
    #[test]
    fn planner_run_binary_roundtrip_is_stable() {
        let spec = sample_planner_run();
        let expected = spec.to_canonical_json().expect("json");
        let bytes = SpecDocument::PlannerRun(spec.clone()).to_binary();
        let reparsed = SpecDocument::from_binary(&bytes, Path::new(".")).expect("binary");
        match reparsed {
            SpecDocument::PlannerRun(parsed) => {
                assert_eq!(parsed.to_canonical_json().expect("parsed json"), expected)
            }
            _ => panic!("expected planner run document"),
        }
    }

    #[cfg(feature = "backend-ctw")]
    #[test]
    fn planner_run_compile_exposes_compiled_predictor_and_action_bits() {
        let compiled = sample_planner_run()
            .compile()
            .expect("compiled planner run");
        assert_eq!(compiled.action_bits(), 1);
        match compiled.controller() {
            CompiledPlannerController::AiqiDiscounted { predictor, .. } => {
                assert!(matches!(
                    predictor.canonical_spec(),
                    RateBackend::Ctw { depth: 8 }
                ));
            }
            _ => panic!("expected compiled aiqi controller"),
        }
    }

    #[cfg(feature = "backend-ctw")]
    #[test]
    fn tune_document_binary_roundtrip_is_stable() {
        let spec = TuneSpec {
            assets: vec![AssetBinding {
                id: "dataset".to_string(),
                path: "input.bin".to_string(),
            }],
            input_asset: "dataset".to_string(),
            baseline_candidate: CompressionBackend::Rate {
                rate_backend: RateBackend::Ctw { depth: 8 },
                coder: crate::coders::CoderType::AC,
                framing: crate::compression::FramingMode::Framed,
            },
            controller: TuneControllerSpec::AnnealedHillClimbing(
                AnnealedHillClimbingTuneControllerSpec {
                    max_mutation_radius: 2,
                },
            ),
            bounds: TuneBoundsSpec {
                allowed_backends: vec!["ctw".to_string()],
                forbidden_backends: vec!["zpaq".to_string()],
                parameter_ranges: vec![TuneParameterRangeSpec {
                    parameter: "mixture.alpha".to_string(),
                    min: 0.1,
                    max: 0.5,
                }],
                max_experts: 4,
                max_mixture_nesting_depth: 2,
                min_experts: Some(1),
                allow_duplicate_experts: Some(false),
                required_experts: vec!["ctw".to_string()],
                forbidden_expert_pairs: vec![],
            },
            eval_time_limit_seconds: 1.0,
            time_budget_seconds: 10.0,
            min_throughput_bytes_per_second: 1024.0,
            max_memory_bytes: 1 << 20,
            output_config_path: "best.json".to_string(),
            seed: 7,
            report_path: Some("report.json".to_string()),
        };
        let expected = spec.to_canonical_json().expect("json");
        let bytes = SpecDocument::Tune(spec.clone()).to_binary();
        let reparsed = SpecDocument::from_binary(&bytes, Path::new(".")).expect("binary");
        match reparsed {
            SpecDocument::Tune(parsed) => {
                assert_eq!(parsed.to_canonical_json().expect("parsed json"), expected)
            }
            _ => panic!("expected tune document"),
        }
    }

    #[cfg(feature = "backend-ctw")]
    #[test]
    fn tune_compile_model_bytes_ignore_outer_request_controls() {
        let base = TuneSpec {
            assets: vec![AssetBinding {
                id: "dataset".to_string(),
                path: "input.bin".to_string(),
            }],
            input_asset: "dataset".to_string(),
            baseline_candidate: CompressionBackend::Rate {
                rate_backend: RateBackend::Ctw { depth: 8 },
                coder: crate::coders::CoderType::AC,
                framing: crate::compression::FramingMode::Framed,
            },
            controller: TuneControllerSpec::AnnealedHillClimbing(
                AnnealedHillClimbingTuneControllerSpec {
                    max_mutation_radius: 2,
                },
            ),
            bounds: TuneBoundsSpec {
                allowed_backends: vec!["ctw".to_string()],
                forbidden_backends: vec![],
                parameter_ranges: vec![],
                max_experts: 4,
                max_mixture_nesting_depth: 2,
                min_experts: Some(1),
                allow_duplicate_experts: Some(false),
                required_experts: vec![],
                forbidden_expert_pairs: vec![],
            },
            eval_time_limit_seconds: 1.0,
            time_budget_seconds: 10.0,
            min_throughput_bytes_per_second: 1024.0,
            max_memory_bytes: 1 << 20,
            output_config_path: "best-a.json".to_string(),
            seed: 7,
            report_path: Some("report-a.json".to_string()),
        };
        let mut other = base.clone();
        other.output_config_path = "best-b.json".to_string();
        other.report_path = Some("report-b.json".to_string());

        let compiled_a = base.compile().expect("compiled tune a");
        let compiled_b = other.compile().expect("compiled tune b");
        assert_eq!(
            compiled_a.baseline_candidate().canonical_bytes().as_slice(),
            compiled_b.baseline_candidate().canonical_bytes().as_slice()
        );
        assert_eq!(
            compiled_a.baseline_candidate_model_bytes(),
            compiled_b.baseline_candidate_model_bytes()
        );
    }

    #[cfg(not(feature = "backend-ctw"))]
    #[test]
    fn planner_run_validation_reports_missing_backend_feature() {
        let spec = PlannerRunSpec {
            assets: Vec::new(),
            environment: EnvironmentSpec::Builtin {
                builtin: BuiltinEnvironmentSpec::CoinFlip,
            },
            interface: PlannerInterfaceSpec {
                observation_bits: 1,
                observation_stream_len: 1,
                observation_key_mode: ObservationKeyMode::FullStream,
                reward_bits: 1,
                agent_actions: 2,
                min_reward: 0,
                max_reward: 1,
                reward_offset: 0,
            },
            controller: ControllerSpec::AiqiDiscounted(AiqiDiscountedControllerSpec {
                predictor: RateBackend::Ctw { depth: 8 },
                predictor_max_order: 8,
                discount_gamma: 0.99,
                return_horizon: 2,
                return_bins: 8,
                augmentation_period: 2,
                history_prune_keep_steps: None,
                baseline_exploration: 0.01,
            }),
            runtime: PlannerRuntimeSpec {
                random_seed: Some(7),
                learn_cycles: Some(4),
                eval_cycles: Some(2),
                terminate_lifetime: 4,
                log_every: 1,
                perf: false,
                vm_perf_only: false,
                explore_epsilon: 0.0,
                explore_gamma: 1.0,
            },
        };
        let err = spec.validate().expect_err("missing feature should fail");
        assert!(
            err.to_string()
                .contains("requires infotheory feature 'backend-ctw'"),
            "{err}"
        );
    }
}
