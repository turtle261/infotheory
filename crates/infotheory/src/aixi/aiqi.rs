//! AIQI implementation from "A Model-Free Universal AI".
//!
//! This module implements a model-free universal agent that predicts
//! discretized H-step returns directly from augmented interaction history.
//! The implementation follows the phase-indexed periodic augmentation in
//! "A Model-Free Universal AI":
//! for return horizon `H` and period `N >= H`, each phase model only inserts
//! returns at indices `i % N == phase`.

use crate::aixi::common::{
    Action, ActionAlphabet, PerceptVal, RandomGenerator, Reward, RewardEncodingError,
    bits_for_cardinality, byte_packed_percept_bits, nonnegative_reward_encoding_bounds,
    resolve_random_seed, validate_aiqi_byte_packed_alignment, validate_reward_encoding_bounds,
};
use crate::aixi::model::{
    Predictor, PredictorBuildError, build_aiqi_predictor, default_aixi_bit_stream_semantics,
};
use crate::aixi::planner_spec::{PlannerInterfaceConfig, build_default_planner_run_spec};
use crate::aixi::return_law::{
    ReturnLabelCodec, ReturnLawEvaluator, ReturnPrefixUpdate, predict_expected_label,
};
use crate::api::{BitStreamSemantics, RateBackend, validate_rate_backend};
use crate::spec::{
    AiqiDiscountedControllerSpec, CompiledPlannerController, CompiledPlannerRunSpec,
    ControllerSpec, PlannerRunSpec, SpecError,
};
use std::error::Error;
use std::fmt;

/// Error returned by AIQI configuration validation, construction, and transition ingestion.
#[derive(Debug)]
#[non_exhaustive]
pub enum AiqiError {
    /// `return_horizon` was zero.
    ReturnHorizonZero,
    /// `return_bins` was zero.
    ReturnBinsZero,
    /// The augmentation period was smaller than the return horizon.
    AugmentationPeriodTooShort {
        /// The configured augmentation period.
        augmentation_period: usize,
        /// The configured return horizon.
        return_horizon: usize,
    },
    /// The discount factor was outside `(0, 1)`.
    InvalidDiscountGamma {
        /// The invalid discount factor value.
        value: f64,
    },
    /// The baseline exploration probability was outside `(0, 1]`.
    InvalidBaselineExploration {
        /// The invalid baseline exploration value.
        value: f64,
    },
    /// The configured reward range is not representable.
    RewardEncoding(RewardEncodingError),
    /// The configured rate backend failed validation.
    InvalidRateBackend(crate::error::InfotheoryError),
    /// The configured rate backend violates AIQI runtime requirements.
    UnsupportedRateBackend {
        /// Human-readable explanation of why the backend is unsupported.
        reason: &'static str,
    },
    /// Planner-run spec compilation failed.
    Spec(SpecError),
    /// The compiled planner-run controller kind was not discounted AIQI.
    ControllerKindMismatch,
    /// Predictor construction failed.
    Predictor(PredictorBuildError),
    /// An observed action was outside the configured action alphabet.
    ActionOutOfRange {
        /// The out-of-range action token.
        action: Action,
        /// The configured action alphabet cardinality.
        agent_actions: ActionAlphabet,
    },
    /// The observation stream length did not match the configured interface.
    ObservationStreamLengthMismatch {
        /// Expected observation stream length.
        expected: usize,
        /// Actual observation stream length.
        actual: usize,
    },
    /// An observed reward was outside the configured reward range.
    RewardOutOfRange {
        /// Out-of-range reward value.
        reward: Reward,
        /// Minimum configured reward.
        min_reward: Reward,
        /// Maximum configured reward.
        max_reward: Reward,
    },
    /// An observation value exceeded the configured observation bit width.
    ObservationValueOutOfRange {
        /// Out-of-range observation value.
        observation: PerceptVal,
        /// Configured observation bit width.
        observation_bits: usize,
        /// Maximum representable observation value for `observation_bits`.
        maximum: PerceptVal,
    },
    /// A shifted observed reward became negative.
    NegativeEncodedReward {
        /// Original reward value.
        reward: Reward,
        /// Configured reward offset.
        reward_offset: Reward,
    },
    /// A shifted observed reward exceeded the configured reward bit capacity.
    EncodedRewardTooLarge {
        /// Shifted reward value after applying offset.
        shifted_reward: i128,
        /// Configured reward bit width.
        reward_bits: usize,
        /// Maximum representable encoded reward for `reward_bits`.
        maximum_encoded: u128,
    },
    /// A requested global step is no longer present in retained history.
    HistoryIndexOutOfRange {
        /// Requested global step index.
        global_step: usize,
        /// First retained global step index.
        history_base_step: usize,
        /// Last observed global step index.
        total_steps_observed: usize,
    },
    /// A phase-model update required a return bin that has not been computed.
    MissingReturnBin {
        /// Global step whose return bin was missing.
        step: usize,
        /// Augmentation phase that required the return bin.
        phase: usize,
    },
}

impl fmt::Display for AiqiError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ReturnHorizonZero => f.write_str("return_horizon must be >= 1"),
            Self::ReturnBinsZero => f.write_str("return_bins must be >= 1"),
            Self::AugmentationPeriodTooShort {
                augmentation_period,
                return_horizon,
            } => write!(
                f,
                "augmentation_period must be >= return_horizon (got N={augmentation_period}, H={return_horizon})"
            ),
            Self::InvalidDiscountGamma { value } => write!(
                f,
                "discount_gamma must be in (0, 1) for AIQI as defined in \"A Model-Free Universal AI\", got {value}"
            ),
            Self::InvalidBaselineExploration { value } => write!(
                f,
                "baseline_exploration (tau) must be in (0, 1] for AIQI as defined in \"A Model-Free Universal AI\", got {value}"
            ),
            Self::RewardEncoding(err) => write!(f, "{err}"),
            Self::InvalidRateBackend(err) => write!(f, "invalid rate_backend: {err}"),
            Self::UnsupportedRateBackend { reason } => f.write_str(reason),
            Self::Spec(err) => write!(f, "{err}"),
            Self::ControllerKindMismatch => {
                f.write_str("compiled planner run does not contain a discounted AIQI controller")
            }
            Self::Predictor(err) => write!(f, "{err}"),
            Self::ActionOutOfRange {
                action,
                agent_actions,
            } => write!(
                f,
                "action out of range: action={action} but agent_actions={agent_actions}"
            ),
            Self::ObservationStreamLengthMismatch { expected, actual } => write!(
                f,
                "observation stream length mismatch: expected {expected}, got {actual}"
            ),
            Self::RewardOutOfRange {
                reward,
                min_reward,
                max_reward,
            } => write!(
                f,
                "reward out of configured range: reward={reward} not in [{min_reward}, {max_reward}]"
            ),
            Self::ObservationValueOutOfRange {
                observation,
                observation_bits,
                maximum,
            } => write!(
                f,
                "observation value {observation} does not fit observation_bits={observation_bits} (max={maximum})"
            ),
            Self::NegativeEncodedReward {
                reward,
                reward_offset,
            } => write!(
                f,
                "encoded reward became negative after offset: reward={reward} offset={reward_offset}"
            ),
            Self::EncodedRewardTooLarge {
                shifted_reward,
                reward_bits,
                maximum_encoded,
            } => write!(
                f,
                "encoded reward {shifted_reward} exceeds reward_bits={reward_bits} capacity {maximum_encoded}"
            ),
            Self::HistoryIndexOutOfRange {
                global_step,
                history_base_step,
                total_steps_observed,
            } => write!(
                f,
                "global step {global_step} out of retained history range [{history_base_step}, {total_steps_observed}]"
            ),
            Self::MissingReturnBin { step, phase } => write!(
                f,
                "missing return bin for step {step} in phase {phase} while pushing augmented history"
            ),
        }
    }
}

impl Error for AiqiError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::RewardEncoding(err) => Some(err),
            Self::InvalidRateBackend(err) => Some(err),
            Self::Spec(err) => Some(err),
            Self::Predictor(err) => Some(err),
            _ => None,
        }
    }
}

impl From<RewardEncodingError> for AiqiError {
    fn from(value: RewardEncodingError) -> Self {
        Self::RewardEncoding(value)
    }
}

impl From<SpecError> for AiqiError {
    fn from(value: SpecError) -> Self {
        Self::Spec(value)
    }
}

/// Configuration parameters for an AIQI agent.
#[derive(Clone)]
#[non_exhaustive]
pub struct AiqiConfig {
    /// Predictive backend.
    pub rate_backend: RateBackend,
    /// Bit-stream semantics used to adapt generic rate backends to AIQI symbols.
    pub bit_stream_semantics: BitStreamSemantics,
    /// Number of bits used to encode observations.
    pub observation_bits: usize,
    /// Number of observation symbols per environment step.
    pub observation_stream_len: usize,
    /// Number of bits used to encode rewards.
    pub reward_bits: usize,
    /// Number of valid actions.
    pub agent_actions: ActionAlphabet,
    /// Minimum possible environment reward.
    pub min_reward: Reward,
    /// Maximum possible environment reward.
    pub max_reward: Reward,
    /// Offset applied before encoding reward bits.
    pub reward_offset: Reward,
    /// Discount factor used when constructing H-step returns.
    pub discount_gamma: f64,
    /// Return horizon `H`.
    pub return_horizon: usize,
    /// Number of discretization bins `M` for returns.
    ///
    /// Non-power-of-two alphabets are represented by fixed-width binary labels
    /// with invalid leaves excluded from the exact return law.
    pub return_bins: usize,
    /// Augmentation period `N` (must satisfy `N >= H`).
    pub augmentation_period: usize,
    /// Optional history retention knob for bounded memory growth.
    ///
    /// - `None`: keep full history (default behavior, no pruning).
    /// - `Some(k)`: keep at least the most recent `k` steps, while also
    ///   preserving all steps still required for exact return construction and
    ///   deferred phase-model advancement.
    pub history_prune_keep_steps: Option<usize>,
    /// Baseline epsilon-greedy exploration probability `tau`.
    pub baseline_exploration: f64,
    /// Optional deterministic RNG seed for action selection/exploration.
    ///
    /// When `None`, planner runtime canonicalizes this to seed `0`.
    pub random_seed: Option<u64>,
}

impl Default for AiqiConfig {
    fn default() -> Self {
        Self {
            rate_backend: RateBackend::Ctw { depth: 8 },
            bit_stream_semantics: default_aixi_bit_stream_semantics(),
            observation_bits: 1,
            observation_stream_len: 1,
            reward_bits: 1,
            agent_actions: ActionAlphabet::try_from_usize(2)
                .expect("default action alphabet must be non-zero"),
            min_reward: 0,
            max_reward: 1,
            reward_offset: 0,
            discount_gamma: 0.99,
            return_horizon: 4,
            return_bins: 8,
            augmentation_period: 4,
            history_prune_keep_steps: None,
            baseline_exploration: 0.01,
            random_seed: None,
        }
    }
}

impl AiqiConfig {
    fn canonical_predictor_backend(&self) -> RateBackend {
        self.rate_backend.clone()
    }

    fn canonical_planner_run_spec(&self) -> PlannerRunSpec {
        let predictor = self.canonical_predictor_backend();
        build_default_planner_run_spec(
            PlannerInterfaceConfig {
                observation_bits: self.observation_bits,
                observation_stream_len: self.observation_stream_len,
                observation_key_mode: crate::aixi::common::ObservationKeyMode::FullStream,
                reward_bits: self.reward_bits,
                agent_actions: self.agent_actions,
            },
            ControllerSpec::AiqiDiscounted(AiqiDiscountedControllerSpec {
                predictor,
                bit_stream_semantics: self.bit_stream_semantics,
                discount_gamma: self.discount_gamma,
                return_horizon: self.return_horizon,
                return_bins: self.return_bins,
                augmentation_period: self.augmentation_period,
                history_prune_keep_steps: self.history_prune_keep_steps,
                baseline_exploration: self.baseline_exploration,
            }),
            self.random_seed,
        )
    }

    fn compile_planner_run_spec(&self) -> Result<CompiledPlannerRunSpec, AiqiError> {
        self.canonical_planner_run_spec()
            .compile()
            .map_err(AiqiError::from)
    }

    fn validate_runtime_invariants(&self) -> Result<(), AiqiError> {
        if self.return_horizon == 0 {
            return Err(AiqiError::ReturnHorizonZero);
        }
        if self.return_bins == 0 {
            return Err(AiqiError::ReturnBinsZero);
        }
        if self.augmentation_period < self.return_horizon {
            return Err(AiqiError::AugmentationPeriodTooShort {
                augmentation_period: self.augmentation_period,
                return_horizon: self.return_horizon,
            });
        }
        if !(0.0 < self.discount_gamma && self.discount_gamma < 1.0) {
            return Err(AiqiError::InvalidDiscountGamma {
                value: self.discount_gamma,
            });
        }
        if !(0.0 < self.baseline_exploration && self.baseline_exploration <= 1.0) {
            return Err(AiqiError::InvalidBaselineExploration {
                value: self.baseline_exploration,
            });
        }
        validate_reward_encoding_bounds(
            self.min_reward,
            self.max_reward,
            self.reward_offset,
            self.reward_bits,
        )?;
        if matches!(
            self.bit_stream_semantics,
            BitStreamSemantics::BytePacked { .. }
        ) {
            let action_bits = self.agent_actions.action_bits();
            let percept_bits = byte_packed_percept_bits(
                self.observation_bits,
                self.observation_stream_len,
                self.reward_bits,
            );
            let return_bits = bits_for_cardinality(self.return_bins);
            validate_aiqi_byte_packed_alignment(action_bits, percept_bits, return_bits)
                .map_err(|reason| AiqiError::UnsupportedRateBackend { reason })?;
        }

        validate_rate_backend(&self.rate_backend).map_err(AiqiError::InvalidRateBackend)?;
        if !rate_backend_supports_aiqi_frozen_conditioning(&self.rate_backend) {
            return Err(AiqiError::UnsupportedRateBackend {
                reason: "AIQI strict mode requires frozen context updates; configured rate_backend contains zpaq which does not provide strict frozen conditioning",
            });
        }
        Ok(())
    }

    /// Validate configuration constraints.
    pub fn validate(&self) -> Result<(), AiqiError> {
        self.validate_runtime_invariants()?;
        self.compile_planner_run_spec().map(|_| ())
    }

    /// Test-only accessor for the private `canonical_predictor_backend` mapping.
    ///
    /// Used by cross-config alias-symmetry tests in [`crate::aixi::agent`].
    #[doc(hidden)]
    #[cfg(test)]
    pub(crate) fn canonical_predictor_backend_for_test(&self) -> RateBackend {
        self.canonical_predictor_backend()
    }
}

#[derive(Clone)]
struct AiqiRuntimeConfig {
    observation_bits: usize,
    observation_stream_len: usize,
    reward_bits: usize,
    agent_actions: ActionAlphabet,
    min_reward: Reward,
    max_reward: Reward,
    reward_offset: Reward,
    discount_gamma: f64,
    return_horizon: usize,
    return_bins: usize,
    augmentation_period: usize,
    history_prune_keep_steps: Option<usize>,
    baseline_exploration: f64,
    random_seed: u64,
}

impl AiqiRuntimeConfig {
    fn from_config(config: &AiqiConfig) -> Self {
        Self {
            observation_bits: config.observation_bits,
            observation_stream_len: config.observation_stream_len.max(1),
            reward_bits: config.reward_bits,
            agent_actions: config.agent_actions,
            min_reward: config.min_reward,
            max_reward: config.max_reward,
            reward_offset: config.reward_offset,
            discount_gamma: config.discount_gamma,
            return_horizon: config.return_horizon,
            return_bins: config.return_bins,
            augmentation_period: config.augmentation_period,
            history_prune_keep_steps: config.history_prune_keep_steps,
            baseline_exploration: config.baseline_exploration,
            random_seed: resolve_random_seed(config.random_seed),
        }
    }

    fn from_compiled(compiled: &CompiledPlannerRunSpec) -> Result<Self, AiqiError> {
        let interface = compiled.interface();
        let runtime = compiled.runtime();
        let (
            discount_gamma,
            return_horizon,
            return_bins,
            augmentation_period,
            history_prune_keep_steps,
            baseline_exploration,
        ) = match compiled.controller() {
            CompiledPlannerController::AiqiDiscounted {
                discount_gamma,
                return_horizon,
                return_bins,
                augmentation_period,
                history_prune_keep_steps,
                baseline_exploration,
                ..
            } => (
                *discount_gamma,
                *return_horizon,
                *return_bins,
                *augmentation_period,
                *history_prune_keep_steps,
                *baseline_exploration,
            ),
            _ => return Err(AiqiError::ControllerKindMismatch),
        };

        let (min_reward, max_reward, reward_offset) =
            nonnegative_reward_encoding_bounds(interface.reward_bits);
        Ok(Self {
            observation_bits: interface.observation_bits,
            observation_stream_len: interface.observation_stream_len.max(1),
            reward_bits: interface.reward_bits,
            agent_actions: interface.agent_actions,
            min_reward,
            max_reward,
            reward_offset,
            discount_gamma,
            return_horizon,
            return_bins,
            augmentation_period,
            history_prune_keep_steps,
            baseline_exploration,
            random_seed: resolve_random_seed(runtime.random_seed),
        })
    }
}

#[derive(Clone, Debug)]
struct StepRecord {
    action: Action,
    observations: Vec<PerceptVal>,
    reward: Reward,
}

struct PhaseModel {
    predictor: Box<dyn Predictor>,
    // Largest step index for which this phase model has consumed
    // the augmented stream up to and including that step's percept.
    last_augmented_step: usize,
}

/// AIQI agent with phase-indexed augmented return predictors.
pub struct AiqiAgent {
    config: AiqiRuntimeConfig,
    phases: Vec<PhaseModel>,
    steps: Vec<StepRecord>,
    return_bins_by_step: Vec<Option<u64>>,
    // Global 1-based index of steps[0] / return_bins_by_step[0].
    history_base_step: usize,
    // Total number of transitions observed so far (global 1-based max step index).
    total_steps_observed: usize,
    action_bits: usize,
    return_label_codec: ReturnLabelCodec,
    use_generic_planner: bool,
    rng: RandomGenerator,
}

impl AiqiAgent {
    /// Construct a new AIQI agent.
    pub fn new(config: AiqiConfig) -> Result<Self, AiqiError> {
        config.validate_runtime_invariants()?;
        let compiled = config.compile_planner_run_spec()?;
        let runtime = AiqiRuntimeConfig::from_config(&config);
        Self::from_compiled_config(runtime, &compiled)
    }

    /// Construct a new AIQI agent directly from a compiled planner-run spec.
    pub fn from_compiled_planner_run(compiled: &CompiledPlannerRunSpec) -> Result<Self, AiqiError> {
        let config = AiqiRuntimeConfig::from_compiled(compiled)?;
        Self::from_compiled_config(config, compiled)
    }

    fn from_compiled_config(
        config: AiqiRuntimeConfig,
        compiled: &CompiledPlannerRunSpec,
    ) -> Result<Self, AiqiError> {
        let (predictor, augmentation_period, return_bins, bit_stream_semantics) =
            match compiled.controller() {
                CompiledPlannerController::AiqiDiscounted {
                    predictor,
                    augmentation_period,
                    return_bins,
                    bit_stream_semantics,
                    ..
                } => (
                    predictor,
                    *augmentation_period,
                    *return_bins,
                    *bit_stream_semantics,
                ),
                _ => return Err(AiqiError::ControllerKindMismatch),
            };
        let action_bits = compiled.action_bits();
        let return_label_codec = ReturnLabelCodec::value_monotone(return_bins);
        let return_bits = return_label_codec.bits();
        let uses_native_reversible_binary_predictor = bit_stream_semantics
            == BitStreamSemantics::BinaryTokens
            && predictor.supports_native_bit_prediction()
            && predictor.supports_reversible_bit_updates();
        let use_generic_planner = !uses_native_reversible_binary_predictor;

        let mut phases = Vec::with_capacity(augmentation_period);
        for _ in 0..augmentation_period {
            phases.push(PhaseModel {
                predictor: build_aiqi_predictor(predictor, return_bits, bit_stream_semantics)
                    .map_err(AiqiError::Predictor)?,
                last_augmented_step: 0,
            });
        }

        let rng = RandomGenerator::from_seed(config.random_seed);

        Ok(Self {
            action_bits,
            return_label_codec,
            use_generic_planner,
            config,
            phases,
            steps: Vec::new(),
            return_bins_by_step: Vec::new(),
            history_base_step: 1,
            total_steps_observed: 0,
            rng,
        })
    }

    /// Number of transitions incorporated so far.
    pub fn steps_observed(&self) -> usize {
        self.total_steps_observed
    }

    /// Returns the configured action alphabet cardinality.
    pub fn num_actions(&self) -> ActionAlphabet {
        self.config.agent_actions
    }

    /// Returns the resolved deterministic seed used by this AIQI agent.
    pub fn resolved_random_seed(&self) -> u64 {
        self.config.random_seed
    }

    pub(crate) fn reseed_random(&mut self, seed: u64) {
        self.config.random_seed = seed;
        self.rng = RandomGenerator::from_seed(seed);
    }

    /// Select the next action from the current history.
    pub fn get_planned_action(&mut self) -> Action {
        self.get_planned_action_with_extra_exploration_flag(0.0).0
    }

    /// Select the next action and report whether it was sampled for exploration.
    pub fn get_planned_action_with_extra_exploration_flag(
        &mut self,
        extra_exploration: f64,
    ) -> (Action, bool) {
        let extra = extra_exploration.clamp(0.0, 1.0);
        let tau = self.config.baseline_exploration.clamp(0.0, 1.0);
        let effective = 1.0 - (1.0 - tau) * (1.0 - extra);
        if effective > 0.0 && self.rng.gen_bool(effective) {
            (
                self.rng.gen_range(self.config.agent_actions.get()) as u64,
                true,
            )
        } else {
            let q_values = self.estimate_q_values();
            let greedy_action = argmax_with_fixed_tie_break(&q_values) as u64;
            (greedy_action, false)
        }
    }

    /// Select the next action, adding optional extra exploration.
    ///
    /// The extra exploration probability is combined as
    /// `p = 1 - (1 - tau) * (1 - extra)`, where `tau` is the baseline
    /// exploration in [`AiqiConfig`].
    pub fn get_planned_action_with_extra_exploration(&mut self, extra_exploration: f64) -> Action {
        self.get_planned_action_with_extra_exploration_flag(extra_exploration)
            .0
    }

    /// Record one environment transition `(action, observations, reward)`.
    ///
    /// This appends to history and, when enough future rewards are known,
    /// computes and learns one newly available discretized return.
    pub fn observe_transition(
        &mut self,
        action: Action,
        observations: &[PerceptVal],
        reward: Reward,
    ) -> Result<(), AiqiError> {
        if action as usize >= self.config.agent_actions.get() {
            return Err(AiqiError::ActionOutOfRange {
                action,
                agent_actions: self.config.agent_actions,
            });
        }

        let expected_obs = self.config.observation_stream_len.max(1);
        if observations.len() != expected_obs {
            return Err(AiqiError::ObservationStreamLengthMismatch {
                expected: expected_obs,
                actual: observations.len(),
            });
        }

        if reward < self.config.min_reward || reward > self.config.max_reward {
            return Err(AiqiError::RewardOutOfRange {
                reward,
                min_reward: self.config.min_reward,
                max_reward: self.config.max_reward,
            });
        }

        let obs_max = max_value_for_bits(self.config.observation_bits);
        for &obs in observations {
            if obs > obs_max {
                return Err(AiqiError::ObservationValueOutOfRange {
                    observation: obs,
                    observation_bits: self.config.observation_bits,
                    maximum: obs_max,
                });
            }
        }

        let rew_shifted = (reward as i128) + (self.config.reward_offset as i128);
        if rew_shifted < 0 {
            return Err(AiqiError::NegativeEncodedReward {
                reward,
                reward_offset: self.config.reward_offset,
            });
        }
        if self.config.reward_bits < 64 {
            let max_enc = (1u128 << self.config.reward_bits) - 1;
            if (rew_shifted as u128) > max_enc {
                return Err(AiqiError::EncodedRewardTooLarge {
                    shifted_reward: rew_shifted,
                    reward_bits: self.config.reward_bits,
                    maximum_encoded: max_enc,
                });
            }
        }

        self.steps.push(StepRecord {
            action,
            observations: observations.to_vec(),
            reward,
        });
        self.total_steps_observed += 1;
        self.return_bins_by_step.push(None);

        self.maybe_learn_new_return()?;
        self.maybe_prune_history();
        Ok(())
    }

    fn maybe_learn_new_return(&mut self) -> Result<(), AiqiError> {
        let t = self.total_steps_observed;
        let h = self.config.return_horizon;
        if t < h {
            return Ok(());
        }

        // Newly available return index (1-based): i = t - H + 1.
        let i = t + 1 - h;
        let bin = self.compute_return_bin(i);
        let local_idx = self.local_index(i)?;
        self.return_bins_by_step[local_idx] = Some(bin);

        let phase = i % self.config.augmentation_period;
        self.advance_phase_model_to_step(phase, i)
    }

    fn estimate_q_values(&mut self) -> Vec<f64> {
        if self.use_generic_planner {
            return self.estimate_q_values_generic();
        }

        let step = self.total_steps_observed + 1;
        let phase = step % self.config.augmentation_period;
        let config = &self.config;
        let steps = &self.steps;
        let return_bins_by_step = &self.return_bins_by_step;
        let history_base_step = self.history_base_step;
        let action_bits = self.action_bits;
        let return_label_codec = self.return_label_codec;
        let token_ctx = AiqiAugmentedTokenContext {
            config,
            history_base_step,
            steps,
            return_bins_by_step,
            action_bits,
            return_label_codec,
            phase,
        };

        let mut q_values = vec![0.0; self.config.agent_actions.get()];
        let mut pushed_fast_forward = 0usize;

        {
            let model = &mut self.phases[phase];
            let start = (model.last_augmented_step + 1).max(history_base_step);
            let end = step.saturating_sub(1);
            if start <= end {
                for idx in start..=end {
                    pushed_fast_forward +=
                        push_step_tokens_history(&token_ctx, model.predictor.as_mut(), idx);
                }
            }

            for (action, q_value) in q_values
                .iter_mut()
                .enumerate()
                .take(self.config.agent_actions.get())
            {
                let pushed_action = push_encoded_bits_history(
                    model.predictor.as_mut(),
                    action as u64,
                    self.action_bits,
                );
                let expected_label = predict_expected_label(
                    model.predictor.as_mut(),
                    self.return_label_codec,
                    ReturnPrefixUpdate::Training,
                    ReturnLawEvaluator::SharedPrefix,
                );
                *q_value = expected_label / self.config.return_bins as f64;
                pop_history_bits(model.predictor.as_mut(), pushed_action);
            }

            pop_history_bits(model.predictor.as_mut(), pushed_fast_forward);
        }

        q_values
    }

    fn estimate_q_values_generic(&mut self) -> Vec<f64> {
        let step = self.total_steps_observed + 1;
        let phase = step % self.config.augmentation_period;

        let model = &self.phases[phase];
        let mut context_predictor = model.predictor.boxed_clone();
        let token_ctx = AiqiAugmentedTokenContext {
            config: &self.config,
            history_base_step: self.history_base_step,
            steps: &self.steps,
            return_bins_by_step: &self.return_bins_by_step,
            action_bits: self.action_bits,
            return_label_codec: self.return_label_codec,
            phase,
        };

        let start = (model.last_augmented_step + 1).max(self.history_base_step);
        let end = step.saturating_sub(1);
        if start <= end {
            for idx in start..=end {
                push_augmented_step_tokens_commit(&token_ctx, context_predictor.as_mut(), idx)
                    .expect(
                        "generic planner retained history must contain required augmented return",
                    );
            }
        }

        let mut q_values = vec![0.0; self.config.agent_actions.get()];
        for (action, q_value) in q_values
            .iter_mut()
            .enumerate()
            .take(self.config.agent_actions.get())
        {
            let mut action_predictor = context_predictor.boxed_clone();
            let _ = push_encoded_bits_commit_history(
                action_predictor.as_mut(),
                action as u64,
                self.action_bits,
            );
            let expected_label = predict_expected_label(
                action_predictor.as_mut(),
                self.return_label_codec,
                ReturnPrefixUpdate::Training,
                ReturnLawEvaluator::SharedPrefix,
            );
            *q_value = expected_label / self.config.return_bins as f64;
        }

        q_values
    }

    fn advance_phase_model_to_step(
        &mut self,
        phase: usize,
        target_step: usize,
    ) -> Result<(), AiqiError> {
        let token_ctx = AiqiAugmentedTokenContext {
            config: &self.config,
            history_base_step: self.history_base_step,
            steps: &self.steps,
            return_bins_by_step: &self.return_bins_by_step,
            action_bits: self.action_bits,
            return_label_codec: self.return_label_codec,
            phase,
        };
        let model = &mut self.phases[phase];
        if target_step <= model.last_augmented_step {
            return Ok(());
        }

        let start = (model.last_augmented_step + 1).max(token_ctx.history_base_step);
        for idx in start..=target_step {
            push_augmented_step_tokens_commit(&token_ctx, model.predictor.as_mut(), idx)?;
        }

        model.last_augmented_step = target_step;
        Ok(())
    }

    fn compute_return_bin(&self, start_step: usize) -> u64 {
        let h = self.config.return_horizon;
        let gamma = self.config.discount_gamma;

        debug_assert!(0.0 < gamma && gamma < 1.0);
        let reward_range = (self.config.max_reward - self.config.min_reward) as f64;

        // Paper definition: R_{t,H} = (1-gamma) * sum_{k=0}^{H-1} gamma^k r_{t+k}.
        let mut total = 0.0f64;
        let mut gk = 1.0f64;
        for k in 0..h {
            let idx = start_step + k;
            let local_idx = self
                .local_index(idx)
                .expect("return computation requires in-range history");
            let r = self.steps[local_idx].reward;
            let rn = if reward_range <= 0.0 {
                0.0
            } else {
                ((r - self.config.min_reward) as f64 / reward_range).clamp(0.0, 1.0)
            };
            total += gk * rn;
            gk *= gamma;
        }
        let ret = ((1.0 - gamma) * total).clamp(0.0, 1.0);

        let mut bin = (ret * (self.config.return_bins as f64)).floor() as u64;
        let max_bin = (self.config.return_bins as u64).saturating_sub(1);
        if bin > max_bin {
            bin = max_bin;
        }
        bin
    }

    fn local_index(&self, global_step: usize) -> Result<usize, AiqiError> {
        if global_step < self.history_base_step || global_step > self.total_steps_observed {
            return Err(AiqiError::HistoryIndexOutOfRange {
                global_step,
                history_base_step: self.history_base_step,
                total_steps_observed: self.total_steps_observed,
            });
        }
        Ok(global_step - self.history_base_step)
    }

    fn maybe_prune_history(&mut self) {
        let Some(keep_steps) = self.config.history_prune_keep_steps else {
            return;
        };
        if self.steps.is_empty() {
            return;
        }

        let min_phase_committed = self
            .phases
            .iter()
            .map(|phase| phase.last_augmented_step)
            .min()
            .unwrap_or(0);

        // For the next return update, we must retain steps from
        // (t+2-H) onward (1-based indexing). Everything before that is no
        // longer needed for exact H-step return construction.
        let next_start_needed = self
            .total_steps_observed
            .saturating_add(2)
            .saturating_sub(self.config.return_horizon);
        let returns_safe_drop_upto = next_start_needed.saturating_sub(1);

        let mut safe_drop_upto = min_phase_committed.min(returns_safe_drop_upto);

        // Optional retention floor: keep at least `keep_steps` most recent
        // transitions in memory for diagnostics/debugging.
        let keep_floor_drop_upto = self.total_steps_observed.saturating_sub(keep_steps);
        safe_drop_upto = safe_drop_upto.min(keep_floor_drop_upto);

        if safe_drop_upto < self.history_base_step {
            return;
        }

        let drain_count = safe_drop_upto - self.history_base_step + 1;
        if drain_count == 0 || drain_count > self.steps.len() {
            return;
        }

        self.steps.drain(0..drain_count);
        self.return_bins_by_step.drain(0..drain_count);
        self.history_base_step += drain_count;
    }
}

struct AiqiAugmentedTokenContext<'a> {
    config: &'a AiqiRuntimeConfig,
    history_base_step: usize,
    steps: &'a [StepRecord],
    return_bins_by_step: &'a [Option<u64>],
    action_bits: usize,
    return_label_codec: ReturnLabelCodec,
    phase: usize,
}

fn push_step_tokens_history(
    ctx: &AiqiAugmentedTokenContext<'_>,
    predictor: &mut dyn Predictor,
    idx: usize,
) -> usize {
    let mut pushed = 0usize;
    pushed += push_action_tokens_history(
        ctx.history_base_step,
        ctx.steps,
        ctx.action_bits,
        predictor,
        idx,
    );

    if idx % ctx.config.augmentation_period == ctx.phase {
        let local_idx = idx - ctx.history_base_step;
        if let Some(bin) = ctx.return_bins_by_step[local_idx] {
            pushed += ctx.return_label_codec.push_label_history(predictor, bin);
        }
    }

    pushed
        + push_percept_tokens_history(ctx.config, ctx.history_base_step, ctx.steps, predictor, idx)
}

fn push_augmented_step_tokens_commit(
    ctx: &AiqiAugmentedTokenContext<'_>,
    predictor: &mut dyn Predictor,
    idx: usize,
) -> Result<usize, AiqiError> {
    let mut pushed = 0usize;
    pushed += push_action_tokens_commit_history(
        ctx.history_base_step,
        ctx.steps,
        ctx.action_bits,
        predictor,
        idx,
    );

    if idx % ctx.config.augmentation_period == ctx.phase {
        let local_idx = idx - ctx.history_base_step;
        let bin = ctx.return_bins_by_step[local_idx].ok_or(AiqiError::MissingReturnBin {
            step: idx,
            phase: ctx.phase,
        })?;
        pushed += ctx.return_label_codec.push_label_commit(predictor, bin);
    }

    Ok(pushed
        + push_percept_tokens_commit_history(
            ctx.config,
            ctx.history_base_step,
            ctx.steps,
            predictor,
            idx,
        ))
}

fn push_action_tokens_history(
    history_base_step: usize,
    steps: &[StepRecord],
    action_bits: usize,
    predictor: &mut dyn Predictor,
    idx: usize,
) -> usize {
    let action = steps[idx - history_base_step].action;
    push_encoded_bits_history(predictor, action, action_bits)
}

fn push_action_tokens_commit_history(
    history_base_step: usize,
    steps: &[StepRecord],
    action_bits: usize,
    predictor: &mut dyn Predictor,
    idx: usize,
) -> usize {
    let action = steps[idx - history_base_step].action;
    push_encoded_bits_commit_history(predictor, action, action_bits)
}

fn push_percept_tokens_history(
    config: &AiqiRuntimeConfig,
    history_base_step: usize,
    steps: &[StepRecord],
    predictor: &mut dyn Predictor,
    idx: usize,
) -> usize {
    let step = &steps[idx - history_base_step];
    let mut pushed = 0usize;
    for &obs in &step.observations {
        pushed += push_encoded_bits_history(predictor, obs, config.observation_bits);
    }
    pushed
        + push_encoded_reward_history(
            predictor,
            step.reward,
            config.reward_bits,
            config.reward_offset,
        )
}

fn push_percept_tokens_commit_history(
    config: &AiqiRuntimeConfig,
    history_base_step: usize,
    steps: &[StepRecord],
    predictor: &mut dyn Predictor,
    idx: usize,
) -> usize {
    let step = &steps[idx - history_base_step];
    let mut pushed = 0usize;
    for &obs in &step.observations {
        pushed += push_encoded_bits_commit_history(predictor, obs, config.observation_bits);
    }
    pushed
        + push_encoded_reward_commit_history(
            predictor,
            step.reward,
            config.reward_bits,
            config.reward_offset,
        )
}

fn rate_backend_supports_aiqi_frozen_conditioning(backend: &RateBackend) -> bool {
    backend
        .compile()
        .map(|compiled| compiled.supports_frozen_conditioning())
        .unwrap_or(false)
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
) -> usize {
    let shifted = (reward as i128) + (offset as i128);
    let as_u64 = if shifted <= 0 {
        0
    } else if shifted > (u64::MAX as i128) {
        u64::MAX
    } else {
        shifted as u64
    };
    push_encoded_bits_history(predictor, as_u64, bits)
}

fn push_encoded_reward_commit_history(
    predictor: &mut dyn Predictor,
    reward: Reward,
    bits: usize,
    offset: Reward,
) -> usize {
    let shifted = (reward as i128) + (offset as i128);
    let as_u64 = if shifted <= 0 {
        0
    } else if shifted > (u64::MAX as i128) {
        u64::MAX
    } else {
        shifted as u64
    };
    push_encoded_bits_commit_history(predictor, as_u64, bits)
}

fn pop_history_bits(predictor: &mut dyn Predictor, bits: usize) {
    for _ in 0..bits {
        predictor.pop_history();
    }
}

fn argmax_with_fixed_tie_break(values: &[f64]) -> usize {
    let mut best_value = f64::NEG_INFINITY;
    let mut best_idx = 0usize;
    for (i, &v) in values.iter().enumerate() {
        if v > best_value {
            best_value = v;
            best_idx = i;
        }
    }
    best_idx
}

#[cfg(all(test, feature = "all-backends"))]
mod tests {
    use super::*;
    use crate::aixi::environment::Environment;
    use crate::aixi::return_law::{
        ReturnLabelBitOrder, ReturnLabelCodec, ReturnLawEvaluator, ReturnPrefixUpdate,
        predict_return_law,
    };
    use crate::aixi::test_envs::DeterministicBinaryEnv;
    use crate::api::{MixtureKind, MixtureSpec};
    use std::sync::{Arc, Mutex};

    fn basic_config() -> AiqiConfig {
        AiqiConfig {
            rate_backend: RateBackend::Ctw { depth: 8 },
            bit_stream_semantics: crate::api::BitStreamSemantics::BinaryTokens,
            observation_bits: 1,
            observation_stream_len: 1,
            reward_bits: 1,
            agent_actions: ActionAlphabet::try_from_usize(2)
                .expect("test fixture action alphabet must be non-zero"),
            min_reward: 0,
            max_reward: 1,
            reward_offset: 0,
            discount_gamma: 0.99,
            return_horizon: 2,
            return_bins: 8,
            augmentation_period: 2,
            history_prune_keep_steps: None,
            baseline_exploration: 0.01,
            random_seed: Some(7),
        }
    }

    fn generic_mixture_config() -> AiqiConfig {
        AiqiConfig {
            rate_backend: RateBackend::Mixture {
                spec: Arc::new(
                    MixtureSpec::new(
                        MixtureKind::Bayes,
                        vec![
                            crate::api::MixtureExpertSpec {
                                name: Some("ctw".to_string()),
                                log_prior: 0.0,
                                backend: RateBackend::Ctw { depth: 8 },
                            },
                            crate::api::MixtureExpertSpec {
                                name: Some("match".to_string()),
                                log_prior: 0.0,
                                backend: RateBackend::Match {
                                    hash_bits: 16,
                                    min_len: 2,
                                    max_len: 16,
                                    base_mix: 0.05,
                                    confidence_scale: 1.0,
                                },
                            },
                        ],
                    )
                    .with_alpha(0.03),
                ),
            },
            random_seed: Some(11),
            baseline_exploration: 0.01,
            ..basic_config()
        }
    }

    fn native_reversible_mixture_config() -> AiqiConfig {
        AiqiConfig {
            rate_backend: RateBackend::Mixture {
                spec: Arc::new(
                    MixtureSpec::new(
                        MixtureKind::Bayes,
                        vec![
                            crate::api::MixtureExpertSpec {
                                name: Some("ctw".to_string()),
                                log_prior: 0.0,
                                backend: RateBackend::Ctw { depth: 8 },
                            },
                            crate::api::MixtureExpertSpec {
                                name: Some("fac-ctw".to_string()),
                                log_prior: 0.0,
                                backend: RateBackend::FacCtw {
                                    base_depth: 8,
                                    num_percept_bits: 1,
                                    encoding_bits: 1,
                                    msb_first: None,
                                },
                            },
                        ],
                    )
                    .with_alpha(0.03),
                ),
            },
            random_seed: Some(13),
            baseline_exploration: 0.01,
            ..basic_config()
        }
    }

    fn run_ctw_trace(agent: &mut AiqiAgent, cycles: usize) -> (Vec<Action>, i64) {
        let mut env = DeterministicBinaryEnv::default();
        let mut actions = Vec::with_capacity(cycles);
        let mut total_reward = 0i64;

        for _ in 0..cycles {
            let action = agent.get_planned_action();
            actions.push(action);
            env.perform_action(action);
            let obs_stream = env.drain_observations();
            let reward = env.get_reward();
            agent
                .observe_transition(action, &obs_stream, reward)
                .expect("transition should be accepted");
            total_reward += reward;
        }

        (actions, total_reward)
    }

    #[test]
    fn programmatic_aiqi_preserves_explicit_signed_reward_contract() {
        let mut config = basic_config();
        config.reward_bits = 3;
        config.min_reward = -2;
        config.max_reward = 3;
        config.reward_offset = 2;

        let mut agent = AiqiAgent::new(config).expect("signed reward config should be valid");
        assert_eq!(agent.config.min_reward, -2);
        assert_eq!(agent.config.max_reward, 3);
        assert_eq!(agent.config.reward_offset, 2);

        agent
            .observe_transition(0, &[0], -2)
            .expect("signed reward within explicit config bounds should be accepted");
    }

    #[derive(Clone, Default)]
    struct CountingPredictor {
        update_calls: usize,
        commit_update_calls: usize,
        update_history_calls: usize,
        commit_update_history_calls: usize,
        revert_calls: usize,
        pop_history_calls: usize,
    }

    impl Predictor for CountingPredictor {
        fn update(&mut self, _sym: bool) {
            self.update_calls += 1;
        }

        fn commit_update(&mut self, _sym: bool) {
            self.commit_update_calls += 1;
        }

        fn update_history(&mut self, _sym: bool) {
            self.update_history_calls += 1;
        }

        fn commit_update_history(&mut self, _sym: bool) {
            self.commit_update_history_calls += 1;
        }

        fn revert(&mut self) {
            self.revert_calls += 1;
        }

        fn pop_history(&mut self) {
            self.pop_history_calls += 1;
        }

        fn predict_prob(&mut self, sym: bool) -> f64 {
            if sym { 0.75 } else { 0.25 }
        }

        fn model_name(&self) -> String {
            "CountingPredictor".to_string()
        }

        fn boxed_clone(&self) -> Box<dyn Predictor> {
            Box::new(self.clone())
        }
    }

    #[derive(Clone, Default)]
    struct SharedCallCounts {
        update: usize,
        commit_update: usize,
        update_history: usize,
        commit_update_history: usize,
    }

    #[derive(Clone)]
    struct SharedCountingPredictor {
        counts: Arc<Mutex<SharedCallCounts>>,
    }

    impl SharedCountingPredictor {
        fn new(counts: Arc<Mutex<SharedCallCounts>>) -> Self {
            Self { counts }
        }
    }

    impl Predictor for SharedCountingPredictor {
        fn update(&mut self, _sym: bool) {
            self.counts.lock().unwrap().update += 1;
        }

        fn commit_update(&mut self, _sym: bool) {
            self.counts.lock().unwrap().commit_update += 1;
        }

        fn update_history(&mut self, _sym: bool) {
            self.counts.lock().unwrap().update_history += 1;
        }

        fn commit_update_history(&mut self, _sym: bool) {
            self.counts.lock().unwrap().commit_update_history += 1;
        }

        fn revert(&mut self) {}

        fn pop_history(&mut self) {}

        fn predict_prob(&mut self, sym: bool) -> f64 {
            if sym { 0.75 } else { 0.25 }
        }

        fn model_name(&self) -> String {
            "SharedCountingPredictor".to_string()
        }

        fn boxed_clone(&self) -> Box<dyn Predictor> {
            Box::new(self.clone())
        }
    }

    #[derive(Clone, Default)]
    struct ReturnLearningPredictor {
        saw_training_one: bool,
        rollback: Vec<bool>,
    }

    impl Predictor for ReturnLearningPredictor {
        fn update(&mut self, sym: bool) {
            self.rollback.push(self.saw_training_one);
            if sym {
                self.saw_training_one = true;
            }
        }

        fn commit_update(&mut self, sym: bool) {
            if sym {
                self.saw_training_one = true;
            }
        }

        fn update_history(&mut self, _sym: bool) {}

        fn commit_update_history(&mut self, _sym: bool) {}

        fn revert(&mut self) {
            self.saw_training_one = self
                .rollback
                .pop()
                .expect("test predictor rollback underflow");
        }

        fn pop_history(&mut self) {}

        fn predict_prob(&mut self, sym: bool) -> f64 {
            let p1 = if self.saw_training_one { 0.75 } else { 0.25 };
            if sym { p1 } else { 1.0 - p1 }
        }

        fn model_name(&self) -> String {
            "ReturnLearningPredictor".to_string()
        }

        fn boxed_clone(&self) -> Box<dyn Predictor> {
            Box::new(self.clone())
        }
    }

    #[derive(Clone, Default)]
    struct ActionConditionedBernoulliPredictor {
        history: Vec<bool>,
    }

    impl ActionConditionedBernoulliPredictor {
        fn p_one_after_action(&self) -> f64 {
            if self.history.first().copied().unwrap_or(false) {
                0.75
            } else {
                0.25
            }
        }
    }

    impl Predictor for ActionConditionedBernoulliPredictor {
        fn update(&mut self, sym: bool) {
            self.history.push(sym);
        }

        fn commit_update(&mut self, sym: bool) {
            self.history.push(sym);
        }

        fn update_history(&mut self, sym: bool) {
            self.history.push(sym);
        }

        fn commit_update_history(&mut self, sym: bool) {
            self.history.push(sym);
        }

        fn revert(&mut self) {
            self.history
                .pop()
                .expect("test predictor rollback underflow");
        }

        fn pop_history(&mut self) {
            self.history
                .pop()
                .expect("test predictor history underflow");
        }

        fn predict_prob(&mut self, sym: bool) -> f64 {
            let p_one = self.p_one_after_action();
            if sym { p_one } else { 1.0 - p_one }
        }

        fn model_name(&self) -> String {
            "ActionConditionedBernoulliPredictor".to_string()
        }

        fn boxed_clone(&self) -> Box<dyn Predictor> {
            Box::new(self.clone())
        }
    }

    #[derive(Clone)]
    struct ScopedReturnLearningPredictor {
        saw_training_one: bool,
        rollback: Vec<bool>,
        clone_count: Arc<Mutex<usize>>,
    }

    impl ScopedReturnLearningPredictor {
        fn new(clone_count: Arc<Mutex<usize>>) -> Self {
            Self {
                saw_training_one: false,
                rollback: Vec::new(),
                clone_count,
            }
        }
    }

    impl Predictor for ScopedReturnLearningPredictor {
        fn update(&mut self, sym: bool) {
            self.rollback.push(self.saw_training_one);
            if sym {
                self.saw_training_one = true;
            }
        }

        fn commit_update(&mut self, sym: bool) {
            if sym {
                self.saw_training_one = true;
            }
        }

        fn revert(&mut self) {
            self.saw_training_one = self
                .rollback
                .pop()
                .expect("test predictor rollback underflow");
        }

        fn predict_prob(&mut self, sym: bool) -> f64 {
            let p1 = if self.saw_training_one { 0.75 } else { 0.25 };
            if sym { p1 } else { 1.0 - p1 }
        }

        fn model_name(&self) -> String {
            "ScopedReturnLearningPredictor".to_string()
        }

        fn boxed_clone(&self) -> Box<dyn Predictor> {
            *self.clone_count.lock().unwrap() += 1;
            Box::new(self.clone())
        }
    }

    #[test]
    fn config_rejects_invalid_period() {
        let mut cfg = basic_config();
        cfg.augmentation_period = 1;
        cfg.return_horizon = 2;
        let err = cfg
            .validate()
            .expect_err("N < H must be rejected to match \"A Model-Free Universal AI\"");
        assert!(matches!(
            err,
            AiqiError::AugmentationPeriodTooShort {
                augmentation_period: 1,
                return_horizon: 2
            }
        ));
    }

    #[test]
    fn config_accepts_non_power_of_two_return_bins() {
        let mut cfg = basic_config();
        cfg.return_bins = 3;
        cfg.validate()
            .expect("non-power-of-two return_bins are valid AIQI discretization levels");
    }

    #[test]
    fn non_power_of_two_aiqi_runtime_normalizes_invalid_return_leaf_and_plans() {
        let mut cfg = basic_config();
        cfg.return_bins = 3;
        cfg.return_horizon = 1;
        cfg.augmentation_period = 1;
        cfg.baseline_exploration = f64::MIN_POSITIVE;
        cfg.random_seed = Some(3);

        let mut agent = AiqiAgent::new(cfg).expect("non-power-of-two AIQI config should build");
        agent.phases[0].predictor = Box::new(ActionConditionedBernoulliPredictor::default());

        let mut action_one_predictor = ActionConditionedBernoulliPredictor::default();
        action_one_predictor.update_history(true);
        let law = predict_return_law(
            &mut action_one_predictor,
            agent.return_label_codec,
            ReturnPrefixUpdate::Training,
            ReturnLawEvaluator::SharedPrefix,
        );
        assert_eq!(law.probabilities.len(), 3);
        assert_eq!(law.stats.invalid_leaves, 1);
        assert!(
            (law.probabilities.iter().sum::<f64>() - 1.0).abs() < 1e-12,
            "non-power-of-two AIQI law must normalize only valid labels: {:?}",
            law.probabilities
        );
        let expected_action_one_law = [1.0 / 7.0, 3.0 / 7.0, 3.0 / 7.0];
        for (actual, expected) in law.probabilities.iter().zip(expected_action_one_law.iter()) {
            assert!(
                (actual - expected).abs() < 1e-12,
                "expected action-1 law {:?}, got {:?}",
                expected_action_one_law,
                law.probabilities
            );
        }

        let q_values = agent.estimate_q_values();
        assert_eq!(q_values.len(), 2);
        assert!(
            (q_values[0] - 0.2).abs() < 1e-12,
            "action 0 should decode the normalized [0.6, 0.2, 0.2] law, got {q_values:?}"
        );
        assert!(
            (q_values[1] - (3.0 / 7.0)).abs() < 1e-12,
            "action 1 should decode the normalized [1/7, 3/7, 3/7] law, got {q_values:?}"
        );

        agent.reseed_random(3);
        let (action, explored) = agent.get_planned_action_with_extra_exploration_flag(0.0);
        assert_eq!(action, 1);
        assert!(!explored);
    }

    #[test]
    fn forced_aiqi_exploration_skips_value_descent() {
        let mut agent = AiqiAgent::new(basic_config()).expect("valid aiqi config");
        let counts = Arc::new(Mutex::new(SharedCallCounts::default()));
        let decision_phase = (agent.total_steps_observed + 1) % agent.config.augmentation_period;
        agent.phases[decision_phase].predictor =
            Box::new(SharedCountingPredictor::new(counts.clone()));

        let (_action, explored) = agent.get_planned_action_with_extra_exploration_flag(1.0);

        assert!(explored);
        let snapshot = counts.lock().unwrap().clone();
        assert_eq!(snapshot.update, 0);
        assert_eq!(snapshot.update_history, 0);
        assert_eq!(snapshot.commit_update, 0);
        assert_eq!(snapshot.commit_update_history, 0);
    }

    #[test]
    fn config_rejects_zpaq_rate_backend_in_strict_mode() {
        let mut cfg = basic_config();
        cfg.rate_backend = RateBackend::Zpaq {
            method: crate::api::ZpaqMethodSpec::literal("1"),
        };
        let err = cfg
            .validate()
            .expect_err("strict AIQI must reject zpaq rate backend");
        assert!(matches!(err, AiqiError::UnsupportedRateBackend { .. }));
    }

    #[test]
    fn config_rejects_nonpaper_gamma_or_tau() {
        let mut cfg = basic_config();
        cfg.discount_gamma = 1.0;
        let err = cfg
            .validate()
            .expect_err("gamma=1 must be rejected for strict paper AIQI");
        assert!(matches!(
            err,
            AiqiError::InvalidDiscountGamma { value: 1.0 }
        ));

        cfg = basic_config();
        cfg.baseline_exploration = 0.0;
        let err = cfg
            .validate()
            .expect_err("tau=0 must be rejected for strict paper AIQI");
        assert!(matches!(
            err,
            AiqiError::InvalidBaselineExploration { value: 0.0 }
        ));
    }

    #[test]
    fn byte_packed_config_allows_observation_and_reward_to_share_a_byte() {
        let mut cfg = basic_config();
        cfg.bit_stream_semantics = BitStreamSemantics::BytePacked {
            order: crate::prediction::BitOrder::MsbFirst,
        };
        cfg.agent_actions = ActionAlphabet::try_from_usize(256)
            .expect("test fixture action alphabet must be byte-aligned");
        cfg.observation_bits = 3;
        cfg.reward_bits = 5;
        cfg.return_bins = 256;

        cfg.validate().expect(
            "byte-packed AIQI should allow observations and reward to share one percept byte",
        );
    }

    #[test]
    fn aiqi_estimates_action_values_after_observations() {
        let mut agent = AiqiAgent::new(basic_config()).expect("valid aiqi config");
        for _ in 0..8 {
            agent
                .observe_transition(1, &[1], 1)
                .expect("transition should be accepted");
        }

        let action = agent.get_planned_action();
        assert!(action <= 1);
    }

    #[test]
    fn fac_ctw_predictor_uses_return_bit_width() {
        let mut cfg = basic_config();
        cfg.return_bins = 8; // return_bits=3
        cfg.rate_backend = RateBackend::FacCtw {
            base_depth: 8,
            num_percept_bits: bits_for_cardinality(cfg.return_bins),
            encoding_bits: 1,
            msb_first: None,
        };

        let agent = AiqiAgent::new(cfg).expect("valid aiqi config");
        let name = agent.phases[0].predictor.model_name();
        assert!(
            name.contains("k=3"),
            "FAC-CTW should factorize over return bits only, model_name={name}"
        );
    }

    #[test]
    fn ac_ctw_path_uses_single_tree_predictor() {
        let agent = AiqiAgent::new(basic_config()).expect("valid aiqi config");
        let name = agent.phases[0].predictor.model_name();
        assert!(
            name.starts_with("AC-CTW"),
            "ac-ctw should map to the single-tree CTW predictor, model_name={name}"
        );
    }

    #[test]
    fn distribution_rollout_uses_update_and_revert_when_requested() {
        let mut predictor = CountingPredictor::default();
        let law = predict_return_law(
            &mut predictor,
            ReturnLabelCodec::value_monotone(4),
            ReturnPrefixUpdate::Training,
            ReturnLawEvaluator::SharedPrefix,
        );

        assert_eq!(law.probabilities.len(), 4);
        assert_eq!(law.stats.logical_queries, 3);
        assert_eq!(predictor.update_calls, 6);
        assert_eq!(predictor.revert_calls, 6);
        assert_eq!(predictor.update_history_calls, 0);
        assert_eq!(predictor.pop_history_calls, 0);
    }

    #[test]
    fn distribution_rollout_uses_history_path_when_not_requested() {
        let mut predictor = CountingPredictor::default();
        let law = predict_return_law(
            &mut predictor,
            ReturnLabelCodec::value_monotone(4),
            ReturnPrefixUpdate::FrozenHistory,
            ReturnLawEvaluator::SharedPrefix,
        );

        assert_eq!(law.probabilities.len(), 4);
        assert_eq!(law.stats.logical_queries, 3);
        assert_eq!(predictor.update_calls, 0);
        assert_eq!(predictor.revert_calls, 0);
        assert_eq!(predictor.update_history_calls, 6);
        assert_eq!(predictor.pop_history_calls, 6);
    }

    #[test]
    fn generic_distribution_rollout_trains_on_return_symbols() {
        let mut predictor = ReturnLearningPredictor::default();
        let law = predict_return_law(
            &mut predictor,
            ReturnLabelCodec::value_monotone(4),
            ReturnPrefixUpdate::Training,
            ReturnLawEvaluator::SharedPrefix,
        );
        let probs = law.probabilities;

        assert_eq!(probs.len(), 4);
        assert!((probs.iter().sum::<f64>() - 1.0).abs() < 1e-12);
        assert!(
            probs[3] > probs[2],
            "training on the first return bit should make bin 11 likelier than 10; got {:?}",
            probs
        );
        assert!(
            (probs[0] - 0.5625).abs() < 1e-12,
            "expected exact normalized mass for 00, got {:?}",
            probs
        );
        assert!(
            !predictor.saw_training_one,
            "shared-prefix rollout must restore the caller's predictor state"
        );
    }

    #[test]
    fn generic_distribution_rollout_does_not_clone_per_label() {
        let clone_count = Arc::new(Mutex::new(0usize));
        let mut predictor = ScopedReturnLearningPredictor::new(clone_count.clone());
        let law = predict_return_law(
            &mut predictor,
            ReturnLabelCodec::value_monotone(4),
            ReturnPrefixUpdate::Training,
            ReturnLawEvaluator::SharedPrefix,
        );
        let probs = law.probabilities;

        assert_eq!(probs.len(), 4);
        assert!(
            probs[3] > probs[2],
            "shared-prefix training on return bits should preserve autoregressive semantics"
        );
        assert_eq!(
            *clone_count.lock().unwrap(),
            0,
            "shared-prefix evaluation should not clone once per return bin"
        );
        assert!(
            !predictor.saw_training_one,
            "shared-prefix rollout must restore the caller's predictor state"
        );
    }

    #[test]
    fn aiqi_return_label_codec_is_value_monotone() {
        let agent = AiqiAgent::new(basic_config()).expect("valid aiqi config");
        assert_eq!(
            agent.return_label_codec.order(),
            ReturnLabelBitOrder::MsbFirst
        );
        assert_eq!(
            agent.return_label_codec.label_range_for_prefix(0, 1),
            Some((0, 3))
        );
        assert_eq!(
            agent.return_label_codec.label_range_for_prefix(1, 1),
            Some((4, 7))
        );
    }

    #[test]
    fn return_bin_for_gamma_less_than_one_matches_paper_h_step_return() {
        let mut cfg = basic_config();
        cfg.discount_gamma = 0.5;
        cfg.return_bins = 8;

        let mut agent = AiqiAgent::new(cfg).expect("valid aiqi config");
        agent
            .observe_transition(0, &[0], 1)
            .expect("first transition stored");
        agent
            .observe_transition(0, &[0], 0)
            .expect("second transition should produce first return");

        let bin = agent.return_bins_by_step[0].expect("first return should be available");
        // Paper target: R_{t,H} = (1-gamma) * sum_{k=0}^{H-1} gamma^k r_{t+k}.
        // For rewards [1, 0], gamma=0.5, H=2 this equals 0.5.
        // With M=8 bins this maps to floor(8 * 0.5) = 4.
        assert_eq!(bin, 4);
    }

    #[test]
    fn optional_history_pruning_bounds_retained_state_without_losing_progress() {
        let mut cfg = basic_config();
        cfg.return_horizon = 3;
        cfg.augmentation_period = 4;
        cfg.history_prune_keep_steps = Some(8);

        let mut agent = AiqiAgent::new(cfg).expect("valid aiqi config");
        for i in 0..256usize {
            let action = (i % 2) as u64;
            let obs = [(i % 2) as u64];
            let rew = (i % 2) as i64;
            agent
                .observe_transition(action, &obs, rew)
                .expect("transition should be accepted");
        }

        // Global progress should be preserved even when retained history is bounded.
        assert_eq!(agent.steps_observed(), 256);
        assert!(
            agent.history_base_step > 1,
            "history should have been pruned"
        );
        assert!(
            agent.steps.len() < agent.steps_observed(),
            "retained history should be smaller than total observed"
        );

        let action = agent.get_planned_action();
        assert!(action <= 1);
    }

    #[test]
    fn committed_phase_advancement_uses_commit_predictor_paths() {
        let mut agent = AiqiAgent::new(basic_config()).expect("valid aiqi config");
        let counts = Arc::new(Mutex::new(SharedCallCounts::default()));
        agent.phases[1].predictor = Box::new(SharedCountingPredictor::new(counts.clone()));
        agent.phases[1].last_augmented_step = 0;
        agent.history_base_step = 1;
        agent.total_steps_observed = 1;
        agent.steps = vec![StepRecord {
            action: 1,
            observations: vec![1],
            reward: 1,
        }];
        agent.return_bins_by_step = vec![Some(3)];

        agent
            .advance_phase_model_to_step(1, 1)
            .expect("phase advancement should succeed");

        let snapshot = counts.lock().unwrap().clone();
        assert_eq!(snapshot.commit_update, 3);
        assert_eq!(snapshot.commit_update_history, 3);
        assert_eq!(snapshot.update, 0);
        assert_eq!(snapshot.update_history, 0);
    }

    #[test]
    fn generic_planner_trains_on_returns_and_freezes_conditioning_tokens() {
        let mut cfg = basic_config();
        cfg.rate_backend = RateBackend::Match {
            hash_bits: 16,
            min_len: 2,
            max_len: 16,
            base_mix: 0.05,
            confidence_scale: 1.0,
        };

        let mut agent = AiqiAgent::new(cfg).expect("valid aiqi config");
        let counts = Arc::new(Mutex::new(SharedCallCounts::default()));
        agent.phases[1].predictor = Box::new(SharedCountingPredictor::new(counts.clone()));
        agent.phases[1].last_augmented_step = 0;
        agent.history_base_step = 1;
        agent.total_steps_observed = 2;
        agent.steps = vec![
            StepRecord {
                action: 1,
                observations: vec![1],
                reward: 1,
            },
            StepRecord {
                action: 0,
                observations: vec![0],
                reward: 0,
            },
        ];
        agent.return_bins_by_step = vec![Some(3), None];

        let q_values = agent.estimate_q_values_generic();

        assert_eq!(q_values.len(), agent.config.agent_actions.get());
        let snapshot = counts.lock().unwrap().clone();
        assert!(
            snapshot.update > 0,
            "generic planner should train on hypothetical return-prefix symbols"
        );
        assert_eq!(snapshot.update_history, 0);
        assert!(
            snapshot.commit_update > 0,
            "generic planner should train on committed augmented return symbols"
        );
        assert!(
            snapshot.commit_update_history > 0,
            "generic planner should keep action/percept conditioning frozen"
        );
    }

    #[test]
    fn native_reversible_mixture_uses_reversible_aiqi_planner() {
        let config = native_reversible_mixture_config();
        let agent = AiqiAgent::new(config).expect("native reversible mixture should build");
        assert!(
            !agent.use_generic_planner,
            "mixtures composed of native reversible bit predictors should keep the reversible planner"
        );
        assert_eq!(
            agent.return_label_codec.order(),
            ReturnLabelBitOrder::MsbFirst
        );
    }

    #[test]
    fn compiled_aiqi_runtime_matches_legacy_config_for_generic_mixture_backend() {
        let config = generic_mixture_config();
        let compiled = config
            .compile_planner_run_spec()
            .expect("generic planner run should compile");
        let mut legacy = AiqiAgent::new(config).expect("legacy aiqi config");
        let mut canonical =
            AiqiAgent::from_compiled_planner_run(&compiled).expect("compiled aiqi config");

        let legacy_trace = run_ctw_trace(&mut legacy, 32);
        let canonical_trace = run_ctw_trace(&mut canonical, 32);
        assert_eq!(canonical_trace, legacy_trace);
    }
}

#[cfg(all(test, feature = "backend-ctw"))]
mod signed_reward_contract_tests {
    use super::*;

    fn action_alphabet(n: usize) -> ActionAlphabet {
        ActionAlphabet::try_from_usize(n).expect("test action alphabet must be non-zero")
    }

    #[test]
    fn programmatic_aiqi_preserves_explicit_signed_reward_contract_under_ctw() {
        let config = AiqiConfig {
            rate_backend: RateBackend::Ctw { depth: 8 },
            bit_stream_semantics: crate::api::BitStreamSemantics::BinaryTokens,
            observation_bits: 1,
            observation_stream_len: 1,
            reward_bits: 3,
            agent_actions: action_alphabet(2),
            min_reward: -2,
            max_reward: 3,
            reward_offset: 2,
            discount_gamma: 0.99,
            return_horizon: 2,
            return_bins: 8,
            augmentation_period: 2,
            history_prune_keep_steps: None,
            baseline_exploration: 0.01,
            random_seed: Some(7),
        };

        let mut agent = AiqiAgent::new(config).expect("signed reward config should be valid");
        assert_eq!(agent.config.min_reward, -2);
        assert_eq!(agent.config.max_reward, 3);
        assert_eq!(agent.config.reward_offset, 2);

        agent
            .observe_transition(0, &[0], -2)
            .expect("signed reward within explicit config bounds should be accepted");
    }
}
