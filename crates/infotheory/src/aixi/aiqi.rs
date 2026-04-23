//! AIQI implementation from "A Model-Free Universal AI".
//!
//! This module implements a model-free universal agent that predicts
//! discretized H-step returns directly from augmented interaction history.
//! The implementation follows the phase-indexed periodic augmentation in
//! "A Model-Free Universal AI":
//! for return horizon `H` and period `N >= H`, each phase model only inserts
//! returns at indices `i % N == phase`.

use crate::aixi::common::{
    Action, PerceptVal, RandomGenerator, Reward, bits_for_cardinality,
    validate_reward_encoding_bounds,
};
use crate::aixi::model::{Predictor, build_aiqi_predictor};
use crate::aixi::planner_spec::{PlannerInterfaceConfig, build_default_planner_run_spec};
use crate::api::{RateBackend, validate_rate_backend};
use crate::spec::{
    AiqiDiscountedControllerSpec, CompiledPlannerController, CompiledPlannerRunSpec,
    ControllerSpec, PlannerRunSpec,
};
#[cfg(feature = "backend-rwkv")]
use std::path::PathBuf;

/// Configuration parameters for an AIQI agent.
#[derive(Clone)]
pub struct AiqiConfig {
    /// Predictive backend.
    ///
    /// - `ac-ctw` / `ctw` / `ctw-context-tree`: AIQI-CTW path from
    ///   "A Model-Free Universal AI".
    /// - `fac-ctw`: factorized CTW extension.
    /// - `rosa` / `rwkv`: pluggable predictor extensions.
    /// - `zpaq`: intentionally unsupported for AIQI strict conditioning.
    pub algorithm: String,
    /// Context depth for CTW/FAC-CTW backends.
    pub ct_depth: usize,
    /// Number of bits used to encode observations.
    pub observation_bits: usize,
    /// Number of observation symbols per environment step.
    pub observation_stream_len: usize,
    /// Number of bits used to encode rewards.
    pub reward_bits: usize,
    /// Number of valid actions.
    pub agent_actions: usize,
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
    /// This implementation uses exact fixed-width binary encoding of return bins,
    /// so `return_bins` must be a power of two.
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
    /// When `None`, a fresh runtime-derived seed is used.
    pub random_seed: Option<u64>,
    /// Optional generic rate backend.
    ///
    /// When set, this takes precedence over `algorithm` and routes AIQI
    /// prediction through the shared `RateBackend` abstraction.
    pub rate_backend: Option<RateBackend>,
    /// Max-order hint for `rate_backend` constructors that use it (for example ROSA).
    pub rate_backend_max_order: i64,
    /// Optional RWKV model path.
    ///
    /// Required only when selecting `algorithm="rwkv"` and no `rate_backend`
    /// override is configured.
    pub rwkv_model_path: Option<String>,
    /// Optional ROSA max order.
    pub rosa_max_order: Option<i64>,
    /// Optional ZPAQ method string.
    pub zpaq_method: Option<String>,
}

impl AiqiConfig {
    fn canonical_predictor_backend(&self) -> Result<RateBackend, String> {
        if let Some(rate_backend) = &self.rate_backend {
            return Ok(rate_backend.clone());
        }

        match self.algorithm.as_str() {
            "ctw" | "ac-ctw" | "ctw-context-tree" => Ok(RateBackend::Ctw {
                depth: self.ct_depth,
            }),
            "fac-ctw" => Ok(RateBackend::FacCtw {
                base_depth: self.ct_depth,
                num_percept_bits: bits_for_cardinality(self.return_bins),
                encoding_bits: 1,
            }),
            "rosa" => Ok(RateBackend::RosaPlus),
            #[cfg(feature = "backend-rwkv")]
            "rwkv" => {
                let path = self.rwkv_model_path.as_ref().ok_or_else(|| {
                    "algorithm=rwkv requires rwkv_model_path when no rate_backend override is configured; for method-string RWKV configure rate_backend rwkv/rwkv7"
                        .to_string()
                })?;
                Ok(RateBackend::Rwkv7Method {
                    method: crate::rwkvzip::MethodSpec::File {
                        path: PathBuf::from(path),
                        policy: None,
                    },
                })
            }
            #[cfg(not(feature = "backend-rwkv"))]
            "rwkv" => Err("algorithm=rwkv requires backend-rwkv feature".to_string()),
            "zpaq" => Err(
                "AIQI strict mode does not support algorithm=zpaq; configure a backend with strict frozen conditioning"
                    .to_string(),
            ),
            other => Err(format!("Unknown AIQI algorithm: {other}")),
        }
    }

    fn canonical_planner_run_spec(&self) -> Result<PlannerRunSpec, String> {
        let predictor = self.canonical_predictor_backend()?;
        Ok(build_default_planner_run_spec(
            PlannerInterfaceConfig {
                observation_bits: self.observation_bits,
                observation_stream_len: self.observation_stream_len,
                observation_key_mode: crate::aixi::common::ObservationKeyMode::FullStream,
                reward_bits: self.reward_bits,
                agent_actions: self.agent_actions,
                min_reward: self.min_reward,
                max_reward: self.max_reward,
                reward_offset: self.reward_offset,
            },
            ControllerSpec::AiqiDiscounted(AiqiDiscountedControllerSpec {
                predictor,
                predictor_max_order: self.rate_backend_max_order,
                discount_gamma: self.discount_gamma,
                return_horizon: self.return_horizon,
                return_bins: self.return_bins,
                augmentation_period: self.augmentation_period,
                history_prune_keep_steps: self.history_prune_keep_steps,
                baseline_exploration: self.baseline_exploration,
            }),
            self.random_seed,
        ))
    }

    fn compile_planner_run_spec(&self) -> Result<CompiledPlannerRunSpec, String> {
        self.canonical_planner_run_spec()?
            .compile()
            .map_err(|err| err.to_string())
    }

    fn validate_runtime_invariants(&self) -> Result<(), String> {
        if self.agent_actions == 0 {
            return Err("agent_actions must be >= 1".to_string());
        }
        if self.return_horizon == 0 {
            return Err("return_horizon must be >= 1".to_string());
        }
        if self.return_bins == 0 {
            return Err("return_bins must be >= 1".to_string());
        }
        if !self.return_bins.is_power_of_two() {
            return Err(format!(
                "return_bins must be a power of two for exact binary return encoding, got {}",
                self.return_bins
            ));
        }
        if self.augmentation_period < self.return_horizon {
            return Err(format!(
                "augmentation_period must be >= return_horizon (got N={}, H={})",
                self.augmentation_period, self.return_horizon
            ));
        }
        if !(0.0 < self.discount_gamma && self.discount_gamma < 1.0) {
            return Err(format!(
                "discount_gamma must be in (0, 1) for AIQI as defined in \"A Model-Free Universal AI\", got {}",
                self.discount_gamma
            ));
        }
        if !(0.0 < self.baseline_exploration && self.baseline_exploration <= 1.0) {
            return Err(format!(
                "baseline_exploration (tau) must be in (0, 1] for AIQI as defined in \"A Model-Free Universal AI\", got {}",
                self.baseline_exploration
            ));
        }
        validate_reward_encoding_bounds(
            self.min_reward,
            self.max_reward,
            self.reward_offset,
            self.reward_bits,
        )?;

        // `rate_backend` takes precedence over `algorithm`; only validate
        // algorithm choices when no backend override is configured.
        if self.rate_backend.is_none() {
            match self.algorithm.as_str() {
                "ctw" | "fac-ctw" | "ac-ctw" | "ctw-context-tree" | "rosa" => {}
                "zpaq" => {
                    return Err(
                        "AIQI strict mode does not support algorithm=zpaq: zpaq backends do not provide strict frozen conditioning"
                            .to_string(),
                    )
                }
                #[cfg(feature = "backend-rwkv")]
                "rwkv" => {}
                #[cfg(not(feature = "backend-rwkv"))]
                "rwkv" => {
                    return Err("algorithm=rwkv requires backend-rwkv feature".to_string())
                }
                other => return Err(format!("Unknown AIQI algorithm: {other}")),
            }
        }

        if let Some(rate_backend) = &self.rate_backend {
            validate_rate_backend(rate_backend)
                .map_err(|err| format!("invalid rate_backend: {err}"))?;
            if !rate_backend_supports_aiqi_frozen_conditioning(rate_backend) {
                return Err(
                    "AIQI strict mode requires frozen context updates; configured rate_backend contains zpaq which does not provide strict frozen conditioning"
                        .to_string(),
                );
            }
        }

        #[cfg(feature = "backend-rwkv")]
        if self.rate_backend.is_none() && self.algorithm == "rwkv" {
            match self.rwkv_model_path.as_deref() {
                Some(path) if !path.trim().is_empty() => {}
                _ => {
                    return Err(
                        "algorithm=rwkv requires rwkv_model_path when no rate_backend override is configured; for method-string RWKV configure rate_backend rwkv/rwkv7"
                            .to_string(),
                    )
                }
            }
        }
        Ok(())
    }

    /// Validate configuration constraints.
    pub fn validate(&self) -> Result<(), String> {
        self.validate_runtime_invariants()?;
        self.compile_planner_run_spec().map(|_| ())
    }
}

#[derive(Clone)]
struct AiqiRuntimeConfig {
    observation_bits: usize,
    observation_stream_len: usize,
    reward_bits: usize,
    agent_actions: usize,
    min_reward: Reward,
    max_reward: Reward,
    reward_offset: Reward,
    discount_gamma: f64,
    return_horizon: usize,
    return_bins: usize,
    augmentation_period: usize,
    history_prune_keep_steps: Option<usize>,
    baseline_exploration: f64,
    random_seed: Option<u64>,
}

impl AiqiRuntimeConfig {
    fn from_compiled(compiled: &CompiledPlannerRunSpec) -> Result<Self, String> {
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
            _ => {
                return Err(
                    "compiled planner run does not contain a discounted AIQI controller"
                        .to_string(),
                );
            }
        };

        Ok(Self {
            observation_bits: interface.observation_bits,
            observation_stream_len: interface.observation_stream_len.max(1),
            reward_bits: interface.reward_bits,
            agent_actions: interface.agent_actions,
            min_reward: interface.min_reward,
            max_reward: interface.max_reward,
            reward_offset: interface.reward_offset,
            discount_gamma,
            return_horizon,
            return_bins,
            augmentation_period,
            history_prune_keep_steps,
            baseline_exploration,
            random_seed: runtime.random_seed,
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
    return_bits: usize,
    use_generic_planner: bool,
    distribution_uses_training_updates: bool,
    rng: RandomGenerator,
}

impl AiqiAgent {
    /// Construct a new AIQI agent.
    pub fn new(config: AiqiConfig) -> Result<Self, String> {
        config.validate_runtime_invariants()?;
        let compiled = config.compile_planner_run_spec()?;
        let runtime = AiqiRuntimeConfig::from_compiled(&compiled)?;
        Self::from_compiled_config(runtime, &compiled)
    }

    /// Construct a new AIQI agent directly from a compiled planner-run spec.
    pub fn from_compiled_planner_run(compiled: &CompiledPlannerRunSpec) -> Result<Self, String> {
        let config = AiqiRuntimeConfig::from_compiled(compiled)?;
        Self::from_compiled_config(config, compiled)
    }

    fn from_compiled_config(
        config: AiqiRuntimeConfig,
        compiled: &CompiledPlannerRunSpec,
    ) -> Result<Self, String> {
        let (predictor, predictor_max_order, augmentation_period, return_bins) =
            match compiled.controller() {
                CompiledPlannerController::AiqiDiscounted {
                    predictor,
                    predictor_max_order,
                    augmentation_period,
                    return_bins,
                    ..
                } => (
                    predictor,
                    *predictor_max_order,
                    *augmentation_period,
                    *return_bins,
                ),
                _ => {
                    return Err(
                        "compiled planner run is not a discounted AIQI controller configuration"
                            .to_string(),
                    );
                }
            };
        let action_bits = compiled.action_bits();
        let return_bits = bits_for_cardinality(return_bins);
        let use_generic_planner = aiqi_requires_generic_planner_backend(predictor.canonical_spec());
        let distribution_uses_training_updates = matches!(
            predictor.canonical_spec(),
            RateBackend::Ctw { .. } | RateBackend::FacCtw { .. }
        );

        let mut phases = Vec::with_capacity(augmentation_period);
        for _ in 0..augmentation_period {
            phases.push(PhaseModel {
                predictor: build_aiqi_predictor(predictor, predictor_max_order, return_bits)?,
                last_augmented_step: 0,
            });
        }

        let rng = if let Some(seed) = config.random_seed {
            RandomGenerator::from_seed(seed)
        } else {
            RandomGenerator::new()
        };

        Ok(Self {
            action_bits,
            return_bits,
            use_generic_planner,
            distribution_uses_training_updates,
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

    /// Returns the configured number of actions.
    pub fn num_actions(&self) -> usize {
        self.config.agent_actions
    }

    /// Select the next action from the current history.
    pub fn get_planned_action(&mut self) -> Action {
        let q_values = self.estimate_q_values();
        let greedy_action = argmax_with_fixed_tie_break(&q_values) as u64;
        if self.config.baseline_exploration > 0.0
            && self
                .rng
                .gen_bool(self.config.baseline_exploration.clamp(0.0, 1.0))
        {
            self.rng.gen_range(self.config.agent_actions) as u64
        } else {
            greedy_action
        }
    }

    /// Select the next action, adding optional extra exploration.
    ///
    /// The extra exploration probability is combined as
    /// `p = 1 - (1 - tau) * (1 - extra)`, where `tau` is the baseline
    /// exploration in [`AiqiConfig`].
    pub fn get_planned_action_with_extra_exploration(&mut self, extra_exploration: f64) -> Action {
        let extra = extra_exploration.clamp(0.0, 1.0);
        let tau = self.config.baseline_exploration.clamp(0.0, 1.0);
        let effective = 1.0 - (1.0 - tau) * (1.0 - extra);
        let q_values = self.estimate_q_values();
        let greedy_action = argmax_with_fixed_tie_break(&q_values) as u64;
        if effective > 0.0 && self.rng.gen_bool(effective) {
            self.rng.gen_range(self.config.agent_actions) as u64
        } else {
            greedy_action
        }
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
    ) -> Result<(), String> {
        if action as usize >= self.config.agent_actions {
            return Err(format!(
                "action out of range: action={} but agent_actions={}",
                action, self.config.agent_actions
            ));
        }

        let expected_obs = self.config.observation_stream_len.max(1);
        if observations.len() != expected_obs {
            return Err(format!(
                "observation stream length mismatch: expected {}, got {}",
                expected_obs,
                observations.len()
            ));
        }

        if reward < self.config.min_reward || reward > self.config.max_reward {
            return Err(format!(
                "reward out of configured range: reward={} not in [{}, {}]",
                reward, self.config.min_reward, self.config.max_reward
            ));
        }

        let obs_max = max_value_for_bits(self.config.observation_bits);
        for &obs in observations {
            if obs > obs_max {
                return Err(format!(
                    "observation value {} does not fit observation_bits={} (max={})",
                    obs, self.config.observation_bits, obs_max
                ));
            }
        }

        let rew_shifted = (reward as i128) + (self.config.reward_offset as i128);
        if rew_shifted < 0 {
            return Err(format!(
                "encoded reward became negative after offset: reward={} offset={}",
                reward, self.config.reward_offset
            ));
        }
        if self.config.reward_bits < 64 {
            let max_enc = (1u128 << self.config.reward_bits) - 1;
            if (rew_shifted as u128) > max_enc {
                return Err(format!(
                    "encoded reward {} exceeds reward_bits={} capacity {}",
                    rew_shifted, self.config.reward_bits, max_enc
                ));
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

    fn maybe_learn_new_return(&mut self) -> Result<(), String> {
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
        let return_bits = self.return_bits;

        let mut q_values = vec![0.0; self.config.agent_actions];
        let mut pushed_fast_forward = 0usize;

        {
            let model = &mut self.phases[phase];
            let start = (model.last_augmented_step + 1).max(history_base_step);
            let end = step.saturating_sub(1);
            if start <= end {
                for idx in start..=end {
                    pushed_fast_forward += push_step_tokens_history(
                        config,
                        history_base_step,
                        steps,
                        return_bins_by_step,
                        action_bits,
                        return_bits,
                        model.predictor.as_mut(),
                        phase,
                        idx,
                    );
                }
            }

            for action in 0..self.config.agent_actions {
                let pushed_action = push_encoded_bits_history(
                    model.predictor.as_mut(),
                    action as u64,
                    self.action_bits,
                );
                let dist = Self::predict_return_distribution(
                    self.config.return_bins,
                    self.return_bits,
                    model.predictor.as_mut(),
                    self.distribution_uses_training_updates,
                );
                q_values[action] = expectation_from_distribution(&dist);
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

        let start = (model.last_augmented_step + 1).max(self.history_base_step);
        let end = step.saturating_sub(1);
        if start <= end {
            for idx in start..=end {
                push_augmented_step_tokens_commit(
                    &self.config,
                    self.history_base_step,
                    &self.steps,
                    &self.return_bins_by_step,
                    self.action_bits,
                    self.return_bits,
                    context_predictor.as_mut(),
                    phase,
                    idx,
                )
                .expect("generic planner retained history must contain required augmented return");
            }
        }

        let mut q_values = vec![0.0; self.config.agent_actions];
        for action in 0..self.config.agent_actions {
            let mut action_predictor = context_predictor.boxed_clone();
            let _ = push_encoded_bits_commit_history(
                action_predictor.as_mut(),
                action as u64,
                self.action_bits,
            );
            let dist = Self::predict_return_distribution_from_base_predictor(
                self.config.return_bins,
                self.return_bits,
                action_predictor.as_ref(),
            );
            q_values[action] = expectation_from_distribution(&dist);
        }

        q_values
    }

    fn predict_return_distribution(
        return_bins: usize,
        return_bits: usize,
        predictor: &mut dyn Predictor,
        use_training_updates: bool,
    ) -> Vec<f64> {
        debug_assert!(return_bins.is_power_of_two());
        if return_bins == 1 {
            return vec![1.0];
        }

        let mut probs = vec![0.0; return_bins];
        for (bin, slot) in probs.iter_mut().enumerate() {
            let mut p = 1.0f64;
            let mut v = bin as u64;
            for _ in 0..return_bits {
                let bit = (v & 1) == 1;
                v >>= 1;
                let q = predictor.predict_prob(bit).clamp(1e-12, 1.0 - 1e-12);
                p *= q;
                if use_training_updates {
                    predictor.update(bit);
                } else {
                    predictor.update_history(bit);
                }
            }
            if use_training_updates {
                revert_bits(predictor, return_bits);
            } else {
                pop_history_bits(predictor, return_bits);
            }
            *slot = p;
        }

        let sum: f64 = probs.iter().sum();
        if !sum.is_finite() || sum <= 0.0 {
            let u = 1.0 / (return_bins as f64);
            probs.fill(u);
            return probs;
        }

        for p in &mut probs {
            *p /= sum;
        }
        probs
    }

    fn predict_return_distribution_from_base_predictor(
        return_bins: usize,
        return_bits: usize,
        base_predictor: &dyn Predictor,
    ) -> Vec<f64> {
        debug_assert!(return_bins.is_power_of_two());
        if return_bins == 1 {
            return vec![1.0];
        }

        let mut probs = vec![0.0; return_bins];
        for (bin, slot) in probs.iter_mut().enumerate() {
            let mut predictor = base_predictor.boxed_clone();
            let mut p = 1.0f64;
            let mut v = bin as u64;
            for _ in 0..return_bits {
                let bit = (v & 1) == 1;
                v >>= 1;
                let q = predictor.predict_prob(bit).clamp(1e-12, 1.0 - 1e-12);
                p *= q;
                predictor.commit_update(bit);
            }
            *slot = p;
        }

        let sum: f64 = probs.iter().sum();
        if !sum.is_finite() || sum <= 0.0 {
            let u = 1.0 / (return_bins as f64);
            probs.fill(u);
            return probs;
        }

        for p in &mut probs {
            *p /= sum;
        }
        probs
    }

    fn advance_phase_model_to_step(
        &mut self,
        phase: usize,
        target_step: usize,
    ) -> Result<(), String> {
        let config = &self.config;
        let steps = &self.steps;
        let return_bins_by_step = &self.return_bins_by_step;
        let history_base_step = self.history_base_step;
        let action_bits = self.action_bits;
        let return_bits = self.return_bits;
        let model = &mut self.phases[phase];
        if target_step <= model.last_augmented_step {
            return Ok(());
        }

        let start = (model.last_augmented_step + 1).max(history_base_step);
        for idx in start..=target_step {
            push_augmented_step_tokens_commit(
                config,
                history_base_step,
                steps,
                return_bins_by_step,
                action_bits,
                return_bits,
                model.predictor.as_mut(),
                phase,
                idx,
            )?;
        }

        model.last_augmented_step = target_step;
        Ok(())
    }

    fn compute_return_bin(&self, start_step: usize) -> u64 {
        let h = self.config.return_horizon;
        let gamma = self.config.discount_gamma;

        debug_assert!(gamma > 0.0 && gamma < 1.0);
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

    fn local_index(&self, global_step: usize) -> Result<usize, String> {
        if global_step < self.history_base_step || global_step > self.total_steps_observed {
            return Err(format!(
                "global step {} out of retained history range [{}, {}]",
                global_step, self.history_base_step, self.total_steps_observed
            ));
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

fn push_step_tokens_history(
    config: &AiqiRuntimeConfig,
    history_base_step: usize,
    steps: &[StepRecord],
    return_bins_by_step: &[Option<u64>],
    action_bits: usize,
    return_bits: usize,
    predictor: &mut dyn Predictor,
    phase: usize,
    idx: usize,
) -> usize {
    let mut pushed = 0usize;
    pushed += push_action_tokens_history(history_base_step, steps, action_bits, predictor, idx);

    if idx % config.augmentation_period == phase {
        let local_idx = idx - history_base_step;
        if let Some(bin) = return_bins_by_step[local_idx] {
            pushed += push_encoded_bits_history(predictor, bin, return_bits);
        }
    }

    pushed + push_percept_tokens_history(config, history_base_step, steps, predictor, idx)
}

fn push_augmented_step_tokens_commit(
    config: &AiqiRuntimeConfig,
    history_base_step: usize,
    steps: &[StepRecord],
    return_bins_by_step: &[Option<u64>],
    action_bits: usize,
    return_bits: usize,
    predictor: &mut dyn Predictor,
    phase: usize,
    idx: usize,
) -> Result<usize, String> {
    let mut pushed = 0usize;
    pushed +=
        push_action_tokens_commit_history(history_base_step, steps, action_bits, predictor, idx);

    if idx % config.augmentation_period == phase {
        let local_idx = idx - history_base_step;
        let bin = return_bins_by_step[local_idx].ok_or_else(|| {
            format!(
                "missing return bin for step {} in phase {} while pushing augmented history",
                idx, phase
            )
        })?;
        pushed += push_encoded_bits_commit(predictor, bin, return_bits);
    }

    Ok(pushed
        + push_percept_tokens_commit_history(config, history_base_step, steps, predictor, idx))
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

fn aiqi_requires_generic_planner_backend(backend: &RateBackend) -> bool {
    !matches!(
        backend,
        RateBackend::Ctw { .. } | RateBackend::FacCtw { .. }
    )
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

fn push_encoded_bits_commit(predictor: &mut dyn Predictor, value: u64, bits: usize) -> usize {
    let mut v = value;
    for _ in 0..bits {
        predictor.commit_update((v & 1) == 1);
        v >>= 1;
    }
    bits
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

fn revert_bits(predictor: &mut dyn Predictor, bits: usize) {
    for _ in 0..bits {
        predictor.revert();
    }
}

fn expectation_from_distribution(probs: &[f64]) -> f64 {
    if probs.is_empty() {
        return 0.0;
    }
    let m = probs.len() as f64;
    probs
        .iter()
        .enumerate()
        .map(|(i, p)| (i as f64 / m) * p)
        .sum::<f64>()
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
    use crate::aixi::test_envs::DeterministicBinaryEnv;
    use crate::api::{MixtureKind, MixtureSpec};
    use std::sync::{Arc, Mutex};

    fn basic_config() -> AiqiConfig {
        AiqiConfig {
            algorithm: "ac-ctw".to_string(),
            ct_depth: 8,
            observation_bits: 1,
            observation_stream_len: 1,
            reward_bits: 1,
            agent_actions: 2,
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
            rate_backend: None,
            rate_backend_max_order: 20,
            rwkv_model_path: None,
            rosa_max_order: None,
            zpaq_method: None,
        }
    }

    fn generic_mixture_config() -> AiqiConfig {
        AiqiConfig {
            rate_backend: Some(RateBackend::Mixture {
                spec: Arc::new(
                    MixtureSpec::new(
                        MixtureKind::Bayes,
                        vec![
                            crate::api::MixtureExpertSpec {
                                name: Some("ctw".to_string()),
                                log_prior: 0.0,
                                max_order: -1,
                                backend: RateBackend::Ctw { depth: 8 },
                            },
                            crate::api::MixtureExpertSpec {
                                name: Some("match".to_string()),
                                log_prior: 0.0,
                                max_order: -1,
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
            }),
            random_seed: Some(11),
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
    }

    impl Predictor for ReturnLearningPredictor {
        fn update(&mut self, sym: bool) {
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

        fn revert(&mut self) {}

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

    #[test]
    fn config_rejects_invalid_period() {
        let mut cfg = basic_config();
        cfg.augmentation_period = 1;
        cfg.return_horizon = 2;
        let err = cfg
            .validate()
            .expect_err("N < H must be rejected to match \"A Model-Free Universal AI\"");
        assert!(err.contains("augmentation_period"));
    }

    #[test]
    fn config_rejects_non_power_of_two_return_bins() {
        let mut cfg = basic_config();
        cfg.return_bins = 3;
        let err = cfg
            .validate()
            .expect_err("non-power-of-two return_bins should be rejected");
        assert!(err.contains("power of two"));
    }

    #[test]
    fn config_rejects_zpaq_algorithm_in_strict_mode() {
        let mut cfg = basic_config();
        cfg.algorithm = "zpaq".to_string();
        let err = cfg
            .validate()
            .expect_err("strict AIQI must reject zpaq algorithm mode");
        assert!(err.contains("strict mode"));
    }

    #[test]
    fn config_rejects_zpaq_rate_backend_in_strict_mode() {
        let mut cfg = basic_config();
        cfg.rate_backend = Some(RateBackend::Zpaq {
            method: crate::api::ZpaqMethodSpec::literal("1"),
        });
        let err = cfg
            .validate()
            .expect_err("strict AIQI must reject zpaq rate backend");
        assert!(err.contains("strict frozen conditioning"));
    }

    #[test]
    fn config_rejects_nonpaper_gamma_or_tau() {
        let mut cfg = basic_config();
        cfg.discount_gamma = 1.0;
        let err = cfg
            .validate()
            .expect_err("gamma=1 must be rejected for strict paper AIQI");
        assert!(err.contains("discount_gamma"));

        cfg = basic_config();
        cfg.baseline_exploration = 0.0;
        let err = cfg
            .validate()
            .expect_err("tau=0 must be rejected for strict paper AIQI");
        assert!(err.contains("baseline_exploration"));
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
        cfg.algorithm = "fac-ctw".to_string();
        cfg.return_bins = 8; // return_bits=3

        let agent = AiqiAgent::new(cfg).expect("valid aiqi config");
        let name = agent.phases[0].predictor.model_name();
        assert!(
            name.contains("k=3"),
            "FAC-CTW should factorize over return bits only, model_name={name}"
        );
    }

    #[test]
    fn ac_ctw_path_uses_single_tree_predictor() {
        let mut cfg = basic_config();
        cfg.algorithm = "ac-ctw".to_string();

        let agent = AiqiAgent::new(cfg).expect("valid aiqi config");
        let name = agent.phases[0].predictor.model_name();
        assert!(
            name.starts_with("AC-CTW"),
            "ac-ctw should map to the single-tree CTW predictor, model_name={name}"
        );
    }

    #[test]
    fn ctw_alias_matches_ac_ctw_predictor() {
        let mut cfg = basic_config();
        cfg.algorithm = "ctw".to_string();

        let agent = AiqiAgent::new(cfg).expect("valid aiqi config");
        let name = agent.phases[0].predictor.model_name();
        assert!(
            name.starts_with("AC-CTW"),
            "ctw alias should map to paper AIQI-CTW predictor, model_name={name}"
        );
    }

    #[test]
    fn distribution_rollout_uses_update_and_revert_when_requested() {
        let mut predictor = CountingPredictor::default();
        let probs = AiqiAgent::predict_return_distribution(4, 2, &mut predictor, true);

        assert_eq!(probs.len(), 4);
        assert_eq!(predictor.update_calls, 8);
        assert_eq!(predictor.revert_calls, 8);
        assert_eq!(predictor.update_history_calls, 0);
        assert_eq!(predictor.pop_history_calls, 0);
    }

    #[test]
    fn distribution_rollout_uses_history_path_when_not_requested() {
        let mut predictor = CountingPredictor::default();
        let probs = AiqiAgent::predict_return_distribution(4, 2, &mut predictor, false);

        assert_eq!(probs.len(), 4);
        assert_eq!(predictor.update_calls, 0);
        assert_eq!(predictor.revert_calls, 0);
        assert_eq!(predictor.update_history_calls, 8);
        assert_eq!(predictor.pop_history_calls, 8);
    }

    #[test]
    fn generic_distribution_rollout_trains_on_return_symbols() {
        let predictor = ReturnLearningPredictor::default();
        let probs = AiqiAgent::predict_return_distribution_from_base_predictor(4, 2, &predictor);

        assert_eq!(probs.len(), 4);
        assert!((probs.iter().sum::<f64>() - 1.0).abs() < 1e-12);
        assert!(
            probs[3] > probs[1],
            "training on the first return bit should make bin 11 likelier than 01; got {:?}",
            probs
        );
        assert!(
            (probs[0] - 0.5625).abs() < 1e-12,
            "expected exact normalized mass for 00, got {:?}",
            probs
        );
    }

    #[test]
    fn ac_ctw_rollout_uses_training_updates() {
        let mut cfg = basic_config();
        cfg.algorithm = "ac-ctw".to_string();

        let agent = AiqiAgent::new(cfg).expect("valid aiqi config");
        assert!(
            agent.distribution_uses_training_updates,
            "ac-ctw should use update/revert during return distribution rollout"
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
        cfg.rate_backend = Some(RateBackend::Match {
            hash_bits: 16,
            min_len: 2,
            max_len: 16,
            base_mix: 0.05,
            confidence_scale: 1.0,
        });

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

        assert_eq!(q_values.len(), agent.config.agent_actions);
        let snapshot = counts.lock().unwrap().clone();
        assert_eq!(snapshot.update, 0);
        assert_eq!(snapshot.update_history, 0);
        assert!(
            snapshot.commit_update > 0,
            "generic planner should train on augmented return symbols"
        );
        assert!(
            snapshot.commit_update_history > 0,
            "generic planner should keep action/percept conditioning frozen"
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
