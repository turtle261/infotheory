//! AIQI (Universal AI with Q-Induction) implementation.
//!
//! This module implements a model-free universal agent that predicts
//! discretized H-step returns directly from augmented interaction history.
//! The implementation follows the paper's phase-indexed periodic augmentation:
//! for return horizon `H` and period `N >= H`, each phase model only inserts
//! returns at indices `i % N == phase`.

use crate::aixi::common::{Action, PerceptVal, RandomGenerator, Reward};
use crate::aixi::model::{CtwPredictor, FacCtwPredictor, Predictor, RateBackendBitPredictor};
#[cfg(feature = "backend-rwkv")]
use crate::load_rwkv7_model_from_path;
use crate::{CalibratedSpec, MixtureSpec, RateBackend};
use std::sync::Arc;

/// Configuration parameters for an AIQI agent.
#[derive(Clone)]
pub struct AiqiConfig {
    /// Predictive backend.
    ///
    /// - `ac-ctw` / `ctw` / `ctw-context-tree`: paper AIQI-CTW path.
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
    /// Validate configuration constraints.
    pub fn validate(&self) -> Result<(), String> {
        if self.agent_actions == 0 {
            return Err("agent_actions must be >= 1".to_string());
        }
        if self.return_horizon == 0 {
            return Err("return_horizon must be >= 1".to_string());
        }
        if self.return_bins == 0 {
            return Err("return_bins must be >= 1".to_string());
        }
        if self.augmentation_period < self.return_horizon {
            return Err(format!(
                "augmentation_period must be >= return_horizon (got N={}, H={})",
                self.augmentation_period, self.return_horizon
            ));
        }
        if !(0.0 < self.discount_gamma && self.discount_gamma < 1.0) {
            return Err(format!(
                "discount_gamma must be in (0, 1) for paper-correct AIQI, got {}",
                self.discount_gamma
            ));
        }
        if !(0.0 < self.baseline_exploration && self.baseline_exploration <= 1.0) {
            return Err(format!(
                "baseline_exploration (tau) must be in (0, 1] for paper-correct AIQI, got {}",
                self.baseline_exploration
            ));
        }
        if self.max_reward < self.min_reward {
            return Err(format!(
                "max_reward must be >= min_reward (got {} < {})",
                self.max_reward, self.min_reward
            ));
        }

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
            "rwkv" => return Err("algorithm=rwkv requires backend-rwkv feature".to_string()),
            other => return Err(format!("Unknown AIQI algorithm: {other}")),
        }

        if let Some(rate_backend) = &self.rate_backend {
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

        let min_shifted = (self.min_reward as i128) + (self.reward_offset as i128);
        let max_shifted = (self.max_reward as i128) + (self.reward_offset as i128);
        if min_shifted < 0 {
            return Err(format!(
                "reward_offset too small: min_reward + reward_offset must be >= 0 (got {})",
                min_shifted
            ));
        }
        if self.reward_bits < 64 {
            let max_enc = (1u128 << self.reward_bits) - 1;
            if (max_shifted as u128) > max_enc {
                return Err(format!(
                    "reward_bits too small for configured reward range: max shifted reward {} exceeds {}",
                    max_shifted, max_enc
                ));
            }
        }

        Ok(())
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
    config: AiqiConfig,
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
        config.validate()?;

        let action_bits = bits_for_cardinality(config.agent_actions);
        let return_bits = bits_for_cardinality(config.return_bins);
        let use_generic_planner = aiqi_requires_generic_planner(&config);
        let distribution_uses_training_updates = config.rate_backend.is_none()
            && matches!(
                config.algorithm.as_str(),
                "ctw" | "fac-ctw" | "ac-ctw" | "ctw-context-tree"
            );

        let mut phases = Vec::with_capacity(config.augmentation_period);
        for _ in 0..config.augmentation_period {
            phases.push(PhaseModel {
                predictor: build_predictor(&config, return_bits)?,
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
                push_step_tokens_history(
                    &self.config,
                    self.history_base_step,
                    &self.steps,
                    &self.return_bins_by_step,
                    self.action_bits,
                    self.return_bits,
                    context_predictor.as_mut(),
                    phase,
                    idx,
                );
            }
        }

        let mut q_values = vec![0.0; self.config.agent_actions];
        for action in 0..self.config.agent_actions {
            let mut action_predictor = context_predictor.boxed_clone();
            let _ = push_encoded_bits_history(
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
                predictor.update_history(bit);
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
            push_action_tokens_history(
                history_base_step,
                steps,
                action_bits,
                model.predictor.as_mut(),
                idx,
            );

            if idx % config.augmentation_period == phase {
                let local_idx = idx - history_base_step;
                let bin = return_bins_by_step[local_idx].ok_or_else(|| {
                    format!(
                        "missing return bin for step {} in phase {} while advancing model",
                        idx, phase
                    )
                })?;
                push_encoded_bits_train(model.predictor.as_mut(), bin, return_bits);
            }

            push_percept_tokens_history(
                config,
                history_base_step,
                steps,
                model.predictor.as_mut(),
                idx,
            );
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
    config: &AiqiConfig,
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

fn push_percept_tokens_history(
    config: &AiqiConfig,
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

fn build_predictor(config: &AiqiConfig, return_bits: usize) -> Result<Box<dyn Predictor>, String> {
    if let Some(rate_backend) = config.rate_backend.clone() {
        let bit_backend = adapt_rate_backend_for_bit_tokens(rate_backend);
        let predictor = RateBackendBitPredictor::new(bit_backend, config.rate_backend_max_order)?;
        return Ok(Box::new(predictor));
    }

    match config.algorithm.as_str() {
        "ctw" | "ac-ctw" | "ctw-context-tree" => Ok(Box::new(CtwPredictor::new(config.ct_depth))),
        "fac-ctw" => {
            // AIQI-FAC-CTW extension: factorized return-bit modeling.
            Ok(Box::new(FacCtwPredictor::new(config.ct_depth, return_bits)))
        }
        "rosa" => {
            let max_order = config
                .rosa_max_order
                .unwrap_or(config.rate_backend_max_order);
            let bit_backend = adapt_rate_backend_for_bit_tokens(RateBackend::RosaPlus);
            let predictor = RateBackendBitPredictor::new(bit_backend, max_order)?;
            Ok(Box::new(predictor))
        }
        #[cfg(feature = "backend-rwkv")]
        "rwkv" => {
            let path = config.rwkv_model_path.as_ref().ok_or_else(|| {
                "algorithm=rwkv requires rwkv_model_path when no rate_backend override is configured; for method-string RWKV configure rate_backend rwkv/rwkv7"
                    .to_string()
            })?;
            let model_arc = load_rwkv7_model_from_path(path);
            let bit_backend =
                adapt_rate_backend_for_bit_tokens(RateBackend::Rwkv7 { model: model_arc });
            let predictor = RateBackendBitPredictor::new(bit_backend, config.rate_backend_max_order)?;
            Ok(Box::new(predictor))
        }
        #[cfg(not(feature = "backend-rwkv"))]
        "rwkv" => Err("algorithm=rwkv requires backend-rwkv feature".to_string()),
        "zpaq" => Err(
            "AIQI strict mode does not support algorithm=zpaq; configure a backend with strict frozen conditioning"
                .to_string(),
        ),
        _ => Err(format!("Unknown AIQI algorithm: {}", config.algorithm)),
    }
}

fn adapt_rate_backend_for_bit_tokens(backend: RateBackend) -> RateBackend {
    match backend {
        RateBackend::Ctw { depth } => RateBackend::FacCtw {
            base_depth: depth,
            num_percept_bits: 1,
            encoding_bits: 1,
        },
        RateBackend::FacCtw { base_depth, .. } => RateBackend::FacCtw {
            base_depth,
            num_percept_bits: 1,
            encoding_bits: 1,
        },
        RateBackend::Mixture { spec } => {
            let experts = spec
                .experts
                .iter()
                .map(|expert| crate::MixtureExpertSpec {
                    name: expert.name.clone(),
                    log_prior: expert.log_prior,
                    max_order: expert.max_order,
                    backend: adapt_rate_backend_for_bit_tokens(expert.backend.clone()),
                })
                .collect();

            let mut adapted = MixtureSpec::new(spec.kind, experts).with_alpha(spec.alpha);
            if let Some(decay) = spec.decay {
                adapted = adapted.with_decay(decay);
            }
            RateBackend::Mixture {
                spec: Arc::new(adapted),
            }
        }
        RateBackend::Calibrated { spec } => RateBackend::Calibrated {
            spec: Arc::new(CalibratedSpec {
                base: adapt_rate_backend_for_bit_tokens(spec.base.clone()),
                context: spec.context,
                bins: spec.bins,
                learning_rate: spec.learning_rate,
                bias_clip: spec.bias_clip,
            }),
        },
        other => other,
    }
}

fn rate_backend_supports_aiqi_frozen_conditioning(backend: &RateBackend) -> bool {
    match backend {
        RateBackend::Zpaq { .. } => false,
        RateBackend::Mixture { spec } => spec
            .experts
            .iter()
            .all(|expert| rate_backend_supports_aiqi_frozen_conditioning(&expert.backend)),
        RateBackend::Calibrated { spec } => {
            rate_backend_supports_aiqi_frozen_conditioning(&spec.base)
        }
        _ => true,
    }
}

fn aiqi_requires_generic_planner(config: &AiqiConfig) -> bool {
    config.rate_backend.is_some()
        || !matches!(
            config.algorithm.as_str(),
            "ctw" | "fac-ctw" | "ac-ctw" | "ctw-context-tree"
        )
}

fn bits_for_cardinality(cardinality: usize) -> usize {
    let n = cardinality.max(1);
    let mut bits = 0usize;
    while (1usize << bits) < n {
        bits += 1;
    }
    bits.max(1)
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

fn push_encoded_bits_train(predictor: &mut dyn Predictor, value: u64, bits: usize) -> usize {
    let mut v = value;
    for _ in 0..bits {
        predictor.update((v & 1) == 1);
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

#[cfg(test)]
mod tests {
    use super::*;

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

    #[derive(Clone, Default)]
    struct CountingPredictor {
        update_calls: usize,
        update_history_calls: usize,
        revert_calls: usize,
        pop_history_calls: usize,
    }

    impl Predictor for CountingPredictor {
        fn update(&mut self, _sym: bool) {
            self.update_calls += 1;
        }

        fn update_history(&mut self, _sym: bool) {
            self.update_history_calls += 1;
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

    #[test]
    fn config_rejects_invalid_period() {
        let mut cfg = basic_config();
        cfg.augmentation_period = 1;
        cfg.return_horizon = 2;
        let err = cfg
            .validate()
            .expect_err("N < H must be rejected for paper-correct augmentation");
        assert!(err.contains("augmentation_period"));
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
            method: "1".to_string(),
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
}
