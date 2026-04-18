//! The core AIXI agent implementation.
//!
//! This module defines the `Agent` struct, which ties together a world model
//! (Predictor) and a planner (SearchTree) to form a complete autonomous entity.

use crate::aixi::common::{
    Action, ObservationKeyMode, PerceptVal, RandomGenerator, Reward, decode, encode,
    observation_repr_from_stream, validate_reward_encoding_bounds,
};
use crate::aixi::mcts::{AgentSimulator, SearchTree};
use crate::aixi::model::{Predictor, build_mc_aixi_predictor};
use crate::aixi::planner_spec::{PlannerInterfaceConfig, build_coin_flip_planner_run_spec};
use crate::api::{RateBackend, validate_rate_backend};
use crate::spec::{
    CompiledPlannerController, CompiledPlannerRunSpec, ControllerSpec, McAixiControllerSpec,
    PlannerRunSpec,
};
use crate::validate_zpaq_rate_method;
#[cfg(any(feature = "backend-mamba", feature = "backend-rwkv"))]
use std::path::PathBuf;

/// Configuration parameters for an AIXI agent.
#[derive(Clone)]
pub struct AgentConfig {
    /// The predictive algorithm to use ("ctw", "rosa", "rwkv", "mamba", "zpaq").
    pub algorithm: String,
    /// Context depth for the CTW model.
    pub ct_depth: usize,
    /// Planning horizon for MCTS.
    pub agent_horizon: usize,
    /// Number of bits used to encode observations.
    pub observation_bits: usize,
    /// Number of observation symbols per action (stream length).
    pub observation_stream_len: usize,
    /// Strategy for mapping observation streams into search keys.
    pub observation_key_mode: ObservationKeyMode,
    /// Number of bits used to encode rewards.
    pub reward_bits: usize,
    /// Number of possible actions.
    pub agent_actions: usize,
    /// Number of MCTS simulations per planning step.
    pub num_simulations: usize,
    /// Constant governing exploration vs exploitation in UCT.
    pub exploration_exploitation_ratio: f64,
    /// Discount factor for future rewards (1.0 = undiscounted).
    pub discount_gamma: f64,
    /// Minimum possible instantaneous reward in the environment.
    pub min_reward: Reward,
    /// Maximum possible instantaneous reward in the environment.
    pub max_reward: Reward,
    /// Reward offset applied before encoding rewards as unsigned bits.
    ///
    /// Paper-compatible encoding shifts rewards by an offset so all encoded values are non-negative.
    pub reward_offset: Reward,
    /// Optional deterministic RNG seed for planning/simulation behavior.
    ///
    /// When `None`, a fresh runtime-derived seed is used.
    pub random_seed: Option<u64>,
    /// Optional generic rate backend override.
    ///
    /// When set, this takes precedence over `algorithm` and routes MC-AIXI
    /// through the shared `RateBackend` abstraction.
    pub rate_backend: Option<RateBackend>,
    /// Max-order hint for `rate_backend` constructors that use it (for example ROSA).
    pub rate_backend_max_order: i64,
    /// Path to the RWKV model weights (if using "rwkv").
    pub rwkv_model_path: Option<String>,
    /// Optional RWKV method string for hosted/browser-safe construction.
    pub rwkv_method: Option<String>,
    /// Path to the Mamba model weights (if using "mamba").
    pub mamba_model_path: Option<String>,
    /// Optional Mamba method string for hosted/browser-safe construction.
    pub mamba_method: Option<String>,
    /// Maximum Markov order for the ROSA model (if using "rosa").
    pub rosa_max_order: Option<i64>,
    /// ZPAQ method string for the rate model (if using "zpaq").
    pub zpaq_method: Option<String>,
}

impl AgentConfig {
    fn canonical_predictor_backend(&self) -> Result<RateBackend, String> {
        if let Some(rate_backend) = &self.rate_backend {
            return Ok(rate_backend.clone());
        }

        match self.algorithm.as_str() {
            "ctw" | "fac-ctw" => Ok(RateBackend::FacCtw {
                base_depth: self.ct_depth,
                num_percept_bits: (self.observation_bits * self.observation_stream_len.max(1))
                    + self.reward_bits,
                encoding_bits: 1,
            }),
            "ac-ctw" | "ctw-context-tree" => Ok(RateBackend::Ctw {
                depth: self.ct_depth,
            }),
            "rosa" => Ok(RateBackend::RosaPlus),
            #[cfg(feature = "backend-rwkv")]
            "rwkv" => {
                if let Some(method) = self
                    .rwkv_method
                    .as_deref()
                    .map(str::trim)
                    .filter(|value| !value.is_empty())
                {
                    Ok(RateBackend::Rwkv7Method {
                        method: crate::rwkvzip::parse_method_spec(method)
                            .map_err(|err| format!("Invalid RWKV method for AIXI: {err}"))?,
                    })
                } else {
                    let path = self.rwkv_model_path.as_ref().ok_or_else(|| {
                        "algorithm=rwkv requires rwkv_model_path or rwkv_method when no rate_backend override is configured"
                            .to_string()
                    })?;
                    Ok(RateBackend::Rwkv7Method {
                        method: crate::rwkvzip::MethodSpec::File {
                            path: PathBuf::from(path),
                            policy: None,
                        },
                    })
                }
            }
            #[cfg(not(feature = "backend-rwkv"))]
            "rwkv" => Err("algorithm=rwkv requires backend-rwkv feature".to_string()),
            #[cfg(feature = "backend-mamba")]
            "mamba" => {
                if let Some(method) = self
                    .mamba_method
                    .as_deref()
                    .map(str::trim)
                    .filter(|value| !value.is_empty())
                {
                    Ok(RateBackend::MambaMethod {
                        method: crate::mambazip::parse_method_spec(method)
                            .map_err(|err| format!("Invalid Mamba method for AIXI: {err}"))?,
                    })
                } else {
                    let path = self.mamba_model_path.as_ref().ok_or_else(|| {
                        "algorithm=mamba requires mamba_model_path or mamba_method when no rate_backend override is configured"
                            .to_string()
                    })?;
                    Ok(RateBackend::MambaMethod {
                        method: crate::mambazip::MethodSpec::File {
                            path: PathBuf::from(path),
                            policy: None,
                        },
                    })
                }
            }
            #[cfg(not(feature = "backend-mamba"))]
            "mamba" => Err("algorithm=mamba requires backend-mamba feature".to_string()),
            "zpaq" => Ok(RateBackend::Zpaq {
                method: crate::api::ZpaqMethodSpec::literal(
                    self.zpaq_method.clone().unwrap_or_else(|| "1".to_string()),
                ),
            }),
            other => Err(format!("Unknown algorithm: {other}")),
        }
    }

    fn canonical_planner_run_spec(&self) -> Result<PlannerRunSpec, String> {
        let predictor = self.canonical_predictor_backend()?;
        Ok(build_coin_flip_planner_run_spec(
            PlannerInterfaceConfig {
                observation_bits: self.observation_bits,
                observation_stream_len: self.observation_stream_len,
                observation_key_mode: self.observation_key_mode,
                reward_bits: self.reward_bits,
                agent_actions: self.agent_actions,
                min_reward: self.min_reward,
                max_reward: self.max_reward,
                reward_offset: self.reward_offset,
            },
            ControllerSpec::McAixi(McAixiControllerSpec {
                predictor,
                predictor_max_order: self.rate_backend_max_order,
                agent_horizon: self.agent_horizon,
                num_simulations: self.num_simulations,
                exploration_exploitation_ratio: self.exploration_exploitation_ratio,
                discount_gamma: self.discount_gamma,
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
        if self.agent_horizon == 0 {
            return Err("agent_horizon must be >= 1".to_string());
        }
        if self.num_simulations == 0 {
            return Err("num_simulations must be >= 1".to_string());
        }
        if self.exploration_exploitation_ratio <= 0.0 {
            return Err("exploration_exploitation_ratio must be > 0".to_string());
        }
        if !(0.0..=1.0).contains(&self.discount_gamma) {
            return Err(format!(
                "discount_gamma must be in [0, 1] for MC-AIXI, got {}",
                self.discount_gamma
            ));
        }
        validate_reward_encoding_bounds(
            self.min_reward,
            self.max_reward,
            self.reward_offset,
            self.reward_bits,
        )?;

        if let Some(rate_backend) = &self.rate_backend {
            validate_rate_backend(rate_backend)
                .map_err(|err| format!("invalid rate_backend: {err}"))?;
            let compiled = rate_backend.compile().map_err(|err| err.to_string())?;
            if compiled.contains_zpaq() {
                return Err(
                    "MC-AIXI strict generic rate_backend support requires reversible action conditioning; configured rate_backend contains zpaq which does not provide the reversible action conditioning required by \"A Monte-Carlo AIXI Approximation\""
                        .to_string(),
                );
            }
            return Ok(());
        }

        match self.algorithm.as_str() {
            "ctw" | "fac-ctw" | "ac-ctw" | "ctw-context-tree" | "rosa" => {}
            #[cfg(feature = "backend-rwkv")]
            "rwkv" => {
                let has_method = self
                    .rwkv_method
                    .as_deref()
                    .map(str::trim)
                    .is_some_and(|v| !v.is_empty());
                let has_path = self
                    .rwkv_model_path
                    .as_deref()
                    .map(str::trim)
                    .is_some_and(|v| !v.is_empty());
                if !(has_method || has_path) {
                    return Err(
                        "algorithm=rwkv requires rwkv_model_path or rwkv_method when no rate_backend override is configured"
                            .to_string(),
                    );
                }
            }
            #[cfg(not(feature = "backend-rwkv"))]
            "rwkv" => return Err("algorithm=rwkv requires backend-rwkv feature".to_string()),
            #[cfg(feature = "backend-mamba")]
            "mamba" => {
                let has_method = self
                    .mamba_method
                    .as_deref()
                    .map(str::trim)
                    .is_some_and(|v| !v.is_empty());
                let has_path = self
                    .mamba_model_path
                    .as_deref()
                    .map(str::trim)
                    .is_some_and(|v| !v.is_empty());
                if !(has_method || has_path) {
                    return Err(
                        "algorithm=mamba requires mamba_model_path or mamba_method when no rate_backend override is configured"
                            .to_string(),
                    );
                }
            }
            #[cfg(not(feature = "backend-mamba"))]
            "mamba" => return Err("algorithm=mamba requires backend-mamba feature".to_string()),
            "zpaq" => {
                let method = self.zpaq_method.as_deref().unwrap_or("1");
                if let Err(err) = validate_zpaq_rate_method(method) {
                    return Err(format!("Invalid zpaq method for AIXI: {err}"));
                }
            }
            other => return Err(format!("Unknown algorithm: {other}")),
        }

        Ok(())
    }

    /// Validate configuration constraints for MC-AIXI.
    pub fn validate(&self) -> Result<(), String> {
        self.validate_runtime_invariants()?;
        self.compile_planner_run_spec().map(|_| ())
    }
}

#[derive(Clone)]
struct AgentRuntimeConfig {
    agent_horizon: usize,
    observation_bits: usize,
    observation_stream_len: usize,
    observation_key_mode: ObservationKeyMode,
    reward_bits: usize,
    agent_actions: usize,
    num_simulations: usize,
    exploration_exploitation_ratio: f64,
    discount_gamma: f64,
    min_reward: Reward,
    max_reward: Reward,
    reward_offset: Reward,
    random_seed: Option<u64>,
}

impl AgentRuntimeConfig {
    fn from_compiled(compiled: &CompiledPlannerRunSpec) -> Result<Self, String> {
        let interface = compiled.interface();
        let runtime = compiled.runtime();
        let (agent_horizon, num_simulations, exploration_exploitation_ratio, discount_gamma) =
            match compiled.controller() {
                CompiledPlannerController::McAixi {
                    agent_horizon,
                    num_simulations,
                    exploration_exploitation_ratio,
                    discount_gamma,
                    ..
                } => (
                    *agent_horizon,
                    *num_simulations,
                    *exploration_exploitation_ratio,
                    *discount_gamma,
                ),
                _ => {
                    return Err(
                        "compiled planner run does not contain an MC-AIXI controller".to_string(),
                    );
                }
            };

        Ok(Self {
            agent_horizon,
            observation_bits: interface.observation_bits,
            observation_stream_len: interface.observation_stream_len.max(1),
            observation_key_mode: interface.observation_key_mode,
            reward_bits: interface.reward_bits,
            agent_actions: interface.agent_actions,
            num_simulations,
            exploration_exploitation_ratio,
            discount_gamma,
            min_reward: interface.min_reward,
            max_reward: interface.max_reward,
            reward_offset: interface.reward_offset,
            random_seed: runtime.random_seed,
        })
    }
}

/// A complete MC-AIXI agent.
///
/// The agent maintains an internal world model and a planning tree. It can
/// be used for both live interaction with an environment and for
/// "imaginary" simulations during planning.
pub struct Agent {
    /// The world model used for prediction.
    model: Box<dyn Predictor>,
    /// The MCTS planner, temporarily taken during search.
    planner: Option<SearchTree>,
    /// Configuration settings.
    config: AgentRuntimeConfig,

    /// Total number of interaction cycles.
    age: u64,
    /// Accumulated reward.
    total_reward: f64,

    /// Pre-calculated bit depth for actions based on `agent_actions`.
    action_bits: usize,

    /// Internal PRNG for simulations.
    rng: RandomGenerator,

    /// Recycled buffer for observation generation during planning.
    obs_buffer: Vec<u64>,
    /// Recycled buffer for symbol processing.
    sym_buffer: Vec<bool>,
}

impl Agent {
    /// Creates a new `Agent` with the given configuration.
    pub fn new(config: AgentConfig) -> Self {
        Self::try_new(config).unwrap_or_else(|err| panic!("Invalid MC-AIXI config: {err}"))
    }

    /// Creates a new `Agent` with the given configuration, returning a validation error on failure.
    pub fn try_new(config: AgentConfig) -> Result<Self, String> {
        config.validate_runtime_invariants()?;
        let compiled = config.compile_planner_run_spec()?;
        let runtime = AgentRuntimeConfig::from_compiled(&compiled)?;
        Self::from_compiled_config(runtime, &compiled)
    }

    /// Creates a new `Agent` from a compiled planner-run spec.
    pub fn from_compiled_planner_run(compiled: &CompiledPlannerRunSpec) -> Result<Self, String> {
        let config = AgentRuntimeConfig::from_compiled(compiled)?;
        Self::from_compiled_config(config, compiled)
    }

    fn from_compiled_config(
        config: AgentRuntimeConfig,
        compiled: &CompiledPlannerRunSpec,
    ) -> Result<Self, String> {
        let (predictor, predictor_max_order) = match compiled.controller() {
            CompiledPlannerController::McAixi {
                predictor,
                predictor_max_order,
                ..
            } => (predictor, *predictor_max_order),
            _ => {
                return Err(
                    "compiled planner run is not an MC-AIXI controller configuration".to_string(),
                );
            }
        };
        let percept_bits = (compiled.interface().observation_bits
            * compiled.interface().observation_stream_len.max(1))
            + compiled.interface().reward_bits;
        let model = build_mc_aixi_predictor(predictor, predictor_max_order, percept_bits)?;

        let rng = if let Some(seed) = config.random_seed {
            RandomGenerator::from_seed(seed)
        } else {
            RandomGenerator::new()
        };

        Ok(Self {
            model,
            planner: Some(SearchTree::new()),
            config,
            age: 0,
            total_reward: 0.0,
            action_bits: compiled.action_bits(),
            rng,
            obs_buffer: Vec::with_capacity(128),
            sym_buffer: Vec::with_capacity(64),
        })
    }

    fn clone_for_simulation(&self, seed: u64) -> Self {
        Self {
            model: self.model.boxed_clone(),
            planner: None,
            config: self.config.clone(),
            age: self.age,
            total_reward: self.total_reward,
            action_bits: self.action_bits,
            rng: self.rng.fork_with(seed),
            obs_buffer: Vec::with_capacity(128),
            sym_buffer: Vec::with_capacity(64),
        }
    }

    /// Resets the agent's interaction statistics.
    pub fn reset(&mut self) {
        self.age = 0;
        self.total_reward = 0.0;
    }

    /// Primary interface for decision making.
    ///
    /// Uses MCTS to find the action that maximizes expected future reward.
    pub fn get_planned_action(
        &mut self,
        prev_obs_stream: &[PerceptVal],
        prev_rew: Reward,
        prev_act: Action,
    ) -> Action {
        let mut planner = self.planner.take().expect("Planner missing");
        let num_sim = self.config.num_simulations;
        let action = planner.search(self, prev_obs_stream, prev_rew, prev_act, num_sim);
        self.planner = Some(planner);
        action
    }

    /// Updates the world model with real-world percepts.
    pub fn model_update_percept(&mut self, observation: PerceptVal, reward: Reward) {
        self.model_update_percept_stream(&[observation], reward);
    }

    /// Updates the world model with an observation stream and a terminal reward.
    pub fn model_update_percept_stream(&mut self, observations: &[PerceptVal], reward: Reward) {
        debug_assert!(
            !observations.is_empty() || self.config.observation_bits == 0,
            "percept update missing observation stream"
        );
        let mut percept_syms = Vec::new();
        for &obs in observations {
            encode(&mut percept_syms, obs, self.config.observation_bits);
        }
        crate::aixi::common::encode_reward_offset(
            &mut percept_syms,
            reward,
            self.config.reward_bits,
            self.config.reward_offset,
        );

        for &sym in &percept_syms {
            self.model.commit_update(sym);
        }

        self.total_reward += reward as f64;
    }

    /// Computes the observation key used for search-tree branching.
    pub fn observation_repr_from_stream(&self, observations: &[PerceptVal]) -> Vec<PerceptVal> {
        observation_repr_from_stream(
            self.config.observation_key_mode,
            observations,
            self.config.observation_bits,
        )
    }

    /// Explicitly updates the world model with an action.
    pub fn model_update_action_external(&mut self, action: Action) {
        self.sym_buffer.clear();
        encode(&mut self.sym_buffer, action, self.action_bits);

        for &sym in &self.sym_buffer {
            self.model.commit_update_history(sym);
        }
    }
}

impl AgentSimulator for Agent {
    fn get_num_actions(&self) -> usize {
        self.config.agent_actions
    }

    fn get_num_observation_bits(&self) -> usize {
        self.config.observation_bits
    }

    fn observation_stream_len(&self) -> usize {
        self.config.observation_stream_len.max(1)
    }

    fn observation_key_mode(&self) -> ObservationKeyMode {
        self.config.observation_key_mode
    }

    fn get_num_reward_bits(&self) -> usize {
        self.config.reward_bits
    }

    fn horizon(&self) -> usize {
        self.config.agent_horizon
    }

    fn max_reward(&self) -> Reward {
        self.config.max_reward
    }

    fn min_reward(&self) -> Reward {
        self.config.min_reward
    }

    fn reward_offset(&self) -> i64 {
        self.config.reward_offset
    }

    fn get_explore_exploit_ratio(&self) -> f64 {
        self.config.exploration_exploitation_ratio
    }

    fn discount_gamma(&self) -> f64 {
        self.config.discount_gamma
    }

    fn model_update_action(&mut self, action: Action) {
        self.sym_buffer.clear();
        encode(&mut self.sym_buffer, action, self.action_bits);

        for &sym in &self.sym_buffer {
            self.model.update_history(sym);
        }
    }

    fn gen_percept_and_update(&mut self, bits: usize) -> u64 {
        self.sym_buffer.clear();
        for _ in 0..bits {
            let prob_1 = self.model.predict_one();
            let sym = self.rng.gen_bool(prob_1);
            self.model.update(sym);
            self.sym_buffer.push(sym);
        }
        decode(&self.sym_buffer, bits)
    }

    fn begin_simulation(&mut self) {
        self.model.begin_rollback_scope();
    }

    fn gen_percepts_and_update(&mut self) -> (Vec<PerceptVal>, Reward) {
        let obs_bits = self.config.observation_bits;
        let obs_len = self.config.observation_stream_len.max(1);

        self.obs_buffer.clear();
        for _ in 0..obs_len {
            let p = self.gen_percept_and_update(obs_bits);
            self.obs_buffer.push(p);
        }

        let obs_repr = observation_repr_from_stream(
            self.config.observation_key_mode,
            &self.obs_buffer,
            obs_bits,
        );
        let rew_bits = self.config.reward_bits;
        let rew_u = self.gen_percept_and_update(rew_bits);
        let rew = (rew_u as i64) - self.config.reward_offset;

        // Mark that we've completed a percept cycle (ready for next action)

        (obs_repr, rew)
    }

    fn gen_range(&mut self, end: usize) -> usize {
        self.rng.gen_range(end)
    }

    fn gen_f64(&mut self) -> f64 {
        self.rng.gen_f64()
    }

    fn model_revert(&mut self, steps: usize) {
        if self.model.rollback_scope() {
            return;
        }
        let obs_bits = self.config.observation_bits * self.config.observation_stream_len.max(1);
        let percept_bits = obs_bits + self.config.reward_bits;

        for _ in 0..steps {
            for _ in 0..percept_bits {
                self.model.revert();
            }
            for _ in 0..self.action_bits {
                self.model.pop_history();
            }
        }
    }

    fn boxed_clone_with_seed(&self, seed: u64) -> Box<dyn AgentSimulator> {
        Box::new(self.clone_for_simulation(seed))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(feature = "all-backends")]
    use crate::aixi::environment::{CtwTest, Environment};
    #[cfg(feature = "all-backends")]
    use crate::api::{MixtureExpertSpec, MixtureKind, MixtureSpec, RateBackend};
    use std::sync::{Arc, Mutex};

    #[derive(Clone, Default)]
    struct CallCounts {
        update: usize,
        commit_update: usize,
        update_history: usize,
        commit_update_history: usize,
        begin_scope: usize,
        rollback_scope: usize,
        revert: usize,
        pop_history: usize,
    }

    #[derive(Clone)]
    struct InstrumentedPredictor {
        counts: Arc<Mutex<CallCounts>>,
    }

    impl InstrumentedPredictor {
        fn new(counts: Arc<Mutex<CallCounts>>) -> Self {
            Self { counts }
        }
    }

    impl Predictor for InstrumentedPredictor {
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

        fn revert(&mut self) {
            self.counts.lock().unwrap().revert += 1;
        }

        fn pop_history(&mut self) {
            self.counts.lock().unwrap().pop_history += 1;
        }

        fn begin_rollback_scope(&mut self) {
            self.counts.lock().unwrap().begin_scope += 1;
        }

        fn rollback_scope(&mut self) -> bool {
            self.counts.lock().unwrap().rollback_scope += 1;
            true
        }

        fn predict_prob(&mut self, sym: bool) -> f64 {
            if sym { 0.75 } else { 0.25 }
        }

        fn model_name(&self) -> String {
            "InstrumentedPredictor".to_string()
        }

        fn boxed_clone(&self) -> Box<dyn Predictor> {
            Box::new(self.clone())
        }
    }

    fn basic_runtime_config() -> AgentRuntimeConfig {
        AgentRuntimeConfig {
            agent_horizon: 2,
            observation_bits: 2,
            observation_stream_len: 2,
            observation_key_mode: ObservationKeyMode::FullStream,
            reward_bits: 3,
            agent_actions: 4,
            num_simulations: 2,
            exploration_exploitation_ratio: 1.0,
            discount_gamma: 0.95,
            min_reward: -2,
            max_reward: 3,
            reward_offset: 2,
            random_seed: Some(7),
        }
    }

    fn test_agent(model: Box<dyn Predictor>) -> Agent {
        let config = basic_runtime_config();
        let action_bits = if config.agent_actions <= 1 {
            1
        } else {
            (usize::BITS - (config.agent_actions - 1).leading_zeros()) as usize
        };
        Agent {
            action_bits,
            model,
            planner: Some(SearchTree::new()),
            config,
            age: 0,
            total_reward: 0.0,
            rng: RandomGenerator::from_seed(7),
            obs_buffer: Vec::with_capacity(128),
            sym_buffer: Vec::with_capacity(64),
        }
    }

    #[cfg(feature = "all-backends")]
    fn generic_mixture_config() -> AgentConfig {
        AgentConfig {
            algorithm: "ignored-by-rate-backend".to_string(),
            ct_depth: 8,
            agent_horizon: 5,
            observation_bits: 1,
            observation_stream_len: 1,
            observation_key_mode: ObservationKeyMode::FullStream,
            reward_bits: 1,
            agent_actions: 2,
            num_simulations: 60,
            exploration_exploitation_ratio: 1.4,
            discount_gamma: 1.0,
            min_reward: 0,
            max_reward: 1,
            reward_offset: 0,
            random_seed: Some(2026),
            rate_backend: Some(RateBackend::Mixture {
                spec: Arc::new(
                    MixtureSpec::new(
                        MixtureKind::Convex,
                        vec![
                            MixtureExpertSpec {
                                name: Some("ctw".to_string()),
                                log_prior: 0.0,
                                max_order: -1,
                                backend: RateBackend::Ctw { depth: 8 },
                            },
                            MixtureExpertSpec {
                                name: Some("rosa".to_string()),
                                log_prior: 0.0,
                                max_order: 8,
                                backend: RateBackend::RosaPlus,
                            },
                        ],
                    )
                    .with_alpha(1.25),
                ),
            }),
            rate_backend_max_order: 8,
            rwkv_model_path: None,
            rwkv_method: None,
            mamba_model_path: None,
            mamba_method: None,
            rosa_max_order: Some(8),
            zpaq_method: None,
        }
    }

    #[cfg(feature = "all-backends")]
    fn run_ctw_trace(agent: &mut Agent, cycles: usize) -> (Vec<Action>, i64) {
        let mut env = CtwTest::new();
        let mut actions = Vec::with_capacity(cycles);
        let mut total_reward = 0i64;
        let mut obs_stream = env.drain_observations();
        let mut prev_rew = env.get_reward();
        let mut prev_act = 0;

        for _ in 0..cycles {
            agent.model_update_percept_stream(&obs_stream, prev_rew);
            let action = agent.get_planned_action(&obs_stream, prev_rew, prev_act);
            actions.push(action);
            agent.model_update_action_external(action);

            env.perform_action(action);
            obs_stream = env.drain_observations();
            let rew = env.get_reward();
            agent.model_update_percept_stream(&obs_stream, rew);
            total_reward += rew;
            prev_rew = rew;
            prev_act = action;
        }

        (actions, total_reward)
    }

    #[test]
    fn external_history_updates_use_committed_predictor_paths() {
        let counts = Arc::new(Mutex::new(CallCounts::default()));
        let mut agent = test_agent(Box::new(InstrumentedPredictor::new(counts.clone())));

        agent.model_update_percept_stream(&[1, 2], 1);
        agent.model_update_action_external(3);

        let snapshot = counts.lock().unwrap().clone();
        assert_eq!(snapshot.commit_update, 7);
        assert_eq!(snapshot.commit_update_history, 2);
        assert_eq!(snapshot.update, 0);
        assert_eq!(snapshot.update_history, 0);
    }

    #[test]
    fn simulation_revert_prefers_predictor_scope_when_available() {
        let counts = Arc::new(Mutex::new(CallCounts::default()));
        let mut agent = test_agent(Box::new(InstrumentedPredictor::new(counts.clone())));

        AgentSimulator::begin_simulation(&mut agent);
        agent.model_revert(3);

        let snapshot = counts.lock().unwrap().clone();
        assert_eq!(snapshot.begin_scope, 1);
        assert_eq!(snapshot.rollback_scope, 1);
        assert_eq!(snapshot.revert, 0);
        assert_eq!(snapshot.pop_history, 0);
    }

    #[cfg(feature = "all-backends")]
    #[test]
    fn compiled_mcaixi_runtime_matches_legacy_config_for_generic_mixture_backend() {
        let config = generic_mixture_config();
        let compiled = config
            .compile_planner_run_spec()
            .expect("generic planner run should compile");
        let mut legacy = Agent::try_new(config).expect("legacy config agent");
        let mut canonical =
            Agent::from_compiled_planner_run(&compiled).expect("compiled planner-run agent");

        let legacy_trace = run_ctw_trace(&mut legacy, 32);
        let canonical_trace = run_ctw_trace(&mut canonical, 32);
        assert_eq!(canonical_trace, legacy_trace);
    }
}
