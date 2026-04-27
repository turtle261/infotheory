//! The core AIXI agent implementation.
//!
//! This module defines the `Agent` struct, which ties together a world model
//! (Predictor) and an explicit MCTS planner state to form a complete autonomous
//! entity.

use crate::aixi::common::{
    Action, MctsStrategy, ObservationKeyMode, PerceptVal, RandomGenerator, Reward,
    RewardEncodingError, decode, encode, observation_repr_from_stream, resolve_random_seed,
    validate_reward_encoding_bounds, warn_parallel_uct_workers_one_once,
};
use crate::aixi::mcts::{
    AgentSimulator, ParallelUctPlanner, ParallelUctPlannerInitError, RhoUctPlanner,
};
use crate::aixi::model::{Predictor, PredictorBuildError, build_mc_aixi_predictor};
use crate::aixi::planner_spec::{PlannerInterfaceConfig, build_default_planner_run_spec};
use crate::api::{RateBackend, validate_rate_backend};
use crate::spec::{
    CompiledPlannerController, CompiledPlannerRunSpec, ControllerSpec, McAixiControllerSpec,
    PlannerRunSpec, SpecError,
};
use std::error::Error;
use std::fmt;

/// Error returned by MC-AIXI configuration validation and construction.
#[derive(Debug)]
#[non_exhaustive]
pub enum AgentError {
    /// `agent_actions` was zero.
    AgentActionsZero,
    /// `agent_horizon` was zero.
    AgentHorizonZero,
    /// `num_simulations` was zero.
    NumSimulationsZero,
    /// The UCT exploration/exploitation constant was non-positive.
    InvalidExplorationExploitationRatio {
        /// The invalid exploration/exploitation ratio value.
        value: f64,
    },
    /// The configured MC-AIXI discount factor was outside `[0, 1]`.
    InvalidDiscountGamma {
        /// The invalid discount factor value.
        value: f64,
    },
    /// The configured reward range is not representable.
    RewardEncoding(RewardEncodingError),
    /// The configured rate backend failed validation.
    InvalidRateBackend(crate::error::InfotheoryError),
    /// The configured rate backend violates MC-AIXI runtime requirements.
    UnsupportedRateBackend {
        /// Human-readable explanation of why the backend is unsupported.
        reason: &'static str,
    },
    /// Planner-run spec compilation failed.
    Spec(SpecError),
    /// The compiled planner-run controller kind was not MC-AIXI.
    ControllerKindMismatch,
    /// Predictor construction failed.
    Predictor(PredictorBuildError),
    /// Parallel UCT planner construction failed (e.g. invalid `bu_uct_m_max`).
    ParallelUctPlannerInit(ParallelUctPlannerInitError),
}

impl fmt::Display for AgentError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::AgentActionsZero => f.write_str("agent_actions must be >= 1"),
            Self::AgentHorizonZero => f.write_str("agent_horizon must be >= 1"),
            Self::NumSimulationsZero => f.write_str("num_simulations must be >= 1"),
            Self::InvalidExplorationExploitationRatio { value: _ } => {
                f.write_str("exploration_exploitation_ratio must be > 0")
            }
            Self::InvalidDiscountGamma { value } => {
                write!(
                    f,
                    "discount_gamma must be in [0, 1] for MC-AIXI, got {value}"
                )
            }
            Self::RewardEncoding(err) => write!(f, "{err}"),
            Self::InvalidRateBackend(err) => write!(f, "invalid rate_backend: {err}"),
            Self::UnsupportedRateBackend { reason } => f.write_str(reason),
            Self::Spec(err) => write!(f, "{err}"),
            Self::ControllerKindMismatch => {
                f.write_str("compiled planner run does not contain an MC-AIXI controller")
            }
            Self::Predictor(err) => write!(f, "{err}"),
            Self::ParallelUctPlannerInit(err) => {
                write!(f, "parallel_uct planner construction failed: {err}")
            }
        }
    }
}

impl Error for AgentError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::RewardEncoding(err) => Some(err),
            Self::InvalidRateBackend(err) => Some(err),
            Self::Spec(err) => Some(err),
            Self::Predictor(err) => Some(err),
            Self::ParallelUctPlannerInit(err) => Some(err),
            _ => None,
        }
    }
}

impl From<RewardEncodingError> for AgentError {
    fn from(value: RewardEncodingError) -> Self {
        Self::RewardEncoding(value)
    }
}

impl From<SpecError> for AgentError {
    fn from(value: SpecError) -> Self {
        Self::Spec(value)
    }
}

impl From<ParallelUctPlannerInitError> for AgentError {
    fn from(value: ParallelUctPlannerInitError) -> Self {
        Self::ParallelUctPlannerInit(value)
    }
}

/// Configuration parameters for an AIXI agent.
#[derive(Clone)]
#[non_exhaustive]
pub struct AgentConfig {
    /// Predictive backend used by MC-AIXI.
    pub rate_backend: RateBackend,
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
    /// Explicit MCTS strategy.
    pub mcts_strategy: MctsStrategy,
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
    /// When `None`, planner runtime canonicalizes this to seed `0`.
    pub random_seed: Option<u64>,
    /// Max-order hint for `rate_backend` constructors that use it (for example ROSA).
    pub rate_backend_max_order: i64,
}

impl Default for AgentConfig {
    fn default() -> Self {
        Self {
            rate_backend: RateBackend::Ctw { depth: 8 },
            agent_horizon: 5,
            observation_bits: 1,
            observation_stream_len: 1,
            observation_key_mode: ObservationKeyMode::FullStream,
            reward_bits: 1,
            agent_actions: 2,
            num_simulations: 100,
            mcts_strategy: MctsStrategy::RhoUct,
            exploration_exploitation_ratio: 1.0,
            discount_gamma: 1.0,
            min_reward: 0,
            max_reward: 1,
            reward_offset: 0,
            random_seed: None,
            rate_backend_max_order: 8,
        }
    }
}

impl AgentConfig {
    fn canonical_predictor_backend(&self) -> RateBackend {
        self.rate_backend.clone()
    }

    fn canonical_planner_run_spec(&self) -> PlannerRunSpec {
        let predictor = self.canonical_predictor_backend();
        build_default_planner_run_spec(
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
                mcts_strategy: self.mcts_strategy,
                exploration_exploitation_ratio: self.exploration_exploitation_ratio,
                discount_gamma: self.discount_gamma,
            }),
            self.random_seed,
        )
    }

    fn compile_planner_run_spec(&self) -> Result<CompiledPlannerRunSpec, AgentError> {
        self.canonical_planner_run_spec()
            .compile()
            .map_err(AgentError::from)
    }

    fn validate_runtime_invariants(&self) -> Result<(), AgentError> {
        if self.agent_actions == 0 {
            return Err(AgentError::AgentActionsZero);
        }
        if self.agent_horizon == 0 {
            return Err(AgentError::AgentHorizonZero);
        }
        if self.num_simulations == 0 {
            return Err(AgentError::NumSimulationsZero);
        }
        match self.mcts_strategy {
            MctsStrategy::RhoUct => {}
            MctsStrategy::ParallelUct {
                workers,
                bu_uct_m_max,
            } => {
                // `workers` is `NonZeroUsize`, so the `>= 1` invariant is
                // type-enforced and no runtime check is needed here.
                if workers.get() == 1 {
                    warn_parallel_uct_workers_one_once();
                }
                if let Some(m_max) = bu_uct_m_max {
                    if !(0.0 < m_max && m_max < 1.0) {
                        return Err(AgentError::Spec(SpecError::new(
                            "controller.mcts_strategy.bu_uct_m_max must be in (0, 1)",
                        )));
                    }
                }
            }
        }
        if self.exploration_exploitation_ratio <= 0.0 {
            return Err(AgentError::InvalidExplorationExploitationRatio {
                value: self.exploration_exploitation_ratio,
            });
        }
        if !(0.0..=1.0).contains(&self.discount_gamma) {
            return Err(AgentError::InvalidDiscountGamma {
                value: self.discount_gamma,
            });
        }
        validate_reward_encoding_bounds(
            self.min_reward,
            self.max_reward,
            self.reward_offset,
            self.reward_bits,
        )?;

        validate_rate_backend(&self.rate_backend).map_err(AgentError::InvalidRateBackend)?;
        let compiled = self.rate_backend.compile().map_err(AgentError::from)?;
        if compiled.contains_zpaq() {
            return Err(AgentError::UnsupportedRateBackend {
                reason: "MC-AIXI strict generic rate_backend support requires reversible action conditioning; configured rate_backend contains zpaq which does not provide the reversible action conditioning required by \"A Monte-Carlo AIXI Approximation\"",
            });
        }

        Ok(())
    }

    /// Validate configuration constraints for MC-AIXI.
    pub fn validate(&self) -> Result<(), AgentError> {
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
    mcts_strategy: MctsStrategy,
    exploration_exploitation_ratio: f64,
    discount_gamma: f64,
    min_reward: Reward,
    max_reward: Reward,
    reward_offset: Reward,
    random_seed: u64,
}

impl AgentRuntimeConfig {
    fn from_compiled(compiled: &CompiledPlannerRunSpec) -> Result<Self, AgentError> {
        let interface = compiled.interface();
        let runtime = compiled.runtime();
        let (
            agent_horizon,
            num_simulations,
            mcts_strategy,
            exploration_exploitation_ratio,
            discount_gamma,
        ) = match compiled.controller() {
            CompiledPlannerController::McAixi {
                agent_horizon,
                num_simulations,
                mcts_strategy,
                exploration_exploitation_ratio,
                discount_gamma,
                ..
            } => (
                *agent_horizon,
                *num_simulations,
                *mcts_strategy,
                *exploration_exploitation_ratio,
                *discount_gamma,
            ),
            _ => return Err(AgentError::ControllerKindMismatch),
        };

        Ok(Self {
            agent_horizon,
            observation_bits: interface.observation_bits,
            observation_stream_len: interface.observation_stream_len.max(1),
            observation_key_mode: interface.observation_key_mode,
            reward_bits: interface.reward_bits,
            agent_actions: interface.agent_actions,
            num_simulations,
            mcts_strategy,
            exploration_exploitation_ratio,
            discount_gamma,
            min_reward: interface.min_reward,
            max_reward: interface.max_reward,
            reward_offset: interface.reward_offset,
            random_seed: resolve_random_seed(runtime.random_seed),
        })
    }
}

enum PlannerState {
    RhoUct(RhoUctPlanner),
    ParallelUct(ParallelUctPlanner),
}

impl PlannerState {
    /// Construct the planner backend selected by `strategy`.
    ///
    /// `workers == 0` is type-prevented by [`MctsStrategy::ParallelUct`]. The
    /// only remaining failure mode is an out-of-range `bu_uct_m_max`, which
    /// is surfaced via [`AgentError::ParallelUctPlannerInit`]. Callers must
    /// chain this through `?` (typically from `Agent::from_compiled_config`)
    /// rather than panicking at construction time.
    fn new(strategy: MctsStrategy) -> Result<Self, AgentError> {
        match strategy {
            MctsStrategy::RhoUct => Ok(Self::RhoUct(RhoUctPlanner::new())),
            MctsStrategy::ParallelUct {
                workers,
                bu_uct_m_max,
            } => Ok(Self::ParallelUct(ParallelUctPlanner::new(
                workers,
                bu_uct_m_max,
            )?)),
        }
    }

    fn search(
        &mut self,
        agent: &mut dyn AgentSimulator,
        prev_obs_stream: &[PerceptVal],
        prev_rew: Reward,
        prev_act: Action,
        samples: usize,
    ) -> Action {
        match self {
            Self::RhoUct(planner) => {
                planner.search(agent, prev_obs_stream, prev_rew, prev_act, samples)
            }
            Self::ParallelUct(planner) => {
                planner.search_validated(agent, prev_obs_stream, prev_rew, prev_act, samples)
            }
        }
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
    planner: Option<PlannerState>,
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
    pub fn try_new(config: AgentConfig) -> Result<Self, AgentError> {
        config.validate_runtime_invariants()?;
        let compiled = config.compile_planner_run_spec()?;
        let runtime = AgentRuntimeConfig::from_compiled(&compiled)?;
        Self::from_compiled_config(runtime, &compiled)
    }

    /// Creates a new `Agent` from a compiled planner-run spec.
    pub fn from_compiled_planner_run(
        compiled: &CompiledPlannerRunSpec,
    ) -> Result<Self, AgentError> {
        let config = AgentRuntimeConfig::from_compiled(compiled)?;
        Self::from_compiled_config(config, compiled)
    }

    fn from_compiled_config(
        config: AgentRuntimeConfig,
        compiled: &CompiledPlannerRunSpec,
    ) -> Result<Self, AgentError> {
        let (predictor, predictor_max_order) = match compiled.controller() {
            CompiledPlannerController::McAixi {
                predictor,
                predictor_max_order,
                ..
            } => (predictor, *predictor_max_order),
            _ => return Err(AgentError::ControllerKindMismatch),
        };
        let percept_bits = (compiled.interface().observation_bits
            * compiled.interface().observation_stream_len.max(1))
            + compiled.interface().reward_bits;
        let model = build_mc_aixi_predictor(predictor, predictor_max_order, percept_bits)
            .map_err(AgentError::Predictor)?;

        let rng = RandomGenerator::from_seed(config.random_seed);

        let planner = PlannerState::new(config.mcts_strategy)?;
        Ok(Self {
            model,
            planner: Some(planner),
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

    /// Returns the resolved deterministic seed used by this agent.
    pub fn resolved_random_seed(&self) -> u64 {
        self.config.random_seed
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
    use crate::aixi::environment::Environment;
    #[cfg(feature = "all-backends")]
    use crate::aixi::test_envs::DeterministicBinaryEnv;
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
            mcts_strategy: MctsStrategy::RhoUct,
            exploration_exploitation_ratio: 1.0,
            discount_gamma: 0.95,
            min_reward: -2,
            max_reward: 3,
            reward_offset: 2,
            random_seed: 7,
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
            planner: Some(
                PlannerState::new(config.mcts_strategy)
                    .expect("test fixture mcts_strategy must be valid"),
            ),
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
            rate_backend: RateBackend::Mixture {
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
            },
            agent_horizon: 5,
            observation_bits: 1,
            observation_stream_len: 1,
            observation_key_mode: ObservationKeyMode::FullStream,
            reward_bits: 1,
            agent_actions: 2,
            num_simulations: 60,
            mcts_strategy: MctsStrategy::RhoUct,
            exploration_exploitation_ratio: 1.4,
            discount_gamma: 1.0,
            min_reward: 0,
            max_reward: 1,
            reward_offset: 0,
            random_seed: Some(2026),
            rate_backend_max_order: 8,
        }
    }

    #[cfg(feature = "all-backends")]
    fn run_ctw_trace(agent: &mut Agent, cycles: usize) -> (Vec<Action>, i64) {
        let mut env = DeterministicBinaryEnv::default();
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

    /// Stable shape-only descriptor for a `RateBackend` variant, used for
    /// alias-equivalence assertions without requiring `Debug` on the enum.
    fn backend_shape(backend: &crate::api::RateBackend) -> &'static str {
        use crate::api::RateBackend;
        match backend {
            RateBackend::Ctw { .. } => "ctw",
            RateBackend::FacCtw { .. } => "fac-ctw",
            RateBackend::RosaPlus => "rosaplus",
            _ => "other",
        }
    }

    #[test]
    fn explicit_rate_backend_semantics_are_symmetric_between_mc_aixi_and_aiqi() {
        let mut agent_cfg = AgentConfig::default();
        agent_cfg.rate_backend = crate::api::RateBackend::Ctw { depth: 8 };
        let agent_ctw = agent_cfg.canonical_predictor_backend();

        let mut aiqi_cfg = crate::aixi::aiqi::AiqiConfig::default();
        aiqi_cfg.rate_backend = crate::api::RateBackend::Ctw { depth: 8 };
        let aiqi_ctw = aiqi_cfg.canonical_predictor_backend_for_test();

        assert_eq!(
            backend_shape(&agent_ctw),
            "ctw",
            "MC-AIXI: 'ctw' must produce a single-tree CTW backend"
        );
        assert_eq!(
            backend_shape(&aiqi_ctw),
            "ctw",
            "AIQI: 'ctw' must produce a single-tree CTW backend"
        );

        agent_cfg.rate_backend = crate::api::RateBackend::RosaPlus;
        aiqi_cfg.rate_backend = crate::api::RateBackend::RosaPlus;
        let agent_rosa = agent_cfg.canonical_predictor_backend();
        let aiqi_rosa = aiqi_cfg.canonical_predictor_backend_for_test();
        assert_eq!(
            backend_shape(&agent_rosa),
            "rosaplus",
            "MC-AIXI: 'rosa' must produce ROSA+"
        );
        assert_eq!(
            backend_shape(&aiqi_rosa),
            "rosaplus",
            "AIQI: 'rosa' must produce ROSA+"
        );
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
