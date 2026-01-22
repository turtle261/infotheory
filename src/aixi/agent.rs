//! The core AIXI agent implementation.
//!
//! This module defines the `Agent` struct, which ties together a world model
//! (Predictor) and a planner (SearchTree) to form a complete autonomous entity.

use crate::aixi::common::{
    Action, ObservationKeyMode, PerceptVal, RandomGenerator, Reward, decode, encode,
    observation_repr_from_stream,
};
use crate::aixi::mcts::{AgentSimulator, SearchTree};
use crate::aixi::model::{CtwPredictor, FacCtwPredictor, Predictor, RosaPredictor, RwkvPredictor};
use crate::load_rwkv7_model_from_path;

/// Configuration parameters for an AIXI agent.
#[derive(Clone, Debug)]
pub struct AgentConfig {
    /// The predictive algorithm to use ("ctw", "rosa", "rwkv").
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
    /// Path to the RWKV model weights (if using "rwkv").
    pub rwkv_model_path: Option<String>,
    /// Maximum Markov order for the ROSA model (if using "rosa").
    pub rosa_max_order: Option<i64>,
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
    config: AgentConfig,

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
        let mut action_bits = 0;
        let mut c = 1;
        let mut i = 1;
        while i < config.agent_actions {
            i *= 2;
            action_bits = c;
            c += 1;
        }
        if config.agent_actions == 1 {
            action_bits = 1;
        }

        let model: Box<dyn Predictor> = match config.algorithm.as_str() {
            // FAC-CTW is the default and recommended CTW variant per the paper
            "ctw" | "fac-ctw" => {
                let obs_len = config.observation_stream_len.max(1);
                let percept_bits = (config.observation_bits * obs_len) + config.reward_bits;
                Box::new(FacCtwPredictor::new(config.ct_depth, percept_bits))
            }
            // AC-CTW is the legacy single-tree variant
            "ac-ctw" | "ctw-context-tree" => Box::new(CtwPredictor::new(config.ct_depth)),
            "rosa" => {
                let max_order = config.rosa_max_order.unwrap_or(20);
                Box::new(RosaPredictor::new(max_order))
            }
            "rwkv" => {
                let path = config
                    .rwkv_model_path
                    .as_ref()
                    .expect("RWKV model path required");
                let model_arc = load_rwkv7_model_from_path(path);
                Box::new(RwkvPredictor::new(model_arc))
            }
            _ => panic!("Unknown algorithm: {}", config.algorithm),
        };

        Self {
            model,
            planner: Some(SearchTree::new()),
            config,
            age: 0,
            total_reward: 0.0,
            action_bits,
            rng: RandomGenerator::new(),
            obs_buffer: Vec::with_capacity(128),
            sym_buffer: Vec::with_capacity(64),
        }
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
            self.model.update(sym);
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
        self.model_update_action(action);
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
