//! The core AIXI agent implementation.
//!
//! This module defines the `Agent` struct, which ties together a world model 
//! (Predictor) and a planner (SearchTree) to form a complete autonomous entity.

use crate::aixi::mcts::{SearchTree, AgentSimulator};
use crate::aixi::model::{Predictor, CtwPredictor, RosaPredictor, RwkvPredictor};
use crate::aixi::common::{Action, PerceptVal, Reward, encode, decode, RandomGenerator};
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
    /// Number of bits used to encode rewards.
    pub reward_bits: usize,
    /// Number of possible actions.
    pub agent_actions: usize,
    /// Number of MCTS simulations per planning step.
    pub num_simulations: usize,
    /// Constant governing exploration vs exploitation in UCT.
    pub exploration_exploitation_ratio: f64,
    /// Path to the RWKV model weights (if using "rwkv").
    pub rwkv_model_path: Option<String>,
    /// Maximum Markov order for the ROSA model (if using "rosa").
    pub rosa_max_order: Option<i64>,
}

/// A complete MC-AIXI-CTW agent.
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
    
    /// State tracking to ensure model update consistency.
    is_last_update_percept: bool,

    /// Internal PRNG for simulations.
    rng: RandomGenerator,
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
        if config.agent_actions == 1 { action_bits = 1; }

        let model: Box<dyn Predictor> = match config.algorithm.as_str() {
            "ctw" | "ctw-context-tree" => {
                Box::new(CtwPredictor::new(config.ct_depth))
            },
            "rosa" => {
                let max_order = config.rosa_max_order.unwrap_or(20);
                Box::new(RosaPredictor::new(max_order))
            },
            "rwkv" => {
                let path = config.rwkv_model_path.as_ref().expect("RWKV model path required");
                let model_arc = load_rwkv7_model_from_path(path);
                Box::new(RwkvPredictor::new(model_arc))
            },
            _ => panic!("Unknown algorithm: {}", config.algorithm),
        };

        Self {
            model,
            planner: Some(SearchTree::new()),
            config,
            age: 0,
            total_reward: 0.0,
            action_bits,
            is_last_update_percept: true,
            rng: RandomGenerator::new(),
        }
    }
    
    /// Resets the agent's interaction statistics.
    pub fn reset(&mut self) {
        self.age = 0;
        self.total_reward = 0.0;
        self.is_last_update_percept = true;
    }

    /// Primary interface for decision making. 
    /// 
    /// Uses MCTS to find the action that maximizes expected future reward.
    pub fn get_planned_action(&mut self, prev_obs: PerceptVal, prev_rew: Reward, prev_act: Action) -> Action {
         let mut planner = self.planner.take().expect("Planner missing");
         let num_sim = self.config.num_simulations;
         let action = planner.search(self, prev_obs, prev_rew, prev_act, num_sim);
         self.planner = Some(planner);
         action
    }
    
    /// Updates the world model with real-world percepts.
    pub fn model_update_percept(&mut self, observation: PerceptVal, reward: Reward) {
        let mut percept_syms = Vec::new();
        encode(&mut percept_syms, observation, self.config.observation_bits);
        encode(&mut percept_syms, reward, self.config.reward_bits);
        
        for &sym in &percept_syms {
            self.model.update(sym);
        }
        
        self.total_reward += reward as f64;
        self.is_last_update_percept = true;
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
    
    fn get_num_reward_bits(&self) -> usize {
        self.config.reward_bits
    }
    
    fn horizon(&self) -> usize {
        self.config.agent_horizon
    }
    
    fn max_reward(&self) -> Reward {
        (1 << self.config.reward_bits) - 1
    }
    
    fn min_reward(&self) -> Reward {
        0
    }
    
    fn get_explore_exploit_ratio(&self) -> f64 {
        self.config.exploration_exploitation_ratio
    }

    fn model_update_action(&mut self, action: Action) {
        let mut action_syms = Vec::new();
        encode(&mut action_syms, action, self.action_bits);
        
        for &sym in &action_syms {
            self.model.update_history(sym);
        }
        
        self.is_last_update_percept = false;
    }
    
    fn gen_percept_and_update(&mut self, bits: usize) -> u64 {
        let mut syms = Vec::with_capacity(bits);
        for _ in 0..bits {
            let prob_1 = self.model.predict_one();
            let sym = self.rng.gen_bool(prob_1);
            self.model.update(sym);
            syms.push(sym);
        }
        decode(&syms, bits)
    }

    fn gen_range(&mut self, end: usize) -> usize {
        self.rng.gen_range(end)
    }
    
    fn gen_f64(&mut self) -> f64 {
        self.rng.gen_f64()
    }
    
    fn model_revert(&mut self, steps: usize) {
        let percept_bits = self.config.observation_bits + self.config.reward_bits;
        
        for _ in 0..steps {
            for _ in 0..percept_bits {
                self.model.revert();
            }
            for _ in 0..self.action_bits {
                self.model.pop_history();
            }
        }
    }
}
