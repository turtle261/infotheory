//! Monte Carlo Tree Search (MCTS) for AIXI.
//!
//! This module implements the planning component of MC-AIXI. It use an upper
//! confidence bounds applied to trees (UCT) approach to select actions
//! by simulating future interactions with a world model.

use crate::aixi::common::{
    Action, ObservationKeyMode, PerceptVal, Reward, observation_key_from_stream,
};
use rayon::prelude::*;
use std::collections::HashMap;

/// Interface for an agent that can be simulated during MCTS.
///
/// This trait allows the MCTS algorithm to interact with an agent
/// (like `Agent` in `agent.rs`) to perform "imagined" actions and
/// receive "imagined" percepts during planning.
pub trait AgentSimulator: Send + Sync {
    /// Returns the number of possible actions the agent can perform.
    fn get_num_actions(&self) -> usize;

    /// Returns the bit-width used to encode observations.
    fn get_num_observation_bits(&self) -> usize;

    /// Returns the number of observation symbols per action.
    fn observation_stream_len(&self) -> usize {
        1
    }

    /// Returns the observation key mode for search-tree branching.
    fn observation_key_mode(&self) -> ObservationKeyMode {
        ObservationKeyMode::First
    }

    /// Returns the bit-width used to encode rewards.
    fn get_num_reward_bits(&self) -> usize;

    /// Returns the planning horizon (depth of simulations).
    fn horizon(&self) -> usize;

    /// Returns the maximum possible reward value.
    fn max_reward(&self) -> Reward;

    /// Returns the minimum possible reward value.
    fn min_reward(&self) -> Reward;

    /// Returns the reward offset used to ensure encoded rewards are non-negative.
    ///
    /// Paper-compatible encoding uses unsigned reward bits and shifts rewards by an offset.
    fn reward_offset(&self) -> i64 {
        0
    }

    /// Returns the exploration-exploitation constant (often denoted as C).
    fn get_explore_exploit_ratio(&self) -> f64 {
        1.0
    }

    /// Returns the discount factor for future rewards.
    fn discount_gamma(&self) -> f64 {
        1.0
    }

    /// Updates the internal model state with a simulated action.
    fn model_update_action(&mut self, action: Action);

    /// Generates a simulated percept and updates the model state.
    fn gen_percept_and_update(&mut self, bits: usize) -> u64;

    /// Reverts the model state to a previous point in the simulation.
    fn model_revert(&mut self, steps: usize);

    /// Generates a random value in `[0, end)`.
    fn gen_range(&mut self, end: usize) -> usize;

    /// Generates a random `f64` in `[0, 1)`.
    fn gen_f64(&mut self) -> f64;

    /// Creates a boxed clone of this simulator for parallel search.
    fn boxed_clone(&self) -> Box<dyn AgentSimulator> {
        self.boxed_clone_with_seed(0)
    }

    /// Creates a boxed clone of this simulator, re-seeding any RNG state.
    fn boxed_clone_with_seed(&self, seed: u64) -> Box<dyn AgentSimulator>;

    /// Normalizes a reward value to [0, 1] based on the agent's range and horizon.
    fn norm_reward(&self, reward: f64) -> f64 {
        let min = self.min_reward() as f64;
        let max = self.max_reward() as f64;
        let h = self.horizon() as f64;
        let gamma = self.discount_gamma().clamp(0.0, 1.0);
        let range = if (gamma - 1.0).abs() < 1e-9 {
            (max - min) * h
        } else {
            (max - min) * ((1.0 - gamma.powi(h as i32)) / (1.0 - gamma))
        };
        if range.abs() < 1e-9 {
            0.5
        } else {
            (reward - (min * h)) / range
        }
    }

    /// Helper to generate a percept stream, update the model, and return a search key + reward.
    fn gen_percepts_and_update(&mut self) -> (PerceptVal, Reward) {
        let obs_bits = self.get_num_observation_bits();
        let obs_len = self.observation_stream_len().max(1);
        let mut observations = Vec::with_capacity(obs_len);
        for _ in 0..obs_len {
            observations.push(self.gen_percept_and_update(obs_bits));
        }

        let obs_key = observation_key_from_stream(self.observation_key_mode(), &observations, obs_bits);
        let rew_bits = self.get_num_reward_bits();
        let rew_u = self.gen_percept_and_update(rew_bits);
        let rew = (rew_u as i64) - self.reward_offset();
        (self.percept_key(obs_key, rew_u), rew)
    }

    /// Combines observation and reward into a single search-tree key.
    fn percept_key(&self, obs_key: PerceptVal, rew_u: u64) -> u64 {
        let obs_bits = self.get_num_observation_bits();
        let rew_bits = self.get_num_reward_bits();
        let can_pack = self.observation_stream_len() == 1
            && matches!(self.observation_key_mode(), ObservationKeyMode::First | ObservationKeyMode::Last)
            && obs_bits + rew_bits <= 63;
        if can_pack {
            obs_key + (rew_u << obs_bits)
        } else {
            obs_key.rotate_left(17) ^ rew_u.wrapping_mul(0x9E3779B97F4A7C15)
        }
    }
}

/// A node in the MCTS search tree.
///
/// Nodes can be either OR-nodes (representing an agent choice) or
/// chance nodes (representing an environment response).
#[derive(Clone)]
pub struct SearchNode {
    /// Number of times this node has been visited during search.
    visits: u32,
    /// The current mean reward estimated for this node.
    mean: f64,
    /// Whether this is a chance node (observation/reward) rather than an action node.
    is_chance_node: bool,
    /// Maps from Action or Percept key to child nodes.
    children: HashMap<u64, SearchNode>,
}

impl SearchNode {
    /// Creates a new `SearchNode`.
    pub fn new(is_chance_node: bool) -> Self {
        Self {
            visits: 0,
            mean: 0.0,
            is_chance_node,
            children: HashMap::new(),
        }
    }

    /// Selects the best action from this node based on accumulated mean rewards.
    pub fn best_action(&self, agent: &mut dyn AgentSimulator) -> Action {
        let mut best_actions = Vec::new();
        let mut best_mean = -f64::INFINITY;

        for (&action, child) in &self.children {
            let mean = child.mean;
            if mean > best_mean {
                best_mean = mean;
                best_actions.clear();
                best_actions.push(action);
            } else if (mean - best_mean).abs() < 1e-9 {
                best_actions.push(action);
            }
        }

        if best_actions.is_empty() {
            return 0;
        }

        let idx = agent.gen_range(best_actions.len());
        best_actions[idx] as Action
    }

    fn expectation(&self) -> f64 {
        self.mean
    }

    fn apply_delta(&mut self, base: &SearchNode, updated: &SearchNode) {
        if self.is_chance_node != base.is_chance_node
            || self.is_chance_node != updated.is_chance_node
        {
            return;
        }

        let base_visits = base.visits as f64;
        let updated_visits = updated.visits as f64;
        if updated_visits < base_visits {
            return;
        }

        let delta_visits = updated.visits - base.visits;
        if delta_visits > 0 {
            let base_sum = base.mean * base_visits;
            let updated_sum = updated.mean * updated_visits;
            let delta_sum = updated_sum - base_sum;
            let total_visits = self.visits + delta_visits;
            let total_sum = self.mean * (self.visits as f64) + delta_sum;
            self.visits = total_visits;
            self.mean = if total_visits > 0 {
                total_sum / (total_visits as f64)
            } else {
                0.0
            };
        }

        for (key, updated_child) in &updated.children {
            if let Some(base_child) = base.children.get(key) {
                if let Some(self_child) = self.children.get_mut(key) {
                    self_child.apply_delta(base_child, updated_child);
                } else {
                    let mut child = SearchNode::new(updated_child.is_chance_node);
                    child.apply_delta(
                        &SearchNode::new(updated_child.is_chance_node),
                        updated_child,
                    );
                    self.children.insert(*key, child);
                }
            } else if let Some(self_child) = self.children.get_mut(key) {
                let empty = SearchNode::new(updated_child.is_chance_node);
                self_child.apply_delta(&empty, updated_child);
            } else {
                let mut child = SearchNode::new(updated_child.is_chance_node);
                child.apply_delta(
                    &SearchNode::new(updated_child.is_chance_node),
                    updated_child,
                );
                self.children.insert(*key, child);
            }
        }
    }

    /// Selects an action to explore, potentially creating a new child node.
    fn select_action(
        &mut self,
        agent: &mut dyn AgentSimulator,
        _horizon: usize,
    ) -> (&mut SearchNode, Action) {
        let num_actions = agent.get_num_actions();

        let mut unvisited = Vec::new();
        for a in 0..num_actions {
            if !self.children.contains_key(&(a as u64)) {
                unvisited.push(a as u64);
            }
        }

        let action;
        if !unvisited.is_empty() {
            let idx = agent.gen_range(unvisited.len());
            action = unvisited[idx];
            self.children.insert(action, SearchNode::new(true));
        } else {
            // UCT Formula: exploit + explore
            let c = agent.get_explore_exploit_ratio();
            let mut best_val = -f64::INFINITY;
            let mut best_action = 0;
            let log_visits = (self.visits as f64).ln().max(0.0);
            for (&a, child) in &self.children {
                let exploit = agent.norm_reward(child.expectation());
                let explore = if child.visits > 0 {
                    c * (log_visits / child.visits as f64).sqrt()
                } else {
                    f64::INFINITY
                };
                let val = exploit + explore;
                if val > best_val {
                    best_val = val;
                    best_action = a;
                }
            }
            action = best_action;
        }

        agent.model_update_action(action as Action);
        (self.children.get_mut(&action).unwrap(), action as Action)
    }

    /// Performs a single simulation (sample) from this node.
    pub fn sample(
        &mut self,
        agent: &mut dyn AgentSimulator,
        horizon: usize,
        total_horizon: usize,
    ) -> f64 {
        if horizon == 0 {
            agent.model_revert(total_horizon);
            return 0.0;
        }

        let reward;
        if self.is_chance_node {
            let (key, rew) = agent.gen_percepts_and_update();
            let child = self
                .children
                .entry(key)
                .or_insert_with(|| SearchNode::new(false));
            reward = (rew as f64) + agent.discount_gamma() * child.sample(agent, horizon - 1, total_horizon);
        } else if self.visits == 0 {
            reward = Self::playout(agent, horizon, total_horizon);
        } else {
            let (child, _act) = self.select_action(agent, horizon);
            reward = child.sample(agent, horizon, total_horizon);
        }

        // Update mean logic:
        self.mean = (reward + (self.visits as f64) * self.mean) / ((self.visits + 1) as f64);
        self.visits += 1;

        reward
    }

    /// Performs a randomized simulation until the horizon is reached.
    fn playout(agent: &mut dyn AgentSimulator, horizon: usize, total_horizon: usize) -> f64 {
        let mut total_rew = 0.0;
        let num_actions = agent.get_num_actions();
        let gamma = agent.discount_gamma().clamp(0.0, 1.0);
        let mut discount = 1.0;

        for _ in 0..horizon {
            let act = agent.gen_range(num_actions);
            agent.model_update_action(act as Action);
            let (_key, rew) = agent.gen_percepts_and_update();
            total_rew += discount * (rew as f64);
            discount *= gamma;
        }

        agent.model_revert(total_horizon);
        total_rew
    }
}

/// Manages the MCTS tree and provides the `search` entry point.
pub struct SearchTree {
    root: Option<SearchNode>,
}

impl SearchTree {
    /// Creates a new `SearchTree`.
    pub fn new() -> Self {
        Self {
            root: Some(SearchNode::new(false)),
        }
    }

    /// Performs several MCTS simulations to find the best next action.
    pub fn search(
        &mut self,
        agent: &mut dyn AgentSimulator,
        prev_obs: u64,
        prev_rew: Reward,
        prev_act: u64,
        samples: usize,
    ) -> Action {
        self.prune_tree(agent, prev_obs, prev_rew, prev_act);

        let root = self.root.as_mut().unwrap();
        let h = agent.horizon();
        let threads = rayon::current_num_threads().max(1);
        if samples < 2 || threads < 2 {
            for _ in 0..samples {
                root.sample(agent, h, h);
            }
            return root.best_action(agent);
        }

        let workers = threads.min(samples);
        let base = samples / workers;
        let extra = samples % workers;
        let snapshot = root.clone();

        let mut agents = Vec::with_capacity(workers);
        for i in 0..workers {
            let seed = agent.gen_f64().to_bits() ^ (i as u64);
            agents.push(agent.boxed_clone_with_seed(seed));
        }

        let results: Vec<SearchNode> = agents
            .into_par_iter()
            .enumerate()
            .map(|(i, mut local_agent)| {
                let mut local_root = snapshot.clone();
                let iterations = base + usize::from(i < extra);
                for _ in 0..iterations {
                    local_root.sample(local_agent.as_mut(), h, h);
                }
                local_root
            })
            .collect();

        for local in &results {
            root.apply_delta(&snapshot, local);
        }

        root.best_action(agent)
    }

    /// Prunes the tree, keeping only relevant subtrees based on the previous interaction.
    fn prune_tree(
        &mut self,
        agent: &mut dyn AgentSimulator,
        prev_obs: u64,
        prev_rew: Reward,
        prev_act: u64,
    ) {
        if self.root.is_none() {
            self.root = Some(SearchNode::new(false));
            return;
        }

        let mut old_root = self.root.take().unwrap();

        // Find chance child (prev_act)
        if let Some(mut chance_child) = old_root.children.remove(&prev_act) {
            let offset = agent.reward_offset();
            let key_rew_u = (prev_rew + offset) as u64;
            let key = agent.percept_key(prev_obs, key_rew_u);

            if let Some(action_child) = chance_child.children.remove(&key) {
                self.root = Some(action_child);
            } else {
                self.root = Some(SearchNode::new(false));
            }
        } else {
            self.root = Some(SearchNode::new(false));
        }
    }
}
