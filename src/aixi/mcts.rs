//! Monte Carlo Tree Search (MCTS) for AIXI.
//!
//! This module implements the planning component of MC-AIXI. It use an upper
//! confidence bounds applied to trees (UCT) approach to select actions
//! by simulating future interactions with a world model.

use crate::aixi::common::{Action, PerceptVal, Reward};
use std::collections::HashMap;

/// Interface for an agent that can be simulated during MCTS.
///
/// This trait allows the MCTS algorithm to interact with an agent
/// (like `Agent` in `agent.rs`) to perform "imagined" actions and
/// receive "imagined" percepts during planning.
pub trait AgentSimulator {
    /// Returns the number of possible actions the agent can perform.
    fn get_num_actions(&self) -> usize;

    /// Returns the bit-width used to encode observations.
    fn get_num_observation_bits(&self) -> usize;

    /// Returns the bit-width used to encode rewards.
    fn get_num_reward_bits(&self) -> usize;

    /// Returns the planning horizon (depth of simulations).
    fn horizon(&self) -> usize;

    /// Returns the maximum possible reward value.
    fn max_reward(&self) -> Reward;

    /// Returns the minimum possible reward value.
    fn min_reward(&self) -> Reward;

    /// Returns the exploration-exploitation constant (often denoted as C).
    fn get_explore_exploit_ratio(&self) -> f64 {
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

    /// Helper to generate both an observation and a reward.
    fn gen_percepts_and_update(&mut self) -> (PerceptVal, Reward) {
        let obs = self.gen_percept_and_update(self.get_num_observation_bits());
        let rew = self.gen_percept_and_update(self.get_num_reward_bits());
        (obs, rew)
    }
}

/// A node in the MCTS search tree.
///
/// Nodes can be either OR-nodes (representing an agent choice) or
/// chance nodes (representing an environment response).
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

    /// Selects an action to explore, potentially creating a new child node.
    fn select_action(
        &mut self,
        agent: &mut dyn AgentSimulator,
        horizon: usize,
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
            let range = (agent.max_reward() - agent.min_reward()) as f64;
            let norm = 1.0 / (horizon as f64 * range).max(1e-9);

            for (&a, child) in &self.children {
                let exploit = child.expectation() * norm;
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
            let obs = agent.gen_percept_and_update(agent.get_num_observation_bits());
            let rew = agent.gen_percept_and_update(agent.get_num_reward_bits());

            let obs_bits = agent.get_num_observation_bits();
            let key = obs + (rew << obs_bits);

            let child = self
                .children
                .entry(key)
                .or_insert_with(|| SearchNode::new(false));
            reward = (rew as f64) + child.sample(agent, horizon - 1, total_horizon);
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

        for _ in 0..horizon {
            let act = agent.gen_range(num_actions);
            agent.model_update_action(act as Action);
            let (_obs, rew) = agent.gen_percepts_and_update();
            total_rew += rew as f64;
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
        prev_rew: u64,
        prev_act: u64,
        samples: usize,
    ) -> Action {
        self.prune_tree(agent, prev_obs, prev_rew, prev_act);

        let root = self.root.as_mut().unwrap();
        let h = agent.horizon();

        for _ in 0..samples {
            root.sample(agent, h, h);
        }

        root.best_action(agent)
    }

    /// Prunes the tree, keeping only relevant subtrees based on the previous interaction.
    fn prune_tree(
        &mut self,
        agent: &mut dyn AgentSimulator,
        prev_obs: u64,
        prev_rew: u64,
        prev_act: u64,
    ) {
        if self.root.is_none() {
            self.root = Some(SearchNode::new(false));
            return;
        }

        let mut old_root = self.root.take().unwrap();

        // Find chance child (prev_act)
        if let Some(mut chance_child) = old_root.children.remove(&prev_act) {
            let obs_bits = agent.get_num_observation_bits();
            let key = prev_obs + (prev_rew << obs_bits);

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
