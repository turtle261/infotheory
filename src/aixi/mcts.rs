//! Monte Carlo Tree Search (MCTS) for AIXI.
//!
//! This module implements the planning component of MC-AIXI. It use an upper
//! confidence bounds applied to trees (UCT) approach to select actions
//! by simulating future interactions with a world model.

use crate::aixi::common::{
    Action, ObservationKeyMode, PerceptVal, Reward, observation_repr_from_stream,
};
use rayon::prelude::*;
use std::collections::HashMap;

#[derive(Clone, Debug, Eq, PartialEq, Hash)]
struct PerceptOutcome {
    observations: Vec<PerceptVal>,
    reward: Reward,
}

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
        ObservationKeyMode::FullStream
    }

    /// Returns the observation representation used for tree branching.
    fn observation_repr_from_stream(&self, observations: &[PerceptVal]) -> Vec<PerceptVal> {
        observation_repr_from_stream(
            self.observation_key_mode(),
            observations,
            self.get_num_observation_bits(),
        )
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
    ///
    /// For discounted rewards, the cumulative range is `sum_{t=0}^{h-1} gamma^t * (max - min)`.
    /// Similarly, the minimum cumulative reward is `sum_{t=0}^{h-1} gamma^t * min`.
    fn norm_reward(&self, reward: f64) -> f64 {
        let min = self.min_reward() as f64;
        let max = self.max_reward() as f64;
        let h = self.horizon() as f64;
        let gamma = self.discount_gamma().clamp(0.0, 1.0);

        // Discounted sum factor: sum_{t=0}^{h-1} gamma^t = (1 - gamma^h) / (1 - gamma) for gamma != 1
        let discount_sum = if (gamma - 1.0).abs() < 1e-9 {
            h
        } else {
            (1.0 - gamma.powi(h as i32)) / (1.0 - gamma)
        };

        let range = (max - min) * discount_sum;
        let min_cumulative = min * discount_sum;

        if range.abs() < 1e-9 {
            0.5
        } else {
            (reward - min_cumulative) / range
        }
    }

    /// Helper to generate a percept stream, update the model, and return a search key + reward.
    fn gen_percepts_and_update(&mut self) -> (Vec<PerceptVal>, Reward) {
        let obs_bits = self.get_num_observation_bits();
        let obs_len = self.observation_stream_len().max(1);
        let mut observations = Vec::with_capacity(obs_len);
        for _ in 0..obs_len {
            observations.push(self.gen_percept_and_update(obs_bits));
        }

        let obs_key = self.observation_repr_from_stream(&observations);
        let rew_bits = self.get_num_reward_bits();
        let rew_u = self.gen_percept_and_update(rew_bits);
        let rew = (rew_u as i64) - self.reward_offset();
        (obs_key, rew)
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
    /// Children indexed by action (action nodes only).
    action_children: Vec<Option<SearchNode>>,
    /// Children indexed by percept outcome (chance nodes only).
    percept_children: HashMap<PerceptOutcome, SearchNode>,
}

impl SearchNode {
    /// Creates a new `SearchNode`.
    pub fn new(is_chance_node: bool) -> Self {
        Self {
            visits: 0,
            mean: 0.0,
            is_chance_node,
            action_children: Vec::new(),
            percept_children: HashMap::new(),
        }
    }

    /// Selects the best action from this node based on accumulated mean rewards.
    pub fn best_action(&self, agent: &mut dyn AgentSimulator) -> Action {
        let mut best_actions = Vec::new();
        let mut best_mean = -f64::INFINITY;

        for (action, child) in self.action_children.iter().enumerate() {
            let Some(child) = child.as_ref() else {
                continue;
            };
            let mean = child.mean;
            if mean > best_mean {
                best_mean = mean;
                best_actions.clear();
                best_actions.push(action as u64);
            } else if (mean - best_mean).abs() < 1e-9 {
                best_actions.push(action as u64);
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

        if self.is_chance_node {
            for (key, updated_child) in &updated.percept_children {
                if let Some(base_child) = base.percept_children.get(key) {
                    if let Some(self_child) = self.percept_children.get_mut(key) {
                        self_child.apply_delta(base_child, updated_child);
                    } else {
                        let mut child = SearchNode::new(updated_child.is_chance_node);
                        child.apply_delta(
                            &SearchNode::new(updated_child.is_chance_node),
                            updated_child,
                        );
                        self.percept_children.insert(key.clone(), child);
                    }
                } else if let Some(self_child) = self.percept_children.get_mut(key) {
                    let empty = SearchNode::new(updated_child.is_chance_node);
                    self_child.apply_delta(&empty, updated_child);
                } else {
                    let mut child = SearchNode::new(updated_child.is_chance_node);
                    child.apply_delta(
                        &SearchNode::new(updated_child.is_chance_node),
                        updated_child,
                    );
                    self.percept_children.insert(key.clone(), child);
                }
            }
        } else {
            let max_len = base
                .action_children
                .len()
                .max(updated.action_children.len());
            if self.action_children.len() < max_len {
                self.action_children.resize_with(max_len, || None);
            }
            for idx in 0..max_len {
                let base_child = base.action_children.get(idx).and_then(|c| c.as_ref());
                let updated_child = updated.action_children.get(idx).and_then(|c| c.as_ref());
                let Some(updated_child) = updated_child else {
                    continue;
                };
                match (base_child, self.action_children.get_mut(idx)) {
                    (Some(base_child), Some(Some(self_child))) => {
                        self_child.apply_delta(base_child, updated_child);
                    }
                    (Some(base_child), Some(slot @ None)) => {
                        let mut child = SearchNode::new(updated_child.is_chance_node);
                        child.apply_delta(base_child, updated_child);
                        *slot = Some(child);
                    }
                    (None, Some(Some(self_child))) => {
                        let empty = SearchNode::new(updated_child.is_chance_node);
                        self_child.apply_delta(&empty, updated_child);
                    }
                    (None, Some(slot @ None)) => {
                        let mut child = SearchNode::new(updated_child.is_chance_node);
                        child.apply_delta(
                            &SearchNode::new(updated_child.is_chance_node),
                            updated_child,
                        );
                        *slot = Some(child);
                    }
                    _ => {}
                }
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

        if self.action_children.len() < num_actions {
            self.action_children.resize_with(num_actions, || None);
        }

        let mut unvisited = Vec::new();
        for a in 0..num_actions {
            if self.action_children[a].is_none() {
                unvisited.push(a as u64);
            }
        }

        let action;
        if !unvisited.is_empty() {
            let idx = agent.gen_range(unvisited.len());
            action = unvisited[idx];
            self.action_children[action as usize] = Some(SearchNode::new(true));
        } else {
            // UCT Formula: exploit + explore
            let c = agent.get_explore_exploit_ratio();
            let mut best_val = -f64::INFINITY;
            let mut best_action = 0;
            let log_visits = (self.visits as f64).ln().max(0.0);
            for (a, child) in self.action_children.iter().enumerate() {
                let Some(child) = child.as_ref() else {
                    continue;
                };
                let exploit = agent.norm_reward(child.expectation());
                let explore = if child.visits > 0 {
                    c * (log_visits / child.visits as f64).sqrt()
                } else {
                    f64::INFINITY
                };
                let val = exploit + explore;
                if val > best_val {
                    best_val = val;
                    best_action = a as u64;
                }
            }
            action = best_action;
        }

        agent.model_update_action(action as Action);
        (
            self.action_children[action as usize]
                .as_mut()
                .expect("missing action child"),
            action as Action,
        )
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
            let (obs, rew) = agent.gen_percepts_and_update();
            let key = PerceptOutcome {
                observations: obs,
                reward: rew,
            };
            let child = self
                .percept_children
                .entry(key)
                .or_insert_with(|| SearchNode::new(false));
            reward = (rew as f64)
                + agent.discount_gamma() * child.sample(agent, horizon - 1, total_horizon);
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
        prev_obs_stream: &[PerceptVal],
        prev_rew: Reward,
        prev_act: u64,
        samples: usize,
    ) -> Action {
        self.prune_tree(agent, prev_obs_stream, prev_rew, prev_act);

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
        prev_obs_stream: &[PerceptVal],
        prev_rew: Reward,
        prev_act: u64,
    ) {
        if self.root.is_none() {
            self.root = Some(SearchNode::new(false));
            return;
        }

        let mut old_root = self.root.take().unwrap();

        // Find chance child (prev_act)
        let action_child_opt = if old_root.action_children.len() > prev_act as usize {
            old_root.action_children[prev_act as usize].take()
        } else {
            None
        };

        if let Some(mut chance_child) = action_child_opt {
            let obs_repr = agent.observation_repr_from_stream(prev_obs_stream);
            let key = PerceptOutcome {
                observations: obs_repr,
                reward: prev_rew,
            };

            if let Some(action_child) = chance_child.percept_children.remove(&key) {
                self.root = Some(action_child);
            } else {
                self.root = Some(SearchNode::new(false));
            }
        } else {
            self.root = Some(SearchNode::new(false));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::aixi::common::ObservationKeyMode;

    #[derive(Clone)]
    struct DummyAgent {
        obs_bits: usize,
        rew_bits: usize,
        horizon: usize,
        min_reward: Reward,
        max_reward: Reward,
        key_mode: ObservationKeyMode,
    }

    impl DummyAgent {
        fn new(obs_bits: usize, key_mode: ObservationKeyMode) -> Self {
            Self {
                obs_bits,
                rew_bits: 8,
                horizon: 5,
                min_reward: -1,
                max_reward: 1,
                key_mode,
            }
        }
    }

    impl AgentSimulator for DummyAgent {
        fn get_num_actions(&self) -> usize {
            4
        }

        fn get_num_observation_bits(&self) -> usize {
            self.obs_bits
        }

        fn observation_key_mode(&self) -> ObservationKeyMode {
            self.key_mode
        }

        fn get_num_reward_bits(&self) -> usize {
            self.rew_bits
        }

        fn horizon(&self) -> usize {
            self.horizon
        }

        fn max_reward(&self) -> Reward {
            self.max_reward
        }

        fn min_reward(&self) -> Reward {
            self.min_reward
        }

        fn model_update_action(&mut self, _action: Action) {}

        fn gen_percept_and_update(&mut self, _bits: usize) -> u64 {
            0
        }

        fn model_revert(&mut self, _steps: usize) {}

        fn gen_range(&mut self, _end: usize) -> usize {
            0
        }

        fn gen_f64(&mut self) -> f64 {
            0.0
        }

        fn boxed_clone_with_seed(&self, _seed: u64) -> Box<dyn AgentSimulator> {
            Box::new(self.clone())
        }
    }

    fn build_tree_with_key(
        agent: &DummyAgent,
        prev_act: u64,
        prev_obs_stream: &[PerceptVal],
        prev_rew: Reward,
        kept_mean: f64,
        kept_visits: u32,
    ) -> SearchTree {
        let mut old_root = SearchNode::new(false);
        old_root.action_children.resize(prev_act as usize + 1, None);

        let mut chance_child = SearchNode::new(true);
        let mut kept = SearchNode::new(false);
        kept.mean = kept_mean;
        kept.visits = kept_visits;

        let obs_repr = agent.observation_repr_from_stream(prev_obs_stream);
        let key = PerceptOutcome {
            observations: obs_repr,
            reward: prev_rew,
        };
        chance_child.percept_children.insert(key, kept);

        old_root.action_children[prev_act as usize] = Some(chance_child);
        SearchTree {
            root: Some(old_root),
        }
    }

    #[test]
    fn prune_tree_keeps_matching_subtree() {
        let prev_act = 2u64;
        let prev_obs_stream = vec![9u64, 2u64, 7u64];
        let prev_rew: Reward = 3;

        let mut agent = DummyAgent::new(3, ObservationKeyMode::FullStream);
        let mut tree = build_tree_with_key(&agent, prev_act, &prev_obs_stream, prev_rew, 123.0, 7);

        tree.prune_tree(&mut agent, &prev_obs_stream, prev_rew, prev_act);

        let root = tree.root.as_ref().expect("root should exist");
        assert!(!root.is_chance_node);
        assert_eq!(root.mean, 123.0);
        assert_eq!(root.visits, 7);
    }

    #[test]
    fn prune_tree_resets_when_action_missing() {
        let prev_act = 10u64;
        let prev_obs_stream = vec![1u64];
        let prev_rew: Reward = 0;

        let mut agent = DummyAgent::new(1, ObservationKeyMode::FullStream);
        let mut tree = SearchTree::new();

        tree.prune_tree(&mut agent, &prev_obs_stream, prev_rew, prev_act);

        let root = tree.root.as_ref().unwrap();
        assert!(!root.is_chance_node);
        assert_eq!(root.visits, 0);
        assert_eq!(root.mean, 0.0);
    }

    #[test]
    fn prune_tree_resets_when_percept_key_missing() {
        let prev_act = 0u64;
        let prev_obs_stream = vec![1u64, 2u64];
        let prev_rew: Reward = 1;

        let mut agent = DummyAgent::new(4, ObservationKeyMode::Last);

        // Build tree keyed on a different reward so the percept key won't match.
        let mut tree = build_tree_with_key(&agent, prev_act, &prev_obs_stream, prev_rew + 1, 9.0, 2);

        tree.prune_tree(&mut agent, &prev_obs_stream, prev_rew, prev_act);

        let root = tree.root.as_ref().unwrap();
        assert!(!root.is_chance_node);
        assert_eq!(root.visits, 0);
        assert_eq!(root.mean, 0.0);
    }
}
