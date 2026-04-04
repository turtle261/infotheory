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

/// Hash key for a sampled percept outcome at a chance node.
///
/// Both the observation representation and the immediate reward are required
/// to identify the correct continuation subtree for generic environments.
/// Some environments can emit the same observation alongside different
/// rewards, so observation-only keys would incorrectly merge distinct
/// successor states during search-tree reuse.
#[derive(Clone, Debug, Eq, PartialEq, Hash)]
struct PerceptOutcome {
    /// Observation symbols used for chance-node branching.
    observations: Box<[PerceptVal]>,
    /// Immediate reward observed on the sampled edge.
    reward: Reward,
}

impl PerceptOutcome {
    /// Creates a compact percept key from an observation stream and reward.
    fn new(observations: Vec<PerceptVal>, reward: Reward) -> Self {
        Self {
            observations: observations.into_boxed_slice(),
            reward,
        }
    }
}

/// Interface for an agent that can be simulated during MCTS.
///
/// This trait allows the MCTS algorithm to interact with an agent
/// (like `Agent` in `agent.rs`) to perform "imagined" actions and
/// receive "imagined" percepts during planning.
pub trait AgentSimulator: Send {
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

    /// Marks the start of a new simulation rollout.
    fn begin_simulation(&mut self) {}

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

    fn effective_visits(&self, base: Option<&SearchNode>) -> u32 {
        self.visits + base.map_or(0, |node| node.visits)
    }

    fn effective_mean(&self, base: Option<&SearchNode>) -> f64 {
        let base_visits = base.map_or(0, |node| node.visits);
        let total_visits = self.visits + base_visits;
        if total_visits == 0 {
            return 0.0;
        }

        let base_sum = base.map_or(0.0, |node| node.mean * (node.visits as f64));
        let delta_sum = self.mean * (self.visits as f64);
        (base_sum + delta_sum) / (total_visits as f64)
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

    fn merge_delta(&mut self, delta: &SearchNode) {
        if self.is_chance_node != delta.is_chance_node {
            return;
        }

        if delta.visits > 0 {
            let total_visits = self.visits + delta.visits;
            let total_sum =
                self.mean * (self.visits as f64) + delta.mean * (delta.visits as f64);
            self.visits = total_visits;
            self.mean = if total_visits > 0 {
                total_sum / (total_visits as f64)
            } else {
                0.0
            };
        }

        if self.is_chance_node {
            for (key, delta_child) in &delta.percept_children {
                let child = self
                    .percept_children
                    .entry(key.clone())
                    .or_insert_with(|| SearchNode::new(delta_child.is_chance_node));
                child.merge_delta(delta_child);
            }
        } else {
            if self.action_children.len() < delta.action_children.len() {
                self.action_children
                    .resize_with(delta.action_children.len(), || None);
            }
            for (idx, delta_child) in delta.action_children.iter().enumerate() {
                let Some(delta_child) = delta_child.as_ref() else {
                    continue;
                };
                let child = self.action_children[idx]
                    .get_or_insert_with(|| SearchNode::new(delta_child.is_chance_node));
                child.merge_delta(delta_child);
            }
        }
    }

    /// Selects an action to explore, potentially creating a new child node.
    fn select_action(&mut self, agent: &mut dyn AgentSimulator) -> (&mut SearchNode, Action) {
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
            // Match reference MC-AIXI UCB scaling:
            // priority = E[return] + horizon*max_reward*sqrt(C*log(N)/n)
            let c = agent.get_explore_exploit_ratio().max(0.0);
            let explore_bias = (agent.horizon() as f64) * (agent.max_reward() as f64).max(0.0);
            let mut best_val = -f64::INFINITY;
            let mut best_action = None;
            let mut num_maximal_actions = 0usize;
            let log_visits = (self.visits as f64).ln().max(0.0);
            for (a, child) in self.action_children.iter().enumerate() {
                let Some(child) = child.as_ref() else {
                    continue;
                };
                let nvisits = child.visits as f64;
                let val = child.expectation() + explore_bias * ((c * log_visits) / nvisits).sqrt();
                debug_assert!(
                    val.is_finite(),
                    "UCB score must be finite for visited MC-AIXI action children"
                );
                match val.total_cmp(&best_val) {
                    std::cmp::Ordering::Greater => {
                        best_val = val;
                        best_action = Some(a as u64);
                        num_maximal_actions = 1;
                    }
                    std::cmp::Ordering::Equal => {
                        num_maximal_actions += 1;
                        // Tie-break from "A Monte-Carlo AIXI Approximation":
                        // choose uniformly among maximal actions.
                        // Reservoir sampling keeps this O(1) in memory without a tie list.
                        if agent.gen_range(num_maximal_actions) == 0 {
                            best_action = Some(a as u64);
                        }
                    }
                    std::cmp::Ordering::Less => {}
                }
            }
            action = best_action.expect("visited MC-AIXI node must have a maximal action");
        }

        agent.model_update_action(action as Action);
        (
            self.action_children[action as usize]
                .as_mut()
                .expect("missing action child"),
            action as Action,
        )
    }

    fn select_action_delta(
        &mut self,
        base: Option<&SearchNode>,
        agent: &mut dyn AgentSimulator,
    ) -> Action {
        let num_actions = agent.get_num_actions();

        if self.action_children.len() < num_actions {
            self.action_children.resize_with(num_actions, || None);
        }

        let mut unvisited = Vec::new();
        for action_idx in 0..num_actions {
            let local_child = self.action_children[action_idx].as_ref();
            let base_child = base
                .and_then(|node| node.action_children.get(action_idx))
                .and_then(|child| child.as_ref());
            let effective_visits =
                local_child.map_or(0, |node| node.visits) + base_child.map_or(0, |node| node.visits);
            if effective_visits == 0 {
                unvisited.push(action_idx as u64);
            }
        }

        let action = if !unvisited.is_empty() {
            let idx = agent.gen_range(unvisited.len());
            unvisited[idx]
        } else {
            let c = agent.get_explore_exploit_ratio().max(0.0);
            let explore_bias = (agent.horizon() as f64) * (agent.max_reward() as f64).max(0.0);
            let mut best_val = -f64::INFINITY;
            let mut best_action = None;
            let mut num_maximal_actions = 0usize;
            let log_visits = (self.effective_visits(base) as f64).ln().max(0.0);

            for action_idx in 0..num_actions {
                let local_child = self.action_children[action_idx].as_ref();
                let base_child = base
                    .and_then(|node| node.action_children.get(action_idx))
                    .and_then(|child| child.as_ref());
                let child_visits =
                    local_child.map_or(0, |node| node.visits) + base_child.map_or(0, |node| node.visits);
                if child_visits == 0 {
                    continue;
                }
                let child_mean = match local_child {
                    Some(local_child) => local_child.effective_mean(base_child),
                    None => base_child
                        .map(SearchNode::expectation)
                        .expect("visited child should exist in delta or base tree"),
                };
                let val =
                    child_mean + explore_bias * ((c * log_visits) / (child_visits as f64)).sqrt();
                debug_assert!(
                    val.is_finite(),
                    "UCB score must be finite for visited MC-AIXI action children"
                );
                match val.total_cmp(&best_val) {
                    std::cmp::Ordering::Greater => {
                        best_val = val;
                        best_action = Some(action_idx as u64);
                        num_maximal_actions = 1;
                    }
                    std::cmp::Ordering::Equal => {
                        num_maximal_actions += 1;
                        if agent.gen_range(num_maximal_actions) == 0 {
                            best_action = Some(action_idx as u64);
                        }
                    }
                    std::cmp::Ordering::Less => {}
                }
            }

            best_action.expect("visited MC-AIXI node must have a maximal action")
        };

        let action_idx = action as usize;
        self.action_children[action_idx].get_or_insert_with(|| SearchNode::new(true));
        action as Action
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
            let key = PerceptOutcome::new(obs, rew);
            let child = self
                .percept_children
                .entry(key)
                .or_insert_with(|| SearchNode::new(false));
            reward = (rew as f64)
                + agent.discount_gamma() * child.sample(agent, horizon - 1, total_horizon);
        } else if self.visits == 0 {
            reward = Self::playout(agent, horizon, total_horizon);
        } else {
            let (child, _act) = self.select_action(agent);
            reward = child.sample(agent, horizon, total_horizon);
        }

        // Update mean logic:
        self.mean = (reward + (self.visits as f64) * self.mean) / ((self.visits + 1) as f64);
        self.visits += 1;

        reward
    }

    fn sample_delta(
        &mut self,
        base: Option<&SearchNode>,
        agent: &mut dyn AgentSimulator,
        horizon: usize,
        total_horizon: usize,
    ) -> f64 {
        if horizon == 0 {
            agent.model_revert(total_horizon);
            return 0.0;
        }

        let reward = if self.is_chance_node {
            let (obs, rew) = agent.gen_percepts_and_update();
            let key = PerceptOutcome::new(obs, rew);
            let base_child = base.and_then(|node| node.percept_children.get(&key));
            let child = self
                .percept_children
                .entry(key)
                .or_insert_with(|| SearchNode::new(false));
            (rew as f64)
                + agent.discount_gamma()
                    * child.sample_delta(base_child, agent, horizon - 1, total_horizon)
        } else if self.effective_visits(base) == 0 {
            Self::playout(agent, horizon, total_horizon)
        } else {
            let action = self.select_action_delta(base, agent);
            agent.model_update_action(action);
            let action_idx = action as usize;
            let base_child = base
                .and_then(|node| node.action_children.get(action_idx))
                .and_then(|child| child.as_ref());
            let child = self.action_children[action_idx]
                .as_mut()
                .expect("missing delta action child");
            child.sample_delta(base_child, agent, horizon, total_horizon)
        };

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

        let h = agent.horizon();
        let threads = rayon::current_num_threads().max(1);
        if samples < 2 || threads < 2 {
            let root = self.root.as_mut().unwrap();
            for _ in 0..samples {
                agent.begin_simulation();
                root.sample(agent, h, h);
            }
            return root.best_action(agent);
        }

        let workers = threads.min(samples);
        let base = samples / workers;
        let extra = samples % workers;
        let mut agents = Vec::with_capacity(workers);
        for i in 0..workers {
            let seed = agent.gen_f64().to_bits() ^ (i as u64);
            agents.push(agent.boxed_clone_with_seed(seed));
        }

        let results: Vec<SearchNode> = {
            let snapshot = self.root.as_ref().expect("search tree root");
            agents
                .into_par_iter()
                .enumerate()
                .map(|(i, mut local_agent)| {
                    let mut local_root = SearchNode::new(snapshot.is_chance_node);
                    let iterations = base + usize::from(i < extra);
                    for _ in 0..iterations {
                        local_agent.begin_simulation();
                        local_root.sample_delta(Some(snapshot), local_agent.as_mut(), h, h);
                    }
                    local_root
                })
                .collect()
        };

        let root = self.root.as_mut().expect("search tree root");
        for local in &results {
            root.merge_delta(local);
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
            let key = PerceptOutcome::new(obs_repr, prev_rew);

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

impl Default for SearchTree {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::aixi::common::ObservationKeyMode;
    use std::sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    };

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
        let key = PerceptOutcome::new(obs_repr, prev_rew);
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

        // Build tree keyed on a different observation key so pruning misses it.
        let mut tree = build_tree_with_key(&agent, prev_act, &[9u64], prev_rew, 9.0, 2);

        tree.prune_tree(&mut agent, &prev_obs_stream, prev_rew, prev_act);

        let root = tree.root.as_ref().unwrap();
        assert!(!root.is_chance_node);
        assert_eq!(root.visits, 0);
        assert_eq!(root.mean, 0.0);
    }

    #[test]
    fn prune_tree_resets_when_reward_mismatch_shares_observation_key() {
        let prev_act = 1u64;
        let prev_obs_stream = vec![4u64, 5u64];
        let kept_rew: Reward = -2;
        let requested_rew: Reward = 2;

        let mut agent = DummyAgent::new(6, ObservationKeyMode::FullStream);
        let mut tree = build_tree_with_key(&agent, prev_act, &prev_obs_stream, kept_rew, 77.0, 11);

        tree.prune_tree(&mut agent, &prev_obs_stream, requested_rew, prev_act);

        let root = tree.root.as_ref().unwrap();
        assert!(!root.is_chance_node);
        assert_eq!(root.visits, 0);
        assert_eq!(root.mean, 0.0);
    }

    #[derive(Clone)]
    struct BeginCountingAgent {
        begins: Arc<AtomicUsize>,
    }

    impl AgentSimulator for BeginCountingAgent {
        fn get_num_actions(&self) -> usize {
            2
        }

        fn get_num_observation_bits(&self) -> usize {
            1
        }

        fn get_num_reward_bits(&self) -> usize {
            1
        }

        fn horizon(&self) -> usize {
            1
        }

        fn max_reward(&self) -> Reward {
            1
        }

        fn min_reward(&self) -> Reward {
            0
        }

        fn model_update_action(&mut self, _action: Action) {}

        fn gen_percept_and_update(&mut self, _bits: usize) -> u64 {
            0
        }

        fn begin_simulation(&mut self) {
            self.begins.fetch_add(1, Ordering::Relaxed);
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

    #[test]
    fn search_calls_begin_simulation_for_each_rollout() {
        let begins = Arc::new(AtomicUsize::new(0));
        let mut agent = BeginCountingAgent {
            begins: begins.clone(),
        };
        let mut tree = SearchTree::new();

        let _ = tree.search(&mut agent, &[0], 0, 0, 5);
        assert_eq!(begins.load(Ordering::Relaxed), 5);
    }

    #[derive(Clone)]
    struct TieBreakAgent {
        next_range: Arc<AtomicUsize>,
    }

    impl AgentSimulator for TieBreakAgent {
        fn get_num_actions(&self) -> usize {
            4
        }

        fn get_num_observation_bits(&self) -> usize {
            1
        }

        fn get_num_reward_bits(&self) -> usize {
            1
        }

        fn horizon(&self) -> usize {
            1
        }

        fn max_reward(&self) -> Reward {
            1
        }

        fn min_reward(&self) -> Reward {
            0
        }

        fn get_explore_exploit_ratio(&self) -> f64 {
            0.0
        }

        fn model_update_action(&mut self, _action: Action) {}

        fn gen_percept_and_update(&mut self, _bits: usize) -> u64 {
            0
        }

        fn model_revert(&mut self, _steps: usize) {}

        fn gen_range(&mut self, end: usize) -> usize {
            self.next_range
                .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |value| {
                    Some(value.saturating_sub(1))
                })
                .expect("range source should be initialized")
                % end
        }

        fn gen_f64(&mut self) -> f64 {
            0.0
        }

        fn boxed_clone_with_seed(&self, _seed: u64) -> Box<dyn AgentSimulator> {
            Box::new(self.clone())
        }
    }

    #[test]
    fn select_action_uses_uniform_tie_break_for_maximal_ucb_actions() {
        let mut node = SearchNode::new(false);
        node.visits = 16;
        node.action_children = vec![
            Some(SearchNode {
                visits: 5,
                mean: 0.1,
                is_chance_node: true,
                action_children: Vec::new(),
                percept_children: HashMap::new(),
            }),
            Some(SearchNode {
                visits: 5,
                mean: 0.9,
                is_chance_node: true,
                action_children: Vec::new(),
                percept_children: HashMap::new(),
            }),
            Some(SearchNode {
                visits: 5,
                mean: 0.2,
                is_chance_node: true,
                action_children: Vec::new(),
                percept_children: HashMap::new(),
            }),
            Some(SearchNode {
                visits: 5,
                mean: 0.9,
                is_chance_node: true,
                action_children: Vec::new(),
                percept_children: HashMap::new(),
            }),
        ];

        let mut agent = TieBreakAgent {
            next_range: Arc::new(AtomicUsize::new(0)),
        };

        let (_child, action) = node.select_action(&mut agent);
        assert_eq!(
            action, 3,
            "exactly tied maximal UCB actions should be chosen uniformly; scripted RNG selected the later maximal action"
        );
    }

    #[derive(Clone)]
    struct DeterministicRewardAgent {
        last_action: Action,
        emit_reward: bool,
    }

    impl AgentSimulator for DeterministicRewardAgent {
        fn get_num_actions(&self) -> usize {
            2
        }

        fn get_num_observation_bits(&self) -> usize {
            1
        }

        fn get_num_reward_bits(&self) -> usize {
            1
        }

        fn horizon(&self) -> usize {
            1
        }

        fn max_reward(&self) -> Reward {
            1
        }

        fn min_reward(&self) -> Reward {
            0
        }

        fn get_explore_exploit_ratio(&self) -> f64 {
            0.0
        }

        fn model_update_action(&mut self, action: Action) {
            self.last_action = action;
            self.emit_reward = false;
        }

        fn gen_percept_and_update(&mut self, _bits: usize) -> u64 {
            if self.emit_reward {
                self.emit_reward = false;
                self.last_action
            } else {
                self.emit_reward = true;
                0
            }
        }

        fn model_revert(&mut self, _steps: usize) {
            self.emit_reward = false;
        }

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

    #[test]
    fn parallel_search_merges_sparse_worker_deltas_correctly() {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(2)
            .build()
            .expect("thread pool");

        let (parallel_action, parallel_visits, visited_children) = pool.install(|| {
            let mut agent = DeterministicRewardAgent {
                last_action: 0,
                emit_reward: false,
            };
            let mut tree = SearchTree::new();
            let action = tree.search(&mut agent, &[0], 0, 0, 8);
            let root = tree.root.as_ref().expect("root");
            let visits = root.visits;
            let visited_children = root
                .action_children
                .iter()
                .filter_map(|child| child.as_ref())
                .filter(|child| child.visits > 0)
                .count();
            (action, visits, visited_children)
        });

        assert_eq!(parallel_visits, 8);
        assert!(parallel_action < 2);
        assert!(visited_children > 0);
    }
}
