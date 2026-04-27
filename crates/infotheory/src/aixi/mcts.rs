//! Monte Carlo Tree Search (MCTS) for AIXI.
//!
//! The sequential `rho_uct` planner follows "A Monte-Carlo AIXI
//! Approximation". Parallel planners are explicit and live in a separate
//! backend rather than being inferred from the simulation count.

mod parallel_uct;
mod rho_uct;

use crate::aixi::common::{
    Action, ObservationKeyMode, PerceptVal, Reward, observation_repr_from_stream,
};

pub use parallel_uct::{ParallelUctPlanner, ParallelUctPlannerInitError, ParallelUctSearchError};
pub use rho_uct::RhoUctPlanner;

use std::collections::HashMap;

/// Hash key for a sampled percept outcome at a chance node.
///
/// Both the observation representation and the immediate reward are required
/// to identify the correct continuation subtree for generic environments.
/// Some environments can emit the same observation alongside different rewards,
/// so observation-only keys would incorrectly merge distinct successor states
/// during search-tree reuse.
#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub(crate) struct PerceptOutcome {
    /// Observation symbols used for chance-node branching.
    observations: Box<[PerceptVal]>,
    /// Immediate reward observed on the sampled edge.
    reward: Reward,
}

impl PerceptOutcome {
    /// Creates a compact percept key from an observation stream and reward.
    pub(crate) fn new(observations: Vec<PerceptVal>, reward: Reward) -> Self {
        Self {
            observations: observations.into_boxed_slice(),
            reward,
        }
    }

    pub(crate) fn reward(&self) -> Reward {
        self.reward
    }
}

pub(crate) fn prune_key(
    agent: &dyn AgentSimulator,
    prev_obs_stream: &[PerceptVal],
    prev_rew: Reward,
) -> PerceptOutcome {
    let obs_repr = agent.observation_repr_from_stream(prev_obs_stream);
    PerceptOutcome::new(obs_repr, prev_rew)
}

/// Interface for an agent that can be simulated during MCTS.
///
/// This trait allows the MCTS algorithm to interact with an agent
/// (like `Agent` in `agent.rs`) to perform imagined actions and receive
/// imagined percepts during planning.
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
    /// Paper-compatible encoding uses unsigned reward bits and shifts rewards by
    /// an offset.
    fn reward_offset(&self) -> i64 {
        0
    }

    /// Returns the exploration-exploitation constant.
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

    /// Returns the discounted cumulative reward bounds for the given horizon.
    fn cumulative_reward_bounds(&self, horizon: usize) -> (f64, f64) {
        let min = self.min_reward() as f64;
        let max = self.max_reward() as f64;
        let gamma = self.discount_gamma().clamp(0.0, 1.0);
        let sum = discounted_horizon_sum(gamma, horizon);
        (min * sum, max * sum)
    }

    /// Normalizes a reward value to `[0, 1]` for a particular remaining horizon.
    ///
    /// For `gamma == 1` this is action-equivalent to the `aixictwx`
    /// finite-horizon normalization. For `gamma < 1` this is the discounted
    /// finite-horizon analogue, matching the discounted return actually backed
    /// up by the planner.
    fn norm_reward_for_horizon(&self, reward: f64, horizon: usize) -> f64 {
        let (min_cumulative, max_cumulative) = self.cumulative_reward_bounds(horizon);
        let range = max_cumulative - min_cumulative;
        if range.abs() < 1e-12 {
            0.5
        } else {
            ((reward - min_cumulative) / range).clamp(0.0, 1.0)
        }
    }

    /// Backward-compatible normalization using the full planning horizon.
    fn norm_reward(&self, reward: f64) -> f64 {
        self.norm_reward_for_horizon(reward, self.horizon())
    }

    /// Helper to generate a percept stream, update the model, and return a
    /// search key plus reward.
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

pub(crate) fn discounted_horizon_sum(gamma: f64, horizon: usize) -> f64 {
    if horizon == 0 {
        return 0.0;
    }
    if (gamma - 1.0).abs() < 1e-12 {
        horizon as f64
    } else {
        (1.0 - gamma.powi(horizon as i32)) / (1.0 - gamma)
    }
}

pub(crate) fn random_rollout(agent: &mut dyn AgentSimulator, horizon: usize) -> f64 {
    let num_actions = agent.get_num_actions();
    let gamma = agent.discount_gamma().clamp(0.0, 1.0);
    let mut total_reward = 0.0;
    let mut discount = 1.0;

    for _ in 0..horizon {
        let action = agent.gen_range(num_actions) as Action;
        agent.model_update_action(action);
        let (_obs, reward) = agent.gen_percepts_and_update();
        total_reward += discount * (reward as f64);
        discount *= gamma;
    }

    total_reward
}

pub(crate) fn ensure_action_slots<T>(slots: &mut Vec<Option<T>>, num_actions: usize) {
    if slots.len() < num_actions {
        slots.resize_with(num_actions, || None);
    }
}

pub(crate) fn choose_uniform_unvisited<T>(
    agent: &mut dyn AgentSimulator,
    slots: &[Option<T>],
    num_actions: usize,
) -> Option<usize> {
    let mut unvisited = Vec::new();
    for action_idx in 0..num_actions {
        if slots.get(action_idx).and_then(Option::as_ref).is_none() {
            unvisited.push(action_idx);
        }
    }
    if unvisited.is_empty() {
        None
    } else {
        Some(unvisited[agent.gen_range(unvisited.len())])
    }
}

pub(crate) fn best_action_from_action_values(
    action_values: impl Iterator<Item = (usize, f64)>,
    num_actions: usize,
    agent: &mut dyn AgentSimulator,
) -> Action {
    let mut best_actions = Vec::new();
    let mut best_value = -f64::INFINITY;

    for (action_idx, value) in action_values {
        match value.total_cmp(&best_value) {
            std::cmp::Ordering::Greater => {
                best_value = value;
                best_actions.clear();
                best_actions.push(action_idx as Action);
            }
            std::cmp::Ordering::Equal => best_actions.push(action_idx as Action),
            std::cmp::Ordering::Less => {}
        }
    }

    if best_actions.is_empty() {
        return agent.gen_range(num_actions.max(1)) as Action;
    }

    best_actions[agent.gen_range(best_actions.len())]
}

pub(crate) type PerceptMap<T> = HashMap<PerceptOutcome, T>;
