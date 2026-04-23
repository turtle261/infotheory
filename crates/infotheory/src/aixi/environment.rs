//! Environment contract for AIXI/AIQI planners.
//!
//! Core AIXI is intentionally environment-agnostic. Concrete environments are
//! provided through optional integrations (for example `aixi-gameengine` and
//! `aixi-vm`) or by user-defined implementations.

use crate::aixi::common::{Action, PerceptVal, Reward};

/// Interface for an agent's environment.
pub trait Environment {
    /// Executes an action in the environment and updates internal state.
    fn perform_action(&mut self, action: Action);

    /// Returns the current observation produced by the environment.
    fn get_observation(&self) -> PerceptVal;

    /// Returns the observation stream emitted by the last action.
    ///
    /// Default behavior is a single-symbol stream.
    fn drain_observations(&mut self) -> Vec<PerceptVal> {
        vec![self.get_observation()]
    }

    /// Returns the current reward produced by the environment.
    fn get_reward(&self) -> Reward;

    /// Returns true if the environment has reached a terminal state.
    fn is_finished(&self) -> bool;

    /// Returns the number of bits used to encode observations.
    fn get_observation_bits(&self) -> usize;

    /// Returns the number of bits used to encode rewards.
    fn get_reward_bits(&self) -> usize;

    /// Returns the number of bits required to represent all valid actions.
    fn get_action_bits(&self) -> usize;

    /// Reseeds stochastic state for deterministic runs.
    fn set_random_seed(&mut self, _seed: u64) {}

    /// Returns the total number of valid actions available.
    fn get_num_actions(&self) -> usize {
        1usize << self.get_action_bits()
    }

    /// Returns the maximum possible reward value in this environment.
    fn max_reward(&self) -> Reward {
        let bits: usize = self.get_reward_bits();
        if bits == 0 {
            return 0;
        }
        if bits >= 64 {
            i64::MAX
        } else {
            (1i64 << (bits - 1)) - 1
        }
    }

    /// Returns the minimum possible reward value in this environment.
    fn min_reward(&self) -> Reward {
        let bits: usize = self.get_reward_bits();
        if bits == 0 {
            return 0;
        }
        if bits >= 64 {
            i64::MIN
        } else {
            -(1i64 << (bits - 1))
        }
    }
}
