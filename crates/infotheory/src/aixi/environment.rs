//! Environment contract for AIXI/AIQI planners.
//!
//! Core AIXI is intentionally environment-agnostic. Concrete environments are
//! provided through optional integrations (for example `aixi-gameengine` and
//! `aixi-vm`) or by user-defined implementations.

use crate::aixi::common::{
    Action, ActionAlphabet, PerceptVal, Reward, action_alphabet_from_action_bits,
};

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
    fn get_num_actions(&self) -> ActionAlphabet {
        action_alphabet_from_action_bits(self.get_action_bits())
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

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Clone, Copy)]
    struct DummyEnv {
        observation: PerceptVal,
        reward: Reward,
        observation_bits: usize,
        reward_bits: usize,
        action_bits: usize,
        finished: bool,
    }

    impl Environment for DummyEnv {
        fn perform_action(&mut self, _action: Action) {}

        fn get_observation(&self) -> PerceptVal {
            self.observation
        }

        fn get_reward(&self) -> Reward {
            self.reward
        }

        fn is_finished(&self) -> bool {
            self.finished
        }

        fn get_observation_bits(&self) -> usize {
            self.observation_bits
        }

        fn get_reward_bits(&self) -> usize {
            self.reward_bits
        }

        fn get_action_bits(&self) -> usize {
            self.action_bits
        }
    }

    #[test]
    fn default_environment_helpers_are_consistent() {
        let mut env = DummyEnv {
            observation: 7,
            reward: -2,
            observation_bits: 3,
            reward_bits: 4,
            action_bits: 2,
            finished: false,
        };

        env.set_random_seed(1234);
        env.perform_action(1);

        assert_eq!(env.drain_observations(), vec![7]);
        assert_eq!(env.get_num_actions().get(), 4);
        assert_eq!(env.max_reward(), 7);
        assert_eq!(env.min_reward(), -8);
        assert_eq!(env.get_reward(), -2);
        assert!(!env.is_finished());
        assert_eq!(env.get_observation_bits(), 3);
    }

    #[test]
    fn reward_bound_helpers_cover_zero_and_wide_bit_ranges() {
        let zero_bits = DummyEnv {
            observation: 0,
            reward: 0,
            observation_bits: 1,
            reward_bits: 0,
            action_bits: 1,
            finished: false,
        };
        assert_eq!(zero_bits.min_reward(), 0);
        assert_eq!(zero_bits.max_reward(), 0);

        let wide_bits = DummyEnv {
            reward_bits: 64,
            ..zero_bits
        };
        assert_eq!(wide_bits.min_reward(), i64::MIN);
        assert_eq!(wide_bits.max_reward(), i64::MAX);
    }
}
