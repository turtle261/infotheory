#![allow(dead_code)]

use infotheory::aixi::common::{Action, PerceptVal, RandomGenerator, Reward};
use infotheory::aixi::environment::Environment;

#[derive(Default)]
pub struct DeterministicBinaryEnv {
    cycle: usize,
    last_action: Action,
    obs: PerceptVal,
    rew: Reward,
}

impl DeterministicBinaryEnv {
    pub fn new() -> Self {
        Self::default()
    }
}

impl Environment for DeterministicBinaryEnv {
    fn perform_action(&mut self, action: Action) {
        self.obs = if self.cycle == 0 {
            0
        } else {
            (self.last_action + 1) % 2
        };
        self.rew = if action == self.obs { 1 } else { 0 };
        self.last_action = action;
        self.cycle += 1;
    }

    fn get_observation(&self) -> PerceptVal {
        self.obs
    }

    fn get_reward(&self) -> Reward {
        self.rew
    }

    fn is_finished(&self) -> bool {
        false
    }

    fn get_observation_bits(&self) -> usize {
        1
    }

    fn get_reward_bits(&self) -> usize {
        1
    }

    fn get_action_bits(&self) -> usize {
        1
    }

    fn min_reward(&self) -> Reward {
        0
    }

    fn max_reward(&self) -> Reward {
        1
    }
}

pub struct SeededCoinFlipEnv {
    p: f64,
    obs: PerceptVal,
    rew: Reward,
    rng: RandomGenerator,
}

impl SeededCoinFlipEnv {
    pub fn new(p: f64) -> Self {
        let mut env = Self {
            p,
            obs: 0,
            rew: 0,
            rng: RandomGenerator::from_seed(1),
        };
        env.obs = env.next_observation();
        env
    }

    fn next_observation(&mut self) -> PerceptVal {
        if self.rng.gen_bool(self.p) { 1 } else { 0 }
    }
}

impl Environment for SeededCoinFlipEnv {
    fn perform_action(&mut self, action: Action) {
        self.obs = self.next_observation();
        self.rew = if action == self.obs { 1 } else { 0 };
    }

    fn get_observation(&self) -> PerceptVal {
        self.obs
    }

    fn get_reward(&self) -> Reward {
        self.rew
    }

    fn is_finished(&self) -> bool {
        false
    }

    fn get_observation_bits(&self) -> usize {
        1
    }

    fn get_reward_bits(&self) -> usize {
        1
    }

    fn get_action_bits(&self) -> usize {
        1
    }

    fn min_reward(&self) -> Reward {
        0
    }

    fn max_reward(&self) -> Reward {
        1
    }
}
