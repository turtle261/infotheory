#![cfg(feature = "aixi")]

use infotheory::aixi::common::{Action, ObservationKeyMode, Reward};
use infotheory::aixi::mcts::AgentSimulator;

struct NormRewardHarness {
    discount_gamma: f64,
    horizon: usize,
    min_reward: Reward,
    max_reward: Reward,
}

fn approx_eq(a: f64, b: f64, eps: f64) {
    assert!(
        (a - b).abs() <= eps,
        "expected {a} ≈ {b} (|diff|={})",
        (a - b).abs()
    );
}

impl AgentSimulator for NormRewardHarness {
    fn get_num_actions(&self) -> usize {
        2
    }

    fn get_num_observation_bits(&self) -> usize {
        1
    }

    fn observation_key_mode(&self) -> ObservationKeyMode {
        ObservationKeyMode::FullStream
    }

    fn get_num_reward_bits(&self) -> usize {
        8
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

    fn reward_offset(&self) -> i64 {
        (-self.min_reward).max(0)
    }

    fn discount_gamma(&self) -> f64 {
        self.discount_gamma
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
        Box::new(Self {
            discount_gamma: self.discount_gamma,
            horizon: self.horizon,
            min_reward: self.min_reward,
            max_reward: self.max_reward,
        })
    }
}

fn mk_agent(
    discount_gamma: f64,
    horizon: usize,
    min_reward: i64,
    max_reward: i64,
) -> NormRewardHarness {
    NormRewardHarness {
        discount_gamma,
        horizon,
        min_reward,
        max_reward,
    }
}

#[test]
fn norm_reward_undiscounted_hits_endpoints() {
    let horizon = 5;
    let min = -2;
    let max = 6;
    let agent = mk_agent(1.0, horizon, min, max);

    let sum = horizon as f64;
    let min_cum = (min as f64) * sum;
    let max_cum = (max as f64) * sum;

    let z0 = agent.norm_reward(min_cum);
    let z1 = agent.norm_reward(max_cum);

    approx_eq(z0, 0.0, 1e-12);
    approx_eq(z1, 1.0, 1e-12);
}

#[test]
fn norm_reward_discounted_hits_endpoints() {
    let horizon = 10;
    let min = -1;
    let max = 3;
    let gamma = 0.7;
    let agent = mk_agent(gamma, horizon, min, max);

    let sum = (1.0 - gamma.powi(horizon as i32)) / (1.0 - gamma);
    let min_cum = (min as f64) * sum;
    let max_cum = (max as f64) * sum;

    let z0 = agent.norm_reward(min_cum);
    let z1 = agent.norm_reward(max_cum);

    approx_eq(z0, 0.0, 1e-10);
    approx_eq(z1, 1.0, 1e-10);
}

#[test]
fn norm_reward_midpoint_is_half() {
    let horizon = 7;
    let min = -4;
    let max = 4;
    let gamma = 0.5;
    let agent = mk_agent(gamma, horizon, min, max);

    let sum = (1.0 - gamma.powi(horizon as i32)) / (1.0 - gamma);
    let mid_cum = ((min + max) as f64 / 2.0) * sum;

    let z = agent.norm_reward(mid_cum);
    approx_eq(z, 0.5, 1e-10);
}
