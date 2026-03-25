use infotheory::aixi::agent::{Agent, AgentConfig};
use infotheory::aixi::common::ObservationKeyMode;
use infotheory::aixi::mcts::AgentSimulator;

fn approx_eq(a: f64, b: f64, eps: f64) {
    assert!(
        (a - b).abs() <= eps,
        "expected {a} ≈ {b} (|diff|={})",
        (a - b).abs()
    );
}

fn mk_agent(discount_gamma: f64, horizon: usize, min_reward: i64, max_reward: i64) -> Agent {
    Agent::new(AgentConfig {
        algorithm: "ctw".to_string(),
        ct_depth: 8,
        agent_horizon: horizon,
        observation_bits: 1,
        observation_stream_len: 1,
        observation_key_mode: ObservationKeyMode::FullStream,
        reward_bits: 8,
        agent_actions: 2,
        num_simulations: 1,
        exploration_exploitation_ratio: 1.0,
        discount_gamma,
        min_reward,
        max_reward,
        reward_offset: 0,
        random_seed: Some(13),
        rwkv_model_path: None,
        rosa_max_order: None,
        zpaq_method: None,
    })
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
