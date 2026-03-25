use infotheory::aixi::agent::{Agent, AgentConfig};
use infotheory::aixi::environment::{BiasedRockPaperScissor, Environment};
use std::time::{Duration, Instant};

fn bench_agent(
    mut agent: Agent,
    mut env: Box<dyn Environment>,
    cycles: usize,
    warmup: usize,
) -> Duration {
    let mut prev_action = 0u64;
    let mut obs_stream = env.drain_observations();
    let mut rew = env.get_reward();

    // Warmup
    for _ in 0..warmup {
        agent.model_update_percept_stream(&obs_stream, rew);
        let action = agent.get_planned_action(&obs_stream, rew, prev_action);
        agent.model_update_action_external(action);
        env.perform_action(action);
        obs_stream = env.drain_observations();
        rew = env.get_reward();
        prev_action = action;
    }

    let now = Instant::now();
    for _ in 0..cycles {
        agent.model_update_percept_stream(&obs_stream, rew);
        let action = agent.get_planned_action(&obs_stream, rew, prev_action);
        agent.model_update_action_external(action);
        env.perform_action(action);
        obs_stream = env.drain_observations();
        rew = env.get_reward();
        prev_action = action;
    }
    now.elapsed()
}

fn main() {
    // Keep these fixed so backends are comparable.
    let cycles = 2_000usize;
    let warmup = 200usize;

    // Environment chosen because it has non-trivial observation+reward bits (2+2) and 3 actions.
    let env_name = "biased-rock-paper-scissor";

    // Benchmark config: moderate horizon + simulations so MCTS is the dominant cost.
    // reward_offset is required for correct unsigned reward encoding.
    let base_cfg = |algorithm: &str| AgentConfig {
        algorithm: algorithm.to_string(),
        ct_depth: 32,
        agent_horizon: 5,
        observation_bits: 2,
        observation_stream_len: 1,
        observation_key_mode: infotheory::aixi::common::ObservationKeyMode::FullStream,
        reward_bits: 2,
        agent_actions: 3,
        num_simulations: 400,
        exploration_exploitation_ratio: 1.4,
        discount_gamma: 1.0,
        min_reward: -1,
        max_reward: 1,
        reward_offset: 1,
        random_seed: Some(1),
        rwkv_model_path: None,
        rwkv_method: None,
        mamba_model_path: None,
        mamba_method: None,
        rosa_max_order: Some(20),
        zpaq_method: None,
    };

    let benches = [("fac-ctw", base_cfg("fac-ctw")), ("rosa", base_cfg("rosa"))];

    println!(
        "MC-AIXI benchmark (env={}, warmup={}, cycles={})",
        env_name, warmup, cycles
    );
    for (name, cfg) in benches {
        // Fresh env per backend to keep interaction distribution identical.
        // (Environment is stochastic; we intentionally measure typical runtime not identical traces.)
        let env: Box<dyn Environment> = Box::new(BiasedRockPaperScissor::new());
        let agent = Agent::new(cfg.clone());
        let elapsed = bench_agent(agent, env, cycles, warmup);

        let ns_per_cycle = (elapsed.as_nanos() as f64) / (cycles as f64);
        let cycles_per_s = (cycles as f64) / elapsed.as_secs_f64().max(1e-12);
        println!(
            "{:>7}: {:>8.3} ms total | {:>10.1} ns/cycle | {:>10.1} cycles/s",
            name,
            elapsed.as_secs_f64() * 1e3,
            ns_per_cycle,
            cycles_per_s
        );
    }
}
