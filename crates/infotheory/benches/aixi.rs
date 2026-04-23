#[cfg(not(feature = "aixi-gameengine"))]
fn main() {
    eprintln!("bench 'aixi' requires feature 'aixi-gameengine'");
}

#[cfg(feature = "aixi-gameengine")]
mod bench_impl {
    use infotheory::aixi::agent::{Agent, AgentConfig};
    use infotheory::aixi::environment::Environment;
    use infotheory::aixi::gameengine::build_builtin_environment;
    use infotheory::api::{
        MixtureExpertSpec, MixtureKind, MixtureScheduleMode, MixtureSpec, RateBackend,
    };
    use infotheory::spec::BuiltinEnvironmentSpec;
    use std::sync::Arc;
    use std::time::{Duration, Instant};

    fn env_usize(name: &str, default: usize) -> usize {
        std::env::var(name)
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .filter(|&value| value > 0)
            .unwrap_or(default)
    }

    fn bench_agent(
        mut agent: Agent,
        mut env: Box<dyn Environment>,
        cycles: usize,
        warmup: usize,
    ) -> Duration {
        let mut prev_action = 0u64;
        let mut obs_stream = env.drain_observations();
        let mut rew = env.get_reward();

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

    pub(super) fn run() {
        let cycles = env_usize("AIXI_BENCH_CYCLES", 2_000);
        let warmup = env_usize("AIXI_BENCH_WARMUP", 200);
        let env_name = "blackjack";

        let base_cfg = |algorithm: &str| AgentConfig {
            algorithm: algorithm.to_string(),
            ct_depth: 32,
            agent_horizon: 5,
            observation_bits: 64,
            observation_stream_len: 4,
            observation_key_mode: infotheory::aixi::common::ObservationKeyMode::FullStream,
            reward_bits: 2,
            agent_actions: 2,
            num_simulations: 400,
            exploration_exploitation_ratio: 1.4,
            discount_gamma: 1.0,
            min_reward: -1,
            max_reward: 1,
            reward_offset: 1,
            random_seed: Some(1),
            rate_backend: None,
            rate_backend_max_order: 20,
            rwkv_model_path: None,
            rwkv_method: None,
            mamba_model_path: None,
            mamba_method: None,
            rosa_max_order: Some(20),
            zpaq_method: None,
        };

        let make_mixture =
            |kind: MixtureKind, alpha: f64, schedule: MixtureScheduleMode| RateBackend::Mixture {
                spec: Arc::new(
                    MixtureSpec::new(
                        kind,
                        vec![
                            MixtureExpertSpec {
                                name: Some("ctw".to_string()),
                                log_prior: 0.0,
                                max_order: -1,
                                backend: RateBackend::Ctw { depth: 32 },
                            },
                            MixtureExpertSpec {
                                name: Some("rosa".to_string()),
                                log_prior: 0.0,
                                max_order: 20,
                                backend: RateBackend::RosaPlus,
                            },
                        ],
                    )
                    .with_schedule(schedule)
                    .with_alpha(alpha),
                ),
            };

        let rate_backend_cfg = |backend: RateBackend| {
            let mut cfg = base_cfg("mixture");
            cfg.rate_backend = Some(backend);
            cfg
        };

        let benches = [
            ("fac-ctw", base_cfg("fac-ctw")),
            ("rosa", base_cfg("rosa")),
            ("rate-ctw", rate_backend_cfg(RateBackend::Ctw { depth: 32 })),
            ("rate-rosa", rate_backend_cfg(RateBackend::RosaPlus)),
            (
                "mix-bayes",
                rate_backend_cfg(make_mixture(
                    MixtureKind::Bayes,
                    0.01,
                    MixtureScheduleMode::Default,
                )),
            ),
            (
                "mix-switch",
                rate_backend_cfg(make_mixture(
                    MixtureKind::Switching,
                    0.17,
                    MixtureScheduleMode::Default,
                )),
            ),
            (
                "mix-switch-thm",
                rate_backend_cfg(make_mixture(
                    MixtureKind::Switching,
                    0.99,
                    MixtureScheduleMode::Theorem,
                )),
            ),
            (
                "mix-convex",
                rate_backend_cfg(make_mixture(
                    MixtureKind::Convex,
                    1.25,
                    MixtureScheduleMode::Default,
                )),
            ),
            (
                "mix-convex-thm",
                rate_backend_cfg(make_mixture(
                    MixtureKind::Convex,
                    7.5,
                    MixtureScheduleMode::Theorem,
                )),
            ),
        ];

        println!(
            "MC-AIXI benchmark (env={}, warmup={}, cycles={})",
            env_name, warmup, cycles
        );
        for (name, cfg) in benches {
            let env = build_builtin_environment(BuiltinEnvironmentSpec::Blackjack)
                .expect("blackjack builtin env");
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
}

#[cfg(feature = "aixi-gameengine")]
fn main() {
    bench_impl::run();
}
