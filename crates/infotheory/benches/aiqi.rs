use infotheory::aixi::aiqi::{AiqiAgent, AiqiConfig};
use infotheory::api::{MixtureExpertSpec, MixtureKind, MixtureSpec, RateBackend};
use std::hint::black_box;
use std::sync::Arc;
use std::time::Instant;

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|&value| value > 0)
        .unwrap_or(default)
}

fn base_cfg(algorithm: &str) -> AiqiConfig {
    let mut cfg = AiqiConfig::default();
    cfg.algorithm = algorithm.to_string();
    cfg.ct_depth = 12;
    cfg.observation_bits = 1;
    cfg.observation_stream_len = 1;
    cfg.reward_bits = 1;
    cfg.agent_actions = 2;
    cfg.min_reward = 0;
    cfg.max_reward = 1;
    cfg.reward_offset = 0;
    cfg.discount_gamma = 0.99;
    cfg.return_horizon = 4;
    cfg.return_bins = 8;
    cfg.augmentation_period = 4;
    cfg.history_prune_keep_steps = None;
    cfg.baseline_exploration = 1e-12;
    cfg.random_seed = Some(7);
    cfg.rate_backend = None;
    cfg.rate_backend_max_order = 8;
    cfg.rwkv_model_path = None;
    cfg.rosa_max_order = None;
    cfg.zpaq_method = None;
    cfg
}

fn mixture_backend() -> RateBackend {
    let experts = vec![
        {
            let mut expert = MixtureExpertSpec::new(RateBackend::Ctw { depth: 10 });
            expert.name = Some("ctw".to_string());
            expert.log_prior = 0.0;
            expert.max_order = -1;
            expert
        },
        {
            let mut expert = MixtureExpertSpec::new(RateBackend::RosaPlus);
            expert.name = Some("rosa".to_string());
            expert.log_prior = 0.0;
            expert.max_order = 8;
            expert
        },
    ];
    RateBackend::Mixture {
        spec: Arc::new(MixtureSpec::new(MixtureKind::Bayes, experts).with_alpha(0.05)),
    }
}

fn seed_history(agent: &mut AiqiAgent, steps: usize) {
    for step in 0..steps {
        let action = (step % 2) as u64;
        let obs = [(step % 2) as u64];
        let rew = (step % 2) as i64;
        agent
            .observe_transition(action, &obs, rew)
            .expect("seed transition should be accepted");
    }
}

fn main() {
    let iterations = env_usize("AIQI_BENCH_ITERS", 1_000);
    let seed_steps = env_usize("AIQI_BENCH_HISTORY", 128);

    let rate_backend_cfg = |backend: RateBackend| {
        let mut cfg = base_cfg("ignored-by-rate-backend");
        cfg.rate_backend = Some(backend);
        cfg
    };

    let benches = [
        ("ctw", base_cfg("ctw")),
        ("rosaplus", base_cfg("rosaplus")),
        ("rate-rosa", rate_backend_cfg(RateBackend::RosaPlus)),
        ("mix-bayes", rate_backend_cfg(mixture_backend())),
    ];

    println!(
        "AIQI planning benchmark (seed_steps={}, iterations={})",
        seed_steps, iterations
    );

    for (name, cfg) in benches {
        let mut agent = AiqiAgent::new(cfg).expect("valid AIQI config");
        seed_history(&mut agent, seed_steps);

        let now = Instant::now();
        for _ in 0..iterations {
            black_box(agent.get_planned_action());
        }
        let elapsed = now.elapsed();
        let ns_per_iter = (elapsed.as_nanos() as f64) / (iterations as f64);
        let iters_per_s = (iterations as f64) / elapsed.as_secs_f64().max(1e-12);

        println!(
            "{:>12}: {:>8.3} ms total | {:>10.1} ns/plan | {:>10.1} plans/s",
            name,
            elapsed.as_secs_f64() * 1e3,
            ns_per_iter,
            iters_per_s
        );
    }
}
