#![cfg(all(feature = "aixi", feature = "all-backends"))]

//! AIQI validation tests.

use infotheory::aixi::aiqi::{AiqiAgent, AiqiConfig};
use infotheory::aixi::common::DEFAULT_RANDOM_SEED;
use infotheory::aixi::environment::Environment;
mod support;
use infotheory::aixi::model::RateBackendBitPredictor;
use infotheory::api::{MixtureKind, MixtureSpec, RateBackend};
use std::sync::Arc;
use support::aixi_envs::{DeterministicBinaryEnv, SeededCoinFlipEnv};

fn base_config() -> AiqiConfig {
    let mut cfg = AiqiConfig::default();
    cfg.algorithm = "ctw".to_string();
    cfg.ct_depth = 8;
    cfg.observation_bits = 1;
    cfg.observation_stream_len = 1;
    cfg.reward_bits = 1;
    cfg.agent_actions = 2;
    cfg.min_reward = 0;
    cfg.max_reward = 1;
    cfg.reward_offset = 0;
    cfg.discount_gamma = 0.99;
    cfg.return_horizon = 2;
    cfg.return_bins = 8;
    cfg.augmentation_period = 2;
    cfg.history_prune_keep_steps = None;
    cfg.baseline_exploration = 0.01;
    cfg.random_seed = Some(11);
    cfg.rate_backend = None;
    cfg.rate_backend_max_order = 20;
    cfg.rwkv_model_path = None;
    cfg.rosa_max_order = None;
    cfg.zpaq_method = None;
    cfg
}

#[test]
fn aiqi_config_rejects_period_shorter_than_horizon() {
    let mut cfg = base_config();
    cfg.return_horizon = 3;
    cfg.augmentation_period = 2;
    let err = cfg.validate().expect_err("N < H must be rejected");
    assert!(err.contains("augmentation_period"));
}

#[test]
fn aiqi_config_rejects_non_power_of_two_return_bins() {
    let mut cfg = base_config();
    cfg.return_bins = 3;
    let err = cfg
        .validate()
        .expect_err("non-power-of-two return_bins must be rejected");
    assert!(err.contains("power of two"));
}

#[test]
fn aiqi_config_rejects_zpaq_algorithm_in_strict_mode() {
    let mut cfg = base_config();
    cfg.algorithm = "zpaq".to_string();
    let err = cfg
        .validate()
        .expect_err("strict AIQI should reject zpaq algorithm mode");
    assert!(err.contains("strict mode"));
}

#[test]
fn aiqi_config_allows_unknown_algorithm_when_rate_backend_overrides() {
    let mut cfg = base_config();
    cfg.algorithm = "unknown-backend-name".to_string();
    cfg.rate_backend = Some(RateBackend::Match {
        hash_bits: 16,
        min_len: 2,
        max_len: 16,
        base_mix: 0.05,
        confidence_scale: 1.0,
    });
    cfg.validate()
        .expect("rate_backend override should make algorithm non-binding");
}

#[test]
fn aiqi_config_allows_algorithm_zpaq_when_rate_backend_overrides() {
    let mut cfg = base_config();
    cfg.algorithm = "zpaq".to_string();
    cfg.rate_backend = Some(RateBackend::RosaPlus);
    cfg.validate()
        .expect("rate_backend override should ignore algorithm=zpaq");
}

#[test]
fn aiqi_config_rejects_zpaq_rate_backend_in_strict_mode() {
    let mut cfg = base_config();
    cfg.rate_backend = Some(RateBackend::Zpaq {
        method: infotheory::api::ZpaqMethodSpec::literal("1"),
    });
    let err = cfg
        .validate()
        .expect_err("strict AIQI should reject zpaq rate backend");
    assert!(err.contains("strict frozen conditioning"));
}

#[test]
fn aiqi_config_rejects_invalid_programmatic_mixture_rate_backend() {
    let mut cfg = base_config();
    cfg.rate_backend = Some(RateBackend::Mixture {
        spec: Arc::new(MixtureSpec::new(MixtureKind::Bayes, vec![])),
    });
    let err = cfg
        .validate()
        .expect_err("empty mixture backend should be rejected");
    assert!(err.contains("invalid rate_backend"));
    assert!(err.contains("must include at least one expert"));
}

#[test]
fn aiqi_coinflip_smoke_runs() {
    let mut agent = AiqiAgent::new(base_config()).expect("valid AIQI config");
    let mut env = SeededCoinFlipEnv::new(0.8);

    let mut total_reward = 0i64;
    for _ in 0..64 {
        let action = agent.get_planned_action();
        env.perform_action(action);
        let obs_stream = env.drain_observations();
        let rew = env.get_reward();
        agent
            .observe_transition(action, &obs_stream, rew)
            .expect("transition must be accepted");
        total_reward += rew;
    }

    assert!(total_reward >= 0);
}

#[test]
fn aiqi_learns_ctw_test_pattern() {
    let mut cfg = base_config();
    cfg.discount_gamma = 0.7;
    cfg.return_horizon = 4;
    cfg.augmentation_period = 4;
    cfg.ct_depth = 10;
    cfg.baseline_exploration = 1e-6;

    let mut agent = AiqiAgent::new(cfg).expect("valid AIQI config");
    let mut env = DeterministicBinaryEnv::new();

    let mut total_reward = 0i64;
    for _ in 0..120 {
        let action = agent.get_planned_action();
        env.perform_action(action);
        let obs_stream = env.drain_observations();
        let rew = env.get_reward();
        agent
            .observe_transition(action, &obs_stream, rew)
            .expect("transition must be accepted");
        total_reward += rew;
    }

    assert!(
        total_reward > 50,
        "AIQI failed to learn DeterministicBinaryEnv pattern; total_reward={total_reward}"
    );
}

#[test]
fn aiqi_with_generic_rate_backend_smoke_runs() {
    let mut cfg = base_config();
    cfg.rate_backend = Some(RateBackend::Match {
        hash_bits: 16,
        min_len: 2,
        max_len: 16,
        base_mix: 0.05,
        confidence_scale: 1.0,
    });
    cfg.rate_backend_max_order = 8;

    let mut agent = AiqiAgent::new(cfg).expect("valid AIQI config");
    let mut env = SeededCoinFlipEnv::new(0.7);

    for _ in 0..24 {
        let action = agent.get_planned_action();
        env.perform_action(action);
        let obs_stream = env.drain_observations();
        let rew = env.get_reward();
        agent
            .observe_transition(action, &obs_stream, rew)
            .expect("transition must be accepted");
    }

    assert!(agent.steps_observed() >= 24);
}

#[test]
fn aiqi_with_rosa_generic_planner_smoke_runs() {
    let mut cfg = base_config();
    cfg.algorithm = "rosaplus".to_string();
    cfg.rosa_max_order = Some(8);

    let mut agent = AiqiAgent::new(cfg).expect("valid AIQI config");
    let mut env = SeededCoinFlipEnv::new(0.7);

    for _ in 0..24 {
        let action = agent.get_planned_action();
        env.perform_action(action);
        let obs_stream = env.drain_observations();
        let rew = env.get_reward();
        agent
            .observe_transition(action, &obs_stream, rew)
            .expect("transition must be accepted");
    }

    assert!(agent.steps_observed() >= 24);
}

#[test]
fn aiqi_optional_history_pruning_smoke_runs() {
    let mut cfg = base_config();
    cfg.return_horizon = 3;
    cfg.augmentation_period = 4;
    cfg.history_prune_keep_steps = Some(16);

    let mut agent = AiqiAgent::new(cfg).expect("valid AIQI config");
    let mut env = SeededCoinFlipEnv::new(0.7);

    for _ in 0..128 {
        let action = agent.get_planned_action();
        env.perform_action(action);
        let obs_stream = env.drain_observations();
        let rew = env.get_reward();
        agent
            .observe_transition(action, &obs_stream, rew)
            .expect("transition must be accepted");
    }

    assert_eq!(agent.steps_observed(), 128);
}

#[test]
fn aiqi_seeded_policy_is_reproducible() {
    let mut cfg = base_config();
    cfg.baseline_exploration = 0.35;
    cfg.random_seed = Some(987654321);

    let mut a = AiqiAgent::new(cfg.clone()).expect("valid AIQI config");
    let mut b = AiqiAgent::new(cfg).expect("valid AIQI config");

    for step in 0..128usize {
        let act_a = a.get_planned_action();
        let act_b = b.get_planned_action();
        assert_eq!(act_a, act_b, "action mismatch at step {step}");

        let obs = [(step % 2) as u64];
        let rew = (step % 2) as i64;
        a.observe_transition(act_a, &obs, rew)
            .expect("transition should be accepted");
        b.observe_transition(act_b, &obs, rew)
            .expect("transition should be accepted");
    }
}

#[test]
fn aiqi_omitted_seed_matches_explicit_default_seed() {
    let mut cfg_omitted = base_config();
    cfg_omitted.random_seed = None;
    cfg_omitted.baseline_exploration = 0.2;
    let mut cfg_explicit = cfg_omitted.clone();
    cfg_explicit.random_seed = Some(DEFAULT_RANDOM_SEED);

    let mut a = AiqiAgent::new(cfg_omitted).expect("agent with omitted seed");
    let mut b = AiqiAgent::new(cfg_explicit).expect("agent with explicit default seed");

    assert_eq!(a.resolved_random_seed(), DEFAULT_RANDOM_SEED);
    assert_eq!(b.resolved_random_seed(), DEFAULT_RANDOM_SEED);

    for step in 0..96usize {
        let act_a = a.get_planned_action();
        let act_b = b.get_planned_action();
        assert_eq!(act_a, act_b, "action mismatch at step {step}");

        let obs = [((step + 1) % 2) as u64];
        let rew = (step % 2) as i64;
        a.observe_transition(act_a, &obs, rew)
            .expect("transition should be accepted");
        b.observe_transition(act_b, &obs, rew)
            .expect("transition should be accepted");
    }
}

#[test]
fn aiqi_different_seeds_can_change_exploration_trace() {
    let mut cfg_a = base_config();
    cfg_a.baseline_exploration = 0.45;
    cfg_a.random_seed = Some(11);
    let mut cfg_b = cfg_a.clone();
    cfg_b.random_seed = Some(12);

    let mut a = AiqiAgent::new(cfg_a).expect("agent A");
    let mut b = AiqiAgent::new(cfg_b).expect("agent B");

    let mut diverged = false;
    for step in 0..128usize {
        let act_a = a.get_planned_action();
        let act_b = b.get_planned_action();
        if act_a != act_b {
            diverged = true;
            break;
        }
        let obs = [(step % 2) as u64];
        let rew = ((step + 1) % 2) as i64;
        a.observe_transition(act_a, &obs, rew)
            .expect("transition should be accepted");
        b.observe_transition(act_b, &obs, rew)
            .expect("transition should be accepted");
    }

    assert!(
        diverged,
        "different random_seed values should be able to produce different exploratory traces"
    );
}

#[test]
fn rate_backend_bit_predictor_rejects_zpaq_backend() {
    let err = match RateBackendBitPredictor::new(
        RateBackend::Zpaq {
            method: infotheory::api::ZpaqMethodSpec::literal("1"),
        },
        8,
    ) {
        Ok(_) => panic!("zpaq must be rejected in RateBackendBitPredictor"),
        Err(err) => err,
    };
    assert!(err.contains("does not support zpaq backends"));
}

#[cfg(feature = "backend-rwkv")]
#[test]
fn aiqi_config_rejects_rwkv_without_model_path_when_no_rate_backend() {
    let mut cfg = base_config();
    cfg.algorithm = "rwkv7".to_string();
    cfg.rwkv_model_path = None;
    cfg.rate_backend = None;

    let err = match AiqiAgent::new(cfg) {
        Ok(_) => panic!("expected rwkv7 config to fail without a model path"),
        Err(err) => err,
    };
    assert!(
        err.contains("rwkv_model_path") || err.contains("backend-rwkv"),
        "unexpected error: {err}"
    );
}

#[cfg(feature = "backend-rwkv")]
#[test]
fn aiqi_config_allows_rwkv_without_model_path_with_rate_backend_override() {
    let mut cfg = base_config();
    cfg.algorithm = "rwkv7".to_string();
    cfg.rwkv_model_path = None;
    cfg.rate_backend = Some(RateBackend::RosaPlus);

    cfg.validate()
        .expect("rate_backend override should avoid requiring rwkv_model_path");
}
