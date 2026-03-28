//! AIQI validation tests.

use infotheory::aixi::aiqi::{AiqiAgent, AiqiConfig};
use infotheory::aixi::environment::{CoinFlip, CtwTest, Environment};
use infotheory::aixi::model::RateBackendBitPredictor;
use infotheory::{MixtureKind, MixtureSpec, RateBackend};
use std::sync::Arc;

fn base_config() -> AiqiConfig {
    AiqiConfig {
        algorithm: "ac-ctw".to_string(),
        ct_depth: 8,
        observation_bits: 1,
        observation_stream_len: 1,
        reward_bits: 1,
        agent_actions: 2,
        min_reward: 0,
        max_reward: 1,
        reward_offset: 0,
        discount_gamma: 0.99,
        return_horizon: 2,
        return_bins: 8,
        augmentation_period: 2,
        history_prune_keep_steps: None,
        baseline_exploration: 0.01,
        random_seed: Some(11),
        rate_backend: None,
        rate_backend_max_order: 20,
        rwkv_model_path: None,
        rosa_max_order: None,
        zpaq_method: None,
    }
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
        method: "1".to_string(),
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
    let mut env = CoinFlip::new(0.8);

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
    let mut env = CtwTest::new();

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
        "AIQI failed to learn CtwTest pattern; total_reward={total_reward}"
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
    let mut env = CoinFlip::new(0.7);

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
    cfg.algorithm = "rosa".to_string();
    cfg.rosa_max_order = Some(8);

    let mut agent = AiqiAgent::new(cfg).expect("valid AIQI config");
    let mut env = CoinFlip::new(0.7);

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
    let mut env = CoinFlip::new(0.7);

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
fn rate_backend_bit_predictor_rejects_zpaq_backend() {
    let err = match RateBackendBitPredictor::new(
        RateBackend::Zpaq {
            method: "1".to_string(),
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
    cfg.algorithm = "rwkv".to_string();
    cfg.rwkv_model_path = None;
    cfg.rate_backend = None;

    let err = cfg
        .validate()
        .expect_err("algorithm=rwkv without path and without rate_backend override must fail");
    assert!(err.contains("rwkv_model_path"));
}

#[cfg(feature = "backend-rwkv")]
#[test]
fn aiqi_config_allows_rwkv_without_model_path_with_rate_backend_override() {
    let mut cfg = base_config();
    cfg.algorithm = "rwkv".to_string();
    cfg.rwkv_model_path = None;
    cfg.rate_backend = Some(RateBackend::RosaPlus);

    cfg.validate()
        .expect("rate_backend override should avoid requiring rwkv_model_path");
}
