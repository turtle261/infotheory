//! AIQI validation tests.

use infotheory::RateBackend;
use infotheory::aixi::aiqi::{AiqiAgent, AiqiConfig};
use infotheory::aixi::environment::{CoinFlip, CtwTest, Environment};
use infotheory::aixi::model::RateBackendBitPredictor;

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
fn aiqi_config_rejects_zpaq_algorithm_in_strict_mode() {
    let mut cfg = base_config();
    cfg.algorithm = "zpaq".to_string();
    let err = cfg
        .validate()
        .expect_err("strict AIQI should reject zpaq algorithm mode");
    assert!(err.contains("strict mode"));
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
#[should_panic(expected = "does not support zpaq backends")]
fn rate_backend_bit_predictor_rejects_zpaq_backend() {
    let _ = RateBackendBitPredictor::new(
        RateBackend::Zpaq {
            method: "1".to_string(),
        },
        8,
    );
}
