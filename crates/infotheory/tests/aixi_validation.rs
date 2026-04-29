#![cfg(all(feature = "aixi", feature = "all-backends"))]

//! AIXI Module Validation Tests
//!
//! Tests for predictors, environments, and agents.

use infotheory::aixi::agent::{Agent, AgentConfig, AgentError};
use infotheory::aixi::common::{Action, ActionAlphabet, DEFAULT_RANDOM_SEED, ObservationKeyMode};
use infotheory::aixi::environment::Environment;
mod support;
use infotheory::aixi::model::{
    CtwPredictor, Predictor, RateBackendBitPredictor, RateBackendBitPredictorConfig, RosaPredictor,
};
use infotheory::api::{
    MAX_MIXTURE_NESTING, MixtureExpertSpec, MixtureKind, MixtureSpec, RateBackend,
};
use std::sync::Arc;
use support::aixi_envs::{DeterministicBinaryEnv, SeededCoinFlipEnv};

// ============================================================================
// Predictor Consistency Tests
// ============================================================================

fn test_predictor_sum_to_one(mut predictor: Box<dyn Predictor>, name: &str) {
    // Feed some history
    for &sym in &[true, false, true, true, false] {
        predictor.update(sym);
    }

    let p_true = predictor.predict_prob(true);
    let p_false = predictor.predict_prob(false);

    // Check they sum to 1.0 (binary predictor)
    let sum = p_true + p_false;
    println!("{name}: P(1)={p_true:.6}, P(0)={p_false:.6}, Sum={sum:.6}");
    assert!(
        (sum - 1.0).abs() < 1e-6,
        "{name}: Probabilities must sum to 1.0, got {p_true} + {p_false} = {sum}"
    );

    // Check range
    assert!(
        (0.0..=1.0).contains(&p_true),
        "{name}: Prob out of range: {p_true}"
    );
}

#[test]
fn ctw_probabilities_valid() {
    test_predictor_sum_to_one(Box::new(CtwPredictor::new(8)), "CTW");
}

#[test]
fn rosa_probabilities_valid() {
    test_predictor_sum_to_one(Box::new(RosaPredictor::new(8)), "ROSA");
}

fn test_predictor_revert(mut predictor: Box<dyn Predictor>, name: &str) {
    let history = [true, false, true, true, false, false, true];

    // Update all
    for &sym in &history {
        predictor.update(sym);
    }
    let prob_after_updates = predictor.predict_prob(true);

    // Revert all
    for _ in &history {
        predictor.revert();
    }

    // Should be back to initial state (approx 0.5 for uniform prior)
    let prob_reverted = predictor.predict_prob(true);

    println!("{name}: After full revert, p(1) = {prob_reverted}");
    assert!(
        (prob_reverted - 0.5).abs() < 0.1,
        "{name}: Reverted predictor should be roughly uninformed (0.5), got {prob_reverted}"
    );

    // Re-apply and check we get same result as before
    for &sym in &history {
        predictor.update(sym);
    }
    let prob_redo = predictor.predict_prob(true);
    assert!(
        (prob_redo - prob_after_updates).abs() < 1e-9,
        "{name}: Deterministic replay failed. {prob_redo} != {prob_after_updates}"
    );
}

#[test]
fn ctw_update_revert_consistency() {
    test_predictor_revert(Box::new(CtwPredictor::new(8)), "CTW");
}

#[test]
fn rosa_update_revert_consistency() {
    test_predictor_revert(Box::new(RosaPredictor::new(8)), "ROSA");
}

fn nested_generic_backend() -> RateBackend {
    let inner = MixtureSpec::new(
        MixtureKind::Bayes,
        vec![
            {
                let mut expert = MixtureExpertSpec::new(RateBackend::Ctw { depth: 6 });
                expert.name = Some("ctw".to_string());
                expert.log_prior = 0.0;
                expert
            },
            {
                let mut expert = MixtureExpertSpec::new(RateBackend::Match {
                    hash_bits: 18,
                    min_len: 2,
                    max_len: 32,
                    base_mix: 0.05,
                    confidence_scale: 1.0,
                });
                expert.name = Some("match".to_string());
                expert.log_prior = 0.0;
                expert
            },
        ],
    )
    .with_alpha(0.03);
    let outer = MixtureSpec::new(
        MixtureKind::Convex,
        vec![
            {
                let mut expert = MixtureExpertSpec::new(RateBackend::Mixture {
                    spec: Arc::new(inner),
                });
                expert.name = Some("nested".to_string());
                expert.log_prior = 0.0;
                expert
            },
            {
                let mut expert = MixtureExpertSpec::new(RateBackend::Ppmd {
                    order: 4,
                    memory_mb: 8,
                });
                expert.name = Some("ppmd".to_string());
                expert.log_prior = 0.0;
                expert
            },
        ],
    )
    .with_alpha(1.25);
    RateBackend::Mixture {
        spec: Arc::new(outer),
    }
}

fn predictor_snapshot(predictor: &mut dyn Predictor) -> (f64, f64) {
    (predictor.predict_prob(false), predictor.predict_prob(true))
}

fn assert_snapshot_eq(actual: (f64, f64), expected: (f64, f64), label: &str) {
    assert!(
        (actual.0 - expected.0).abs() < 1e-12 && (actual.1 - expected.1).abs() < 1e-12,
        "{label}: expected {:?}, got {:?}",
        expected,
        actual
    );
}

#[test]
fn rate_backend_bit_predictor_roundtrips_nested_mixtures() {
    let config =
        RateBackendBitPredictorConfig::compile(nested_generic_backend(), 1e-12).expect("config");
    let mut predictor = RateBackendBitPredictor::new(config).expect("valid predictor");

    let initial = predictor_snapshot(&mut predictor);

    predictor.update(true);
    let after_update = predictor_snapshot(&mut predictor);
    predictor.revert();
    assert_snapshot_eq(
        predictor_snapshot(&mut predictor),
        initial,
        "revert after update",
    );

    predictor.update(true);
    assert_snapshot_eq(
        predictor_snapshot(&mut predictor),
        after_update,
        "redo after update",
    );

    predictor.update_history(false);
    let after_frozen = predictor_snapshot(&mut predictor);
    predictor.pop_history();
    assert_snapshot_eq(
        predictor_snapshot(&mut predictor),
        after_update,
        "pop_history after frozen update",
    );

    predictor.update_history(false);
    assert_snapshot_eq(
        predictor_snapshot(&mut predictor),
        after_frozen,
        "redo after frozen update",
    );
}

#[test]
fn rate_backend_bit_predictor_roundtrips_sequitur_backend() {
    let config =
        RateBackendBitPredictorConfig::compile(RateBackend::Sequitur { context_bytes: 32 }, 1e-12)
            .expect("config");
    let mut predictor = RateBackendBitPredictor::new(config).expect("valid sequitur predictor");

    let initial = predictor_snapshot(&mut predictor);

    predictor.update(true);
    let after_update = predictor_snapshot(&mut predictor);
    predictor.revert();
    assert_snapshot_eq(
        predictor_snapshot(&mut predictor),
        initial,
        "sequitur revert after update",
    );

    predictor.update(true);
    assert_snapshot_eq(
        predictor_snapshot(&mut predictor),
        after_update,
        "sequitur redo after update",
    );

    predictor.update_history(false);
    let after_frozen = predictor_snapshot(&mut predictor);
    predictor.pop_history();
    assert_snapshot_eq(
        predictor_snapshot(&mut predictor),
        after_update,
        "sequitur pop_history after frozen update",
    );

    predictor.update_history(false);
    assert_snapshot_eq(
        predictor_snapshot(&mut predictor),
        after_frozen,
        "sequitur redo after frozen update",
    );
}

// ============================================================================
// Environment Tests
// ============================================================================

#[test]
fn ctw_test_env_is_deterministic() {
    let mut env1 = DeterministicBinaryEnv::new();
    let mut env2 = DeterministicBinaryEnv::new();

    for i in 0..50 {
        let action = (i % 2) as Action;
        env1.perform_action(action);
        env2.perform_action(action);

        assert_eq!(
            env1.get_observation(),
            env2.get_observation(),
            "Obs mismatch at step {i}"
        );
        assert_eq!(
            env1.get_reward(),
            env2.get_reward(),
            "Reward mismatch at step {i}"
        );
    }
}

// ============================================================================
// Agent / MCTS Tests
// ============================================================================

fn run_agent_env<T: Environment>(agent: &mut Agent, mut env: T, cycles: usize) -> f64 {
    let mut total_reward = 0.0;
    let mut obs_stream = env.drain_observations();
    let mut prev_rew = env.get_reward();
    let mut prev_act = 0;

    for _ in 0..cycles {
        agent.model_update_percept_stream(&obs_stream, prev_rew);
        let action = agent.get_planned_action(&obs_stream, prev_rew, prev_act);

        // Update model with chosen action (so model sees: ...p a p a p a...)
        agent.model_update_action_external(action);

        env.perform_action(action);

        obs_stream = env.drain_observations();
        let rew = env.get_reward();

        // Update model with observed percept stream
        agent.model_update_percept_stream(&obs_stream, rew);

        total_reward += rew as f64;
        prev_rew = rew;
        prev_act = action;

        if env.is_finished() {
            break;
        }
    }
    total_reward
}

fn generic_agent_config(rate_backend: RateBackend) -> AgentConfig {
    let mut cfg = AgentConfig::default();
    cfg.rate_backend = rate_backend;
    cfg.agent_horizon = 5;
    cfg.observation_bits = 1;
    cfg.observation_stream_len = 1;
    cfg.observation_key_mode = ObservationKeyMode::FullStream;
    cfg.reward_bits = 1;
    cfg.agent_actions =
        ActionAlphabet::try_from_usize(2).expect("test fixture action alphabet must be valid");
    cfg.num_simulations = 60;
    cfg.exploration_exploitation_ratio = 1.4;
    cfg.discount_gamma = 1.0;
    cfg.min_reward = 0;
    cfg.max_reward = 1;
    cfg.reward_offset = 0;
    cfg.random_seed = Some(2026);
    cfg
}

fn mixture_backend(kind: MixtureKind) -> RateBackend {
    let experts = vec![
        {
            let mut expert = MixtureExpertSpec::new(RateBackend::Ctw { depth: 8 });
            expert.name = Some("ctw".to_string());
            expert.log_prior = 0.0;
            expert
        },
        {
            let mut expert = MixtureExpertSpec::new(RateBackend::RosaPlus { max_order: 8 });
            expert.name = Some("rosa".to_string());
            expert.log_prior = 0.0;
            expert
        },
    ];
    let alpha = match kind {
        MixtureKind::Switching => 0.05,
        MixtureKind::Convex => 1.25,
        _ => 0.03,
    };
    RateBackend::Mixture {
        spec: Arc::new(MixtureSpec::new(kind, experts).with_alpha(alpha)),
    }
}

fn deeply_nested_bayes_backend(depth: usize) -> RateBackend {
    let mut backend = RateBackend::Ctw { depth: 4 };
    for level in 0..depth {
        backend = RateBackend::Mixture {
            spec: Arc::new(MixtureSpec::new(
                MixtureKind::Bayes,
                vec![{
                    let mut expert = MixtureExpertSpec::new(backend);
                    expert.name = Some(format!("level-{level}"));
                    expert.log_prior = 0.0;
                    expert
                }],
            )),
        };
    }
    backend
}

#[test]
fn agent_solves_ctw_test_environment() {
    let mut config = AgentConfig::default();
    config.rate_backend = RateBackend::FacCtw {
        base_depth: 8,
        num_percept_bits: 2,
        encoding_bits: 1,
    };
    config.agent_horizon = 8;
    config.observation_bits = 1;
    config.observation_stream_len = 1;
    config.observation_key_mode = infotheory::aixi::common::ObservationKeyMode::FullStream;
    config.reward_bits = 1;
    config.agent_actions =
        ActionAlphabet::try_from_usize(2).expect("test fixture action alphabet must be valid");
    config.num_simulations = 200;
    config.exploration_exploitation_ratio = 2.0;
    config.discount_gamma = 1.0;
    config.min_reward = 0;
    config.max_reward = 1;
    config.reward_offset = 0;
    config.random_seed = Some(17);

    let mut agent = Agent::new(config);
    let env = DeterministicBinaryEnv::new();

    let cycles = 100;
    let total_reward = run_agent_env(&mut agent, env, cycles);

    println!(
        "Agent Total Reward on DeterministicBinaryEnv (100 cycles): {}",
        total_reward
    );

    // Agent should learn pattern and get reasonable reward
    assert!(
        total_reward > 50.0,
        "Agent failed to learn DeterministicBinaryEnv pattern. Reward: {total_reward}"
    );
}

#[test]
fn agent_regret_sublinear_coinflip() {
    let mut config = AgentConfig::default();
    config.rate_backend = RateBackend::FacCtw {
        base_depth: 4,
        num_percept_bits: 2,
        encoding_bits: 1,
    };
    config.agent_horizon = 4;
    config.observation_bits = 1;
    config.observation_stream_len = 1;
    config.observation_key_mode = infotheory::aixi::common::ObservationKeyMode::FullStream;
    config.reward_bits = 1;
    config.agent_actions =
        ActionAlphabet::try_from_usize(2).expect("test fixture action alphabet must be valid");
    config.num_simulations = 100;
    config.exploration_exploitation_ratio = 1.0;
    config.discount_gamma = 1.0;
    config.min_reward = 0;
    config.max_reward = 1;
    config.reward_offset = 0;
    config.random_seed = Some(23);

    let mut agent = Agent::new(config);
    let env = SeededCoinFlipEnv::new(0.8);

    let cycles = 500;
    let total_reward = run_agent_env(&mut agent, env, cycles);

    let expected_optimal = 0.8 * cycles as f64;
    let regret = expected_optimal - total_reward;
    let regret_per_step = regret / cycles as f64;

    println!(
        "SeededCoinFlipEnv(0.8): Reward={total_reward}, Opt={expected_optimal}, Regret/step={regret_per_step:.4}"
    );

    // Regret should be reasonable (< 0.25 per step)
    assert!(regret_per_step < 0.25, "Regret too high: {regret_per_step}");
}

#[test]
fn agent_seeded_policy_is_reproducible_on_deterministic_env() {
    let mut config = AgentConfig::default();
    config.rate_backend = RateBackend::FacCtw {
        base_depth: 8,
        num_percept_bits: 2,
        encoding_bits: 1,
    };
    config.agent_horizon = 6;
    config.observation_bits = 1;
    config.observation_stream_len = 1;
    config.observation_key_mode = infotheory::aixi::common::ObservationKeyMode::FullStream;
    config.reward_bits = 1;
    config.agent_actions =
        ActionAlphabet::try_from_usize(2).expect("test fixture action alphabet must be valid");
    config.num_simulations = 80;
    config.exploration_exploitation_ratio = 1.4;
    config.discount_gamma = 1.0;
    config.min_reward = 0;
    config.max_reward = 1;
    config.reward_offset = 0;
    config.random_seed = Some(12345);

    let mut a = Agent::new(config.clone());
    let mut b = Agent::new(config);
    let mut env_a = DeterministicBinaryEnv::new();
    let mut env_b = DeterministicBinaryEnv::new();

    let mut obs_a = env_a.drain_observations();
    let mut obs_b = env_b.drain_observations();
    let mut rew_a = env_a.get_reward();
    let mut rew_b = env_b.get_reward();
    let mut prev_a = 0u64;
    let mut prev_b = 0u64;

    for step in 0..64usize {
        assert_eq!(obs_a, obs_b, "observation mismatch at step {step}");
        assert_eq!(rew_a, rew_b, "reward mismatch at step {step}");

        a.model_update_percept_stream(&obs_a, rew_a);
        b.model_update_percept_stream(&obs_b, rew_b);

        let act_a = a.get_planned_action(&obs_a, rew_a, prev_a);
        let act_b = b.get_planned_action(&obs_b, rew_b, prev_b);
        assert_eq!(act_a, act_b, "action mismatch at step {step}");

        a.model_update_action_external(act_a);
        b.model_update_action_external(act_b);

        env_a.perform_action(act_a);
        env_b.perform_action(act_b);
        obs_a = env_a.drain_observations();
        obs_b = env_b.drain_observations();
        rew_a = env_a.get_reward();
        rew_b = env_b.get_reward();
        prev_a = act_a;
        prev_b = act_b;
    }
}

#[test]
fn agent_omitted_seed_matches_explicit_default_seed() {
    let mut cfg_omitted = generic_agent_config(RateBackend::Ctw { depth: 8 });
    cfg_omitted.random_seed = None;
    let mut cfg_explicit = cfg_omitted.clone();
    cfg_explicit.random_seed = Some(DEFAULT_RANDOM_SEED);

    let mut a = Agent::try_new(cfg_omitted).expect("agent with omitted seed");
    let mut b = Agent::try_new(cfg_explicit).expect("agent with explicit default seed");

    assert_eq!(a.resolved_random_seed(), DEFAULT_RANDOM_SEED);
    assert_eq!(b.resolved_random_seed(), DEFAULT_RANDOM_SEED);

    let mut env_a = DeterministicBinaryEnv::new();
    let mut env_b = DeterministicBinaryEnv::new();

    let mut obs_a = env_a.drain_observations();
    let mut obs_b = env_b.drain_observations();
    let mut rew_a = env_a.get_reward();
    let mut rew_b = env_b.get_reward();
    let mut prev_a = 0u64;
    let mut prev_b = 0u64;

    for step in 0..64usize {
        a.model_update_percept_stream(&obs_a, rew_a);
        b.model_update_percept_stream(&obs_b, rew_b);

        let act_a = a.get_planned_action(&obs_a, rew_a, prev_a);
        let act_b = b.get_planned_action(&obs_b, rew_b, prev_b);
        assert_eq!(act_a, act_b, "action mismatch at step {step}");

        a.model_update_action_external(act_a);
        b.model_update_action_external(act_b);

        env_a.perform_action(act_a);
        env_b.perform_action(act_b);
        obs_a = env_a.drain_observations();
        obs_b = env_b.drain_observations();
        rew_a = env_a.get_reward();
        rew_b = env_b.get_reward();
        prev_a = act_a;
        prev_b = act_b;
    }
}

#[test]
fn agent_different_seeds_can_change_stochastic_trace() {
    let mut seed_pair = None;
    for left in 1u64..256 {
        let left_draw = infotheory::aixi::common::RandomGenerator::from_seed(left).gen_range(2);
        for right in (left + 1)..256 {
            let right_draw =
                infotheory::aixi::common::RandomGenerator::from_seed(right).gen_range(2);
            if left_draw != right_draw {
                seed_pair = Some((left, right, left_draw, right_draw));
                break;
            }
        }
        if seed_pair.is_some() {
            break;
        }
    }
    let (left_seed, right_seed, left_draw, right_draw) =
        seed_pair.expect("expected to find a seed pair with different first draws");
    assert_ne!(left_draw, right_draw);

    let mut left_rng = infotheory::aixi::common::RandomGenerator::from_seed(left_seed);
    let mut right_rng = infotheory::aixi::common::RandomGenerator::from_seed(right_seed);
    assert_ne!(left_rng.gen_range(2), right_rng.gen_range(2));
}

#[test]
fn agent_config_accepts_explicit_programmatic_rate_backend() {
    let cfg = generic_agent_config(RateBackend::Ppmd {
        order: 4,
        memory_mb: 8,
    });
    assert!(cfg.validate().is_ok());
    let mut agent = Agent::try_new(cfg).expect("explicit rate_backend should be valid");
    let action = agent.get_planned_action(&[0], 0, 0);
    assert!(action < 2);
}

#[test]
fn agent_config_rejects_zpaq_rate_backend_in_strict_mode() {
    let cfg = generic_agent_config(RateBackend::Mixture {
        spec: Arc::new(MixtureSpec::new(
            MixtureKind::Bayes,
            vec![{
                let mut expert = MixtureExpertSpec::new(RateBackend::Zpaq {
                    method: infotheory::api::ZpaqMethodSpec::literal("1"),
                });
                expert.name = Some("bad-zpaq".to_string());
                expert.log_prior = 0.0;
                expert
            }],
        )),
    });
    let err = cfg
        .validate()
        .expect_err("zpaq-backed generic MC-AIXI should be rejected");
    assert!(matches!(err, AgentError::UnsupportedRateBackend { .. }));
}

#[test]
fn agent_config_rejects_invalid_programmatic_mixture_rate_backend() {
    let cfg = generic_agent_config(RateBackend::Mixture {
        spec: Arc::new(MixtureSpec::new(MixtureKind::Bayes, vec![])),
    });
    let err = cfg
        .validate()
        .expect_err("empty mixture backend should be rejected");
    assert!(matches!(err, AgentError::InvalidRateBackend(_)));
}

#[test]
fn agent_config_rejects_programmatic_mixture_nesting_overflow() {
    let cfg = generic_agent_config(deeply_nested_bayes_backend(MAX_MIXTURE_NESTING + 1));
    let err = cfg
        .validate()
        .expect_err("overly deep nested mixture should be rejected");
    assert!(matches!(err, AgentError::InvalidRateBackend(_)));
}

#[test]
fn agent_with_generic_mixture_backends_smoke_runs() {
    for (kind, label) in [
        (MixtureKind::Bayes, "bayes"),
        (MixtureKind::Switching, "switching"),
        (MixtureKind::Convex, "convex"),
    ] {
        let mut agent =
            Agent::try_new(generic_agent_config(mixture_backend(kind))).expect("valid mixture");
        let total_reward = run_agent_env(&mut agent, DeterministicBinaryEnv::new(), 48);
        assert!(
            total_reward > 16.0,
            "{label} mixture backend reward too low on DeterministicBinaryEnv: {total_reward}"
        );
    }
}
