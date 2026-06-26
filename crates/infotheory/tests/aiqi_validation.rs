#![cfg(all(feature = "aixi", feature = "all-backends"))]

//! AIQI validation tests.

use infotheory::aixi::aiqi::{AiqiAgent, AiqiConfig, AiqiError};
use infotheory::aixi::common::{ActionAlphabet, DEFAULT_RANDOM_SEED};
use infotheory::aixi::environment::Environment;
mod support;
use infotheory::aixi::model::{
    RateBackendBitPredictor, RateBackendBitPredictorConfig, RateBackendBitPredictorError,
};
use infotheory::api::{BitOrder, BitStreamSemantics, MixtureKind, MixtureSpec, RateBackend};
use std::sync::Arc;
use support::aixi_envs::{DeterministicBinaryEnv, SeededCoinFlipEnv};

fn base_config() -> AiqiConfig {
    let mut cfg = AiqiConfig::default();
    cfg.rate_backend = RateBackend::Ctw { depth: 8 };
    cfg.bit_stream_semantics = infotheory::api::BitStreamSemantics::BinaryTokens;
    cfg.observation_bits = 1;
    cfg.observation_stream_len = 1;
    cfg.reward_bits = 1;
    cfg.agent_actions =
        ActionAlphabet::try_from_usize(2).expect("test fixture action alphabet must be valid");
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
    cfg
}

#[test]
fn default_aiqi_config_validates() {
    AiqiConfig::default()
        .validate()
        .expect("default AIQI config should satisfy its own contract");
}

fn aiqi_mixture_backend(kind: MixtureKind) -> RateBackend {
    let experts = vec![
        {
            let mut expert = infotheory::api::MixtureExpertSpec::new(RateBackend::Ctw { depth: 8 });
            expert.name = Some("ctw".to_string());
            expert.log_prior = 0.0;
            expert
        },
        {
            let mut expert = infotheory::api::MixtureExpertSpec::new(RateBackend::FacCtw {
                base_depth: 8,
                num_percept_bits: 8,
                encoding_bits: 1,
                msb_first: None,
            });
            expert.name = Some("fac-ctw".to_string());
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

fn run_aiqi_env<T: Environment>(agent: &mut AiqiAgent, mut env: T, cycles: usize) -> i64 {
    let mut total_reward = 0i64;
    for _ in 0..cycles {
        let action = agent.get_planned_action();
        env.perform_action(action);
        let obs_stream = env.drain_observations();
        let rew = env.get_reward();
        agent
            .observe_transition(action, &obs_stream, rew)
            .expect("transition must be accepted");
        total_reward += rew;
    }
    total_reward
}

#[test]
fn aiqi_config_rejects_period_shorter_than_horizon() {
    let mut cfg = base_config();
    cfg.return_horizon = 3;
    cfg.augmentation_period = 2;
    let err = cfg.validate().expect_err("N < H must be rejected");
    assert!(matches!(
        err,
        AiqiError::AugmentationPeriodTooShort {
            augmentation_period: 2,
            return_horizon: 3
        }
    ));
}

#[test]
fn aiqi_config_accepts_non_power_of_two_return_bins() {
    let mut cfg = base_config();
    cfg.return_bins = 3;
    cfg.validate()
        .expect("non-power-of-two return_bins are valid AIQI discretization levels");
}

#[test]
fn aiqi_config_rejects_zpaq_rate_backend_in_strict_mode() {
    let mut cfg = base_config();
    cfg.rate_backend = RateBackend::Zpaq {
        method: infotheory::api::ZpaqMethodSpec::literal("1"),
    };
    let err = cfg
        .validate()
        .expect_err("strict AIQI should reject zpaq rate backend");
    assert!(matches!(err, AiqiError::UnsupportedRateBackend { .. }));
}

#[test]
fn aiqi_config_rejects_invalid_programmatic_mixture_rate_backend() {
    let mut cfg = base_config();
    cfg.rate_backend = RateBackend::Mixture {
        spec: Arc::new(MixtureSpec::new(MixtureKind::Bayes, vec![])),
    };
    let err = cfg
        .validate()
        .expect_err("empty mixture backend should be rejected");
    assert!(matches!(err, AiqiError::InvalidRateBackend(_)));
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
    cfg.rate_backend = RateBackend::Ctw { depth: 10 };
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
    cfg.rate_backend = RateBackend::Match {
        hash_bits: 16,
        min_len: 2,
        max_len: 16,
        base_mix: 0.05,
        confidence_scale: 1.0,
    };

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
    cfg.rate_backend = RateBackend::RosaPlus { max_order: 20 };

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
fn aiqi_bytepacked_ctw_planner_handles_shared_percept_byte() {
    let mut cfg = base_config();
    cfg.bit_stream_semantics = BitStreamSemantics::BytePacked {
        order: BitOrder::MsbFirst,
    };
    cfg.rate_backend = RateBackend::Ctw { depth: 8 };
    cfg.agent_actions =
        ActionAlphabet::try_from_usize(129).expect("129 actions require one byte of action bits");
    cfg.observation_bits = 3;
    cfg.observation_stream_len = 1;
    cfg.reward_bits = 5;
    cfg.min_reward = 0;
    cfg.max_reward = 31;
    cfg.reward_offset = 0;
    cfg.return_bins = 256;
    cfg.return_horizon = 2;
    cfg.augmentation_period = 2;
    cfg.baseline_exploration = 1e-12;
    cfg.random_seed = Some(0x0A10_1B17);

    let mut left = AiqiAgent::new(cfg.clone()).expect("valid byte-packed AIQI CTW config");
    let mut right = AiqiAgent::new(cfg).expect("valid replay byte-packed AIQI CTW config");

    let mut planned_actions = Vec::new();
    for step in 0..8usize {
        let planned_left = left.get_planned_action();
        let planned_right = right.get_planned_action();
        assert_eq!(
            planned_left, planned_right,
            "byte-packed AIQI planning must be deterministic at step {step}"
        );
        planned_actions.push(planned_left);

        let action = (planned_left ^ ((step as u64).wrapping_mul(37))) % 129;
        let observation = [((0b101usize ^ (step * 3) ^ action as usize) & 0b111) as u64];
        let reward = ((0b10001usize ^ (step * 5) ^ action as usize) & 0b1_1111) as i64;

        left.observe_transition(action, &observation, reward)
            .expect("left byte-packed transition should be accepted");
        right
            .observe_transition(action, &observation, reward)
            .expect("right byte-packed transition should be accepted");
    }

    assert_eq!(left.steps_observed(), 8);
    assert_eq!(right.steps_observed(), 8);
    assert!(
        planned_actions.iter().all(|&action| action < 129),
        "all byte-packed AIQI actions must stay inside the configured alphabet: {planned_actions:?}"
    );
}

/// Exercises `BitStreamSemantics::BytePacked` combined with `RateBackend::FacCtw`
/// inside an AIQI planner loop — covering the FacCtw + BytePacked planner path
/// that is distinct from both the native-CTW BytePacked path and the
/// BinaryTokens FacCtw path already exercised elsewhere.
///
/// Two independently-constructed agents with identical configuration and seed
/// must produce identical action sequences across multiple plan/observe cycles,
/// and every planned action must lie within the configured alphabet.
#[test]
fn aiqi_bytepacked_fac_ctw_planner_handles_shared_percept_byte() {
    let mut cfg = base_config();
    cfg.bit_stream_semantics = BitStreamSemantics::BytePacked {
        order: BitOrder::MsbFirst,
    };
    cfg.rate_backend = RateBackend::FacCtw {
        base_depth: 8,
        // 3-bit observation + 5-bit reward share one percept byte.
        num_percept_bits: 8,
        encoding_bits: 1,
        msb_first: Some(true),
    };
    cfg.agent_actions =
        ActionAlphabet::try_from_usize(129).expect("129 actions require one byte of action bits");
    cfg.observation_bits = 3;
    cfg.observation_stream_len = 1;
    cfg.reward_bits = 5;
    cfg.min_reward = 0;
    cfg.max_reward = 31;
    cfg.reward_offset = 0;
    cfg.return_bins = 256;
    cfg.return_horizon = 2;
    cfg.augmentation_period = 2;
    cfg.baseline_exploration = 1e-12;
    cfg.random_seed = Some(0xA17F_AC17);

    let mut left = AiqiAgent::new(cfg.clone()).expect("valid byte-packed AIQI FAC-CTW config");
    let mut right = AiqiAgent::new(cfg).expect("valid replay byte-packed AIQI FAC-CTW config");

    let mut planned_actions = Vec::new();
    for step in 0..8usize {
        let planned_left = left.get_planned_action();
        let planned_right = right.get_planned_action();
        assert_eq!(
            planned_left, planned_right,
            "byte-packed AIQI FAC-CTW planning must be deterministic at step {step}"
        );
        planned_actions.push(planned_left);

        let action = (planned_left ^ ((step as u64).wrapping_mul(19))) % 129;
        let observation = [((0b110usize ^ (step * 7) ^ action as usize) & 0b111) as u64];
        let reward = ((0b01101usize ^ (step * 9) ^ action as usize) & 0b1_1111) as i64;

        left.observe_transition(action, &observation, reward)
            .expect("left byte-packed FAC-CTW transition should be accepted");
        right
            .observe_transition(action, &observation, reward)
            .expect("right byte-packed FAC-CTW transition should be accepted");
    }

    assert_eq!(left.steps_observed(), 8);
    assert_eq!(right.steps_observed(), 8);
    assert!(
        planned_actions.iter().all(|&action| action < 129),
        "all byte-packed AIQI FAC-CTW actions must stay inside configured alphabet: {planned_actions:?}"
    );
}

/// `BitStreamSemantics::BinaryTokens` with 8-bit MSB FacCtw in a full AIQI planner loop.
/// Native path is [`FacCtwPredictor`] (per `percept_bits` lanes), not byte-prefix MSB hooks.
#[test]
fn aiqi_binarytokens_fac_ctw_native_planner_integration() {
    let compiled = RateBackend::FacCtw {
        base_depth: 8,
        num_percept_bits: 8,
        encoding_bits: 8,
        msb_first: Some(true),
    }
    .compile()
    .expect("compile fac-ctw backend");
    let caps = compiled.capabilities();
    assert!(caps.supports_native_bit_prediction);
    assert!(caps.supports_reversible_bit_updates);

    let mut cfg = base_config();
    cfg.bit_stream_semantics = BitStreamSemantics::BinaryTokens;
    cfg.rate_backend = RateBackend::FacCtw {
        base_depth: 8,
        num_percept_bits: 8,
        encoding_bits: 8,
        msb_first: Some(true),
    };
    cfg.agent_actions =
        ActionAlphabet::try_from_usize(129).expect("129 actions require one byte of action bits");
    cfg.observation_bits = 3;
    cfg.observation_stream_len = 1;
    cfg.reward_bits = 5;
    cfg.min_reward = 0;
    cfg.max_reward = 31;
    cfg.reward_offset = 0;
    cfg.return_bins = 256;
    cfg.return_horizon = 2;
    cfg.augmentation_period = 2;
    cfg.baseline_exploration = 1e-12;
    cfg.random_seed = Some(0xB17A_1701);

    let mut left = AiqiAgent::new(cfg.clone()).expect("valid BinaryTokens AIQI FAC-CTW config");
    let mut right = AiqiAgent::new(cfg).expect("valid replay BinaryTokens AIQI FAC-CTW config");

    let mut planned_actions = Vec::new();
    for step in 0..8usize {
        let planned_left = left.get_planned_action();
        let planned_right = right.get_planned_action();
        assert_eq!(
            planned_left, planned_right,
            "BinaryTokens AIQI FAC-CTW planning must be deterministic at step {step}"
        );
        planned_actions.push(planned_left);

        let action = (planned_left ^ ((step as u64).wrapping_mul(23))) % 129;
        let observation = [((0b101usize ^ (step * 3) ^ action as usize) & 0b111) as u64];
        let reward = ((0b10001usize ^ (step * 5) ^ action as usize) & 0b1_1111) as i64;

        left.observe_transition(action, &observation, reward)
            .expect("left BinaryTokens FAC-CTW transition should be accepted");
        right
            .observe_transition(action, &observation, reward)
            .expect("right BinaryTokens FAC-CTW transition should be accepted");
    }

    assert_eq!(left.steps_observed(), 8);
    assert_eq!(right.steps_observed(), 8);
    assert!(
        planned_actions.iter().all(|&action| action < 129),
        "all BinaryTokens AIQI FAC-CTW actions must stay inside configured alphabet: {planned_actions:?}"
    );
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
    let config = RateBackendBitPredictorConfig::compile(
        RateBackend::Zpaq {
            method: infotheory::api::ZpaqMethodSpec::literal("1"),
        },
        1e-12,
    )
    .expect("zpaq compiles before bit-predictor capability check");
    let err = match RateBackendBitPredictor::new(config) {
        Ok(_) => panic!("zpaq must be rejected in RateBackendBitPredictor"),
        Err(err) => err,
    };
    assert!(matches!(err, RateBackendBitPredictorError::UnsupportedZpaq));
}

#[test]
fn aiqi_learns_ctw_pattern_with_fac_ctw_world_model() {
    let mut cfg = base_config();
    cfg.discount_gamma = 0.7;
    cfg.return_horizon = 4;
    cfg.augmentation_period = 4;
    cfg.baseline_exploration = 1e-6;
    cfg.rate_backend = RateBackend::FacCtw {
        base_depth: 10,
        num_percept_bits: 8,
        encoding_bits: 1,
        msb_first: None,
    };

    let mut agent = AiqiAgent::new(cfg).expect("valid AIQI FAC-CTW config");
    let total_reward = run_aiqi_env(&mut agent, DeterministicBinaryEnv::new(), 120);
    assert!(
        total_reward > 50,
        "AIQI FAC-CTW world model failed to learn deterministic pattern; total_reward={total_reward}"
    );
}

#[test]
fn aiqi_mixture_world_models_learn_deterministic_pattern() {
    for (kind, label) in [
        (MixtureKind::Bayes, "bayes"),
        (MixtureKind::Switching, "switching"),
        (MixtureKind::Convex, "convex"),
    ] {
        let mut cfg = base_config();
        cfg.discount_gamma = 0.8;
        cfg.return_horizon = 4;
        cfg.augmentation_period = 4;
        cfg.baseline_exploration = 0.01;
        cfg.rate_backend = aiqi_mixture_backend(kind);

        let mut agent = AiqiAgent::new(cfg).expect("valid AIQI mixture config");
        let total_reward = run_aiqi_env(&mut agent, DeterministicBinaryEnv::new(), 96);
        assert!(
            total_reward > 35,
            "{label} AIQI mixture world model reward too low on deterministic pattern: {total_reward}"
        );
    }
}

#[test]
fn aiqi_mixture_world_models_are_seed_deterministic() {
    for (kind, label) in [
        (MixtureKind::Bayes, "bayes"),
        (MixtureKind::Switching, "switching"),
        (MixtureKind::Convex, "convex"),
    ] {
        let mut cfg = base_config();
        cfg.rate_backend = aiqi_mixture_backend(kind);
        cfg.baseline_exploration = 0.3;
        cfg.random_seed = Some(20260429);

        let mut a = AiqiAgent::new(cfg.clone()).expect("mixture agent A");
        let mut b = AiqiAgent::new(cfg).expect("mixture agent B");

        for step in 0..96usize {
            let act_a = a.get_planned_action();
            let act_b = b.get_planned_action();
            assert_eq!(
                act_a, act_b,
                "{label} action mismatch at step {step} under equal seed/history"
            );

            let obs = [((step + 1) % 2) as u64];
            let rew = (step % 2) as i64;
            a.observe_transition(act_a, &obs, rew)
                .expect("transition should be accepted");
            b.observe_transition(act_b, &obs, rew)
                .expect("transition should be accepted");
        }
    }
}
