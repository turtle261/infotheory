//! AIXI Module Validation Tests
//!
//! Tests for predictors, environments, and agents.

use infotheory::aixi::agent::{Agent, AgentConfig};
use infotheory::aixi::common::Action;
use infotheory::aixi::environment::{CoinFlip, CtwTest, Environment};
use infotheory::aixi::model::{CtwPredictor, Predictor, RosaPredictor};

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

// ============================================================================
// Environment Tests
// ============================================================================

#[test]
fn ctw_test_env_is_deterministic() {
    let mut env1 = CtwTest::new();
    let mut env2 = CtwTest::new();

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

#[test]
fn agent_solves_ctw_test_environment() {
    let config = AgentConfig {
        algorithm: "ctw".into(),
        ct_depth: 8,
        agent_horizon: 8, // Increased from 4
        observation_bits: 1,
        observation_stream_len: 1,
        observation_key_mode: infotheory::aixi::common::ObservationKeyMode::FullStream,
        reward_bits: 1,
        agent_actions: 2,
        num_simulations: 200, // Increased from 50
        exploration_exploitation_ratio: 2.0,
        discount_gamma: 1.0,
        min_reward: 0,
        max_reward: 1,
        reward_offset: 0,
        random_seed: Some(17),
        rwkv_model_path: None,
        rwkv_method: None,
        mamba_model_path: None,
        mamba_method: None,
        rosa_max_order: None,
        zpaq_method: None,
    };

    let mut agent = Agent::new(config);
    let env = CtwTest::new();

    let cycles = 100;
    let total_reward = run_agent_env(&mut agent, env, cycles);

    println!(
        "Agent Total Reward on CtwTest (100 cycles): {}",
        total_reward
    );

    // Agent should learn pattern and get reasonable reward
    assert!(
        total_reward > 50.0,
        "Agent failed to learn CtwTest pattern. Reward: {total_reward}"
    );
}

#[test]
fn agent_regret_sublinear_coinflip() {
    let config = AgentConfig {
        algorithm: "ctw".into(),
        ct_depth: 4,
        agent_horizon: 4, // Increased from 2
        observation_bits: 1,
        observation_stream_len: 1,
        observation_key_mode: infotheory::aixi::common::ObservationKeyMode::FullStream,
        reward_bits: 1,
        agent_actions: 2,
        num_simulations: 100, // Increased from 20
        exploration_exploitation_ratio: 1.0,
        discount_gamma: 1.0,
        min_reward: 0,
        max_reward: 1,
        reward_offset: 0,
        random_seed: Some(23),
        rwkv_model_path: None,
        rwkv_method: None,
        mamba_model_path: None,
        mamba_method: None,
        rosa_max_order: None,
        zpaq_method: None,
    };

    let mut agent = Agent::new(config);
    let env = CoinFlip::new(0.8);

    let cycles = 500;
    let total_reward = run_agent_env(&mut agent, env, cycles);

    let expected_optimal = 0.8 * cycles as f64;
    let regret = expected_optimal - total_reward;
    let regret_per_step = regret / cycles as f64;

    println!(
        "CoinFlip(0.8): Reward={total_reward}, Opt={expected_optimal}, Regret/step={regret_per_step:.4}"
    );

    // Regret should be reasonable (< 0.25 per step)
    assert!(regret_per_step < 0.25, "Regret too high: {regret_per_step}");
}

#[test]
fn agent_seeded_policy_is_reproducible_on_deterministic_env() {
    let config = AgentConfig {
        algorithm: "ctw".into(),
        ct_depth: 8,
        agent_horizon: 6,
        observation_bits: 1,
        observation_stream_len: 1,
        observation_key_mode: infotheory::aixi::common::ObservationKeyMode::FullStream,
        reward_bits: 1,
        agent_actions: 2,
        num_simulations: 80,
        exploration_exploitation_ratio: 1.4,
        discount_gamma: 1.0,
        min_reward: 0,
        max_reward: 1,
        reward_offset: 0,
        random_seed: Some(12345),
        rwkv_model_path: None,
        rwkv_method: None,
        mamba_model_path: None,
        mamba_method: None,
        rosa_max_order: None,
        zpaq_method: None,
    };

    let mut a = Agent::new(config.clone());
    let mut b = Agent::new(config);
    let mut env_a = CtwTest::new();
    let mut env_b = CtwTest::new();

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
