import infotheory_rs as ait
import pytest


def test_aixi_env_smoke():
    env = ait.CoinFlipEnv(0.5)
    env.perform_action(0)
    assert env.get_observation() in (0, 1)
    assert isinstance(env.get_reward(), int)


def test_agent_config_and_agent_smoke():
    cfg = ait.AgentConfig(
        algorithm="fac-ctw",
        ct_depth=8,
        agent_horizon=2,
        observation_bits=1,
        observation_stream_len=1,
        reward_bits=1,
        agent_actions=2,
        num_simulations=8,
        min_reward=0,
        max_reward=1,
        reward_offset=0,
    )
    agent = ait.Agent(cfg)
    action = agent.get_planned_action([0], 0, 0)
    assert action in (0, 1)


def test_aiqi_config_and_agent_smoke():
    cfg = ait.AiqiConfig(
        algorithm="ac-ctw",
        ct_depth=8,
        observation_bits=1,
        observation_stream_len=1,
        reward_bits=1,
        agent_actions=2,
        min_reward=0,
        max_reward=1,
        reward_offset=0,
        discount_gamma=0.99,
        return_horizon=2,
        return_bins=8,
        augmentation_period=2,
        baseline_exploration=0.01,
    )
    agent = ait.AiqiAgent(cfg)
    action = agent.get_planned_action()
    assert action in (0, 1)
    agent.observe_transition(action, [0], 1)
    assert agent.steps_observed() == 1


def test_run_aiqi_with_environment_smoke():
    env = ait.CoinFlipEnv(0.7)
    cfg = ait.AiqiConfig(
        algorithm="ac-ctw",
        ct_depth=6,
        observation_bits=1,
        observation_stream_len=1,
        reward_bits=1,
        agent_actions=2,
        min_reward=0,
        max_reward=1,
        reward_offset=0,
        discount_gamma=0.99,
        return_horizon=2,
        return_bins=8,
        augmentation_period=2,
        baseline_exploration=0.01,
    )
    summary = ait.run_aiqi_with_environment(
        env,
        cfg,
        learn_cycles=6,
        eval_cycles=4,
        terminate_lifetime=6,
        explore_epsilon=0.0,
        explore_gamma=1.0,
        check_finished=False,
    )
    assert summary["learn_cycles_completed"] == 6
    assert summary["eval_cycles_completed"] == 4
    assert isinstance(summary["eval_average_reward"], float)


def test_run_aiqi_with_generic_rate_backend_smoke():
    env = ait.CoinFlipEnv(0.7)
    cfg = ait.AiqiConfig(
        algorithm="ac-ctw",
        ct_depth=6,
        observation_bits=1,
        observation_stream_len=1,
        reward_bits=1,
        agent_actions=2,
        min_reward=0,
        max_reward=1,
        reward_offset=0,
        discount_gamma=0.99,
        return_horizon=2,
        return_bins=8,
        augmentation_period=2,
        baseline_exploration=0.01,
        rate_backend=ait.RateBackend.ppmd(order=4, memory_mb=8),
        rate_backend_max_order=8,
    )
    summary = ait.run_aiqi_with_environment(
        env,
        cfg,
        learn_cycles=4,
        eval_cycles=2,
        terminate_lifetime=4,
        explore_epsilon=0.0,
        explore_gamma=1.0,
        check_finished=False,
    )
    assert summary["learn_cycles_completed"] == 4
    assert summary["eval_cycles_completed"] == 2


def test_run_mcaixi_with_generic_mixture_rate_backend_smoke():
    env = ait.CoinFlipEnv(0.7)
    mixture = ait.RateBackend.mixture(
        ait.MixtureSpec(
            ait.MixtureKind.Convex,
            [
                ait.MixtureExpertSpec(
                    ait.RateBackend.ctw(6), max_order=-1, log_prior=0.0, name="ctw"
                ),
                ait.MixtureExpertSpec(
                    ait.RateBackend.ppmd(order=4, memory_mb=8),
                    max_order=-1,
                    log_prior=0.0,
                    name="ppmd",
                ),
            ],
            alpha=1.25,
        )
    )
    cfg = ait.AgentConfig(
        algorithm="zpaq",
        ct_depth=6,
        agent_horizon=2,
        observation_bits=1,
        observation_stream_len=1,
        reward_bits=1,
        agent_actions=2,
        num_simulations=16,
        exploration_exploitation_ratio=1.2,
        discount_gamma=1.0,
        min_reward=0,
        max_reward=1,
        reward_offset=0,
        random_seed=77,
        rate_backend=mixture,
        rate_backend_max_order=8,
        zpaq_method="1",
    )
    summary = ait.run_agent_with_environment(
        env,
        cfg,
        learn_cycles=4,
        eval_cycles=2,
        terminate_lifetime=4,
        explore_epsilon=0.0,
        explore_gamma=1.0,
        check_finished=False,
    )
    assert summary["learn_cycles_completed"] == 4
    assert summary["eval_cycles_completed"] == 2
    assert summary["last_action"] in (0, 1)


def test_aiqi_rejects_zpaq_algorithm_in_strict_mode():
    with pytest.raises(ValueError, match="strict mode"):
        ait.AiqiConfig(
            algorithm="zpaq",
            ct_depth=6,
            observation_bits=1,
            observation_stream_len=1,
            reward_bits=1,
            agent_actions=2,
            min_reward=0,
            max_reward=1,
            reward_offset=0,
            discount_gamma=0.99,
            return_horizon=2,
            return_bins=8,
            augmentation_period=2,
            baseline_exploration=0.01,
        )


def test_aiqi_rejects_non_power_of_two_return_bins():
    with pytest.raises(ValueError, match="power of two"):
        ait.AiqiConfig(
            algorithm="ac-ctw",
            ct_depth=6,
            observation_bits=1,
            observation_stream_len=1,
            reward_bits=1,
            agent_actions=2,
            min_reward=0,
            max_reward=1,
            reward_offset=0,
            discount_gamma=0.99,
            return_horizon=2,
            return_bins=3,
            augmentation_period=2,
            baseline_exploration=0.01,
        )


def test_aiqi_rejects_zpaq_rate_backend_in_strict_mode():
    with pytest.raises(ValueError, match="strict frozen conditioning"):
        ait.AiqiConfig(
            algorithm="ac-ctw",
            ct_depth=6,
            observation_bits=1,
            observation_stream_len=1,
            reward_bits=1,
            agent_actions=2,
            min_reward=0,
            max_reward=1,
            reward_offset=0,
            discount_gamma=0.99,
            return_horizon=2,
            return_bins=8,
            augmentation_period=2,
            baseline_exploration=0.01,
            rate_backend=ait.RateBackend.zpaq("1"),
            rate_backend_max_order=8,
        )


def test_mcaixi_rejects_zpaq_rate_backend_in_strict_mode():
    with pytest.raises(ValueError, match="A Monte-Carlo AIXI Approximation"):
        ait.AgentConfig(
            algorithm="fac-ctw",
            ct_depth=6,
            agent_horizon=2,
            observation_bits=1,
            observation_stream_len=1,
            reward_bits=1,
            agent_actions=2,
            num_simulations=8,
            min_reward=0,
            max_reward=1,
            reward_offset=0,
            rate_backend=ait.RateBackend.zpaq("1"),
            rate_backend_max_order=8,
        )


def test_aiqi_optional_history_pruning_smoke():
    cfg = ait.AiqiConfig(
        algorithm="ac-ctw",
        ct_depth=6,
        observation_bits=1,
        observation_stream_len=1,
        reward_bits=1,
        agent_actions=2,
        min_reward=0,
        max_reward=1,
        reward_offset=0,
        discount_gamma=0.99,
        return_horizon=3,
        return_bins=8,
        augmentation_period=4,
        history_prune_keep_steps=16,
        baseline_exploration=0.01,
    )
    agent = ait.AiqiAgent(cfg)
    env = ait.CoinFlipEnv(0.7)
    for _ in range(64):
        action = agent.get_planned_action()
        env.perform_action(action)
        agent.observe_transition(action, [env.get_observation()], env.get_reward())
    assert agent.steps_observed() == 64


def test_mcaixi_seed_reproducibility_with_deterministic_env():
    cfg = ait.AgentConfig(
        algorithm="ctw",
        ct_depth=8,
        agent_horizon=4,
        observation_bits=1,
        observation_stream_len=1,
        reward_bits=1,
        agent_actions=2,
        num_simulations=60,
        exploration_exploitation_ratio=1.4,
        discount_gamma=1.0,
        min_reward=0,
        max_reward=1,
        reward_offset=0,
        random_seed=424242,
    )

    s1 = ait.run_agent_with_environment(
        ait.CtwTestEnv(),
        cfg,
        learn_cycles=40,
        eval_cycles=20,
        terminate_lifetime=40,
        explore_epsilon=0.15,
        explore_gamma=0.99,
        check_finished=False,
    )
    s2 = ait.run_agent_with_environment(
        ait.CtwTestEnv(),
        cfg,
        learn_cycles=40,
        eval_cycles=20,
        terminate_lifetime=40,
        explore_epsilon=0.15,
        explore_gamma=0.99,
        check_finished=False,
    )

    assert s1["learn_total_reward"] == s2["learn_total_reward"]
    assert s1["eval_total_reward"] == s2["eval_total_reward"]
    assert s1["last_action"] == s2["last_action"]
    assert s1["last_observation_stream"] == s2["last_observation_stream"]


def test_aiqi_seed_reproducibility_with_deterministic_env():
    cfg = ait.AiqiConfig(
        algorithm="ac-ctw",
        ct_depth=8,
        observation_bits=1,
        observation_stream_len=1,
        reward_bits=1,
        agent_actions=2,
        min_reward=0,
        max_reward=1,
        reward_offset=0,
        discount_gamma=0.99,
        return_horizon=4,
        return_bins=16,
        augmentation_period=4,
        baseline_exploration=0.2,
        random_seed=131313,
    )

    s1 = ait.run_aiqi_with_environment(
        ait.CtwTestEnv(),
        cfg,
        learn_cycles=40,
        eval_cycles=20,
        terminate_lifetime=40,
        explore_epsilon=0.0,
        explore_gamma=1.0,
        check_finished=False,
    )
    s2 = ait.run_aiqi_with_environment(
        ait.CtwTestEnv(),
        cfg,
        learn_cycles=40,
        eval_cycles=20,
        terminate_lifetime=40,
        explore_epsilon=0.0,
        explore_gamma=1.0,
        check_finished=False,
    )

    assert s1["learn_total_reward"] == s2["learn_total_reward"]
    assert s1["eval_total_reward"] == s2["eval_total_reward"]
    assert s1["last_action"] == s2["last_action"]
    assert s1["last_observation_stream"] == s2["last_observation_stream"]
