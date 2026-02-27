import infotheory_rs as ait


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
