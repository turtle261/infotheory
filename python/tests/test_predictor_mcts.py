import infotheory_rs as ait


def test_predictor_wrappers_smoke():
    p = ait.CtwPredictor(8)
    q = p.predict_one()
    assert 0.0 <= q <= 1.0
    p.update(True)
    p.revert()
    assert isinstance(p.model_name(), str)


def test_search_tree_smoke():
    cfg = ait.AgentConfig(
        rate_backend=ait.RateBackend.ctw(8),
        agent_horizon=2,
        observation_bits=1,
        observation_stream_len=1,
        reward_bits=1,
        agent_actions=2,
        num_simulations=4,
        min_reward=0,
        max_reward=1,
        reward_offset=0,
    )
    agent = ait.Agent(cfg)
    tree = ait.SearchTree()
    a = tree.search(agent, [0], 0, 0, 4)
    assert a in (0, 1)
