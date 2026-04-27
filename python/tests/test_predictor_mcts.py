import pytest

import infotheory_rs as ait


def _agent_config(mcts_strategy=None):
    kwargs = {
        "rate_backend": ait.RateBackend.ctw(8),
        "agent_horizon": 2,
        "observation_bits": 1,
        "observation_stream_len": 1,
        "reward_bits": 1,
        "agent_actions": 2,
        "num_simulations": 4,
        "min_reward": 0,
        "max_reward": 1,
        "reward_offset": 0,
    }
    if mcts_strategy is not None:
        kwargs["mcts_strategy"] = mcts_strategy
    return ait.AgentConfig(**kwargs)


def test_predictor_wrappers_smoke():
    p = ait.CtwPredictor(8)
    q = p.predict_one()
    assert 0.0 <= q <= 1.0
    p.update(True)
    p.revert()
    assert isinstance(p.model_name(), str)


def test_mcts_strategy_python_surface():
    rho_uct = ait.MctsStrategy.rho_uct()
    assert rho_uct.kind == "rho_uct"
    assert rho_uct.workers is None
    assert rho_uct.bu_uct_m_max is None

    parallel_wu = ait.MctsStrategy.parallel_uct(4)
    assert parallel_wu.kind == "parallel_uct"
    assert parallel_wu.workers == 4
    assert parallel_wu.bu_uct_m_max is None

    parallel_bu = ait.MctsStrategy.parallel_uct(4, 0.8)
    assert parallel_bu.kind == "parallel_uct"
    assert parallel_bu.workers == 4
    assert parallel_bu.bu_uct_m_max == pytest.approx(0.8)


def test_mcts_strategy_parallel_uct_rejects_invalid_parameters():
    with pytest.raises(ValueError, match="workers"):
        ait.MctsStrategy.parallel_uct(0)
    with pytest.raises(ValueError, match="bu_uct_m_max"):
        ait.MctsStrategy.parallel_uct(2, 1.0)


@pytest.mark.parametrize(
    "mcts_strategy",
    [
        None,
        ait.MctsStrategy.rho_uct(),
        ait.MctsStrategy.parallel_uct(2),
        ait.MctsStrategy.parallel_uct(2, 0.8),
    ],
)
def test_search_tree_smoke(mcts_strategy):
    agent = ait.Agent(_agent_config(mcts_strategy))
    tree = ait.SearchTree()
    if mcts_strategy is not None:
        tree = ait.SearchTree(mcts_strategy)
        assert tree.mcts_strategy.kind == mcts_strategy.kind
    else:
        assert tree.mcts_strategy.kind == "rho_uct"
    a = tree.search(agent, [0], 0, 0, 4)
    assert a in (0, 1)


def test_search_node_is_not_exported():
    assert not hasattr(ait, "SearchNode")
    with pytest.raises(AttributeError, match="SearchNode"):
        getattr(ait, "SearchNode")
