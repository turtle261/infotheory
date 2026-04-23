import pytest

import infotheory_rs as ait


pytestmark = pytest.mark.skipif(
    not hasattr(ait, "CoinFlipEnv"),
    reason="requires the aixi-gameengine feature",
)


def test_coinflip_env_preserves_bias_and_seed_api():
    heads = ait.CoinFlipEnv(1.0, random_seed=7)
    heads.perform_action(1)
    assert heads.get_observation() == 1
    assert heads.get_reward() == 1

    heads.set_random_seed(11)
    heads.perform_action(0)
    assert heads.get_observation() == 1
    assert heads.get_reward() == 0

    tails = ait.CoinFlipEnv(0.0, random_seed=7)
    tails.perform_action(0)
    assert tails.get_observation() == 0
    assert tails.get_reward() == 1


@pytest.mark.parametrize("p", [-0.1, 1.1, float("nan")])
def test_coinflip_env_rejects_invalid_probability(p):
    with pytest.raises(ValueError, match="finite probability"):
        ait.CoinFlipEnv(p)
