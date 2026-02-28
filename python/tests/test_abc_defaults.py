import infotheory_rs as ait
import random


class _Predictor(ait.PredictorABC):
    def __init__(self):
        self.hist = []

    def update(self, sym: bool):
        self.hist.append(bool(sym))

    def revert(self):
        if self.hist:
            self.hist.pop()

    def predict_prob(self, sym: bool) -> float:
        return 0.9 if sym else 0.1


class _Env(ait.EnvironmentABC):
    def __init__(self):
        self.obs = 1
        self.rew = 0

    def perform_action(self, action: int):
        self.obs = int(action) & 1
        self.rew = self.obs

    def get_observation(self) -> int:
        return self.obs

    def get_reward(self) -> int:
        return self.rew

    def is_finished(self) -> bool:
        return False

    def get_observation_bits(self) -> int:
        return 1

    def get_reward_bits(self) -> int:
        return 1

    def get_action_bits(self) -> int:
        return 1


class _Sim(ait.AgentSimulatorABC):
    def __init__(self):
        self.rng = random.Random(0)
        self.obs = 0

    def get_num_actions(self) -> int:
        return 2

    def get_num_observation_bits(self) -> int:
        return 1

    def get_num_reward_bits(self) -> int:
        return 1

    def horizon(self) -> int:
        return 2

    def max_reward(self) -> int:
        return 1

    def min_reward(self) -> int:
        return 0

    def model_update_action(self, action: int):
        self.obs = action & 1

    def gen_percept_and_update(self, bits: int) -> int:
        return self.obs if bits == 1 else 0

    def model_revert(self, steps: int):
        return None

    def gen_range(self, end: int) -> int:
        return self.rng.randrange(end)

    def gen_f64(self) -> float:
        return self.rng.random()


def test_predictor_default_methods():
    p = _Predictor()
    p.update_history(True)
    assert p.hist == [True]
    p.pop_history()
    assert p.hist == []
    assert p.predict_one() == p.predict_prob(True)
    assert p.model_name() == "_Predictor"
    clone = p.boxed_clone_with_seed(123)
    assert isinstance(clone, _Predictor)


def test_environment_default_drain_observations():
    env = _Env()
    env.perform_action(1)
    assert env.drain_observations() == [1]


def test_agent_simulator_default_methods():
    sim = _Sim()
    assert sim.observation_stream_len() == 1
    assert sim.observation_key_mode() == "fullstream"
    assert sim.reward_offset() == 0
    assert sim.get_explore_exploit_ratio() == 1.0
    assert sim.discount_gamma() == 1.0
    clone = sim.boxed_clone_with_seed(42)
    assert isinstance(clone, _Sim)
