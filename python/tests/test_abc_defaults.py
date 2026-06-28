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


class _ForkableSim(ait.AgentSimulatorABC):
    def __init__(self):
        self.rng = ait.RandomGenerator()
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
        return self.rng.gen_range(end)

    def gen_f64(self) -> float:
        return self.rng.gen_f64()


class _SharedRngSim(ait.AgentSimulatorABC):
    __slots__ = ("rng", "mirror", "obs")

    def __init__(self):
        self.rng = random.Random(0)
        self.mirror = self.rng
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


class _StringSlotSim(ait.AgentSimulatorABC):
    __slots__ = "rng"

    def __init__(self):
        self.rng = random.Random(0)

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
        return None

    def gen_percept_and_update(self, bits: int) -> int:
        return 0

    def model_revert(self, steps: int):
        return None

    def gen_range(self, end: int) -> int:
        return self.rng.randrange(end)

    def gen_f64(self) -> float:
        return self.rng.random()


class _NestedRngContainerSim(ait.AgentSimulatorABC):
    def __init__(self):
        self.obs = 0
        self.state = {
            "primary": random.Random(0),
            "secondary": [random.Random(1), random.Random(2)],
        }

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
        return self.state["primary"].randrange(end)

    def gen_f64(self) -> float:
        return self.state["secondary"][0].random()


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
    assert sim.observation_key_mode() == "full_stream"
    assert sim.reward_offset() == 0
    assert sim.get_explore_exploit_ratio() == 1.0
    assert sim.discount_gamma() == 1.0
    clone = sim.boxed_clone_with_seed(42)
    assert isinstance(clone, _Sim)


def test_agent_simulator_default_clone_reseeds_python_random():
    sim = _Sim()
    clone_a = sim.boxed_clone_with_seed(42)
    clone_b = sim.boxed_clone_with_seed(42)
    clone_c = sim.boxed_clone_with_seed(43)

    seq_a = [clone_a.gen_range(1 << 30) for _ in range(8)]
    seq_b = [clone_b.gen_range(1 << 30) for _ in range(8)]
    seq_c = [clone_c.gen_range(1 << 30) for _ in range(8)]

    assert seq_a == seq_b
    assert seq_a != seq_c


def test_agent_simulator_default_clone_forks_infotheory_rng():
    sim = _ForkableSim()
    clone_a = sim.boxed_clone_with_seed(101)
    clone_b = sim.boxed_clone_with_seed(101)
    clone_c = sim.boxed_clone_with_seed(202)

    seq_a = [clone_a.gen_range(1 << 30) for _ in range(8)]
    seq_b = [clone_b.gen_range(1 << 30) for _ in range(8)]
    seq_c = [clone_c.gen_range(1 << 30) for _ in range(8)]

    assert seq_a == seq_b
    assert seq_a != seq_c


def test_agent_simulator_default_clone_preserves_shared_references():
    sim = _SharedRngSim()
    clone = sim.boxed_clone_with_seed(1234)
    assert clone.rng is clone.mirror


def test_agent_simulator_default_clone_supports_string_slots():
    sim = _StringSlotSim()
    clone_a = sim.boxed_clone_with_seed(11)
    clone_b = sim.boxed_clone_with_seed(11)
    clone_c = sim.boxed_clone_with_seed(12)

    seq_a = [clone_a.gen_range(1 << 30) for _ in range(8)]
    seq_b = [clone_b.gen_range(1 << 30) for _ in range(8)]
    seq_c = [clone_c.gen_range(1 << 30) for _ in range(8)]

    assert seq_a == seq_b
    assert seq_a != seq_c


def test_agent_simulator_default_clone_reseeds_nested_rng_containers():
    sim = _NestedRngContainerSim()
    clone_a = sim.boxed_clone_with_seed(2001)
    clone_b = sim.boxed_clone_with_seed(2001)
    clone_c = sim.boxed_clone_with_seed(2002)

    def _draw_sequences(clone):
        primary = [clone.state["primary"].randrange(1 << 30) for _ in range(5)]
        second0 = [clone.state["secondary"][0].randrange(1 << 30) for _ in range(5)]
        second1 = [clone.state["secondary"][1].randrange(1 << 30) for _ in range(5)]
        return primary, second0, second1

    seq_a = _draw_sequences(clone_a)
    seq_b = _draw_sequences(clone_b)
    seq_c = _draw_sequences(clone_c)

    assert seq_a == seq_b
    assert seq_a != seq_c
