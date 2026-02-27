import infotheory_rs as ait
import pathlib
import subprocess
import sys


class DummyPredictor(ait.PredictorABC):
    def __init__(self):
        self.hist = []

    def update(self, sym: bool):
        self.hist.append(bool(sym))

    def revert(self):
        if self.hist:
            self.hist.pop()

    def predict_prob(self, sym: bool) -> float:
        return 0.75 if sym else 0.25


class DummyEnv(ait.EnvironmentABC):
    def __init__(self):
        self.obs = 0
        self.rew = 0

    def perform_action(self, action: int):
        self.obs = action & 1
        self.rew = 1 if self.obs == 1 else 0

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


class DummySim(ait.AgentSimulatorABC):
    def __init__(self):
        self._rng = ait.RandomGenerator()
        self._obs = 0

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
        self._obs = action & 1

    def gen_percept_and_update(self, bits: int) -> int:
        if bits == 1:
            return self._obs
        return 0

    def model_revert(self, steps: int):
        return None

    def gen_range(self, end: int) -> int:
        return self._rng.gen_range(end)

    def gen_f64(self) -> float:
        return self._rng.gen_f64()

    def boxed_clone_with_seed(self, seed: int):
        c = DummySim()
        c._rng = self._rng.fork_with(seed)
        c._obs = self._obs
        return c


def test_predictor_probe_with_python_callback_object():
    probs, name = ait.predictor_probe(DummyPredictor(), steps=5)
    assert len(probs) == 5
    assert all(0.0 <= p <= 1.0 for p in probs)
    assert isinstance(name, str)


def test_environment_probe_with_python_callback_object():
    rows = ait.environment_probe(DummyEnv(), [0, 1, 1, 0])
    assert len(rows) == 4
    assert rows[1][1] == 1


def test_search_with_simulator_adapter():
    action = ait.search_with_simulator(DummySim(), [0], 0, 0, 4)
    assert action in (0, 1)


def _repo_root() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parents[2]


def test_predictor_callback_exception_is_fatal():
    code = """
import infotheory_rs as ait
class BadPred(ait.PredictorABC):
    def update(self, sym: bool): pass
    def revert(self): pass
    def predict_prob(self, sym: bool) -> float:
        raise RuntimeError("predictor prob boom")
ait.predictor_probe(BadPred(), steps=1)
"""
    proc = subprocess.run(
        [sys.executable, "-c", code],
        cwd=_repo_root(),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    assert proc.returncode != 0
    assert "Predictor.predict_prob" in proc.stderr
    assert "predictor prob boom" in proc.stderr


def test_environment_callback_exception_is_fatal():
    code = """
import infotheory_rs as ait
class BadEnv(ait.EnvironmentABC):
    def perform_action(self, action: int):
        raise RuntimeError("environment action boom")
    def get_observation(self) -> int: return 0
    def get_reward(self) -> int: return 0
    def is_finished(self) -> bool: return False
    def get_observation_bits(self) -> int: return 1
    def get_reward_bits(self) -> int: return 1
    def get_action_bits(self) -> int: return 1
ait.environment_probe(BadEnv(), [0])
"""
    proc = subprocess.run(
        [sys.executable, "-c", code],
        cwd=_repo_root(),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    assert proc.returncode != 0
    assert "Environment.perform_action" in proc.stderr
    assert "environment action boom" in proc.stderr
