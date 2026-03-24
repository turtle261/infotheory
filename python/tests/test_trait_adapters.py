import infotheory_rs as ait
import os
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


class DummySimWithKeyMode(DummySim):
    def __init__(self, mode: str):
        super().__init__()
        self._mode = mode

    def observation_key_mode(self):
        return self._mode


class RunnerTupleEnv(ait.EnvironmentABC):
    def __init__(self):
        self.obs = 0
        self.rew = 0
        self.finished = False
        self.perform_calls = 0
        self.get_obs_calls = 0
        self.get_rew_calls = 0

    def perform_action(self, action: int):
        self.perform_calls += 1
        self.obs = action & 1
        self.rew = 1 if self.obs else 0
        return (self.obs, self.rew)

    def get_observation(self) -> int:
        self.get_obs_calls += 1
        return self.obs

    def get_reward(self) -> int:
        self.get_rew_calls += 1
        return self.rew

    def is_finished(self) -> bool:
        return self.finished

    def get_observation_bits(self) -> int:
        return 1

    def get_reward_bits(self) -> int:
        return 1

    def get_action_bits(self) -> int:
        return 1


def _test_agent_config() -> ait.AgentConfig:
    return ait.AgentConfig(
        algorithm="ac-ctw",
        ct_depth=8,
        agent_horizon=2,
        observation_bits=1,
        observation_stream_len=1,
        reward_bits=1,
        agent_actions=2,
        num_simulations=8,
        exploration_exploitation_ratio=1.41,
        discount_gamma=1.0,
        min_reward=0,
        max_reward=1,
        reward_offset=0,
    )


def test_predictor_probe_with_python_callback_object():
    probs, name = ait.predictor_probe(DummyPredictor(), steps=5)
    assert len(probs) == 5
    assert all(0.0 <= p <= 1.0 for p in probs)
    assert isinstance(name, str)


def test_environment_probe_with_python_callback_object():
    rows = ait.environment_probe(DummyEnv(), [0, 1, 1, 0])
    assert len(rows) == 4
    assert rows[1][1] == 1


def test_run_agent_with_environment_adapter():
    env = DummyEnv()
    summary = ait.run_agent_with_environment(
        env,
        _test_agent_config(),
        learn_cycles=4,
        eval_cycles=6,
        terminate_lifetime=10,
    )
    assert summary["learn_cycles_completed"] == 4
    assert summary["eval_cycles_completed"] == 6
    assert isinstance(summary["eval_average_reward"], float)


def test_run_agent_with_environment_uses_perform_action_tuple_fast_path():
    env = RunnerTupleEnv()
    summary = ait.run_agent_with_environment(
        env,
        _test_agent_config(),
        learn_cycles=10,
        eval_cycles=0,
        check_finished=False,
    )
    assert summary["learn_cycles_completed"] == 10
    assert env.perform_calls == 10
    assert env.get_obs_calls == 1
    assert env.get_rew_calls == 1


def test_search_with_simulator_adapter():
    action = ait.search_with_simulator(DummySim(), [0], 0, 0, 4)
    assert action in (0, 1)


def test_search_with_simulator_accepts_cli_observation_key_aliases():
    for mode in ("full", "full-stream", "stream-hash"):
        action = ait.search_with_simulator(DummySimWithKeyMode(mode), [0], 0, 0, 4)
        assert action in (0, 1)


def _repo_root() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parents[2]


def test_search_with_simulator_respects_rayon_num_threads_env():
    code = """
import infotheory_rs as ait
class ProbeSim(ait.AgentSimulatorABC):
    def __init__(self):
        self._rng = ait.RandomGenerator()
        self._obs = 0
        self.seed_clone_calls = 0
    def get_num_actions(self) -> int: return 2
    def get_num_observation_bits(self) -> int: return 1
    def get_num_reward_bits(self) -> int: return 1
    def horizon(self) -> int: return 2
    def max_reward(self) -> int: return 1
    def min_reward(self) -> int: return 0
    def model_update_action(self, action: int): self._obs = action & 1
    def gen_percept_and_update(self, bits: int) -> int: return self._obs if bits == 1 else 0
    def model_revert(self, steps: int): return None
    def gen_range(self, end: int) -> int: return self._rng.gen_range(end)
    def gen_f64(self) -> float: return self._rng.gen_f64()
    def boxed_clone_with_seed(self, seed: int):
        self.seed_clone_calls += 1
        c = ProbeSim()
        c._rng = self._rng.fork_with(seed)
        c._obs = self._obs
        return c
sim = ProbeSim()
ait.search_with_simulator(sim, [0], 0, 0, 8)
print(sim.seed_clone_calls)
"""
    env = dict(os.environ)
    env["RAYON_NUM_THREADS"] = "2"
    proc = subprocess.run(
        [sys.executable, "-c", code],
        cwd=_repo_root(),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    assert int(proc.stdout.strip()) >= 1


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
