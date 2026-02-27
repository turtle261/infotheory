import infotheory_rs as ait


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


class ErrorPredictor(ait.PredictorABC):
    def update(self, sym: bool):
        raise RuntimeError("predictor update boom")

    def revert(self):
        raise RuntimeError("predictor revert boom")

    def predict_prob(self, sym: bool) -> float:
        raise RuntimeError("predictor prob boom")


def test_predictor_callback_exception_printed_and_defaulted(capsys):
    probs, name = ait.predictor_probe(ErrorPredictor(), steps=3)
    assert probs == [0.5, 0.5, 0.5]
    assert isinstance(name, str)
    err = capsys.readouterr().err
    assert "predictor prob boom" in err


class ErrorEnv(ait.EnvironmentABC):
    def perform_action(self, action: int):
        raise RuntimeError("environment action boom")

    def get_observation(self) -> int:
        return 0

    def get_reward(self) -> int:
        return 0

    def is_finished(self) -> bool:
        return False

    def get_observation_bits(self) -> int:
        return 1

    def get_reward_bits(self) -> int:
        return 1

    def get_action_bits(self) -> int:
        return 1


def test_environment_callback_exception_printed_and_continues(capsys):
    rows = ait.environment_probe(ErrorEnv(), [0, 1])
    assert rows == [(0, 0, False), (0, 0, False)]
    err = capsys.readouterr().err
    assert "environment action boom" in err
