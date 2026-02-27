"""Python bindings for `infotheory` (import name: `infotheory_rs`).

This module re-exports symbols from the native extension and provides:
- ergonomic wrappers for common entry points (`ncd_paths`, `ncd_bytes`)
- abstract base classes for Python-driven AIXI trait adapters

Callback error policy:
- Exceptions raised inside `PredictorABC`, `EnvironmentABC`, or
  `AgentSimulatorABC` callbacks are treated as fatal by the Rust shim layer.
- The process exits after printing callback context and traceback. This avoids
  silently continuing MCTS/planning with corrupted fallback values.
"""

from . import _core as _c
from abc import ABC, abstractmethod

for _name in dir(_c):
    if not _name.startswith("_"):
        globals()[_name] = getattr(_c, _name)


def ncd_paths(x, y, method="5", variant="vitanyi", backend=None):
    return _c.ncd_paths(x, y, method, variant, backend=backend)


def ncd_bytes(x, y, method="5", variant="vitanyi", backend=None):
    return _c.ncd_bytes(x, y, method, variant, backend=backend)


class PredictorABC(ABC):
    """Python-side adapter for the Rust `Predictor` trait."""

    @abstractmethod
    def update(self, sym: bool): ...

    def update_history(self, sym: bool):
        self.update(sym)

    @abstractmethod
    def revert(self): ...

    def pop_history(self):
        self.revert()

    @abstractmethod
    def predict_prob(self, sym: bool) -> float: ...

    def predict_one(self) -> float:
        return self.predict_prob(True)

    def model_name(self) -> str:
        return self.__class__.__name__

    def boxed_clone_with_seed(self, seed: int):
        import copy
        return copy.deepcopy(self)


class EnvironmentABC(ABC):
    """Python-side adapter for the Rust `Environment` trait."""

    @abstractmethod
    def perform_action(self, action: int): ...

    @abstractmethod
    def get_observation(self) -> int: ...

    def drain_observations(self):
        return [self.get_observation()]

    @abstractmethod
    def get_reward(self) -> int: ...

    @abstractmethod
    def is_finished(self) -> bool: ...

    @abstractmethod
    def get_observation_bits(self) -> int: ...

    @abstractmethod
    def get_reward_bits(self) -> int: ...

    @abstractmethod
    def get_action_bits(self) -> int: ...


class AgentSimulatorABC(ABC):
    """Python-side adapter for the Rust `AgentSimulator` trait."""

    @abstractmethod
    def get_num_actions(self) -> int: ...

    @abstractmethod
    def get_num_observation_bits(self) -> int: ...

    def observation_stream_len(self) -> int:
        return 1

    def observation_key_mode(self):
        return "fullstream"

    @abstractmethod
    def get_num_reward_bits(self) -> int: ...

    @abstractmethod
    def horizon(self) -> int: ...

    @abstractmethod
    def max_reward(self) -> int: ...

    @abstractmethod
    def min_reward(self) -> int: ...

    def reward_offset(self) -> int:
        return 0

    def get_explore_exploit_ratio(self) -> float:
        return 1.0

    def discount_gamma(self) -> float:
        return 1.0

    @abstractmethod
    def model_update_action(self, action: int): ...

    @abstractmethod
    def gen_percept_and_update(self, bits: int) -> int: ...

    @abstractmethod
    def model_revert(self, steps: int): ...

    @abstractmethod
    def gen_range(self, end: int) -> int: ...

    @abstractmethod
    def gen_f64(self) -> float: ...

    def boxed_clone_with_seed(self, seed: int):
        import copy
        return copy.deepcopy(self)


__all__ = [name for name in globals() if not name.startswith("_")]
