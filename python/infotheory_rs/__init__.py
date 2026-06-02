"""Python bindings for `infotheory` (import name: `infotheory_rs`).

This module re-exports symbols from the native extension and provides:

- ergonomic wrappers for common entry points (`ncd_paths`, `ncd_bytes`)
- abstract base classes for Python-driven AIXI trait adapters
- bit-level predictive primitives (`RateBackendBitSession`, `RateBackendBitSessionCheckpoint`, `BytePrefixMass`, `BinaryPrediction`)
- bit-stream semantics and ordering controls (`BitStreamSemantics`, `BitOrder`)
- context entrypoint for bit sessions (`InfotheoryCtx.rate_backend_bit_session`)

Python `RateBackendBitSession` exposes `predict_bit`, `predict_one`, `step_bit`, `observe_bit`, `condition_bit`, `reset_frozen`, `begin_bit_stream`, `checkpoint`, `restore_checkpoint`, `clear_checkpoints_if_supported`, and `finish`. The checkpoint object is `RateBackendBitSessionCheckpoint`. See the native docstring on RateBackendBitSession for details.

Accepted string aliases:

- `BitOrder`: `"msb"`, `"msbfirst"`, `"msb_first"`, `"lsb"`, `"lsbfirst"`, `"lsb_first"`
- `BitStreamSemantics`: `"bytepacked"`, `"byte_packed"`, `"byte"` map to
  `BitStreamSemantics.byte_packed(order=BitOrder.MsbFirst)`; `"binarytokens"`,
  `"binary_tokens"`, `"binary"`, `"bit"` map to
  `BitStreamSemantics.binary_tokens()`

Callback error policy:

- Exceptions raised inside `PredictorABC`, `EnvironmentABC`, or
  `AgentSimulatorABC` callbacks are treated as fatal by the Rust shim layer.
- The process exits after printing callback context and traceback. This avoids
  silently continuing MCTS/planning with corrupted fallback values.
"""

from . import _core as _c
from abc import ABC, abstractmethod

_REMOVED_PUBLIC_SYMBOLS = frozenset({"SearchNode"})

for _name in dir(_c):
    if not _name.startswith("_") and _name not in _REMOVED_PUBLIC_SYMBOLS:
        globals()[_name] = getattr(_c, _name)


def ncd_paths(x, y, method="5", variant="vitanyi", backend=None):
    return _c.ncd_paths(x, y, method, variant, backend=backend)


def ncd_bytes(x, y, method="5", variant="vitanyi", backend=None):
    return _c.ncd_bytes(x, y, method, variant, backend=backend)


def _deepcopy_with_seeded_rng_fields(obj, seed: int):
    import copy
    import random
    import types

    def _slot_names(instance):
        cls = type(instance)
        for base in cls.__mro__:
            base_slots = getattr(base, "__slots__", ())
            if isinstance(base_slots, str):
                base_slots = (base_slots,)
            for slot in base_slots:
                if slot not in {"__dict__", "__weakref__"}:
                    yield slot

    def _is_forkable_rng(value):
        return (
            hasattr(value, "fork_with")
            and callable(value.fork_with)
            and hasattr(value, "gen_range")
            and callable(value.gen_range)
            and hasattr(value, "gen_f64")
            and callable(value.gen_f64)
        )

    def _iter_children(value):
        if isinstance(value, dict):
            for key, child in value.items():
                yield key
                yield child
            return

        if isinstance(value, (list, tuple, set, frozenset)):
            for child in value:
                yield child
            return

        if hasattr(value, "__dict__"):
            for child in vars(value).values():
                yield child

        for slot in _slot_names(value):
            if hasattr(value, slot):
                yield getattr(value, slot)

    seed = int(seed) & ((1 << 64) - 1)
    memo = {}
    seen = set()
    stack = [obj]

    while stack:
        value = stack.pop()
        value_id = id(value)
        if value_id in seen:
            continue
        seen.add(value_id)

        if _is_forkable_rng(value):
            memo[value_id] = value.fork_with(seed)
        elif isinstance(value, random.Random):
            copied = copy.deepcopy(value, memo)
            copied.seed(seed)
            memo[value_id] = copied
        elif isinstance(
            value,
            (
                str,
                bytes,
                bytearray,
                memoryview,
                int,
                float,
                bool,
                complex,
                type(None),
                types.FunctionType,
                types.BuiltinFunctionType,
                types.MethodType,
                types.ModuleType,
                type,
            ),
        ):
            continue
        else:
            stack.extend(_iter_children(value))

    return copy.deepcopy(obj, memo)


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
        return "full_stream"

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
        return _deepcopy_with_seeded_rng_fields(self, seed)


__all__ = [name for name in globals() if not name.startswith("_")]
