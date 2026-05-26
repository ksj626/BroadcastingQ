from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import pickle

import numpy as np

from core.spaces import DiscreteActionSpace, MultiDiscreteSpace


class QTable:
    def __init__(
        self,
        observation_space: MultiDiscreteSpace,
        action_space: DiscreteActionSpace,
        initial_value: float = 0.0,
    ) -> None:
        self.observation_space = observation_space
        self.action_space = action_space
        self.initial_value = float(initial_value)
        self.values = defaultdict(self._new_row)

    def state_index(self, obs) -> int:
        return self.observation_space.to_index(obs)

    def row(self, state_index: int) -> np.ndarray:
        return self.values[int(state_index)]

    def get(self, state_index: int, action: int) -> float:
        return float(self.row(state_index)[int(action)])

    def add(self, state_index: int, action: int, value: float) -> float:
        row = self.row(state_index)
        row[int(action)] += float(value)
        return float(row[int(action)])

    def save(self, path: str | Path) -> None:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "initial_value": self.initial_value,
            "action_count": self.action_space.n,
            "values": dict(self.values),
        }
        with out.open("wb") as f:
            pickle.dump(payload, f)

    def load(self, path: str | Path) -> None:
        with Path(path).open("rb") as f:
            payload = pickle.load(f)
        action_count = int(payload.get("action_count", self.action_space.n))
        if action_count != self.action_space.n:
            raise ValueError(f"Checkpoint action count {action_count} does not match {self.action_space.n}")
        self.initial_value = float(payload.get("initial_value", self.initial_value))
        self.values = defaultdict(self._new_row)
        for state_index, row in payload.get("values", {}).items():
            arr = np.asarray(row, dtype=np.float32)
            if arr.shape != (self.action_space.n,):
                raise ValueError(f"Invalid Q row shape for state {state_index}: {arr.shape}")
            self.values[int(state_index)] = arr.copy()

    def _new_row(self) -> np.ndarray:
        return np.full(self.action_space.n, self.initial_value, dtype=np.float32)
