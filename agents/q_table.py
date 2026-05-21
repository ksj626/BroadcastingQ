from __future__ import annotations

from pathlib import Path

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
        self.values = np.full(
            (observation_space.size, action_space.n),
            float(initial_value),
            dtype=np.float32,
        )

    def state_index(self, obs) -> int:
        return self.observation_space.to_index(obs)

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        np.save(path, self.values)

    def load(self, path: str | Path) -> None:
        self.values[...] = np.load(path)
