from __future__ import annotations

import numpy as np


class ReplayBuffer:
    def __init__(self, capacity: int, obs_shape: tuple[int, ...], obs_dtype=np.int64) -> None:
        self.capacity = int(capacity)
        self.obs = np.zeros((self.capacity, *obs_shape), dtype=obs_dtype)
        self.next_obs = np.zeros((self.capacity, *obs_shape), dtype=obs_dtype)
        self.actions = np.zeros(self.capacity, dtype=np.int64)
        self.rewards = np.zeros(self.capacity, dtype=np.float32)
        self.terminated = np.zeros(self.capacity, dtype=np.bool_)
        self.truncated = np.zeros(self.capacity, dtype=np.bool_)
        self._pos = 0
        self._size = 0

    def add(self, obs, action: int, reward: float, next_obs, terminated: bool, truncated: bool) -> None:
        self.obs[self._pos] = np.asarray(obs)
        self.actions[self._pos] = int(action)
        self.rewards[self._pos] = float(reward)
        self.next_obs[self._pos] = np.asarray(next_obs)
        self.terminated[self._pos] = bool(terminated)
        self.truncated[self._pos] = bool(truncated)
        self._pos = (self._pos + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(self, batch_size: int, rng: np.random.Generator) -> dict[str, np.ndarray]:
        if self._size < batch_size:
            raise ValueError("not enough samples in replay buffer")
        idx = rng.integers(0, self._size, size=int(batch_size))
        return {
            "obs": self.obs[idx],
            "actions": self.actions[idx],
            "rewards": self.rewards[idx],
            "next_obs": self.next_obs[idx],
            "terminated": self.terminated[idx],
            "truncated": self.truncated[idx],
        }

    def __len__(self) -> int:
        return self._size
