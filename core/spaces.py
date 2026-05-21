from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property

import numpy as np


@dataclass(frozen=True)
class MultiDiscreteSpace:
    nvec: list[int]

    def __post_init__(self) -> None:
        if not self.nvec:
            raise ValueError("nvec must contain at least one factor")
        if any(int(n) <= 0 for n in self.nvec):
            raise ValueError("all nvec entries must be positive")
        object.__setattr__(self, "nvec", [int(n) for n in self.nvec])

    def sample(self, rng: np.random.Generator) -> np.ndarray:
        return np.array([rng.integers(n) for n in self.nvec], dtype=np.int64)

    def contains(self, x) -> bool:
        arr = np.asarray(x)
        if arr.shape != (len(self.nvec),):
            return False
        if not np.issubdtype(arr.dtype, np.integer):
            return False
        return bool(np.all(arr >= 0) and np.all(arr < np.asarray(self.nvec)))

    def flatten(self, x) -> np.ndarray:
        arr = self._validate_array(x)
        out = np.zeros(self.flat_dim, dtype=np.float32)
        offset = 0
        for value, n in zip(arr, self.nvec):
            out[offset + int(value)] = 1.0
            offset += n
        return out

    def to_index(self, x) -> int:
        arr = self._validate_array(x)
        index = 0
        multiplier = 1
        for value, radix in zip(reversed(arr), reversed(self.nvec)):
            index += int(value) * multiplier
            multiplier *= radix
        return int(index)

    def from_index(self, index: int) -> np.ndarray:
        index = int(index)
        if index < 0 or index >= self.size:
            raise ValueError(f"index {index} is outside [0, {self.size})")
        values = []
        remaining = index
        for radix in reversed(self.nvec):
            values.append(remaining % radix)
            remaining //= radix
        return np.array(list(reversed(values)), dtype=np.int64)

    @cached_property
    def size(self) -> int:
        return int(np.prod(self.nvec, dtype=np.int64))

    @cached_property
    def flat_dim(self) -> int:
        return int(sum(self.nvec))

    def _validate_array(self, x) -> np.ndarray:
        arr = np.asarray(x, dtype=np.int64)
        if not self.contains(arr):
            raise ValueError(f"{x!r} is not contained in MultiDiscreteSpace({self.nvec})")
        return arr


@dataclass(frozen=True)
class DiscreteActionSpace:
    n: int

    def __post_init__(self) -> None:
        if int(self.n) <= 0:
            raise ValueError("n must be positive")
        object.__setattr__(self, "n", int(self.n))

    def sample(self, rng: np.random.Generator) -> int:
        return int(rng.integers(self.n))

    def contains(self, a) -> bool:
        if isinstance(a, (bool, np.bool_)):
            return False
        value = int(a)
        return value == a and 0 <= value < self.n
