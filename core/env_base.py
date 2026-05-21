from __future__ import annotations

from core.spaces import DiscreteActionSpace, MultiDiscreteSpace


class BaseEnv:
    observation_space: MultiDiscreteSpace
    action_space: DiscreteActionSpace

    def reset(self, seed: int | None = None):
        raise NotImplementedError

    def step(self, action: int):
        raise NotImplementedError

    def render(self, mode: str = "rgb_array"):
        raise NotImplementedError

    def close(self) -> None:
        pass
