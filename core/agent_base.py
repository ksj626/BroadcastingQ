from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class Transition:
    obs: Any
    action: int
    reward: float
    next_obs: Any
    terminated: bool
    truncated: bool
    info: dict
    next_action: Optional[int] = None

    @property
    def done(self) -> bool:
        return self.terminated or self.truncated


class BaseAgent:
    def act(self, obs, explore: bool = True) -> int:
        raise NotImplementedError

    def update(self, transition: Transition) -> dict:
        return {}

    def begin_episode(self) -> None:
        pass

    def end_episode(self) -> None:
        pass

    def save(self, path: str) -> None:
        pass

    def load(self, path: str) -> None:
        pass
