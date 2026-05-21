from __future__ import annotations

import numpy as np

from core.agent_base import BaseAgent
from core.spaces import DiscreteActionSpace


class RandomAgent(BaseAgent):
    def __init__(self, action_space: DiscreteActionSpace, rng: np.random.Generator, **kwargs) -> None:
        self.action_space = action_space
        self.rng = rng

    def act(self, obs, explore: bool = True) -> int:
        return self.action_space.sample(self.rng)
