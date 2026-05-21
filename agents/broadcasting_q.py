from __future__ import annotations

from core.agent_base import BaseAgent


class SBQAgent(BaseAgent):
    """
    TODO:
    Implement Structural Broadcasting Q-learning.

    Intended future form:
        Q(s,a) = sum_g sqrt(eta_g) * theta[a, g, s_g]

    This class is intentionally not implemented yet.
    """

    def __init__(self, *args, **kwargs):
        raise NotImplementedError("SBQ is intentionally left as TODO.")
