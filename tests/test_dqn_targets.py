import numpy as np
import pytest

torch = pytest.importorskip("torch")

from agents.dqn import DQNAgent
from core.agent_base import Transition
from core.spaces import DiscreteActionSpace, MultiDiscreteSpace


def test_dqn_bootstraps_through_truncation():
    obs_space = MultiDiscreteSpace([2])
    action_space = DiscreteActionSpace(2)
    agent = DQNAgent(
        obs_space,
        action_space,
        np.random.default_rng(0),
        gamma=0.5,
        learning_rate=0.0,
        batch_size=1,
        warmup_steps=1,
        replay_size=4,
        hidden_sizes=[],
        device="cpu",
    )
    with torch.no_grad():
        agent.q_network.net[0].weight.zero_()
        agent.q_network.net[0].bias.zero_()
        agent.target_network.net[0].weight.zero_()
        agent.target_network.net[0].bias[:] = torch.tensor([2.0, 1.0])

    info = agent.update(
        Transition(
            obs=np.array([0], dtype=np.int64),
            action=0,
            reward=1.0,
            next_obs=np.array([1], dtype=np.int64),
            terminated=False,
            truncated=True,
            info={},
        )
    )

    assert info["loss"] == pytest.approx(4.0)
    assert info["td_error_mean"] == pytest.approx(2.0)
