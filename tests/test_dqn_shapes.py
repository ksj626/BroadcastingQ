import numpy as np
import pytest

pytest.importorskip("minigrid")
torch = pytest.importorskip("torch")

from agents.dqn import DQNAgent
from applications.key_door.env import KeyDoorEnv
from core.agent_base import Transition
from core.utils import load_yaml


def test_dqn_shapes_and_update():
    config = load_yaml("applications/key_door/config_dqn.yaml")
    env = KeyDoorEnv(config)
    try:
        rng = np.random.default_rng(0)
        agent = DQNAgent(
            env.observation_space,
            env.action_space,
            rng,
            batch_size=2,
            warmup_steps=1,
            replay_size=10,
            hidden_sizes=[16],
            target_update_interval=2,
            device="cpu",
        )

        obs, _ = env.reset(seed=0)
        flat = env.observation_space.flatten(obs)
        assert flat.shape == (env.observation_space.flat_dim,)

        x = torch.as_tensor(np.stack([flat, flat]), dtype=torch.float32)
        q_values = agent.q_network(x)
        assert q_values.shape == (2, env.action_space.n)

        for _ in range(2):
            action = env.action_space.sample(rng)
            next_obs, reward, terminated, truncated, info = env.step(action)
            out = agent.update(Transition(obs, action, reward, next_obs, terminated, truncated, info))
            obs = next_obs
            if terminated or truncated:
                obs, _ = env.reset()
        assert "loss" in out
    finally:
        env.close()
