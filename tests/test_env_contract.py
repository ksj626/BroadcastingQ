import numpy as np
import pytest

pytest.importorskip("minigrid")

from applications.key_door.env import KeyDoorEnv
from core.spaces import MultiDiscreteSpace
from core.utils import load_yaml


def test_keydoor_env_contract():
    config = load_yaml("applications/key_door/config_qlearning.yaml")
    env = KeyDoorEnv(config)
    try:
        obs, info = env.reset(seed=0)
        assert isinstance(env.observation_space, MultiDiscreteSpace)
        assert isinstance(obs, np.ndarray)
        assert np.issubdtype(obs.dtype, np.integer)
        assert obs.shape == (len(env.observation_space.nvec),)
        assert env.observation_space.contains(obs)

        rng = np.random.default_rng(0)
        for _ in range(10):
            action = env.action_space.sample(rng)
            next_obs, reward, terminated, truncated, info = env.step(action)
            assert env.observation_space.contains(next_obs)
            assert isinstance(reward, float)
            assert isinstance(terminated, bool)
            assert isinstance(truncated, bool)
            assert isinstance(info, dict)
            if terminated or truncated:
                env.reset()

        frame = env.render(mode="rgb_array")
        assert isinstance(frame, np.ndarray)
        assert frame.ndim == 3
        assert frame.shape[-1] == 3
    finally:
        env.close()
