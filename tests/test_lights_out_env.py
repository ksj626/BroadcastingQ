import numpy as np
import pytest

from applications.lights_out.env import LightsOutEnv
from core.spaces import MultiDiscreteSpace
from core.utils import load_yaml


def make_config(**env_updates):
    config = load_yaml("applications/lights_out/config_qlearning_3x3.yaml")
    config["env"].update(env_updates)
    return config


def test_lights_out_env_contract():
    env = LightsOutEnv(make_config())
    obs, info = env.reset(seed=0)

    assert isinstance(env.observation_space, MultiDiscreteSpace)
    assert env.observation_space.size == 512
    assert env.action_space.n == 9
    assert isinstance(obs, np.ndarray)
    assert np.issubdtype(obs.dtype, np.integer)
    assert obs.shape == (9,)
    assert env.observation_space.contains(obs)
    assert isinstance(info["lights_on"], int)

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

    with pytest.raises(ValueError):
        env.step(env.action_space.n)


def test_lights_out_center_toggle_pattern_is_original_rule():
    env = LightsOutEnv(make_config(start_mode="off", action_slip_prob=0.0))
    env.reset(seed=0)

    next_obs, reward, terminated, truncated, info = env.step(4)
    expected = np.array(
        [
            [0, 1, 0],
            [1, 1, 1],
            [0, 1, 0],
        ],
        dtype=np.int64,
    )

    assert np.array_equal(next_obs.reshape(3, 3), expected)
    assert reward == 0.0
    assert terminated is False
    assert truncated is False
    assert info["lights_on"] == 5


def test_lights_out_solved_board_returns_sparse_success():
    start_board = [
        [0, 1, 0],
        [1, 1, 1],
        [0, 1, 0],
    ]
    env = LightsOutEnv(make_config(start_board=start_board, action_slip_prob=0.0))
    env.reset(seed=0)

    next_obs, reward, terminated, truncated, info = env.step(4)

    assert np.array_equal(next_obs, np.zeros(9, dtype=np.int64))
    assert reward == 1.0
    assert terminated is True
    assert truncated is False
    assert info["success"] is True
    assert info["lights_on"] == 0


def test_lights_out_seeded_reset_is_reproducible():
    env = LightsOutEnv(make_config(scramble_steps=9, action_slip_prob=0.0))

    obs1, info1 = env.reset(seed=123)
    obs2, info2 = env.reset(seed=123)

    assert np.array_equal(obs1, obs2)
    assert info1["lights_on"] == info2["lights_on"]


def test_lights_out_action_slip_reports_replacement():
    env = LightsOutEnv(make_config(start_mode="off", action_slip_prob=1.0))
    env.reset(seed=0)

    _, _, _, _, info = env.step(0)

    assert info["requested_action"] == 0
    assert info["action_slipped"] is True
    assert info["executed_action"] != 0
    assert 0 <= info["executed_action"] < env.action_space.n
