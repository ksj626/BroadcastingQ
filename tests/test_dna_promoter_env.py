import numpy as np
import pytest

from applications.dna_promoter.env import DNAPromoterEnv
from core.spaces import MultiDiscreteSpace
from core.utils import load_yaml


def make_config(**env_updates):
    config = load_yaml("applications/dna_promoter/config_qlearning.yaml")
    config["env"].update(env_updates)
    return config


def test_dna_promoter_env_contract():
    env = DNAPromoterEnv(make_config())
    obs, info = env.reset(seed=0)

    assert isinstance(env.observation_space, MultiDiscreteSpace)
    assert env.observation_space.size == 4096
    assert env.action_space.n == 18
    assert isinstance(obs, np.ndarray)
    assert np.issubdtype(obs.dtype, np.integer)
    assert obs.shape == (6,)
    assert env.observation_space.contains(obs)
    assert isinstance(info["sequence"], str)

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


def test_dna_actions_always_change_exactly_one_base():
    env = DNAPromoterEnv(make_config(start_sequence="AAAAAA", target_sequences=["TTTTTT"]))
    env.reset(seed=0)

    for action in range(env.action_space.n):
        env.state = env.sequence_to_obs("AAAAAA")
        env.steps = 0
        before = env.obs_to_sequence(env.state)
        env.step(action)
        after = env.obs_to_sequence(env.state)
        assert env.hamming_distance(before, after) == 1


def test_dna_target_transition_returns_sparse_success():
    env = DNAPromoterEnv(make_config(start_sequence="AATAAT", target_sequences=["TATAAT"]))
    env.reset(seed=0)

    # At position 0, delta 3 maps A -> T under the cyclic mutation encoding.
    next_obs, reward, terminated, truncated, info = env.step(2)

    assert env.obs_to_sequence(next_obs) == "TATAAT"
    assert reward == 1.0
    assert terminated is True
    assert truncated is False
    assert info["success"] is True
    assert info["target_hit"] is True
    assert info["hamming_distance"] == 0


def test_dna_non_target_transition_and_hamming_info():
    env = DNAPromoterEnv(make_config(start_sequence="AAAAAA", target_sequences=["TATAAT"]))
    obs, info = env.reset(seed=0)

    assert env.obs_to_sequence(obs) == "AAAAAA"
    assert info["nearest_target"] == "TATAAT"
    assert info["hamming_distance"] == 3

    next_obs, reward, terminated, truncated, info = env.step(0)
    assert env.obs_to_sequence(next_obs) == "CAAAAA"
    assert reward == 0.0
    assert terminated is False
    assert truncated is False
    assert info["target_hit"] is False
    assert info["nearest_target"] == "TATAAT"
    assert info["hamming_distance"] == 3
