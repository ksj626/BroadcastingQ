import numpy as np

from core.spaces import MultiDiscreteSpace


def test_multidiscrete_contains():
    space = MultiDiscreteSpace([5, 5, 2, 3])
    assert space.contains(np.array([2, 1, 0, 2], dtype=np.int64))
    assert not space.contains(np.array([5, 1, 0, 2], dtype=np.int64))
    assert not space.contains(np.array([1, 0, 2], dtype=np.int64))
    assert not space.contains(np.array([1.5, 1, 0, 2]))


def test_to_index_from_index_roundtrip():
    space = MultiDiscreteSpace([3, 4, 2])
    for index in range(space.size):
        obs = space.from_index(index)
        assert space.contains(obs)
        assert space.to_index(obs) == index


def test_flatten_dimension_and_values():
    space = MultiDiscreteSpace([5, 5, 2, 3])
    flat = space.flatten(np.array([2, 1, 0, 2], dtype=np.int64))
    assert flat.shape == (space.flat_dim,)
    assert flat.sum() == len(space.nvec)
    assert flat.dtype == np.float32
