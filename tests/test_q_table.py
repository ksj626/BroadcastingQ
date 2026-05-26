from collections import defaultdict

import numpy as np

from agents.q_table import QTable
from core.spaces import DiscreteActionSpace, MultiDiscreteSpace


def test_q_table_is_sparse_and_lazy():
    q_table = QTable(MultiDiscreteSpace([2] * 25), DiscreteActionSpace(25))

    assert isinstance(q_table.values, defaultdict)
    assert len(q_table.values) == 0

    q_table.add(123, 4, 1.5)

    assert len(q_table.values) == 1
    assert q_table.get(123, 4) == np.float32(1.5)
    assert q_table.row(123).shape == (25,)


def test_q_table_save_load_roundtrip(tmp_path):
    q_table = QTable(MultiDiscreteSpace([3, 3]), DiscreteActionSpace(4), initial_value=0.25)
    q_table.add(2, 1, 1.0)
    q_table.add(5, 3, -0.5)

    path = tmp_path / "q_table.pkl"
    q_table.save(path)

    loaded = QTable(MultiDiscreteSpace([3, 3]), DiscreteActionSpace(4))
    loaded.load(path)

    assert len(loaded.values) == 2
    assert loaded.get(2, 1) == np.float32(1.25)
    assert loaded.get(5, 3) == np.float32(-0.25)
    assert loaded.get(8, 0) == np.float32(0.25)
