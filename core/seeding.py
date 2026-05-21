from __future__ import annotations

import random
import warnings

import numpy as np


def set_global_seed(seed: int) -> np.random.Generator:
    random.seed(seed)
    np.random.seed(seed)
    warnings.filterwarnings(
        "ignore",
        message="CUDA initialization: The NVIDIA driver on your system is too old.*",
        category=UserWarning,
    )
    try:
        import torch

        torch.manual_seed(seed)
    except ImportError:
        pass
    return np.random.default_rng(seed)
