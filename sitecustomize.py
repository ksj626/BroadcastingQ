"""Project-local Python startup defaults."""

from __future__ import annotations

import warnings


warnings.filterwarnings(
    "ignore",
    message="CUDA initialization: The NVIDIA driver on your system is too old.*",
    category=UserWarning,
)
