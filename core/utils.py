from __future__ import annotations

import importlib
from pathlib import Path

import yaml


def import_from_string(path: str):
    module_path, object_name = path.split(":")
    module = importlib.import_module(module_path)
    return getattr(module, object_name)


def load_yaml(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dir(path: str | Path) -> Path:
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out
