"""Config loading and shared path helpers.

Every module reads settings through here so paths, the valid-hour window, the
target column and the split years each have exactly one definition.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "config" / "settings.yaml"


@lru_cache(maxsize=1)
def load_config(path: Path | None = None) -> dict[str, Any]:
    with open(path or CONFIG_PATH) as handle:
        return yaml.safe_load(handle)


def resolve(key: str) -> Path:
    """Resolve a `paths:` entry to an absolute path, creating it if needed."""
    cfg = load_config()
    target = PROJECT_ROOT / cfg["paths"][key]
    if target.suffix:
        target.parent.mkdir(parents=True, exist_ok=True)
    else:
        target.mkdir(parents=True, exist_ok=True)
    return target


def env_secret(name: str) -> str | None:
    """Secrets come from the environment only - never from the repo.

    Nothing in this project needs one: DfT road traffic counts are published
    open, key-free, under the Open Government Licence.
    """
    return os.environ.get(name)
