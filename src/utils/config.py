"""YAML configuration loading.

All hyper-parameters, paths and ROS topics live in ``config/*.yaml``; code only
receives plain dictionaries / dataclasses built from them.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Union

import yaml

PathLike = Union[str, Path]

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "config"


def load_yaml(path: PathLike) -> Dict[str, Any]:
    """Loads a YAML file, expanding ``${ENV_VAR}`` and ``~`` in string values.

    Environment expansion lets machine-specific paths (dataset location) stay out of
    the repository: ``kitti_root: ${KITTI_ROOT}``.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    path = Path(path)
    if not path.is_absolute() and not path.exists():
        path = CONFIG_DIR / path
    with open(path, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    return _expand(data)


def _expand(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _expand(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_expand(item) for item in value]
    if isinstance(value, str):
        return os.path.expanduser(os.path.expandvars(value))
    return value
