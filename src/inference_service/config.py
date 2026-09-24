"""Loads config/config.yaml once per process, resolves relative paths against
the repo root (not the process's CWD, which varies between pytest/uvicorn/
Docker), and expands ${VAR} placeholders from the environment (.env first)."""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Union

import yaml
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"


def _expand(value: Any) -> Any:
    if isinstance(value, str):
        return os.path.expandvars(value)
    if isinstance(value, dict):
        return {k: _expand(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand(v) for v in value]
    return value


def _to_namespace(value: Any) -> Any:
    if isinstance(value, dict):
        return SimpleNamespace(**{k: _to_namespace(v) for k, v in value.items()})
    return value


@lru_cache(maxsize=1)
def get_config() -> SimpleNamespace:
    """Process-wide singleton — config.yaml is parsed once, not per request."""
    load_dotenv(PROJECT_ROOT / ".env")  # no-op if .env doesn't exist
    with open(DEFAULT_CONFIG_PATH, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return _to_namespace(_expand(raw))


def resolve_path(relative_path: Union[str, Path]) -> Path:
    """Resolve a config-file path against the repo root."""
    p = Path(relative_path)
    return p if p.is_absolute() else PROJECT_ROOT / p
