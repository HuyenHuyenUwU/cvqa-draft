"""YAML configuration loading.

A top-level YAML may contain an ``includes`` mapping ``{key: relative/path.yaml}``;
each included file is loaded (relative to the including file) and nested under
``key``. Remaining keys of the top-level file override included values.
"""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from .exceptions import ConfigError


def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_yaml(path) -> Dict[str, Any]:
    p = Path(path)
    if not p.is_file():
        raise ConfigError(f"Config file not found: {p}")
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_config(path, overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    p = Path(path)
    cfg = load_yaml(p)
    includes = cfg.pop("includes", None) or {}
    merged: Dict[str, Any] = {}
    for key, rel in includes.items():
        merged[key] = load_yaml(p.parent / rel)
    merged = deep_merge(merged, cfg)
    if overrides:
        merged = deep_merge(merged, overrides)
    return merged


def save_config(cfg: Dict[str, Any], path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
