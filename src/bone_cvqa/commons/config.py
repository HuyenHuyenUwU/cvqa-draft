"""Nạp cấu hình YAML: base.yaml + (data|model|training|inference).yaml được gộp theo thứ tự."""
import copy
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import yaml

from .exceptions import ConfigError

DEFAULT_CONFIG_DIR = Path(__file__).resolve().parents[3] / "configs"


def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def load_yaml(path: Path) -> Dict[str, Any]:
    if not Path(path).is_file():
        raise ConfigError(f"Không tìm thấy file cấu hình: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_config(names: Iterable[str] = ("base",), config_dir: Optional[Path] = None,
                overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Ví dụ: load_config(["base", "data", "model", "training"])."""
    config_dir = Path(config_dir) if config_dir else DEFAULT_CONFIG_DIR
    config: Dict[str, Any] = {}
    for name in names:
        config = deep_merge(config, load_yaml(config_dir / f"{name}.yaml"))
    return deep_merge(config, overrides or {})


def parse_overrides(items: Optional[List[str]]) -> Dict[str, Any]:
    """['train.epochs=3', 'external_models.vision_encoder.device=cpu'] -> dict lồng nhau."""
    result: Dict[str, Any] = {}
    for item in items or []:
        key, value = item.split("=", 1)
        *parents, last = key.split(".")
        node = result
        for parent in parents:
            node = node.setdefault(parent, {})
        node[last] = yaml.safe_load(value)
    return result


def save_config(config: Dict[str, Any], path: Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False, allow_unicode=True)
