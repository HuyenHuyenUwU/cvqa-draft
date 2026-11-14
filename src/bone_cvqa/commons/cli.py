"""Helper CLI dùng chung để MỖI module có thể chạy độc lập: python -m bone_cvqa.<module>.run ..."""
import argparse
import json
from typing import Any, Dict, List, Optional

import torch

from .config import load_config, parse_overrides
from .logger import setup_logging


def base_parser(description: str, config_names: str = "base,model") -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--configs", default=config_names, help="các file YAML trong configs/, ngăn bằng dấu phẩy")
    parser.add_argument("--set", nargs="*", default=[], help="ghi đè dạng a.b.c=giá_trị")
    return parser


def load_from_args(args: argparse.Namespace) -> Dict[str, Any]:
    config = load_config(args.configs.split(","), overrides=parse_overrides(args.set))
    setup_logging(config.get("log_level", "INFO"))
    return config


def describe(value: Any) -> Any:
    """Tóm tắt kết quả (shape tensor, trạng thái) để in JSON gọn."""
    if isinstance(value, torch.Tensor):
        return {"shape": list(value.shape), "dtype": str(value.dtype).replace("torch.", "")}
    if hasattr(value, "__dataclass_fields__"):
        return {name: describe(getattr(value, name)) for name in value.__dataclass_fields__}
    if isinstance(value, dict):
        return {key: describe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [describe(item) for item in value[:5]]
    return value


def print_json(payload: Any) -> None:
    print(json.dumps(describe(payload), indent=2, ensure_ascii=False, default=str))
