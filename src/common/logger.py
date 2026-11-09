"""Logging helpers. Placeholder modules must call `log_placeholder` so that the
log always states that a module has NOT performed its intended function."""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional

_FMT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"


def setup_logging(level: str = "INFO", log_file: Optional[str] = None) -> None:
    root = logging.getLogger("cvqa")
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    if not root.handlers:
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(logging.Formatter(_FMT))
        root.addHandler(sh)
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file)
        fh.setFormatter(logging.Formatter(_FMT))
        root.addHandler(fh)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(f"cvqa.{name}")


def log_placeholder(logger: logging.Logger, module_name: str) -> str:
    msg = (f"[TODO] {module_name} is a placeholder. "
           f"It will be implemented in a later development stage.")
    logger.warning(msg)
    return msg
