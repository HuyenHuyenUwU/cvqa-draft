"""Logging dùng chung. Placeholder gọi `log_placeholder` để log luôn nói rõ module CHƯA làm việc thật."""
import logging
import sys
from pathlib import Path
from typing import Optional

LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
ROOT_LOGGER_NAME = "bone_cvqa"


def setup_logging(level: str = "INFO", log_file: Optional[str] = None) -> None:
    root = logging.getLogger(ROOT_LOGGER_NAME)
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    if not any(isinstance(h, logging.StreamHandler) for h in root.handlers):
        stream = logging.StreamHandler(sys.stdout)
        stream.setFormatter(logging.Formatter(LOG_FORMAT))
        root.addHandler(stream)
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter(LOG_FORMAT))
        root.addHandler(file_handler)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(f"{ROOT_LOGGER_NAME}.{name}")


def placeholder_message(module_name: str) -> str:
    return (f"[TODO] {module_name} is a placeholder. "
            f"It will be implemented in a later development stage.")
