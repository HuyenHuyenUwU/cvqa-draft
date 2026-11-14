"""Model ngoài: import các adapter để chúng tự đăng ký vào registry."""
from .base import BaseModelAdapter, TorchModelAdapter
from .llm import api_llm, hf_causal_lm, llava_med, medgemma  # noqa: F401
from .registry import build_adapter, list_adapters, register_adapter
from .text import bert, pubmedbert  # noqa: F401
from .vision import biomedclip, resnet50  # noqa: F401

__all__ = ["BaseModelAdapter", "TorchModelAdapter", "build_adapter", "register_adapter", "list_adapters"]
