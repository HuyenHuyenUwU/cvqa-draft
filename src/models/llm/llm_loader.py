"""Local open-weight LLM loading (Hugging Face Transformers). No paid API, no hosted inference.

Default candidate: Qwen/Qwen2.5-1.5B-Instruct (Apache-2.0 according to its model card at the time of
writing - re-check the card before redistributing). Weights are loaded from ``local_dir``; a download only
happens when ``allow_download: true`` is set explicitly. 4-bit loading needs the optional bitsandbytes package
and a CUDA GPU; ``device_map`` needs ``accelerate``.
"""
from __future__ import annotations

from typing import Any, Dict

import torch

from ...common.exceptions import ModelLoadError
from ...common.logger import get_logger

log = get_logger("models.llm")
_DTYPES = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}


def load_llm(cfg: Dict[str, Any]):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    path = cfg.get("local_dir") or (cfg["model_id"] if cfg.get("allow_download", False) else None)
    if path is None:
        raise ModelLoadError("LLM weights not available: set model.llm.local_dir, or allow_download: true "
                             "to approve downloading %s." % cfg.get("model_id"))
    kw: Dict[str, Any] = {"dtype": _DTYPES[cfg.get("dtype", "float32")]}
    if cfg.get("device_map"):
        kw["device_map"] = cfg["device_map"]
    if cfg.get("load_in_4bit"):
        try:
            from transformers import BitsAndBytesConfig
            kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
        except Exception as e:  # pragma: no cover
            raise ModelLoadError(f"4-bit loading requested but unavailable: {e}") from e
    llm = AutoModelForCausalLM.from_pretrained(path, **kw)
    tok = AutoTokenizer.from_pretrained(path)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return llm, tok


def freeze_llm(llm: torch.nn.Module) -> None:
    for p in llm.parameters():
        p.requires_grad = False
    llm.eval()
