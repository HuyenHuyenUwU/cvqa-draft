from __future__ import annotations

import math
from typing import Any, Dict

import torch


def build_optimizer(model: torch.nn.Module, cfg: Dict[str, Any]):
    """AdamW with two LR groups: pretrained encoders that were (partly) unfrozen vs. new modules."""
    lr, lr_enc = float(cfg.get("lr", 1e-4)), float(cfg.get("lr_encoder", 1e-5))
    wd = float(cfg.get("weight_decay", 0.01))
    enc, new = [], []
    for n, p in model.named_parameters():
        if p.requires_grad:
            (enc if n.startswith(("visual.", "text.")) else new).append(p)
    groups = [g for g in ({"params": new, "lr": lr}, {"params": enc, "lr": lr_enc}) if g["params"]]
    if not groups:
        raise ValueError("No trainable parameters")
    return torch.optim.AdamW(groups, weight_decay=wd)


def build_scheduler(optimizer, cfg: Dict[str, Any], total_steps: int):
    warm = int(cfg.get("warmup_steps", 0))

    def f(step):
        if step < warm:
            return (step + 1) / max(1, warm)
        prog = (step - warm) / max(1, total_steps - warm)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, prog)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, f)
