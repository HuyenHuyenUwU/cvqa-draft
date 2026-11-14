"""Backpropagation & Optimizer (Offline): AdamW trên các tham số requires_grad + lịch cosine có warm-up."""
import math
from typing import Any, Dict

import torch


def build_optimizer(system: torch.nn.Module, train_config: Dict[str, Any]) -> torch.optim.Optimizer:
    parameters = [p for p in system.parameters() if p.requires_grad]
    if not parameters:
        raise ValueError("Không có tham số nào để train (tất cả đang đóng băng).")
    return torch.optim.AdamW(parameters, lr=float(train_config["lr"]), weight_decay=float(train_config.get("weight_decay", 0.01)))


def build_scheduler(optimizer: torch.optim.Optimizer, total_steps: int, warmup_steps: int = 0):
    def factor(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)
