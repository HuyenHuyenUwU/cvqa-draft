"""Training checkpoints = inference checkpoint (models.model_io) + training state.
Saved: model weights (trainable), optimizer, scheduler, epoch, best metric, model config, dataset split
info, training config, active loss components."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

from ..common.exceptions import CheckpointError
from ..models.model_io import save_model_checkpoint


def save_training_checkpoint(path, model, cfg: Dict[str, Any], optimizer, scheduler, epoch: int,
                             best_metric: Optional[float], split_info: Dict[str, Any],
                             active_losses: List[str], skipped_losses: Dict[str, str]) -> None:
    extra = {"optimizer_state": optimizer.state_dict() if optimizer else None,
             "scheduler_state": scheduler.state_dict() if scheduler else None,
             "epoch": epoch, "best_metric": best_metric, "split_info": split_info,
             "train_config": cfg.get("train", {}), "active_losses": active_losses, "skipped_losses": skipped_losses}
    save_model_checkpoint(path, model, cfg, extra)


def resume_training_state(path, optimizer, scheduler, map_location="cpu") -> Dict[str, Any]:
    if not Path(path).is_file():
        raise CheckpointError(f"Checkpoint not found: {path}")
    ck = torch.load(path, map_location=map_location, weights_only=False)
    if ck.get("optimizer_state") and optimizer:
        optimizer.load_state_dict(ck["optimizer_state"])
    if ck.get("scheduler_state") and scheduler:
        scheduler.load_state_dict(ck["scheduler_state"])
    return {"epoch": ck["epoch"], "best_metric": ck["best_metric"], "model_state": ck["model_state"]}
