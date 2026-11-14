"""Checkpoint: chỉ lưu tham số requires_grad (projection, fusion, ... và phần model ngoài nếu được mở khóa).
Model ngoài đóng băng được dựng lại từ cấu hình (external_models). Có cả optimizer/scheduler/epoch/split/loss active."""
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

from .commons.config import deep_merge
from .commons.exceptions import CheckpointError
from .system import BoneCVQASystem, build_system

FORMAT_VERSION = 2
MODEL_KEYS = ["d_model", "external_models", "image_processing", "question_processing", "ontology_processing",
              "cross_modal_fusion", "decoder"]


def trainable_state(system: torch.nn.Module) -> Dict[str, torch.Tensor]:
    names = {n for n, p in system.named_parameters() if p.requires_grad}
    return {k: v.detach().cpu() for k, v in system.state_dict().items() if k in names}


def save_checkpoint(path, system: BoneCVQASystem, config: Dict[str, Any], optimizer=None, scheduler=None, epoch: int = 0,
                    best_metric: Optional[float] = None, split_info: Optional[dict] = None,
                    active_losses: Optional[List[str]] = None, skipped_losses: Optional[dict] = None) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"format_version": FORMAT_VERSION, "model_state": trainable_state(system),
                "model_config": {k: config[k] for k in MODEL_KEYS if k in config},
                "optimizer_state": optimizer.state_dict() if optimizer else None,
                "scheduler_state": scheduler.state_dict() if scheduler else None,
                "epoch": epoch, "best_metric": best_metric, "split_info": split_info or {},
                "train_config": config.get("train", {}), "active_losses": active_losses or [],
                "skipped_losses": skipped_losses or {}}, path)


def load_system_from_checkpoint(path, config_override: Optional[Dict[str, Any]] = None) -> Tuple[BoneCVQASystem, Dict[str, Any]]:
    """Dựng lại hệ thống từ checkpoint. `config_override` để sửa đường dẫn model ngoài trên máy hiện tại."""
    if not Path(path).is_file():
        raise CheckpointError(f"Không tìm thấy checkpoint: {path}")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("format_version") != FORMAT_VERSION:
        raise CheckpointError(f"Định dạng checkpoint không hỗ trợ: {checkpoint.get('format_version')}")
    config = deep_merge(checkpoint["model_config"], config_override or {})
    system = build_system(config)
    missing, unexpected = system.load_state_dict(checkpoint["model_state"], strict=False)
    trainable = {n for n, p in system.named_parameters() if p.requires_grad}
    missing_trainable = [k for k in missing if k in trainable]
    if unexpected or missing_trainable:
        raise CheckpointError(f"State không khớp. unexpected={unexpected[:5]} missing_trainable={missing_trainable[:5]}")
    return system, checkpoint
