"""Vòng lặp huấn luyện OFFLINE: bật gradient (execution_context TRAIN) -> JointLoss -> backward -> optimizer.

Gradient accumulation, gradient clipping, AMP (CUDA), validation mỗi epoch, early stopping, lưu best/last.
Validation dùng chế độ INFER (không gradient) nhưng vẫn tính loss để chọn checkpoint tốt nhất.
"""
import json
import math
from pathlib import Path
from typing import Any, Dict, Optional

import torch

from ..commons.constants import ExecutionMode
from ..commons.execution_mode import execution_context
from ..commons.logger import get_logger
from ..system import BoneCVQASystem
from ..checkpoint_io import save_checkpoint
from .joint_loss import JointLoss, JointLossResult
from .optimizer import build_optimizer, build_scheduler

log = get_logger("training.trainer")


class Trainer:
    def __init__(self, system: BoneCVQASystem, config: Dict[str, Any], train_loader, val_loader, device: torch.device,
                 split_info: Optional[dict], checkpoint_dir: str) -> None:
        self.system, self.config, self.device = system, config, device
        self.train_loader, self.val_loader = train_loader, val_loader
        self.train_config = config["train"]
        self.accum = max(1, int(self.train_config.get("grad_accum_steps", 1)))
        self.loss_fn = JointLoss(config.get("loss_weights"))
        self.optimizer = build_optimizer(system, self.train_config)
        steps = max(1, math.ceil(len(train_loader) / self.accum) * int(self.train_config["epochs"]))
        self.scheduler = build_scheduler(self.optimizer, steps, int(self.train_config.get("warmup_steps", 0)))
        self.use_amp = bool(self.train_config.get("amp", False)) and device.type == "cuda"
        self.amp_dtype = torch.bfloat16 if self.use_amp and torch.cuda.is_bf16_supported() else torch.float16
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp and self.amp_dtype == torch.float16)
        self.split_info, self.checkpoint_dir = split_info, Path(checkpoint_dir)
        self.active_losses, self.skipped_losses = set(), {}
        self.best_val: Optional[float] = None
        self.history = []

    def _compute_loss(self, batch: Dict[str, Any]) -> JointLossResult:
        answers = [a if bool(has) else None for a, has in zip(batch["answer"], batch["has_answer"])]
        with torch.autocast(self.device.type, dtype=self.amp_dtype, enabled=self.use_amp):
            context = self.system.encode(batch["image"].to(self.device), batch["question"], batch["conversation_history"])
            logits, targets = self.system.answer_logits(context, answers)
            return self.loss_fn(logits, targets)

    def _train_epoch(self, epoch: int) -> float:
        total, count = 0.0, 0
        self.optimizer.zero_grad(set_to_none=True)
        limit = self.train_config.get("max_train_batches")
        with execution_context(self.system, ExecutionMode.TRAIN):          # BẬT gradient
            for step, batch in enumerate(self.train_loader):
                if limit is not None and step >= limit:
                    break
                result = self._compute_loss(batch)
                self.skipped_losses.update(result.skipped)
                if result.total is None:
                    log.warning("batch %d: không có term loss active -> bỏ qua", step)
                    continue
                self.active_losses.update(result.active)
                self.scaler.scale(result.total / self.accum).backward()
                if (step + 1) % self.accum == 0 or step + 1 == len(self.train_loader):
                    self.scaler.unscale_(self.optimizer)
                    clip = float(self.train_config.get("grad_clip", 1.0))
                    if clip > 0:
                        torch.nn.utils.clip_grad_norm_([p for p in self.system.parameters() if p.requires_grad], clip)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad(set_to_none=True)
                    self.scheduler.step()
                total += float(result.total.detach()); count += 1
                if (step + 1) % int(self.train_config.get("log_every", 20)) == 0:
                    log.info("epoch %d step %d loss %.4f %s", epoch, step + 1, float(result.total.detach()), result.components)
        return total / max(1, count)

    def _validate(self) -> float:
        total, count = 0.0, 0
        limit = self.train_config.get("max_val_batches")
        with execution_context(self.system, ExecutionMode.INFER):          # TẮT gradient
            for step, batch in enumerate(self.val_loader):
                if limit is not None and step >= limit:
                    break
                result = self._compute_loss(batch)
                if result.total is not None:
                    total += float(result.total); count += 1
        return total / count if count else float("nan")

    def _save(self, name: str, epoch: int) -> None:
        save_checkpoint(self.checkpoint_dir / name, self.system, self.config, self.optimizer, self.scheduler, epoch,
                        self.best_val, self.split_info, sorted(self.active_losses), self.skipped_losses)

    def fit(self) -> Dict[str, Any]:
        patience, bad_epochs = int(self.train_config.get("early_stopping_patience", 0)), 0
        for epoch in range(1, int(self.train_config["epochs"]) + 1):
            train_loss, val_loss = self._train_epoch(epoch), self._validate()
            self.history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
            log.info("epoch %d | train %.4f | val %.4f | active=%s", epoch, train_loss, val_loss, sorted(self.active_losses))
            if not math.isnan(val_loss) and (self.best_val is None or val_loss < self.best_val):
                self.best_val, bad_epochs = val_loss, 0
                self._save("best.pt", epoch)
            else:
                bad_epochs += 1
            self._save("last.pt", epoch)
            if patience and bad_epochs >= patience:
                log.info("early stopping tại epoch %d", epoch)
                break
        summary = {"history": self.history, "best_val_loss": self.best_val, "active_losses": sorted(self.active_losses),
                   "skipped_losses": self.skipped_losses}
        (self.checkpoint_dir / "train_summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False))
        return summary
