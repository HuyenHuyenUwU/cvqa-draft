"""Offline trainer: joint loss -> backprop -> optimizer (the 'Multi-task Learning' block).

Features: AdamW (2 LR groups), warm-up + cosine schedule, gradient accumulation, gradient clipping,
mixed precision (CUDA only), validation each epoch, early stopping, best/last checkpoints, reproducible seed.
The trainer states honestly which loss terms are active; with answer-only supervision it warns that the
multi-task objective is NOT fully trained.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, Optional

import torch

from ..common.logger import get_logger
from ..models.cvqa_inputs import build_model_inputs
from .checkpoint import save_training_checkpoint
from .losses import JointLoss
from .optimizer import build_optimizer, build_scheduler

log = get_logger("offline.trainer")


class Trainer:
    def __init__(self, model, cfg: Dict[str, Any], train_loader, val_loader, qp, ontology, device,
                 split_info: Dict[str, Any], ckpt_dir: str):
        self.model, self.cfg, self.device = model.to(device), cfg, device
        self.tl, self.vl, self.qp, self.onto = train_loader, val_loader, qp, ontology
        self.tc = cfg.get("train", {})
        self.split_info, self.ckpt_dir = split_info, Path(ckpt_dir)
        lc = cfg.get("losses", {})
        self.loss_fn = JointLoss(lc.get("weights"), bool(lc.get("heatmap_bce", True)))
        self.accum = max(1, int(self.tc.get("grad_accum_steps", 1)))
        steps = max(1, math.ceil(len(train_loader) / self.accum) * int(self.tc.get("epochs", 1)))
        self.opt = build_optimizer(model, self.tc)
        self.sched = build_scheduler(self.opt, self.tc, steps)
        self.use_amp = bool(self.tc.get("amp", False)) and device.type == "cuda"
        self.amp_dtype = torch.bfloat16 if (self.use_amp and torch.cuda.is_bf16_supported()) else torch.float16
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp and self.amp_dtype == torch.float16)
        self.active_seen, self.skipped_seen, self.loc_steps = set(), {}, 0
        self.best, self.bad_epochs, self.history = None, 0, []

    def _forward(self, batch, train: bool):
        inp = build_model_inputs(self.model, self.qp, self.onto, batch["question"], batch["conversation_history"])
        answers = [a if bool(h) else None for a, h in zip(batch["answer"], batch["has_answer"])]
        with torch.autocast(self.device.type, dtype=self.amp_dtype, enabled=self.use_amp):
            out = self.model(batch["image"].to(self.device), inp["conv_texts"], inp["type_ids"], inp["prompts"],
                             answers=answers if any(a is not None for a in answers) else None,
                             gt_box=batch["gt_box"], has_box=batch["has_box"],
                             use_gt_roi=bool(self.tc.get("use_gt_roi", True)), knowledge=inp["knowledge"])
            return self.loss_fn(out, batch)

    def _train_epoch(self, epoch: int) -> float:
        self.model.train()
        self.opt.zero_grad(set_to_none=True)
        run, n, max_b = 0.0, 0, self.tc.get("max_train_batches")
        for i, batch in enumerate(self.tl):
            if max_b is not None and i >= max_b:
                break
            res = self._forward(batch, True)
            self.skipped_seen.update(res["skipped"])
            if res["total"] is None:
                log.warning("batch %d: no active loss term -> skipped (%s)", i, res["skipped"])
                continue
            self.active_seen.update(res["active"])
            self.loc_steps += int("loc" in res["active"])
            self.scaler.scale(res["total"] / self.accum).backward()
            if (i + 1) % self.accum == 0 or i + 1 == len(self.tl):
                self.scaler.unscale_(self.opt)
                clip = float(self.tc.get("grad_clip", 1.0))
                if clip > 0:
                    torch.nn.utils.clip_grad_norm_([p for p in self.model.parameters() if p.requires_grad], clip)
                self.scaler.step(self.opt); self.scaler.update()
                self.opt.zero_grad(set_to_none=True); self.sched.step()
            run += float(res["total"].detach()); n += 1
            if (i + 1) % int(self.tc.get("log_every", 20)) == 0:
                log.info("epoch %d it %d loss %.4f %s", epoch, i + 1, float(res["total"].detach()), res["components"])
        return run / max(1, n)

    @torch.no_grad()
    def _validate(self) -> float:
        self.model.eval()
        tot, n = 0.0, 0
        for i, batch in enumerate(self.vl):
            if self.tc.get("max_val_batches") is not None and i >= self.tc["max_val_batches"]:
                break
            res = self._forward(batch, False)
            if res["total"] is not None:
                tot += float(res["total"]); n += 1
        return tot / n if n else float("nan")

    def _save(self, name: str, epoch: int):
        save_training_checkpoint(self.ckpt_dir / name, self.model, self.cfg, self.opt, self.sched, epoch, self.best,
                                 self.split_info, sorted(self.active_seen), self.skipped_seen)

    def fit(self) -> Dict[str, Any]:
        epochs, patience = int(self.tc.get("epochs", 1)), int(self.tc.get("early_stopping_patience", 0))
        for ep in range(1, epochs + 1):
            tr = self._train_epoch(ep)
            self.model.localizer_trained = self.loc_steps >= int(self.tc.get("min_loc_steps", 50))
            va = self._validate()
            self.history.append({"epoch": ep, "train_loss": tr, "val_loss": va, "active_losses": sorted(self.active_seen)})
            log.info("epoch %d | train %.4f | val %.4f | active=%s | localizer_trained=%s", ep, tr, va,
                     sorted(self.active_seen), self.model.localizer_trained)
            improved = not math.isnan(va) and (self.best is None or va < self.best)
            if improved:
                self.best, self.bad_epochs = va, 0
                self._save("best.pt", ep)
            else:
                self.bad_epochs += 1
            self._save("last.pt", ep)
            if patience and self.bad_epochs >= patience:
                log.info("early stopping at epoch %d", ep)
                break
        if self.active_seen != {"ans", "loc"} and self.active_seen != {"ans", "loc", "rat", "kg", "align"}:
            log.warning("Multi-task objective is NOT fully trained. Active: %s. Skipped: %s",
                        sorted(self.active_seen), self.skipped_seen)
        summary = {"history": self.history, "best_val_loss": self.best, "active_losses": sorted(self.active_seen),
                   "skipped_losses": self.skipped_seen, "localizer_trained": self.model.localizer_trained}
        (self.ckpt_dir / "train_summary.json").write_text(json.dumps(summary, indent=1))
        return summary
