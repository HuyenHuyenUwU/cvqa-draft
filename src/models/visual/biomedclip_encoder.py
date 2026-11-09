"""BiomedCLIP vision encoder (open_clip implementation).

Checkpoint : microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224 (loaded through open_clip;
             a ``local-dir:`` copy is used unless ``allow_download: true`` is set explicitly).
Feature extraction (verified against open_clip's TimmModel source):
  tokens   = trunk.forward_features(x)                 # [B, prefix + N, Dv]  (prefix = CLS token)
  patch    = tokens[:, num_prefix_tokens:]             # patch_features        [B, N, Dv]
  global   = visual.head(trunk.forward_head(tokens))   # global_visual_embedding [B, Dg]  (CLIP space)
NOTE: Dv (ViT width, 768 for ViT-B) and Dg (CLIP embedding, 512) differ; fusion projects each
separately, so no dimension is hard-coded anywhere.
If the visual tower has no ``trunk`` (non-timm variant) patch_features is None and the
localisation branch reports ``not_available``.
"""
from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn

from ...common.exceptions import ModelLoadError
from ...common.logger import get_logger

log = get_logger("models.visual")


@dataclass
class VisualOutput:
    patch_features: Optional[torch.Tensor]       # [B, N, Dv] or None
    global_embedding: torch.Tensor               # [B, Dg]
    grid_size: Optional[Tuple[int, int]]         # (h, w) patch grid


def load_biomedclip(cfg: Dict[str, Any]):
    """Returns (open_clip_model, tokenizer). Never downloads unless allow_download is true."""
    import open_clip
    if cfg.get("local_dir"):
        name = f"local-dir:{cfg['local_dir']}"
    elif cfg.get("allow_download", False):
        name = cfg["model_id"]
        log.warning("Downloading BiomedCLIP weights from %s (allow_download=true)", name)
    else:
        raise ModelLoadError(
            "BiomedCLIP weights not available: set model.biomedclip.local_dir to a local copy, "
            "or set allow_download: true to approve the download explicitly.")
    model = open_clip.create_model_and_transforms(name)[0]
    tok = open_clip.get_tokenizer(name, context_length=int(cfg.get("context_length", 256)))
    return model, tok


class BiomedCLIPVisualEncoder(nn.Module):
    def __init__(self, visual: nn.Module, image_size: int = 224, freeze: bool = True,
                 unfreeze_last_n_blocks: int = 0):
        super().__init__()
        self.visual = visual
        self.trunk = getattr(visual, "trunk", None)
        self.image_size = int(image_size)
        self.set_trainable(freeze, unfreeze_last_n_blocks)
        self.patch_dim, self.global_dim, self.grid_size = self._probe()

    # ---- freezing ----
    def set_trainable(self, freeze: bool, unfreeze_last_n_blocks: int = 0) -> None:
        for p in self.visual.parameters():
            p.requires_grad = not freeze
        self._fully_frozen = freeze and unfreeze_last_n_blocks <= 0
        if freeze and unfreeze_last_n_blocks > 0 and self.trunk is not None:
            blocks = list(getattr(self.trunk, "blocks", []))[-unfreeze_last_n_blocks:]
            mods = blocks + [m for m in (getattr(self.trunk, "norm", None), getattr(self.visual, "head", None)) if m]
            for m in mods:
                for p in m.parameters():
                    p.requires_grad = True

    def train(self, mode: bool = True):
        super().train(mode and not self._fully_frozen)   # frozen tower stays in eval (no dropout drift)
        return self

    @torch.no_grad()
    def _probe(self):
        was = self.training
        self.eval()
        x = torch.zeros(1, 3, self.image_size, self.image_size)
        dev = next(self.visual.parameters()).device
        out = self._extract(x.to(dev))
        self.train(was)
        pd = out.patch_features.shape[-1] if out.patch_features is not None else None
        return pd, out.global_embedding.shape[-1], out.grid_size

    def _extract(self, images: torch.Tensor) -> VisualOutput:
        if self.trunk is not None and hasattr(self.trunk, "forward_features"):
            tokens = self.trunk.forward_features(images)
            prefix = getattr(self.trunk, "num_prefix_tokens", 1)
            g = self.visual.head(self.trunk.forward_head(tokens))
            patches = tokens[:, prefix:]
            ge = getattr(getattr(self.trunk, "patch_embed", None), "grid_size", None)
            grid = tuple(ge) if ge is not None else None
            if grid is None or grid[0] * grid[1] != patches.shape[1]:
                s = int(round(patches.shape[1] ** 0.5))
                grid = (s, s)
            return VisualOutput(patches, g, grid)
        return VisualOutput(None, self.visual(images), None)

    def forward(self, images: torch.Tensor) -> VisualOutput:
        ctx = torch.no_grad() if self._fully_frozen else nullcontext()
        with ctx:
            return self._extract(images)
