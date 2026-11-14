"""Ablation: BiomedCLIP (open_clip) làm vision encoder. CHƯA kiểm thử với trọng số thật.

Cùng giao diện đầu ra với ResNet50VisionAdapter:
  patch_features [B,N,Dv] (bỏ token prefix), global_feature [B,Dg] (không gian CLIP), grid_size.
Cần `local_dir` (thư mục chứa open_clip_config.json + weights) hoặc allow_download=true.
"""
from typing import Any, Dict

from ...commons.exceptions import ModelLoadError
from ..base import TorchModelAdapter
from ..registry import register_adapter


def extract_vit_features(visual: Any, images):
    """Trích đặc trưng từ tháp ảnh open_clip dạng TimmModel (đã kiểm chứng trên mã nguồn open_clip)."""
    trunk = visual.trunk
    tokens = trunk.forward_features(images)
    prefix = getattr(trunk, "num_prefix_tokens", 1)
    global_feature = visual.head(trunk.forward_head(tokens))
    patches = tokens[:, prefix:]
    grid = tuple(trunk.patch_embed.grid_size)
    return patches, global_feature, grid


@register_adapter("vision_encoder", "open_clip", "huggingface")
class BiomedCLIPVisionAdapter(TorchModelAdapter):
    def load(self) -> None:
        import open_clip
        if self.config.get("local_dir"):
            name = f"local-dir:{self.config['local_dir']}"
        elif self.config.get("allow_download", False):
            name = f"hf-hub:{self.config['name']}"
        else:
            raise ModelLoadError("BiomedCLIP: đặt local_dir hoặc allow_download: true.")
        clip_model = open_clip.create_model_and_transforms(name)[0]
        self.model = clip_model.visual
        self._finish_load()

    def predict(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        patches, global_feature, grid = extract_vit_features(self.model, inputs["images"].to(self.device))
        return {"patch_features": patches, "global_feature": global_feature, "grid_size": grid}
