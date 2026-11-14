"""Vision encoder cơ bản nhất: ResNet50 (torchvision).  provider=torchvision | tiny_random (cùng lớp, trọng số ngẫu nhiên).

predict({"images": [B,3,H,W]}) -> {
    "patch_features": [B, N, 2048]   bản đồ đặc trưng layer4 được trải phẳng (N = h*w),
    "global_feature": [B, 2048]      average pooling,
    "grid_size": (h, w),
}
Trọng số: `weights_path` (state_dict local) hoặc `pretrained: true` + `allow_download: true`. Mặc định: ngẫu nhiên.
"""
from typing import Any, Dict

import torch
import torchvision
from torch import nn

from ...commons.exceptions import ModelLoadError
from ..base import TorchModelAdapter
from ..registry import register_adapter


@register_adapter("vision_encoder", "torchvision", "tiny_random")
class ResNet50VisionAdapter(TorchModelAdapter):
    output_dim = 2048

    def load(self) -> None:
        weights = None
        if self.config.get("pretrained"):
            if not self.config.get("allow_download", False):
                raise ModelLoadError("pretrained=true cần allow_download=true (hoặc dùng weights_path local).")
            weights = torchvision.models.ResNet50_Weights.DEFAULT
        resnet = torchvision.models.resnet50(weights=weights)
        if self.config.get("weights_path"):
            resnet.load_state_dict(torch.load(self.config["weights_path"], map_location="cpu"))
        self.model = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
                                   resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4)
        self._finish_load()

    def predict(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        feature_map = self.model(inputs["images"].to(self.device))        # [B, 2048, h, w]
        batch, channels, height, width = feature_map.shape
        return {"patch_features": feature_map.flatten(2).transpose(1, 2),
                "global_feature": feature_map.mean(dim=(2, 3)),
                "grid_size": (height, width)}
