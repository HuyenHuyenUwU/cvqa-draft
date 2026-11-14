"""Dữ liệu TỔNG HỢP theo layout FracAtlas giả định, chỉ để test/smoke test (không phải dữ liệu bệnh nhân)."""
import json
from pathlib import Path

import numpy as np
from PIL import Image


def make_fake_fracatlas(root: Path, n_fractured: int = 6, n_normal: int = 6, size=(96, 128)) -> Path:
    rng = np.random.default_rng(0)
    images, annotations, index = [], [], 0
    for sub, count in (("Fractured", n_fractured), ("Non_fractured", n_normal)):
        folder = Path(root) / "images" / sub
        folder.mkdir(parents=True, exist_ok=True)
        for _ in range(count):
            array = (rng.random((size[1], size[0])) * 200 + 20).astype(np.uint8)
            name = f"IMG{index:04d}.jpg"
            Image.fromarray(array).save(folder / name)
            images.append({"id": index, "file_name": name, "width": size[0], "height": size[1]})
            if sub == "Fractured":
                annotations.append({"id": index, "image_id": index, "bbox": [10, 20, 40, 30]})
            index += 1
    annotation_dir = Path(root) / "Annotations" / "COCO JSON"
    annotation_dir.mkdir(parents=True, exist_ok=True)
    (annotation_dir / "COCO_fracture_masks.json").write_text(json.dumps({"images": images, "annotations": annotations}))
    return Path(root)
