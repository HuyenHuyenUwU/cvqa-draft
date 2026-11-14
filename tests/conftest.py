import json
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))


def make_fake_fracatlas(root: Path, n_fractured=6, n_normal=6, size=(96, 128)):
    """Dữ liệu tổng hợp theo layout FracAtlas GIẢ ĐỊNH (không phải dữ liệu bệnh nhân thật)."""
    rng = np.random.default_rng(0)
    images, annotations, index = [], [], 0
    for sub, count in (("Fractured", n_fractured), ("Non_fractured", n_normal)):
        folder = root / "images" / sub
        folder.mkdir(parents=True, exist_ok=True)
        for _ in range(count):
            array = (rng.random((size[1], size[0])) * 200 + 20).astype(np.uint8)
            name = f"IMG{index:04d}.jpg"
            Image.fromarray(array).save(folder / name)
            images.append({"id": index, "file_name": name, "width": size[0], "height": size[1]})
            if sub == "Fractured":
                annotations.append({"id": index, "image_id": index, "bbox": [10, 20, 40, 30]})
            index += 1
    annotation_dir = root / "Annotations" / "COCO JSON"
    annotation_dir.mkdir(parents=True, exist_ok=True)
    (annotation_dir / "COCO_fracture_masks.json").write_text(json.dumps({"images": images, "annotations": annotations}))
    return root


@pytest.fixture()
def fake_fracatlas(tmp_path):
    return make_fake_fracatlas(tmp_path / "fracatlas")


@pytest.fixture()
def xray_path(tmp_path):
    rng = np.random.default_rng(1)
    base = np.linspace(30, 220, 160)[None, :].repeat(160, 0) + rng.normal(0, 25, (160, 160))
    path = tmp_path / "xray.png"
    Image.fromarray(np.clip(base, 0, 255).astype(np.uint8)).save(path)
    return str(path)


@pytest.fixture(scope="session")
def tiny_cfg():
    from tests.helpers import tiny_config  # noqa
    return tiny_config()


@pytest.fixture(scope="session")
def tiny_system(tiny_cfg):
    from bone_cvqa.system import build_system
    return build_system(tiny_cfg)
