import json
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def make_fake_fracatlas(root: Path, n_frac=6, n_non=6, size=(96, 128)):
    """Synthetic data in the ASSUMED FracAtlas layout (no real patient data)."""
    rng = np.random.default_rng(0)
    imgs, anns = [], []
    iid = 0
    for sub, n in (("Fractured", n_frac), ("Non_fractured", n_non)):
        d = root / "images" / sub
        d.mkdir(parents=True, exist_ok=True)
        for k in range(n):
            arr = (rng.random((size[1], size[0])) * 200 + 20).astype(np.uint8)
            name = f"IMG{iid:04d}.jpg"
            Image.fromarray(arr).save(d / name)
            imgs.append({"id": iid, "file_name": name, "width": size[0], "height": size[1]})
            if sub == "Fractured":
                anns.append({"id": iid, "image_id": iid, "bbox": [10, 20, 40, 30]})
            iid += 1
    ann_dir = root / "Annotations" / "COCO JSON"
    ann_dir.mkdir(parents=True, exist_ok=True)
    (ann_dir / "COCO_fracture_masks.json").write_text(json.dumps({"images": imgs, "annotations": anns}))
    return root


@pytest.fixture()
def fake_fracatlas(tmp_path):
    return make_fake_fracatlas(tmp_path / "fracatlas")


@pytest.fixture()
def data_cfg(fake_fracatlas):
    return {"name": "fracatlas", "root": str(fake_fracatlas),
            "split": {"ratios": [0.5, 0.25, 0.25], "seed": 1}, "qa": {"source": "weak_template"}}


def tiny_cfg(**over):
    cfg = {
        "model_source": "tiny_random",
        "preprocess": {"image_size": 64},
        "biomedclip": {"freeze": True}, "text_encoder": {"max_length": 96, "freeze": True},
        "llm": {"freeze": True, "max_prompt_tokens": 96, "max_answer_tokens": 16},
        "fusion": {"d_model": 32, "n_layers": 1, "n_heads": 2, "n_soft_tokens": 4, "adapter_heads": 2, "dropout": 0.0},
        "localization": {"enabled": True, "hidden": 32},
        "generation": {"max_new_tokens": 6, "do_sample": False},
    }
    cfg.update(over)
    return cfg


@pytest.fixture(scope="session")
def tiny_model_cfg():
    return tiny_cfg()


@pytest.fixture()
def tiny_model(tiny_model_cfg):
    from src.models.model_io import build_cvqa_model
    return build_cvqa_model(tiny_model_cfg)
