from bone_cvqa.data.loaders import make_dataloader
from bone_cvqa.commons.config import deep_merge


def test_fracatlas_loader(tiny_cfg, fake_fracatlas):
    config = deep_merge(tiny_cfg, {"data": {"datasets": {"fracatlas": {"root": str(fake_fracatlas)}}, "split": {"ratios": [0.6, 0.2, 0.2], "seed": 3}}})
    batch = next(iter(make_dataloader(config, "train", False)))
    assert batch["image"].shape[1:] == (3, 64, 64)
    assert "has_box" in batch and "has_answer" in batch
    assert batch["metadata"][0].get("qa_source", "template_weak_supervision") == "template_weak_supervision"
