import logging

import torch

from bone_cvqa.image_processing.dicom_preprocessing import DicomPreprocessor
from bone_cvqa.image_processing.pipeline import ImageProcessingPipeline


def test_preprocess_shape(tiny_cfg, xray_path):
    pre = DicomPreprocessor(tiny_cfg["image_processing"]["preprocess"])(xray_path)
    assert pre.tensor.shape == (3, 64, 64)
    assert pre.original_size_hw == (160, 160)


def test_forward_and_placeholders(tiny_cfg, xray_path, caplog):
    pipeline = ImageProcessingPipeline(tiny_cfg)
    image = pipeline.preprocess(xray_path).tensor.unsqueeze(0)
    with caplog.at_level(logging.WARNING):
        out = pipeline(image)
    assert out.global_embedding.embedding.shape == (1, 32)
    assert out.localization.status == "not_implemented"
    assert out.roi.status == "skipped"
    assert "[TODO] Lesion Localization is a placeholder" in caplog.text


def test_stops_at_stage(tiny_cfg, xray_path):
    pipeline = ImageProcessingPipeline(tiny_cfg)
    image = pipeline.preprocess(xray_path).tensor.unsqueeze(0)
    assert "localization" not in pipeline.run_until(image, "vision_encoder")


def test_standalone_cli(tiny_cfg, xray_path, capsys):
    from bone_cvqa.image_processing import run
    assert run.main is not None
