import pytest
import torch

from bone_cvqa.external_models import build_adapter, list_adapters
from bone_cvqa.external_models.base import BaseModelAdapter


def test_registry_lists_roles_and_providers():
    registry = list_adapters()
    text = str(registry)
    for provider in ("torchvision", "huggingface", "tiny_random"):
        assert provider in text


def test_vision_adapter_interface_load_predict_unload(tiny_cfg):
    adapter = build_adapter("vision_encoder", tiny_cfg["external_models"]["vision_encoder"])
    assert isinstance(adapter, BaseModelAdapter)
    adapter.load()
    out = adapter.predict({"images": torch.rand(2, 3, 64, 64)})
    assert out["patch_features"].dim() == 3 and out["global_feature"].shape[0] == 2
    adapter.unload()


def test_unknown_provider_raises(tiny_cfg):
    with pytest.raises(Exception):
        build_adapter("vision_encoder", {"provider": "does_not_exist"})


def test_api_placeholder_returns_explicit_status():
    adapter = build_adapter("medical_llm", {"provider": "api", "name": "x", "device": "cpu"})
    adapter.load()
    assert adapter.predict({"prompt": "hi"}).get("status") == "not_implemented"
