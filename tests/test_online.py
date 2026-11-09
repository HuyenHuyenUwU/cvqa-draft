import json
import sys

import numpy as np
import pytest
import torch
from PIL import Image

from src.data.image_preprocessing import ImagePreprocessor
from src.models.model_io import build_cvqa_model, save_model_checkpoint
from src.online.image_quality import ImageQualityAssessor
from src.online.inference_engine import InferenceEngine
from src.online.output_formatter import validate_output_schema
from src.online.pipeline import OnlinePipeline
from src.online.warning_generator import WarningGenerator
from src.question_processing.question_scope_detector import RuleBasedQuestionScopeDetector
from tests.conftest import tiny_cfg


@pytest.fixture()
def xray(tmp_path):
    rng = np.random.default_rng(1)
    base = np.linspace(30, 220, 160)[None, :].repeat(160, 0) + rng.normal(0, 25, (160, 160))
    p = tmp_path / "x.png"
    Image.fromarray(np.clip(base, 0, 255).astype(np.uint8)).save(p)
    return str(p)


@pytest.fixture()
def pipeline():
    cfg = tiny_cfg()
    return OnlinePipeline(build_cvqa_model(cfg), ImagePreprocessor(cfg["preprocess"]), torch.device("cpu"),
                          {"generation": {"max_new_tokens": 4}})


def test_online_does_not_import_offline_trainer(tmp_path):
    cfg = tiny_cfg()
    save_model_checkpoint(tmp_path / "m.pt", build_cvqa_model(cfg), cfg)
    for m in [m for m in sys.modules if m.startswith("src.offline")]:
        del sys.modules[m]
    InferenceEngine.from_checkpoint(str(tmp_path / "m.pt"), {"device": "cpu"})
    assert not [m for m in sys.modules if m.startswith("src.offline")]


def test_online_from_checkpoint_end_to_end(tmp_path, xray):
    cfg = tiny_cfg()
    save_model_checkpoint(tmp_path / "m.pt", build_cvqa_model(cfg), cfg)
    eng = InferenceEngine.from_checkpoint(str(tmp_path / "m.pt"), {"device": "cpu", "generation": {"max_new_tokens": 4}})
    out = eng.infer(xray, "Is there a fracture in this image?", [], "s1")
    assert out["pipeline_status"] == "completed" and isinstance(out["answer"], str)
    assert validate_output_schema(out) == []
    json.dumps(out)                                              # must be JSON-serialisable


def test_missing_ontology_uncertainty_and_localizer_are_explicit(pipeline, xray):
    out = pipeline.run(xray, "Is there a fracture?", [], "s")
    assert out["evidence_graph"] == {"nodes": [], "edges": [], "status": "not_implemented"}
    assert out["visual_knowledge_explanation"] == {"text": None, "status": "not_implemented"}
    assert out["uncertainty"] == {"score": None, "status": "not_implemented"}
    assert out["confidence"]["score"] is None and out["confidence"]["status"] == "not_implemented"
    assert out["localization"]["status"] == "not_available" and out["localization"]["value"] is None  # untrained
    assert out["debug"]["fusion_status"]["knowledge"] == "not_implemented"
    codes = [w["code"] for w in out["warnings"]]
    assert "uncertainty_unavailable" in codes and "low_confidence" not in codes


def test_empty_and_nonempty_history(pipeline, xray):
    a = pipeline.run(xray, "Where is the lesion?", [], "a")
    b = pipeline.run(xray, "Where is it?", [{"question": "Is there a fracture?", "answer": "Yes."}], "b")
    assert a["pipeline_status"] == b["pipeline_status"] == "completed"


def test_out_of_scope_skips_generation(pipeline, xray):
    out = pipeline.run(xray, "What is the capital of France?", [], "s")
    assert out["question_scope"]["label"] == "out_of_scope" and out["answer"] is None
    assert out["pipeline_status"] == "completed_out_of_scope"
    assert "out_of_scope_question" in [w["code"] for w in out["warnings"]]
    assert "limitation" in out["question_scope"]["reason"].lower()      # reason explains the fallback policy
    assert validate_output_schema(out) == []


def test_invalid_image_handling(pipeline, tmp_path):
    bad = tmp_path / "bad.png"
    bad.write_bytes(b"garbage")
    out = pipeline.run(str(bad), "Is there a fracture?", [], "s")
    assert out["pipeline_status"] == "failed_invalid_image" and out["answer"] is None
    assert out["image_quality"]["label"] == "invalid"
    assert "invalid_image" in [w["code"] for w in out["warnings"]]
    assert validate_output_schema(out) == []
    out2 = pipeline.run(str(tmp_path / "nope.png"), "Is there a fracture?", [], "s")
    assert out2["pipeline_status"] == "failed_invalid_image"


def test_constant_image_flagged_but_pipeline_follows_diagram(pipeline, tmp_path):
    p = tmp_path / "flat.png"
    Image.fromarray(np.full((160, 160), 100, np.uint8)).save(p)
    out = pipeline.run(str(p), "Is there a fracture?", [], "s")
    assert out["image_quality"]["label"] == "invalid" and "invalid_image" in [w["code"] for w in out["warnings"]]
    assert out["pipeline_status"] == "completed"                 # default: quality only feeds the Warning Generator
    pipeline.block_on_invalid_quality = True
    assert pipeline.run(str(p), "Is there a fracture?", [], "s")["pipeline_status"] == "blocked_invalid_image_quality"


def test_image_quality_heuristics():
    iqa = ImageQualityAssessor()
    rng = np.random.default_rng(0)
    good = np.clip(rng.random((128, 128)) * 0.8 + 0.1, 0, 1).astype(np.float32)
    assert iqa.assess(good, (256, 256))["label"] == "good"
    low = (0.5 + 0.005 * rng.random((128, 128))).astype(np.float32)
    assert iqa.assess(low, (256, 256))["label"] == "invalid"
    assert iqa.assess(good * 0.1 + 0.45, (256, 256))["label"] == "low"             # low contrast
    assert iqa.assess(good, (64, 64))["warnings"] == ["low_resolution"]            # wrong dimensions
    nan = good.copy(); nan[0, 0] = np.nan
    assert iqa.assess(nan)["label"] == "invalid" and iqa.assess(None)["label"] == "invalid"


def test_scope_detector_fallbacks():
    d = RuleBasedQuestionScopeDetector()
    assert d.detect("Is there a fracture in the femur?")["label"] == "in_scope"
    assert d.detect("Who won the football match?")["label"] == "out_of_scope"
    assert d.detect("tell me about bananas", {"entities": ["x"]})["label"] == "in_scope"
    assert d.detect("Who won the football match?")["label"] == "out_of_scope"      # 'foot' must not match 'football'
    assert d.detect("Is it osteosarcoma?")["label"] == "in_scope"                  # stem match
    assert d.detect("Where is it?")["label"] == "out_of_scope"                     # no context -> fallback policy
    assert d.detect("Where is it?", history=[("Is there a fracture?", "Yes")])["label"] == "in_scope"
    assert RuleBasedQuestionScopeDetector(unmatched_policy="in_scope").detect("hello")["label"] == "in_scope"


def test_warning_generator_never_invents_low_confidence():
    w = WarningGenerator(0.5)
    iq, sc = {"label": "good", "warnings": []}, {"label": "in_scope"}
    codes = lambda c: [x["code"] for x in w.generate(iq, sc, {"status": "not_implemented"}, c)]
    assert "low_confidence" not in codes({"score": None, "status": "not_implemented"})
    assert "low_confidence" not in codes({"score": 0.1, "status": "not_implemented"})      # score w/o valid method
    assert "low_confidence" in codes({"score": 0.1, "status": "available"})
    assert w.generate(iq, sc, {"status": "available", "score": 0.2}, {"score": 0.9, "status": "available"}) == []
