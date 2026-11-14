from bone_cvqa.question_processing.pipeline import QuestionProcessingPipeline
from bone_cvqa.question_processing.scope_detector.detector import QuestionScopeDetector


def test_forward_shapes(tiny_cfg):
    pipeline = QuestionProcessingPipeline(tiny_cfg)
    out = pipeline(["Is there a fracture?", "Where is it?"], [[], []])
    assert out.features.token_features.shape[0] == 2 and out.features.token_features.shape[-1] == tiny_cfg["d_model"]
    assert out.scope is None            # Offline: không có scope detector


def test_scope_detector_online_only(tiny_cfg):
    pipeline = QuestionProcessingPipeline(tiny_cfg)
    out = pipeline(["Is there a fracture in the femur?", "What is the capital of France?"], [[], []], QuestionScopeDetector({}))
    assert [s["label"] for s in out.scope] == ["in_scope", "out_of_scope"]
