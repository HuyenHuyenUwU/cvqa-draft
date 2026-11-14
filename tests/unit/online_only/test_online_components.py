from bone_cvqa.image_processing.image_quality.assessor import ImageQualityAssessor
from bone_cvqa.trustworthy_verification.final_output import validate_final_output
from bone_cvqa.uncertainty_estimation.estimator import ConfidenceScorer, UncertaintyEstimator


def test_quality_assessor_runs(xray_path):
    from PIL import Image
    import numpy as np
    gray = np.asarray(Image.open(xray_path).convert("L"), dtype=np.float32) / 255.0
    result = ImageQualityAssessor({}).assess(gray, gray.shape)
    assert "label" in result


def test_uncertainty_confidence_not_fabricated():
    assert ConfidenceScorer().score(UncertaintyEstimator().estimate()).get("score") is None
