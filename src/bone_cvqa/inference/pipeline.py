"""Pipeline ONLINE (một mẫu): tắt gradient, không loss, không cập nhật tham số.

    DICOM Preprocessing -> Image Quality Assessment (chỉ Online)
    -> [system.encode: Question Processing + Scope Detector (chỉ Online) -> Image Processing -> Ontology -> Fusion]
    -> Answer Generation -> Uncertainty/Confidence (placeholder, chỉ Online) -> Warning Generator -> Final Output
Dùng CHÍNH `BoneCVQASystem` (cùng module với Offline). Không import bone_cvqa.training.
"""
from typing import Any, Dict, List, Optional

import torch

from ..commons.constants import ExecutionMode, Status
from ..commons.exceptions import InvalidImageError
from ..commons.execution_mode import execution_context
from ..commons.logger import get_logger
from ..image_processing.dicom_preprocessing import DicomPreprocessor
from ..image_processing.image_quality.assessor import ImageQualityAssessor
from ..question_processing.scope_detector.detector import QuestionScopeDetector
from ..system import BoneCVQASystem
from ..trustworthy_verification.pipeline import TrustworthyVerification
from ..uncertainty_estimation.estimator import ConfidenceScorer, UncertaintyEstimator

log = get_logger("inference.pipeline")


class OnlinePipeline:
    def __init__(self, system: BoneCVQASystem, config: Dict[str, Any]) -> None:
        inference_cfg = config["inference"]
        self.system, self.inference_cfg = system, inference_cfg
        self.preprocessor = DicomPreprocessor(config["image_processing"]["preprocess"])
        self.quality = ImageQualityAssessor(inference_cfg.get("image_quality"))
        scope_cfg = inference_cfg.get("scope", {})
        self.scope_detector = QuestionScopeDetector(scope_cfg.get("keywords"), scope_cfg.get("unmatched_policy", "out_of_scope"))
        self.uncertainty, self.confidence = UncertaintyEstimator(), ConfidenceScorer()
        self.verification = TrustworthyVerification(float(inference_cfg.get("warnings", {}).get("confidence_threshold", 0.5)))
        self.device = next(system.parameters()).device

    def run(self, image_path: str, question: str, history: Optional[List[Any]] = None, sample_id: str = "sample") -> Dict[str, Any]:
        history = history or []
        placeholders = self.system.answer_generation.placeholder_outputs()
        uncertainty = self.uncertainty.estimate()
        confidence = self.confidence.score(uncertainty)

        def finish(answer, quality, scope, status, debug=None):
            return self.verification.finalize(sample_id, answer, placeholders, uncertainty, confidence, quality, scope, status, debug)

        try:
            image = self.preprocessor(image_path)
        except InvalidImageError as error:
            quality = {"status": Status.AVAILABLE, "label": "invalid", "score": 0.0, "warnings": [f"unreadable: {error}"], "metrics": {}}
            return finish(None, quality, {"label": None, "status": Status.SKIPPED, "reason": "image invalid"}, "failed_invalid_image")
        quality = self.quality.assess(image.gray_unit, image.original_size_hw)

        with execution_context(self.system, ExecutionMode.INFER):          # TẮT gradient
            context = self.system.encode(image.tensor.unsqueeze(0).to(self.device), [question], [history], self.scope_detector)
            scope = context.question.scope[0]
            if context.early_exit_reason == "out_of_scope":
                log.info("out_of_scope -> bỏ qua sinh câu trả lời")
                return finish(None, quality, scope, "completed_out_of_scope")
            if quality["label"] == "invalid" and self.inference_cfg.get("block_on_invalid_quality", False):
                return finish(None, quality, scope, "blocked_invalid_image_quality")
            answer = self.system.generate_answers(context, self.inference_cfg["generation"])[0]
        debug = {"question_type": context.question.type_prompts[0].question_type,
                 "fusion_status": context.fusion.modality_status}
        return finish(answer, quality, scope, "completed", debug)
