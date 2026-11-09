"""Online orchestration (diagram: pipeline_online.png). Imports nothing from ``src.offline``.

DICOM Preprocessing -> Image Quality Assessment -> (Vision Encoder -> Localization -> RoI / Global)
Question: Language Encoder -> Entity Linking -> Question Scope Detector -> (in-scope) Type-Conditional Prompt
        | out-of-scope -> Warning Generator (answer generation skipped)
Ontology placeholders -> Cross-modal Fusion -> LLM Decoder -> Answer / Heatmap-Box / Evidence Graph /
Visual+Knowledge Explanation / Uncertainty / Confidence -> Warning Generator -> Final Output
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch

from ..common.exceptions import InvalidImageError
from ..common.logger import get_logger
from ..common.schemas import Status
from ..data.image_preprocessing import ImagePreprocessor
from ..models.cvqa_inputs import build_model_inputs
from ..ontology.branch import OntologyBranch
from ..ontology.evidence_graph import PlaceholderEvidenceGraph, PlaceholderVisualKnowledgeExplanation
from ..question_processing.processor import QuestionProcessor
from ..question_processing.question_scope_detector import RuleBasedQuestionScopeDetector
from ..uncertainty.estimator import PlaceholderConfidenceEstimator, PlaceholderUncertaintyEstimator
from .image_quality import ImageQualityAssessor
from .output_formatter import format_output, localization_block
from .warning_generator import WarningGenerator

log = get_logger("online.pipeline")
NA = {"status": Status.NOT_AVAILABLE}


class OnlinePipeline:
    def __init__(self, model, preprocessor: ImagePreprocessor, device, cfg: Optional[Dict[str, Any]] = None):
        cfg = cfg or {}
        self.model, self.pre, self.device, self.cfg = model.to(device).eval(), preprocessor, device, cfg
        self.iqa = ImageQualityAssessor(cfg.get("image_quality"))
        sc = cfg.get("scope", {})
        self.scope = RuleBasedQuestionScopeDetector(sc.get("keywords"), sc.get("unmatched_policy", "out_of_scope"))
        self.qp, self.onto = QuestionProcessor(), OntologyBranch()
        self.evidence, self.vke = PlaceholderEvidenceGraph(), PlaceholderVisualKnowledgeExplanation()
        self.unc, self.conf = PlaceholderUncertaintyEstimator(), PlaceholderConfidenceEstimator()
        self.warner = WarningGenerator(float(cfg.get("warnings", {}).get("confidence_threshold", 0.5)))
        self.gen = cfg.get("generation", {})
        self.block_on_invalid_quality = bool(cfg.get("block_on_invalid_quality", False))
        self.include_heatmap = bool(cfg.get("include_heatmap", False))

    def run(self, image_path: str, question: str, history: Optional[List[Any]] = None,
            sample_id: str = "sample") -> Dict[str, Any]:
        history = history or []
        evi, vke = self.evidence.predict(), self.vke.explain()
        unc, conf = self.unc.estimate(), self.conf.estimate(None)
        loc_na = localization_block(None, False)

        # 1. Image Processing: DICOM preprocessing + quality assessment
        try:
            img = self.pre(image_path)
            iq = self.iqa.assess(img.gray_minmax, img.original_size)
        except InvalidImageError as e:
            iq = {"status": Status.AVAILABLE, "label": "invalid", "score": 0.0, "warnings": [f"unreadable: {e}"], "metrics": {}}
            scope = {"label": None, "status": Status.SKIPPED, "reason": "image invalid", "method": None}
            return format_output(sample_id, None, loc_na, evi, vke, unc, conf, iq, scope,
                                 self.warner.generate(iq, None, unc, conf), "failed_invalid_image")

        # 2. Question Processing (+ Question Scope Detector)
        qinfo = self.qp.process(question, history)
        scope = self.scope.detect(question, qinfo["entities"], ontology_available=self.onto.index.available,
                                  history=history)
        if scope["label"] == "out_of_scope":
            log.info("out_of_scope question -> skipping answer generation")
            return format_output(sample_id, None, loc_na, evi, vke, unc, conf, iq, scope,
                                 self.warner.generate(iq, scope, unc, conf), "completed_out_of_scope")
        if iq["label"] == "invalid" and self.block_on_invalid_quality:
            return format_output(sample_id, None, loc_na, evi, vke, unc, conf, iq, scope,
                                 self.warner.generate(iq, scope, unc, conf), "blocked_invalid_image_quality")

        # 3-5. Ontology (placeholders) -> Fusion -> LLM decoder -> Answer Generation
        inp = build_model_inputs(self.model, self.qp, self.onto, [question], [history])
        with torch.no_grad():
            out = self.model.generate(img.tensor.unsqueeze(0).to(self.device), inp["conv_texts"], inp["type_ids"],
                                      inp["prompts"], knowledge=inp["knowledge"],
                                      max_new_tokens=self.gen.get("max_new_tokens"),
                                      do_sample=bool(self.gen.get("do_sample", False)),
                                      temperature=float(self.gen.get("temperature", 1.0)),
                                      top_p=float(self.gen.get("top_p", 1.0)), return_scores=False)
        loc = localization_block(out["loc"], out["localizer_trained"], self.include_heatmap, out["grid_size"])
        unc = self.unc.estimate(logits=out["scores"], multimodal_repr=None, answer=out["texts"][0])
        conf = self.conf.estimate(unc)
        warnings = self.warner.generate(iq, scope, unc, conf)
        return format_output(sample_id, out["texts"][0], loc, evi, vke, unc, conf, iq, scope, warnings, "completed",
                             extra={"question_type": qinfo["type_prompt"]["question_type"],
                                    "fusion_status": out["fusion_status"]})
