"""Chạy ĐỘC LẬP khối Cross-modal Fusion + Decoder trên đầu vào tổng hợp (không cần ảnh thật):
    python -m bone_cvqa.cross_modal_fusion.run [--set d_model=32 ...]"""
import torch

from ..commons.cli import base_parser, load_from_args, print_json
from ..commons.constants import ExecutionMode, Status
from ..commons.execution_mode import execution_context
from ..commons.types import (GlobalEmbedding, ImageProcessingOutput, LocalizationResult, OntologyOutput,
                             QuestionFeatures, QuestionProcessingOutput, RoiFeatures, TypePrompt, VisionFeatures)
from .pipeline import CrossModalFusionPipeline


def main(argv=None) -> None:
    parser = base_parser("Cross-modal Fusion (chạy độc lập, dữ liệu tổng hợp)")
    config = load_from_args(parser.parse_args(argv))
    d = int(config["d_model"])
    image = ImageProcessingOutput(VisionFeatures(torch.randn(1, 4, d), torch.randn(1, 8), (2, 2)),
                                  LocalizationResult(Status.NOT_IMPLEMENTED), RoiFeatures(Status.SKIPPED),
                                  GlobalEmbedding(torch.randn(1, d)))
    question = QuestionProcessingOutput(QuestionFeatures(torch.randn(1, 6, d), torch.ones(1, 6, dtype=torch.bool), torch.randn(1, d)),
                                        ["q"], [{"entities": []}], [TypePrompt("other", 5, "Task: answer.")])
    pipeline = CrossModalFusionPipeline(config)
    with execution_context(pipeline, ExecutionMode.INFER):
        print_json(pipeline.fuse(question, image, OntologyOutput(Status.NOT_IMPLEMENTED)))


if __name__ == "__main__":
    main()
