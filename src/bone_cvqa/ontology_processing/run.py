"""Chạy ĐỘC LẬP khối Ontology Processing (placeholder): python -m bone_cvqa.ontology_processing.run"""
import torch

from ..commons.cli import base_parser, print_json
from ..commons.constants import Status
from ..commons.logger import setup_logging
from ..commons.types import (GlobalEmbedding, ImageProcessingOutput, LocalizationResult, RoiFeatures, VisionFeatures)
from .pipeline import OntologyProcessingPipeline


def main(argv=None) -> None:
    base_parser("Ontology Processing (placeholder, chạy độc lập)").parse_args(argv)
    setup_logging("INFO")
    vision = VisionFeatures(torch.zeros(1, 4, 8), torch.zeros(1, 8), (2, 2))
    image = ImageProcessingOutput(vision, LocalizationResult(Status.NOT_IMPLEMENTED), RoiFeatures(Status.SKIPPED),
                                  GlobalEmbedding(torch.zeros(1, 8)))
    print_json(OntologyProcessingPipeline()(image, [{"entities": []}]))


if __name__ == "__main__":
    main()
