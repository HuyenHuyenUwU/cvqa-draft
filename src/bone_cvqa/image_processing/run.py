"""Chạy ĐỘC LẬP khối Image Processing:
    python -m bone_cvqa.image_processing.run --image xray.png --until vision_encoder
`--until`: dicom_preprocessing | image_quality | vision_encoder | lesion_localization | roi_features | global_embedding
"""
import torch

from ..commons.cli import base_parser, load_from_args, print_json
from ..commons.execution_mode import execution_context
from ..commons.constants import ExecutionMode
from .pipeline import ImageProcessingPipeline


def main(argv=None) -> None:
    parser = base_parser("Image Processing (chạy độc lập)")
    parser.add_argument("--image", required=True)
    parser.add_argument("--until", default="global_embedding")
    parser.add_argument("--quality-thresholds", default=None)
    args = parser.parse_args(argv)
    config = load_from_args(args)
    pipeline = ImageProcessingPipeline(config)
    preprocessed = pipeline.preprocess(args.image)
    if args.until == "dicom_preprocessing":
        return print_json({"tensor": preprocessed.tensor, "original_size_hw": preprocessed.original_size_hw})
    if args.until == "image_quality":
        from .image_quality.assessor import ImageQualityAssessor   # chỉ-Online: import khi thật sự dùng
        return print_json(ImageQualityAssessor().assess(preprocessed.gray_unit, preprocessed.original_size_hw))
    with execution_context(pipeline, ExecutionMode.INFER):
        print_json(pipeline.run_until(preprocessed.tensor.unsqueeze(0), args.until))


if __name__ == "__main__":
    main()
