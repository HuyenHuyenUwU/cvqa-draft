"""Chạy ĐỘC LẬP khối Question Processing:
    python -m bone_cvqa.question_processing.run --question "Is there a fracture?" [--history h.json] [--with-scope]
`--with-scope` bật Question Scope Detector (chỉ-Online)."""
import json
from pathlib import Path

from ..commons.cli import base_parser, load_from_args, print_json
from ..commons.constants import ExecutionMode
from ..commons.execution_mode import execution_context
from .pipeline import QuestionProcessingPipeline


def main(argv=None) -> None:
    parser = base_parser("Question Processing (chạy độc lập)")
    parser.add_argument("--question", required=True)
    parser.add_argument("--history", default=None)
    parser.add_argument("--with-scope", action="store_true")
    args = parser.parse_args(argv)
    config = load_from_args(args)
    history = json.loads(Path(args.history).read_text()) if args.history else []
    detector = None
    if args.with_scope:
        from .scope_detector.detector import QuestionScopeDetector
        detector = QuestionScopeDetector()
    pipeline = QuestionProcessingPipeline(config)
    with execution_context(pipeline, ExecutionMode.INFER):
        print_json(pipeline([args.question], [history], detector))


if __name__ == "__main__":
    main()
