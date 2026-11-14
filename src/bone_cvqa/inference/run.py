"""Chạy Online một mẫu:
    python -m bone_cvqa.inference.run --checkpoint checkpoints/exp01/best.pt --image xray.png --question "Is there a fracture?"
"""
import json
from pathlib import Path

from ..commons.cli import base_parser, load_from_args
from .engine import InferenceEngine


def main(argv=None) -> None:
    parser = base_parser("Online inference", "base,model,inference")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--question", required=True)
    parser.add_argument("--history", default=None, help='JSON: [{"question": "...", "answer": "..."}]')
    parser.add_argument("--sample-id", default="sample_001")
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)
    config = load_from_args(args)
    history = json.loads(Path(args.history).read_text()) if args.history else []
    result = InferenceEngine.from_checkpoint(args.checkpoint, config).infer(args.image, args.question, history, args.sample_id)
    text = json.dumps(result, indent=2, ensure_ascii=False)
    print(text)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
