"""Đánh giá trên split cố định (chế độ INFER):
    python scripts/evaluate.py --checkpoint checkpoints/exp01/best.pt --split test [--set ...]"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from bone_cvqa.checkpoint_io import load_system_from_checkpoint  # noqa: E402
from bone_cvqa.commons.config import load_config, parse_overrides  # noqa: E402
from bone_cvqa.commons.device import resolve_device  # noqa: E402
from bone_cvqa.commons.logger import setup_logging  # noqa: E402
from bone_cvqa.data.loaders import make_dataloader  # noqa: E402
from bone_cvqa.evaluation.evaluator import evaluate  # noqa: E402

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", default="base,data,model,training,inference")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--set", nargs="*", default=[])
    args = parser.parse_args()
    config = load_config(args.configs.split(","), overrides=parse_overrides(args.set))
    setup_logging(config.get("log_level", "INFO"))
    system, _ = load_system_from_checkpoint(args.checkpoint, {"external_models": config["external_models"]})
    loader = make_dataloader(config, args.split, False, str(Path(config["paths"]["output_dir"]) / "split_metadata.json"))
    result = evaluate(system, loader, resolve_device(config["device"]), config["inference"]["generation"],
                      config["paths"]["output_dir"], args.split)
    print(json.dumps(result, indent=2, ensure_ascii=False))
