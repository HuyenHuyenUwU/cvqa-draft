"""Offline evaluation on a fixed split:
   python scripts/evaluate.py --config configs/offline.yaml --checkpoint checkpoints/exp/best.pt --split test"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.train import parse_overrides  # noqa: E402
from src.common.config import load_config  # noqa: E402
from src.offline.pipeline import OfflinePipeline  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/offline.yaml")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--set", nargs="*")
    a = ap.parse_args()
    cfg = load_config(a.config, parse_overrides(a.set))
    print(json.dumps(OfflinePipeline(cfg).evaluate(a.checkpoint, a.split), indent=1))
