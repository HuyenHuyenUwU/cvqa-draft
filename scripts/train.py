"""OFFLINE train:  python scripts/train.py [--configs base,data,model,training] [--set train.epochs=3 data.datasets.fracatlas.root=/path]"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from bone_cvqa.commons.config import load_config, parse_overrides  # noqa: E402
from bone_cvqa.training.pipeline import OfflinePipeline  # noqa: E402

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", default="base,data,model,training")
    parser.add_argument("--set", nargs="*", default=[])
    args = parser.parse_args()
    summary = OfflinePipeline(load_config(args.configs.split(","), overrides=parse_overrides(args.set))).train()
    print({k: summary[k] for k in ("best_val_loss", "active_losses")})
