"""Offline training:  python scripts/train.py --config configs/offline.yaml [--set train.epochs=3 ...]"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common.config import deep_merge, load_config  # noqa: E402
from src.offline.pipeline import OfflinePipeline  # noqa: E402


def parse_overrides(items):
    out = {}
    for it in items or []:
        k, v = it.split("=", 1)
        d = out
        *path, last = k.split(".")
        for p in path:
            d = d.setdefault(p, {})
        import yaml
        d[last] = yaml.safe_load(v)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/offline.yaml")
    ap.add_argument("--set", nargs="*", help="dotted overrides, e.g. data.root=/path train.epochs=2")
    a = ap.parse_args()
    cfg = load_config(a.config, parse_overrides(a.set))
    print(OfflinePipeline(cfg).train())
