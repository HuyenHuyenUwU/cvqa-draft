"""Online inference (no training code imported):
python scripts/infer.py --config configs/online.yaml --checkpoint checkpoints/exp01/best.pt \
       --image xray.jpg --question "Is there a fracture?" [--history history.json] [--out result.json]
history.json: [{"question": "...", "answer": "..."}, ...]"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common.config import load_config  # noqa: E402
from src.online.inference_engine import InferenceEngine  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/online.yaml")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--image", required=True)
    ap.add_argument("--question", required=True)
    ap.add_argument("--history", default=None)
    ap.add_argument("--sample-id", default="sample_001")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    hist = json.loads(Path(a.history).read_text()) if a.history else []
    res = InferenceEngine.from_checkpoint(a.checkpoint, load_config(a.config)).infer(a.image, a.question, hist, a.sample_id)
    txt = json.dumps(res, indent=2, ensure_ascii=False)
    print(txt)
    if a.out:
        Path(a.out).write_text(txt, encoding="utf-8")
