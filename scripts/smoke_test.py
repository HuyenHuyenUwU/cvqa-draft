"""CPU smoke test: synthetic FracAtlas-like data + tiny RANDOM models (no downloads, no real data).
Runs Offline train -> Offline eval -> Online inference from the saved checkpoint, and checks the output schema.
Answers are meaningless (random weights); this verifies plumbing only.

   python scripts/smoke_test.py [--workdir outputs/smoke]
"""
import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from conftest import make_fake_fracatlas, tiny_cfg  # noqa: E402
from src.offline.pipeline import OfflinePipeline  # noqa: E402
from src.online.inference_engine import InferenceEngine  # noqa: E402
from src.online.output_formatter import validate_output_schema  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--workdir", default=None)
    a = ap.parse_args()
    wd = Path(a.workdir) if a.workdir else Path(tempfile.mkdtemp(prefix="cvqa_smoke_"))
    if wd.exists():
        shutil.rmtree(wd)
    wd.mkdir(parents=True)
    root = make_fake_fracatlas(wd / "fracatlas")
    cfg = tiny_cfg()
    cfg.update({"data": {"name": "fracatlas", "root": str(root), "split": {"ratios": [0.6, 0.2, 0.2], "seed": 3},
                         "qa": {"source": "weak_template"}},
                "train": {"epochs": 2, "batch_size": 4, "grad_accum_steps": 2, "lr": 1e-3, "min_loc_steps": 1,
                          "amp": False, "log_every": 1000, "max_train_batches": 3, "max_val_batches": 2},
                "losses": {"weights": {"ans": 1.0, "loc": 0.5}}, "seed": 0, "device": "cpu",
                "output": {"output_dir": str(wd / "out"), "checkpoint_dir": str(wd / "ck"), "log_dir": str(wd / "logs")}})
    pipe = OfflinePipeline(cfg)
    summary = pipe.train()
    print("TRAIN active losses:", summary["active_losses"], "| skipped:", list(summary["skipped_losses"]))
    ev = pipe.evaluate(str(wd / "ck" / "best.pt"), "test")
    print("EVAL:", json.dumps({k: ev[k] for k in ("num_samples", "answer", "localization")}, ensure_ascii=False))

    eng = InferenceEngine.from_checkpoint(str(wd / "ck" / "best.pt"), {"device": "cpu", "generation": {"max_new_tokens": 6}})
    img = next((root / "images" / "Fractured").glob("*.jpg"))
    res = eng.infer(str(img), "Is there a fracture in this X-ray image?", [], "sample_001")
    probs = validate_output_schema(res)
    print("ONLINE:", json.dumps(res, indent=2, ensure_ascii=False))
    (wd / "example_output.json").write_text(json.dumps(res, indent=2, ensure_ascii=False))
    oos = eng.infer(str(img), "What is the capital of France?", [], "sample_002")
    assert not probs and not validate_output_schema(oos), probs
    assert oos["pipeline_status"] == "completed_out_of_scope" and oos["answer"] is None
    print("SMOKE TEST PASSED (plumbing only; random weights). Output:", wd)
