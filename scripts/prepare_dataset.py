"""Build deterministic split metadata and a PREVIEW of the generated (weak-supervision) QA pairs.
   python scripts/prepare_dataset.py --config configs/offline.yaml --out outputs/prepared
Nothing is downloaded; generated QA is tagged template_weak_supervision, never expert ground truth."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common.config import load_config  # noqa: E402
from src.data.dataset_factory import build_dataset  # noqa: E402
from src.data.image_preprocessing import ImagePreprocessor  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/offline.yaml")
    ap.add_argument("--out", default="outputs/prepared")
    a = ap.parse_args()
    cfg = load_config(a.config)
    pre = ImagePreprocessor(cfg["preprocess"])
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    for sp in ("train", "val", "test"):
        ds = build_dataset(cfg["data"], sp, pre, str(out / "split_metadata.json"))
        with open(out / f"qa_preview_{sp}.jsonl", "w", encoding="utf-8") as f:
            for it in ds.items:
                f.write(json.dumps({"image_id": it["record"]["image_id"], "turn": it["turn"], "question": it["qa"]["question"],
                                    "answer": it["qa"]["answer"], "answer_source": it["qa"].get("answer_source"),
                                    "lesion_status": it["lesion"]["status"]}, ensure_ascii=False) + "\n")
        print(sp, len(ds.records), "images", len(ds), "QA samples")
