"""End-to-end (train -> evaluate -> infer) trên dữ liệu TỔNG HỢP + model nhỏ ngẫu nhiên, chạy được trên CPU, không tải gì.
Chỉ kiểm tra đường ống nối với nhau (câu trả lời vô nghĩa vì trọng số ngẫu nhiên).
    python scripts/run_end2end.py [--workdir outputs/end2end]
Với dữ liệu/model thật: chạy lần lượt scripts/train.py -> evaluate.py -> infer.py (xem README)."""
import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bone_cvqa.checkpoint_io import load_system_from_checkpoint  # noqa: E402
from bone_cvqa.commons.config import deep_merge  # noqa: E402
from bone_cvqa.commons.device import resolve_device  # noqa: E402
from bone_cvqa.data.loaders import make_dataloader  # noqa: E402
from bone_cvqa.data.synthetic import make_fake_fracatlas  # noqa: E402
from bone_cvqa.evaluation.evaluator import evaluate  # noqa: E402
from bone_cvqa.inference.engine import InferenceEngine  # noqa: E402
from bone_cvqa.trustworthy_verification.final_output import validate_final_output  # noqa: E402
from bone_cvqa.training.pipeline import OfflinePipeline  # noqa: E402
from tests.helpers import tiny_config  # noqa: E402

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--workdir", default="outputs/end2end")
    args = parser.parse_args()
    work = Path(args.workdir)
    shutil.rmtree(work, ignore_errors=True)
    root = make_fake_fracatlas(work / "fracatlas")
    config = deep_merge(tiny_config(), {
        "data": {"datasets": {"fracatlas": {"root": str(root)}}, "split": {"ratios": [0.6, 0.2, 0.2], "seed": 3}},
        "paths": {"output_dir": str(work / "out"), "checkpoint_dir": str(work / "ckpt"), "log_dir": str(work / "logs")}})
    print("== 1/3 TRAIN (offline) ==")
    summary = OfflinePipeline(config).train()
    print("active losses:", summary["active_losses"], "| skipped:", sorted(summary["skipped_losses"]))
    checkpoint = str(work / "ckpt" / "best.pt")
    print("== 2/3 EVALUATE ==")
    system, _ = load_system_from_checkpoint(checkpoint, {"external_models": config["external_models"]})
    result = evaluate(system, make_dataloader(config, "test", False), resolve_device("cpu"),
                      config["inference"]["generation"], str(work / "out"), "test")
    print(json.dumps(result["answer"]))
    print("== 3/3 INFER (online) ==")
    engine = InferenceEngine.from_checkpoint(checkpoint, config)
    image = next((root / "images" / "Fractured").glob("*.jpg"))
    answer = engine.infer(str(image), "Is there a fracture in this X-ray image?", [], "sample_001")
    print(json.dumps(answer, indent=2, ensure_ascii=False))
    assert not validate_final_output(answer)
    print("END-TO-END OK (đường ống thông; câu trả lời vô nghĩa do trọng số ngẫu nhiên)")
