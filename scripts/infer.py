"""ONLINE infer: python scripts/infer.py --checkpoint ... --image ... --question ...  (xem bone_cvqa/inference/run.py)"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from bone_cvqa.inference.run import main  # noqa: E402

if __name__ == "__main__":
    main()
