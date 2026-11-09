"""Dataset inspection report (Phase 1 helper). Does NOT assume any schema: it walks the
directory and reports extensions, tabular headers and JSON keys so adapters/YAML can be set.

Usage: python scripts/inspect_dataset.py --root /path/to/dataset --out outputs/inspect_<name>.json
"""
import argparse
import csv
import json
from collections import Counter
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", default="outputs/dataset_inspection.json")
    ap.add_argument("--max-json-keys", type=int, default=30)
    a = ap.parse_args()
    root = Path(a.root)
    files = [p for p in root.rglob("*") if p.is_file()]
    report = {"root": str(root), "num_files": len(files),
              "extensions": dict(Counter(p.suffix.lower() for p in files)),
              "top_level_dirs": {d.name: sum(1 for _ in d.rglob("*")) for d in root.iterdir() if d.is_dir()},
              "tables": {}, "json": {}, "notes": []}
    for p in files:
        rel = str(p.relative_to(root))
        if p.suffix.lower() == ".csv":
            with open(p, "r", encoding="utf-8-sig", newline="") as f:
                rd = csv.reader(f)
                header = next(rd, [])
                rows = sum(1 for _ in rd)
            report["tables"][rel] = {"columns": header, "rows": rows}
        elif p.suffix.lower() in (".xlsx", ".xls"):
            report["notes"].append(f"{rel}: Excel file - export to CSV before using the adapters")
        elif p.suffix.lower() == ".json" and p.stat().st_size < 200_000_000:
            try:
                with open(p, "r", encoding="utf-8") as f:
                    j = json.load(f)
                report["json"][rel] = {"type": type(j).__name__,
                                       "keys": list(j.keys())[: a.max_json_keys] if isinstance(j, dict) else len(j)}
            except Exception as e:
                report["json"][rel] = {"error": str(e)}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=1, ensure_ascii=False)[:4000])
    print(f"\nFull report -> {a.out}")


if __name__ == "__main__":
    main()
