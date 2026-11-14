"""Deterministic, leakage-aware splitting.

Group key = patient_id when available (patient-level split), else image_id.
Assignment uses a seeded hash of the group key, so adding images never moves old ones.
An official split file ({"train":[ids],"val":[ids],"test":[ids]}) takes priority.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..commons.exceptions import DatasetError


def _bucket(key: str, seed: int) -> float:
    h = hashlib.sha256(f"{seed}:{key}".encode()).hexdigest()
    return int(h[:12], 16) / float(16 ** 12)


def assign_splits(records: List[Dict[str, Any]], ratios=(0.7, 0.15, 0.15), seed: int = 42,
                  official_split_file: Optional[str] = None) -> Dict[str, str]:
    """Returns {image_id: split_name}."""
    if official_split_file:
        with open(official_split_file, "r", encoding="utf-8") as f:
            official = json.load(f)
        inv = {str(i): s for s, ids in official.items() for i in ids}
        missing = [r["image_id"] for r in records if str(r["image_id"]) not in inv]
        if missing:
            raise DatasetError(f"{len(missing)} images missing from official split file, e.g. {missing[:3]}")
        return {str(r["image_id"]): inv[str(r["image_id"])] for r in records}
    tr, va, _ = ratios
    out = {}
    for r in records:
        key = str(r.get("patient_id") or r["image_id"])
        u = _bucket(key, seed)
        out[str(r["image_id"])] = "train" if u < tr else ("val" if u < tr + va else "test")
    return out


def check_no_leakage(records: List[Dict[str, Any]], split_of: Dict[str, str]) -> None:
    seen: Dict[str, str] = {}
    for r in records:
        if not r.get("patient_id"):
            continue
        s = split_of[str(r["image_id"])]
        if seen.setdefault(str(r["patient_id"]), s) != s:
            raise DatasetError(f"Patient {r['patient_id']} appears in several splits (leakage)")


def save_split_metadata(path, records, split_of, seed, ratios, group_by) -> None:
    meta = {"seed": seed, "ratios": list(ratios), "group_by": group_by,
            "counts": {s: sum(1 for v in split_of.values() if v == s) for s in ("train", "val", "test")},
            "assignment": split_of}
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=1)
