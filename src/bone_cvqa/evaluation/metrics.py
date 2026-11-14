"""Chỉ số cơ bản cho câu trả lời tự do (không cần thư viện ngoài).
exact_match, token_f1, rouge_l (LCS), yes_no_accuracy (khi đáp án chuẩn bắt đầu bằng yes/no).
Localization: not_implemented (Lesion Localization còn là placeholder)."""
import re
from collections import Counter
from typing import Dict, List


def normalize_text(text: str) -> str:
    return " ".join(re.sub(r"[^\w\s]", " ", (text or "").lower()).split())


def exact_match(prediction: str, reference: str) -> float:
    return float(normalize_text(prediction) == normalize_text(reference))


def token_f1(prediction: str, reference: str) -> float:
    pred, ref = normalize_text(prediction).split(), normalize_text(reference).split()
    if not pred or not ref:
        return float(pred == ref)
    overlap = sum((Counter(pred) & Counter(ref)).values())
    if overlap == 0:
        return 0.0
    precision, recall = overlap / len(pred), overlap / len(ref)
    return 2 * precision * recall / (precision + recall)


def rouge_l(prediction: str, reference: str) -> float:
    pred, ref = normalize_text(prediction).split(), normalize_text(reference).split()
    if not pred or not ref:
        return float(pred == ref)
    table = [[0] * (len(ref) + 1) for _ in range(len(pred) + 1)]
    for i, p in enumerate(pred, 1):
        for j, r in enumerate(ref, 1):
            table[i][j] = table[i - 1][j - 1] + 1 if p == r else max(table[i - 1][j], table[i][j - 1])
    lcs = table[-1][-1]
    if lcs == 0:
        return 0.0
    precision, recall = lcs / len(pred), lcs / len(ref)
    return 2 * precision * recall / (precision + recall)


def first_yes_no(text: str):
    words = normalize_text(text).split()
    return words[0] if words and words[0] in ("yes", "no") else None


def answer_metrics(predictions: List[str], references: List[str]) -> Dict[str, object]:
    count = len(references)
    if count == 0:
        return {"n": 0}
    yes_no_pairs = [(first_yes_no(p), first_yes_no(r)) for p, r in zip(predictions, references) if first_yes_no(r)]
    return {
        "n": count,
        "exact_match": sum(exact_match(p, r) for p, r in zip(predictions, references)) / count,
        "token_f1": sum(token_f1(p, r) for p, r in zip(predictions, references)) / count,
        "rouge_l": sum(rouge_l(p, r) for p, r in zip(predictions, references)) / count,
        "yes_no_accuracy": (sum(p == r for p, r in yes_no_pairs) / len(yes_no_pairs)) if yes_no_pairs else None,
        "yes_no_n": len(yes_no_pairs),
    }
