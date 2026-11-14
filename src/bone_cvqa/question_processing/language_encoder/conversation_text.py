"""Dựng chuỗi hội thoại có dấu phân tách rõ ràng. Không bao giờ chứa đáp án của câu hỏi hiện tại.

    [HISTORY]
    User: ...
    Assistant: ...

    [CURRENT QUESTION]
    ...
"""
from typing import Any, Dict, Iterable, List, Optional


def normalize_history(history: Optional[Iterable[Any]]) -> List[Dict[str, str]]:
    """Chấp nhận {"question","answer"}, {"role","content"} hoặc (câu hỏi, đáp án); trả về danh sách lượt theo thứ tự."""
    turns: List[Dict[str, str]] = []
    for item in history or []:
        if isinstance(item, dict) and "role" in item:
            turns.append({"role": item["role"], "content": str(item["content"])})
        elif isinstance(item, dict):
            turns.append({"role": "user", "content": str(item.get("question", ""))})
            if item.get("answer") is not None:
                turns.append({"role": "assistant", "content": str(item["answer"])})
        elif isinstance(item, (tuple, list)) and len(item) == 2:
            turns.append({"role": "user", "content": str(item[0])})
            turns.append({"role": "assistant", "content": str(item[1])})
        else:
            raise ValueError(f"Lượt hội thoại không hỗ trợ: {item!r}")
    return turns


def build_conversation_text(question: str, history: Optional[Iterable[Any]] = None) -> str:
    lines: List[str] = []
    turns = normalize_history(history)
    if turns:
        lines.append("[HISTORY]")
        for turn in turns:
            speaker = "User" if turn["role"] == "user" else "Assistant"
            lines.append(f"{speaker}: {turn['content']}")
        lines.append("")
    lines += ["[CURRENT QUESTION]", question.strip()]
    return "\n".join(lines)
