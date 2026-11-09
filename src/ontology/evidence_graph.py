"""Placeholders for the 'Evidence Graph' and 'Visual + Knowledge Explanation' outputs (diagram blocks).
Both depend on the ontology branch and therefore return explicit not_implemented results."""
from __future__ import annotations

from ..common.logger import get_logger, log_placeholder
from ..common.schemas import Status

log = get_logger("ontology.evidence")


class PlaceholderEvidenceGraph:
    def __init__(self):
        self._calls = 0

    def predict(self, *_, **__):
        if self._calls == 0:
            log_placeholder(log, "Evidence Graph generation")
        self._calls += 1
        return {"nodes": [], "edges": [], "status": Status.NOT_IMPLEMENTED}


class PlaceholderVisualKnowledgeExplanation:
    def __init__(self):
        self._calls = 0

    def explain(self, *_, **__):
        if self._calls == 0:
            log_placeholder(log, "Visual + Knowledge Explanation")
        self._calls += 1
        return {"text": None, "status": Status.NOT_IMPLEMENTED}
