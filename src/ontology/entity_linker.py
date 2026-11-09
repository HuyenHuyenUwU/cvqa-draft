from __future__ import annotations

from ..common.logger import get_logger, log_placeholder
from .interfaces import OntologyEntityLinker, empty_links

log = get_logger("ontology.entity_linker")


class PlaceholderOntologyEntityLinker(OntologyEntityLinker):
    """[placeholder] Ontology-side Entity Linking (distinct from question-side linking)."""

    def __init__(self):
        self._calls = 0

    def link(self, retrieval=None):
        if self._calls == 0:
            log_placeholder(log, "Ontology Entity Linking")
        self._calls += 1
        return empty_links()
