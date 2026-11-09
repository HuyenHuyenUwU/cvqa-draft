from __future__ import annotations

from ..common.logger import get_logger, log_placeholder
from .interfaces import OntologyIndexRef, OntologyRetriever, empty_subgraph

log = get_logger("ontology.retriever")


class PlaceholderOntologyRetriever(OntologyRetriever):
    """[placeholder] Ontology Retriever: returns an empty subgraph, status=not_implemented."""

    def __init__(self):
        self._calls = 0

    def retrieve(self, roi_features=None, global_embedding=None, question_entities=None,
                 ontology_index: OntologyIndexRef = OntologyIndexRef()):
        if self._calls == 0:
            log_placeholder(log, "Ontology Retriever")      # first call at WARNING, then silent (avoids log floods)
        self._calls += 1
        return empty_subgraph()
