from __future__ import annotations

from ..common.logger import get_logger, log_placeholder
from .interfaces import KnowledgeEmbedder, empty_knowledge_embedding

log = get_logger("ontology.knowledge_embedding")


class PlaceholderKnowledgeEmbedding(KnowledgeEmbedder):
    """[placeholder] Knowledge Embedding: embedding=None (never random), status=not_implemented."""

    def __init__(self):
        self._calls = 0

    def embed(self, subgraph=None, linked_entities=None):
        if self._calls == 0:
            log_placeholder(log, "Knowledge Embedding")
        self._calls += 1
        return empty_knowledge_embedding()
