"""Bone Disease Ontology Processing block: Retriever -> Entity Linking -> Knowledge Embedding.
All three stages are placeholders; this class only chains them and exposes one result dict that is
passed to Cross-modal Fusion (which masks the knowledge slot when status != available)."""
from __future__ import annotations

from typing import Any, Dict, Optional

from .entity_linker import PlaceholderOntologyEntityLinker
from .interfaces import OntologyIndexRef
from .knowledge_embedding import PlaceholderKnowledgeEmbedding
from .retriever import PlaceholderOntologyRetriever


class OntologyBranch:
    def __init__(self, index: Optional[OntologyIndexRef] = None, retriever=None, linker=None, embedder=None):
        self.index = index or OntologyIndexRef(None)
        self.retriever = retriever or PlaceholderOntologyRetriever()
        self.linker = linker or PlaceholderOntologyEntityLinker()
        self.embedder = embedder or PlaceholderKnowledgeEmbedding()

    def run(self, roi_features=None, global_embedding=None, question_entities=None) -> Dict[str, Any]:
        sub = self.retriever.retrieve(roi_features, global_embedding, question_entities, self.index)
        links = self.linker.link(sub)
        know = self.embedder.embed(sub, links)
        return {"subgraph": sub, "linked_entities": links, "knowledge": know}
