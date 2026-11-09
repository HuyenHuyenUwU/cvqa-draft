"""Interfaces + result schemas for the Bone Disease Ontology branch (ALL PLACEHOLDERS for now).

Retriever          (ROI feats, global emb, question entities, ontology index) -> subgraph
Ontology EntityLinker (retrieved candidates)                                  -> linked entities
KnowledgeEmbedding (subgraph, linked entities)                                -> embedding (or None)

No module here may return invented concepts, triples or random embeddings.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Optional

from ..common.schemas import Status


@dataclass
class OntologyIndexRef:
    """Reference to a (future) indexed ontology. ``path=None`` means 'not available'."""
    path: Optional[str] = None

    @property
    def available(self) -> bool:
        return self.path is not None


def empty_subgraph() -> Dict[str, Any]:
    return {"nodes": [], "edges": [], "status": Status.NOT_IMPLEMENTED}


def empty_links() -> Dict[str, Any]:
    return {"entities": [], "status": Status.NOT_IMPLEMENTED}


def empty_knowledge_embedding() -> Dict[str, Any]:
    # embedding is None on purpose: fusion masks the knowledge slot instead of consuming random values
    return {"embedding": None, "status": Status.NOT_IMPLEMENTED}


class OntologyRetriever(ABC):
    @abstractmethod
    def retrieve(self, roi_features, global_embedding, question_entities, ontology_index: OntologyIndexRef) -> Dict[str, Any]: ...


class OntologyEntityLinker(ABC):
    @abstractmethod
    def link(self, retrieval: Dict[str, Any]) -> Dict[str, Any]: ...


class KnowledgeEmbedder(ABC):
    @abstractmethod
    def embed(self, subgraph: Dict[str, Any], linked_entities: Dict[str, Any]) -> Dict[str, Any]: ...
