"""Khối 3 — Bone Disease Ontology Processing (tất cả là placeholder).  Retriever -> Entity Linking -> Knowledge Embedding.
Dùng chung Offline/Online; baseline CVQA vẫn chạy khi nhánh này chưa có."""
from typing import Any, List, Optional

from ..commons.types import ImageProcessingOutput, OntologyOutput
from .entity_linking.linker import OntologyEntityLinker
from .knowledge_embedding.embedder import KnowledgeEmbedder
from .retriever.retriever import OntologyRetriever


class OntologyProcessingPipeline:
    def __init__(self, config: Optional[dict] = None) -> None:
        self.index_path = (config or {}).get("ontology_processing", {}).get("index_path")
        self.retriever = OntologyRetriever()
        self.entity_linker = OntologyEntityLinker()
        self.knowledge_embedder = KnowledgeEmbedder()

    def __call__(self, image: ImageProcessingOutput, question_entities: List[Any]) -> OntologyOutput:
        subgraph = self.retriever.retrieve(image.roi, image.global_embedding, question_entities, self.index_path)
        linked = self.entity_linker.link(subgraph)
        knowledge = self.knowledge_embedder.embed(subgraph, linked)
        return OntologyOutput(status=knowledge["status"], subgraph=subgraph, linked_entities=linked,
                              knowledge_embedding=knowledge["embedding"])
