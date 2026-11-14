"""[PLACEHOLDER] Ontology Retriever.  in: ROI features, global embedding, entity câu hỏi, Indexed Bone Disease Ontology
out: subgraph truy xuất được. Hiện trả subgraph rỗng."""
from typing import Any, Dict

from ...commons.constants import Status
from ...commons.placeholder import PlaceholderModule


class OntologyRetriever(PlaceholderModule):
    module_name = "Ontology Retriever"

    def empty_result(self) -> Dict[str, Any]:
        return {"nodes": [], "edges": [], "status": Status.NOT_IMPLEMENTED}

    def retrieve(self, roi, global_embedding, question_entities, ontology_index=None) -> Dict[str, Any]:
        return self.run(roi, global_embedding, question_entities, ontology_index)
