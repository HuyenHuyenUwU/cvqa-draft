"""[PLACEHOLDER] Knowledge Embedding.  in: subgraph + thực thể đã link  ->  out: embedding tri thức [B, D].
Hiện trả embedding=None (KHÔNG sinh vector ngẫu nhiên); Cross-modal Fusion sẽ che mask slot này."""
from typing import Any, Dict

from ...commons.constants import Status
from ...commons.placeholder import PlaceholderModule


class KnowledgeEmbedder(PlaceholderModule):
    module_name = "Knowledge Embedding"

    def empty_result(self) -> Dict[str, Any]:
        return {"embedding": None, "status": Status.NOT_IMPLEMENTED}

    def embed(self, subgraph: Dict[str, Any], linked_entities: Dict[str, Any]) -> Dict[str, Any]:
        return self.run(subgraph, linked_entities)
