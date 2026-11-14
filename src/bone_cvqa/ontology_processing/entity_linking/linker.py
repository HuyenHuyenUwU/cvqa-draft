"""[PLACEHOLDER] Entity Linking phía ontology.  in: ứng viên đã truy xuất -> out: thực thể ontology đã link."""
from typing import Any, Dict

from ...commons.constants import Status
from ...commons.placeholder import PlaceholderModule


class OntologyEntityLinker(PlaceholderModule):
    module_name = "Ontology Entity Linking"

    def empty_result(self) -> Dict[str, Any]:
        return {"entities": [], "status": Status.NOT_IMPLEMENTED}

    def link(self, retrieval: Dict[str, Any]) -> Dict[str, Any]:
        return self.run(retrieval)
