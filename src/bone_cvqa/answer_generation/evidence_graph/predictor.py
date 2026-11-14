"""[PLACEHOLDER] Evidence Graph (đầu ra khối 5). Phụ thuộc nhánh ontology chưa có."""
from typing import Any, Dict

from ...commons.constants import Status
from ...commons.placeholder import PlaceholderModule


class EvidenceGraphPredictor(PlaceholderModule):
    module_name = "Evidence Graph"

    def empty_result(self) -> Dict[str, Any]:
        return {"nodes": [], "edges": [], "status": Status.NOT_IMPLEMENTED}
