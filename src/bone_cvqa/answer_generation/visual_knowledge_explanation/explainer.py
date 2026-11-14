"""[PLACEHOLDER] Visual + Knowledge Explanation (đầu ra khối 5). Không được bịa bằng chứng."""
from typing import Any, Dict

from ...commons.constants import Status
from ...commons.placeholder import PlaceholderModule


class VisualKnowledgeExplainer(PlaceholderModule):
    module_name = "Visual + Knowledge Explanation"

    def empty_result(self) -> Dict[str, Any]:
        return {"text": None, "status": Status.NOT_IMPLEMENTED}
