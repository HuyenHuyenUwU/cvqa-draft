"""[PLACEHOLDER] Dựng index Bone Disease Ontology. Chưa có ontology/thuật toán index -> chỉ ghi log, KHÔNG tạo file giả."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from bone_cvqa.commons.logger import setup_logging  # noqa: E402
from bone_cvqa.commons.placeholder import PlaceholderModule  # noqa: E402


class OntologyIndexBuilder(PlaceholderModule):
    module_name = "Ontology Index Builder"


if __name__ == "__main__":
    setup_logging("INFO")
    print(OntologyIndexBuilder().run())
