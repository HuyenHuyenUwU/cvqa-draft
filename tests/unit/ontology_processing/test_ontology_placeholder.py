import logging

from bone_cvqa.ontology_processing.pipeline import OntologyProcessingPipeline


def test_no_fabricated_evidence(tiny_system, tiny_cfg, caplog):
    import torch
    image = tiny_system.image_processing(torch.rand(1, 3, 64, 64))
    with caplog.at_level(logging.WARNING):
        out = OntologyProcessingPipeline(tiny_cfg)(image, [[]])
    assert out.status == "not_implemented"
    assert out.knowledge_embedding is None
    assert out.subgraph in ({}, None) or out.subgraph.get("status") == "not_implemented"
    assert "[TODO]" in caplog.text
