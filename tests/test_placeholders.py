import logging

import pytest

from src.ontology.branch import OntologyBranch
from src.ontology.evidence_graph import PlaceholderEvidenceGraph, PlaceholderVisualKnowledgeExplanation
from src.ontology.entity_linker import PlaceholderOntologyEntityLinker
from src.ontology.knowledge_embedding import PlaceholderKnowledgeEmbedding
from src.ontology.retriever import PlaceholderOntologyRetriever
from src.question_processing.entity_linking import PlaceholderQuestionEntityLinker
from src.uncertainty.estimator import PlaceholderConfidenceEstimator, PlaceholderUncertaintyEstimator


@pytest.fixture(autouse=True)
def _propagate():
    lg = logging.getLogger("cvqa")
    old = lg.propagate
    lg.propagate = True            # let caplog see records
    yield
    lg.propagate = old


def test_ontology_modules_return_not_implemented_and_log(caplog):
    with caplog.at_level(logging.WARNING, logger="cvqa"):
        r = PlaceholderOntologyRetriever().retrieve()
        e = PlaceholderOntologyEntityLinker().link(r)
        k = PlaceholderKnowledgeEmbedding().embed(r, e)
    assert r == {"nodes": [], "edges": [], "status": "not_implemented"}
    assert e == {"entities": [], "status": "not_implemented"}
    assert k == {"embedding": None, "status": "not_implemented"}          # never random embeddings
    text = caplog.text
    for name in ("Ontology Retriever", "Ontology Entity Linking", "Knowledge Embedding"):
        assert f"[TODO] {name} is a placeholder. It will be implemented in a later development stage." in text


def test_question_entity_linking_placeholder():
    assert PlaceholderQuestionEntityLinker().link("any") == {"entities": [], "status": "not_implemented"}


def test_uncertainty_and_confidence_are_null(caplog):
    with caplog.at_level(logging.WARNING, logger="cvqa"):
        u = PlaceholderUncertaintyEstimator().estimate(logits=None, answer="yes")
        c = PlaceholderConfidenceEstimator().estimate(u)
    assert u == {"score": None, "status": "not_implemented"} and c == {"score": None, "status": "not_implemented"}
    assert "[TODO] Uncertainty Estimation is a placeholder" in caplog.text
    assert "[TODO] Confidence Score is a placeholder" in caplog.text


def test_evidence_and_explanation_placeholders():
    assert PlaceholderEvidenceGraph().predict()["status"] == "not_implemented"
    assert PlaceholderVisualKnowledgeExplanation().explain() == {"text": None, "status": "not_implemented"}


def test_branch_chain_and_log_once(caplog):
    b = OntologyBranch()
    with caplog.at_level(logging.WARNING, logger="cvqa"):
        for _ in range(3):
            out = b.run()
    assert out["knowledge"]["embedding"] is None and out["subgraph"]["nodes"] == []
    assert caplog.text.count("Ontology Retriever is a placeholder") == 1      # no log flooding


def test_type_conditional_prompt_is_rule_based():
    from src.question_processing.type_conditional_prompt import TypeConditionalPromptBuilder
    b = TypeConditionalPromptBuilder()
    cases = {"Where is the lesion?": "anatomy_location", "Is there a fracture?": "fracture_identification",
             "Why do you say so?": "explanation", "What disease is this?": "disease_identification",
             "Describe the cortical margin": "radiographic_sign", "Hello": "other"}
    for q, t in cases.items():
        r = b.build(q, {"entities": []})
        assert r["question_type"] == t and r["method"] == "rule_based_keywords"
    assert "Relevant concepts" not in b.build("Is there a fracture?", {"entities": []})["prompt_text"]   # nothing invented
