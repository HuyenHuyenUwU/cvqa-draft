"""Shared (Offline + Online) conversion of raw questions/histories into model inputs:
Question Processing (entity linking placeholder + Type-Conditional Prompt) and the ontology placeholders."""
from __future__ import annotations

from typing import Any, Dict, List

import torch

from ..ontology.branch import OntologyBranch
from ..question_processing.processor import QuestionProcessor


def build_model_inputs(model, qp: QuestionProcessor, ontology: OntologyBranch, questions: List[str],
                       histories: List[Any]) -> Dict[str, Any]:
    infos = [qp.process(q, h) for q, h in zip(questions, histories)]
    prompts = [model.decoder.build_prompt(q, h, i["type_prompt"]["prompt_text"])
               for q, h, i in zip(questions, histories, infos)]
    onto = ontology.run(None, None, [i["entities"] for i in infos])   # placeholder ignores its inputs
    return {"conv_texts": [i["conversation_text"] for i in infos],
            "type_ids": torch.tensor([i["type_prompt"]["type_id"] for i in infos], dtype=torch.long),
            "prompts": prompts, "knowledge": onto["knowledge"], "ontology": onto, "question_info": infos}
