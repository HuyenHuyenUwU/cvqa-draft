"""REST API for the CVQA online inference pipeline.

Run from the repository root:
    python -m uvicorn webdemo.backend.app:app --reload
"""
from __future__ import annotations

import json
import os
import tempfile
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from src.common.config import load_config
from src.common.exceptions import CheckpointError, InvalidImageError
from src.online.inference_engine import InferenceEngine

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs" / "online.yaml"


@lru_cache(maxsize=1)
def get_engine() -> InferenceEngine:
    """Load the model once per worker, rather than once per request."""
    config_path = Path(os.getenv("CVQA_CONFIG", str(DEFAULT_CONFIG)))
    config = load_config(config_path)
    checkpoint = os.getenv("CVQA_CHECKPOINT", config.get("checkpoint", ""))
    if not checkpoint:
        raise RuntimeError("Set CVQA_CHECKPOINT or configure checkpoint in online.yaml")
    return InferenceEngine.from_checkpoint(checkpoint, config)


def _parse_history(raw: str) -> List[Any]:
    if not raw.strip():
        return []
    try:
        history = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail=f"history must be valid JSON: {exc.msg}") from exc
    if not isinstance(history, list):
        raise HTTPException(status_code=400, detail="history must be a JSON array")
    return history


app = FastAPI(title="CVQA Web Demo API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CVQA_ALLOWED_ORIGINS", "http://localhost:5173").split(","),
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/api/infer")
async def infer(
    image: UploadFile = File(...),
    question: str = Form(...),
    history: str = Form("[]"),
    sample_id: str = Form("webdemo"),
) -> Dict[str, Any]:
    if not question.strip():
        raise HTTPException(status_code=400, detail="question must not be empty")
    if not image.filename:
        raise HTTPException(status_code=400, detail="image filename is required")

    parsed_history = _parse_history(history)
    suffix = Path(image.filename).suffix.lower() or ".img"
    temp_path: Optional[str] = None
    try:
        contents = await image.read()
        if not contents:
            raise HTTPException(status_code=400, detail="uploaded image is empty")
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(contents)
            temp_path = tmp.name
        return get_engine().infer(temp_path, question, parsed_history, sample_id.strip() or "webdemo")
    except HTTPException:
        raise
    except (InvalidImageError, CheckpointError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    finally:
        if temp_path:
            Path(temp_path).unlink(missing_ok=True)
