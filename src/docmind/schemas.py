"""Pydantic request/response models for the API."""
from __future__ import annotations

from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=1000)
    top_k: Optional[int] = Field(None, ge=1, le=10)


class Source(BaseModel):
    id: int
    source: str
    page: int
    score: float
    text: str


class AskResponse(BaseModel):
    answer: str
    refused: bool
    sources: List[Source]
    latency_ms: float


class IngestResponse(BaseModel):
    file: str
    chunks_added: int
    documents: Dict[str, int]
