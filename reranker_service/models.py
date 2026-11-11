from __future__ import annotations

from pydantic import BaseModel, Field


class RerankRequest(BaseModel):
    query: str = Field(..., min_length=1)
    documents: list[str] = Field(..., min_length=1)
    top_n: int | None = Field(None, ge=1)
    normalize: bool = True
    # Cohere/Jina-style flag; False omits `document` from each result.
    return_documents: bool = True
    # None = server default (RERANKER_MAX_LENGTH).
    max_length: int | None = Field(None, ge=1)
    # Accepted for Cohere/Jina-client compatibility; the served model is fixed.
    model: str | None = None


class RerankResult(BaseModel):
    index: int
    document: str | None = None
    relevance_score: float


class RerankUsage(BaseModel):
    total_pairs: int


class RerankResponse(BaseModel):
    results: list[RerankResult]
    model: str
    usage: RerankUsage


class HealthResponse(BaseModel):
    status: str
    model: str
    device: str
    ready: bool
