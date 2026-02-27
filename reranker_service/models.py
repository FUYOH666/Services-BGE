from __future__ import annotations

from pydantic import BaseModel, Field


class RerankRequest(BaseModel):
    query: str = Field(..., min_length=1)
    documents: list[str] = Field(..., min_length=1)
    top_n: int | None = None
    normalize: bool = True


class RerankResult(BaseModel):
    index: int
    document: str
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
