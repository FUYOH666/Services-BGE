from __future__ import annotations

from pydantic import BaseModel, Field


class EmbeddingRequest(BaseModel):
    input: list[str] = Field(..., min_length=1, description="List of texts to embed")
    return_dense: bool = True
    return_sparse: bool = True
    return_colbert: bool = False
    max_length: int = Field(512, ge=1, le=8192)


class SparseQdrant(BaseModel):
    indices: list[int]
    values: list[float]


class EmbeddingItem(BaseModel):
    index: int
    dense_embedding: list[float] | None = None
    sparse_embedding: dict[str, float] | None = None
    sparse_qdrant: SparseQdrant | None = None
    colbert_embedding: list[list[float]] | None = None


class UsageInfo(BaseModel):
    total_tokens: int = 0


class EmbeddingResponse(BaseModel):
    object: str = "list"
    data: list[EmbeddingItem]
    model: str
    usage: UsageInfo


class HealthResponse(BaseModel):
    status: str
    model: str
    device: str
    ready: bool
