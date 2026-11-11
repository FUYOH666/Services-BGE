from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator


class EmbeddingRequest(BaseModel):
    # OpenAI accepts a single string or a list of strings.
    input: list[str] = Field(..., description="Text or list of texts to embed")
    # Accepted for OpenAI-client compatibility; the served model is fixed.
    model: str | None = None
    encoding_format: Literal["float", "base64"] = "float"
    # OpenAI `dimensions`. Only Matryoshka backends accept a smaller size.
    dimensions: int | None = Field(None, ge=1)
    # Cohere/Voyage-style hint for asymmetric models: queries get the query
    # prompt, documents the document prompt. Ignored by bge-m3.
    input_type: Literal["query", "document"] | None = None
    user: str | None = None
    return_dense: bool = True
    return_sparse: bool = True
    return_colbert: bool = False
    # None = server default (EMBEDDING_MAX_LENGTH).
    max_length: int | None = Field(None, ge=1)
    # Drop the duplicate `dense_embedding` and token-readable `sparse_embedding`.
    compact: bool = False

    @field_validator("input", mode="before")
    @classmethod
    def _coerce_input(cls, value):
        if isinstance(value, str):
            value = [value]
        if isinstance(value, list) and len(value) == 0:
            raise ValueError("input must contain at least one text")
        return value


class SparseQdrant(BaseModel):
    indices: list[int]
    values: list[float]


class EmbeddingItem(BaseModel):
    object: str = "embedding"
    index: int
    # OpenAI-compatible clients (including OpenClaw memory search) read this
    # canonical field. Keep dense_embedding for existing hybrid-search clients.
    # With encoding_format="base64" both hold little-endian float32 base64.
    embedding: list[float] | str | None = None
    dense_embedding: list[float] | str | None = None
    sparse_embedding: dict[str, float] | None = None
    sparse_qdrant: SparseQdrant | None = None
    colbert_embedding: list[list[float]] | None = None


class UsageInfo(BaseModel):
    prompt_tokens: int = 0
    total_tokens: int = 0


class EmbeddingResponse(BaseModel):
    object: str = "list"
    data: list[EmbeddingItem]
    model: str
    usage: UsageInfo


class ModelCard(BaseModel):
    id: str
    object: str = "model"
    owned_by: str = "local"
    backend: str
    capabilities: dict[str, bool]
    dimensions: int | None = None
    max_length: int
    max_length_cap: int
    device: str
    fp16: bool


class ModelList(BaseModel):
    object: str = "list"
    data: list[ModelCard]


class HealthResponse(BaseModel):
    status: str
    model: str
    device: str
    ready: bool
