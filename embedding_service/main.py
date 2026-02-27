from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException

from embedding_service.config import settings
from embedding_service.models import (
    EmbeddingItem,
    EmbeddingRequest,
    EmbeddingResponse,
    HealthResponse,
    SparseQdrant,
    UsageInfo,
)

logging.basicConfig(
    level=settings.log_level,
    format="%(asctime)s · %(levelname)s · embedding · %(message)s",
)
logger = logging.getLogger(__name__)

_state: dict[str, Any] = {}


def _sparse_to_qdrant(lexical_weights: dict) -> SparseQdrant:
    indices = []
    values = []
    for token_id, weight in lexical_weights.items():
        if weight > 0:
            indices.append(int(token_id))
            values.append(float(weight))
    return SparseQdrant(indices=indices, values=values)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Loading model %s (device=%s, fp16=%s)", settings.model, settings.device, settings.use_fp16)
    t0 = time.monotonic()

    from FlagEmbedding import BGEM3FlagModel

    kwargs: dict[str, Any] = {"use_fp16": settings.use_fp16}
    if settings.device == "mps":
        kwargs["devices"] = ["mps"]

    _state["model"] = BGEM3FlagModel(settings.model, **kwargs)
    _state["ready"] = True

    elapsed = time.monotonic() - t0
    logger.info("Model loaded in %.1f s", elapsed)

    yield

    _state.clear()
    logger.info("Model unloaded")


app = FastAPI(
    title="BGE-M3 Embedding Service",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/healthz", response_model=HealthResponse)
async def healthz():
    return HealthResponse(
        status="ok" if _state.get("ready") else "loading",
        model=settings.model,
        device=settings.device,
        ready=_state.get("ready", False),
    )


@app.get("/health", response_model=HealthResponse)
async def health():
    return await healthz()


@app.post("/v1/embeddings", response_model=EmbeddingResponse)
async def create_embeddings(request: EmbeddingRequest):
    if not _state.get("ready"):
        raise HTTPException(status_code=503, detail="Model not ready")

    model = _state["model"]
    texts = request.input

    logger.info("Encoding %d texts (dense=%s, sparse=%s, colbert=%s, max_len=%d)",
                len(texts), request.return_dense, request.return_sparse,
                request.return_colbert, request.max_length)

    t0 = time.monotonic()
    output = model.encode(
        texts,
        batch_size=settings.batch_size,
        max_length=request.max_length,
        return_dense=request.return_dense,
        return_sparse=request.return_sparse,
        return_colbert_vecs=request.return_colbert,
    )
    elapsed = time.monotonic() - t0
    logger.info("Encoded %d texts in %.3f s (%.1f texts/s)", len(texts), elapsed,
                len(texts) / elapsed if elapsed > 0 else 0)

    dense_vecs = output.get("dense_vecs")
    lexical_weights = output.get("lexical_weights")
    colbert_vecs = output.get("colbert_vecs")

    readable_sparse = None
    if lexical_weights is not None:
        if not isinstance(lexical_weights, list):
            lexical_weights = [lexical_weights]
        readable_sparse = model.convert_id_to_token(lexical_weights)
        if isinstance(readable_sparse, dict):
            readable_sparse = [readable_sparse]

    data: list[EmbeddingItem] = []
    for i in range(len(texts)):
        item = EmbeddingItem(index=i)

        if dense_vecs is not None:
            item.dense_embedding = dense_vecs[i].tolist()

        if lexical_weights is not None:
            item.sparse_embedding = readable_sparse[i] if readable_sparse else None
            item.sparse_qdrant = _sparse_to_qdrant(lexical_weights[i])

        if colbert_vecs is not None:
            item.colbert_embedding = colbert_vecs[i].tolist()

        data.append(item)

    return EmbeddingResponse(
        data=data,
        model=settings.model,
        usage=UsageInfo(total_tokens=0),
    )
