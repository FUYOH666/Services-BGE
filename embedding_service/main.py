from __future__ import annotations

import asyncio
import base64
import logging
import time
from contextlib import asynccontextmanager
from typing import Any

import numpy as np
from fastapi import Depends, FastAPI, HTTPException, Response, status
from fastapi.responses import JSONResponse

from embedding_service.config import settings
from embedding_service.models import (
    EmbeddingItem,
    EmbeddingRequest,
    EmbeddingResponse,
    HealthResponse,
    SparseQdrant,
    UsageInfo,
)
from service_common.runtime import (
    count_tokens,
    enforce_text_limits,
    make_api_key_dependency,
    maybe_clear_mps_cache,
    resolve_max_length,
    resolve_model_path,
)

logging.basicConfig(
    level=settings.log_level,
    format="%(asctime)s · %(levelname)s · embedding · %(message)s",
)
logger = logging.getLogger(__name__)

_state: dict[str, Any] = {}

require_api_key = make_api_key_dependency(lambda: settings.api_key)


def _sparse_to_qdrant(lexical_weights: dict) -> SparseQdrant:
    indices = []
    values = []
    for token_id, weight in lexical_weights.items():
        if weight > 0:
            indices.append(int(token_id))
            values.append(float(weight))
    return SparseQdrant(indices=indices, values=values)


def _encode_dense(vector: Any, encoding_format: str) -> list[float] | str:
    if encoding_format == "base64":
        if not isinstance(vector, np.ndarray):
            vector = vector.tolist()
        array = np.asarray(vector, dtype="<f4")
        return base64.b64encode(array.tobytes()).decode("ascii")
    return vector.tolist()


def _health_response(*, ready: bool, status_text: str) -> HealthResponse:
    return HealthResponse(
        status=status_text,
        model=settings.model,
        device=settings.device,
        ready=ready,
    )


def _warmup(model: Any) -> int | None:
    """Run one small inference so the first real request is not slow.

    Returns the dense vector size, used to validate the OpenAI `dimensions` field.
    """
    output = model.encode(
        ["warmup"],
        batch_size=1,
        max_length=32,
        return_dense=True,
        return_sparse=True,
        return_colbert_vecs=False,
    )
    maybe_clear_mps_cache(settings.device, settings.mps_cache_threshold_mb)
    dense = output.get("dense_vecs")
    if dense is None:
        return None
    return len(dense[0].tolist())


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Loading model %s (device=%s, fp16=%s)", settings.model, settings.device, settings.use_fp16)
    t0 = time.monotonic()
    _state["ready"] = False
    _state["inference_lock"] = asyncio.Lock()

    from FlagEmbedding import BGEM3FlagModel

    kwargs: dict[str, Any] = {"use_fp16": settings.use_fp16}
    if settings.device == "mps":
        kwargs["devices"] = ["mps"]

    try:
        model_path = resolve_model_path(settings.model, settings.model_revision)
        model = BGEM3FlagModel(model_path, **kwargs)
        _state["model"] = model
        if settings.warmup:
            _state["dense_dim"] = _warmup(model)
        _state["ready"] = True
    except Exception:
        logger.exception("Failed to load model %s", settings.model)
        raise

    elapsed = time.monotonic() - t0
    logger.info("Model loaded%s in %.1f s", " and warmed up" if settings.warmup else "", elapsed)

    yield

    _state.clear()
    logger.info("Model unloaded")


app = FastAPI(
    title="BGE-M3 Embedding Service",
    version="0.2.0",
    lifespan=lifespan,
)


@app.get("/livez", response_model=HealthResponse)
async def livez():
    return _health_response(ready=_state.get("ready", False), status_text="ok")


@app.get("/readyz", response_model=HealthResponse)
async def readyz(response: Response):
    ready = _state.get("ready", False)
    if not ready:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return _health_response(ready=ready, status_text="ok" if ready else "loading")


@app.get("/healthz", response_model=HealthResponse)
async def healthz(response: Response):
    return await readyz(response)


@app.get("/health", response_model=HealthResponse)
async def health(response: Response):
    return await readyz(response)


@app.post("/v1/embeddings", response_model=EmbeddingResponse, dependencies=[Depends(require_api_key)])
async def create_embeddings(request: EmbeddingRequest):
    if not _state.get("ready"):
        raise HTTPException(status_code=503, detail="Model not ready")

    texts = request.input
    enforce_text_limits(texts, max_items=settings.max_items, max_chars=settings.max_text_chars, what="texts")
    max_length = resolve_max_length(request.max_length, settings.max_length, settings.max_length_cap)

    dense_dim = _state.get("dense_dim")
    if request.dimensions is not None and dense_dim is not None and request.dimensions != dense_dim:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"{settings.model} only returns {dense_dim}-dimensional vectors",
        )

    model = _state["model"]
    inference_lock: asyncio.Lock = _state["inference_lock"]

    logger.debug("Encoding %d texts (dense=%s, sparse=%s, colbert=%s, max_len=%d)",
                 len(texts), request.return_dense, request.return_sparse,
                 request.return_colbert, max_length)

    def _run() -> tuple[dict, int]:
        output = model.encode(
            texts,
            batch_size=settings.batch_size,
            max_length=max_length,
            return_dense=request.return_dense,
            return_sparse=request.return_sparse,
            return_colbert_vecs=request.return_colbert,
        )
        tokens = count_tokens(getattr(model, "tokenizer", None), texts, max_length)
        return output, tokens

    async with inference_lock:
        t0 = time.monotonic()
        output, total_tokens = await asyncio.to_thread(_run)
        elapsed = time.monotonic() - t0
        maybe_clear_mps_cache(settings.device, settings.mps_cache_threshold_mb)

    log = logger.warning if elapsed * 1000 >= settings.slow_request_ms else logger.debug
    log("Encoded %d texts (%d tokens) in %.3f s", len(texts), total_tokens, elapsed)

    dense_vecs = output.get("dense_vecs")
    lexical_weights = output.get("lexical_weights")
    colbert_vecs = output.get("colbert_vecs")

    if dense_vecs is not None and request.dimensions is not None and dense_dim is None:
        actual = len(dense_vecs[0].tolist())
        if request.dimensions != actual:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"{settings.model} only returns {actual}-dimensional vectors",
            )

    readable_sparse = None
    if lexical_weights is not None:
        if not isinstance(lexical_weights, list):
            lexical_weights = [lexical_weights]
        if not request.compact:
            readable_sparse = model.convert_id_to_token(lexical_weights)
            if isinstance(readable_sparse, dict):
                readable_sparse = [readable_sparse]

    data: list[EmbeddingItem] = []
    for i in range(len(texts)):
        item = EmbeddingItem(index=i)

        if dense_vecs is not None:
            dense_embedding = _encode_dense(dense_vecs[i], request.encoding_format)
            item.embedding = dense_embedding
            if not request.compact:
                item.dense_embedding = dense_embedding

        if lexical_weights is not None:
            item.sparse_embedding = readable_sparse[i] if readable_sparse else None
            item.sparse_qdrant = _sparse_to_qdrant(lexical_weights[i])

        if colbert_vecs is not None:
            item.colbert_embedding = colbert_vecs[i].tolist()

        data.append(item)

    response = EmbeddingResponse(
        data=data,
        model=settings.model,
        usage=UsageInfo(prompt_tokens=total_tokens, total_tokens=total_tokens),
    )
    if request.compact:
        return JSONResponse(content=response.model_dump(exclude_none=True))
    return response
