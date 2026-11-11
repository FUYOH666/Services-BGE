from __future__ import annotations

import asyncio
import base64
import logging
import time
from contextlib import asynccontextmanager
from dataclasses import asdict
from typing import Any

import numpy as np
from fastapi import Depends, FastAPI, HTTPException, Response, status
from fastapi.responses import JSONResponse

from embedding_service.backends import load_backend
from embedding_service.config import settings
from embedding_service.models import (
    EmbeddingItem,
    EmbeddingRequest,
    EmbeddingResponse,
    HealthResponse,
    ModelCard,
    ModelList,
    SparseQdrant,
    UsageInfo,
)
from service_common.runtime import (
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


def _warmup(backend: Any) -> int | None:
    """Run one small inference so the first real request is not slow.

    Returns the native dense size, used to validate the OpenAI `dimensions` field.
    """
    output = backend.encode(
        ["warmup"],
        max_length=32,
        return_dense=True,
        return_sparse=backend.capabilities.sparse,
        return_colbert=False,
        readable_sparse=False,
    )
    maybe_clear_mps_cache(settings.device, settings.mps_cache_threshold_mb)
    if output.dense is None:
        return None
    return len(output.dense[0].tolist())


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(
        "Loading model %s (backend=%s, device=%s, fp16=%s)",
        settings.model, settings.backend, settings.device, settings.use_fp16,
    )
    t0 = time.monotonic()
    _state["ready"] = False
    _state["inference_lock"] = asyncio.Lock()

    try:
        model_path = resolve_model_path(settings.model, settings.model_revision)
        backend = load_backend(settings, model_path)
        _state["backend"] = backend
        if settings.warmup:
            _state["dense_dim"] = _warmup(backend)
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
    title="Services-BGE · Embedding",
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


@app.get("/v1/models", response_model=ModelList)
async def list_models():
    backend = _state.get("backend")
    if backend is None:
        raise HTTPException(status_code=503, detail="Model not ready")
    card = ModelCard(
        id=settings.model,
        backend=settings.backend,
        capabilities=asdict(backend.capabilities),
        dimensions=_state.get("dense_dim"),
        max_length=settings.max_length,
        max_length_cap=settings.max_length_cap,
        device=settings.device,
        fp16=settings.use_fp16,
    )
    return ModelList(data=[card])


@app.post("/v1/embeddings", response_model=EmbeddingResponse, dependencies=[Depends(require_api_key)])
async def create_embeddings(request: EmbeddingRequest):
    if not _state.get("ready"):
        raise HTTPException(status_code=503, detail="Model not ready")

    backend = _state["backend"]
    caps = backend.capabilities
    texts = request.input
    enforce_text_limits(texts, max_items=settings.max_items, max_chars=settings.max_text_chars, what="texts")
    max_length = resolve_max_length(request.max_length, settings.max_length, settings.max_length_cap)

    # Defaults ask for sparse output; dense-only backends skip it silently, but an
    # explicit request for an output the backend cannot produce is an error.
    explicit = request.model_fields_set
    for flag, supported, label in (
        ("return_sparse", caps.sparse, "sparse"),
        ("return_colbert", caps.colbert, "ColBERT"),
    ):
        if getattr(request, flag) and not supported and flag in explicit:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"{settings.model} ({settings.backend}) does not produce {label} vectors",
            )
    return_sparse = request.return_sparse and caps.sparse
    return_colbert = request.return_colbert and caps.colbert

    dense_dim = _state.get("dense_dim")
    dimensions = request.dimensions
    if dimensions is not None and dense_dim is not None:
        if dimensions == dense_dim:
            dimensions = None
        elif not caps.matryoshka or dimensions > dense_dim:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"{settings.model} only returns {dense_dim}-dimensional vectors",
            )

    inference_lock: asyncio.Lock = _state["inference_lock"]
    logger.debug("Encoding %d texts (dense=%s, sparse=%s, colbert=%s, max_len=%d, input_type=%s)",
                 len(texts), request.return_dense, return_sparse, return_colbert, max_length, request.input_type)

    async with inference_lock:
        t0 = time.monotonic()
        output = await asyncio.to_thread(
            backend.encode,
            texts,
            max_length=max_length,
            return_dense=request.return_dense,
            return_sparse=return_sparse,
            return_colbert=return_colbert,
            readable_sparse=not request.compact,
            input_type=request.input_type,
            dimensions=dimensions,
        )
        elapsed = time.monotonic() - t0
        maybe_clear_mps_cache(settings.device, settings.mps_cache_threshold_mb)

    log = logger.warning if elapsed * 1000 >= settings.slow_request_ms else logger.debug
    log("Encoded %d texts (%d tokens) in %.3f s", len(texts), output.tokens, elapsed)

    if output.dense is not None and request.dimensions is not None and dense_dim is None:
        actual = len(output.dense[0].tolist())
        if request.dimensions != actual:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"{settings.model} only returns {actual}-dimensional vectors",
            )

    data: list[EmbeddingItem] = []
    for i in range(len(texts)):
        item = EmbeddingItem(index=i)

        if output.dense is not None:
            dense_embedding = _encode_dense(output.dense[i], request.encoding_format)
            item.embedding = dense_embedding
            if not request.compact:
                item.dense_embedding = dense_embedding

        if output.sparse is not None:
            item.sparse_embedding = output.sparse_readable[i] if output.sparse_readable else None
            item.sparse_qdrant = _sparse_to_qdrant(output.sparse[i])

        if output.colbert is not None:
            item.colbert_embedding = output.colbert[i].tolist()

        data.append(item)

    response = EmbeddingResponse(
        data=data,
        model=settings.model,
        usage=UsageInfo(prompt_tokens=output.tokens, total_tokens=output.tokens),
    )
    if request.compact:
        return JSONResponse(content=response.model_dump(exclude_none=True))
    return response
