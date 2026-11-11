from __future__ import annotations

import asyncio
import logging
import time
from contextlib import asynccontextmanager
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Response, status
from fastapi.responses import JSONResponse

from reranker_service.backends import load_backend
from reranker_service.config import settings
from reranker_service.models import (
    HealthResponse,
    ModelCard,
    ModelList,
    RerankRequest,
    RerankResponse,
    RerankResult,
    RerankUsage,
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
    format="%(asctime)s · %(levelname)s · reranker · %(message)s",
)
logger = logging.getLogger(__name__)

_state: dict[str, Any] = {}

require_api_key = make_api_key_dependency(lambda: settings.api_key)


def _health_response(*, ready: bool, status_text: str) -> HealthResponse:
    return HealthResponse(
        status=status_text,
        model=settings.model,
        device=settings.device,
        ready=ready,
    )


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
            backend.score([["warmup", "warmup"]], max_length=32, normalize=True)
            maybe_clear_mps_cache(settings.device, settings.mps_cache_threshold_mb)
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
    title="Services-BGE · Reranker",
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
        instruction=getattr(backend, "instruction", None),
        max_length=settings.max_length,
        max_length_cap=settings.max_length_cap,
        device=settings.device,
        fp16=settings.use_fp16,
    )
    return ModelList(data=[card])


@app.post("/v1/rerank", response_model=RerankResponse, dependencies=[Depends(require_api_key)])
async def rerank(request: RerankRequest):
    if not _state.get("ready"):
        raise HTTPException(status_code=503, detail="Model not ready")

    enforce_text_limits(
        request.documents,
        max_items=settings.max_documents,
        max_chars=settings.max_text_chars,
        what="documents",
    )
    enforce_text_limits([request.query], max_items=0, max_chars=settings.max_text_chars, what="queries")
    max_length = resolve_max_length(request.max_length, settings.max_length, settings.max_length_cap)

    backend = _state["backend"]
    if request.instruction and not backend.supports_instruction:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"{settings.model} ({settings.backend}) does not take an instruction",
        )
    inference_lock: asyncio.Lock = _state["inference_lock"]
    pairs = [[request.query, doc] for doc in request.documents]

    logger.debug("Reranking %d pairs (normalize=%s, max_len=%d)", len(pairs), request.normalize, max_length)

    async with inference_lock:
        t0 = time.monotonic()
        scores = await asyncio.to_thread(
            backend.score,
            pairs,
            max_length=max_length,
            normalize=request.normalize,
            instruction=request.instruction,
        )
        elapsed = time.monotonic() - t0
        maybe_clear_mps_cache(settings.device, settings.mps_cache_threshold_mb)

    log = logger.warning if elapsed * 1000 >= settings.slow_request_ms else logger.debug
    log("Reranked %d pairs in %.3f s", len(pairs), elapsed)

    results = [
        RerankResult(
            index=i,
            document=request.documents[i] if request.return_documents else None,
            relevance_score=float(s),
        )
        for i, s in enumerate(scores)
    ]
    results.sort(key=lambda r: r.relevance_score, reverse=True)

    if request.top_n is not None:
        results = results[: request.top_n]

    response = RerankResponse(
        results=results,
        model=settings.model,
        usage=RerankUsage(total_pairs=len(pairs)),
    )
    if not request.return_documents:
        return JSONResponse(content=response.model_dump(exclude_none=True))
    return response
