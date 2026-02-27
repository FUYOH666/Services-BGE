from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException

from reranker_service.config import settings
from reranker_service.models import (
    HealthResponse,
    RerankRequest,
    RerankResponse,
    RerankResult,
    RerankUsage,
)

logging.basicConfig(
    level=settings.log_level,
    format="%(asctime)s · %(levelname)s · reranker · %(message)s",
)
logger = logging.getLogger(__name__)

_state: dict[str, Any] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Loading model %s (device=%s, fp16=%s)", settings.model, settings.device, settings.use_fp16)
    t0 = time.monotonic()

    from FlagEmbedding import FlagReranker

    kwargs: dict[str, Any] = {"use_fp16": settings.use_fp16}
    if settings.device == "mps":
        kwargs["devices"] = ["mps"]

    _state["model"] = FlagReranker(settings.model, **kwargs)
    _state["ready"] = True

    elapsed = time.monotonic() - t0
    logger.info("Model loaded in %.1f s", elapsed)

    yield

    _state.clear()
    logger.info("Model unloaded")


app = FastAPI(
    title="BGE-Reranker-v2-m3 Service",
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


@app.post("/v1/rerank", response_model=RerankResponse)
async def rerank(request: RerankRequest):
    if not _state.get("ready"):
        raise HTTPException(status_code=503, detail="Model not ready")

    model = _state["model"]
    pairs = [[request.query, doc] for doc in request.documents]

    logger.info("Reranking %d pairs (normalize=%s)", len(pairs), request.normalize)

    t0 = time.monotonic()
    scores = model.compute_score(pairs, normalize=request.normalize)
    elapsed = time.monotonic() - t0

    if isinstance(scores, (int, float)):
        scores = [scores]

    logger.info("Reranked %d pairs in %.3f s (%.1f pairs/s)", len(pairs), elapsed,
                len(pairs) / elapsed if elapsed > 0 else 0)

    results = [
        RerankResult(index=i, document=request.documents[i], relevance_score=float(s))
        for i, s in enumerate(scores)
    ]
    results.sort(key=lambda r: r.relevance_score, reverse=True)

    if request.top_n is not None:
        results = results[: request.top_n]

    return RerankResponse(
        results=results,
        model=settings.model,
        usage=RerankUsage(total_pairs=len(pairs)),
    )
