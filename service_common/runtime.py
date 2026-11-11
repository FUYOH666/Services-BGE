"""Runtime helpers shared by both services: auth, MPS cache policy, limits."""
from __future__ import annotations

import logging
import secrets
from collections.abc import Callable, Sequence

from fastapi import Header, HTTPException, status

logger = logging.getLogger(__name__)

_MB = 1024 * 1024


def make_api_key_dependency(get_key: Callable[[], str | None]):
    """Return a FastAPI dependency that enforces an optional shared API key.

    When no key is configured the dependency is a no-op, so existing local
    clients keep working. When a key is set, either `Authorization: Bearer <key>`
    or `X-API-Key: <key>` is accepted.
    """

    async def require_api_key(
        authorization: str | None = Header(default=None),
        x_api_key: str | None = Header(default=None),
    ) -> None:
        expected = get_key()
        if not expected:
            return

        supplied = x_api_key
        if authorization and authorization.lower().startswith("bearer "):
            supplied = authorization[7:].strip()

        if not supplied or not secrets.compare_digest(supplied, expected):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or missing API key",
                headers={"WWW-Authenticate": "Bearer"},
            )

    return require_api_key


def maybe_clear_mps_cache(device: str, threshold_mb: int) -> None:
    """Release cached MPS allocator blocks.

    `threshold_mb <= 0` keeps the historical behaviour (clear after every
    request). A positive value only clears once the Metal driver holds more
    than that many megabytes, which avoids paying the cost on every call.
    """
    if device != "mps":
        return

    import torch

    if threshold_mb > 0:
        allocated = torch.mps.driver_allocated_memory()
        if allocated < threshold_mb * _MB:
            return
        logger.debug("MPS driver memory %.0f MB >= %d MB, clearing cache", allocated / _MB, threshold_mb)

    torch.mps.empty_cache()


def enforce_text_limits(texts: Sequence[str], *, max_items: int, max_chars: int, what: str) -> None:
    """Reject oversized requests with 413 before they reach the model."""
    if max_items > 0 and len(texts) > max_items:
        raise HTTPException(
            status_code=413,
            detail=f"Too many {what}: {len(texts)} > limit {max_items}",
        )
    if max_chars > 0:
        for i, text in enumerate(texts):
            if len(text) > max_chars:
                raise HTTPException(
                    status_code=413,
                    detail=f"{what[:-1].capitalize()} {i} is {len(text)} characters, limit is {max_chars}",
                )


def resolve_max_length(requested: int | None, default: int, cap: int) -> int:
    """Pick the effective token limit and refuse values above the configured cap."""
    value = default if requested is None else requested
    if cap > 0 and value > cap:
        raise HTTPException(
            status_code=422,
            detail=f"max_length {value} exceeds the configured cap {cap}",
        )
    return value


def resolve_model_path(model: str, revision: str | None) -> str:
    """Return a local snapshot path when a revision is pinned, else the model id.

    Pinning a revision makes restarts reproducible and lets the service start
    with `HF_HUB_OFFLINE=1` from the local cache.
    """
    if not revision:
        return model

    from huggingface_hub import snapshot_download

    path = snapshot_download(repo_id=model, revision=revision)
    logger.info("Resolved %s@%s to %s", model, revision, path)
    return path


def count_tokens(tokenizer, texts: list[str], max_length: int) -> int:
    """Count tokens the model actually sees (after truncation). 0 if unknown."""
    if tokenizer is None:
        return 0
    try:
        encoded = tokenizer(texts, truncation=True, max_length=max_length)
        return sum(len(ids) for ids in encoded["input_ids"])
    except Exception:  # token accounting must never break inference
        logger.debug("Token counting failed", exc_info=True)
        return 0
