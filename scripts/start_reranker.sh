#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."

echo "Starting BGE-Reranker-v2-m3 Service on port ${RERANKER_PORT:-9002}..."
exec uv run uvicorn reranker_service.main:app \
    --host "${RERANKER_HOST:-0.0.0.0}" \
    --port "${RERANKER_PORT:-9002}" \
    --workers "${RERANKER_WORKERS:-1}" \
    --log-level "${LOG_LEVEL:-info}"
