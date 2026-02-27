#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."

echo "Starting BGE-M3 Embedding Service on port ${EMBEDDING_PORT:-9001}..."
exec uv run uvicorn embedding_service.main:app \
    --host "${EMBEDDING_HOST:-0.0.0.0}" \
    --port "${EMBEDDING_PORT:-9001}" \
    --workers "${EMBEDDING_WORKERS:-1}" \
    --log-level "${LOG_LEVEL:-info}"
