#!/bin/bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_DIR"

if [ -f "$PROJECT_DIR/.env" ]; then
    set -a
    # shellcheck disable=SC1091
    source "$PROJECT_DIR/.env"
    set +a
fi

export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"

service="${1:-}"

case "$service" in
    embedding)
        module="embedding_service.main:app"
        host="${EMBEDDING_HOST:-0.0.0.0}"
        port="${EMBEDDING_PORT:-9001}"
        workers="${EMBEDDING_WORKERS:-1}"
        log_level="${EMBEDDING_LOG_LEVEL:-INFO}"
        ;;
    reranker)
        module="reranker_service.main:app"
        host="${RERANKER_HOST:-0.0.0.0}"
        port="${RERANKER_PORT:-9002}"
        workers="${RERANKER_WORKERS:-1}"
        log_level="${RERANKER_LOG_LEVEL:-INFO}"
        ;;
    *)
        echo "Usage: $(basename "$0") <embedding|reranker>"
        exit 1
        ;;
esac

log_level="$(printf '%s' "$log_level" | tr '[:upper:]' '[:lower:]')"

echo "Starting $service on $host:$port (workers=$workers, log_level=$log_level)"
exec uv run uvicorn "$module" \
    --host "$host" \
    --port "$port" \
    --workers "$workers" \
    --log-level "$log_level"
