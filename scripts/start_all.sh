#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."

EMBED_PORT="${EMBEDDING_PORT:-9001}"
RERANK_PORT="${RERANKER_PORT:-9002}"
PIDFILE_DIR="/tmp/services-bge"
mkdir -p "$PIDFILE_DIR"

# Kill stale processes on target ports
for port in "$EMBED_PORT" "$RERANK_PORT"; do
    pids=$(lsof -ti :"$port" 2>/dev/null || true)
    if [ -n "$pids" ]; then
        echo "Killing stale process on port $port (PIDs: $pids)..."
        echo "$pids" | xargs kill -9 2>/dev/null || true
        sleep 1
    fi
done

echo "=== Starting BGE Services ==="

echo "[1/2] Starting Embedding Service (port $EMBED_PORT)..."
uv run uvicorn embedding_service.main:app \
    --host "${EMBEDDING_HOST:-0.0.0.0}" \
    --port "$EMBED_PORT" \
    --workers "${EMBEDDING_WORKERS:-1}" \
    --log-level "${LOG_LEVEL:-info}" &
echo $! > "$PIDFILE_DIR/embedding.pid"

echo "[2/2] Starting Reranker Service (port $RERANK_PORT)..."
uv run uvicorn reranker_service.main:app \
    --host "${RERANKER_HOST:-0.0.0.0}" \
    --port "$RERANK_PORT" \
    --workers "${RERANKER_WORKERS:-1}" \
    --log-level "${LOG_LEVEL:-info}" &
echo $! > "$PIDFILE_DIR/reranker.pid"

echo ""
echo "PIDs:"
echo "  embedding = $(cat "$PIDFILE_DIR/embedding.pid")"
echo "  reranker  = $(cat "$PIDFILE_DIR/reranker.pid")"
echo ""
echo "Waiting for models to load..."
echo "Check health:"
echo "  curl http://localhost:$EMBED_PORT/healthz"
echo "  curl http://localhost:$RERANK_PORT/healthz"
echo ""
echo "Stop with: ./scripts/stop_all.sh"

wait
