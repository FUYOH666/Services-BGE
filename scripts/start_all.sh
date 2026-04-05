#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."

PIDFILE_DIR="/tmp/services-bge"
mkdir -p "$PIDFILE_DIR"

if [ -f ".env" ]; then
    set -a
    # shellcheck disable=SC1091
    source ".env"
    set +a
fi

EMBED_PORT="${EMBEDDING_PORT:-9001}"
RERANK_PORT="${RERANKER_PORT:-9002}"

ensure_port_available() {
    local port="$1"
    local pids
    pids="$(lsof -ti :"$port" 2>/dev/null || true)"
    if [ -n "$pids" ]; then
        echo "ERROR: Port $port is already in use by PID(s): $pids"
        echo "Refusing to kill unrelated processes automatically."
        exit 1
    fi
}

for port in "$EMBED_PORT" "$RERANK_PORT"; do
    ensure_port_available "$port"
done

echo "=== Starting BGE Services ==="

echo "[1/2] Starting Embedding Service (port $EMBED_PORT)..."
"$PWD/scripts/run_service.sh" embedding &
echo $! > "$PIDFILE_DIR/embedding.pid"

echo "[2/2] Starting Reranker Service (port $RERANK_PORT)..."
"$PWD/scripts/run_service.sh" reranker &
echo $! > "$PIDFILE_DIR/reranker.pid"

echo ""
echo "PIDs:"
echo "  embedding = $(cat "$PIDFILE_DIR/embedding.pid")"
echo "  reranker  = $(cat "$PIDFILE_DIR/reranker.pid")"
echo ""
echo "Waiting for models to load..."
echo "Check health:"
echo "  curl http://localhost:$EMBED_PORT/readyz"
echo "  curl http://localhost:$RERANK_PORT/readyz"
echo ""
echo "Stop with: ./scripts/stop_all.sh"

wait
