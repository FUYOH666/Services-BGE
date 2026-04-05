#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")/.."

if [ -f ".env" ]; then
    set -a
    # shellcheck disable=SC1091
    source ".env"
    set +a
fi

EMBED_PORT="${EMBEDDING_PORT:-9001}"
RERANK_PORT="${RERANKER_PORT:-9002}"
PIDFILE_DIR="/tmp/services-bge"

echo "=== Stopping BGE Services ==="

for svc in embedding reranker; do
    pidfile="$PIDFILE_DIR/$svc.pid"
    if [ -f "$pidfile" ]; then
        pid=$(cat "$pidfile")
        if kill -0 "$pid" 2>/dev/null; then
            echo "Stopping $svc (PID $pid)..."
            kill "$pid"
        else
            echo "$svc (PID $pid) already stopped"
        fi
        rm -f "$pidfile"
    fi
done

for port in "$EMBED_PORT" "$RERANK_PORT"; do
    pids=$(lsof -ti :"$port" 2>/dev/null || true)
    if [ -n "$pids" ]; then
        echo "Warning: port $port is still in use by PID(s): $pids"
        echo "Leaving that process untouched."
    fi
done

echo "Done."
