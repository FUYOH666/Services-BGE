#!/bin/bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_DIR"

usage() {
    echo "Usage: $(basename "$0") <embedding|reranker>"
    echo "       $(basename "$0") --profile <name>     (profiles/<name>.env, see profiles/README.md)"
    exit 1
}

service=""
profile=""
case "${1:-}" in
    --profile)
        profile="${2:-}"
        [ -n "$profile" ] || usage
        ;;
    embedding|reranker)
        service="$1"
        ;;
    *)
        usage
        ;;
esac

# Precedence: variables passed by the caller > profile > .env > built-in defaults.
caller_env="$(export -p)"

if [ -f "$PROJECT_DIR/.env" ]; then
    set -a
    # shellcheck disable=SC1091
    source "$PROJECT_DIR/.env"
    set +a
fi

if [ -n "$profile" ]; then
    profile_file="$PROJECT_DIR/profiles/$profile.env"
    if [ ! -f "$profile_file" ]; then
        echo "Unknown profile '$profile' (no $profile_file)"
        exit 1
    fi
    set -a
    # shellcheck disable=SC1090
    source "$profile_file"
    set +a
    service="${PROFILE_SERVICE:-}"
    if [ "$service" != "embedding" ] && [ "$service" != "reranker" ]; then
        echo "Profile '$profile' must set PROFILE_SERVICE=embedding or PROFILE_SERVICE=reranker"
        exit 1
    fi
fi

eval "$caller_env"

export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"

case "$service" in
    embedding)
        module="embedding_service.main:app"
        host="${EMBEDDING_HOST:-127.0.0.1}"
        port="${EMBEDDING_PORT:-9001}"
        workers="${EMBEDDING_WORKERS:-1}"
        log_level="${EMBEDDING_LOG_LEVEL:-INFO}"
        access_log="${EMBEDDING_ACCESS_LOG:-true}"
        model="${EMBEDDING_MODEL:-BAAI/bge-m3}"
        revision="${EMBEDDING_MODEL_REVISION:-}"
        ;;
    reranker)
        module="reranker_service.main:app"
        host="${RERANKER_HOST:-127.0.0.1}"
        port="${RERANKER_PORT:-9002}"
        workers="${RERANKER_WORKERS:-1}"
        log_level="${RERANKER_LOG_LEVEL:-INFO}"
        access_log="${RERANKER_ACCESS_LOG:-true}"
        model="${RERANKER_MODEL:-BAAI/bge-reranker-v2-m3}"
        revision="${RERANKER_MODEL_REVISION:-}"
        ;;
esac

# Start offline when the model is already in the local Hugging Face cache, so a
# boot without network does not stall on huggingface.co retries. An explicit
# HF_HUB_OFFLINE (0 or 1) in the environment or .env always wins.
if [ -z "${HF_HUB_OFFLINE:-}" ]; then
    if uv run --no-sync python - "$model" "$revision" <<'PY' 2>/dev/null
import os
import sys

from huggingface_hub import try_to_load_from_cache

model, revision = sys.argv[1], sys.argv[2] or None
if os.path.isdir(model):
    sys.exit(0)
cached = try_to_load_from_cache(model, "config.json", revision=revision)
sys.exit(0 if isinstance(cached, str) else 1)
PY
    then
        export HF_HUB_OFFLINE=1
    fi
fi

log_level="$(printf '%s' "$log_level" | tr '[:upper:]' '[:lower:]')"
access_flag="--access-log"
case "$(printf '%s' "$access_log" | tr '[:upper:]' '[:lower:]')" in
    false|0|no|off) access_flag="--no-access-log" ;;
esac

echo "Starting $service${profile:+ (profile $profile)} on $host:$port (workers=$workers, log_level=$log_level, hf_offline=${HF_HUB_OFFLINE:-0})"
exec uv run uvicorn "$module" \
    --host "$host" \
    --port "$port" \
    --workers "$workers" \
    --log-level "$log_level" \
    "$access_flag"
