#!/bin/bash
# Pull the team canon (read-only) and refresh the shared memory index once.
# Used by the launchd agent com.services-bge.memory-sync (./scripts/service.sh memory-sync on)
# and safe to run by hand. Settings come from .env (MCP_*).
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_DIR"

export PATH="$HOME/.local/bin:$HOME/.cargo/bin:/opt/homebrew/bin:/usr/local/bin:$PATH"
export PYTHONUNBUFFERED=1

echo "=== memory sync $(date '+%Y-%m-%d %H:%M:%S') ==="
exec uv run --quiet services-bge-mcp sync
