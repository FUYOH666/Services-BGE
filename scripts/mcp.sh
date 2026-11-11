#!/bin/bash
# Start the MCP memory server over stdio. Point an agent's MCP config at this file:
# it needs no arguments, works from any working directory and finds uv even when
# the agent is a GUI app started with a minimal PATH. Settings come from .env (MCP_*).
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_DIR"

export PATH="$HOME/.local/bin:$HOME/.cargo/bin:/opt/homebrew/bin:/usr/local/bin:$PATH"
export PYTHONUNBUFFERED=1

# stdout carries the MCP protocol; uv and the server log to stderr only.
exec uv run --quiet services-bge-mcp "$@"
