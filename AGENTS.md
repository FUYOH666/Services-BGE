# AGENTS.md

Guidance for AI coding agents (Claude Code, Codex, Cursor, Copilot, …) working in this repository. Humans: see [CONTRIBUTING.md](CONTRIBUTING.md).

## What this is

Services-BGE: local embedding and reranking APIs for hybrid search on Apple Silicon, plus tools to compare models. Two FastAPI processes, one model each:

- `embedding_service/` — `POST /v1/embeddings` on `127.0.0.1:9001` (production model BAAI/bge-m3: dense + sparse + ColBERT).
- `reranker_service/` — `POST /v1/rerank` on `127.0.0.1:9002` (production model BAAI/bge-reranker-v2-m3).

Both start at login as launchd agents (`launchd/*.plist`, installed by `scripts/service.sh install`). About eight local projects call them, so the HTTP contract matters more than anything else here.

A third process, `mcp_server/` (`services-bge-mcp`), is an MCP server that gives AI agents long-term memory on top of the two services and Qdrant (`127.0.0.1:6333`): stdio via `scripts/mcp.sh`, or streamable HTTP on `127.0.0.1:9003`. It loads no model.

## Layout

| Path | What |
|---|---|
| `embedding_service/main.py`, `reranker_service/main.py` | HTTP layer: validation, limits, auth, response shape |
| `embedding_service/backends.py`, `reranker_service/backends.py` | Model backends (`bge-m3`, `sentence-transformers`; `flag-reranker`, `cross-encoder`, `qwen3-reranker`) |
| `*/config.py` | pydantic-settings, env prefix `EMBEDDING_` / `RERANKER_` |
| `service_common/runtime.py` | Shared: API key, limits, MPS cache policy, model path resolution, token counting |
| `profiles/*.env` | One file per model deployment; `scripts/sidecar.sh` runs them on 9011+ |
| `mcp_server/` | MCP memory server: `server.py` (tools, transports), `memory.py` (Qdrant logic), `chunking.py`, `services.py` (HTTP clients), `config.py` (`MCP_*`) |
| `scripts/run_service.sh` | Entry point for launchd and sidecars (`.env`, profile, offline mode) |
| `scripts/parity_check.py` | Compares two deployments (vectors, rerank order, latency, memory) |
| `eval/` | Retrieval eval (`run_eval.py`), comparison table (`compare.py`), agent memory bench (`memory_bench.py`), datasets, results; `eval/private/` is gitignored |
| `tests/test_service_endpoints.py` | Contract and behaviour tests with fake models |
| `tests/test_mcp_server.py` | MCP tools with fake services and in-memory Qdrant |
| `tests/test_memory_bench.py` | Memory bench on the synthetic set, offline |
| `docs/` | API, configuration, models, operations, roadmap, brand assets |

## Commands

```bash
uv sync --extra dev                       # Python 3.12 env
uv run --extra dev pytest -q              # fast, never downloads models
uv lock --check
bash -n scripts/*.sh                      # shell syntax
./scripts/sidecar.sh list                 # profiles and their state
uv run python scripts/parity_check.py     # A = 9001/9002 vs B = 9101/9102
uv run python eval/run_eval.py --dataset eval/datasets/demo_ru --name <label>
uv run python scripts/mcp_smoke.py       # MCP round trip on live services + Qdrant, scratch collection
uv run python eval/memory_bench.py --bench eval/datasets/memory_synthetic   # memory retrieval + poisoning, scratch collection
```

CI (`.github/workflows/test.yml`) runs pytest on Ubuntu with only the web stack installed, plus a secret and personal-path scan.

## Rules

- **The default response shape is a contract.** Never remove or rename `embedding`, `dense_embedding`, `sparse_embedding`, `sparse_qdrant`, `colbert_embedding`, `index`, `relevance_score`, `document`, `usage`. New behaviour must be opt-in. The contract tests must stay green.
- **Do not stop or restart the user's running services** (launchd labels `com.services-bge.embedding`, `com.services-bge.reranker`, ports 9001/9002). Test changes on side ports: `EMBEDDING_PORT=9101 ./scripts/run_service.sh embedding`, then `scripts/parity_check.py`.
- Tests must not download models or reach the network: stub `FlagEmbedding`, `sentence_transformers` and friends via `sys.modules` like the existing fixtures.
- Keep the default bind address `127.0.0.1`.
- MCP server: only touch Qdrant collections with `MCP_COLLECTION_PREFIX` (`mcp_`); never write to stdout on the stdio transport; destructive tools need an explicit target. Smoke tests use a scratch collection and delete it.
- Trust tiers come from where text was read (`MCP_CANON_PATHS`), never from the caller; keep `remember` without a tier argument. Private golden sets and their results live in `eval/private/` only.
- New models go in as a profile first. Only mark a profile *verified* in `docs/MODELS.md` after it loaded and scored on real hardware.
- Model or precision changes need evidence: `parity_check.py` for drift, `eval/run_eval.py` for quality.
- No absolute personal paths, IPs or secrets in tracked files (CI greps for them). Use `~` or `$HOME`.
- Keep `CHANGELOG.md` updated for user-visible changes.
