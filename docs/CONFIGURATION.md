# Configuration

Settings come from environment variables. Precedence, highest first:

1. variables set on the command line (`EMBEDDING_PORT=9101 ./scripts/run_service.sh embedding`)
2. a profile, when started with `--profile <name>` or `scripts/sidecar.sh` (`profiles/<name>.env`)
3. `.env` in the project root (copy `.env.example`)
4. built-in defaults (`embedding_service/config.py`, `reranker_service/config.py`)

After editing `.env`, run `./scripts/service.sh restart`.

## Embedding service (`EMBEDDING_*`)

| Variable | Default | Meaning |
|---|---|---|
| `EMBEDDING_HOST` | `127.0.0.1` | Bind address. `0.0.0.0` exposes the service on LAN/Tailscale — set an API key too |
| `EMBEDDING_PORT` | `9001` | |
| `EMBEDDING_BACKEND` | `bge-m3` | `bge-m3` or `sentence-transformers` ([MODELS.md](MODELS.md)) |
| `EMBEDDING_MODEL` | `BAAI/bge-m3` | Hugging Face id or local path |
| `EMBEDDING_MODEL_REVISION` | — | Commit sha or tag; makes restarts reproducible |
| `EMBEDDING_DEVICE` | `mps` | `mps`, `cpu` or `cuda` |
| `EMBEDDING_USE_FP16` | `false` | Half precision. Saves ~0.6 GB on bge-m3; measure first with `scripts/parity_check.py` |
| `EMBEDDING_MAX_LENGTH` | `512` | Default tokens per text when a request sends no `max_length` |
| `EMBEDDING_MAX_LENGTH_CAP` | `8192` | Highest `max_length` a request may ask for |
| `EMBEDDING_BATCH_SIZE` | `12` | Texts per forward pass |
| `EMBEDDING_MAX_ITEMS` | `256` | Texts per request (`413` above). `0` disables |
| `EMBEDDING_MAX_TEXT_CHARS` | `100000` | Characters per text (`413` above). `0` disables |
| `EMBEDDING_QUERY_PROMPT` | — | sentence-transformers: literal prefix for `input_type: "query"` (`\n` allowed) |
| `EMBEDDING_QUERY_PROMPT_NAME` | — | …or the name of a prompt stored in the model's config (e.g. `query`) |
| `EMBEDDING_DOCUMENT_PROMPT` / `_NAME` | — | Same for documents (requests without `input_type` count as documents) |
| `EMBEDDING_MATRYOSHKA` | `false` | Allow `dimensions` smaller than the native size |
| `EMBEDDING_TRUST_REMOTE_CODE` | `false` | Needed by some Hugging Face models with custom code; review the code first |
| `EMBEDDING_WORKERS` | `1` | uvicorn workers. Keep `1`: each worker loads its own model copy |
| `EMBEDDING_LOG_LEVEL` | `INFO` | `DEBUG` logs every request |
| `EMBEDDING_ACCESS_LOG` | `true` | uvicorn access log line per request |
| `EMBEDDING_SLOW_REQUEST_MS` | `2000` | Requests slower than this are logged at `WARNING` |
| `EMBEDDING_WARMUP` | `true` | One inference before `readyz` turns green |
| `EMBEDDING_MPS_CACHE_THRESHOLD_MB` | `0` | `0`: empty the MPS cache after every request. `>0`: only above this much Metal driver memory |
| `EMBEDDING_API_KEY` | — | Overrides `SERVICE_API_KEY` for this service |

## Reranker service (`RERANKER_*`)

| Variable | Default | Meaning |
|---|---|---|
| `RERANKER_HOST` / `RERANKER_PORT` | `127.0.0.1` / `9002` | |
| `RERANKER_BACKEND` | `flag-reranker` | `flag-reranker`, `cross-encoder` or `qwen3-reranker` |
| `RERANKER_MODEL` | `BAAI/bge-reranker-v2-m3` | |
| `RERANKER_MODEL_REVISION` | — | |
| `RERANKER_DEVICE` / `RERANKER_USE_FP16` | `mps` / `false` | |
| `RERANKER_BATCH_SIZE` | `32` | Pairs per forward pass |
| `RERANKER_MAX_LENGTH` / `_CAP` | `512` / `8192` | Tokens per query + document |
| `RERANKER_MAX_DOCUMENTS` | `1000` | Documents per request |
| `RERANKER_MAX_TEXT_CHARS` | `100000` | Characters per query or document |
| `RERANKER_INSTRUCTION` | — | Default task instruction for `qwen3-reranker` |
| `RERANKER_TRUST_REMOTE_CODE` | `false` | |
| `RERANKER_WORKERS`, `_LOG_LEVEL`, `_ACCESS_LOG`, `_SLOW_REQUEST_MS`, `_WARMUP`, `_MPS_CACHE_THRESHOLD_MB`, `_API_KEY` | | Same as for the embedding service |

## MCP server (`MCP_*`)

The [agent memory server](AGENT-MEMORY.md) reads the environment and the project's `.env` itself (variables already set win), wherever the agent starts it from. After editing `.env`, restart the agent (stdio) or `./scripts/run_service.sh mcp` (HTTP).

| Variable | Default | Meaning |
|---|---|---|
| `MCP_TRANSPORT` | `stdio` | `stdio` or `http`; `--transport` on the command line wins |
| `MCP_HOST` / `MCP_PORT` / `MCP_PATH` | `127.0.0.1` / `9003` / `/mcp` | HTTP address. A non-loopback host is refused without `MCP_API_KEY` |
| `MCP_API_KEY` | — | Required from HTTP clients when set (Bearer or `X-API-Key`); `/healthz` stays open |
| `MCP_LOG_LEVEL` | `INFO` | Logs go to stderr |
| `MCP_EMBEDDING_URL` | `http://127.0.0.1:9001` | Embedding service (any profile with the same API, e.g. a sidecar) |
| `MCP_RERANKER_URL` | `http://127.0.0.1:9002` | Rerank service |
| `MCP_EMBEDDING_API_KEY` | `EMBEDDING_API_KEY`, then `SERVICE_API_KEY` | Key sent to the embedding service |
| `MCP_RERANKER_API_KEY` | `RERANKER_API_KEY`, then `SERVICE_API_KEY` | Key sent to the rerank service |
| `MCP_QDRANT_URL` | `http://127.0.0.1:6333` | Qdrant REST endpoint |
| `MCP_QDRANT_API_KEY` | — | Qdrant API key, if your Qdrant has one |
| `MCP_REQUEST_TIMEOUT` | `120` | Seconds per call to the services or Qdrant |
| `MCP_COLLECTION_PREFIX` | `mcp_` | Qdrant name prefix of memory collections; others are invisible to agents |
| `MCP_DEFAULT_COLLECTION` | `memory` | Collection used when a tool call names none |
| `MCP_CHUNK_TOKENS` / `MCP_CHUNK_OVERLAP_TOKENS` | `500` / `80` | Passage size and overlap (estimated tokens) |
| `MCP_MAX_LENGTH` | `1024` | `max_length` sent to both services; keep it above `MCP_CHUNK_TOKENS` |
| `MCP_EMBED_BATCH_SIZE` | `32` | Passages per `/v1/embeddings` call |
| `MCP_PREFETCH_LIMIT` / `MCP_RERANK_CANDIDATES` | `50` / `20` | Dense and sparse candidates before fusion; fused candidates reranked |
| `MCP_ALLOWED_PATHS` | `~` | Folders `remember(path=...)` may read, separated by `:` |
| `MCP_FILE_EXTENSIONS` | `.md,.markdown,.mdx,.txt,.rst` | File types it reads |
| `MCP_MAX_FILE_BYTES` / `MCP_MAX_FILES` / `MCP_MAX_CHUNKS` | `2000000` / `500` / `5000` | Per file / files per call / passages per call |

## Common

| Variable | Default | Meaning |
|---|---|---|
| `SERVICE_API_KEY` | — | Shared key for both services' `POST` endpoints (Bearer or `X-API-Key`) |
| `HF_HUB_OFFLINE` | auto | Unset: `run_service.sh` turns it on when the model is already cached. `0`/`1` force it |
| `TOKENIZERS_PARALLELISM` | `false` | Avoids fork warnings from Hugging Face tokenizers on macOS |
| `SIDECAR_READY_TIMEOUT` | `900` | Seconds `sidecar.sh up` waits for a model to download and load |

## Profiles

A profile (`profiles/<name>.env`) sets `PROFILE_SERVICE=embedding|reranker` plus every model-specific variable of that service, including empty ones, so nothing leaks in from `.env`. See [MODELS.md](MODELS.md#adding-a-model).
