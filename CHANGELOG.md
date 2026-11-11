# Changelog

## Unreleased

Agent memory (new, opt-in; the embedding and rerank APIs are unchanged):

- `mcp_server/`, console script `services-bge-mcp`: an MCP server (official `mcp` SDK 2.x) that gives Claude Code, Cursor, OpenClaw, Hermes and other agents shared long-term memory on top of the two services and Qdrant, over stdio (`scripts/mcp.sh`) or streamable HTTP on `127.0.0.1:9003` (`./scripts/run_service.sh mcp`)
- tools: `remember` (notes, or `.md`/`.txt` files and folders, chunked into ~500-token passages with heading paths and overlap), `search` (bge-m3 dense + sparse with RRF fusion in Qdrant `query_points`, then bge-reranker-v2-m3; falls back to fusion order when the reranker is down), `list_collections`, `list_sources`, `forget`, `rerank`, `embed`
- idempotent writes: stable point ids from source + passage number; unchanged passages are not re-embedded; a re-read file replaces its old passages
- safety: memory collections are prefixed `mcp_` so other collections stay invisible; `forget` needs an explicit `source` or `everything=true`; file reading limited to `MCP_ALLOWED_PATHS` (default `~`), text extensions and size limits; the collection records its embedding model and refuses vectors from another one; HTTP binds to loopback with DNS-rebinding protection and refuses other hosts without `MCP_API_KEY`; `SERVICE_API_KEY` is sent to the services when set
- `MCP_*` settings in `docs/CONFIGURATION.md`; setup for each agent in `docs/AGENT-MEMORY.md`; `scripts/mcp_smoke.py` checks the full loop on a scratch collection and deletes it
- dependencies: `mcp>=2.3,<3`, `qdrant-client>=1.16`; CI installs both and runs the MCP tests against fake services and in-memory Qdrant

## 0.2.0 — 2026-11-11

API (additive; default responses are unchanged and pinned by contract tests):

- embeddings accept a single string as `input` (OpenAI style) and the OpenAI `model`, `user`, `dimensions` and `encoding_format` fields; `encoding_format=base64` returns float32 base64
- each embedding item now has `object: "embedding"`; `usage.total_tokens` / `prompt_tokens` report real token counts instead of `0`
- `compact: true` drops the duplicate `dense_embedding`, token-readable `sparse_embedding` and null fields (708 KB → 353 KB, 99 KB with base64, for 16 mixed texts)
- rerank: `return_documents`, per-request `max_length`, `model` accepted

Model switching (research tool):

- pluggable backends: embeddings `bge-m3` (FlagEmbedding) and `sentence-transformers` (any dense model, query/document prompts via `input_type`, Matryoshka `dimensions`); rerankers `flag-reranker`, `cross-encoder` and `qwen3-reranker` (official yes/no-logit scoring with an `instruction`)
- `GET /v1/models` on both services: backend, capabilities, dimensions, limits
- `profiles/*.env` (bge-m3, bge-reranker-v2-m3, Qwen3-Embedding-0.6B, Qwen3-Reranker-0.6B/4B, multilingual-e5-large) and `scripts/sidecar.sh` to run challengers on 9011+ next to production; `run_service.sh --profile`
- eval: `input_type` for asymmetric models, `--sparse-url` for cross-model hybrids, `eval/compare.py` for one table across runs; demo matrix in `eval/results/COMPARISON.md`
- the default bge path stays bit-identical to the previous release (`parity_check.py`)

Fixes:

- `EMBEDDING_MAX_LENGTH` is now the default token limit (it was ignored; 512 was hard-coded)
- `RERANKER_BATCH_SIZE` is now applied (FlagEmbedding's default of 128 was used)
- rerank `top_n` must be ≥ 1 (`-1` silently dropped the last result)

Operations and security:

- default host is `127.0.0.1`; exposing on LAN/Tailscale is an explicit opt-in, ideally with the new optional `SERVICE_API_KEY` (Bearer or `X-API-Key`)
- request limits: texts per request, documents per rerank, characters per text (413) and a `max_length` cap (422)
- models are warmed up before `readyz` reports ready
- `run_service.sh` starts with `HF_HUB_OFFLINE=1` when the model is cached (no more huggingface.co retries on a boot without network); optional `*_MODEL_REVISION` pin; variables passed on the command line win over `.env`
- per-request inference logs moved to `DEBUG`, slow requests logged at `WARNING`, configurable uvicorn access log, `service.sh rotate-logs`
- optional `*_MPS_CACHE_THRESHOLD_MB` (default keeps clearing after every request)

Tooling and docs:

- `scripts/parity_check.py`: compare two deployments (vector drift, rerank order, latency, response size, memory); fp32 branch vs live was bit-identical; fp16 measurements recorded in `docs/ROADMAP.md`
- `eval/run_eval.py` + `eval/datasets/demo_ru`: retrieval eval (recall, nDCG, MRR, HS-prefix accuracy) for any endpoints; demo baseline in `eval/results/`
- `scripts/test_services.py` takes `--embed-url` / `--rerank-url`
- GitHub Actions runs the test suite; `psutil` added to dev extras
- `docs/ROADMAP.md` (first written 2026-09-10) committed and updated with the November 2026 model landscape, quantization rules and the replacement gate
- documentation rewritten to the repository canon shared with VoiceToText: light/dark SVG banner and pipeline diagram, social preview, badges, `README.ru.md`, `docs/API.md`, `docs/CONFIGURATION.md`, `docs/MODELS.md`, `docs/OPERATIONS.md`, `AGENTS.md`, `llms.txt`, `CONTRIBUTING.md`, `CITATION.cff`, issue and PR templates; CI adds secret/path scans, lockfile and shell checks
- corrected memory figures (~7.4 GB for both services); the old cover art moved to `docs/assets/cover-art.png`

## 2026-08-21

- embedding response now includes the OpenAI-compatible `embedding` field (same values as `dense_embedding`, which is kept for existing hybrid-search clients)

## 2026-07-17

- added project cover image at `assets/services-bge-cover.png` and show it at the top of `README.md`

## 2026-04-06

- added MIT `LICENSE`, maintainer contacts in `README.md`, `AUTHORS.md`, and `pyproject.toml`
- added `SECURITY.md` for private vulnerability reports
- use legal name **Aleksandr Mordvinov** in copyright and package metadata (GitHub: @FUYOH666)

## 2026-04-05

- aligned `launchd`, `.env`, and service startup through `scripts/run_service.sh`
- made `service.sh`, `start_all.sh`, and `stop_all.sh` refuse to kill unrelated processes on ports `9001` and `9002`
- added explicit `livez` and `readyz` endpoints while keeping `healthz` as a readiness alias
- kept `/health` as a backward-compatible readiness alias for older clients
- moved blocking model inference off the event loop and serialized inference with an async lock
- normalized tokenizer parallelism to the canonical `false` string across runtime config and launchd
- added lightweight pytest coverage for readiness and happy-path API responses using fake model stubs
- updated README and smoke test commands for the new operational behavior
