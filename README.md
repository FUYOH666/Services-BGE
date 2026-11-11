<p align="center">
  <img src="assets/services-bge-cover.png" alt="Services-BGE — BGE-M3 hybrid search on Apple Silicon" width="420">
</p>

# Services-BGE

[![tests](https://github.com/FUYOH666/Services-BGE/actions/workflows/test.yml/badge.svg)](https://github.com/FUYOH666/Services-BGE/actions/workflows/test.yml)

BGE-M3 embedding and BGE-Reranker-v2-m3 microservices for hybrid search on Apple Silicon, plus an eval harness to compare them against newer models on your own data.

## Architecture

Two FastAPI services live in one project:

| Service | Model | Port | Purpose |
|---------|-------|------|---------|
| Embedding | `BAAI/bge-m3` | `9001` | Dense + sparse (+ optional ColBERT) vectors for hybrid search |
| Reranker | `BAAI/bge-reranker-v2-m3` | `9002` | Cross-encoder relevance scoring |

Runtime characteristics:

- Apple Silicon, PyTorch `mps`, one worker per service, inference serialized per process
- `launchd` for auto-start and crash restart
- Explicit `readyz` and `livez` endpoints; models are warmed up before `readyz` turns green
- Starts offline from the local Hugging Face cache, so a boot without network does not stall
- Request limits, optional API key, MPS cache policy, all configured in `.env`

### Why launchd and not Docker on a Mac

Containers on macOS (Docker Desktop, Apple `container`, OrbStack) cannot reach the Apple GPU through Metal; Podman only forwards Vulkan for llama.cpp. PyTorch MPS therefore has to run natively, and `launchd` is the native supervisor. Docker remains the right tool for Qdrant and for a future Linux/CUDA deployment.

Ready-made servers (Text Embeddings Inference, llama.cpp, vllm-metal, MLX, Ollama, Docker Model Runner) can serve bge-m3 dense vectors, but as of November 2026 none of them returns bge-m3's sparse lexical weights and ColBERT vectors on Metal. That is why this project wraps FlagEmbedding itself.

## Requirements

- macOS on Apple Silicon
- Python `3.12`
- [uv](https://docs.astral.sh/uv/)
- About `7.5 GB` of memory for both services in fp32 (measured physical footprint ≈ 3.7 GB per process on an M4 Max), plus headroom

## Quick Start

```bash
cd ~/development/Services-BGE
uv sync
cp .env.example .env                  # optional, to customize host/ports/limits
./scripts/service.sh install
./scripts/service.sh start
./scripts/service.sh status
```

Useful commands:

```bash
./scripts/service.sh stop
./scripts/service.sh restart
./scripts/service.sh logs
./scripts/service.sh rotate-logs      # gzip + truncate logs over 10 MB, keep 5 archives
./scripts/service.sh uninstall
```

`install` renders `launchd` plist files for the current machine and project path. Settings are read from `.env` by `scripts/run_service.sh`, so changing `.env` only requires `./scripts/service.sh restart`.

Logs are written to `~/Library/Logs/Services-BGE/`. Per-request inference lines are logged at `DEBUG`; requests slower than `*_SLOW_REQUEST_MS` are logged at `WARNING`.

Manual foreground fallback:

```bash
./scripts/start_all.sh
./scripts/stop_all.sh
```

Run a side copy (for example a branch or an fp16 profile) without touching the live services; variables set on the command line win over `.env`:

```bash
EMBEDDING_PORT=9101 ./scripts/run_service.sh embedding
RERANKER_PORT=9102 RERANKER_USE_FP16=true ./scripts/run_service.sh reranker
```

## Health Endpoints

Each service exposes:

- `GET /livez` - process is alive
- `GET /readyz` - model is loaded, warmed up and ready for inference
- `GET /healthz` - backward-compatible alias for `readyz`
- `GET /health` - backward-compatible alias for `readyz`

```bash
curl http://localhost:9001/readyz
curl http://localhost:9002/readyz
```

## API

### Embedding — `POST /v1/embeddings`

OpenAI-compatible (verified with the official Python `openai` SDK, which requests base64 by default), with hybrid-search extensions.

```bash
curl -X POST http://localhost:9001/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "input": ["бесшовные трубы из нержавеющей стали"],
    "return_dense": true,
    "return_sparse": true
  }'
```

| Field | Default | Notes |
|---|---|---|
| `input` | required | a string or a list of strings |
| `model`, `user` | — | accepted for OpenAI compatibility; the served model is fixed |
| `encoding_format` | `float` | `base64` returns little-endian float32 base64 (≈7× smaller with `compact`) |
| `dimensions` | — | only the native size (1024) is accepted; bge-m3 is not Matryoshka |
| `return_dense` / `return_sparse` / `return_colbert` | `true` / `true` / `false` | |
| `max_length` | `EMBEDDING_MAX_LENGTH` (512) | capped by `EMBEDDING_MAX_LENGTH_CAP` (8192) |
| `compact` | `false` | omits the duplicate `dense_embedding`, the token-readable `sparse_embedding` and null fields |

Response items always keep the fields existing clients rely on: `index`, `embedding` (OpenAI), `dense_embedding`, `sparse_embedding` (token → weight), `sparse_qdrant` (`indices`/`values`, ready for a Qdrant sparse vector) and `colbert_embedding`. `usage.total_tokens` reports the tokens the model actually saw.

### Reranker — `POST /v1/rerank`

```bash
curl -X POST http://localhost:9002/v1/rerank \
  -H "Content-Type: application/json" \
  -d '{
    "query": "бесшовные трубы из нержавеющей стали",
    "documents": ["7304 — Трубы бесшовные", "7318 — Винты, болты"],
    "normalize": true,
    "top_n": 1
  }'
```

| Field | Default | Notes |
|---|---|---|
| `query`, `documents` | required | |
| `top_n` | all | must be ≥ 1 |
| `normalize` | `true` | sigmoid scores in `[0, 1]` |
| `return_documents` | `true` | `false` returns only `index` + `relevance_score` (Cohere/Jina style) |
| `max_length` | `RERANKER_MAX_LENGTH` (512) | query + document tokens, capped by `RERANKER_MAX_LENGTH_CAP` |

### Errors and limits

| Status | When |
|---|---|
| `401` | `SERVICE_API_KEY` (or a per-service key) is set and the request has no matching `Authorization: Bearer …` / `X-API-Key` header |
| `413` | more than `EMBEDDING_MAX_ITEMS` (256) texts / `RERANKER_MAX_DOCUMENTS` (1000) documents, or a text longer than `*_MAX_TEXT_CHARS` |
| `422` | invalid body, `top_n < 1`, or `max_length` above the cap |
| `400` | unsupported `dimensions` |
| `503` | model still loading |

## Configuration

All runtime settings live in `.env` and are documented in `.env.example`. Highlights:

- **Host** defaults to `127.0.0.1`. To expose the services on LAN or Tailscale set `EMBEDDING_HOST=0.0.0.0` / `RERANKER_HOST=0.0.0.0` **and** a `SERVICE_API_KEY`.
- **Offline start**: when `HF_HUB_OFFLINE` is unset, `run_service.sh` turns it on automatically if the model is already cached. `*_MODEL_REVISION` pins a commit for reproducible restarts.
- **MPS cache**: `*_MPS_CACHE_THRESHOLD_MB=0` empties the MPS cache after every request (historical behaviour that keeps memory flat). A positive value only empties it above that many MB of Metal driver memory, trading memory for latency.
- **Precision**: `*_USE_FP16=false` by default. Measured on an M4 Max with torch 2.13, fp16 saves 0.6–1.2 GB per process with near-identical outputs, but the reranker gets slower on 32 pairs, so fp32 stays the default (details in [docs/ROADMAP.md](docs/ROADMAP.md)).
- `service.sh` refuses to kill unrelated processes that already occupy `9001` or `9002`.

## Testing

```bash
uv run --extra dev pytest                                # fake models, no downloads; also runs in CI
uv run python scripts/test_services.py                   # smoke test against running services
uv run python scripts/test_services.py --embed-url http://127.0.0.1:9101 --rerank-url http://127.0.0.1:9102
```

The pytest suite includes contract tests that pin the response fields downstream projects read, so API changes stay additive.

### Parity between two deployments

`scripts/parity_check.py` compares a reference (A) and a candidate (B): dense cosine, sparse top-10 overlap, rerank Spearman and top-5 order, latency, response size and process memory. Use it before switching precision, upgrading torch/FlagEmbedding or deploying a branch:

```bash
uv run python scripts/parity_check.py \
  --a-embed http://127.0.0.1:9001 --b-embed http://127.0.0.1:9101 \
  --a-rerank http://127.0.0.1:9002 --b-rerank http://127.0.0.1:9102
```

### Retrieval quality

`eval/run_eval.py` scores dense, sparse, hybrid RRF and reranked retrieval (recall, nDCG, MRR, HS-code prefix accuracy, latency) for any endpoint URLs. It is how challenger models are judged; see [eval/README.md](eval/README.md) and [docs/ROADMAP.md](docs/ROADMAP.md).

## Operational Notes

- Retrieval A/B plan (BGE stays the control, challengers on side ports 9011/9012): [docs/ROADMAP.md](docs/ROADMAP.md).
- The canonical production path is `dense + sparse` vectors from BGE-M3 into Qdrant, then second-stage `reranker`; `colbert_embedding` is available for experiments but is not part of the default runtime contract.
- With Qdrant ≥ 1.19 use the `/query` API (the legacy `/search` endpoints were removed): prefetch dense and sparse, fuse with (weighted) RRF, then rerank the top-k here.
- If you still see `resource_tracker: leaked semaphore objects` during shutdown on macOS, treat it as a known platform/runtime warning unless it starts affecting startup or inference.

## Performance

Measured on Apple M4 Max with MPS and fp32 (warm, median, via `scripts/parity_check.py`):

| Operation | Latency |
|-----------|---------|
| BGE-M3 encode, 8 short texts, dense + sparse | ~55–65 ms |
| BGE-M3 encode, 8 texts × ~450 tokens, dense + sparse | ~530 ms |
| BGE-M3 encode, 1 short text, dense only | ~30 ms |
| Reranker, 8 pairs | ~80 ms |
| Reranker, 32 pairs | ~215 ms |

Response size for 16 mixed texts with dense + sparse: 708 KB default, 353 KB with `compact`, 99 KB with `compact` + `encoding_format=base64`.

## Resource Usage

| Model | Disk (HF cache) | Memory footprint (fp32) |
|-------|------|-----------|
| BGE-M3 | ~2.2 GB weights (the full snapshot incl. ONNX is ~6.4 GB) | ~3.7 GB |
| BGE-Reranker-v2-m3 | ~2.1 GB | ~3.7 GB |
| **Total** | | **~7.4 GB** |

## License

This project is licensed under the [MIT License](LICENSE). If you use or adapt it, keep the copyright and license notice with your copy (see the `LICENSE` file).

## Maintainer & contact

**Aleksandr Mordvinov** ([@FUYOH666](https://github.com/FUYOH666)) — questions, ideas, or collaboration:

- Email: [iamfuyoh@gmail.com](mailto:iamfuyoh@gmail.com)
- Telegram: [@ScanovichAI](https://t.me/ScanovichAI)

See [AUTHORS.md](AUTHORS.md) for attribution notes (including third-party models and libraries). Security-sensitive reports: [SECURITY.md](SECURITY.md).
