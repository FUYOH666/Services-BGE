# Services-BGE

BGE-M3 embedding and BGE-Reranker-v2-m3 microservices for hybrid search on Apple Silicon.

## Architecture

Two FastAPI services live in one project:

| Service | Model | Port | Purpose |
|---------|-------|------|---------|
| Embedding | `BAAI/bge-m3` | `9001` | Dense + sparse vectors for hybrid search |
| Reranker | `BAAI/bge-reranker-v2-m3` | `9002` | Cross-encoder relevance scoring |

Runtime characteristics:

- Apple Silicon only, using `mps`
- One worker per service
- `launchd` for auto-start and crash restart
- Explicit `readyz` and `livez` endpoints
- MPS cache is cleared after inference to keep memory usage stable
- HuggingFace tokenizer parallelism is forced to `false` to reduce fork-related noise on macOS

## Requirements

- macOS on Apple Silicon
- Python `3.12`
- [uv](https://docs.astral.sh/uv/)
- About `3-4 GB` RAM for both models together, plus headroom for startup and fragmentation

## Quick Start

```bash
cd ~/development/Services-BGE
uv sync
cp .env.example .env                  # optional, if you want custom host/port/log settings
./scripts/service.sh install
./scripts/service.sh start
./scripts/service.sh status
```

Useful commands:

```bash
./scripts/service.sh stop
./scripts/service.sh restart
./scripts/service.sh logs
./scripts/service.sh uninstall
```

`install` renders `launchd` plist files for the current machine and project path. Host/port/workers/log level are read from `.env` by `scripts/run_service.sh`, so changing `.env` only requires `./scripts/service.sh restart`.

Logs are written to `~/Library/Logs/Services-BGE/`.

Manual foreground fallback:

```bash
./scripts/start_all.sh
./scripts/stop_all.sh
```

## Health Endpoints

Each service exposes:

- `GET /livez` - process is alive
- `GET /readyz` - model is loaded and ready for inference
- `GET /healthz` - backward-compatible alias for `readyz`
- `GET /health` - backward-compatible alias for `readyz`

Examples:

```bash
curl http://localhost:9001/livez
curl http://localhost:9001/readyz
curl http://localhost:9002/livez
curl http://localhost:9002/readyz
```

## API Examples

### Embedding

```bash
curl -X POST http://localhost:9001/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "input": ["бесшовные трубы из нержавеющей стали"],
    "return_dense": true,
    "return_sparse": true
  }'
```

### Reranker

```bash
curl -X POST http://localhost:9002/v1/rerank \
  -H "Content-Type: application/json" \
  -d '{
    "query": "бесшовные трубы из нержавеющей стали",
    "documents": ["7304 — Трубы бесшовные", "7318 — Винты, болты"],
    "normalize": true
  }'
```

## Configuration

All runtime settings live in `.env` and are documented in `.env.example`.

Important notes:

- Default `HOST` is `0.0.0.0`, which allows access from other interfaces such as TailScale or LAN.
- If you want local-only access, set `EMBEDDING_HOST=127.0.0.1` and `RERANKER_HOST=127.0.0.1`, then restart the services.
- `service.sh` will refuse to kill unrelated processes that already occupy `9001` or `9002`.

## Testing

Fast local checks:

```bash
uv run --extra dev pytest
uv run python scripts/test_services.py
```

`pytest` uses fake in-memory model stubs, so it validates endpoint behavior without downloading or loading the real BGE models.

## Operational Notes

- The canonical production path is `dense + sparse` vectors from BGE-M3 into Qdrant, then second-stage `reranker`; `colbert_embedding` is available for experiments but is not part of the default runtime contract.
- If you still see `resource_tracker: leaked semaphore objects` during shutdown on macOS, treat it as a known platform/runtime warning unless it starts affecting startup or inference. The services are configured to minimize tokenizer parallelism noise, but the warning can still appear on process teardown.

## Performance

Measured on Apple M4 Max with MPS/Metal and fp32:

| Operation | Latency (warm) | Throughput |
|-----------|---------------|------------|
| BGE-M3 encode (8 texts, dense+sparse) | ~55 ms | ~145 texts/sec |
| BGE Reranker score (8 pairs) | ~112 ms | ~71 pairs/sec |

## Resource Usage

| Model | Disk | RAM (fp32) |
|-------|------|-----------|
| BGE-M3 | ~2.2 GB | ~2 GB |
| BGE-Reranker-v2-m3 | ~2.2 GB | ~1 GB |
| **Total** | **~4.4 GB** | **~3-4 GB with runtime overhead** |

## License

This project is licensed under the [MIT License](LICENSE). If you use or adapt it, keep the copyright and license notice with your copy (see the `LICENSE` file).

## Maintainer & contact

**Aleksandr Mordvinov** ([@FUYOH666](https://github.com/FUYOH666)) — questions, ideas, or collaboration:

- Email: [iamfuyoh@gmail.com](mailto:iamfuyoh@gmail.com)
- Telegram: [@ScanovichAI](https://t.me/ScanovichAI)

See [AUTHORS.md](AUTHORS.md) for attribution notes (including third-party models and libraries). Security-sensitive reports: [SECURITY.md](SECURITY.md).
