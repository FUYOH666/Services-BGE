# Services-BGE

BGE-M3 Embedding & BGE-Reranker-v2-m3 microservices for hybrid search.

```bash
# Установить как системный сервис (один раз)
cd ~/development/Services-BGE && ./scripts/service.sh install && ./scripts/service.sh start

# Проверить статус
./scripts/service.sh status
```

## Architecture

Two FastAPI microservices in one project:

| Service | Model | Port | Purpose |
|---------|-------|------|---------|
| Embedding | `BAAI/bge-m3` (568M params) | 9001 | Dense + sparse vectors for Qdrant hybrid search |
| Reranker | `BAAI/bge-reranker-v2-m3` (568M params) | 9002 | Cross-encoder relevance scoring |

## Requirements

- macOS with Apple Silicon (M1/M2/M3/M4 Max/Ultra)
- Python 3.12
- ~3 GB RAM for both models (MPS/Metal)
- [uv](https://docs.astral.sh/uv/) package manager

## Quick Start

```bash
uv sync                              # Install dependencies (once)
./scripts/service.sh install         # Install as macOS launchd service (once)
./scripts/service.sh start           # Start both services
./scripts/service.sh status          # Check health & PIDs
./scripts/service.sh stop            # Stop both services
./scripts/service.sh restart         # Restart both services
./scripts/service.sh logs            # Tail live logs
./scripts/service.sh uninstall       # Remove from launchd
```

Services auto-start on login and auto-restart on crash via launchd.
Logs: `~/Library/Logs/Services-BGE/`

Manual fallback (without launchd):

```bash
./scripts/start_all.sh               # Runs in foreground terminal
./scripts/stop_all.sh                # Stop manual processes
```

## API

### Embedding Service (port 9001)

```bash
# Health check
curl http://localhost:9001/healthz

# Generate embeddings (dense + sparse)
curl -X POST http://localhost:9001/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "input": ["бесшовные трубы из нержавеющей стали"],
    "return_dense": true,
    "return_sparse": true
  }'
```

### Reranker Service (port 9002)

```bash
# Health check
curl http://localhost:9002/healthz

# Rerank documents
curl -X POST http://localhost:9002/v1/rerank \
  -H "Content-Type: application/json" \
  -d '{
    "query": "бесшовные трубы из нержавеющей стали",
    "documents": ["7304 — Трубы бесшовные", "7318 — Винты, болты"],
    "normalize": true
  }'
```

## Environment Variables

See `.env.example` for all available configuration options.

## Performance (Apple M4 Max, MPS/Metal, fp32)

| Operation | Latency (warm) | Throughput |
|-----------|---------------|------------|
| BGE-M3 encode (8 texts, dense+sparse) | ~55 ms | ~145 texts/sec |
| BGE Reranker score (8 pairs) | ~112 ms | ~71 pairs/sec |

## Resource Usage

| Model | Disk | RAM (fp32) |
|-------|------|-----------|
| BGE-M3 | ~2.2 GB | ~2 GB |
| BGE-Reranker-v2-m3 | ~2.2 GB | ~1 GB |
| **Total** | **~4.4 GB** | **~3 GB** |
