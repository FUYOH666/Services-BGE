# Services-BGE

BGE-M3 Embedding & BGE-Reranker-v2-m3 microservices for hybrid search.

```bash
# Запуск обоих сервисов (из любого терминала)
cd ~/development/Services-BGE && ./scripts/start_all.sh
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
# Install dependencies
uv sync

# Start both services
./scripts/start_all.sh

# Or start individually
uv run uvicorn embedding_service.main:app --port 9001 --workers 1
uv run uvicorn reranker_service.main:app --port 9002 --workers 1
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
