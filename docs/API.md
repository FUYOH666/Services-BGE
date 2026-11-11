# API reference

Two services, one model each. Defaults: embedding on `http://127.0.0.1:9001`, reranker on `http://127.0.0.1:9002`. Sidecars (see [MODELS.md](MODELS.md)) expose the same API on their own ports.

- [Authentication](#authentication)
- [Embeddings — `POST /v1/embeddings`](#embeddings)
- [Rerank — `POST /v1/rerank`](#rerank)
- [Model info — `GET /v1/models`](#model-info)
- [Health](#health)
- [Errors](#errors)
- [Hybrid search with Qdrant](#hybrid-search-with-qdrant)

Interactive OpenAPI docs are served at `/docs` on each port. AI agents can also use both services through the MCP memory server: [AGENT-MEMORY.md](AGENT-MEMORY.md).

## Authentication

None by default (the services listen on loopback only). When `SERVICE_API_KEY` (or `EMBEDDING_API_KEY` / `RERANKER_API_KEY`) is set, `POST` endpoints require one of:

```http
Authorization: Bearer <key>
X-API-Key: <key>
```

Health and `/v1/models` stay open so supervisors and status checks keep working.

## Embeddings

`POST /v1/embeddings` — OpenAI-compatible, with hybrid-search extensions.

### Request

| Field | Type | Default | Notes |
|---|---|---|---|
| `input` | string or string[] | required | 1…`EMBEDDING_MAX_ITEMS` (256) texts, each ≤ `EMBEDDING_MAX_TEXT_CHARS` (100 000) characters |
| `model` | string | — | Accepted for OpenAI clients; the served model is fixed |
| `encoding_format` | `"float"` \| `"base64"` | `"float"` | `base64` = little-endian float32 bytes, base64-encoded |
| `dimensions` | int | native | Must equal the native size, or be smaller on Matryoshka backends |
| `input_type` | `"query"` \| `"document"` | — | Applies the profile's query/document prompt on asymmetric models (Qwen3, e5). Ignored by bge-m3 |
| `return_dense` | bool | `true` | |
| `return_sparse` | bool | `true` | Skipped silently on dense-only backends unless set explicitly (then `400`) |
| `return_colbert` | bool | `false` | bge-m3 only |
| `max_length` | int | `EMBEDDING_MAX_LENGTH` (512) | Tokens per text; above `EMBEDDING_MAX_LENGTH_CAP` → `422` |
| `compact` | bool | `false` | Omit `dense_embedding`, token-readable `sparse_embedding` and null fields |
| `user` | string | — | Accepted and ignored |

### Response

```json
{
  "object": "list",
  "model": "BAAI/bge-m3",
  "usage": {"prompt_tokens": 17, "total_tokens": 17},
  "data": [
    {
      "object": "embedding",
      "index": 0,
      "embedding": [0.0123, -0.0456, "… 1024 floats"],
      "dense_embedding": [0.0123, -0.0456, "… same values"],
      "sparse_embedding": {"▁бес": 0.21, "шов": 0.18},
      "sparse_qdrant": {"indices": [6, 7712, 28149], "values": [0.21, 0.18, 0.11]},
      "colbert_embedding": null
    }
  ]
}
```

| Field | Meaning |
|---|---|
| `embedding` | Dense vector (OpenAI field). L2-normalized: cosine = dot product |
| `dense_embedding` | Same values, kept for older hybrid-search clients |
| `sparse_embedding` | Lexical weights keyed by readable token |
| `sparse_qdrant` | The same weights keyed by token id, ready for `models.SparseVector(**sparse_qdrant)` |
| `colbert_embedding` | One 1024-d vector per token (only with `return_colbert`) |
| `usage.total_tokens` | Tokens the model actually saw, after truncation, including prompts |

The default response shape is a contract: downstream projects read `embedding`, `dense_embedding`, `sparse_embedding`, `sparse_qdrant` and `index`, and the tests pin them.

### Examples

```bash
# OpenAI style
curl -s http://127.0.0.1:9001/v1/embeddings -H 'Content-Type: application/json' \
  -d '{"input": "seamless stainless steel pipe", "model": "bge-m3"}'

# Smallest payload for indexing: dense as base64 + Qdrant sparse
curl -s http://127.0.0.1:9001/v1/embeddings -H 'Content-Type: application/json' \
  -d '{"input": ["doc one", "doc two"], "compact": true, "encoding_format": "base64"}'
```

```python
import base64, numpy as np
vec = np.frombuffer(base64.b64decode(item["embedding"]), dtype="<f4")
```

## Rerank

`POST /v1/rerank` — scores documents against a query and returns them best first. Cohere/Jina-style.

### Request

| Field | Type | Default | Notes |
|---|---|---|---|
| `query` | string | required | ≤ `RERANKER_MAX_TEXT_CHARS` characters |
| `documents` | string[] | required | 1…`RERANKER_MAX_DOCUMENTS` (1000) |
| `top_n` | int ≥ 1 | all | Return only the best `top_n` |
| `normalize` | bool | `true` | `true`: score in [0, 1] (sigmoid); `false`: raw logit |
| `return_documents` | bool | `true` | `false` omits `document` from results |
| `max_length` | int | `RERANKER_MAX_LENGTH` (512) | Query + document tokens; above the cap → `422` |
| `instruction` | string | profile default | Task instruction for `qwen3-reranker`; other backends → `400` |
| `model` | string | — | Accepted and ignored |

### Response

```json
{
  "model": "BAAI/bge-reranker-v2-m3",
  "usage": {"total_pairs": 3},
  "results": [
    {"index": 0, "document": "7304 Seamless tubes", "relevance_score": 0.9849},
    {"index": 1, "document": "7306 Welded tubes", "relevance_score": 0.0068},
    {"index": 2, "document": "7318 Screws, bolts", "relevance_score": 0.0001}
  ]
}
```

`index` points into the request's `documents`. Scores are comparable within one model; do not compare absolute scores across different rerankers.

## Model info

`GET /v1/models` on either service:

```json
{
  "object": "list",
  "data": [{
    "id": "Qwen/Qwen3-Embedding-0.6B",
    "object": "model",
    "owned_by": "local",
    "backend": "sentence-transformers",
    "capabilities": {"dense": true, "sparse": false, "colbert": false, "matryoshka": true, "prompts": true},
    "dimensions": 1024,
    "max_length": 512,
    "max_length_cap": 8192,
    "device": "mps",
    "fp16": false
  }]
}
```

The reranker reports `backend`, `instruction`, `max_length`, `max_length_cap`, `device`, `fp16`.

## Health

| Endpoint | 200 when | Body |
|---|---|---|
| `GET /livez` | the process is up | `{"status": "ok", "model": …, "device": …, "ready": bool}` |
| `GET /readyz` | the model is loaded **and warmed up** | same; `503` with `"status": "loading"` before that |
| `GET /healthz`, `GET /health` | aliases of `/readyz` for older clients | |

## Errors

FastAPI's standard error body: `{"detail": "<message>"}` (validation errors return a list).

| Status | Cause |
|---|---|
| `400` | `dimensions` not supported; explicit `return_sparse` / `return_colbert` on a backend without it; `instruction` on a backend that ignores it |
| `401` | API key configured and missing or wrong |
| `413` | Too many texts / documents, or a text longer than the character limit |
| `422` | Invalid body: empty `input`, `top_n < 1`, `max_length` above the cap, wrong types |
| `503` | Model still loading |

## Hybrid search with Qdrant

A complete loop with [qdrant-client](https://github.com/qdrant/qdrant-client) ≥ 1.10: bge-m3 dense + sparse, fused with RRF in Qdrant, then reranked. Qdrant ≥ 1.19 removed the legacy `/search` endpoints, so use `query_points`.

```python
import httpx
from qdrant_client import QdrantClient, models

EMBED, RERANK = "http://127.0.0.1:9001/v1/embeddings", "http://127.0.0.1:9002/v1/rerank"
qdrant = QdrantClient("http://127.0.0.1:6333")

def embed(texts, input_type="document"):
    r = httpx.post(EMBED, json={"input": texts, "compact": True, "input_type": input_type}, timeout=120)
    r.raise_for_status()
    return r.json()["data"]

docs = ["7304 Seamless tubes and pipes of iron or steel", "7306 Other tubes, welded", "7318 Screws, bolts, nuts"]

qdrant.create_collection(
    "hs",
    vectors_config={"dense": models.VectorParams(size=1024, distance=models.Distance.COSINE)},
    sparse_vectors_config={"sparse": models.SparseVectorParams()},
)
qdrant.upsert("hs", points=[
    models.PointStruct(
        id=i,
        vector={"dense": item["embedding"], "sparse": models.SparseVector(**item["sparse_qdrant"])},
        payload={"text": text},
    )
    for i, (text, item) in enumerate(zip(docs, embed(docs)))
])

query = "seamless stainless steel pipe"
q = embed([query], input_type="query")[0]
candidates = qdrant.query_points(
    "hs",
    prefetch=[
        models.Prefetch(query=q["embedding"], using="dense", limit=50),
        models.Prefetch(query=models.SparseVector(**q["sparse_qdrant"]), using="sparse", limit=50),
    ],
    query=models.FusionQuery(fusion=models.Fusion.RRF),
    limit=20,
    with_payload=True,
).points

texts = [p.payload["text"] for p in candidates]
ranked = httpx.post(RERANK, json={"query": query, "documents": texts, "top_n": 3}).json()["results"]
for r in ranked:
    print(round(r["relevance_score"], 3), r["document"])
```

Tune the fusion before trusting it: on short, paraphrased queries equal-weight RRF can be worse than dense alone (see [eval/README.md](../eval/README.md)); Qdrant ≥ 1.17 supports weighted RRF.
