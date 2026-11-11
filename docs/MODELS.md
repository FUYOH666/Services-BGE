# Models, backends and profiles

Services-BGE is model-agnostic behind a fixed HTTP contract. A **backend** knows how to run one family of models; a **profile** picks a backend, a model and its settings; a **sidecar** is a profile running next to production.

## Backends

### Embedding

| Backend | Runs | Outputs | Notes |
|---|---|---|---|
| `bge-m3` | `BGEM3FlagModel` from [FlagEmbedding](https://github.com/FlagOpen/FlagEmbedding) | dense + sparse + ColBERT | Production. Use for BAAI/bge-m3 and its fine-tunes that keep the sparse/ColBERT heads |
| `sentence-transformers` | any [SentenceTransformer](https://www.sbert.net) | dense | Query/document prompts via `input_type`; Matryoshka `dimensions` when `EMBEDDING_MATRYOSHKA=true`; vectors are L2-normalized |

### Reranker

| Backend | Runs | Notes |
|---|---|---|
| `flag-reranker` | `FlagReranker` (FlagEmbedding) | Production. bge-reranker-v2-m3 and other XLM-R/BERT cross-encoders |
| `cross-encoder` | sentence-transformers `CrossEncoder` | Any single-logit cross-encoder from the Hub; sigmoid applied in fp32 |
| `qwen3-reranker` | `AutoModelForCausalLM` (transformers) | Qwen3-Reranker 0.6B/4B/8B with the official prompt; score = P("yes") = sigmoid(logit_yes − logit_no); takes an `instruction` |

All rerank backends share one scale: `normalize=true` gives [0, 1], `false` gives the raw logit.

## Profiles

| Profile | Port | Model | Download | Memory (fp32 unless noted) | Licence | Status |
|---|---|---|---|---|---|---|
| `bge-m3` | 9001 | BAAI/bge-m3 | ~2.2 GB weights (~6.4 GB snapshot) | ~3.7 GB | MIT | production |
| `bge-reranker-v2-m3` | 9002 | BAAI/bge-reranker-v2-m3 | ~2.1 GB | ~3.7 GB | Apache-2.0 | production |
| `qwen3-embedding-0.6b` | 9011 | Qwen/Qwen3-Embedding-0.6B | ~1.2 GB | ~3.9 GB | Apache-2.0 | verified 2026-11-11 |
| `qwen3-reranker-0.6b` | 9012 | Qwen/Qwen3-Reranker-0.6B | ~1.2 GB | ~3.7 GB | Apache-2.0 | verified 2026-11-11 |
| `multilingual-e5-large` | 9013 | intfloat/multilingual-e5-large | ~2.2 GB | ~4.0 GB | MIT | verified 2026-11-11 |
| `qwen3-reranker-4b` | 9014 | Qwen/Qwen3-Reranker-4B | ~8 GB | ~9 GB (fp16) | Apache-2.0 | candidate: same backend as 0.6B, not yet benchmarked |

Memory is the process footprint on an M4 Max and includes PyTorch overhead.

### Using sidecars

```bash
./scripts/sidecar.sh list
./scripts/sidecar.sh up qwen3-reranker-0.6b          # first start downloads the model
curl -s http://127.0.0.1:9012/v1/models
./scripts/sidecar.sh logs qwen3-reranker-0.6b
./scripts/sidecar.sh down qwen3-reranker-0.6b        # or: down all
```

Sidecars run in the background and are not started at login. Logs: `~/Library/Logs/Services-BGE/sidecar-<profile>.log`.

### Asymmetric models need `input_type`

Qwen3-Embedding and e5 expect different text for queries and documents. Send `"input_type": "query"` for queries; documents need nothing (or `"document"`). `eval/run_eval.py` does this automatically. bge-m3 ignores the field, so clients can send it unconditionally.

### Mixing models

The eval can take dense vectors from a challenger and sparse vectors from bge-m3 (`--sparse-url http://127.0.0.1:9001`), which is how a dense-only model is tested inside a hybrid pipeline without losing lexical matching on codes and part numbers.

## Adding a model

1. Copy the closest profile: `cp profiles/qwen3-embedding-0.6b.env profiles/my-model.env`.
2. Set `EMBEDDING_MODEL` (or `RERANKER_MODEL`), the backend, a free port (odd 9011+ for embeddings, even for rerankers), prompts from the model card, `EMBEDDING_MATRYOSHKA` if the card says Matryoshka, and `*_MAX_LENGTH_CAP` from the model's context length.
3. Read the model card for licence and for `trust_remote_code`; enable remote code only after reviewing it.
4. `./scripts/sidecar.sh up my-model`, then check `GET /v1/models` and a few requests by hand.
5. Run the eval against the control and record the result. Mark the profile *verified* in the table above only after it loaded and scored here.

If a model needs a library or scoring method none of the backends cover, add a backend class with the same `encode(...)` or `score(...)` signature in `embedding_service/backends.py` / `reranker_service/backends.py`, register it in `load_backend`, and add a test with a fake model (see `tests/test_service_endpoints.py`).

## Model landscape (November 2026)

Research summary behind the profiles; details and sources in [ROADMAP.md](ROADMAP.md).

- **bge-m3 is not outdated for Russian/multilingual retrieval**: ruMTEB retrieval 74.79, the best dense model on RusBEIR, and it returns dense, sparse and multi-vector outputs from one pass.
- **The reranker is where newer models gain most.** Qwen3-Reranker-4B scores ~72.7 on MMTEB reranking vs ~58.4 for bge-reranker-v2-m3; on short multilingual queries the 0.6B model is about level with bge-reranker. The demo set agrees directionally (see the README), but only your own data decides.
- **2026 dense challengers worth a profile:** microsoft/harrier-oss-v1-0.6b (MIT, needs an instruction prefix), perplexity-ai/pplx-embed-v1-0.6b (MIT, no prefixes). Not added yet because they have not been verified here.
- **Excluded for commercial use (non-commercial licences):** jina-embeddings v5, jina-reranker v3/v3.5, jina-colbert-v2, zerank-2, llama-embed-nemotron.
- **Quantization:** ≤1B models in fp16/bf16 only; 4–8B in Q8 / MLX 8-bit; no Q4 for embedders (Cyrillic long documents lose up to ~24% nDCG). MLX and GGUF ports of bge-m3 drop the sparse and ColBERT heads.
