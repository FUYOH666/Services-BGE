# Roadmap — retrieval A/B

Status: **ready to start once a real HS-code golden set exists.** Written 2026-09-10, updated **2026-11-11**.

History: on 2026-09-10 the launchd agents were stopped for a host-memory bench; they are installed and running again (9001/9002). The measuring tools this plan was waiting for now exist:

- `eval/run_eval.py` — dense / sparse / hybrid RRF / +rerank metrics (recall, nDCG, MRR, HS-prefix accuracy) against **any** endpoint URLs, so sidecars plug in without code. See `eval/README.md`.
- `scripts/parity_check.py` — numeric drift, latency and memory between two deployments (used for fp16 and library upgrades).

This document is the agreed experiment, not a drop-in model swap.

---

## Principle

Treat this as a **retrieval system**, not a model-name contest.

- `9001` (`BAAI/bge-m3`) is a **hybrid retriever**: dense + lexical sparse + ColBERT. A denser MTEB score on Qwen3-Embedding does **not** prove that replacing `9001` is an upgrade (codes, GOST/TU, SKUs).
- Do **not** uninstall BGE to run quality A/B. Keep BGE as **control**; challengers on **sidecars**. Stop BGE only for an isolated RAM/latency production-candidate bench (as done 2026-09-10 for host memory, not for A/B).
- Do not put Jina in the first benchmark while Apache Qwen is in play (CC BY-NC 4.0).

## Target ports when A/B starts

```text
9001  BGE-M3 existing          CONTROL   dense+sparse+colbert
9002  BGE reranker existing    CONTROL

9011  Qwen3-Embedding-0.6B     SHADOW    dense only (phase 2)
9012  Qwen3-Reranker 0.6B/4B   CHALLENGER  (phase 1)
```

Four combinations on one query set (phase 2, after reranker):

- BGE retrieval → BGE rerank
- BGE retrieval → Qwen rerank
- Qwen retrieval → BGE rerank
- Qwen retrieval → Qwen rerank

## Phase 1 — reranker only (no reindex)

Same BGE-M3 top-30/50, three scorers:

| Role | Model |
|------|--------|
| Control | `BAAI/bge-reranker-v2-m3` on 9002 |
| Latency floor | [mlx-community/Qwen3-Reranker-0.6B-mxfp8](https://huggingface.co/mlx-community/Qwen3-Reranker-0.6B-mxfp8) sidecar |
| Quality candidate | [mlx-community/Qwen3-Reranker-4B-mxfp8](https://huggingface.co/mlx-community/Qwen3-Reranker-4B-mxfp8) (~4.15 GB weights) |

On M4 Max 128 GB, 0.6B size is **not** an advantage by itself. Winner = domain nDCG/MRR/win-rate + latency/RAM.

**Gate:** many MLX “reranker” cards expose `mlx_embeddings.generate()` (embeddings). Official Qwen scoring is CausalLM **yes/no** logits. Verify the scoring path and ranking vs BF16 original before trusting numbers.

`reranker_service` is PyTorch MPS + FlagEmbedding. `RERANKER_MODEL=mlx-community/...` on 9002 is **not** drop-in. Sidecar only. If 4B wins clearly, it is a **future** 9002 candidate, not an immediate launchd swap.

## Phase 2 — embedding as shadow only

[Qwen3-Embedding-0.6B](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B) (MLX 8-bit) is a **dense challenger**, not a `9001` replacement. Dimension 1024 matches width only; vector spaces differ — separate Qdrant collection. Use a customs/VED-style instruction, not generic web search.

## Phase 3 — visual lane

Qwen3-VL Embedding/Reranker 2B MLX: PDF page, table, product photo, nameplate, drawing, screenshot. Does not replace text retrieval.

## Update 2026-11-11 — model landscape and measurements

### Candidates (Apache/MIT only)

| Role | Model | Released | Why |
|---|---|---|---|
| Rerank challenger | `Qwen/Qwen3-Reranker-4B` | 2025-06 | MMTEB-R 72.7 vs 58.4 for bge-reranker-v2-m3; instruction-aware (write a customs/HS instruction); bf16 ≈ 8 GB |
| Rerank floor | `Qwen/Qwen3-Reranker-0.6B` | 2025-06 | Speed baseline. On short multilingual queries bge-reranker-v2-m3 still matches it (HAKARI-Bench 67.4 vs 66.5), so 4B is the real contender |
| Dense shadow (1024-d) | `microsoft/harrier-oss-v1-0.6b` | 2026-04 | MIT, MMTEB v2 69.0; needs an `Instruct: …\nQuery:` prefix |
| Dense shadow (1024-d) | `perplexity-ai/pplx-embed-v1-0.6b` | 2026-02 | MIT, no prompts, native int8/binary, MIRACL 68.2 |
| Dense shadow | `Qwen/Qwen3-Embedding-0.6B` / `-4B` | 2025-06 | Apache, Matryoshka, instruction-aware |
| Scale ceiling | `Qwen3-Embedding-8B`, `ai-sage/Giga-Embeddings-instruct` (RU control), `Qwen3-Reranker-8B` | | Does scale pay off on HS codes? |
| Middle stage | bge-m3 ColBERT vectors (already served), `pplx-embed-v1-late-0.6b` | | Cheap re-score of top-100 before the cross-encoder |
| Lexical | Russian BM25 in Qdrant next to bge-m3 sparse | | bge-m3 sparse is weak (MIRACL 55.5) but free; BM25 helps long documents |

Excluded (non-commercial licences): jina-embeddings v5, jina-reranker v3/v3.5, jina-colbert-v2, zerank-2, llama-embed-nemotron. Not upgrades for this use: USER-bge-m3 (slightly below bge-m3 on RusBEIR), EmbeddingGemma 2 (text quality same as v1).

bge-m3 itself is **not** outdated for Russian retrieval: ruMTEB retrieval 74.79 (above Giga 74.28), best dense model on RusBEIR (61.13), and it is still the only permissive model that returns dense + sparse + multi-vector in one pass.

### Quantization rules

- Models ≤ 1B: fp16/bf16 only. 4–8B: Q8_0 GGUF or MLX 8-bit. **No Q4 for embedders** (fails cosine ≥ 0.99; non-Latin long documents lose up to ~24% nDCG).
- MLX 8-bit and standard GGUF ports of bge-m3 **drop the sparse and ColBERT heads** — not a 9001 replacement.
- Vectors in Qdrant: scalar int8 with rescoring first (≈0.1 point loss); binary only with oversampling + rescoring.

### fp16 on the current stack (M4 Max, torch 2.13, measured 2026-11-11 with `parity_check.py`)

| | fp32 (live) | fp16 | Verdict |
|---|---|---|---|
| Embedding footprint | 3.7 GB | 3.1 GB | dense cos ≥ 0.99999, sparse top-10 identical |
| Embedding 8 long texts | 532 ms | 495 ms | small gain |
| Reranker footprint | 3.7 GB | 2.5 GB | top-5 identical, max score diff 0.0005, Spearman 0.989 (tail of near-zero scores) |
| Reranker 32 pairs | 215 ms | 258 ms | **slower** |

Keep fp32 as default. Re-measure after moving to torch ≥ 2.14 (new MPS fp16/bf16 attention kernel for head size 64) and FlagEmbedding ≥ 1.4.2 (transformers v5 support), both behind `parity_check.py`.

### Demo baseline

`eval/datasets/demo_ru` (36 toy queries): dense recall@1 0.81, equal-weight hybrid RRF 0.56, any +rerank@20 0.75. Toy data, but a warning that fusion weights and the rerank stage must be tuned on the real set, not assumed to help.

### Replacement gate

Replace 9002 (or 9001) only if the challenger wins by **≥ 3 points nDCG@10 or HS-10 top-5 accuracy** on the real held-out set, with rerank p95 for top-50 under 1.5 s and RAM that fits next to the LLM workloads.

## Cloud Agent prompt (when researching HF again)

```text
Дата: 2026-09-10. Хост: Apple M4 Max 128GB. Services-BGE УЖЕ CONTROL:
- 9001 BAAI/bge-m3 FlagEmbedding MPS: dense 1024 + sparse_qdrant + colbert (НЕ drop-in на Qwen)
- 9002 BAAI/bge-reranker-v2-m3 FlagEmbedding MPS
Языки: ru + en. Лицензия прода: MIT/Apache, не CC-BY-NC.

Operational rule: НЕ останавливать 9001/9002 для quality A/B.
Challengers только на отдельных портах (9011 embed shadow, 9012 rerank).
Останавливать baseline только для изолированного RAM/latency benchmark.
Не предлагай квант BGE-M3 как главную идею. Не предлагай замену 9001 на dense-only как upgrade.

Проверь на Hugging Face (карточки + files):
1) mlx-community/Qwen3-Embedding-0.6B-8bit — размер, лицензия, API, dim, instruct. Только shadow dense, не 9001.
2) Qwen3-Reranker MLX 0.6B: какие репо реально yes/no logits + lm_head, какие сконвертированы как embedding. mxfp8 vs 8bit vs 4bit. mlx-lm vs mlx-embeddings vs omlx.
   Отдельно: mlx-community/Qwen3-Reranker-4B-mxfp8 как quality-tier. На M4 Max 128GB сравни 0.6B vs 4B по quality/latency/RAM на rerank top-20/top-50. Минимальный размер модели не считать преимуществом сам по себе.
3) Qwen3-VL-Embedding-2B / VL-Reranker-2B MLX — отдельный visual lane, не text A/B.
4) Jina v5 / reranker v3.5 MLX — зафиксировать BY-NC, не бенчмаркать пока Qwen в игре.
5) Другие Apache/MIT MLX encoder/reranker за 60 дней ~0.6–4B, RU+EN.

Верни таблицу: model_id, роль, диск, RAM, dim, sparse?, лицензия, scoring path (embed vs yes/no), риск sidecar.
Рекомендация фазы 1: BGE retrieval + три reranker. Не пиши код Services-BGE и не меняй launchd.
```
