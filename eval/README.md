# Retrieval eval

`run_eval.py` measures retrieval quality and latency of **any** embedding + rerank
endpoints on a JSONL dataset. It is the gate for every model or config change: a
challenger replaces the control only if it wins here, on your data, not on a public
leaderboard.

## Run

```bash
# control: live BGE services on 9001/9002
uv run python eval/run_eval.py --dataset eval/datasets/demo_ru --name bge-baseline

# challenger on side ports (see docs/ROADMAP.md), same dataset
uv run python eval/run_eval.py --dataset eval/datasets/demo_ru \
    --embed-url http://127.0.0.1:9011 --rerank-url http://127.0.0.1:9012 --name challenger

# options
#   --rerank-url ""        skip reranking
#   --rerank-k 20          candidates passed to the reranker
#   --dense-weight 3       weighted RRF (Qdrant 1.17+ semantics), sparse weight is 1
#   --query-embed-url URL  embed queries with a different endpoint than documents
#   --sparse-url URL       take sparse vectors from another endpoint (e.g. bge-m3 on 9001)
#                          so a dense-only challenger is still tested in a hybrid pipeline
#   --max-length 1024      token limit sent to the embedder

# one table across runs
uv run python eval/compare.py eval/results/*.json --metric recall@1 --out eval/results/COMPARISON.md
```

Queries are sent with `"input_type": "query"` and documents with `"document"`, so
asymmetric models (Qwen3-Embedding, e5) get their prompts; bge-m3 ignores the field.
Challengers come from `profiles/` via `./scripts/sidecar.sh up <profile>`.

Each run writes `results/<timestamp>_<dataset>_<name>.json` (full report) and `.md`
(table). Runs reported:

| run | what it is |
|---|---|
| `dense` | cosine over the dense vectors |
| `sparse` | BGE-M3 lexical matching (sum of shared token weights); only if the endpoint returns `sparse_qdrant` |
| `hybrid_rrf` | reciprocal rank fusion of dense and sparse (`k=60`), like Qdrant `/query` with RRF |
| `<base>+rerank@K` | top-K of that run re-ordered by the reranker |

Metrics: `recall@{1,5,10,50}`, `mrr@10`, `ndcg@10`, and when queries and documents
carry `hs_code`: `code{2,4,6,10}_acc@{1,5}` (does a top-k document share the query's
code prefix). Timing: index time, query-embedding p50/p95, rerank p50/p95.

## Dataset format

```text
<dataset>/corpus.jsonl   {"id": "hs-7304", "text": "...", "hs_code": "7304"}
<dataset>/queries.jsonl  {"id": "q001", "query": "...", "positives": ["hs-7304"], "hs_code": "7304"}
```

`hs_code` is optional. `positives` lists corpus ids that count as relevant.

## Datasets

- `datasets/demo_ru` — 36 Russian product descriptions against 36 HS heading texts
  (4-digit). **A pipeline sanity check, not evidence of model quality**: it is tiny,
  synthetic and the corpus is one short line per heading.
- Real gate: a held-out set of real declarations with verified HS-10 codes from the
  classification project. Put it in `datasets/<name>/`; keep private data out of git.

## Notes from the demo matrix (2026-11-11, M4 Max, fp32)

Full table: [`results/COMPARISON.md`](results/COMPARISON.md). On `demo_ru`:

- Qwen3-Reranker-0.6B on top of bge-m3 lifted recall@1 from 0.81 to **0.86** (rerank
  ~580 ms per 20 documents); bge-reranker-v2-m3 *lowered* it to 0.75 (~150 ms).
- Equal-weight RRF of dense + sparse was worse than dense alone (0.56 vs 0.81);
  `--dense-weight 3` narrows the gap (0.61).
- Qwen3-Embedding-0.6B dense (0.69) and multilingual-e5-large (0.78) did not beat
  bge-m3 dense (0.81).

This is a toy set, so none of this is a verdict. It shows why fusion weights, the
rerank stage and the model choice must be measured on real data.
