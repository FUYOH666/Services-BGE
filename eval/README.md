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
#   --max-length 1024      token limit sent to the embedder
```

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

## Notes from the demo baseline (2026-11-11, M4 Max, fp32)

On `demo_ru`, plain equal-weight RRF of dense + sparse was **worse** than dense alone
(recall@1 0.56 vs 0.81), and reranking the top-20 did not beat dense either (0.75).
`--dense-weight 3` narrows the gap (0.61). This is a toy set, but it shows why fusion
weights and the rerank stage must be tuned on real data rather than assumed to help.
