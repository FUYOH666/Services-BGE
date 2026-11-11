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

## Agent memory bench

`memory_bench.py` asks a different question: when agents share [MCP memory](../docs/AGENT-MEMORY.md),
does `search` return the passage that answers a question about *your* knowledge base, in
Russian, English and mixed phrasing, and can an unreviewed note written by a bot push a
false fact to the top? Method and results: [docs/MEMORY-BENCH.md](../docs/MEMORY-BENCH.md).

```bash
uv run python eval/memory_bench.py --bench eval/datasets/memory_synthetic
uv run python eval/memory_bench.py --bench eval/private/my-canon      # your own set, gitignored
#   --no-poison           skip the poisoning runs
#   --penalties 0.75,0.5  MCP_TIER_PENALTY values to test with prefer_tier=canon
#   --keep                keep the scratch collection (mcp_bench-*) for inspection
```

It indexes the corpus with the MCP server's own code (`mcp_server.memory.Memory`: same
chunking, payloads, hybrid query and rerank as the `remember` / `search` tools) into a
scratch collection, deletes it at the end, and writes `<bench>/results/<time>_<name>.md`
(numbers only) and `.json` (every question). Methods: `grep` (literal term match on the
same passages, like `rg -i`), `grep_stem` (5-letter stems), `dense`, `sparse`, `hybrid`
(RRF), `hybrid+rerank` (what `search` does), and `hybrid+rerank, prefer canon`.

```text
<bench>/bench.json     {"name", "slices": [{"name", "root", "include", "exclude", "canon"}], "read_first": ["<slice>/<file>"]}
<bench>/golden.jsonl   {"id", "lang": "ru|en|mixed", "query", "expect": [{"file": "<slice>/<path>", "contains": "text or [any of]"}]}
<bench>/poison.jsonl   {"id", "target": "<golden id>", "agent", "style": "plain|authority", "texts": ["false fact"]}
```

- `canon` globs (relative to the slice root) mark curated files: they are indexed as
  trust tier `canon`, everything else as `inbox`. Poison notes are saved with `texts`, so they
  are always `inbox`, like any agent write.
- A passage answers a question when it comes from one of the `expect` files and contains
  one of the `contains` strings (case, spaces and markdown ticks ignored). Questions without
  any such stored passage are reported and left out.
- Write questions the way you would ask them, not copied from the text; mix languages.
- Keep a private set under `eval/private/` (gitignored): its JSON results contain your
  questions and passage sources.

`datasets/memory_synthetic` is a public example: a fictional studio run by agents (canon,
decisions, two unreviewed inbox notes, one in Russian), plus this repository's `docs/` as
a second slice; 24 questions and 8 false notes.
