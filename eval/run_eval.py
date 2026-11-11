"""Retrieval evaluation harness for any embedding + rerank endpoints.

Measures dense, sparse, hybrid (RRF) and reranked retrieval on a JSONL dataset, so
model or config changes can be judged on your own data instead of public leaderboards.

    uv run python eval/run_eval.py --dataset eval/datasets/demo_ru --name bge-baseline

    # a challenger on side ports, same dataset
    uv run python eval/run_eval.py --dataset eval/datasets/demo_ru \\
        --embed-url http://127.0.0.1:9011 --rerank-url http://127.0.0.1:9012 --name qwen3-challenger

Dataset layout (see eval/README.md):
    corpus.jsonl   {"id": str, "text": str, "hs_code"?: str}
    queries.jsonl  {"id": str, "query": str, "positives": [corpus ids], "hs_code"?: str}

Endpoints only need the OpenAI-style `/v1/embeddings` (with optional `sparse_qdrant`)
and the `/v1/rerank` shape used by this project.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx
import numpy as np

HEADERS = {"Authorization": f"Bearer {os.environ['SERVICE_API_KEY']}"} if os.environ.get("SERVICE_API_KEY") else {}
RRF_K = 60
CUTOFFS = (1, 5, 10, 50)


# --- data ----------------------------------------------------------------------


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


class Encoded:
    def __init__(self, dense: np.ndarray | None, sparse: list[dict[int, float]] | None):
        self.dense = dense
        self.sparse = sparse


def encode(client: httpx.Client, url: str, texts: list[str], batch: int, latencies: list[float], max_length: int | None) -> Encoded:
    dense_rows: list[list[float]] = []
    sparse_rows: list[dict[int, float]] = []
    has_sparse = True
    for start in range(0, len(texts), batch):
        chunk = texts[start : start + batch]
        payload: dict = {"input": chunk, "return_dense": True, "return_sparse": True}
        if max_length:
            payload["max_length"] = max_length
        t0 = time.perf_counter()
        r = client.post(f"{url}/v1/embeddings", json=payload, headers=HEADERS)
        latencies.append(time.perf_counter() - t0)
        r.raise_for_status()
        for item in sorted(r.json()["data"], key=lambda x: x["index"]):
            dense_rows.append(item.get("embedding") or item.get("dense_embedding"))
            sq = item.get("sparse_qdrant")
            if sq is None:
                has_sparse = False
            else:
                sparse_rows.append(dict(zip(sq["indices"], sq["values"])))

    dense = np.asarray(dense_rows, dtype=np.float32)
    norms = np.linalg.norm(dense, axis=1, keepdims=True)
    dense = dense / np.where(norms == 0, 1, norms)
    return Encoded(dense, sparse_rows if has_sparse else None)


# --- scoring ---------------------------------------------------------------------


def sparse_scores(query: dict[int, float], docs: list[dict[int, float]]) -> np.ndarray:
    """BGE-M3 lexical matching: sum of q_w * d_w over shared token ids."""
    return np.asarray([sum(w * d.get(t, 0.0) for t, w in query.items()) for d in docs], dtype=np.float32)


def rrf(rankings: list[list[int]], weights: list[float] | None = None, k: int = RRF_K) -> list[int]:
    """Reciprocal rank fusion; weights mirror Qdrant's weighted RRF (1.17+)."""
    weights = weights or [1.0] * len(rankings)
    scores: dict[int, float] = {}
    for ranking, weight in zip(rankings, weights):
        for rank, doc in enumerate(ranking):
            scores[doc] = scores.get(doc, 0.0) + weight / (k + rank + 1)
    return sorted(scores, key=scores.get, reverse=True)


def metrics_for(rankings: list[list[int]], positives: list[set[int]]) -> dict[str, float]:
    out: dict[str, float] = {}
    for k in CUTOFFS:
        out[f"recall@{k}"] = statistics.fmean(
            len(set(r[:k]) & p) / len(p) if p else 0.0 for r, p in zip(rankings, positives)
        )
    mrr, ndcg = [], []
    for ranking, pos in zip(rankings, positives):
        rr = 0.0
        for i, doc in enumerate(ranking[:10]):
            if doc in pos:
                rr = 1.0 / (i + 1)
                break
        mrr.append(rr)
        dcg = sum(1.0 / math.log2(i + 2) for i, doc in enumerate(ranking[:10]) if doc in pos)
        idcg = sum(1.0 / math.log2(i + 2) for i in range(min(len(pos), 10)))
        ndcg.append(dcg / idcg if idcg else 0.0)
    out["mrr@10"] = statistics.fmean(mrr)
    out["ndcg@10"] = statistics.fmean(ndcg)
    return out


def code_accuracy(rankings: list[list[int]], query_codes: list[str | None], doc_codes: list[str | None]) -> dict[str, float]:
    """Top-1/top-5 accuracy of the retrieved document's code at several prefix lengths."""
    digits = lambda s: "".join(ch for ch in (s or "") if ch.isdigit())  # noqa: E731
    out: dict[str, float] = {}
    pairs = [(r, digits(q)) for r, q in zip(rankings, query_codes) if q]
    if not pairs:
        return out
    max_len = max(len(code) for _, code in pairs)
    for level in (2, 4, 6, 10):
        if level > max_len:
            break
        for k in (1, 5):
            hits = []
            for ranking, code in pairs:
                if len(code) < level:
                    continue
                found = any(digits(doc_codes[d])[:level] == code[:level] for d in ranking[:k])
                hits.append(1.0 if found else 0.0)
            if hits:
                out[f"code{level}_acc@{k}"] = statistics.fmean(hits)
    return out


# --- main ------------------------------------------------------------------------


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, math.ceil(q * len(ordered)) - 1))
    return ordered[idx]


def main() -> int:
    parser = argparse.ArgumentParser(description="Retrieval eval for embedding + rerank endpoints")
    parser.add_argument("--dataset", required=True, type=Path, help="folder with corpus.jsonl and queries.jsonl")
    parser.add_argument("--embed-url", default="http://127.0.0.1:9001")
    parser.add_argument("--query-embed-url", help="separate endpoint for queries (defaults to --embed-url)")
    parser.add_argument("--rerank-url", default="http://127.0.0.1:9002", help="empty string disables rerank")
    parser.add_argument("--rerank-k", type=int, default=20, help="candidates passed to the reranker")
    parser.add_argument("--dense-weight", type=float, default=1.0, help="dense weight in RRF (sparse weight is 1)")
    parser.add_argument("--retrieve-k", type=int, default=50, help="depth of retrieved lists")
    parser.add_argument("--batch", type=int, default=32, help="texts per embedding request")
    parser.add_argument("--max-length", type=int, help="override token limit sent to the embedder")
    parser.add_argument("--name", default="run", help="label used in the result file name")
    parser.add_argument("--out", type=Path, default=Path(__file__).parent / "results")
    args = parser.parse_args()

    corpus = load_jsonl(args.dataset / "corpus.jsonl")
    queries = load_jsonl(args.dataset / "queries.jsonl")
    id_to_idx = {doc["id"]: i for i, doc in enumerate(corpus)}
    positives = [{id_to_idx[p] for p in q["positives"] if p in id_to_idx} for q in queries]
    embed_url = args.embed_url.rstrip("/")
    query_url = (args.query_embed_url or args.embed_url).rstrip("/")
    rerank_url = args.rerank_url.rstrip("/")
    k = args.retrieve_k

    embed_lat: list[float] = []
    query_lat: list[float] = []
    rerank_lat: list[float] = []

    with httpx.Client(timeout=600) as client:
        t0 = time.perf_counter()
        docs = encode(client, embed_url, [d["text"] for d in corpus], args.batch, embed_lat, args.max_length)
        index_seconds = time.perf_counter() - t0
        qs = encode(client, query_url, [q["query"] for q in queries], 1, query_lat, args.max_length)

        dense_rank = [list(np.argsort(-(docs.dense @ qv))[:k]) for qv in qs.dense]
        runs: dict[str, list[list[int]]] = {"dense": dense_rank}

        if docs.sparse is not None and qs.sparse is not None:
            sparse_rank = [list(np.argsort(-sparse_scores(qsp, docs.sparse))[:k]) for qsp in qs.sparse]
            runs["sparse"] = sparse_rank
            runs["hybrid_rrf"] = [
                rrf([d, s], [args.dense_weight, 1.0])[:k] for d, s in zip(dense_rank, sparse_rank)
            ]

        if rerank_url:
            for base_name in [name for name in ("dense", "hybrid_rrf") if name in runs]:
                reranked = []
                for q, candidates in zip(queries, runs[base_name]):
                    head = [int(c) for c in candidates[: args.rerank_k]]
                    t1 = time.perf_counter()
                    r = client.post(
                        f"{rerank_url}/v1/rerank",
                        json={"query": q["query"], "documents": [corpus[c]["text"] for c in head], "normalize": True},
                        headers=HEADERS,
                    )
                    rerank_lat.append(time.perf_counter() - t1)
                    r.raise_for_status()
                    order = [head[item["index"]] for item in r.json()["results"]]
                    reranked.append(order + [int(c) for c in candidates[args.rerank_k :]])
                runs[f"{base_name}+rerank@{args.rerank_k}"] = reranked

        model_info = {}
        for label, url, path in (("embed", embed_url, "/readyz"), ("rerank", rerank_url, "/readyz")):
            if not url:
                continue
            try:
                model_info[label] = client.get(f"{url}{path}", timeout=5).json().get("model")
            except Exception:
                model_info[label] = None

    query_codes = [q.get("hs_code") for q in queries]
    doc_codes = [d.get("hs_code") for d in corpus]
    results = {}
    for name, rankings in runs.items():
        rankings = [[int(x) for x in r] for r in rankings]
        results[name] = {**metrics_for(rankings, positives), **code_accuracy(rankings, query_codes, doc_codes)}

    report = {
        "name": args.name,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dataset": str(args.dataset),
        "corpus_size": len(corpus),
        "queries": len(queries),
        "endpoints": {"embed": embed_url, "query_embed": query_url, "rerank": rerank_url or None},
        "models": model_info,
        "params": {"retrieve_k": k, "rerank_k": args.rerank_k, "batch": args.batch, "max_length": args.max_length, "rrf_k": RRF_K, "dense_weight": args.dense_weight},
        "timing": {
            "index_seconds": index_seconds,
            "query_embed_p50_ms": percentile(query_lat, 0.5) * 1000,
            "query_embed_p95_ms": percentile(query_lat, 0.95) * 1000,
            "rerank_p50_ms": percentile(rerank_lat, 0.5) * 1000 if rerank_lat else None,
            "rerank_p95_ms": percentile(rerank_lat, 0.95) * 1000 if rerank_lat else None,
        },
        "results": results,
    }

    args.out.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    base = args.out / f"{stamp}_{args.dataset.name}_{args.name}"
    base.with_suffix(".json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    metric_cols = ["recall@1", "recall@5", "recall@10", "mrr@10", "ndcg@10"]
    code_cols = sorted({c for r in results.values() for c in r if c.startswith("code")})
    header = "| run | " + " | ".join(metric_cols + code_cols) + " |"
    lines = [
        f"# Eval: {args.name} on {args.dataset.name}",
        "",
        f"{report['created_at']} · corpus {len(corpus)} · queries {len(queries)} · models {model_info}",
        "",
        header,
        "|" + "---|" * (len(metric_cols) + len(code_cols) + 1),
    ]
    for name, vals in results.items():
        lines.append(f"| {name} | " + " | ".join(f"{vals.get(c, float('nan')):.3f}" for c in metric_cols + code_cols) + " |")
    t = report["timing"]
    lines += [
        "",
        f"Index {t['index_seconds']:.1f} s · query embed p50 {t['query_embed_p50_ms']:.0f} ms / p95 {t['query_embed_p95_ms']:.0f} ms"
        + (f" · rerank p50 {t['rerank_p50_ms']:.0f} ms / p95 {t['rerank_p95_ms']:.0f} ms" if t["rerank_p50_ms"] is not None else ""),
    ]
    markdown = "\n".join(lines) + "\n"
    base.with_suffix(".md").write_text(markdown, encoding="utf-8")
    print(markdown)
    print(f"Saved {base}.json and .md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
