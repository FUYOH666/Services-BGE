"""Merge several eval result files into one comparison table.

    uv run python eval/compare.py eval/results/*.json
    uv run python eval/compare.py eval/results/*.json --metric code4_acc@1 --out eval/results/COMPARISON.md

Rows are (run name, retrieval mode), sorted by the chosen metric.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

COLUMNS = ["recall@1", "recall@5", "recall@10", "mrr@10", "ndcg@10"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("results", nargs="+", type=Path)
    parser.add_argument("--metric", default="ndcg@10", help="sort key")
    parser.add_argument("--out", type=Path, help="also write the table to this markdown file")
    args = parser.parse_args()

    rows = []
    code_cols: set[str] = set()
    datasets = set()
    seen: set[tuple] = set()
    for path in args.results:
        report = json.loads(path.read_text(encoding="utf-8"))
        datasets.add(Path(report["dataset"]).name)
        models = report.get("models", {})
        timing = report.get("timing", {})
        for mode, metrics in report["results"].items():
            code_cols |= {c for c in metrics if c.startswith("code")}
            reranker = models.get("rerank") if "rerank" in mode else None
            base = mode.split("+")[0]
            dense = models.get("embed") if base in ("dense", "hybrid_rrf") else None
            sparse = models.get("sparse") if base in ("sparse", "hybrid_rrf") else None
            key = (dense, sparse, mode, reranker)
            if key in seen:
                continue
            seen.add(key)
            rows.append(
                {
                    "run": report["name"],
                    "mode": mode,
                    "dense": dense,
                    "sparse": sparse,
                    "rerank": reranker,
                    "metrics": metrics,
                    "query_p50": timing.get("query_embed_p50_ms"),
                    "rerank_p50": timing.get("rerank_p50_ms") if reranker else None,
                }
            )

    rows.sort(key=lambda r: r["metrics"].get(args.metric, float("-inf")), reverse=True)
    cols = COLUMNS + sorted(code_cols)
    lines = [
        f"# Retrieval comparison · {', '.join(sorted(datasets))}",
        "",
        f"Sorted by `{args.metric}`. Latency = median per query (embedding) and per rerank call.",
        "",
        "| dense | sparse | mode | reranker | " + " | ".join(cols) + " | embed p50 | rerank p50 |",
        "|---|---|---|---|" + "---|" * (len(cols) + 2),
    ]
    for r in rows:
        vals = " | ".join(f"{r['metrics'].get(c, float('nan')):.3f}" for c in cols)
        q = f"{r['query_p50']:.0f} ms" if r["query_p50"] is not None else "—"
        rr = f"{r['rerank_p50']:.0f} ms" if r["rerank_p50"] is not None else "—"
        short = lambda m: m.split("/")[-1] if m else "—"  # noqa: E731
        lines.append(
            f"| {short(r['dense'])} | {short(r['sparse'])} | {r['mode']} | {short(r['rerank'])} | {vals} | {q} | {rr} |"
        )
    table = "\n".join(lines) + "\n"
    print(table)
    if args.out:
        args.out.write_text(table, encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
