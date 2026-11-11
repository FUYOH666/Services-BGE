"""Compare two running deployments of the services (A = reference, B = candidate).

Use it to prove a change is backward compatible (same model, same numbers) or to
quantify drift from a deliberate change such as fp16 or a library upgrade.

    uv run python scripts/parity_check.py \\
        --a-embed http://127.0.0.1:9001 --b-embed http://127.0.0.1:9101 \\
        --a-rerank http://127.0.0.1:9002 --b-rerank http://127.0.0.1:9102 \\
        --json /tmp/parity.json

Exit code is 0 when every threshold passes, 1 otherwise. Set SERVICE_API_KEY if the
services require a key.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import statistics
import subprocess
import sys
import time
from urllib.parse import urlparse

import httpx

SHORT_TEXTS = [
    "бесшовные трубы из нержавеющей стали марки AISI 304, диаметр 50мм",
    "винты самонарезающие с шестигранной головкой из углеродистой стали",
    "молоко сухое цельное с массовой долей жира 25%",
    "подшипники шариковые радиальные однорядные закрытого типа",
    "кабель силовой медный с изоляцией из сшитого полиэтилена",
    "масло моторное синтетическое SAE 5W-30 API SN",
    "Stainless steel seamless pipe, outer diameter 50 mm, wall 3 mm",
    "Ball bearing, deep groove, single row, sealed, 6204-2RS",
]

LONG_PARAGRAPH = (
    "Описание товара для таможенного декларирования: изделие представляет собой "
    "кабель силовой медный многожильный с изоляцией из сшитого полиэтилена, "
    "номинальное напряжение 1 кВ, сечение жил 3x95 мм2, оболочка из ПВХ пластиката "
    "пониженной горючести, предназначен для передачи и распределения электроэнергии. "
)
LONG_TEXTS = [f"{i}. " + LONG_PARAGRAPH * 8 for i in range(8)]

RERANK_QUERIES = [
    "бесшовные трубы из нержавеющей стали",
    "подшипник шариковый для электродвигателя",
    "сухое молоко для пищевой промышленности",
]
RERANK_DOCS = [
    "7304 — Трубы, трубки и профили полые, бесшовные, из черных металлов (кроме чугунного литья)",
    "7306 — Трубы, трубки и профили полые прочие (например, с открытым швом, сварные, клепаные)",
    "7307 — Фитинги для труб из черных металлов",
    "7318 — Винты, болты, гайки, глухари, ввертные крюки, заклепки, шплинты, шайбы",
    "7326 — Изделия прочие из черных металлов",
    "8482 — Подшипники шариковые или роликовые",
    "8483 — Валы трансмиссионные, корпуса подшипников, подшипники скольжения, шестерни",
    "8501 — Двигатели и генераторы электрические",
    "8544 — Провода изолированные, кабели и другие изолированные электрические проводники",
    "0401 — Молоко и сливки, несгущенные и без добавления сахара",
    "0402 — Молоко и сливки, сгущенные или с добавлением сахара или других подслащивающих веществ",
    "0404 — Молочная сыворотка; продукты из натуральных компонентов молока",
    "1901 — Экстракты солодовые; готовые пищевые продукты из муки, крупы, молока",
    "2710 — Нефть и нефтепродукты, масла смазочные",
    "3926 — Изделия прочие из пластмасс",
    "4011 — Шины и покрышки пневматические резиновые новые",
]

HEADERS = {"Authorization": f"Bearer {os.environ['SERVICE_API_KEY']}"} if os.environ.get("SERVICE_API_KEY") else {}


# --- small numeric helpers (no numpy needed) ---------------------------------


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda i: values[i])
    result = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg = (i + j) / 2
        for k in range(i, j + 1):
            result[order[k]] = avg
        i = j + 1
    return result


def spearman(a: list[float], b: list[float]) -> float:
    ra, rb = ranks(a), ranks(b)
    ma, mb = statistics.fmean(ra), statistics.fmean(rb)
    cov = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    va = math.sqrt(sum((x - ma) ** 2 for x in ra))
    vb = math.sqrt(sum((y - mb) ** 2 for y in rb))
    return cov / (va * vb) if va and vb else 1.0


def jaccard(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if a | b else 1.0


# --- service calls -------------------------------------------------------------


def embed(client: httpx.Client, base: str, texts: list[str], **extra) -> tuple[dict, float, int]:
    payload = {"input": texts, "return_dense": True, "return_sparse": True, **extra}
    t0 = time.perf_counter()
    r = client.post(f"{base}/v1/embeddings", json=payload, headers=HEADERS)
    elapsed = time.perf_counter() - t0
    r.raise_for_status()
    return r.json(), elapsed, len(r.content)


def rerank(client: httpx.Client, base: str, query: str, docs: list[str]) -> tuple[list[float], float]:
    t0 = time.perf_counter()
    r = client.post(f"{base}/v1/rerank", json={"query": query, "documents": docs, "normalize": True}, headers=HEADERS)
    elapsed = time.perf_counter() - t0
    r.raise_for_status()
    scores = [0.0] * len(docs)
    for item in r.json()["results"]:
        scores[item["index"]] = item["relevance_score"]
    return scores, elapsed


def median_latency_ms(fn, rounds: int) -> float:
    fn()  # warm
    samples = []
    for _ in range(rounds):
        t0 = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - t0)
    return statistics.median(samples) * 1000


def footprint_mb(base_url: str) -> float | None:
    """macOS only: physical footprint of the process listening on the URL's port."""
    port = urlparse(base_url).port
    try:
        pid = subprocess.run(
            ["lsof", "-ti", f"tcp:{port}", "-sTCP:LISTEN"], capture_output=True, text=True, timeout=5
        ).stdout.split()[0]
        out = subprocess.run(["footprint", "-p", pid], capture_output=True, text=True, timeout=20).stdout
        match = re.search(r"Footprint:\s*([\d.]+)\s*([KMG]B)", out)
        if not match:
            return None
        value, unit = float(match.group(1)), match.group(2)
        return value * {"KB": 1 / 1024, "MB": 1, "GB": 1024}[unit]
    except Exception:
        return None


# --- checks --------------------------------------------------------------------


def check_embeddings(client: httpx.Client, a: str, b: str, rounds: int) -> dict:
    texts = SHORT_TEXTS + LONG_TEXTS
    ra, _, size_a = embed(client, a, texts)
    rb, _, size_b = embed(client, b, texts)

    cosines, sparse_j, sparse_dmax = [], [], []
    for ia, ib in zip(ra["data"], rb["data"]):
        va = ia.get("embedding") or ia.get("dense_embedding")
        vb = ib.get("embedding") or ib.get("dense_embedding")
        cosines.append(cosine(va, vb))
        sa, sb = ia.get("sparse_qdrant"), ib.get("sparse_qdrant")
        if sa and sb:
            wa = dict(zip(sa["indices"], sa["values"]))
            wb = dict(zip(sb["indices"], sb["values"]))
            top_a = set(sorted(wa, key=wa.get, reverse=True)[:10])
            top_b = set(sorted(wb, key=wb.get, reverse=True)[:10])
            sparse_j.append(jaccard(top_a, top_b))
            sparse_dmax.append(max(abs(wa.get(k, 0.0) - wb.get(k, 0.0)) for k in set(wa) | set(wb)))

    sizes = {"a_default_kb": size_a / 1024, "b_default_kb": size_b / 1024}
    for label, extra in {"b_compact_kb": {"compact": True}, "b_compact_base64_kb": {"compact": True, "encoding_format": "base64"}}.items():
        try:
            _, _, size = embed(client, b, SHORT_TEXTS + LONG_TEXTS, **extra)
            sizes[label] = size / 1024
        except httpx.HTTPStatusError:
            sizes[label] = None

    latency = {}
    for side, base in (("a", a), ("b", b)):
        latency[f"{side}_8_short_ms"] = median_latency_ms(lambda: embed(client, base, SHORT_TEXTS), rounds)
        latency[f"{side}_8_long_ms"] = median_latency_ms(lambda: embed(client, base, LONG_TEXTS), rounds)

    return {
        "dense_cos_min": min(cosines),
        "dense_cos_mean": statistics.fmean(cosines),
        "sparse_top10_jaccard_mean": statistics.fmean(sparse_j) if sparse_j else None,
        "sparse_weight_absdiff_max": max(sparse_dmax) if sparse_dmax else None,
        "sizes": sizes,
        "latency": latency,
    }


def check_rerank(client: httpx.Client, a: str, b: str, rounds: int) -> dict:
    rhos, top5_equal, absdiff = [], [], []
    for query in RERANK_QUERIES:
        sa, _ = rerank(client, a, query, RERANK_DOCS)
        sb, _ = rerank(client, b, query, RERANK_DOCS)
        rhos.append(spearman(sa, sb))
        top_a = sorted(range(len(sa)), key=lambda i: -sa[i])[:5]
        top_b = sorted(range(len(sb)), key=lambda i: -sb[i])[:5]
        top5_equal.append(top_a == top_b)
        absdiff.append(max(abs(x - y) for x, y in zip(sa, sb)))

    latency = {}
    for side, base in (("a", a), ("b", b)):
        latency[f"{side}_8_pairs_ms"] = median_latency_ms(lambda: rerank(client, base, RERANK_QUERIES[0], RERANK_DOCS[:8]), rounds)
        latency[f"{side}_32_pairs_ms"] = median_latency_ms(
            lambda: rerank(client, base, RERANK_QUERIES[0], RERANK_DOCS * 2), rounds
        )

    return {
        "spearman_min": min(rhos),
        "top5_identical": all(top5_equal),
        "score_absdiff_max": max(absdiff),
        "latency": latency,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--a-embed", default="http://127.0.0.1:9001")
    parser.add_argument("--b-embed", default="http://127.0.0.1:9101")
    parser.add_argument("--a-rerank", default="http://127.0.0.1:9002")
    parser.add_argument("--b-rerank", default="http://127.0.0.1:9102")
    parser.add_argument("--skip-embed", action="store_true")
    parser.add_argument("--skip-rerank", action="store_true")
    parser.add_argument("--min-cos", type=float, default=0.999, help="minimum dense cosine A vs B")
    parser.add_argument("--min-sparse-jaccard", type=float, default=0.9, help="mean top-10 sparse token overlap")
    parser.add_argument("--min-spearman", type=float, default=0.99, help="minimum rerank rank correlation")
    parser.add_argument("--rounds", type=int, default=5, help="latency samples per measurement")
    parser.add_argument("--json", help="write the full report to this path")
    args = parser.parse_args()

    report: dict = {"a": {}, "b": {}, "checks": {}}
    failures: list[str] = []

    with httpx.Client(timeout=300) as client:
        if not args.skip_embed:
            emb = check_embeddings(client, args.a_embed.rstrip("/"), args.b_embed.rstrip("/"), args.rounds)
            report["embedding"] = emb
            report["a"]["embed_footprint_mb"] = footprint_mb(args.a_embed)
            report["b"]["embed_footprint_mb"] = footprint_mb(args.b_embed)
            if emb["dense_cos_min"] < args.min_cos:
                failures.append(f"dense cosine min {emb['dense_cos_min']:.6f} < {args.min_cos}")
            if emb["sparse_top10_jaccard_mean"] is not None and emb["sparse_top10_jaccard_mean"] < args.min_sparse_jaccard:
                failures.append(f"sparse top-10 jaccard {emb['sparse_top10_jaccard_mean']:.3f} < {args.min_sparse_jaccard}")

        if not args.skip_rerank:
            rr = check_rerank(client, args.a_rerank.rstrip("/"), args.b_rerank.rstrip("/"), args.rounds)
            report["rerank"] = rr
            report["a"]["rerank_footprint_mb"] = footprint_mb(args.a_rerank)
            report["b"]["rerank_footprint_mb"] = footprint_mb(args.b_rerank)
            if rr["spearman_min"] < args.min_spearman:
                failures.append(f"rerank spearman min {rr['spearman_min']:.4f} < {args.min_spearman}")
            if not rr["top5_identical"]:
                failures.append("rerank top-5 order differs")

    report["failures"] = failures
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2, ensure_ascii=False)

    print("\nPARITY:", "PASS" if not failures else "FAIL\n  - " + "\n  - ".join(failures))
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
