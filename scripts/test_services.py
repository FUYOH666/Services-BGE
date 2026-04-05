"""Smoke tests for both BGE services."""
from __future__ import annotations

import json
import sys
import time

import httpx

EMBED_URL = "http://localhost:9001"
RERANK_URL = "http://localhost:9002"

TEXTS = [
    "бесшовные трубы из нержавеющей стали марки AISI 304, диаметр 50мм",
    "винты самонарезающие с шестигранной головкой из углеродистой стали",
    "молоко сухое цельное с массовой долей жира 25%",
]


def check_embedding_health():
    live = httpx.get(f"{EMBED_URL}/livez", timeout=5)
    live.raise_for_status()
    print(f"  livez:  {json.dumps(live.json(), ensure_ascii=False)}")

    r = httpx.get(f"{EMBED_URL}/readyz", timeout=5)
    r.raise_for_status()
    data = r.json()
    assert data["ready"], f"Embedding not ready: {data}"
    print(f"  readyz: {json.dumps(data, ensure_ascii=False)}")


def check_embedding():
    t0 = time.monotonic()
    r = httpx.post(
        f"{EMBED_URL}/v1/embeddings",
        json={
            "input": TEXTS,
            "return_dense": True,
            "return_sparse": True,
            "return_colbert": False,
            "max_length": 512,
        },
        timeout=60,
    )
    elapsed = time.monotonic() - t0
    r.raise_for_status()
    data = r.json()

    assert len(data["data"]) == len(TEXTS), f"Expected {len(TEXTS)} results, got {len(data['data'])}"

    item = data["data"][0]
    dense_dim = len(item["dense_embedding"])
    sparse_count = len(item["sparse_embedding"])
    qdrant_keys = set(item["sparse_qdrant"].keys())

    assert dense_dim == 1024, f"Expected 1024-dim dense, got {dense_dim}"
    assert sparse_count > 0, "Sparse embedding is empty"
    assert qdrant_keys == {"indices", "values"}, f"Unexpected sparse_qdrant keys: {qdrant_keys}"
    assert len(item["sparse_qdrant"]["indices"]) > 0, "Qdrant sparse indices empty"

    print(f"  time:           {elapsed:.2f} s ({len(TEXTS)/elapsed:.1f} texts/s)")
    print(f"  results:        {len(data['data'])}")
    print(f"  dense dim:      {dense_dim}")
    print(f"  sparse tokens:  {sparse_count}")
    print(f"  qdrant indices: {len(item['sparse_qdrant']['indices'])}")


def check_reranker_health():
    live = httpx.get(f"{RERANK_URL}/livez", timeout=5)
    live.raise_for_status()
    print(f"  livez:  {json.dumps(live.json(), ensure_ascii=False)}")

    r = httpx.get(f"{RERANK_URL}/readyz", timeout=5)
    r.raise_for_status()
    data = r.json()
    assert data["ready"], f"Reranker not ready: {data}"
    print(f"  readyz: {json.dumps(data, ensure_ascii=False)}")


def check_reranker():
    query = "бесшовные трубы из нержавеющей стали"
    documents = [
        "7304 — Трубы, трубки и профили полые, бесшовные, из черных металлов (кроме чугунного литья)",
        "7306 — Трубы, трубки и профили полые прочие (например, с открытым швом, сварные, клепаные)",
        "7318 — Винты, болты, гайки, глухари, ввертные крюки, заклепки",
        "0402 — Молоко и сливки, сгущенные или с добавлением сахара",
        "7307 — Фитинги для труб из черных металлов",
    ]

    t0 = time.monotonic()
    r = httpx.post(
        f"{RERANK_URL}/v1/rerank",
        json={"query": query, "documents": documents, "normalize": True},
        timeout=60,
    )
    elapsed = time.monotonic() - t0
    r.raise_for_status()
    data = r.json()

    assert len(data["results"]) == len(documents)

    top = data["results"][0]
    bottom = data["results"][-1]
    assert top["relevance_score"] > bottom["relevance_score"], "Top should score higher than bottom"

    print(f"  time:  {elapsed:.2f} s ({len(documents)/elapsed:.1f} pairs/s)")
    for res in data["results"]:
        print(f"  [{res['index']}] score={res['relevance_score']:.4f} — {res['document'][:60]}...")


def main():
    ok = True
    for name, fn in [
        ("Embedding Health", check_embedding_health),
        ("Embedding Encode", check_embedding),
        ("Reranker Health", check_reranker_health),
        ("Reranker Score", check_reranker),
    ]:
        print(f"\n=== {name} ===")
        try:
            fn()
            print(f"  PASS")
        except Exception as e:
            print(f"  FAIL: {e}")
            ok = False

    print("\n" + ("=== All tests passed ===" if ok else "=== SOME TESTS FAILED ==="))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
