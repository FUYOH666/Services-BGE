"""Benchmark CPU vs MPS for BGE-M3 and BGE-Reranker on Apple Silicon."""
from __future__ import annotations

import gc
import os
import time

import psutil
import torch

os.environ["TOKENIZERS_PARALLELISM"] = "false"

TEXTS = [
    "бесшовные трубы из нержавеющей стали марки AISI 304, диаметр 50мм",
    "винты самонарезающие с шестигранной головкой из углеродистой стали",
    "молоко сухое цельное с массовой долей жира 25%",
    "подшипники шариковые радиальные однорядные закрытого типа",
    "кабель силовой медный с изоляцией из сшитого полиэтилена",
    "масло моторное синтетическое SAE 5W-30 API SN",
    "фитинги резьбовые из нержавеющей стали для трубопроводов",
    "электродвигатель асинхронный трёхфазный мощностью 5.5 кВт",
]

PAIRS = [
    ["бесшовные трубы из нержавеющей стали", doc]
    for doc in [
        "7304 — Трубы, трубки и профили полые, бесшовные, из черных металлов",
        "7306 — Трубы, трубки и профили полые прочие (сварные, клепаные)",
        "7318 — Винты, болты, гайки, глухари, ввертные крюки, заклепки",
        "0402 — Молоко и сливки, сгущенные или с добавлением сахара",
        "7307 — Фитинги для труб из черных металлов",
        "7326 — Изделия прочие из черных металлов",
        "7304 — Трубы бесшовные из коррозионностойкой стали",
        "8482 — Подшипники шариковые или роликовые",
    ]
]


def mem_mb() -> float:
    return psutil.Process().memory_info().rss / 1024 / 1024


def benchmark_embedding(device: str, rounds: int = 5):
    print(f"\n{'='*60}")
    print(f"  BGE-M3 Embedding — device={device}")
    print(f"{'='*60}")

    mem_before = mem_mb()
    from FlagEmbedding import BGEM3FlagModel

    kwargs = {"use_fp16": False}
    if device == "mps":
        kwargs["devices"] = ["mps"]

    t0 = time.monotonic()
    model = BGEM3FlagModel("BAAI/bge-m3", **kwargs)
    load_time = time.monotonic() - t0
    mem_after_load = mem_mb()
    print(f"  Load time:    {load_time:.1f} s")
    print(f"  RAM after load: {mem_after_load:.0f} MB (+{mem_after_load - mem_before:.0f} MB)")

    # Warmup
    model.encode(TEXTS[:1], batch_size=1, max_length=256, return_dense=True, return_sparse=True, return_colbert_vecs=False)
    if device == "mps":
        torch.mps.empty_cache()

    times = []
    for i in range(rounds):
        t0 = time.monotonic()
        model.encode(TEXTS, batch_size=12, max_length=512, return_dense=True, return_sparse=True, return_colbert_vecs=False)
        elapsed = time.monotonic() - t0
        times.append(elapsed)
        if device == "mps":
            torch.mps.empty_cache()

    mem_after_bench = mem_mb()
    avg = sum(times) / len(times)
    print(f"  Avg latency:  {avg*1000:.0f} ms / {len(TEXTS)} texts")
    print(f"  Throughput:   {len(TEXTS)/avg:.1f} texts/s")
    print(f"  Min/Max:      {min(times)*1000:.0f} / {max(times)*1000:.0f} ms")
    print(f"  RAM after bench: {mem_after_bench:.0f} MB (+{mem_after_bench - mem_after_load:.0f} MB leak)")

    del model
    gc.collect()
    if device == "mps":
        torch.mps.empty_cache()


def benchmark_reranker(device: str, rounds: int = 5):
    print(f"\n{'='*60}")
    print(f"  BGE-Reranker-v2-m3 — device={device}")
    print(f"{'='*60}")

    mem_before = mem_mb()
    from FlagEmbedding import FlagReranker

    kwargs = {"use_fp16": False}
    if device == "mps":
        kwargs["devices"] = ["mps"]

    t0 = time.monotonic()
    model = FlagReranker("BAAI/bge-reranker-v2-m3", **kwargs)
    load_time = time.monotonic() - t0
    mem_after_load = mem_mb()
    print(f"  Load time:    {load_time:.1f} s")
    print(f"  RAM after load: {mem_after_load:.0f} MB (+{mem_after_load - mem_before:.0f} MB)")

    # Warmup
    model.compute_score([PAIRS[0]], normalize=True)
    if device == "mps":
        torch.mps.empty_cache()

    times = []
    for i in range(rounds):
        t0 = time.monotonic()
        model.compute_score(PAIRS, normalize=True)
        elapsed = time.monotonic() - t0
        times.append(elapsed)
        if device == "mps":
            torch.mps.empty_cache()

    mem_after_bench = mem_mb()
    avg = sum(times) / len(times)
    print(f"  Avg latency:  {avg*1000:.0f} ms / {len(PAIRS)} pairs")
    print(f"  Throughput:   {len(PAIRS)/avg:.1f} pairs/s")
    print(f"  Min/Max:      {min(times)*1000:.0f} / {max(times)*1000:.0f} ms")
    print(f"  RAM after bench: {mem_after_bench:.0f} MB (+{mem_after_bench - mem_after_load:.0f} MB leak)")

    del model
    gc.collect()
    if device == "mps":
        torch.mps.empty_cache()


if __name__ == "__main__":
    print(f"PyTorch {torch.__version__}, MPS: {torch.backends.mps.is_available()}")
    print(f"Initial RAM: {mem_mb():.0f} MB")

    benchmark_embedding("cpu")
    benchmark_embedding("mps")
    benchmark_reranker("cpu")
    benchmark_reranker("mps")

    print(f"\nFinal RAM: {mem_mb():.0f} MB")
