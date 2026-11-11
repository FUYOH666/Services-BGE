"""eval/memory_bench.py on the synthetic set, with fake models and in-memory Qdrant.

Checks that the public example bench is consistent (every expected passage exists),
that the grep baseline behaves like a term match, and that a full run (index, all
methods, poisoning, cleanup) works without the network.
"""
from __future__ import annotations

import argparse
import asyncio
import importlib.util
import sys
from pathlib import Path

from test_mcp_server import make_memory

ROOT = Path(__file__).resolve().parent.parent
SYNTHETIC = ROOT / "eval" / "datasets" / "memory_synthetic"

_spec = importlib.util.spec_from_file_location("memory_bench", ROOT / "eval" / "memory_bench.py")
bench = importlib.util.module_from_spec(_spec)
sys.modules["memory_bench"] = bench  # dataclasses look the module up while it loads
_spec.loader.exec_module(bench)


def test_synthetic_bench_is_consistent():
    config, slices, golden, poison = bench.load_bench(SYNTHETIC)
    files = {bench.file_id(slices, p): p for s in slices for p in s.files}
    assert {"kestrel/CANON.md", "docs/AGENT-MEMORY.md"} <= set(files)
    assert "docs/MEMORY-BENCH.md" not in files  # results must not leak into the corpus
    assert set(config["read_first"]) <= set(files)
    for g in golden:
        texts = {f: bench.norm(files[f].read_text(encoding="utf-8")) for f, _ in g.expect if f in files}
        assert any(any(c in texts.get(f, "") for c in contains) for f, contains in g.expect), g.id
    assert {g.lang for g in golden} == set(bench.LANGS)
    assert {p.target for p in poison} <= {g.id for g in golden}
    canon = {bench.file_id(slices, p) for s in slices for p in s.canon}
    assert "kestrel/STATE.md" in canon and not any(f.startswith("kestrel/inbox/") for f in canon)


def test_grep_baseline_is_a_term_match():
    assert bench.grep_terms("What is the pilot price for logistics firms?", stem=False) == ["pilot", "price", "logistics", "firms"]
    assert bench.grep_terms("Сколько стоит пилотов", stem=True) == ["стоит", "пилот"]
    passages = [
        bench.Hit("a.md", "a", "", haystack="the pilot runs four weeks"),
        bench.Hit("b.md", "b", "", haystack="pilot price: eur 900, pilot price floor"),
        bench.Hit("c.md", "c", "", haystack="nothing relevant"),
    ]
    ranked = bench.grep_rank("pilot price", passages, stem=False)
    assert [h.source for h in ranked] == ["b", "a"]


def test_full_run_with_fake_models(tmp_path, capsys):
    memory, services = make_memory(tmp_path)
    args = argparse.Namespace(
        bench=SYNTHETIC, out=tmp_path / "results", collection="bench-test", keep=False,
        no_poison=False, penalties=[0.5],
    )
    report = asyncio.run(bench.run(args, memory=memory))

    assert report["golden"]["left_out"] == []
    assert set(report["clean"]) == set(bench.CLEAN_METHODS)
    for method in bench.CLEAN_METHODS:
        assert 0 <= report["clean"][method]["all"]["recall@1"] <= report["clean"][method]["all"]["recall@5"] <= 1
    assert report["index"]["passages_by_tier"]["inbox"] > 0
    assert report["context"]["read_first_chars"] > 0
    configs = report["poison"]["configs"]
    assert set(configs) == {
        "hybrid+rerank (no tiers)", "hybrid+rerank, prefer canon x0.5", "hybrid+rerank, canon only",
        "hybrid (no rerank, no tiers)", "hybrid, prefer canon x0.5",
    }
    assert configs["hybrid+rerank, canon only"]["false_in_top5"] == 0
    assert configs["hybrid+rerank, prefer canon x0.5"]["false_first"] <= configs["hybrid+rerank (no tiers)"]["false_first"]
    # The scratch collection is dropped and the summary holds numbers, not question text.
    assert "dropped scratch collection 'bench-test'" in capsys.readouterr().err
    summary = next((tmp_path / "results").glob("*.md")).read_text(encoding="utf-8")
    assert "## Poisoning" in summary and "pilot" not in summary.lower()
