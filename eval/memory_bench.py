"""Agent memory bench: does shared memory find the right fact, and can an unreviewed write poison it?

Indexes a corpus into a scratch collection with the MCP server's own code
(`mcp_server.memory.Memory`: the chunking, payloads, hybrid query and rerank behind
the `remember` and `search` tools), runs a golden question set and reports, per
language (ru / en / mixed):

- hit rate at 1 and 5 (recall@1, recall@5) and MRR@10 for a grep baseline (literal
  term match, like `rg -i`), grep on 5-letter stems, dense only, sparse only, hybrid
  RRF and hybrid + rerank;
- context cost: characters of the top-5 passages vs reading the "read first" files;
- poisoning: false facts written as unreviewed notes (tier `inbox`) by a bot source;
  how often one ranks first, with and without trust-tier weighting.

    uv run python eval/memory_bench.py --bench eval/datasets/memory_synthetic
    uv run python eval/memory_bench.py --bench eval/private/<your-bench>      # gitignored

Bench folder (eval/README.md#agent-memory-bench):
    bench.json      slices (folders of .md files), which files are canon, read-first list
    golden.jsonl    {"id", "lang": "ru|en|mixed", "query", "expect": [{"file": "<slice>/<path>", "contains": "..."}]}
    poison.jsonl    {"id", "target": "<golden id>", "agent", "style": "plain|authority", "texts": [...]}

Needs the embedding (:9001) and rerank (:9002) services and Qdrant (:6333). Writes only
to a scratch collection `mcp_bench-*` and deletes it at the end (unless --keep).
Results go to <bench>/results/: a markdown summary with numbers only, and a JSON file
with every question, which stays private when the bench folder is.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import re
import statistics
import sys
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from qdrant_client import models  # noqa: E402

from mcp_server import keyword  # noqa: E402
from mcp_server.chunking import estimate_tokens  # noqa: E402
from mcp_server.config import MCPSettings  # noqa: E402
from mcp_server.memory import Memory  # noqa: E402

TOP_K = 10
LANGS = ("ru", "en", "mixed")
CLEAN_METHODS = ("grep", "grep_stem", "dense", "sparse", "hybrid", "hybrid+rerank", "hybrid+rerank, prefer canon")

_SPACES = re.compile(r"\s+")


def norm(text: str) -> str:
    """Lower case, no markdown emphasis or code ticks, single spaces."""
    text = text.replace("`", "").replace("*", "").replace(" ", " ").replace(" ", " ")
    return _SPACES.sub(" ", text).strip().lower()


# --- data ------------------------------------------------------------------------------


@dataclass
class Slice:
    name: str
    root: Path
    files: list[Path]
    canon: set[Path]


@dataclass
class Golden:
    id: str
    lang: str
    query: str
    expect: list[tuple[str, list[str]]]
    slice: str = ""
    answer_tier: str = ""


@dataclass
class Poison:
    id: str
    target: str
    agent: str
    style: str
    texts: list[str]

    @property
    def source(self) -> str:
        return f"{self.agent}:{self.id}"


@dataclass
class Hit:
    file: str | None  # "<slice>/<path>" for indexed files, None for notes
    source: str
    text: str
    tier: str | None = None
    haystack: str = field(default="", repr=False)


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def load_bench(folder: Path) -> tuple[dict, list[Slice], list[Golden], list[Poison]]:
    config = json.loads((folder / "bench.json").read_text(encoding="utf-8"))
    slices = []
    for spec in config["slices"]:
        root = (folder / spec["root"]).resolve()
        files: set[Path] = set()
        for pattern in spec.get("include", ["**/*.md"]):
            files |= {p.resolve() for p in root.glob(pattern) if p.is_file()}
        for pattern in spec.get("exclude", []):
            files -= {p.resolve() for p in root.glob(pattern)}
        canon: set[Path] = set()
        for pattern in spec.get("canon", spec.get("include", ["**/*.md"])):
            canon |= {p.resolve() for p in root.glob(pattern) if p.is_file()}
        slices.append(Slice(spec["name"], root, sorted(files), canon & files))
    golden = []
    for row in load_jsonl(folder / "golden.jsonl"):
        expect = []
        for e in row["expect"]:
            contains = e["contains"] if isinstance(e["contains"], list) else [e["contains"]]
            expect.append((e["file"], [norm(c) for c in contains]))
        if row["lang"] not in LANGS:
            raise SystemExit(f"golden {row['id']}: lang must be one of {LANGS}")
        golden.append(Golden(row["id"], row["lang"], row["query"], expect))
    poison = [Poison(r["id"], r["target"], r["agent"], r.get("style", "plain"), r["texts"])
              for r in load_jsonl(folder / "poison.jsonl")]
    return config, slices, golden, poison


def file_id(slices: list[Slice], path: Path) -> str:
    for s in slices:
        if path.is_relative_to(s.root):
            return f"{s.name}/{path.relative_to(s.root).as_posix()}"
    raise ValueError(path)


# --- scoring --------------------------------------------------------------------------------


def is_hit(golden: Golden, hit: Hit) -> bool:
    if hit.file is None:
        return False
    text = norm(hit.text)
    return any(hit.file == f and any(c in text for c in contains) for f, contains in golden.expect)


def first_hit(golden: Golden, hits: list[Hit]) -> int | None:
    for rank, hit in enumerate(hits[:TOP_K], start=1):
        if is_hit(golden, hit):
            return rank
    return None


def summarize(ranks: list[int | None]) -> dict:
    if not ranks:
        return {"n": 0}
    return {
        "n": len(ranks),
        "recall@1": statistics.fmean(1.0 if r == 1 else 0.0 for r in ranks),
        "recall@5": statistics.fmean(1.0 if r is not None and r <= 5 else 0.0 for r in ranks),
        "mrr@10": statistics.fmean(1.0 / r if r is not None else 0.0 for r in ranks),
    }


def grouped(golden: list[Golden], ranks: dict[str, int | None]) -> dict[str, dict]:
    groups: dict[str, list[int | None]] = {"all": []}
    for g in golden:
        if g.id not in ranks:
            continue
        r = ranks[g.id]
        for key in ("all", f"lang:{g.lang}", f"slice:{g.slice}", f"answer:{g.answer_tier}"):
            groups.setdefault(key, []).append(r)
    return {key: summarize(values) for key, values in groups.items()}


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, max(0, math.ceil(q * len(ordered)) - 1))]


# --- grep baseline ----------------------------------------------------------------------------


def grep_terms(query: str, stem: bool) -> list[str]:
    return keyword.terms(query, stem=stem)


def grep_rank(query: str, passages: list[Hit], stem: bool) -> list[Hit]:
    """Passages ranked by distinct query terms found (case-insensitive substrings), then by occurrences."""
    return [passages[i] for i, _ in keyword.rank(query, [p.haystack for p in passages], stem=stem, limit=TOP_K)]


# --- the bench -------------------------------------------------------------------------------


class Bench:
    def __init__(self, memory: Memory, collection: str, slices: list[Slice]):
        self.memory = memory
        self.collection = collection
        self.qname = memory.qdrant_name(collection)
        self.slices = slices
        self.source_to_file: dict[str, str] = {}
        self.file_tier: dict[str, str] = {}

    def to_hit(self, payload: dict) -> Hit:
        source = payload.get("source", "")
        return Hit(self.source_to_file.get(source), source, payload.get("text", ""), payload.get("tier"))

    async def index(self) -> dict:
        started = time.perf_counter()
        passages = 0
        for s in self.slices:
            for path in s.files:
                result = await self.memory.remember(self.collection, path=str(path))
                fid = file_id(self.slices, path)
                for source in result.sources:
                    self.source_to_file[source] = fid
                self.file_tier[fid] = "canon" if path in s.canon else "inbox"
                passages += result.passages
        return {"seconds": time.perf_counter() - started, "files": len(self.file_tier), "passages": passages}

    async def stored_passages(self) -> list[Hit]:
        hits: list[tuple[Hit, int]] = []
        offset = None
        while True:
            records, offset = await self.memory.qdrant.scroll(
                self.qname, limit=256, offset=offset, with_payload=True, with_vectors=False
            )
            for record in records:
                payload = record.payload or {}
                hit = self.to_hit(payload)
                hit.haystack = f"{payload.get('section') or ''}\n{hit.text}".lower()
                hits.append((hit, payload.get("chunk", 0)))
            if offset is None:
                break
        hits.sort(key=lambda item: (item[0].source, item[1]))
        return [h for h, _ in hits]

    async def vectors(self, query: str, timings: dict[str, list[float]]) -> tuple[list[Hit], list[Hit]]:
        t0 = time.perf_counter()
        embedded = await self.memory.embedder.embed([query], input_type="query")
        dense, sparse = embedded.items[0].dense, embedded.items[0].sparse
        t1 = time.perf_counter()
        response = await self.memory.qdrant.query_points(self.qname, query=dense, using="dense", limit=TOP_K, with_payload=True)
        dense_hits = [self.to_hit(p.payload or {}) for p in response.points]
        timings["dense"].append(time.perf_counter() - t0)
        sparse_hits: list[Hit] = []
        if sparse and sparse.get("indices"):
            t2 = time.perf_counter()
            response = await self.memory.qdrant.query_points(
                self.qname, query=models.SparseVector(**sparse), using="sparse", limit=TOP_K, with_payload=True
            )
            sparse_hits = [self.to_hit(p.payload or {}) for p in response.points]
            timings["sparse"].append(time.perf_counter() - t2 + (t1 - t0))
        return dense_hits, sparse_hits

    async def search(self, query: str, *, rerank: bool, prefer_tier: str = "", tiers: list[str] | None = None,
                     penalty: float | None = None) -> tuple[list[Hit], float]:
        if penalty is not None:
            self.memory.settings.tier_penalty = penalty
        t0 = time.perf_counter()
        result = await self.memory.search(
            self.collection, query, top_k=TOP_K, rerank=rerank, prefer_tier=prefer_tier, tiers=tiers
        )
        elapsed = time.perf_counter() - t0
        hits = [Hit(self.source_to_file.get(p.source), p.source, p.text, p.tier) for p in result.results]
        return hits, elapsed


async def run(args: argparse.Namespace, memory: Memory | None = None) -> dict:
    """Run the bench; `memory` lets tests pass one wired to fake services."""
    folder = args.bench.resolve()
    config, slices, golden, poison = load_bench(folder)
    name = config.get("name", folder.name)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    collection = args.collection or f"bench-{re.sub(r'[^A-Za-z0-9_-]', '-', name)[:30]}-{stamp}"

    canon_files = sorted({p for s in slices for p in s.canon})
    overrides = {
        "allowed_paths": os.pathsep.join(str(s.root) for s in slices),
        "canon_paths": os.pathsep.join(str(p) for p in canon_files),
        "prefer_tier": "",
    }
    for key in ("embedding_url", "reranker_url", "qdrant_url"):
        if getattr(args, key, None):
            overrides[key] = getattr(args, key)
    if memory is None:
        memory = Memory(MCPSettings(**overrides), use_manifest=False)
    else:
        for key, value in overrides.items():
            setattr(memory.settings, key, value)
    memory.use_manifest = False  # tiers come from bench.json only, never from a synced canon
    settings = memory.settings
    bench = Bench(memory, collection, slices)
    log = lambda msg: print(msg, file=sys.stderr, flush=True)  # noqa: E731

    report: dict = {
        "name": name,
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "bench": str(args.bench),
        "collection": collection,
        "settings": {
            "embedding_url": settings.embedding_url, "reranker_url": settings.reranker_url,
            "qdrant_url": settings.qdrant_url, "chunk_tokens": settings.chunk_tokens,
            "chunk_overlap_tokens": settings.chunk_overlap_tokens, "prefetch_limit": settings.prefetch_limit,
            "rerank_candidates": settings.rerank_candidates, "top_k": TOP_K,
        },
    }
    try:
        log(f"indexing {sum(len(s.files) for s in slices)} files into scratch collection {collection!r} ...")
        report["index"] = await bench.index()
        passages = await bench.stored_passages()
        report["index"]["passages_by_tier"] = {
            tier: sum(1 for p in passages if p.tier == tier) for tier in sorted({p.tier or "none" for p in passages})
        }

        # Every golden question must have at least one stored passage that answers it.
        valid: list[Golden] = []
        invalid: list[str] = []
        for g in golden:
            g.slice = g.expect[0][0].split("/", 1)[0]
            tiers = {bench.file_tier.get(f) for f, _ in g.expect}
            g.answer_tier = "canon" if "canon" in tiers else "inbox"
            if any(is_hit(g, p) for p in passages):
                valid.append(g)
            else:
                invalid.append(g.id)
                log(f"  golden {g.id}: no stored passage contains the expected text; left out")
        report["golden"] = {
            "questions": len(valid), "left_out": invalid,
            "by_lang": {lang: sum(1 for g in valid if g.lang == lang) for lang in LANGS},
            "by_answer_tier": {t: sum(1 for g in valid if g.answer_tier == t) for t in ("canon", "inbox")},
        }

        # --- clean index: every method, every question --------------------------------
        log(f"running {len(valid)} questions x {len(CLEAN_METHODS)} methods ...")
        timings: dict[str, list[float]] = {m: [] for m in CLEAN_METHODS}
        per_question: dict[str, dict] = {}
        ranks: dict[str, dict[str, int | None]] = {m: {} for m in CLEAN_METHODS}
        top5_chars: list[int] = []
        top5_tokens: list[int] = []
        for g in valid:
            runs: dict[str, list[Hit]] = {}
            for method, stem in (("grep", False), ("grep_stem", True)):
                t0 = time.perf_counter()
                runs[method] = grep_rank(g.query, passages, stem)
                timings[method].append(time.perf_counter() - t0)
            runs["dense"], runs["sparse"] = await bench.vectors(g.query, timings)
            runs["hybrid"], elapsed = await bench.search(g.query, rerank=False)
            timings["hybrid"].append(elapsed)
            runs["hybrid+rerank"], elapsed = await bench.search(g.query, rerank=True)
            timings["hybrid+rerank"].append(elapsed)
            # What tier weighting costs when nobody attacks (MCP_TIER_PENALTY default 0.5).
            runs["hybrid+rerank, prefer canon"], elapsed = await bench.search(
                g.query, rerank=True, prefer_tier="canon", penalty=0.5
            )
            timings["hybrid+rerank, prefer canon"].append(elapsed)
            top5_chars.append(sum(len(h.text) for h in runs["hybrid+rerank"][:5]))
            top5_tokens.append(sum(estimate_tokens(h.text) for h in runs["hybrid+rerank"][:5]))
            per_question[g.id] = {"lang": g.lang, "slice": g.slice, "answer_tier": g.answer_tier, "query": g.query,
                                  "clean": {}}
            for method, hits in runs.items():
                rank = first_hit(g, hits)
                ranks[method][g.id] = rank
                per_question[g.id]["clean"][method] = {
                    "rank": rank, "top1": (hits[0].file or hits[0].source) if hits else None,
                }
        report["clean"] = {m: grouped(valid, ranks[m]) for m in CLEAN_METHODS}
        report["latency_ms_p50"] = {m: round(percentile(t, 0.5) * 1000, 1) for m, t in timings.items() if t}

        # --- context cost ---------------------------------------------------------------------
        read_first = config.get("read_first", [])
        read_chars = read_tokens = 0
        for fid in read_first:
            slice_name, rel = fid.split("/", 1)
            root = next(s.root for s in slices if s.name == slice_name)
            text = (root / rel).read_text(encoding="utf-8")
            read_chars += len(text)
            read_tokens += estimate_tokens(text)
        covered = [g for g in valid if any(f in read_first for f, _ in g.expect)]
        mean_top5 = statistics.fmean(top5_chars) if top5_chars else 0.0
        corpus_texts = [p.read_text(encoding="utf-8") for s in slices for p in s.files]
        report["context"] = {
            "read_first_files": len(read_first),
            "read_first_chars": read_chars,
            "read_first_tokens_est": read_tokens,
            "read_first_covers": len(covered) / len(valid) if valid else 0.0,
            "corpus_files": len(corpus_texts),
            "corpus_chars": sum(len(t) for t in corpus_texts),
            "corpus_tokens_est": sum(estimate_tokens(t) for t in corpus_texts),
            "top5_chars_mean": round(mean_top5),
            "top5_tokens_est": round(statistics.fmean(top5_tokens)) if top5_tokens else 0,
            "top5_recall@5": report["clean"]["hybrid+rerank"]["all"].get("recall@5"),
            "ratio": round(read_chars / mean_top5, 1) if mean_top5 else None,
        }

        # --- poisoning ---------------------------------------------------------------------------
        if poison and not args.no_poison:
            log(f"planting {sum(len(p.texts) for p in poison)} false notes from bot sources ...")
            for item in poison:
                await memory.remember(collection, texts=item.texts, source=item.source, tags=["bench-poison"])
            poison_by_source = {p.source: p for p in poison}
            targets = {g.id: g for g in valid if any(p.target == g.id for p in poison)}
            configs: list[tuple[str, dict]] = [("hybrid+rerank (no tiers)", {"rerank": True})]
            for penalty in args.penalties:
                configs.append((f"hybrid+rerank, prefer canon x{penalty:g}",
                                {"rerank": True, "prefer_tier": "canon", "penalty": penalty}))
            configs.append(("hybrid+rerank, canon only", {"rerank": True, "tiers": ["canon"]}))
            configs.append(("hybrid (no rerank, no tiers)", {"rerank": False}))
            configs.append(("hybrid, prefer canon x0.5", {"rerank": False, "prefer_tier": "canon", "penalty": 0.5}))
            log(f"poisoning runs: {len(configs)} rankings x {len(valid)} questions ...")
            report["poison"] = {
                "notes": sum(len(p.texts) for p in poison), "targets": len(targets),
                "targets_by_answer_tier": {t: sum(1 for g in targets.values() if g.answer_tier == t) for t in ("canon", "inbox")},
                "notes_by_style": {s: sum(1 for p in poison if p.style == s) for s in sorted({p.style for p in poison})},
                "styles": sorted({p.style for p in poison}), "configs": {},
            }
            for label, kwargs in configs:
                cfg_ranks: dict[str, int | None] = {}
                first, top5, truth1, latency = [], [], [], []
                first_by_answer: dict[str, list[float]] = {}
                styles: dict[str, list[float]] = {}
                for g in valid:
                    hits, elapsed = await bench.search(g.query, **kwargs)
                    latency.append(elapsed)
                    cfg_ranks[g.id] = first_hit(g, hits)
                    entry = per_question[g.id].setdefault("poisoned", {})
                    entry[label] = {"rank": cfg_ranks[g.id], "top1": (hits[0].file or hits[0].source) if hits else None}
                    if g.id not in targets:
                        continue
                    top_poison = poison_by_source.get(hits[0].source) if hits else None
                    first.append(1.0 if top_poison else 0.0)
                    first_by_answer.setdefault(g.answer_tier, []).append(1.0 if top_poison else 0.0)
                    top5.append(1.0 if any(h.source in poison_by_source for h in hits[:5]) else 0.0)
                    truth1.append(1.0 if cfg_ranks[g.id] == 1 else 0.0)
                    # Per false note: does it rank above the true passage (or appear while the truth does not)?
                    truth_rank = cfg_ranks[g.id] or math.inf
                    for item in (p for p in poison if p.target == g.id):
                        rank = next((i for i, h in enumerate(hits[:TOP_K], 1) if h.source == item.source), math.inf)
                        styles.setdefault(item.style, []).append(1.0 if rank < truth_rank else 0.0)
                report["poison"]["configs"][label] = {
                    "false_first": statistics.fmean(first) if first else None,
                    "false_in_top5": statistics.fmean(top5) if top5 else None,
                    "true_first": statistics.fmean(truth1) if truth1 else None,
                    "false_first_by_answer_tier": {t: statistics.fmean(v) for t, v in sorted(first_by_answer.items())},
                    "false_above_truth_by_style": {s: statistics.fmean(v) for s, v in sorted(styles.items())},
                    "all_questions": grouped(valid, cfg_ranks),
                    "latency_ms_p50": round(percentile(latency, 0.5) * 1000, 1),
                }

        report["questions"] = per_question
    finally:
        if args.keep:
            log(f"kept scratch collection {collection!r}; delete it with forget(collection, everything=true)")
        else:
            try:
                dropped = await memory.forget(collection, everything=True)
                log(f"dropped scratch collection {collection!r} ({dropped.deleted_passages} passages)")
            except Exception as exc:  # the collection may not exist if indexing failed early
                log(f"could not drop {collection!r}: {exc}")
        await memory.aclose()

    out = (args.out or folder / "results").resolve()
    out.mkdir(parents=True, exist_ok=True)
    base = out / f"{stamp}_{name}"
    base.with_suffix(".json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    markdown = render(report, slices)
    base.with_suffix(".md").write_text(markdown, encoding="utf-8")
    print(markdown)
    print(f"Saved {base}.md (numbers only) and .json (per question)", file=sys.stderr)
    return report


# --- report -----------------------------------------------------------------------------------


def _pct(value: float | None) -> str:
    return "—" if value is None else f"{value * 100:.0f}%"


def _num(value: float | None) -> str:
    return "—" if value is None else f"{value:.2f}"


def render(report: dict, slices: list[Slice]) -> str:
    g = report["golden"]
    idx = report["index"]
    lines = [
        f"# Memory bench: {report['name']}",
        "",
        f"{report['created_at']} · {idx['files']} files in {len(slices)} slices "
        f"({', '.join(s.name for s in slices)}) → {idx['passages']} passages "
        f"({', '.join(f'{k} {v}' for k, v in idx['passages_by_tier'].items())}), indexed in {idx['seconds']:.0f} s · "
        f"{g['questions']} questions (ru {g['by_lang']['ru']}, en {g['by_lang']['en']}, mixed {g['by_lang']['mixed']}; "
        f"answer in canon {g['by_answer_tier']['canon']}, in inbox {g['by_answer_tier']['inbox']})"
        + (f" · left out: {len(g['left_out'])}" if g["left_out"] else ""),
        "",
        "## Retrieval on the clean index",
        "",
        "recall@k = share of questions with a passage that holds the answer in the top k; MRR over the top 10.",
        "",
        "| method | R@1 | R@5 | MRR | ru R@1 | ru R@5 | en R@1 | en R@5 | mixed R@1 | mixed R@5 | p50 ms |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for method, groups in report["clean"].items():
        row = [method, _pct(groups["all"]["recall@1"]), _pct(groups["all"]["recall@5"]), _num(groups["all"]["mrr@10"])]
        for lang in LANGS:
            sub = groups.get(f"lang:{lang}", {})
            row += [_pct(sub.get("recall@1")), _pct(sub.get("recall@5"))]
        row.append(f"{report['latency_ms_p50'].get(method, 0):.0f}")
        lines.append("| " + " | ".join(row) + " |")

    lines += ["", "By slice and by where the answer lives (hybrid + rerank, recall@1 / recall@5):", ""]
    best = report["clean"]["hybrid+rerank"]
    grep = report["clean"]["grep"]
    lines += ["| group | n | hybrid+rerank | grep |", "|---|---|---|---|"]
    for key in sorted(k for k in best if k.startswith(("slice:", "answer:"))):
        lines.append(f"| {key} | {best[key]['n']} | {_pct(best[key]['recall@1'])} / {_pct(best[key]['recall@5'])} | "
                     f"{_pct(grep.get(key, {}).get('recall@1'))} / {_pct(grep.get(key, {}).get('recall@5'))} |")

    c = report["context"]
    lines += [
        "",
        "## Context cost",
        "",
        "| what the agent reads | characters | ≈ tokens | holds the answer |",
        "|---|---|---|---|",
        f"| the {c['read_first_files']} read-first files, in full | {c['read_first_chars']:,} | {c['read_first_tokens_est']:,} | "
        f"{_pct(c['read_first_covers'])} of questions |",
        f"| every file in the corpus ({c['corpus_files']}) | {c['corpus_chars']:,} | {c['corpus_tokens_est']:,} | all questions |",
        f"| top-5 passages from search (mean) | {c['top5_chars_mean']:,} | {c['top5_tokens_est']:,} | "
        f"{_pct(c['top5_recall@5'])} of questions |",
    ]
    if c["ratio"]:
        ratio = c["ratio"]
        lines += ["", f"The top-5 passages are {ratio:g}× smaller than the read-first files." if ratio >= 1
                  else f"The top-5 passages are {1 / ratio:.1f}× larger than the read-first files (a small canon)."]

    if "poison" in report:
        p = report["poison"]
        lines += [
            "",
            "## Poisoning",
            "",
            f"{p['notes']} false notes written as unreviewed `inbox` notes from bot sources, aimed at {p['targets']} questions "
            f"({p['targets_by_answer_tier']['canon']} whose true answer is in canon, {p['targets_by_answer_tier']['inbox']} "
            f"whose true answer is itself an inbox note). Styles: "
            + ", ".join(f"{s} {n}" for s, n in p["notes_by_style"].items())
            + "; `authority` notes claim to be approved or to supersede canon.",
            "",
            "| ranking | false fact 1st | — truth in canon | — truth in inbox | false in top 5 | true fact 1st | all questions R@1 | R@5 | p50 ms |",
            "|---|---|---|---|---|---|---|---|---|",
        ]
        for label, r in p["configs"].items():
            allq = r["all_questions"]
            by = r["false_first_by_answer_tier"]
            lines.append(
                f"| {label} | {_pct(r['false_first'])} | {_pct(by.get('canon'))} | {_pct(by.get('inbox'))} | "
                f"{_pct(r['false_in_top5'])} | {_pct(r['true_first'])} | "
                f"{_pct(allq['all']['recall@1'])} | {_pct(allq['all']['recall@5'])} | {r['latency_ms_p50']:.0f} |"
            )
        lines += ["", "Share of false notes ranked above the true passage, by style:", ""]
        lines += ["| ranking | " + " | ".join(p["styles"]) + " |", "|---|" + "---|" * len(p["styles"])]
        for label, r in p["configs"].items():
            lines.append(f"| {label} | " + " | ".join(_pct(r["false_above_truth_by_style"].get(s)) for s in p["styles"]) + " |")

    s = report["settings"]
    lines += [
        "",
        f"Settings: chunks {s['chunk_tokens']} tokens / overlap {s['chunk_overlap_tokens']}, prefetch {s['prefetch_limit']}, "
        f"rerank candidates {s['rerank_candidates']}, top {s['top_k']}.",
    ]
    return "\n".join(line for line in lines if line is not None) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--bench", required=True, type=Path, help="folder with bench.json, golden.jsonl, poison.jsonl")
    parser.add_argument("--out", type=Path, help="results folder (default: <bench>/results)")
    parser.add_argument("--collection", help="scratch collection name (default: bench-<name>-<time>)")
    parser.add_argument("--keep", action="store_true", help="keep the scratch collection")
    parser.add_argument("--no-poison", action="store_true", help="skip the poisoning runs")
    parser.add_argument("--penalties", type=lambda v: [float(x) for x in v.split(",")], default=[0.75, 0.5, 0.25],
                        help="tier penalties to test with prefer_tier=canon (default 0.75,0.5,0.25)")
    parser.add_argument("--embedding-url", dest="embedding_url")
    parser.add_argument("--reranker-url", dest="reranker_url")
    parser.add_argument("--qdrant-url", dest="qdrant_url")
    args = parser.parse_args()
    asyncio.run(run(args))
    return 0


if __name__ == "__main__":
    sys.exit(main())
