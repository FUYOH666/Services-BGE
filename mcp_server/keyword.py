"""Keyword ranking without any model: literal, case-insensitive term matching, like `rg -i`.

Used by search when the embedding service or Qdrant is down (degraded mode over the
local canon files) and by eval/memory_bench.py as the grep baseline.
"""
from __future__ import annotations

import re

# Words nobody would grep for. Words shorter than 3 letters are dropped as well.
STOPWORDS = frozenset("""
the and are was were does did what which who whom how why where when can could should would
our ours you your for with from that this these those there their them they its into about
any much many one has have had now not but all out use used get got still just than then
also more most some such very own each other only same being been will shall may might must
что как какой какая какое какие каких каким кто где когда почему зачем ли можно нужно надо
это этот эта эти том той тот там тут для без при над под через после перед между или если
так также уже ещё еще есть был была были было будет быть нас наш наша наше наши нам нами
мы вы они она оно его её ее их чем чего сейчас сколько который которая которые кого кому
""".split())
_TERM = re.compile(r"\w+")
STEM = 5


def terms(query: str, *, stem: bool = False) -> list[str]:
    """Search terms of a query: lower case, no stop words; with `stem`, cut to 5 letters (пилотов → пилот)."""
    out: list[str] = []
    for term in _TERM.findall(query.lower()):
        if len(term) < 3 or term in STOPWORDS:
            continue
        if stem and len(term) > STEM:
            term = term[:STEM]
        if term not in out:
            out.append(term)
    return out


def rank(query: str, haystacks: list[str], *, stem: bool = False, limit: int | None = None) -> list[tuple[int, float]]:
    """(index, share of query terms found) for texts containing at least one term.

    Ordered by distinct terms found, then by total occurrences, then by position.
    `haystacks` must already be lower case.
    """
    words = terms(query, stem=stem)
    if not words:
        return []
    scored = []
    for i, text in enumerate(haystacks):
        counts = [text.count(w) for w in words]
        found = sum(1 for c in counts if c)
        if found:
            scored.append((-found, -sum(counts), i))
    scored.sort()
    ranked = [(i, -found / len(words)) for found, _, i in scored]
    return ranked[:limit] if limit else ranked
