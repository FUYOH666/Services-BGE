"""Split markdown and plain text into overlapping passages sized for bge-m3.

Token counts are estimated without loading the model's tokenizer. The estimate
is deliberately conservative for XLM-R (bge-m3) on English and Russian, and
the server sends `max_length` well above the passage size, so a passage is
never truncated by the embedding service.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

_WORD = re.compile(r"\w+")
_PUNCT = re.compile(r"[^\w\s]")
_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
_FENCE = re.compile(r"^\s*(```|~~~)")
_SENTENCE_BREAK = re.compile(r"(?<=[.!?…])\s+|\n")


def estimate_tokens(text: str) -> int:
    """About one token per four characters of a word, plus one per punctuation mark."""
    return sum(1 + len(w) // 4 for w in _WORD.findall(text)) + len(_PUNCT.findall(text))


@dataclass(frozen=True)
class Chunk:
    index: int
    text: str
    # Heading path in effect at the start of the passage, e.g. "Setup > macOS".
    section: str | None = None

    def contextual_text(self) -> str:
        """Text sent to the models: the heading path is prepended unless the passage already starts with it."""
        if not self.section or (" > " not in self.section and self.text.lstrip().startswith("#")):
            return self.text
        return f"{self.section}\n\n{self.text}"


@dataclass
class _Unit:
    text: str
    tokens: int
    section: str | None
    heading_level: int = 0
    # Second or later piece of one long paragraph: joined with a space, not a blank line.
    continues: bool = False
    # Copied from the end of the previous passage.
    overlap: bool = False


def _blocks(text: str, markdown: bool) -> list[_Unit]:
    """Paragraphs, code fences and headings, each with the heading path above it."""
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    blocks: list[_Unit] = []
    buf: list[str] = []
    stack: list[tuple[int, str]] = []
    fence: str | None = None

    def section() -> str | None:
        return " > ".join(title for _, title in stack) or None

    def flush() -> None:
        body = "\n".join(buf).strip("\n")
        if body.strip():
            blocks.append(_Unit(body, estimate_tokens(body), section()))
        buf.clear()

    for line in lines:
        if markdown:
            fence_match = _FENCE.match(line)
            if fence is not None:
                buf.append(line)
                if fence_match and fence_match.group(1) == fence:
                    fence = None
                    flush()
                continue
            if fence_match:
                flush()
                fence = fence_match.group(1)
                buf.append(line)
                continue
            heading = _HEADING.match(line)
            if heading:
                flush()
                level = len(heading.group(1))
                while stack and stack[-1][0] >= level:
                    stack.pop()
                stack.append((level, heading.group(2).strip()))
                title = line.strip()
                blocks.append(_Unit(title, estimate_tokens(title), section(), heading_level=level))
                continue
        if not line.strip():
            flush()
            continue
        buf.append(line)
    flush()
    return blocks


def _pieces(text: str, pattern: re.Pattern[str]) -> list[str]:
    """Split after each match, keeping the separators so that "".join() restores the text."""
    out, start = [], 0
    for match in pattern.finditer(text):
        piece = text[start:match.end()]
        if piece.strip():
            out.append(piece)
        start = match.end()
    if text[start:].strip():
        out.append(text[start:])
    return out


def _split_long(text: str, limit: int) -> list[str]:
    """Split one oversized block at sentence ends, then words, then characters."""
    parts: list[str] = []
    buf: list[str] = []
    tokens = 0

    def flush() -> None:
        nonlocal tokens
        joined = "".join(buf).strip()
        if joined:
            parts.append(joined)
        buf.clear()
        tokens = 0

    for sentence in _pieces(text, _SENTENCE_BREAK):
        t = estimate_tokens(sentence)
        if t > limit:
            flush()
            for word in _pieces(sentence, re.compile(r"\s+")):
                wt = estimate_tokens(word)
                if wt > limit:
                    flush()
                    step = max((limit - 1) * 4, 1)
                    parts.extend(word[i:i + step].strip() for i in range(0, len(word), step) if word[i:i + step].strip())
                    continue
                if tokens + wt > limit:
                    flush()
                buf.append(word)
                tokens += wt
            flush()
            continue
        if tokens + t > limit:
            flush()
        buf.append(sentence)
        tokens += t
    flush()
    return parts


def _tail(text: str, budget: int) -> str:
    """The last sentences (or words) of a passage that fit in `budget` tokens."""
    for pattern in (_SENTENCE_BREAK, re.compile(r"\s+")):
        picked: list[str] = []
        total = 0
        for piece in reversed(_pieces(text, pattern)):
            t = estimate_tokens(piece)
            if total + t > budget:
                break
            picked.insert(0, piece)
            total += t
        if picked:
            return "".join(picked).strip()
    return ""


def chunk_text(
    text: str,
    *,
    max_tokens: int = 500,
    overlap_tokens: int = 80,
    markdown: bool = True,
) -> list[Chunk]:
    """Split a document into passages of at most `max_tokens` estimated tokens.

    Markdown headings start a new passage (level 1–2 always, deeper ones once the
    current passage is a third full) and give each passage its heading path.
    Consecutive passages inside one section share up to `overlap_tokens`.
    """
    if max_tokens < 32:
        raise ValueError("max_tokens must be at least 32")
    overlap_tokens = max(0, min(overlap_tokens, max_tokens // 2))
    unit_limit = max_tokens - overlap_tokens

    units: list[_Unit] = []
    for block in _blocks(text, markdown):
        if block.tokens <= unit_limit:
            units.append(block)
            continue
        for i, part in enumerate(_split_long(block.text, unit_limit)):
            units.append(_Unit(part, estimate_tokens(part), block.section, continues=i > 0))

    chunks: list[Chunk] = []
    current: list[_Unit] = []

    def has_new_body() -> bool:
        return any(not u.heading_level and not u.overlap for u in current)

    def size() -> int:
        return sum(u.tokens for u in current)

    def drop_overlap() -> None:
        current[:] = [u for u in current if not u.overlap]

    def emit(with_overlap: bool) -> None:
        nonlocal current
        # Headings at the end belong to the next passage.
        carried: list[_Unit] = []
        while current and current[-1].heading_level:
            carried.insert(0, current.pop())
        # The heading path of the first new paragraph (headings above it included).
        first = next(u for u in current if not u.overlap and not u.heading_level)
        texts: list[str] = []
        for u in current:
            if texts and u.continues:
                texts[-1] += " " + u.text
            else:
                texts.append(u.text)
        chunks.append(Chunk(len(chunks), "\n\n".join(texts).strip(), first.section))
        last_body = [u for u in current if not u.heading_level][-1]
        current = carried
        if with_overlap and not carried and overlap_tokens:
            tail = _tail(last_body.text, overlap_tokens)
            if tail:
                current = [_Unit(tail, estimate_tokens(tail), last_body.section, overlap=True)]

    for unit in units:
        if unit.heading_level:
            if has_new_body() and (unit.heading_level <= 2 or size() >= max_tokens // 3):
                emit(with_overlap=False)
            if not has_new_body():
                drop_overlap()  # a new section does not start with the previous one's tail
            current.append(unit)
            continue
        if has_new_body() and size() + unit.tokens > max_tokens:
            emit(with_overlap=True)
        if size() + unit.tokens > max_tokens:
            drop_overlap()
        current.append(unit)

    if has_new_body():
        emit(with_overlap=False)
    return chunks
