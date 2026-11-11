"""Long-term memory on Qdrant: remember, search, forget.

Each memory collection is a Qdrant collection named `<prefix><name>` with a
named dense vector `dense` (cosine) and a sparse vector `sparse` (bge-m3
lexical weights). Point ids are derived from the source and passage number,
so saving the same thing twice updates it instead of duplicating it.
"""
from __future__ import annotations

import contextlib
import hashlib
import logging
import os
import re
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
from pydantic import BaseModel
from qdrant_client import AsyncQdrantClient, models
from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse

from .chunking import Chunk, chunk_text
from .config import MCPSettings
from .services import EmbeddingClient, RerankClient, ServiceError

logger = logging.getLogger(__name__)

DENSE = "dense"
SPARSE = "sparse"
SCHEMA_VERSION = 1
CREATED_BY = "services-bge-mcp"
_ID_NAMESPACE = uuid.uuid5(uuid.NAMESPACE_URL, "https://github.com/FUYOH666/Services-BGE/mcp_server")
_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,62}$")
_MARKDOWN = {".md", ".markdown", ".mdx"}
_SKIP_DIRS = {"node_modules", "__pycache__", "venv", "site-packages"}
_RETRIEVE_BATCH = 256

Progress = Callable[[float, float | None, str | None], Awaitable[None]]


# --- Tool results ---------------------------------------------------------------


class Passage(BaseModel):
    text: str
    source: str
    score: float
    chunk: int | None = None
    section: str | None = None
    tags: list[str] | None = None
    created_at: str | None = None


class SearchResult(BaseModel):
    collection: str
    query: str
    reranked: bool
    results: list[Passage]
    note: str | None = None


class RememberResult(BaseModel):
    collection: str
    sources: list[str]
    passages: int
    embedded: int
    unchanged: int
    removed: int
    skipped: list[str] = []
    note: str | None = None


class CollectionInfo(BaseModel):
    name: str
    passages: int
    embedding_model: str | None = None


class CollectionsResult(BaseModel):
    collections: list[CollectionInfo]


class SourceInfo(BaseModel):
    source: str
    passages: int


class SourcesResult(BaseModel):
    collection: str
    sources: list[SourceInfo]
    truncated: bool


class ForgetResult(BaseModel):
    collection: str
    deleted_passages: int
    dropped_collection: bool


class RankedDocument(BaseModel):
    index: int
    score: float
    document: str


class RerankResult(BaseModel):
    model: str
    results: list[RankedDocument]


class Vector(BaseModel):
    index: int
    dense: list[float]
    sparse: dict[str, list[float | int]] | None = None


class VectorsResult(BaseModel):
    model: str
    dimensions: int
    vectors: list[Vector]


# --- Internals --------------------------------------------------------------------


@dataclass
class _Document:
    source: str
    text: str
    markdown: bool
    # Saved texts are keyed by content too, so different notes can share a source.
    content_key: str = ""


@dataclass
class _Planned:
    id: str
    source: str
    chunk: Chunk
    hash: str


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _display_path(path: Path) -> str:
    home = Path.home()
    try:
        return "~/" + str(path.relative_to(home))
    except ValueError:
        return str(path)


def _clean_tags(tags: list[str] | None) -> list[str] | None:
    if not tags:
        return None
    seen: list[str] = []
    for tag in tags:
        tag = tag.strip()
        if tag and tag not in seen:
            seen.append(tag)
    return seen or None


def _model_text(payload: dict[str, Any]) -> str:
    return Chunk(0, payload.get("text", ""), payload.get("section")).contextual_text()


class Memory:
    """The memory operations behind the MCP tools. Collaborators are injectable for tests."""

    def __init__(
        self,
        settings: MCPSettings,
        *,
        qdrant: AsyncQdrantClient | None = None,
        embedder: EmbeddingClient | None = None,
        reranker: RerankClient | None = None,
    ) -> None:
        self.settings = settings
        self._qdrant = qdrant
        self.embedder = embedder or EmbeddingClient(
            settings.embedding_url,
            api_key=settings.embedding_api_key,
            timeout=settings.request_timeout,
            max_length=settings.max_length,
            batch_size=settings.embed_batch_size,
        )
        self.reranker = reranker or RerankClient(
            settings.reranker_url,
            api_key=settings.reranker_api_key,
            timeout=settings.request_timeout,
            max_length=settings.max_length,
        )

    @property
    def qdrant(self) -> AsyncQdrantClient:
        if self._qdrant is None:
            self._qdrant = AsyncQdrantClient(
                url=self.settings.qdrant_url,
                api_key=self.settings.qdrant_api_key,
                timeout=int(self.settings.request_timeout),
                check_compatibility=False,
            )
        return self._qdrant

    async def aclose(self) -> None:
        await self.embedder.aclose()
        await self.reranker.aclose()
        if self._qdrant is not None:
            await self._qdrant.close()

    # --- helpers -------------------------------------------------------------------

    def qdrant_name(self, collection: str | None) -> str:
        name = collection or self.settings.default_collection
        if not _NAME.match(name):
            raise ServiceError(
                f"Invalid collection name {name!r}: use letters, digits, '-' and '_' "
                "(up to 63 characters, starting with a letter or digit)."
            )
        return self.settings.collection_prefix + name

    def public_name(self, qdrant_name: str) -> str:
        return qdrant_name[len(self.settings.collection_prefix):]

    @contextlib.asynccontextmanager
    async def _qdrant_errors(self):
        try:
            yield
        except ServiceError:
            raise
        except UnexpectedResponse as exc:
            body = exc.content.decode("utf-8", errors="replace")[:300] if exc.content else exc.reason_phrase
            raise ServiceError(f"Qdrant returned {exc.status_code}: {body}") from exc
        except (ResponseHandlingException, httpx.TransportError, ConnectionError) as exc:
            raise ServiceError(
                f"Qdrant at {self.settings.qdrant_url} is not reachable. Start it, or set MCP_QDRANT_URL."
            ) from exc

    async def _collection(self, qname: str) -> models.CollectionInfo | None:
        async with self._qdrant_errors():
            if not await self.qdrant.collection_exists(qname):
                return None
            return await self.qdrant.get_collection(qname)

    def _check_compatible(self, qname: str, info: models.CollectionInfo, model: str, dim: int) -> None:
        name = self.public_name(qname)
        vectors = info.config.params.vectors
        if not isinstance(vectors, dict) or DENSE not in vectors:
            raise ServiceError(f"Collection {name!r} exists but is not a Services-BGE memory collection (no 'dense' vector).")
        if vectors[DENSE].size != dim:
            raise ServiceError(
                f"Collection {name!r} stores {vectors[DENSE].size}-d vectors, but the embedding service "
                f"returns {dim}-d ones. Point MCP_EMBEDDING_URL at the model the collection was built with."
            )
        stored = (info.config.metadata or {}).get("embedding_model")
        if stored and model and stored != model:
            raise ServiceError(
                f"Collection {name!r} was built with {stored}, but the embedding service now runs {model}. "
                "Vectors from different models cannot be compared: switch the service back, or forget the "
                "collection and remember the content again."
            )

    async def _create_collection(self, qname: str, model: str, dim: int) -> None:
        async with self._qdrant_errors():
            try:
                await self.qdrant.create_collection(
                    qname,
                    vectors_config={DENSE: models.VectorParams(size=dim, distance=models.Distance.COSINE)},
                    sparse_vectors_config={SPARSE: models.SparseVectorParams()},
                    metadata={
                        "created_by": CREATED_BY,
                        "schema_version": SCHEMA_VERSION,
                        "embedding_model": model,
                        "dense_size": dim,
                    },
                )
            except UnexpectedResponse as exc:
                if exc.status_code != 409:  # created concurrently by another agent
                    raise
            for field in ("source", "tags"):
                await self.qdrant.create_payload_index(qname, field, models.PayloadSchemaType.KEYWORD)

    @staticmethod
    def _filter(source: str | None, tags: list[str] | None) -> models.Filter | None:
        must: list = []
        if source:
            must.append(models.FieldCondition(key="source", match=models.MatchValue(value=source)))
        if tags:
            must.append(models.FieldCondition(key="tags", match=models.MatchAny(any=tags)))
        return models.Filter(must=must) if must else None

    # --- remember --------------------------------------------------------------------

    def _collect_files(self, raw: str) -> tuple[list[Path], list[str]]:
        path = Path(raw).expanduser()
        if not path.is_absolute():
            raise ServiceError(f"Use an absolute path or one starting with ~/ (got {raw!r}).")
        path = path.resolve()
        roots = self.settings.allowed_roots()
        if not any(path == root or path.is_relative_to(root) for root in roots):
            allowed = ", ".join(_display_path(r) for r in roots) or "nothing"
            raise ServiceError(f"{_display_path(path)} is outside the folders this server may read ({allowed}; MCP_ALLOWED_PATHS).")
        if not path.exists():
            raise ServiceError(f"{_display_path(path)} does not exist.")
        extensions = self.settings.extensions()
        if path.is_file():
            if path.suffix.lower() not in extensions:
                raise ServiceError(f"Only {', '.join(sorted(extensions))} files can be remembered (MCP_FILE_EXTENSIONS).")
            return [path], []

        files: list[Path] = []
        skipped: list[str] = []
        for dirpath, dirnames, filenames in os.walk(path):
            dirnames[:] = sorted(d for d in dirnames if not d.startswith(".") and d not in _SKIP_DIRS)
            for filename in sorted(filenames):
                file = Path(dirpath) / filename
                if filename.startswith(".") or file.suffix.lower() not in extensions:
                    continue
                real = file.resolve()
                if not any(real.is_relative_to(root) for root in roots):
                    skipped.append(f"{_display_path(file)}: links outside the allowed folders")
                    continue
                files.append(real)
                if len(files) > self.settings.max_files:
                    raise ServiceError(
                        f"{_display_path(path)} has more than {self.settings.max_files} matching files "
                        "(MCP_MAX_FILES); remember a smaller folder."
                    )
        if not files:
            raise ServiceError(f"No {', '.join(sorted(extensions))} files in {_display_path(path)}.")
        return files, skipped

    def _read(self, file: Path) -> str | None:
        if file.stat().st_size > self.settings.max_file_bytes:
            return None
        text = file.read_text(encoding="utf-8", errors="replace")
        return None if "\x00" in text else text

    def _plan(self, documents: list[_Document], tags: list[str] | None) -> list[_Planned]:
        planned: list[_Planned] = []
        tag_key = ",".join(sorted(tags or []))
        for doc in documents:
            chunks = chunk_text(
                doc.text,
                max_tokens=self.settings.chunk_tokens,
                overlap_tokens=self.settings.chunk_overlap_tokens,
                markdown=doc.markdown,
            )
            for chunk in chunks:
                point_id = str(uuid.uuid5(_ID_NAMESPACE, f"{doc.source}\x00{doc.content_key}\x00{chunk.index}"))
                content_hash = _sha(f"{chunk.contextual_text()}\x00{tag_key}")[:16]
                planned.append(_Planned(point_id, doc.source, chunk, content_hash))
            if len(planned) > self.settings.max_chunks:
                raise ServiceError(
                    f"That is more than {self.settings.max_chunks} passages in one call (MCP_MAX_CHUNKS); "
                    "remember fewer files at a time."
                )
        return planned

    async def _existing_hashes(self, qname: str, ids: list[str]) -> dict[str, str]:
        hashes: dict[str, str] = {}
        async with self._qdrant_errors():
            for start in range(0, len(ids), _RETRIEVE_BATCH):
                records = await self.qdrant.retrieve(
                    qname, ids=ids[start:start + _RETRIEVE_BATCH], with_payload=["hash"], with_vectors=False
                )
                for record in records:
                    hashes[str(record.id)] = (record.payload or {}).get("hash", "")
        return hashes

    async def remember(
        self,
        collection: str | None,
        *,
        texts: list[str] | None = None,
        path: str | None = None,
        source: str | None = None,
        tags: list[str] | None = None,
        replace: bool = False,
        progress: Progress | None = None,
    ) -> RememberResult:
        qname = self.qdrant_name(collection)
        tags = _clean_tags(tags)
        source = source.strip() if source and source.strip() else None
        if (texts is None) == (path is None):
            raise ServiceError("Pass either `texts` (a list of strings) or `path` (a file or folder), not both or neither.")

        documents: list[_Document] = []
        skipped: list[str] = []
        replace_sources: set[str] = set()
        if path is not None:
            files, skipped = self._collect_files(path)
            if source and len(files) != 1:
                raise ServiceError("`source` can only be set when `path` is a single file.")
            for file in files:
                text = self._read(file)
                if text is None:
                    skipped.append(f"{_display_path(file)}: larger than MCP_MAX_FILE_BYTES or not text")
                    continue
                doc_source = source or _display_path(file)
                documents.append(_Document(doc_source, text, markdown=file.suffix.lower() in _MARKDOWN))
                # A file is one document: reading it again replaces its old passages.
                replace_sources.add(doc_source)
        else:
            items = [t for t in (texts or []) if t and t.strip()]
            if not items:
                raise ServiceError("`texts` is empty.")
            for text in items:
                digest = _sha(text)
                doc_source = source or f"note:{digest[:12]}"
                documents.append(_Document(doc_source, text, markdown=True, content_key=digest[:16]))
                if replace and source:
                    replace_sources.add(doc_source)

        planned = self._plan(documents, tags)
        sources = list(dict.fromkeys(p.source for p in planned)) or list(dict.fromkeys(d.source for d in documents))
        info = await self._collection(qname)
        existing = await self._existing_hashes(qname, [p.id for p in planned]) if info else {}
        todo = [p for p in planned if existing.get(p.id) != p.hash]
        now = _now()

        batch_size = self.settings.embed_batch_size
        for start in range(0, len(todo), batch_size):
            batch = todo[start:start + batch_size]
            embedded = await self.embedder.embed([p.chunk.contextual_text() for p in batch], input_type="document")
            dim = len(embedded.items[0].dense)
            if info is None:
                await self._create_collection(qname, embedded.model, dim)
                info = await self._collection(qname)
            if info is not None:
                self._check_compatible(qname, info, embedded.model, dim)
            points = []
            for item, emb in zip(batch, embedded.items, strict=True):
                vector: dict[str, Any] = {DENSE: emb.dense}
                if emb.sparse and emb.sparse.get("indices"):
                    vector[SPARSE] = models.SparseVector(**emb.sparse)
                payload: dict[str, Any] = {
                    "text": item.chunk.text,
                    "source": item.source,
                    "chunk": item.chunk.index,
                    "created_at": now,
                    "hash": item.hash,
                }
                if item.chunk.section:
                    payload["section"] = item.chunk.section
                if tags:
                    payload["tags"] = tags
                points.append(models.PointStruct(id=item.id, vector=vector, payload=payload))
            async with self._qdrant_errors():
                await self.qdrant.upsert(qname, points=points, wait=True)
            if progress is not None:
                done = min(start + batch_size, len(todo))
                await progress(done, len(todo), f"Embedded {done} of {len(todo)} passages")

        removed = 0
        if info is not None and replace_sources:
            removed = await self._remove_stale(qname, planned, replace_sources)

        note = None
        if not todo and planned:
            note = "Everything was already in memory and unchanged."
        return RememberResult(
            collection=self.public_name(qname),
            sources=sources[:50],
            passages=len(planned),
            embedded=len(todo),
            unchanged=len(planned) - len(todo),
            removed=removed,
            skipped=skipped,
            note=note,
        )

    async def _remove_stale(self, qname: str, planned: list[_Planned], sources: set[str]) -> int:
        removed = 0
        async with self._qdrant_errors():
            for source in sorted(sources):
                keep = [p.id for p in planned if p.source == source]
                stale = models.Filter(
                    must=[models.FieldCondition(key="source", match=models.MatchValue(value=source))],
                    must_not=[models.HasIdCondition(has_id=keep)] if keep else None,
                )
                count = (await self.qdrant.count(qname, count_filter=stale, exact=True)).count
                if count:
                    await self.qdrant.delete(qname, points_selector=models.FilterSelector(filter=stale), wait=True)
                    removed += count
        return removed

    # --- search ----------------------------------------------------------------------

    async def search(
        self,
        collection: str | None,
        query: str,
        *,
        top_k: int = 5,
        rerank: bool = True,
        tags: list[str] | None = None,
        source: str | None = None,
        min_score: float | None = None,
    ) -> SearchResult:
        qname = self.qdrant_name(collection)
        name = self.public_name(qname)
        if not query.strip():
            raise ServiceError("`query` is empty.")
        info = await self._collection(qname)
        if info is None or not info.points_count:
            return SearchResult(
                collection=name, query=query, reranked=False, results=[],
                note=f"Nothing has been remembered in {name!r} yet. Use list_collections to see what exists.",
            )

        embedded = await self.embedder.embed([query], input_type="query")
        dense = embedded.items[0].dense
        sparse = embedded.items[0].sparse
        self._check_compatible(qname, info, embedded.model, len(dense))

        flt = self._filter(source, _clean_tags(tags))
        pool = max(top_k, self.settings.rerank_candidates) if rerank else top_k
        prefetch_limit = max(self.settings.prefetch_limit, pool)
        async with self._qdrant_errors():
            if sparse and sparse.get("indices"):
                response = await self.qdrant.query_points(
                    qname,
                    prefetch=[
                        models.Prefetch(query=dense, using=DENSE, limit=prefetch_limit, filter=flt),
                        models.Prefetch(query=models.SparseVector(**sparse), using=SPARSE, limit=prefetch_limit, filter=flt),
                    ],
                    query=models.FusionQuery(fusion=models.Fusion.RRF),
                    query_filter=flt,
                    limit=pool,
                    with_payload=True,
                )
            else:
                response = await self.qdrant.query_points(
                    qname, query=dense, using=DENSE, query_filter=flt, limit=pool, with_payload=True
                )
        points = response.points

        note = None
        reranked = False
        ordered: list[tuple[models.ScoredPoint, float]] = [(p, p.score) for p in points[:top_k]]
        if rerank and points:
            try:
                _, ranked = await self.reranker.rerank(
                    query, [_model_text(p.payload or {}) for p in points], top_n=min(top_k, len(points))
                )
                ordered = [(points[i], score) for i, score in ranked]
                reranked = True
            except ServiceError as exc:
                note = f"Reranker unavailable, results are in hybrid-search order. {exc}"
        if min_score is not None:
            ordered = [(p, s) for p, s in ordered if s >= min_score]

        results = []
        for point, score in ordered:
            payload = point.payload or {}
            results.append(Passage(
                text=payload.get("text", ""),
                source=payload.get("source", ""),
                score=round(float(score), 4),
                chunk=payload.get("chunk"),
                section=payload.get("section"),
                tags=payload.get("tags"),
                created_at=payload.get("created_at"),
            ))
        if not results and note is None:
            note = "No passage matched." if min_score is None else f"No passage scored at least {min_score}."
        return SearchResult(collection=name, query=query, reranked=reranked, results=results, note=note)

    # --- browse and delete -------------------------------------------------------------

    async def list_collections(self) -> CollectionsResult:
        prefix = self.settings.collection_prefix
        out = []
        async with self._qdrant_errors():
            response = await self.qdrant.get_collections()
            for item in sorted(response.collections, key=lambda c: c.name):
                if not item.name.startswith(prefix) or not _NAME.match(self.public_name(item.name)):
                    continue
                info = await self.qdrant.get_collection(item.name)
                out.append(CollectionInfo(
                    name=self.public_name(item.name),
                    passages=info.points_count or 0,
                    embedding_model=(info.config.metadata or {}).get("embedding_model"),
                ))
        return CollectionsResult(collections=out)

    async def list_sources(self, collection: str | None, limit: int = 100) -> SourcesResult:
        qname = self.qdrant_name(collection)
        if await self._collection(qname) is None:
            raise ServiceError(f"There is no collection {self.public_name(qname)!r}.")
        async with self._qdrant_errors():
            response = await self.qdrant.facet(qname, key="source", limit=limit, exact=True)
        sources = [SourceInfo(source=str(hit.value), passages=hit.count) for hit in response.hits]
        return SourcesResult(collection=self.public_name(qname), sources=sources, truncated=len(sources) >= limit)

    async def forget(self, collection: str, *, source: str | None = None, everything: bool = False) -> ForgetResult:
        qname = self.qdrant_name(collection)
        name = self.public_name(qname)
        source = source.strip() if source and source.strip() else None
        if everything and source:
            raise ServiceError("Pass either `source` or `everything=true`, not both.")
        if not everything and not source:
            raise ServiceError(
                "Nothing deleted: pass `source` (exactly as list_sources shows it) or `everything=true` "
                "to delete the whole collection."
            )
        info = await self._collection(qname)
        if info is None:
            raise ServiceError(f"There is no collection {name!r}.")
        async with self._qdrant_errors():
            if everything:
                await self.qdrant.delete_collection(qname)
                return ForgetResult(collection=name, deleted_passages=info.points_count or 0, dropped_collection=True)
            flt = self._filter(source, None)
            count = (await self.qdrant.count(qname, count_filter=flt, exact=True)).count
            if count:
                await self.qdrant.delete(qname, points_selector=models.FilterSelector(filter=flt), wait=True)
        return ForgetResult(collection=name, deleted_passages=count, dropped_collection=False)

    # --- raw model access ------------------------------------------------------------------

    async def embed(self, texts: list[str], *, input_type: str = "document", sparse: bool = False) -> VectorsResult:
        result = await self.embedder.embed(texts, input_type=input_type, sparse=sparse)
        vectors = [
            Vector(index=i, dense=item.dense, sparse=item.sparse if sparse else None)
            for i, item in enumerate(result.items)
        ]
        return VectorsResult(model=result.model, dimensions=len(vectors[0].dense) if vectors else 0, vectors=vectors)

    async def rerank(self, query: str, documents: list[str], top_n: int | None = None) -> RerankResult:
        model, ranked = await self.reranker.rerank(query, documents, top_n=top_n)
        return RerankResult(
            model=model,
            results=[RankedDocument(index=i, score=round(score, 4), document=documents[i]) for i, score in ranked],
        )
