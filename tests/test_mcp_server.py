"""MCP memory server: tools, chunking, safety rules.

The embedding and rerank services are replaced by deterministic fakes behind
httpx.MockTransport (bag-of-words vectors, word-overlap scores) and Qdrant by
qdrant-client's in-memory mode, so hybrid search really runs and nothing touches
the network or downloads a model.
"""
from __future__ import annotations

import asyncio
import importlib
import json
import math
import re
import sys
import zlib
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient
from mcp import Client
from qdrant_client import AsyncQdrantClient, models

from mcp_server.chunking import chunk_text, estimate_tokens
from mcp_server.config import MCPSettings
from mcp_server.memory import Memory
from mcp_server.server import _RequireApiKey, build_server, http_app
from mcp_server.services import EmbeddingClient, RerankClient

DIM = 1024
_TOKEN = re.compile(r"\w+")


def _tokens(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN.findall(text)]


class FakeServices:
    """Stands in for :9001 and :9002 and records every request body."""

    def __init__(self, model: str = "BAAI/bge-m3", sparse: bool = True):
        self.model = model
        self.sparse = sparse
        self.embed_requests: list[dict] = []
        self.rerank_requests: list[dict] = []
        self.rerank_down = False

    def _vector(self, text: str) -> tuple[list[float], dict]:
        counts: dict[int, float] = {}
        for token in _tokens(text):
            counts[zlib.crc32(token.encode()) % DIM] = counts.get(zlib.crc32(token.encode()) % DIM, 0.0) + 1.0
        dense = [0.0] * DIM
        for i, c in counts.items():
            dense[i] = c
        norm = math.sqrt(sum(v * v for v in dense)) or 1.0
        dense = [v / norm for v in dense]
        lexical: dict[int, float] = {}
        for token in set(_tokens(text)):
            lexical[zlib.crc32(token.encode()) % 250_000] = 0.3
        indices = sorted(lexical)
        return dense, {"indices": indices, "values": [lexical[i] for i in indices]}

    def embed_handler(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        self.embed_requests.append(body)
        data = []
        for i, text in enumerate(body["input"]):
            dense, sparse = self._vector(text)
            item = {"object": "embedding", "index": i, "embedding": dense}
            if self.sparse and body.get("return_sparse", True):
                item["sparse_qdrant"] = sparse
            data.append(item)
        return httpx.Response(200, json={"object": "list", "model": self.model, "data": data,
                                         "usage": {"prompt_tokens": 1, "total_tokens": 1}})

    def rerank_handler(self, request: httpx.Request) -> httpx.Response:
        if self.rerank_down:
            raise httpx.ConnectError("connection refused")
        body = json.loads(request.content)
        self.rerank_requests.append(body)
        query = set(_tokens(body["query"]))
        scored = []
        for i, doc in enumerate(body["documents"]):
            overlap = len(query & set(_tokens(doc))) / max(len(query), 1)
            scored.append({"index": i, "relevance_score": overlap})
        scored.sort(key=lambda r: r["relevance_score"], reverse=True)
        if body.get("top_n"):
            scored = scored[: body["top_n"]]
        return httpx.Response(200, json={"model": "BAAI/bge-reranker-v2-m3", "results": scored})


def make_memory(tmp_path: Path, services: FakeServices | None = None, qdrant: AsyncQdrantClient | None = None, **overrides):
    services = services or FakeServices()
    settings = MCPSettings(_env_file=None, allowed_paths=str(tmp_path), **overrides)
    memory = Memory(
        settings,
        qdrant=qdrant or AsyncQdrantClient(location=":memory:"),
        embedder=EmbeddingClient("http://embed.test", transport=httpx.MockTransport(services.embed_handler), batch_size=4),
        reranker=RerankClient("http://rerank.test", transport=httpx.MockTransport(services.rerank_handler)),
    )
    return memory, services


async def call(client: Client, tool: str, **arguments):
    result = await client.call_tool(tool, arguments)
    if result.is_error:
        return {"error": result.content[0].text}
    return result.structured_content


NOTES = [
    "The customs classifier uses bge-m3 for retrieval and Qdrant for storage.",
    "Lunch on Fridays is at the Georgian place near the office.",
    "Deploy the reranker on port 9002; the embedding service listens on 9001.",
]


# --- Tools over MCP ---------------------------------------------------------------


def test_tools_are_listed_with_safety_annotations(tmp_path):
    async def scenario():
        memory, _ = make_memory(tmp_path)
        async with Client(build_server(memory.settings, memory)) as client:
            tools = {t.name: t for t in (await client.list_tools()).tools}
            assert set(tools) == {"remember", "search", "list_collections", "list_sources", "forget", "rerank", "embed"}
            assert tools["forget"].annotations.destructive_hint is True
            for name in ("search", "list_collections", "list_sources", "rerank", "embed"):
                assert tools[name].annotations.read_only_hint is True
            assert tools["search"].output_schema["properties"]["results"]
            assert "ctx" not in tools["remember"].input_schema["properties"]

    asyncio.run(scenario())


def test_remember_then_search_round_trip(tmp_path):
    async def scenario():
        memory, services = make_memory(tmp_path)
        async with Client(build_server(memory.settings, memory)) as client:
            saved = await call(client, "remember", texts=NOTES, source="team-notes", tags=["work"])
            assert saved["passages"] == 3 and saved["embedded"] == 3 and saved["unchanged"] == 0
            assert saved["collection"] == "memory" and saved["sources"] == ["team-notes"]

            found = await call(client, "search", query="which port does the reranker use", top_k=2)
            assert found["reranked"] is True
            assert found["results"][0]["text"] == NOTES[2]
            assert found["results"][0]["source"] == "team-notes"
            assert found["results"][0]["tags"] == ["work"]
            assert 0 <= found["results"][0]["score"] <= 1
            assert len(found["results"]) == 2

        # Documents are embedded as documents, the query as a query, both compact.
        assert all(r["compact"] is True for r in services.embed_requests)
        assert {r["input_type"] for r in services.embed_requests[:-1]} == {"document"}
        assert services.embed_requests[-1]["input_type"] == "query"
        assert services.rerank_requests[-1]["top_n"] == 2
        assert services.rerank_requests[-1]["return_documents"] is False

    asyncio.run(scenario())


def test_remembering_the_same_text_twice_does_not_duplicate(tmp_path):
    async def scenario():
        memory, services = make_memory(tmp_path)
        first = await memory.remember(None, texts=NOTES)
        calls = len(services.embed_requests)
        second = await memory.remember(None, texts=NOTES)
        assert first.embedded == 3
        assert second.embedded == 0 and second.unchanged == 3
        assert second.note and "unchanged" in second.note
        assert len(services.embed_requests) == calls  # nothing re-embedded
        count = await memory.qdrant.count("mcp_memory", exact=True)
        assert count.count == 3
        # Without a source every note gets its own id-like source.
        assert all(s.startswith("note:") for s in first.sources)

    asyncio.run(scenario())


def test_texts_under_one_source_accumulate_unless_replace(tmp_path):
    async def scenario():
        memory, _ = make_memory(tmp_path)
        await memory.remember("prefs", texts=["Prefers dark mode."], source="user-prefs")
        await memory.remember("prefs", texts=["Uses Cursor as the editor."], source="user-prefs")
        sources = await memory.list_sources("prefs")
        assert [(s.source, s.passages) for s in sources.sources] == [("user-prefs", 2)]

        replaced = await memory.remember("prefs", texts=["Uses Zed as the editor."], source="user-prefs", replace=True)
        assert replaced.removed == 2
        found = await memory.search("prefs", "editor", top_k=5)
        assert [p.text for p in found.results] == ["Uses Zed as the editor."]

    asyncio.run(scenario())


def test_search_filters_by_tags_and_source(tmp_path):
    async def scenario():
        memory, _ = make_memory(tmp_path)
        await memory.remember(None, texts=["Port 9001 serves embeddings."], source="ops", tags=["infra"])
        await memory.remember(None, texts=["Port 9003 serves the MCP server."], source="mcp", tags=["agents"])
        by_tag = await memory.search(None, "port", tags=["agents"])
        assert [p.source for p in by_tag.results] == ["mcp"]
        by_source = await memory.search(None, "port", source="ops")
        assert [p.source for p in by_source.results] == ["ops"]

    asyncio.run(scenario())


def test_search_without_reranker_and_with_min_score(tmp_path):
    async def scenario():
        memory, services = make_memory(tmp_path)
        await memory.remember(None, texts=NOTES)
        plain = await memory.search(None, "reranker port", rerank=False, top_k=3)
        assert plain.reranked is False and len(plain.results) == 3
        assert services.rerank_requests == []

        strict = await memory.search(None, "reranker port", min_score=0.99)
        assert [p.text for p in strict.results] == [NOTES[2]]

        services.rerank_down = True
        fallback = await memory.search(None, "reranker port")
        assert fallback.reranked is False
        assert fallback.results and "Reranker unavailable" in fallback.note

    asyncio.run(scenario())


def test_dense_only_embedding_model_still_works(tmp_path):
    async def scenario():
        memory, _ = make_memory(tmp_path, services=FakeServices(model="Qwen/Qwen3-Embedding-0.6B", sparse=False))
        await memory.remember(None, texts=NOTES)
        found = await memory.search(None, "Georgian lunch", top_k=1)
        assert found.results[0].text == NOTES[1]

    asyncio.run(scenario())


def test_search_refuses_vectors_from_another_model(tmp_path):
    async def scenario():
        qdrant = AsyncQdrantClient(location=":memory:")
        memory, _ = make_memory(tmp_path, qdrant=qdrant)
        await memory.remember(None, texts=NOTES)
        other, _ = make_memory(tmp_path, services=FakeServices(model="intfloat/multilingual-e5-large"), qdrant=qdrant)
        async with Client(build_server(other.settings, other)) as client:
            result = await call(client, "search", query="lunch")
            assert "was built with BAAI/bge-m3" in result["error"]
            assert "multilingual-e5-large" in result["error"]
            result = await call(client, "remember", texts=["new note"])
            assert "was built with BAAI/bge-m3" in result["error"]

    asyncio.run(scenario())


def test_search_on_an_empty_collection_explains(tmp_path):
    async def scenario():
        memory, services = make_memory(tmp_path)
        found = await memory.search("nothing-here", "anything")
        assert found.results == [] and "Nothing has been remembered" in found.note
        assert services.embed_requests == []

    asyncio.run(scenario())


# --- Files ----------------------------------------------------------------------------


def _write_notes_folder(root: Path) -> Path:
    folder = root / "notes"
    (folder / "sub").mkdir(parents=True)
    (folder / ".hidden").mkdir()
    (folder / "guide.md").write_text(
        "# Guide\n\nServices-BGE runs bge-m3 locally.\n\n## Install\n\nRun uv sync, then service.sh install.\n",
        encoding="utf-8",
    )
    (folder / "sub" / "todo.txt").write_text("Buy a new SSD for the Qdrant snapshots.", encoding="utf-8")
    (folder / ".hidden" / "secret.md").write_text("should never be read", encoding="utf-8")
    (folder / "script.py").write_text("print('not a note')", encoding="utf-8")
    return folder


def test_remember_folder_reads_only_notes_and_rereads_replace(tmp_path):
    async def scenario():
        memory, _ = make_memory(tmp_path, chunk_tokens=40, chunk_overlap_tokens=0)
        folder = _write_notes_folder(tmp_path)
        saved = await memory.remember("docs", path=str(folder))
        assert len(saved.sources) == 2
        assert any(s.endswith("notes/guide.md") for s in saved.sources)
        assert not any("secret" in s or "script" in s for s in saved.sources)

        found = await memory.search("docs", "how to install", top_k=1)
        assert found.results[0].section == "Guide > Install"

        # Unchanged files are not embedded again.
        again = await memory.remember("docs", path=str(folder))
        assert again.embedded == 0 and again.removed == 0

        # A shorter file replaces its old passages.
        (folder / "guide.md").write_text("# Guide\n\nShort now.\n", encoding="utf-8")
        edited = await memory.remember("docs", path=str(folder / "guide.md"))
        assert edited.removed >= 1
        guide = [s for s in (await memory.list_sources("docs")).sources if s.source.endswith("guide.md")]
        assert guide[0].passages == edited.passages

    asyncio.run(scenario())


@pytest.mark.parametrize(
    ("make_path", "message"),
    [
        (lambda tmp: "notes/guide.md", "absolute path"),
        (lambda tmp: str(tmp.parent), "outside the folders"),
        (lambda tmp: str(tmp / "notes" / "script.py"), "files can be remembered"),
        (lambda tmp: str(tmp / "missing.md"), "does not exist"),
    ],
)
def test_remember_path_is_restricted(tmp_path, make_path, message):
    async def scenario():
        memory, _ = make_memory(tmp_path)
        _write_notes_folder(tmp_path)
        async with Client(build_server(memory.settings, memory)) as client:
            result = await call(client, "remember", path=make_path(tmp_path))
            assert message in result["error"]

    asyncio.run(scenario())


def test_remember_needs_exactly_one_input(tmp_path):
    async def scenario():
        memory, _ = make_memory(tmp_path)
        async with Client(build_server(memory.settings, memory)) as client:
            assert "either `texts`" in (await call(client, "remember"))["error"]
            assert "either `texts`" in (await call(client, "remember", texts=["a"], path=str(tmp_path)))["error"]
            assert "empty" in (await call(client, "remember", texts=["  "]))["error"]
            assert "Invalid collection name" in (await call(client, "remember", texts=["a"], collection="../x"))["error"]

    asyncio.run(scenario())


# --- Trust tiers -------------------------------------------------------------------------

CANON_PRICE = "The pilot price: two paid pilots at EUR 900 to 2,400 each."
FALSE_PRICE = "The pilot price is EUR 90."


def _write_canon(root: Path) -> tuple[Path, Path]:
    canon = root / "canon"
    canon.mkdir()
    (canon / "STATE.md").write_text(f"# State\n\n{CANON_PRICE}\n", encoding="utf-8")
    loose = root / "scratch.md"
    loose.write_text("# Scratch\n\nThe pilot price might change next month.\n", encoding="utf-8")
    return canon, loose


def test_without_canon_paths_nothing_gets_a_tier(tmp_path):
    async def scenario():
        memory, _ = make_memory(tmp_path)
        canon, _ = _write_canon(tmp_path)
        await memory.remember(None, path=str(canon))
        await memory.remember(None, texts=[FALSE_PRICE], source="bot-x")
        records, _ = await memory.qdrant.scroll("mcp_memory", with_payload=True)
        assert all("tier" not in r.payload for r in records)
        found = await memory.search(None, "what is the pilot price")
        assert found.prefer_tier is None
        assert all(p.tier is None for p in found.results)

    asyncio.run(scenario())


def test_tier_follows_where_the_text_came_from(tmp_path):
    async def scenario():
        canon, loose = _write_canon(tmp_path)
        memory, _ = make_memory(tmp_path, canon_paths=str(canon))
        await memory.remember(None, path=str(canon))
        await memory.remember(None, path=str(loose))
        await memory.remember(None, texts=[FALSE_PRICE], source="bot-x")
        records, _ = await memory.qdrant.scroll("mcp_memory", with_payload=True)
        tiers = {r.payload["source"].rsplit("/", 1)[-1]: r.payload["tier"] for r in records}
        assert tiers == {"STATE.md": "canon", "scratch.md": "inbox", "bot-x": "inbox"}

        async with Client(build_server(memory.settings, memory)) as client:
            # A writer cannot claim a tier: the tool has no such argument.
            tools = {t.name: t for t in (await client.list_tools()).tools}
            assert "tier" not in tools["remember"].input_schema["properties"]
            assert {"tiers", "prefer_tier"} <= set(tools["search"].input_schema["properties"])

            found = await call(client, "search", query="what is the pilot price", top_k=3)
            by_source = {r["source"].rsplit("/", 1)[-1]: r["tier"] for r in found["results"]}
            assert by_source["bot-x"] == "inbox" and by_source["STATE.md"] == "canon"

            canon_only = await call(client, "search", query="what is the pilot price", tiers=["canon"])
            assert [r["tier"] for r in canon_only["results"]] == ["canon"]

    asyncio.run(scenario())


def test_prefer_tier_puts_canon_above_a_contradicting_note(tmp_path):
    async def scenario():
        canon, _ = _write_canon(tmp_path)
        memory, _ = make_memory(tmp_path, canon_paths=str(canon))
        await memory.remember(None, path=str(canon))
        await memory.remember(None, texts=[FALSE_PRICE], source="bot-x")
        query = "what is the pilot price"

        # Plain ranking: the short false note matches the question best (0.8 vs 0.6).
        plain = await memory.search(None, query)
        assert plain.results[0].text == FALSE_PRICE and plain.prefer_tier is None

        preferred = await memory.search(None, query, prefer_tier="canon")
        assert preferred.prefer_tier == "canon"
        assert CANON_PRICE in preferred.results[0].text and preferred.results[0].tier == "canon"
        # The note is still there, its score multiplied by the penalty.
        note = next(p for p in preferred.results if p.source == "bot-x")
        assert note.score == pytest.approx(0.8 * 0.5)

        # Without the reranker, the fusion order is weighted the same way.
        fused = await memory.search(None, query, prefer_tier="canon", rerank=False)
        assert fused.results[0].tier == "canon"

        # The server default turns it on for agents that do not pass the argument.
        memory.settings.prefer_tier = "canon"
        assert (await memory.search(None, query)).results[0].tier == "canon"
        assert (await memory.search(None, query, prefer_tier="")).results[0].text == FALSE_PRICE

    asyncio.run(scenario())


def test_prefer_tier_survives_a_flood_of_notes(tmp_path):
    async def scenario():
        canon, _ = _write_canon(tmp_path)
        memory, _ = make_memory(tmp_path, canon_paths=str(canon), prefetch_limit=3, rerank_candidates=3)
        await memory.remember(None, path=str(canon))
        flood = [f"The pilot price is EUR 90, update {i}." for i in range(8)]
        await memory.remember(None, texts=flood, source="bot-x")
        query = "what is the pilot price"

        plain = await memory.search(None, query, top_k=3)
        assert all(p.source == "bot-x" for p in plain.results)  # canon pushed out of the pool

        preferred = await memory.search(None, query, top_k=3, prefer_tier="canon")
        assert preferred.results[0].tier == "canon"

    asyncio.run(scenario())


def test_rereading_a_file_after_enabling_tiers_relabels_without_embedding(tmp_path):
    async def scenario():
        canon, _ = _write_canon(tmp_path)
        memory, services = make_memory(tmp_path)
        await memory.remember(None, path=str(canon))
        calls = len(services.embed_requests)

        memory.settings.canon_paths = str(canon)
        again = await memory.remember(None, path=str(canon))
        assert again.embedded == 0 and again.unchanged == again.passages
        assert len(services.embed_requests) == calls
        records, _ = await memory.qdrant.scroll("mcp_memory", with_payload=True)
        assert {r.payload["tier"] for r in records} == {"canon"}

    asyncio.run(scenario())


# --- Forget and isolation ------------------------------------------------------------


def test_forget_requires_an_explicit_target(tmp_path):
    async def scenario():
        memory, _ = make_memory(tmp_path)
        await memory.remember("work", texts=NOTES[:2], source="a")
        await memory.remember("work", texts=NOTES[2:], source="b")
        async with Client(build_server(memory.settings, memory)) as client:
            refused = await call(client, "forget", collection="work")
            assert "Nothing deleted" in refused["error"]
            assert (await memory.qdrant.count("mcp_work", exact=True)).count == 3

            by_source = await call(client, "forget", collection="work", source="a")
            assert by_source == {"collection": "work", "deleted_passages": 2, "dropped_collection": False}

            dropped = await call(client, "forget", collection="work", everything=True)
            assert dropped["dropped_collection"] is True and dropped["deleted_passages"] == 1
            assert not await memory.qdrant.collection_exists("mcp_work")

    asyncio.run(scenario())


def test_other_projects_collections_are_invisible_and_safe(tmp_path):
    async def scenario():
        memory, _ = make_memory(tmp_path)
        await memory.qdrant.create_collection(
            "tnved", vectors_config=models.VectorParams(size=4, distance=models.Distance.COSINE)
        )
        await memory.remember("notes", texts=["hello"])
        listed = await memory.list_collections()
        assert [c.name for c in listed.collections] == ["notes"]
        assert listed.collections[0].embedding_model == "BAAI/bge-m3"
        async with Client(build_server(memory.settings, memory)) as client:
            result = await call(client, "forget", collection="tnved", everything=True)
            assert "no collection 'tnved'" in result["error"]
        assert await memory.qdrant.collection_exists("tnved")

    asyncio.run(scenario())


# --- Thin wrappers and errors --------------------------------------------------------


def test_rerank_and_embed_tools(tmp_path):
    async def scenario():
        memory, services = make_memory(tmp_path)
        async with Client(build_server(memory.settings, memory)) as client:
            ranked = await call(client, "rerank", query="reranker port", documents=NOTES, top_n=1)
            assert ranked["results"] == [{"index": 2, "score": 1.0, "document": NOTES[2]}]

            vectors = await call(client, "embed", texts=["a b"], input_type="query")
            assert vectors["dimensions"] == DIM and vectors["vectors"][0]["sparse"] is None
            assert services.embed_requests[-1]["return_sparse"] is False

            with_sparse = await call(client, "embed", texts=["a b"], sparse=True)
            assert with_sparse["vectors"][0]["sparse"]["indices"]

    asyncio.run(scenario())


def test_unreachable_services_give_actionable_errors(tmp_path):
    def refuse(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    async def scenario():
        settings = MCPSettings(_env_file=None, allowed_paths=str(tmp_path))
        memory = Memory(
            settings,
            qdrant=AsyncQdrantClient(location=":memory:"),
            embedder=EmbeddingClient("http://127.0.0.1:9001", transport=httpx.MockTransport(refuse)),
            reranker=RerankClient("http://127.0.0.1:9002", transport=httpx.MockTransport(refuse)),
        )
        async with Client(build_server(settings, memory)) as client:
            result = await call(client, "remember", texts=["hello"])
            assert "embedding service at http://127.0.0.1:9001 is not reachable" in result["error"]
            assert "service.sh" in result["error"]
            result = await call(client, "rerank", query="q", documents=["d"])
            assert "rerank service" in result["error"]

    asyncio.run(scenario())


def test_service_api_key_is_sent(monkeypatch):
    monkeypatch.setenv("SERVICE_API_KEY", "shared-key")
    monkeypatch.setenv("MCP_RERANKER_API_KEY", "rerank-key")
    settings = MCPSettings(_env_file=None)
    assert settings.embedding_api_key == "shared-key"
    assert settings.reranker_api_key == "rerank-key"

    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["auth"] = request.headers.get("authorization")
        return httpx.Response(200, json={"model": "m", "data": [{"index": 0, "embedding": [1.0]}]})

    client = EmbeddingClient("http://embed.test", api_key=settings.embedding_api_key, transport=httpx.MockTransport(handler))
    asyncio.run(client.embed(["x"]))
    assert seen["auth"] == "Bearer shared-key"


def test_service_errors_are_translated():
    def handler(request: httpx.Request) -> httpx.Response:
        status = {"/v1/embeddings": 401, "/v1/rerank": 503}[request.url.path]
        return httpx.Response(status, json={"detail": "nope"})

    transport = httpx.MockTransport(handler)
    with pytest.raises(Exception, match="rejected the API key"):
        asyncio.run(EmbeddingClient("http://e.test", transport=transport).embed(["x"]))
    with pytest.raises(Exception, match="still loading"):
        asyncio.run(RerankClient("http://r.test", transport=transport).rerank("q", ["d"]))


# --- Contract with the real services (fake models) ----------------------------------------


def test_clients_match_the_real_service_contract(monkeypatch):
    from test_service_endpoints import _fake_flagembedding

    monkeypatch.setitem(sys.modules, "FlagEmbedding", _fake_flagembedding())
    embedding = importlib.import_module("embedding_service.main")
    reranker = importlib.import_module("reranker_service.main")
    for module in (embedding, reranker):
        module._state.clear()
        monkeypatch.setattr(module.settings, "device", "cpu", raising=False)
        monkeypatch.setattr(module.settings, "api_key", None, raising=False)

    with TestClient(embedding.app), TestClient(reranker.app):  # runs startup: loads the fake models
        async def scenario():
            embed = EmbeddingClient("http://e.test", transport=httpx.ASGITransport(app=embedding.app))
            result = await embed.embed(["one", "two"], input_type="query")
            assert result.model == "BAAI/bge-m3"
            assert [len(item.dense) for item in result.items] == [1024, 1024]
            assert set(result.items[0].sparse) == {"indices", "values"}

            rerank = RerankClient("http://r.test", transport=httpx.ASGITransport(app=reranker.app))
            model, ranked = await rerank.rerank("q", ["a", "b", "c"], top_n=2)
            assert model == "BAAI/bge-reranker-v2-m3"
            assert [i for i, _ in ranked] == [0, 1]

        asyncio.run(scenario())

    for module in (embedding, reranker):
        module._state.clear()


# --- HTTP transport ------------------------------------------------------------------


def test_http_transport_requires_api_key_when_configured(tmp_path):
    settings = MCPSettings(_env_file=None, api_key="mcp-secret")
    memory, _ = make_memory(tmp_path)
    app = http_app(build_server(settings, memory), settings)
    assert isinstance(app, _RequireApiKey)

    with TestClient(app, base_url="http://127.0.0.1:9003") as client:
        assert client.get("/healthz").status_code == 200
        init = {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {
            "protocolVersion": "2025-11-25", "capabilities": {}, "clientInfo": {"name": "t", "version": "1"}}}
        headers = {"Accept": "application/json, text/event-stream"}
        assert client.post("/mcp", json=init, headers=headers).status_code == 401
        assert client.post("/mcp", json=init, headers={**headers, "Authorization": "Bearer wrong"}).status_code == 401
        ok = client.post("/mcp", json=init, headers={**headers, "Authorization": "Bearer mcp-secret"})
        assert ok.status_code == 200, ok.text


def test_http_refuses_non_loopback_without_key(monkeypatch):
    from mcp_server import server

    monkeypatch.setenv("MCP_HOST", "0.0.0.0")
    monkeypatch.delenv("MCP_API_KEY", raising=False)
    monkeypatch.setattr(server.MCPSettings, "model_config", {**server.MCPSettings.model_config, "env_file": None})
    with pytest.raises(SystemExit):
        server.main(["--transport", "http"])


# --- Chunking --------------------------------------------------------------------------


def test_chunks_follow_headings_and_overlap():
    sentences = " ".join(f"Sentence {i} describes one detail of the setup." for i in range(80))
    text = f"# Guide\n\nIntro.\n\n## Install\n\n{sentences}\n\n## Usage\n\nShort note.\n"
    chunks = chunk_text(text, max_tokens=120, overlap_tokens=30)

    assert chunks[0].text.startswith("# Guide") and chunks[0].section == "Guide"
    assert chunks[-1].section == "Guide > Usage" and chunks[-1].text.startswith("## Usage")
    assert all(estimate_tokens(c.text) <= 120 for c in chunks)
    install = [c for c in chunks if c.section == "Guide > Install"]
    assert len(install) > 2
    # Consecutive passages in one section share text.
    last_sentence = install[1].text.rsplit("Sentence", 1)[1]
    assert ("Sentence" + last_sentence) in install[2].text
    # The heading path is given to the models when a passage does not start with it.
    assert install[1].contextual_text().startswith("Guide > Install\n\n")
    assert [c.index for c in chunks] == list(range(len(chunks)))


def test_chunks_keep_code_fences_and_handle_odd_input():
    text = "# Code\n\n```bash\n# not a heading\nuv sync\n```\n\nAfter."
    chunks = chunk_text(text, max_tokens=200)
    assert len(chunks) == 1 and "# not a heading" in chunks[0].text
    assert chunks[0].section == "Code"

    assert chunk_text("", max_tokens=100) == []
    blob = chunk_text("x" * 5000, max_tokens=100, overlap_tokens=0)
    assert len(blob) > 1 and all(estimate_tokens(c.text) <= 100 for c in blob)
    plain = chunk_text("# not markdown\n\ntext", max_tokens=100, markdown=False)
    assert plain[0].section is None
    with pytest.raises(ValueError):
        chunk_text("x", max_tokens=10)
