"""Team canon: sync from a git clone, degraded keyword search, status.

The canon repository is a local git repo in a temp folder (no network); the services
are the fakes from test_mcp_server, Qdrant runs in memory, and an outage is a client
that refuses every call.
"""
from __future__ import annotations

import asyncio
import subprocess
from pathlib import Path

import httpx
import pytest
from mcp import Client
from qdrant_client import AsyncQdrantClient

from mcp_server.canon import load_manifest, select_files, sync
from mcp_server.memory import Memory
from mcp_server.server import build_server
from mcp_server.services import EmbeddingClient, RerankClient
from test_mcp_server import FakeServices, call, make_memory

CANON = "# Canon\n\n## Pricing\n\nA pilot costs EUR 900 to 2,400 for four weeks.\n\n## Branches\n\nThe production checker runs on branch stable.\n"
STATE = "# State\n\nGoal: two paid pilots by December.\n"
DECISION = "# 0001 Keep local OCR\n\nWe keep the local OCR engine because client documents must stay on site.\n"
INBOX = "---\nsource: scout\n---\n\n# Scout notes\n\nCold emails get a 2 % reply rate.\n"


def _git(cwd: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-c", "user.name=test", "-c", "user.email=test@example.com", "-c", "commit.gpgsign=false", *args],
        cwd=cwd, check=True, capture_output=True, text=True,
    )
    return result.stdout.strip()


def _canon_repo(root: Path) -> Path:
    repo = root / "origin"
    (repo / "decisions").mkdir(parents=True)
    (repo / "inbox").mkdir()
    (repo / "private").mkdir()
    (repo / "CANON.md").write_text(CANON, encoding="utf-8")
    (repo / "STATE.md").write_text(STATE, encoding="utf-8")
    (repo / "decisions" / "0001-ocr.md").write_text(DECISION, encoding="utf-8")
    (repo / "inbox" / "2026-11-01-scout.md").write_text(INBOX, encoding="utf-8")
    (repo / "private" / "agent-memory.md").write_text("A personal agent memory: never indexed.", encoding="utf-8")
    (repo / "tool.py").write_text("print('code is not indexed')", encoding="utf-8")
    _git(repo, "init", "-q", "-b", "main")
    _git(repo, "add", ".")
    _git(repo, "commit", "-q", "-m", "canon")
    return repo


class DeadQdrant:
    """Qdrant that refuses every call, like a stopped Docker container."""

    def __getattr__(self, name):
        async def refuse(*args, **kwargs):
            raise httpx.ConnectError("connection refused")
        return refuse


def _refuse(request: httpx.Request) -> httpx.Response:
    raise httpx.ConnectError("connection refused")


def test_select_files_takes_only_listed_markdown(tmp_path):
    repo = _canon_repo(tmp_path)
    canon, inbox = select_files(repo, ["CANON.md", "STATE.md", "ORG.md", "decisions"], ["inbox", "CANON.md"])
    assert [p.name for p in canon] == ["CANON.md", "STATE.md", "0001-ocr.md"]  # ORG.md is missing: skipped
    assert [p.name for p in inbox] == ["2026-11-01-scout.md"]  # CANON.md stays canon


def test_sync_indexes_canon_and_inbox_and_is_idempotent(tmp_path):
    async def scenario():
        repo = _canon_repo(tmp_path)
        qdrant = AsyncQdrantClient(location=":memory:")
        services = FakeServices()
        memory, _ = make_memory(tmp_path, services=services, qdrant=qdrant)
        manifest = await sync(memory, repo=str(repo), directory=str(tmp_path / "clone"))

        assert manifest.commit == _git(repo, "rev-parse", "HEAD")
        assert manifest.collection == "canon" and manifest.chunk_tokens == 150
        assert {f.path: f.tier for f in manifest.files} == {
            "CANON.md": "canon", "STATE.md": "canon", "decisions/0001-ocr.md": "canon", "inbox/2026-11-01-scout.md": "inbox",
        }
        assert load_manifest(memory.settings) == manifest
        clone = tmp_path / "clone"
        assert _git(clone, "status", "--porcelain") == ""  # read-only: nothing written into the clone

        # A fresh server process picks the canon up from the manifest.
        server_memory, _ = make_memory(tmp_path, services=services, qdrant=qdrant)
        sources = await server_memory.list_sources("canon")
        assert not any(s.source.endswith(("agent-memory.md", ".py")) for s in sources.sources)
        found = await server_memory.search("canon", "pilot costs EUR", top_k=3)
        assert found.prefer_tier == "canon"  # canon-first by default in the canon collection
        assert found.results[0].tier == "canon" and "EUR 900" in found.results[0].text
        assert (await server_memory.search("canon", "pilot", prefer_tier="")).prefer_tier is None

        # An agent's own note in the canon collection is unreviewed.
        await server_memory.remember("canon", texts=["The pilot costs EUR 90."], source="some-bot")
        tiers = {s.source: s.tier for s in (await server_memory.status()).collections[0].sources}
        assert tiers["some-bot"] == "inbox"

        # Second sync: nothing changed, nothing re-embedded; the agent's note stays.
        again = await sync(make_memory(tmp_path, services=services, qdrant=qdrant)[0])
        assert again.embedded == 0 and again.unchanged == manifest.embedded and again.removed == []

        # A deleted decision disappears from memory on the next sync.
        _git(repo, "rm", "-q", "decisions/0001-ocr.md")
        _git(repo, "commit", "-q", "-m", "drop decision")
        third = await sync(make_memory(tmp_path, services=services, qdrant=qdrant)[0])
        assert third.removed == ["decisions/0001-ocr.md"]
        assert third.commit == _git(repo, "rev-parse", "HEAD")
        left = {s.source for s in (await server_memory.list_sources("canon")).sources}
        assert not any(s.endswith("0001-ocr.md") for s in left) and "some-bot" in left

    asyncio.run(scenario())


def test_sync_needs_a_place_for_the_clone(tmp_path):
    memory, _ = make_memory(tmp_path)
    with pytest.raises(Exception, match="--dir"):
        asyncio.run(sync(memory))
    with pytest.raises(Exception, match="--repo"):
        asyncio.run(sync(memory, directory=str(tmp_path / "missing")))


def _degraded_memory(tmp_path: Path, *, qdrant_down: bool, embedding_down: bool = False, canon: bool = True) -> Memory:
    folder = tmp_path / "canon"
    folder.mkdir(exist_ok=True)
    (folder / "CANON.md").write_text(CANON, encoding="utf-8")
    services = FakeServices()
    memory, _ = make_memory(
        tmp_path, services=services, canon_paths=str(folder) if canon else "",
        qdrant=DeadQdrant() if qdrant_down else AsyncQdrantClient(location=":memory:"),
    )
    return memory


async def _take_down_embedding(memory: Memory) -> None:
    await memory.remember(None, texts=["An agent note saved while everything was up."])
    memory.embedder = EmbeddingClient("http://127.0.0.1:9001", transport=httpx.MockTransport(_refuse))


@pytest.mark.parametrize(("qdrant_down", "embedding_down", "reason"), [
    (True, False, "Qdrant at"),
    (False, True, "embedding service at http://127.0.0.1:9001 is not reachable"),
])
def test_search_falls_back_to_keywords_over_canon(tmp_path, qdrant_down, embedding_down, reason):
    async def scenario():
        memory = _degraded_memory(tmp_path, qdrant_down=qdrant_down, embedding_down=embedding_down)
        if embedding_down:
            await _take_down_embedding(memory)
        async with Client(build_server(memory.settings, memory)) as client:
            found = await call(client, "search", query="Which branch does the production checker use?")
        assert found["degraded"] is True and found["reranked"] is False
        assert reason in found["note"] and "keyword search" in found["note"] and "CANON.md directly" in found["note"]
        top = found["results"][0]
        assert "branch stable" in top["text"] and top["tier"] == "canon" and top["source"].endswith("CANON.md")
        assert 0 < top["score"] <= 1

        # Filters that canon files cannot match give an empty, explained answer.
        none = await memory.search(None, "branch", tags=["work"])
        assert none.degraded and none.results == [] and "No canon passage" in none.note

    asyncio.run(scenario())


def test_outage_without_canon_files_is_still_an_error(tmp_path):
    async def scenario():
        memory = _degraded_memory(tmp_path, qdrant_down=True, embedding_down=False, canon=False)
        async with Client(build_server(memory.settings, memory)) as client:
            result = await call(client, "search", query="branch")
        assert "Qdrant at" in result["error"] and "not reachable" in result["error"]

    asyncio.run(scenario())


def test_status_in_every_mode(tmp_path):
    async def scenario():
        repo = _canon_repo(tmp_path)
        qdrant = AsyncQdrantClient(location=":memory:")
        services = FakeServices()
        await sync(make_memory(tmp_path, services=services, qdrant=qdrant)[0], repo=str(repo), directory=str(tmp_path / "clone"))

        memory, _ = make_memory(tmp_path, services=services, qdrant=qdrant)
        full = await memory.status()
        assert full.mode == "full" and full.canon is not None
        assert "fully working" in full.summary and full.canon.commit[:7] in full.summary
        assert "3 curated files" in full.summary and "1 inbox files" in full.summary
        canon = next(c for c in full.collections if c.name == "canon")
        assert {s.tier for s in canon.sources} == {"canon", "inbox"}
        async with Client(build_server(memory.settings, memory)) as client:
            tool = await call(client, "memory_status")
            assert tool["mode"] == "full" and tool["canon"]["collection"] == "canon"
            assert 'collection="canon"' in (client.instructions or "")

        services.rerank_down = True
        assert (await memory.status()).mode == "no-rerank"

        memory._qdrant = DeadQdrant()
        degraded = await memory.status()
        assert degraded.mode == "degraded" and "keyword" in degraded.summary and "CANON.md directly" in degraded.summary
        found = await memory.search("canon", "production checker branch")
        assert found.degraded and found.results[0].source.endswith("CANON.md")

        nothing, _ = make_memory(tmp_path / "other", qdrant=DeadQdrant())
        down = await nothing.status()
        assert down.mode == "down" and "cannot answer" in down.summary

    (tmp_path / "other").mkdir()
    asyncio.run(scenario())


def test_status_reports_an_unreachable_embedding_service(tmp_path):
    async def scenario():
        memory = _degraded_memory(tmp_path, qdrant_down=False, embedding_down=True)
        await _take_down_embedding(memory)
        status = await memory.status()
        assert status.mode == "degraded"
        assert "not reachable" in status.services["embedding"] and status.services["qdrant"].startswith("ok")

    asyncio.run(scenario())


def test_clients_report_readiness(tmp_path):
    services = FakeServices()
    embed = EmbeddingClient("http://e.test", transport=httpx.MockTransport(services.embed_handler))
    assert asyncio.run(embed.ready()) == (True, "BAAI/bge-m3")
    dead = RerankClient("http://127.0.0.1:9002", transport=httpx.MockTransport(_refuse))
    ok, reason = asyncio.run(dead.ready())
    assert not ok and "not reachable" in reason
