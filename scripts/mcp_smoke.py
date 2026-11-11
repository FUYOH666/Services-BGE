"""End-to-end check of the MCP memory server against the live services and Qdrant.

Starts the server over stdio with scripts/mcp.sh (or connects to a running HTTP server with
--url), saves a few notes and a markdown file into a scratch collection, searches
them, and deletes the collection again, even when a step fails.

    uv run python scripts/mcp_smoke.py
    uv run python scripts/mcp_smoke.py --url http://127.0.0.1:9003/mcp
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import tempfile
import time
from pathlib import Path

from mcp import Client, StdioServerParameters

PROJECT_DIR = Path(__file__).resolve().parent.parent

NOTES = [
    "Services-BGE serves bge-m3 embeddings on port 9001 and the bge-reranker-v2-m3 reranker on port 9002.",
    "Бесшовные трубы из нержавеющей стали классифицируются в товарной позиции 7304.",
    "The team decided on 2026-11-11 to keep fp32 because fp16 was not faster on the M4 Max.",
]

GUIDE = """# Smoke guide

## Qdrant

Qdrant stores the memory collections on port 6333. Each collection has a dense and a sparse vector.

## Agents

Claude Code, Cursor, OpenClaw and Hermes connect to the MCP server over stdio or HTTP on port 9003.
"""

CHECKS = [
    ("which port is the reranker on", "9002"),
    ("нержавеющая труба без шва код", "7304"),
    ("why did we stay on fp32", "fp16"),
    ("where are memory collections stored", "6333"),
]


async def tool(client: Client, name: str, **arguments) -> dict:
    started = time.perf_counter()
    result = await client.call_tool(name, arguments)
    elapsed = (time.perf_counter() - started) * 1000
    if result.is_error:
        raise RuntimeError(f"{name} failed: {result.content[0].text}")
    print(f"  {name:<17} {elapsed:7.0f} ms")
    return result.structured_content


async def run(server, collection: str) -> int:
    failures = 0
    async with Client(server) as client:
        names = sorted(t.name for t in (await client.list_tools()).tools)
        print("tools:", ", ".join(names))
        try:
            saved = await tool(client, "remember", collection=collection, texts=NOTES, source="smoke-notes", tags=["smoke"])
            print("   ", json.dumps({k: saved[k] for k in ("passages", "embedded", "unchanged")}))
            again = await tool(client, "remember", collection=collection, texts=NOTES, source="smoke-notes", tags=["smoke"])
            if again["embedded"] != 0:
                print("FAIL: saving the same notes twice embedded them again")
                failures += 1

            with tempfile.TemporaryDirectory(dir=PROJECT_DIR, prefix=".mcp-smoke-") as folder:
                guide = Path(folder) / "guide.md"
                guide.write_text(GUIDE, encoding="utf-8")
                saved = await tool(client, "remember", collection=collection, path=str(guide))
                print("   ", json.dumps({k: saved[k] for k in ("sources", "passages")}))

            for query, expected in CHECKS:
                found = await tool(client, "search", collection=collection, query=query, top_k=3)
                top = found["results"][0] if found["results"] else {}
                ok = expected in top.get("text", "")
                failures += not ok
                print(f"    {'ok  ' if ok else 'FAIL'} {query!r} -> {top.get('score')} {top.get('source')} "
                      f"{top.get('text', '')[:70]!r} (reranked={found['reranked']})")

            listed = await tool(client, "list_collections")
            print("   ", [c for c in listed["collections"] if c["name"] == collection])
            sources = await tool(client, "list_sources", collection=collection)
            print("   ", sources["sources"])
            ranked = await tool(client, "rerank", query="reranker port", documents=NOTES, top_n=1)
            failures += ranked["results"][0]["index"] != 0
            vectors = await tool(client, "embed", texts=["hello"], sparse=True)
            failures += vectors["dimensions"] != 1024
        finally:
            dropped = await tool(client, "forget", collection=collection, everything=True)
            print("    dropped:", dropped)
    return failures


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--url", help="MCP server URL (default: start one over stdio)")
    parser.add_argument("--collection", default="mcp-smoke", help="scratch collection, deleted at the end")
    args = parser.parse_args()

    # The same launcher the agent configs in docs/AGENT-MEMORY.md use.
    server = args.url or StdioServerParameters(command=str(PROJECT_DIR / "scripts" / "mcp.sh"))
    print("server:", args.url or "stdio (scripts/mcp.sh)")
    failures = asyncio.run(run(server, args.collection))
    print("PASS" if not failures else f"FAIL ({failures})")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
