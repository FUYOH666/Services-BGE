"""MCP server: shared long-term memory and search for AI agents, backed by Services-BGE."""
from __future__ import annotations

import argparse
import logging
import secrets
import sys
from typing import Annotated, Literal

from mcp.server import MCPServer
from mcp.server.mcpserver import Context
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import ToolAnnotations
from pydantic import Field
from starlette.requests import Request
from starlette.responses import JSONResponse

from . import __version__
from .config import MCPSettings, is_loopback
from .memory import (
    CollectionsResult,
    ForgetResult,
    Memory,
    RememberResult,
    RerankResult,
    SearchResult,
    SourcesResult,
    VectorsResult,
)
from .services import ServiceError

logger = logging.getLogger("services_bge_mcp")

INSTRUCTIONS = """\
Long-term memory and search that runs locally on this Mac (Services-BGE: bge-m3 hybrid \
embeddings + bge-reranker-v2-m3, stored in Qdrant). Several agents share it.

- Before answering questions about the user's projects, notes, documents or earlier \
decisions, call `search`. Cite the `source` of passages you use.
- Call `remember` to save durable facts, decisions, summaries and documents (`texts`), \
or to index a file or folder of .md/.txt files (`path`). Never save passwords, API keys \
or other secrets.
- Collections are namespaces (default "{default}"). Use the same name in every agent to \
share memory; `list_collections` and `list_sources` show what exists.
- `forget` deletes permanently; ask the user first.
- A passage may carry a `tier`: `canon` is a curated file, `inbox` an unreviewed write \
by an agent. When they disagree, trust `canon` and say that the inbox passage conflicts.
"""

READ_ONLY = ToolAnnotations(read_only_hint=True, open_world_hint=False)

Collection = Annotated[
    str | None,
    Field(description="Memory namespace: letters, digits, '-' and '_'. Omit to use the shared default collection (normally 'memory'), which all agents see."),
]


def build_server(settings: MCPSettings | None = None, memory: Memory | None = None) -> MCPServer:
    settings = settings or MCPSettings()
    memory = memory or Memory(settings)
    default = settings.default_collection

    server = MCPServer(
        "services-bge",
        title="Services-BGE memory",
        description="Local long-term memory and hybrid search (bge-m3 + reranker + Qdrant)",
        instructions=INSTRUCTIONS.format(default=default),
        website_url="https://github.com/FUYOH666/Services-BGE",
        version=__version__,
        log_level=settings.log_level.upper(),
    )

    async def run(call):
        try:
            return await call
        except ServiceError as exc:
            raise ToolError(str(exc)) from exc

    @server.tool(
        title="Remember",
        annotations=ToolAnnotations(read_only_hint=False, destructive_hint=False, idempotent_hint=True, open_world_hint=False),
    )
    async def remember(
        ctx: Context,
        texts: Annotated[
            list[str] | None,
            Field(description="Things to save, one string per note, fact, decision or document. Long text is split into passages automatically."),
        ] = None,
        path: Annotated[
            str | None,
            Field(description="Instead of texts: a .md/.markdown/.txt/.rst file or a folder of them on this Mac (absolute or ~/...). Folders are read recursively; hidden folders are skipped. Reading a file again replaces its old passages."),
        ] = None,
        collection: Collection = None,
        source: Annotated[
            str | None,
            Field(description="Where this came from (e.g. 'meeting-2026-11-11', a URL, 'project-x/decisions'), shown in search results and used by forget. Default: the file path, or an id per text."),
        ] = None,
        tags: Annotated[list[str] | None, Field(description="Labels to filter search by later, e.g. ['project-x', 'decision'].")] = None,
        replace: Annotated[
            bool,
            Field(description="With texts and source: replace everything saved under that source instead of adding to it."),
        ] = False,
    ) -> RememberResult:
        """Save text or files to long-term memory so you or another agent can find them later with `search`.

        Saving the same text again does not create a duplicate. Returns how many
        passages were stored, newly embedded or already up to date. When the server
        has trust tiers on, notes are stored as tier 'inbox' (unreviewed); only files
        in the curated folders it is configured with become 'canon'.
        """

        async def progress(done: float, total: float | None, message: str | None) -> None:
            await ctx.report_progress(done, total, message)

        return await run(memory.remember(
            collection, texts=texts, path=path, source=source, tags=tags, replace=replace, progress=progress,
        ))

    @server.tool(title="Search memory", annotations=READ_ONLY)
    async def search(
        query: Annotated[str, Field(description="A question or keywords, in any language.")],
        collection: Collection = None,
        top_k: Annotated[int, Field(ge=1, le=50, description="How many passages to return.")] = 5,
        rerank: Annotated[bool, Field(description="Re-order candidates with the cross-encoder reranker (more accurate, ~0.2 s slower).")] = True,
        tags: Annotated[list[str] | None, Field(description="Only passages with at least one of these tags.")] = None,
        source: Annotated[str | None, Field(description="Only passages from this exact source.")] = None,
        min_score: Annotated[float | None, Field(ge=0, description="Drop passages scoring below this. With rerank, scores are 0-1 relevance; under ~0.1 is rarely useful.")] = None,
        tiers: Annotated[list[str] | None, Field(description="Only passages from these trust tiers, e.g. ['canon'] for curated files only.")] = None,
        prefer_tier: Annotated[
            str | None,
            Field(description="Rank this trust tier first when relevance is close, e.g. 'canon': scores of passages from other tiers are multiplied by MCP_TIER_PENALTY (0.5). Default: the server's MCP_PREFER_TIER (normally off)."),
        ] = None,
    ) -> SearchResult:
        """Find the passages in long-term memory most relevant to a query.

        Hybrid search (bge-m3 dense meaning + sparse keyword match, fused in Qdrant),
        then reranked. Results come best first with `source` to cite, `score`, and
        `tier` ('canon' curated / 'inbox' unreviewed) when the server uses trust tiers.
        """
        return await run(memory.search(
            collection, query, top_k=top_k, rerank=rerank, tags=tags, source=source, min_score=min_score,
            tiers=tiers, prefer_tier=prefer_tier,
        ))

    @server.tool(title="List memory collections", annotations=READ_ONLY)
    async def list_collections() -> CollectionsResult:
        """List memory collections with the number of passages in each and the embedding model that built them."""
        return await run(memory.list_collections())

    @server.tool(title="List sources", annotations=READ_ONLY)
    async def list_sources(
        collection: Collection = None,
        limit: Annotated[int, Field(ge=1, le=1000, description="Maximum sources to list.")] = 100,
    ) -> SourcesResult:
        """List the sources (files, notes, URLs) saved in a collection with their passage counts."""
        return await run(memory.list_sources(collection, limit=limit))

    @server.tool(
        title="Forget",
        annotations=ToolAnnotations(read_only_hint=False, destructive_hint=True, idempotent_hint=True, open_world_hint=False),
    )
    async def forget(
        collection: Annotated[str, Field(description="The collection to delete from.")],
        source: Annotated[str | None, Field(description="Delete every passage saved from this source (exactly as list_sources shows it).")] = None,
        everything: Annotated[bool, Field(description="Delete the whole collection. Cannot be undone.")] = False,
    ) -> ForgetResult:
        """Permanently delete memories: one source, or the whole collection with everything=true.

        This cannot be undone. Ask the user before calling it unless they asked you to forget something.
        """
        return await run(memory.forget(collection, source=source, everything=everything))

    @server.tool(title="Rerank documents", annotations=READ_ONLY)
    async def rerank(
        query: Annotated[str, Field(description="The question or need.")],
        documents: Annotated[list[str], Field(min_length=1, max_length=1000, description="Candidate texts to order.")],
        top_n: Annotated[int | None, Field(ge=1, description="Return only the best top_n.")] = None,
    ) -> RerankResult:
        """Order your own candidate texts by relevance to a query with the local cross-encoder.

        Use it when you already have candidates (snippets, search hits from elsewhere).
        Returns indices into `documents`, best first, with 0-1 relevance scores.
        """
        return await run(memory.rerank(query, documents, top_n=top_n))

    @server.tool(title="Embed texts", annotations=READ_ONLY)
    async def embed(
        texts: Annotated[list[str], Field(min_length=1, max_length=8, description="Texts to embed (at most 8).")],
        input_type: Annotated[Literal["query", "document"], Field(description="'query' for search queries, 'document' for indexed text.")] = "document",
        sparse: Annotated[bool, Field(description="Also return bge-m3 sparse keyword weights in Qdrant format.")] = False,
    ) -> VectorsResult:
        """Turn texts into 1024-d bge-m3 vectors (and optional sparse weights).

        Only needed to build your own index; to save or find memories use `remember` and `search`.
        Each vector is several thousand tokens of output.
        """
        return await run(memory.embed(texts, input_type=input_type, sparse=sparse))

    @server.custom_route("/healthz", methods=["GET"])
    async def healthz(_: Request) -> JSONResponse:
        return JSONResponse({"status": "ok", "server": "services-bge-mcp", "version": __version__})

    return server


class _RequireApiKey:
    """ASGI middleware: every HTTP request except /healthz needs the MCP API key."""

    def __init__(self, app, api_key: str) -> None:
        self.app = app
        self.api_key = api_key

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http" and scope.get("path") != "/healthz":
            headers = {k.decode("latin-1").lower(): v.decode("latin-1") for k, v in scope.get("headers", [])}
            supplied = headers.get("x-api-key")
            auth = headers.get("authorization", "")
            if auth.lower().startswith("bearer "):
                supplied = auth[7:].strip()
            if not supplied or not secrets.compare_digest(supplied, self.api_key):
                response = JSONResponse(
                    {"error": "Invalid or missing API key"}, status_code=401, headers={"WWW-Authenticate": "Bearer"}
                )
                await response(scope, receive, send)
                return
        await self.app(scope, receive, send)


def http_app(server: MCPServer, settings: MCPSettings):
    app = server.streamable_http_app(
        streamable_http_path=settings.path,
        stateless_http=True,
        host=settings.host,
    )
    return _RequireApiKey(app, settings.api_key) if settings.api_key else app


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="services-bge-mcp",
        description="MCP server for Services-BGE: shared long-term memory and search for AI agents.",
    )
    parser.add_argument("--transport", choices=["stdio", "http"], help="stdio (default, for agents that start the server) or http")
    parser.add_argument("--host", help="HTTP bind address (default 127.0.0.1)")
    parser.add_argument("--port", type=int, help="HTTP port (default 9003)")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    args = parser.parse_args(argv)

    overrides = {k: v for k, v in {"transport": args.transport, "host": args.host, "port": args.port}.items() if v is not None}
    settings = MCPSettings(**overrides)

    # Logs go to stderr: on stdio, stdout carries the protocol.
    logging.basicConfig(level=settings.log_level.upper(), stream=sys.stderr, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)

    server = build_server(settings)
    if settings.transport == "stdio":
        server.run("stdio")
        return

    if not is_loopback(settings.host) and not settings.api_key:
        parser.error(f"MCP_HOST={settings.host} exposes memory beyond this Mac; set MCP_API_KEY first.")

    import uvicorn

    logger.info("Serving MCP over streamable HTTP on http://%s:%d%s", settings.host, settings.port, settings.path)
    uvicorn.run(http_app(server, settings), host=settings.host, port=settings.port, log_level=settings.log_level.lower())


if __name__ == "__main__":
    main()
