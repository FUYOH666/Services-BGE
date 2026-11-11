"""Settings for the MCP memory server (env prefix `MCP_`)."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_DIR = Path(__file__).resolve().parent.parent


class MCPSettings(BaseSettings):
    # --- Transport ---------------------------------------------------------
    # "stdio" for desktop agents that spawn the server; "http" serves
    # streamable HTTP on host:port + path.
    transport: Literal["stdio", "http"] = "stdio"
    host: str = "127.0.0.1"
    port: int = 9003
    path: str = "/mcp"
    # Required from HTTP clients when set (Bearer or X-API-Key). Mandatory when
    # host is not loopback.
    api_key: str | None = None
    log_level: str = "INFO"

    # --- Services-BGE and Qdrant -------------------------------------------
    embedding_url: str = "http://127.0.0.1:9001"
    reranker_url: str = "http://127.0.0.1:9002"
    # Same resolution order as the services themselves, so one .env serves all.
    embedding_api_key: str | None = Field(
        default=None,
        validation_alias=AliasChoices("MCP_EMBEDDING_API_KEY", "EMBEDDING_API_KEY", "SERVICE_API_KEY"),
    )
    reranker_api_key: str | None = Field(
        default=None,
        validation_alias=AliasChoices("MCP_RERANKER_API_KEY", "RERANKER_API_KEY", "SERVICE_API_KEY"),
    )
    qdrant_url: str = "http://127.0.0.1:6333"
    qdrant_api_key: str | None = None
    # Seconds per call to the services or Qdrant. (Not MCP_TIMEOUT: Claude Code
    # sets that variable for the servers it starts.)
    request_timeout: float = 120.0

    # --- Memory --------------------------------------------------------------
    # Every MCP collection is stored in Qdrant as <prefix><name>, so agents can
    # neither see nor drop collections created by other projects.
    collection_prefix: str = "mcp_"
    default_collection: str = "memory"
    # Passage size, estimated in model tokens (conservative), and overlap.
    chunk_tokens: int = 500
    chunk_overlap_tokens: int = 80
    # max_length sent to the embedding and rerank services (tokens per text).
    max_length: int = 1024
    # Texts per /v1/embeddings call.
    embed_batch_size: int = 32
    # Candidates per vector type before fusion, and fused candidates reranked.
    prefetch_limit: int = 50
    rerank_candidates: int = 20

    # --- remember(path=...) ------------------------------------------------------
    # Folders the server may read, separated by ":" ("~" allowed).
    allowed_paths: str = "~"
    file_extensions: str = ".md,.markdown,.mdx,.txt,.rst"
    max_file_bytes: int = 2_000_000
    max_files: int = 500
    # Passages per remember call.
    max_chunks: int = 5000

    # --- Trust tiers -------------------------------------------------------------
    # Files and folders (":"-separated, "~" allowed) whose passages are stored as
    # tier "canon" (curated). Once set, every other write (notes, other files) is
    # stored as tier "inbox" (unreviewed). Empty: no tiers, as before. The writer
    # cannot choose its tier; it follows from where the text came from.
    canon_paths: str = ""
    # Default for search(prefer_tier=...), e.g. "canon". Empty: plain ranking.
    prefer_tier: str = ""
    # With a preferred tier, scores of passages from other tiers are multiplied by this.
    tier_penalty: float = Field(default=0.5, ge=0.0, le=1.0)

    # --- Team canon (services-bge-mcp sync) -------------------------------------
    # A git repository of curated markdown that every agent reads. `sync` clones or
    # fast-forwards it (read-only), indexes it and writes a manifest to state_dir;
    # repo and dir are remembered there after the first run.
    canon_repo: str = ""  # owner/name, git URL or local path
    canon_dir: str = ""  # where the clone lives, e.g. ~/development/memory
    canon_collection: str = "canon"
    # Relative to canon_dir, ":"-separated; a folder means every .md file in it.
    canon_files: str = "CANON.md:STATE.md:ORG.md:AGENTS.md:repos.md:decisions"
    canon_inbox: str = "inbox"
    # Short passages: same recall as 500 tokens on the memory bench, 3x faster rerank.
    canon_chunk_tokens: int = 150
    canon_chunk_overlap_tokens: int = 20
    # Where sync records what it indexed (canon-sync.json).
    state_dir: str = "~/.services-bge"

    model_config = SettingsConfigDict(
        env_prefix="MCP_",
        # The project's .env, wherever the agent starts the process from.
        env_file=PROJECT_DIR / ".env",
        extra="ignore",
    )

    def allowed_roots(self) -> list[Path]:
        return _paths(self.allowed_paths)

    def canon_roots(self) -> list[Path]:
        return _paths(self.canon_paths)

    def extensions(self) -> set[str]:
        return {e.strip().lower() if e.strip().startswith(".") else "." + e.strip().lower()
                for e in self.file_extensions.split(",") if e.strip()}


def _paths(value: str) -> list[Path]:
    return [Path(raw.strip()).expanduser().resolve() for raw in value.split(os.pathsep) if raw.strip()]


def is_loopback(host: str) -> bool:
    return host in {"127.0.0.1", "localhost", "::1", "[::1]"}
