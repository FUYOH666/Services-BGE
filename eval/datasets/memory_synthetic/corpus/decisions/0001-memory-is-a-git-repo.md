# 0001 — Shared memory is a git repository

Date: 2026-10-02. Status: accepted.

The agents run in different tools and share no runtime. The one place all of them can read is the `kestrel/ops` repository, so it holds the canon. A vector index over it is a cache that is rebuilt from git; it is never the source of truth.
