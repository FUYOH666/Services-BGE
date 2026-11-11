"""The team canon: a read-only clone of curated markdown, synced into one collection.

`services-bge-mcp sync` clones the canon repository or fast-forwards its clone (it never
commits or pushes there), indexes the read-first files and decisions as tier `canon`
and the inbox as tier `inbox` into MCP_CANON_COLLECTION, drops passages of files that
are gone, and writes a manifest (MCP_STATE_DIR/canon-sync.json) saying what it indexed,
from which commit and when.

The server reads that manifest to know which files are canon: they rank first in the
canon collection, every agent write is tier `inbox`, and when the embedding service or
Qdrant is down, search falls back to a keyword search over those files.
"""
from __future__ import annotations

import os
import re
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from shutil import which
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, ValidationError

from .config import MCPSettings
from .services import ServiceError

if TYPE_CHECKING:
    from .memory import Memory

MANIFEST_NAME = "canon-sync.json"
_FRONTMATTER_LINES = 40
_SKIP_REASON = {
    "curated": "already in canon or decisions/",
    "accepted": "already in canon or decisions/",
    "rejected": "rejected by the curator",
}
_MARKDOWN = (".md", ".markdown")
_OWNER_NAME = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")
_GIT_TIMEOUT = 300


class ManifestFile(BaseModel):
    path: str  # relative to the clone
    tier: Literal["canon", "inbox"]
    passages: int


class SkippedFile(BaseModel):
    path: str  # relative to the clone
    status: str | None  # frontmatter status, None when the file has none


class Manifest(BaseModel):
    repo: str | None = None
    dir: str
    commit: str | None = None
    commit_date: str | None = None
    synced_at: str
    collection: str
    chunk_tokens: int
    chunk_overlap_tokens: int
    files: list[ManifestFile]
    inbox_statuses: list[str] = []
    skipped: list[SkippedFile] = []
    embedded: int = 0
    unchanged: int = 0
    removed: list[str] = []
    warnings: list[str] = []

    def root(self) -> Path:
        return Path(self.dir).expanduser()

    def paths(self, tier: str) -> list[Path]:
        return [self.root() / f.path for f in self.files if f.tier == tier]

    def passages(self, tier: str) -> int:
        return sum(f.passages for f in self.files if f.tier == tier)


def display(path: Path) -> str:
    home = Path.home()
    try:
        return "~/" + str(path.relative_to(home))
    except ValueError:
        return str(path)


def manifest_path(settings: MCPSettings) -> Path:
    return Path(settings.state_dir).expanduser() / MANIFEST_NAME


def load_manifest(settings: MCPSettings) -> Manifest | None:
    path = manifest_path(settings)
    try:
        return Manifest.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, ValidationError):
        return None


def save_manifest(settings: MCPSettings, manifest: Manifest) -> Path:
    path = manifest_path(settings)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(manifest.model_dump_json(indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)
    return path


def markdown_files(path: Path) -> list[Path]:
    """A markdown file, or every markdown file under a folder (hidden folders skipped)."""
    if path.is_file():
        return [path] if path.suffix.lower() in _MARKDOWN else []
    if not path.is_dir():
        return []
    found = []
    for file in sorted(path.rglob("*")):
        rel = file.relative_to(path)
        if file.is_file() and file.suffix.lower() in _MARKDOWN and not any(p.startswith(".") for p in rel.parts):
            found.append(file)
    return found


def frontmatter_status(path: Path) -> str | None:
    """`status:` from a markdown file's YAML frontmatter, lower case; None without one."""
    try:
        with path.open(encoding="utf-8", errors="replace") as fh:
            lines = [fh.readline() for _ in range(_FRONTMATTER_LINES)]
    except OSError:
        return None
    if not lines or lines[0].strip() != "---":
        return None
    for line in lines[1:]:
        if line.strip() == "---":
            break
        key, _, value = line.partition(":")
        if key.strip().lower() == "status" and value.strip():
            return value.strip().strip("'\"").lower()
    return None


def skipped_summary(skipped: list[SkippedFile]) -> str | None:
    """Plain words on inbox files sync left out, e.g. '6 curated inbox files skipped: already in canon …'."""
    if not skipped:
        return None
    groups: dict[str | None, list[str]] = {}
    for item in skipped:
        groups.setdefault(item.status, []).append(Path(item.path).name)
    parts = []
    for status, names in groups.items():
        files = "file" if len(names) == 1 else "files"
        if status:
            why = _SKIP_REASON.get(status, f"status '{status}' is not an open proposal")
            part = f"{len(names)} {status} inbox {files} skipped: {why}"
        else:
            part = f"{len(names)} inbox {files} without a status skipped: not a proposal"
        if len(names) <= 3:
            part += f" ({', '.join(names)})"
        parts.append(part)
    return "; ".join(parts) + "."


def _split(value: str) -> list[str]:
    return [item.strip() for item in value.split(os.pathsep) if item.strip()]


def select_files(root: Path, canon: list[str], inbox: list[str]) -> tuple[list[Path], list[Path]]:
    """Canon and inbox files under root; a file listed in both is canon."""
    def expand(patterns: list[str]) -> list[Path]:
        out: list[Path] = []
        for pattern in patterns:
            target = root / pattern
            matches = markdown_files(target) if target.exists() else sorted(
                p for p in root.glob(pattern) if p.is_file() and p.suffix.lower() in _MARKDOWN
            )
            out.extend(m.resolve() for m in matches if m.resolve() not in out)
        return out

    canon_files = expand(canon)
    inbox_files = [p for p in expand(inbox) if p not in canon_files]
    return canon_files, inbox_files


def _run(command: list[str], cwd: Path | None = None) -> subprocess.CompletedProcess:
    env = {**os.environ, "GIT_TERMINAL_PROMPT": "0"}
    return subprocess.run(command, cwd=cwd, capture_output=True, text=True, timeout=_GIT_TIMEOUT, env=env)


def _git(*args: str, cwd: Path | None = None) -> subprocess.CompletedProcess:
    return _run(["git", *args], cwd=cwd)


def git_sync(repo: str | None, root: Path) -> tuple[str | None, str | None, list[str]]:
    """Clone or fast-forward the canon clone, read-only. Returns (commit, commit date, warnings)."""
    warnings: list[str] = []
    if not root.exists():
        if not repo:
            raise ServiceError(f"{display(root)} does not exist; pass --repo to clone the canon there.")
        root.parent.mkdir(parents=True, exist_ok=True)
        local = Path(repo).expanduser()
        if local.exists():
            result = _git("clone", "--quiet", str(local), str(root))
        elif _OWNER_NAME.match(repo) and which("gh"):
            result = _run(["gh", "repo", "clone", repo, str(root), "--", "--quiet"])  # uses gh's auth and protocol
        elif _OWNER_NAME.match(repo):
            result = _git("clone", "--quiet", f"https://github.com/{repo}.git", str(root))
        else:
            result = _git("clone", "--quiet", repo, str(root))
        if result.returncode != 0:
            raise ServiceError(f"Could not clone {repo}: {result.stderr.strip()[:300]}")
    elif (root / ".git").exists():
        if _git("status", "--porcelain", cwd=root).stdout.strip():
            warnings.append(f"{display(root)} has local changes; not pulled, indexed as it is.")
        else:
            pull = _git("pull", "--ff-only", "--quiet", cwd=root)
            if pull.returncode != 0:
                warnings.append(f"git pull failed in {display(root)}, indexed the current checkout: {pull.stderr.strip()[:200]}")
    else:
        warnings.append(f"{display(root)} is not a git clone; indexed as it is, without a commit.")
        return None, None, warnings
    head = _git("rev-parse", "HEAD", cwd=root)
    date = _git("log", "-1", "--format=%cI", cwd=root)
    commit = head.stdout.strip() if head.returncode == 0 else None
    commit_date = date.stdout.strip() if date.returncode == 0 and date.stdout.strip() else None
    return commit, commit_date, warnings


async def sync(memory: Memory, *, repo: str | None = None, directory: str | None = None) -> Manifest:
    """Bring the canon collection in line with the clone; idempotent."""
    settings = memory.settings
    previous = load_manifest(settings)
    repo = repo or (previous.repo if previous else None) or settings.canon_repo or None
    raw_dir = directory or (previous.dir if previous else None) or settings.canon_dir
    if not raw_dir:
        raise ServiceError("Where should the canon clone live? Pass --dir (e.g. ~/development/memory) the first time.")
    root = Path(raw_dir).expanduser().resolve()
    commit, commit_date, warnings = git_sync(repo, root)

    canon_files, inbox_candidates = select_files(root, _split(settings.canon_files), _split(settings.canon_inbox))
    if not canon_files:
        raise ServiceError(f"None of {settings.canon_files!r} exist in {display(root)} (MCP_CANON_FILES).")
    # Only open proposals: curated, accepted or rejected ones already live in canon or decisions/.
    statuses = sorted({s.strip().lower() for s in settings.canon_inbox_statuses.split(",") if s.strip()})
    inbox_files: list[Path] = []
    skipped: list[SkippedFile] = []
    for path in inbox_candidates:
        status = frontmatter_status(path)
        if "*" in statuses or (status is not None and status in statuses):
            inbox_files.append(path)
        else:
            skipped.append(SkippedFile(path=path.relative_to(root).as_posix(), status=status))

    # This run decides tiers from the files just selected, not from an older manifest,
    # and indexes with the short canon passages.
    memory.use_manifest = False
    settings.canon_paths = os.pathsep.join(str(p) for p in canon_files)
    settings.chunk_tokens = settings.canon_chunk_tokens
    settings.chunk_overlap_tokens = settings.canon_chunk_overlap_tokens
    settings.allowed_paths = os.pathsep.join(filter(None, [settings.allowed_paths, str(root)]))
    collection = settings.canon_collection

    files: list[ManifestFile] = []
    kept: set[str] = set()
    embedded = unchanged = 0
    for path in canon_files + inbox_files:
        result = await memory.remember(collection, path=str(path))
        kept.update(result.sources)
        embedded += result.embedded
        unchanged += result.unchanged
        tier = "canon" if path in canon_files else "inbox"
        files.append(ManifestFile(path=path.relative_to(root).as_posix(), tier=tier, passages=result.passages))

    # Passages of files that were deleted, renamed or are no longer selected.
    removed: list[str] = []
    prefix = display(root) + "/"
    listed = await memory.list_sources(collection, limit=1000)
    for item in listed.sources:
        if item.source.startswith(prefix) and item.source not in kept:
            await memory.forget(collection, source=item.source)
            removed.append(item.source[len(prefix):])

    manifest = Manifest(
        repo=repo, dir=display(root), commit=commit, commit_date=commit_date,
        synced_at=datetime.now(UTC).isoformat(timespec="seconds"), collection=collection,
        chunk_tokens=settings.chunk_tokens, chunk_overlap_tokens=settings.chunk_overlap_tokens,
        files=files, inbox_statuses=statuses, skipped=skipped,
        embedded=embedded, unchanged=unchanged, removed=removed, warnings=warnings,
    )
    save_manifest(settings, manifest)
    return manifest


def describe_sync(manifest: Manifest, settings: MCPSettings) -> str:
    commit = manifest.commit[:7] if manifest.commit else "no commit"
    canon_n = sum(1 for f in manifest.files if f.tier == "canon")
    inbox_n = sum(1 for f in manifest.files if f.tier == "inbox")
    lines = [
        f"Synced {manifest.repo or manifest.dir} at {commit} into collection '{manifest.collection}':",
        f"  {canon_n} curated files (tier canon, {manifest.passages('canon')} passages), "
        f"{inbox_n} inbox files (tier inbox, {manifest.passages('inbox')} passages).",
        f"  Newly embedded {manifest.embedded}, unchanged {manifest.unchanged}"
        + (f", removed files: {', '.join(manifest.removed)}" if manifest.removed else ", nothing removed") + ".",
        f"  Clone: {manifest.dir} (read-only). Record: {display(manifest_path(settings))}.",
    ]
    skipped = skipped_summary(manifest.skipped)
    if skipped:
        lines.insert(2, f"  Not indexed: {skipped}")
    lines += [f"  Warning: {w}" for w in manifest.warnings]
    return "\n".join(lines)
