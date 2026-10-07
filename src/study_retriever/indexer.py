from __future__ import annotations

import fnmatch
import hashlib
import logging
import os
import shutil
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from .catalog import Catalog
from .chunking import chunk_document, source_id_for
from .config import SUPPORTED_SUFFIXES, AppConfig
from .locking import InterProcessLock
from .parsers import is_supported, parse_document
from .vector_store import VectorStore

log = logging.getLogger(__name__)


@dataclass(slots=True)
class SyncResult:
    indexed: int = 0
    unchanged: int = 0
    deleted: int = 0
    errors: int = 0
    chunks_added: int = 0

    def as_dict(self) -> dict:
        return {
            "indexed": self.indexed,
            "unchanged": self.unchanged,
            "deleted": self.deleted,
            "errors": self.errors,
            "chunks_added": self.chunks_added,
        }


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


class Indexer:
    def __init__(self, catalog: Catalog, vectors: VectorStore, config: AppConfig, lock: threading.RLock | None = None):
        self.catalog = catalog
        self.vectors = vectors
        self.config = config
        self.lock = lock or threading.RLock()
        self.process_lock = InterProcessLock(catalog.db_path.parent / "index-writer.lock", timeout=600.0)

    def _excluded(self, path: Path, root: Path) -> bool:
        """True when path sits inside an excluded directory below its root (watcher events bypass the walk)."""
        excluded = {name.casefold() for name in self.config.excluded_dir_names}
        try:
            parts = path.relative_to(root).parts[:-1]
        except ValueError:
            parts = path.parts[:-1]
        if any(part.casefold() in excluded for part in parts):
            return True
        if not self.config.exclude_globs:
            return False
        try:
            rel = path.relative_to(root).as_posix()
        except ValueError:
            rel = path.name
        return any(fnmatch.fnmatch(rel.casefold(), g.casefold()) for g in self.config.exclude_globs)

    def _iter_root_files(self, root: Path):
        root = root.expanduser().resolve()
        if root.is_file():
            if is_supported(root) and root.stat().st_size <= self.config.max_file_bytes:
                yield root
            return
        if not root.is_dir():
            return
        excluded = {name.casefold() for name in self.config.excluded_dir_names}
        for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
            dirnames[:] = [d for d in dirnames if d.casefold() not in excluded]
            base = Path(dirpath)
            for name in filenames:
                path = base / name
                try:
                    if (
                        path.suffix.lower() in SUPPORTED_SUFFIXES
                        and not path.is_symlink()
                        and path.stat().st_size <= self.config.max_file_bytes
                        and not self._excluded(path, root)
                    ):
                        yield path
                except OSError:
                    continue

    @staticmethod
    def _relative(path: Path, root: Path) -> str:
        try:
            return str(path.resolve().relative_to(root.resolve()))
        except ValueError:
            return path.name

    def index_file(self, path: Path, root: Path, *, force: bool = False) -> tuple[str, int]:
        # One writer across ChatGPT MCP, CLI, installer, and watcher processes.
        # This keeps SQLite metadata and the separately persisted vector index atomic as a pair.
        with self.process_lock:
            return self._index_file_locked(path, root, force=force)

    _TRANSIENT_ERRORS = ("PermissionError", "OSError", "TimeoutError", "RuntimeError: Source changed")

    @staticmethod
    def _same_stat(existing, stat) -> bool:
        return int(existing["size_bytes"]) == stat.st_size and int(existing["mtime_ns"]) == stat.st_mtime_ns

    def _is_permanent_error(self, existing) -> bool:
        row = self.catalog._conn().execute("SELECT error FROM sources WHERE source_id=?", (existing["source_id"],)).fetchone()
        message = (row[0] if row else "") or ""
        return not message.startswith(self._TRANSIENT_ERRORS)

    def free_disk_mb(self) -> int:
        return int(shutil.disk_usage(self.catalog.db_path.parent).free // 2**20)

    def _disk_ok(self) -> bool:
        free = self.free_disk_mb()
        if free < self.config.min_free_disk_mb:
            log.error("only %d MB free on the data drive (minimum %d); refusing to write the index", free, self.config.min_free_disk_mb)
            return False
        return True

    def _index_file_locked(self, path: Path, root: Path, *, force: bool = False) -> tuple[str, int]:
        path = path.expanduser().resolve()
        root = root.expanduser().resolve()
        if not path.exists() or not path.is_file():
            existing = self.catalog.source_by_path(path)
            if existing:
                with self.lock:
                    keys = self.catalog.vector_keys_for_source(existing["source_id"])
                    self.vectors.delete_keys(keys)
                    self.catalog.delete_source(existing["source_id"])
                return "deleted", 0
            return "missing", 0
        if not is_supported(path):
            return "unsupported", 0
        if self._excluded(path, root):
            return "excluded", 0
        if not self._disk_ok():
            return "disk_full", 0
        if root.is_dir() and not path.is_relative_to(root):
            return "excluded", 0  # junction/symlink target outside the root: never index across the boundary
        stat = path.stat()
        if stat.st_size > self.config.max_file_bytes:
            return "too_large", 0

        existing = self.catalog.source_stat(path)
        if not force and existing and existing["status"] == "ready" and int(existing["size_bytes"]) == stat.st_size and int(existing["mtime_ns"]) == stat.st_mtime_ns:
            return "unchanged", 0

        if not force and existing and existing["status"] == "error" and self._same_stat(existing, stat) and self._is_permanent_error(existing):
            return "error", 0  # recoverable-vs-not (ch. 14): a malformed file stays failed until it changes
        digest = sha256_file(path)
        if not force and existing and existing["status"] == "ready" and existing["sha256"] == digest:
            self.catalog.update_source_stat(path, size_bytes=stat.st_size, mtime_ns=stat.st_mtime_ns)
            return "unchanged", 0

        sid = source_id_for(path)
        old_keys = self.catalog.vector_keys_for_source(sid)
        try:
            # Editors often emit several write events while replacing a file. Retrying here avoids indexing a half-written ZIP/PDF.
            last_exc: Exception | None = None
            doc = None
            for delay in (0.0, 0.35, 0.8):
                if delay:
                    time.sleep(delay)
                try:
                    doc = parse_document(path)
                    last_exc = None
                    break
                except Exception as exc:
                    # Office/PDF writers may expose a temporary but syntactically invalid
                    # container during atomic save (for example zipfile.BadZipFile). Retry
                    # all ordinary parser failures before marking the changed source bad.
                    last_exc = exc
            if doc is None:
                assert last_exc is not None
                raise last_exc
            post_stat = path.stat()
            if post_stat.st_size != stat.st_size or post_stat.st_mtime_ns != stat.st_mtime_ns:
                raise RuntimeError("Source changed while it was being indexed; waiting for the next stable filesystem event")
            chunks = chunk_document(path, digest, doc, self.config.chunk_chars, self.config.overlap_chars)
            new_keys = [c.vector_key for c in chunks]
            # Embedding is the slow step: do it before taking the in-process lock so searches are not blocked.
            new_set, old_set = set(new_keys), set(old_keys)
            # Delta sync: chunks whose content-derived key already exists keep their vectors. force re-embeds all.
            fresh = list(chunks) if force else [c for c in chunks if c.vector_key not in old_set]
            vectors = self.vectors.embed(fresh)
            stale = [k for k in old_keys if k not in new_set]
            with self.lock:
                try:
                    self.vectors.add(fresh, vectors, remove=stale)
                    self.catalog.replace_source(
                        source_id=sid,
                        path=path,
                        root=root,
                        relative_path=self._relative(path, root),
                        title=doc.title,
                        suffix=path.suffix.lower(),
                        size_bytes=stat.st_size,
                        mtime_ns=stat.st_mtime_ns,
                        sha256=digest,
                        metadata=doc.metadata,
                        chunks=chunks,
                    )
                except Exception:
                    # Remove only vectors that did not exist before this attempt. If force=True on unchanged
                    # content, new_keys can equal old_keys and those vectors still belong to the old catalog.
                    self.vectors.delete_keys([key for key in new_keys if key not in old_keys])
                    raise
            return "indexed", len(chunks)
        except Exception as exc:
            log.exception("Failed to index %s", path)
            # The source changed but cannot be parsed reliably. Fail closed: remove the
            # last-good chunks/vectors instead of serving content that no longer matches disk.
            with self.lock:
                self.vectors.delete_keys(old_keys)
                self.catalog.delete_source(sid)
                self.catalog.mark_error(
                    source_id=sid, path=path, root=root, error=f"{type(exc).__name__}: {exc}",
                    size_bytes=stat.st_size, mtime_ns=stat.st_mtime_ns,
                )
            return "error", 0

    def sync_root(self, root: Path, *, force: bool = False) -> SyncResult:
        root = root.expanduser().resolve()
        self.catalog.add_root(root)
        result = SyncResult()
        if not root.exists():
            # The source of truth disappeared while the service was offline. Fail
            # closed by removing stale derived entries, but retain the configured
            # root so the watcher/startup sync can recover if it is recreated.
            for source in self.catalog.sources_under_root(root):
                with self.process_lock, self.lock:
                    self.vectors.delete_keys(self.catalog.vector_keys_for_source(source["source_id"]))
                    self.catalog.delete_source(source["source_id"])
                result.deleted += 1
            return result
        seen: set[str] = set()
        for path in self._iter_root_files(root):
            seen.add(str(path.resolve()))
            status, chunks = self.index_file(path, root, force=force)
            if status == "indexed":
                result.indexed += 1
                result.chunks_added += chunks
            elif status == "unchanged":
                result.unchanged += 1
            elif status in {"error", "disk_full"}:
                result.errors += 1

        for source in self.catalog.sources_under_root(root):
            if source["path"] not in seen:
                with self.process_lock, self.lock:
                    self.vectors.delete_keys(self.catalog.vector_keys_for_source(source["source_id"]))
                    self.catalog.delete_source(source["source_id"])
                result.deleted += 1
        return result

    def sync_all(self, *, force: bool = False) -> dict:
        aggregate = SyncResult()
        per_root: dict[str, dict] = {}
        for root in self.catalog.roots():
            try:
                r = self.sync_root(root, force=force)
                per_root[str(root)] = r.as_dict()
                aggregate.indexed += r.indexed
                aggregate.unchanged += r.unchanged
                aggregate.deleted += r.deleted
                aggregate.errors += r.errors
                aggregate.chunks_added += r.chunks_added
            except Exception as exc:
                per_root[str(root)] = {"error": f"{type(exc).__name__}: {exc}"}
                aggregate.errors += 1
        return {"total": aggregate.as_dict(), "roots": per_root}

    def repair_vectors(self) -> dict:
        """Anti-entropy: diff catalog and vector-index key sets, drop orphans, re-embed only what is missing."""
        with self.process_lock, self.lock:
            catalog_keys = self.catalog.all_vector_keys()
            index_keys = self.vectors.keys()
            orphans = sorted(index_keys - catalog_keys)
            missing = catalog_keys - index_keys
            if orphans:
                self.vectors.delete_keys(orphans)
            restored = 0
            if missing:
                from .models import Chunk

                for batch in self.catalog.iter_chunks(256, vector_keys=missing):
                    chunks = [
                        Chunk(
                            chunk_id=row["chunk_id"], source_id=row["source_id"], vector_key=int(row["vector_key"]),
                            ordinal=int(row["ordinal"]), text=row["text"], embedding_text=row["embedding_text"],
                            title=row["title"], section=row["section"], path=row["path"], source_uri=row["source_uri"],
                            locator=row["locator"],
                        )
                        for row in batch
                    ]
                    self.vectors.add(chunks, self.vectors.embed(chunks), save=False)
                    restored += len(chunks)
                self.vectors.flush()
            return {"orphans_removed": len(orphans), "vectors_restored": restored}

    def rebuild_vectors(self) -> dict:
        if not self._disk_ok():
            raise RuntimeError("Not enough free disk space to rebuild the vector index")
        with self.process_lock, self.lock:
            self.vectors.clear()
            count = 0
            for batch in self.catalog.iter_chunks(256):
                from .models import Chunk
                chunks = [
                    Chunk(
                        chunk_id=row["chunk_id"], source_id=row["source_id"], vector_key=int(row["vector_key"]),
                        ordinal=int(row["ordinal"]), text=row["text"], embedding_text=row["embedding_text"],
                        title=row["title"], section=row["section"], path=row["path"], source_uri=row["source_uri"],
                        locator=row["locator"],
                    )
                    for row in batch
                ]
                self.vectors.add(chunks, self.vectors.embed(chunks), save=False)
                count += len(chunks)
            self.vectors.flush()
            return {"vectors": count, "backend": self.vectors.description}
