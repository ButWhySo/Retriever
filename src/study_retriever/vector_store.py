from __future__ import annotations

import hashlib
import logging
import math
import os
import re
import sqlite3
import threading
from typing import Sequence, TypeAlias

import numpy as np

from .config import AppConfig, Paths
from .locking import InterProcessLock
from .models import Chunk

log = logging.getLogger(__name__)


def _validate_embedding_matrix(vectors: np.ndarray, rows: int, dim: int, context: str) -> np.ndarray:
    """Reject malformed/NaN/zero vectors before they can poison the persistent index."""
    vectors = np.asarray(vectors, dtype=np.float32)
    expected = (rows, dim)
    if vectors.shape != expected:
        raise RuntimeError(f"{context} embedding shape {vectors.shape} != {expected}")
    if not np.isfinite(vectors).all():
        raise RuntimeError(f"{context} embeddings contain NaN or infinite values")
    norms = np.linalg.norm(vectors, axis=1)
    if np.any(norms <= 1e-12):
        raise RuntimeError(f"{context} embeddings contain zero-length vectors")
    return vectors


def _validate_query_vector(vector: np.ndarray, dim: int) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float32).reshape(-1)
    if vector.shape != (dim,):
        raise RuntimeError(f"Query embedding shape {vector.shape} != {(dim,)}")
    if not np.isfinite(vector).all():
        raise RuntimeError("Query embedding contains NaN or infinite values")
    if float(np.linalg.norm(vector)) <= 1e-12:
        raise RuntimeError("Query embedding is a zero-length vector")
    return vector


class FastEmbedUSearchStore:
    """FastEmbed ONNX embeddings + USearch HNSW persisted to one local index file.

    Writers are serialized across processes. Readers detect atomic index-file
    replacement and reload, so a CLI sync and a ChatGPT MCP process cannot silently
    diverge or overwrite each other's vector updates.
    """

    def __init__(self, paths: Paths, config: AppConfig):
        from fastembed import TextEmbedding
        from usearch.index import Index

        self.paths = paths
        self.config = config
        self._Index = Index
        self._lock = threading.RLock()
        self._process_lock = InterProcessLock(paths.home / "vector-index.lock")
        # Keep each Codex/Desktop worker light; ONNX thread pools multiply per MCP process.
        threads = min(2, max(1, (os.cpu_count() or 2) - 1))
        self.model = TextEmbedding(
            model_name=config.model_name,
            cache_dir=str(paths.model_cache),
            threads=threads,
        )
        self.dim = int(self.model.embedding_size)
        slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", config.model_name).strip("_")
        self.index_path = paths.home / f"vectors-{slug}-{self.dim}.usearch"
        self._disk_signature: tuple[int, int] | None = None
        with self._process_lock, self._lock:
            self.index = self._load_index_from_disk()

    def _new_index(self):
        return self._Index(
            ndim=self.dim,
            metric="cos",
            dtype="bf16",
            connectivity=16,
            expansion_add=128,
            expansion_search=64,
        )

    def _signature(self) -> tuple[int, int] | None:
        try:
            stat = self.index_path.stat()
            return stat.st_mtime_ns, stat.st_size
        except FileNotFoundError:
            return None

    def _load_index_from_disk(self):
        signature = self._signature()
        if signature and signature[1] > 0:
            index = self._Index.restore(str(self.index_path), view=False)
            if int(index.ndim) != self.dim:
                raise RuntimeError(
                    f"Vector index dimension {index.ndim} does not match model dimension {self.dim}. Run rebuild-vectors."
                )
        else:
            index = self._new_index()
        self._disk_signature = signature
        return index

    def _refresh_if_changed(self) -> None:
        signature = self._signature()
        if signature != self._disk_signature:
            self.index = self._load_index_from_disk()

    def _save(self) -> None:
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.index_path.with_suffix(self.index_path.suffix + ".tmp")
        try:
            self.index.save(str(tmp))
            os.replace(tmp, self.index_path)
        finally:
            if tmp.exists():
                tmp.unlink(missing_ok=True)
        self._disk_signature = self._signature()

    def embed(self, chunks: Sequence[Chunk]) -> np.ndarray:
        """Embed passages without touching shared state, so callers can do this outside any lock."""
        if not chunks:
            return np.empty((0, self.dim), dtype=np.float32)
        texts = [c.embedding_text for c in chunks]
        return _validate_embedding_matrix(
            np.asarray(
                list(self.model.passage_embed(texts, batch_size=self.config.embedding_batch_size)),
                dtype=np.float32,
            ),
            len(chunks),
            self.dim,
            "Passage",
        )

    def add(self, chunks: Sequence[Chunk], vectors: np.ndarray, *, remove: Sequence[int] = (), save: bool = True) -> None:
        """Insert precomputed vectors and drop stale keys, persisting the index once."""
        if not len(chunks) and not remove:
            return
        keys = np.asarray([c.vector_key for c in chunks], dtype=np.uint64)
        with self._process_lock, self._lock:
            self._refresh_if_changed()
            for key in [int(k) for k in keys if self.index.contains(int(k))]:
                self.index.remove(key)
            if len(keys):
                self.index.add(keys, vectors, threads=0)
            keep = {int(k) for k in keys}
            for key in remove:
                if int(key) not in keep and self.index.contains(int(key)):
                    self.index.remove(int(key))
            if save:
                self._save()

    def flush(self) -> None:
        with self._process_lock, self._lock:
            self._save()

    def keys(self) -> set[int]:
        with self._lock:
            self._refresh_if_changed()
            return {int(k) for k in np.asarray(self.index.keys)}

    def upsert(self, chunks: Sequence[Chunk]) -> None:
        self.add(chunks, self.embed(chunks))

    def delete_keys(self, keys: Sequence[int]) -> None:
        if not keys:
            return
        with self._process_lock, self._lock:
            self._refresh_if_changed()
            changed = False
            for key in keys:
                if self.index.contains(int(key)):
                    self.index.remove(int(key))
                    changed = True
            if changed:
                self._save()

    def search(self, query: str, limit: int) -> list[tuple[int, float]]:
        if not query.strip() or limit < 1:
            return []
        vector = _validate_query_vector(next(iter(self.model.query_embed(query))), self.dim)
        with self._lock:
            self._refresh_if_changed()
            if len(self.index) == 0:
                return []
            matches = self.index.search(vector, min(limit, len(self.index)), threads=1)
            return [(int(match.key), float(1.0 - match.distance)) for match in matches]

    def count(self) -> int:
        with self._lock:
            self._refresh_if_changed()
            return int(len(self.index))

    def clear(self) -> None:
        with self._process_lock, self._lock:
            self.index = self._new_index()
            if self.index_path.exists():
                self.index_path.unlink()
            self._disk_signature = None

    def warmup(self) -> None:
        _validate_query_vector(next(iter(self.model.query_embed("warmup semantic retrieval"))), self.dim)

    @property
    def description(self) -> str:
        return f"FastEmbed {self.config.model_name} ({self.dim}d) + USearch HNSW bf16"


class HashingVectorStore:
    """Deterministic feature-hashing cosine store for offline/recovery operation.

    This is a functional retrieval backend, not a mocked embedding model. Neural semantic
    retrieval remains the default; select this backend explicitly when model/runtime
    dependencies are unavailable.
    """

    def __init__(self, paths: Paths, config: AppConfig, dimensions: int = 2048):
        self.paths = paths
        self.config = config
        self.dim = dimensions
        self.db_path = paths.home / "hash-vectors.sqlite3"
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(self.db_path, timeout=30, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("CREATE TABLE IF NOT EXISTS vectors(key INTEGER PRIMARY KEY, vec BLOB NOT NULL)")
        self._conn.commit()
        self._cache_keys: np.ndarray | None = None
        self._cache_vectors: np.ndarray | None = None
        self._cache_data_version: int | None = None

    @staticmethod
    def _terms(text: str) -> list[str]:
        words = [w.casefold() for w in re.findall(r"[\w.+#-]+", text, flags=re.UNICODE) if w.strip("._-+")]
        return words + [f"{a}\x1f{b}" for a, b in zip(words, words[1:])]

    def _embed(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim, dtype=np.float32)
        counts: dict[str, int] = {}
        for term in self._terms(text):
            counts[term] = counts.get(term, 0) + 1
        for term, count in counts.items():
            digest = hashlib.blake2b(term.encode("utf-8"), digest_size=8).digest()
            raw = int.from_bytes(digest, "little")
            idx = raw % self.dim
            sign = -1.0 if raw & (1 << 63) else 1.0
            vec[idx] += sign * (1.0 + math.log(count))
        norm = float(np.linalg.norm(vec))
        if norm:
            vec /= norm
        return vec

    def _invalidate(self) -> None:
        self._cache_keys = None
        self._cache_vectors = None
        self._cache_data_version = None

    def _data_version(self) -> int:
        return int(self._conn.execute("PRAGMA data_version").fetchone()[0])

    def _load_cache(self) -> tuple[np.ndarray, np.ndarray]:
        data_version = self._data_version()
        if (
            self._cache_keys is None
            or self._cache_vectors is None
            or self._cache_data_version != data_version
        ):
            rows = self._conn.execute("SELECT key, vec FROM vectors ORDER BY key").fetchall()
            self._cache_keys = np.asarray([int(r[0]) for r in rows], dtype=np.int64)
            if rows:
                self._cache_vectors = np.vstack(
                    [np.frombuffer(r[1], dtype=np.float16).astype(np.float32) for r in rows]
                )
            else:
                self._cache_vectors = np.empty((0, self.dim), dtype=np.float32)
            self._cache_data_version = data_version
        return self._cache_keys, self._cache_vectors

    def embed(self, chunks: Sequence[Chunk]) -> np.ndarray:
        if not chunks:
            return np.empty((0, self.dim), dtype=np.float32)
        return np.vstack([self._embed(c.embedding_text) for c in chunks])

    def add(self, chunks: Sequence[Chunk], vectors: np.ndarray, *, remove: Sequence[int] = (), save: bool = True) -> None:
        if not len(chunks) and not remove:
            return
        rows = [(int(c.vector_key), vectors[i].astype(np.float16).tobytes()) for i, c in enumerate(chunks)]
        keep = {int(c.vector_key) for c in chunks}
        with self._lock, self._conn:
            self._conn.executemany("INSERT OR REPLACE INTO vectors(key,vec) VALUES(?,?)", rows)
            self._conn.executemany("DELETE FROM vectors WHERE key=?", ((int(k),) for k in remove if int(k) not in keep))
            self._invalidate()

    def flush(self) -> None:
        return None

    def keys(self) -> set[int]:
        with self._lock:
            return {int(r[0]) for r in self._conn.execute("SELECT key FROM vectors")}

    def upsert(self, chunks: Sequence[Chunk]) -> None:
        self.add(chunks, self.embed(chunks))

    def delete_keys(self, keys: Sequence[int]) -> None:
        if not keys:
            return
        with self._lock, self._conn:
            self._conn.executemany("DELETE FROM vectors WHERE key=?", ((int(k),) for k in keys))
            self._invalidate()

    def search(self, query: str, limit: int) -> list[tuple[int, float]]:
        q = self._embed(query)
        if not np.any(q):
            return []
        with self._lock:
            keys, matrix = self._load_cache()
            if len(keys) == 0:
                return []
            scores = matrix @ q
            n = min(limit, len(keys))
            if n == len(keys):
                idx = np.argsort(-scores)
            else:
                candidates = np.argpartition(-scores, n - 1)[:n]
                idx = candidates[np.argsort(-scores[candidates])]
            return [(int(keys[i]), float(scores[i])) for i in idx]

    def count(self) -> int:
        return int(self._conn.execute("SELECT count(*) FROM vectors").fetchone()[0])

    def clear(self) -> None:
        with self._lock, self._conn:
            self._conn.execute("DELETE FROM vectors")
            self._invalidate()

    def warmup(self) -> None:
        _ = self._embed("warmup semantic retrieval")

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    @property
    def description(self) -> str:
        return f"Feature hashing cosine fallback ({self.dim}d, non-neural)"


VectorStore: TypeAlias = FastEmbedUSearchStore | HashingVectorStore


def create_vector_store(paths: Paths, config: AppConfig) -> VectorStore:
    if config.vector_backend == "hashing":
        return HashingVectorStore(paths, config)
    if config.vector_backend == "fastembed-usearch":
        return FastEmbedUSearchStore(paths, config)
    raise ValueError(f"Unknown vector backend: {config.vector_backend}")
