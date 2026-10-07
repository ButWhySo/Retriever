from __future__ import annotations

import json
import re
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

from .locking import InterProcessLock
from .models import Chunk


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _fts_query(query: str) -> str:
    query = query.replace("\x00", " ")
    tokens = re.findall(r"[^\W_]+(?:[-_+][^\W_]+)*|[A-Za-z_][A-Za-z0-9_+.-]*", query, flags=re.UNICODE)
    seen: set[str] = set()
    cleaned: list[str] = []
    for token in tokens:
        token = token.strip(".-_")
        if not token or token.casefold() in seen:
            continue
        seen.add(token.casefold())
        escaped = token.replace('"', '""')
        cleaned.append(f'"{escaped}"')
        if len(cleaned) >= 24:
            break
    if not cleaned:
        escaped = query.strip().replace('"', '""')
        return f'"{escaped}"' if escaped else '""'
    return " OR ".join(cleaned)


class Catalog:
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        self._db_lock = InterProcessLock(self.db_path.parent / "index-writer.lock", timeout=600.0)
        with self._db_lock:
            self._init_schema()

    def _conn(self) -> sqlite3.Connection:
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(self.db_path, timeout=30.0, check_same_thread=False)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA busy_timeout=30000")
            # Setting WAL can fail immediately when another new process is opening
            # the same database. Schema initialization is process-locked, so set it
            # there once rather than racing on every process's first connection.
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute("PRAGMA foreign_keys=ON")
            self._local.conn = conn
        return conn

    def close(self) -> None:
        conn = getattr(self._local, "conn", None)
        if conn is not None:
            conn.close()
            self._local.conn = None

    def _init_schema(self) -> None:
        conn = self._conn()
        conn.execute("PRAGMA journal_mode=WAL")
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS roots (
                path TEXT PRIMARY KEY,
                enabled INTEGER NOT NULL DEFAULT 1,
                added_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS sources (
                source_id TEXT PRIMARY KEY,
                path TEXT NOT NULL UNIQUE,
                root TEXT NOT NULL,
                relative_path TEXT NOT NULL,
                title TEXT NOT NULL,
                suffix TEXT NOT NULL,
                size_bytes INTEGER NOT NULL,
                mtime_ns INTEGER NOT NULL,
                sha256 TEXT NOT NULL,
                indexed_at TEXT NOT NULL,
                status TEXT NOT NULL,
                error TEXT,
                chunk_count INTEGER NOT NULL DEFAULT 0,
                metadata_json TEXT NOT NULL DEFAULT '{}'
            );
            CREATE INDEX IF NOT EXISTS idx_sources_root ON sources(root);
            CREATE INDEX IF NOT EXISTS idx_sources_path ON sources(path);

            CREATE TABLE IF NOT EXISTS chunks (
                chunk_id TEXT PRIMARY KEY,
                source_id TEXT NOT NULL REFERENCES sources(source_id) ON DELETE CASCADE,
                vector_key INTEGER NOT NULL UNIQUE,
                ordinal INTEGER NOT NULL,
                title TEXT NOT NULL,
                section TEXT NOT NULL DEFAULT '',
                path TEXT NOT NULL,
                source_uri TEXT NOT NULL,
                locator_json TEXT NOT NULL DEFAULT '{}',
                text TEXT NOT NULL,
                embedding_text TEXT NOT NULL,
                UNIQUE(source_id, ordinal)
            );
            CREATE INDEX IF NOT EXISTS idx_chunks_source_ordinal ON chunks(source_id, ordinal);

            CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
                chunk_id UNINDEXED,
                source_id UNINDEXED,
                title,
                section,
                path,
                text,
                tokenize='unicode61 remove_diacritics 2'
            );
            """
        )
        conn.commit()

    def add_root(self, path: Path) -> None:
        p = str(path.expanduser().resolve())
        self._conn().execute(
            "INSERT INTO roots(path, enabled, added_at) VALUES(?,1,?) "
            "ON CONFLICT(path) DO UPDATE SET enabled=1",
            (p, _now()),
        )
        self._conn().commit()

    def remove_root(self, path: Path) -> None:
        p = str(path.expanduser().resolve())
        self._conn().execute("DELETE FROM roots WHERE path=?", (p,))
        self._conn().commit()

    def roots(self) -> list[Path]:
        rows = self._conn().execute("SELECT path FROM roots WHERE enabled=1 ORDER BY path").fetchall()
        return [Path(row["path"]) for row in rows]

    def source_stat(self, path: Path) -> sqlite3.Row | None:
        return self._conn().execute(
            "SELECT source_id, size_bytes, mtime_ns, sha256, status FROM sources WHERE path=?",
            (str(path.resolve()),),
        ).fetchone()

    def update_source_stat(self, path: Path, *, size_bytes: int, mtime_ns: int) -> None:
        self._conn().execute(
            "UPDATE sources SET size_bytes=?, mtime_ns=?, indexed_at=? WHERE path=?",
            (size_bytes, mtime_ns, _now(), str(path.resolve())),
        )
        self._conn().commit()

    def source_by_id(self, source_id: str) -> dict | None:
        row = self._conn().execute("SELECT * FROM sources WHERE source_id=?", (source_id,)).fetchone()
        return dict(row) if row else None

    def source_by_path(self, path: Path) -> dict | None:
        row = self._conn().execute("SELECT * FROM sources WHERE path=?", (str(path.resolve()),)).fetchone()
        return dict(row) if row else None

    def replace_source(
        self,
        *,
        source_id: str,
        path: Path,
        root: Path,
        relative_path: str,
        title: str,
        suffix: str,
        size_bytes: int,
        mtime_ns: int,
        sha256: str,
        metadata: dict,
        chunks: Sequence[Chunk],
    ) -> None:
        conn = self._conn()
        with conn:
            old_ids = [r[0] for r in conn.execute("SELECT chunk_id FROM chunks WHERE source_id=?", (source_id,))]
            if old_ids:
                conn.executemany("DELETE FROM chunks_fts WHERE chunk_id=?", ((cid,) for cid in old_ids))
            conn.execute("DELETE FROM chunks WHERE source_id=?", (source_id,))
            conn.execute(
                """
                INSERT INTO sources(source_id,path,root,relative_path,title,suffix,size_bytes,mtime_ns,sha256,indexed_at,status,error,chunk_count,metadata_json)
                VALUES(?,?,?,?,?,?,?,?,?,?, 'ready', NULL, ?, ?)
                ON CONFLICT(source_id) DO UPDATE SET
                    path=excluded.path, root=excluded.root, relative_path=excluded.relative_path,
                    title=excluded.title, suffix=excluded.suffix, size_bytes=excluded.size_bytes,
                    mtime_ns=excluded.mtime_ns, sha256=excluded.sha256, indexed_at=excluded.indexed_at,
                    status='ready', error=NULL, chunk_count=excluded.chunk_count, metadata_json=excluded.metadata_json
                """,
                (
                    source_id, str(path.resolve()), str(root.resolve()), relative_path, title, suffix,
                    size_bytes, mtime_ns, sha256, _now(), len(chunks), json.dumps(metadata, ensure_ascii=False),
                ),
            )
            conn.executemany(
                """INSERT INTO chunks(chunk_id,source_id,vector_key,ordinal,title,section,path,source_uri,locator_json,text,embedding_text)
                   VALUES(?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    (
                        c.chunk_id, c.source_id, c.vector_key, c.ordinal, c.title, c.section, c.path, c.source_uri,
                        json.dumps(c.locator, ensure_ascii=False), c.text, c.embedding_text,
                    )
                    for c in chunks
                ),
            )
            conn.executemany(
                "INSERT INTO chunks_fts(chunk_id,source_id,title,section,path,text) VALUES(?,?,?,?,?,?)",
                ((c.chunk_id, c.source_id, c.title, c.section, c.path, c.text) for c in chunks),
            )

    def mark_error(self, *, source_id: str, path: Path, root: Path, error: str, size_bytes: int, mtime_ns: int) -> None:
        conn = self._conn()
        relative = str(path.resolve().relative_to(root.resolve())) if path.resolve().is_relative_to(root.resolve()) else path.name
        with conn:
            conn.execute(
                """
                INSERT INTO sources(source_id,path,root,relative_path,title,suffix,size_bytes,mtime_ns,sha256,indexed_at,status,error,chunk_count,metadata_json)
                VALUES(?,?,?,?,?,?,?,?,?,?,'error',?,0,'{}')
                ON CONFLICT(source_id) DO UPDATE SET size_bytes=excluded.size_bytes, mtime_ns=excluded.mtime_ns,
                    indexed_at=excluded.indexed_at, status='error', error=excluded.error
                """,
                (source_id, str(path.resolve()), str(root.resolve()), relative, path.stem, path.suffix.lower(),
                 size_bytes, mtime_ns, "", _now(), error[:4000]),
            )

    def delete_source(self, source_id: str) -> None:
        conn = self._conn()
        with conn:
            ids = [r[0] for r in conn.execute("SELECT chunk_id FROM chunks WHERE source_id=?", (source_id,))]
            if ids:
                conn.executemany("DELETE FROM chunks_fts WHERE chunk_id=?", ((cid,) for cid in ids))
            conn.execute("DELETE FROM sources WHERE source_id=?", (source_id,))

    def sources_under_root(self, root: Path) -> list[dict]:
        rows = self._conn().execute("SELECT * FROM sources WHERE root=? ORDER BY path", (str(root.resolve()),)).fetchall()
        return [dict(r) for r in rows]

    def all_sources(self) -> list[dict]:
        rows = self._conn().execute("SELECT * FROM sources ORDER BY path").fetchall()
        return [dict(r) for r in rows]

    def lexical_search(self, query: str, limit: int) -> list[tuple[str, float]]:
        query = query.replace("\x00", " ")
        match = _fts_query(query)
        if not match or match == '""':
            return []
        try:
            rows = self._conn().execute(
                """
                SELECT chunk_id, bm25(chunks_fts, 0.0, 0.0, 4.0, 2.5, 0.3, 1.0) AS score
                FROM chunks_fts
                WHERE chunks_fts MATCH ?
                ORDER BY score
                LIMIT ?
                """,
                (match, limit),
            ).fetchall()
        except sqlite3.OperationalError:
            escaped = query.replace('"', '""').strip()
            if not escaped:
                return []
            try:
                rows = self._conn().execute(
                    """SELECT chunk_id, bm25(chunks_fts) AS score FROM chunks_fts
                       WHERE chunks_fts MATCH ? ORDER BY score LIMIT ?""",
                    (f'"{escaped}"', limit),
                ).fetchall()
            except sqlite3.OperationalError:
                return []
        return [(str(r["chunk_id"]), float(r["score"])) for r in rows]


    def vector_keys_for_source(self, source_id: str) -> list[int]:
        rows = self._conn().execute("SELECT vector_key FROM chunks WHERE source_id=? ORDER BY ordinal", (source_id,)).fetchall()
        return [int(r[0]) for r in rows]

    def vector_key_map(self, keys: Iterable[int]) -> dict[int, str]:
        values = list(dict.fromkeys(int(k) for k in keys))
        if not values:
            return {}
        out: dict[int, str] = {}
        conn = self._conn()
        for offset in range(0, len(values), 900):
            batch = values[offset:offset + 900]
            placeholders = ",".join("?" for _ in batch)
            rows = conn.execute(f"SELECT vector_key, chunk_id FROM chunks WHERE vector_key IN ({placeholders})", batch).fetchall()
            out.update({int(r["vector_key"]): str(r["chunk_id"]) for r in rows})
        return out

    def iter_chunks(self, batch_size: int = 256, vector_keys: Iterable[int] | None = None):
        """Yield chunk rows in rowid order using keyset pagination (no OFFSET rescans)."""
        wanted = None if vector_keys is None else {int(k) for k in vector_keys}
        last_rowid = 0
        conn = self._conn()
        while True:
            rows = conn.execute(
                "SELECT rowid AS _rowid, * FROM chunks WHERE rowid > ? ORDER BY rowid LIMIT ?", (last_rowid, batch_size)
            ).fetchall()
            if not rows:
                break
            last_rowid = int(rows[-1]["_rowid"])
            batch = []
            for row in rows:
                item = dict(row)
                item.pop("_rowid", None)
                if wanted is not None and int(item["vector_key"]) not in wanted:
                    continue
                item["locator"] = json.loads(item.pop("locator_json"))
                batch.append(item)
            if batch:
                yield batch

    def all_vector_keys(self) -> set[int]:
        return {int(r[0]) for r in self._conn().execute("SELECT vector_key FROM chunks")}

    def get_chunks(self, chunk_ids: Iterable[str]) -> dict[str, dict]:
        ids = list(dict.fromkeys(chunk_ids))
        if not ids:
            return {}
        out: dict[str, dict] = {}
        conn = self._conn()
        for offset in range(0, len(ids), 900):
            batch = ids[offset:offset + 900]
            placeholders = ",".join("?" for _ in batch)
            rows = conn.execute(f"SELECT * FROM chunks WHERE chunk_id IN ({placeholders})", batch).fetchall()
            for row in rows:
                item = dict(row)
                item["locator"] = json.loads(item.pop("locator_json"))
                out[item["chunk_id"]] = item
        return out

    def get_chunk(self, chunk_id: str) -> dict | None:
        return self.get_chunks([chunk_id]).get(chunk_id)

    def neighbors(self, source_id: str, ordinal: int, radius: int = 1) -> list[dict]:
        rows = self._conn().execute(
            "SELECT * FROM chunks WHERE source_id=? AND ordinal BETWEEN ? AND ? ORDER BY ordinal",
            (source_id, max(0, ordinal - radius), ordinal + radius),
        ).fetchall()
        out = []
        for row in rows:
            item = dict(row)
            item["locator"] = json.loads(item.pop("locator_json"))
            out.append(item)
        return out

    def error_sources(self, limit: int = 20) -> list[dict[str, str]]:
        rows = self._conn().execute(
            "SELECT path, error FROM sources WHERE status='error' ORDER BY indexed_at DESC LIMIT ?",
            (max(1, min(int(limit), 100)),),
        ).fetchall()
        return [{"path": str(row["path"]), "error": str(row["error"] or "unknown error")} for row in rows]

    def stats(self) -> dict:
        conn = self._conn()
        source_count = conn.execute("SELECT count(*) FROM sources WHERE status='ready'").fetchone()[0]
        error_count = conn.execute("SELECT count(*) FROM sources WHERE status='error'").fetchone()[0]
        chunk_count = conn.execute("SELECT count(*) FROM chunks").fetchone()[0]
        total_bytes = conn.execute("SELECT COALESCE(sum(size_bytes),0) FROM sources WHERE status='ready'").fetchone()[0]
        return {
            "roots": [str(p) for p in self.roots()],
            "sources": int(source_count),
            "errors": int(error_count),
            "chunks": int(chunk_count),
            "source_bytes": int(total_bytes),
            "db_path": str(self.db_path),
        }
