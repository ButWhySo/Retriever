from __future__ import annotations

import json
import logging
import logging.handlers
import os
import threading
import time
from pathlib import Path

from . import __version__
from .catalog import Catalog
from .config import Paths, load_config, save_config
from .indexer import Indexer
from .search import SearchEngine
from .vector_store import VectorStore, create_vector_store
from .watcher import StudyWatcher


def lower_process_priority() -> None:
    """Keep background indexing from competing with the foreground apps (Windows: BELOW_NORMAL)."""
    try:
        if os.name == "nt":
            import ctypes

            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            kernel32.GetCurrentProcess.restype = ctypes.c_void_p
            kernel32.SetPriorityClass(ctypes.c_void_p(kernel32.GetCurrentProcess()), 0x00004000)
        else:
            os.nice(5)
    except (OSError, AttributeError):
        pass


class Runtime:
    def __init__(self, home: Path | None = None, *, start_watcher: bool = True, reconcile: bool = False):
        self.paths = Paths(home)
        self._watcher_requested = start_watcher
        self.config = load_config(self.paths)
        self._configure_logging()
        self.lock = threading.RLock()
        self.catalog = Catalog(self.paths.db)
        self._reconcile_root_config()
        self.vectors: VectorStore = create_vector_store(self.paths, self.config)
        self.indexer = Indexer(self.catalog, self.vectors, self.config, self.lock)
        self.search = SearchEngine(self.catalog, self.vectors, self.config)
        self.watcher = StudyWatcher(self.catalog, self.indexer, self.config)
        if start_watcher:
            self.watcher.start()
        if reconcile:
            threading.Thread(target=self._safe_reconcile, name="study-retriever-reconcile", daemon=True).start()

    def _reconcile_root_config(self) -> None:
        configured = {str(Path(p).expanduser().resolve()) for p in self.config.roots}
        catalogued = {str(p.resolve()) for p in self.catalog.roots()}
        for path in sorted(configured - catalogued):
            self.catalog.add_root(Path(path))
        merged = sorted(configured | catalogued)
        if merged != sorted(configured):
            self.config.roots = merged
            save_config(self.paths, self.config)

    def _configure_logging(self) -> None:
        root = logging.getLogger("study_retriever")
        if root.handlers:
            return
        root.setLevel(logging.INFO)
        formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
        file_handler = logging.handlers.RotatingFileHandler(
            self.paths.logs / "study-retriever.log", maxBytes=5_000_000, backupCount=3, encoding="utf-8"
        )
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)

    def _safe_reconcile(self) -> None:
        try:
            self.indexer.sync_all()
            repaired = self.indexer.repair_vectors()
            if repaired["orphans_removed"] or repaired["vectors_restored"]:
                logging.getLogger(__name__).warning("Vector/catalog mismatch repaired: %s", repaired)
        except Exception:
            logging.getLogger(__name__).exception("Startup reconciliation failed")

    def restart_watcher(self) -> None:
        self.watcher.stop()
        self.watcher = StudyWatcher(self.catalog, self.indexer, self.config)
        if self._watcher_requested:
            self.watcher.start()

    def add_root(self, path: str, *, sync_now: bool = True) -> dict:
        root = Path(path).expanduser().resolve()
        if not root.exists():
            raise FileNotFoundError(root)
        with self.indexer.process_lock, self.lock:
            self.catalog.add_root(root)
            if str(root) not in self.config.roots:
                self.config.roots.append(str(root))
                save_config(self.paths, self.config)
        result = self.indexer.sync_root(root).as_dict() if sync_now else {"scheduled": False}
        self.restart_watcher()
        return {"root": str(root), "sync": result}

    def remove_root(self, path: str, *, delete_indexed: bool = True) -> dict:
        root = Path(path).expanduser().resolve()
        deleted_sources = 0
        deleted_chunks = 0
        with self.indexer.process_lock, self.lock:
            if delete_indexed:
                for source in self.catalog.sources_under_root(root):
                    keys = self.catalog.vector_keys_for_source(source["source_id"])
                    deleted_chunks += len(keys)
                    self.vectors.delete_keys(keys)
                    self.catalog.delete_source(source["source_id"])
                    deleted_sources += 1
            self.catalog.remove_root(root)
            self.config.roots = [r for r in self.config.roots if Path(r).expanduser().resolve() != root]
            save_config(self.paths, self.config)
        self.restart_watcher()
        return {"root": str(root), "deleted_sources": deleted_sources, "deleted_chunks": deleted_chunks}

    def _heartbeat_age(self) -> float | None:
        """Seconds since any watcher (this process or the background indexer) last reported in; None if never."""
        try:
            beat = json.loads((self.paths.home / "watcher-heartbeat.json").read_text(encoding="utf-8"))
            return round(max(0.0, time.time() - float(beat["ts"])), 1)
        except (OSError, ValueError, KeyError):
            return None

    def status(self) -> dict:
        stats = self.catalog.stats()
        stats.update({
            "version": __version__,
            "vector_backend": self.vectors.description,
            "vectors": self.vectors.count(),
            "model": self.config.model_name,
            "home": str(self.paths.home),
            "vector_count_matches_chunks": self.vectors.count() == stats["chunks"],
            "error_sources": self.catalog.error_sources(),
            "watcher_heartbeat_age_s": self._heartbeat_age(),
        })
        return stats

    def close(self) -> None:
        self.watcher.stop()
        close_vectors = getattr(self.vectors, "close", None)
        if close_vectors is not None:
            close_vectors()
        self.catalog.close()


_runtime: Runtime | None = None
_runtime_lock = threading.Lock()


def get_runtime(*, reconcile: bool = False) -> Runtime:
    global _runtime
    if _runtime is None:
        with _runtime_lock:
            if _runtime is None:
                _runtime = Runtime(start_watcher=True, reconcile=reconcile)
    return _runtime
