from __future__ import annotations

import json
import logging
import os
import queue
import threading
import time
from pathlib import Path

from .catalog import Catalog
from .config import AppConfig
from .indexer import Indexer

log = logging.getLogger(__name__)


class StudyWatcher:
    QUEUE_MAX = 100_000  # event storms (git checkout, mass copy) must not grow memory without bound

    def __init__(self, catalog: Catalog, indexer: Indexer, config: AppConfig):
        self.catalog = catalog
        self.indexer = indexer
        self.config = config
        self._observer = None
        self._queue: queue.Queue[tuple[str, bool]] = queue.Queue(maxsize=self.QUEUE_MAX)
        self.dropped = 0
        self._stop = threading.Event()
        self._worker: threading.Thread | None = None

    def _enqueue(self, raw: str, is_directory: bool) -> None:
        try:
            self._queue.put_nowait((raw, is_directory))
        except queue.Full:
            self.dropped += 1  # the next sync/reconcile pass finds whatever was dropped
            if self.dropped == 1 or self.dropped % 10_000 == 0:
                log.warning("watcher queue full; %d events dropped so far (a full sync will catch up)", self.dropped)

    def start(self) -> bool:
        if not self.config.watcher_enabled or self._observer is not None:
            return False
        try:
            from watchdog.events import FileSystemEventHandler
            from watchdog.observers import Observer
        except ImportError:
            log.warning("watchdog unavailable; automatic filesystem updates disabled")
            return False

        owner = self

        class Handler(FileSystemEventHandler):
            def on_any_event(self, event):
                is_directory = bool(getattr(event, "is_directory", False))
                src = getattr(event, "src_path", None)
                dest = getattr(event, "dest_path", None)
                if src:
                    owner._enqueue(src, is_directory)
                if dest:
                    owner._enqueue(dest, is_directory)

        observer = Observer()
        scheduled: set[tuple[str, bool]] = set()
        handler = Handler()

        def schedule(target: Path, *, recursive: bool) -> None:
            if not target.exists() or not target.is_dir():
                return
            resolved = str(target.resolve())
            key = (resolved, recursive)
            if key in scheduled:
                return
            observer.schedule(handler, resolved, recursive=recursive)
            scheduled.add(key)

        for root in self.catalog.roots():
            rr = root.expanduser().resolve()
            is_dir = self._root_is_directory(rr)
            if is_dir and rr.exists():
                schedule(rr, recursive=True)
            # Also watch the parent non-recursively. This catches deletion/recreation
            # of the configured root itself, including after an editor's atomic rename.
            schedule(rr.parent, recursive=False)
        if not scheduled:
            return False
        observer.start()
        self._observer = observer
        self._worker = threading.Thread(target=self._run, name="study-retriever-watcher", daemon=True)
        self._worker.start()
        return True

    def _root_is_directory(self, root: Path) -> bool:
        if root.is_dir():
            return True
        if root.is_file():
            return False
        # Root may have just been deleted or renamed. Existing indexed descendants
        # distinguish a former directory root from a single-file root.
        rr = root.resolve()
        for source in self.catalog.sources_under_root(rr):
            try:
                if Path(source["path"]).resolve() != rr:
                    return True
            except OSError:
                continue
        return False

    def _find_root(self, path: Path) -> Path | None:
        resolved = path.expanduser().resolve()
        candidates: list[Path] = []
        for root in self.catalog.roots():
            try:
                rr = root.expanduser().resolve()
                if resolved == rr:
                    candidates.append(rr)
                elif self._root_is_directory(rr) and resolved.is_relative_to(rr):
                    candidates.append(rr)
            except (OSError, ValueError):
                continue
        return max(candidates, key=lambda p: len(str(p)), default=None)

    def _sync_directory_event(self, path: Path) -> None:
        root = self._find_root(path)
        if root is None:
            return
        if root.exists():
            self.indexer.sync_root(root)
            return
        # The configured root itself disappeared. Remove stale index entries but
        # keep the configured root so it is picked up again if restored.
        for source in self.catalog.sources_under_root(root):
            self.indexer.index_file(Path(source["path"]), root)

    _RETRY_DELAYS = (2.0, 10.0, 60.0)  # notification-style retry with backoff (ch. 10), then give up

    def _heartbeat(self) -> None:
        """Liveness signal so status can tell whether background indexing is running."""
        try:
            beat = self.catalog.db_path.parent / "watcher-heartbeat.json"
            tmp = beat.with_suffix(".tmp")
            tmp.write_text(json.dumps({"pid": os.getpid(), "ts": time.time(), "pending": self._queue.qsize(), "dropped": self.dropped}), encoding="utf-8")
            os.replace(tmp, beat)
        except OSError:
            pass

    def _run(self) -> None:
        pending: dict[tuple[str, bool], float] = {}
        attempts: dict[tuple[str, bool], int] = {}
        last_beat = 0.0
        while not self._stop.is_set():
            if time.monotonic() - last_beat >= 10.0:
                last_beat = time.monotonic()
                self._heartbeat()
            try:
                item = self._queue.get(timeout=0.2)
                pending[item] = time.monotonic()
            except queue.Empty:
                pass
            now = time.monotonic()
            ready = [item for item, t in pending.items() if now - t >= self.config.watcher_debounce_seconds]
            for raw, is_directory in ready:
                pending.pop((raw, is_directory), None)
                path = Path(raw)
                try:
                    if is_directory:
                        self._sync_directory_event(path)
                        continue
                    root = self._find_root(path)
                    if root is None:
                        existing = self.catalog.source_by_path(path)
                        if existing and not path.exists():
                            root = Path(existing["root"])
                        else:
                            continue
                    key = (raw, is_directory)
                    status, _ = self.indexer.index_file(path, root, force=attempts.get(key, 0) > 0)
                    if status == "error" and attempts.get(key, 0) < len(self._RETRY_DELAYS):
                        delay = self._RETRY_DELAYS[attempts.get(key, 0)]
                        attempts[key] = attempts.get(key, 0) + 1
                        pending[key] = time.monotonic() + delay - self.config.watcher_debounce_seconds
                    else:
                        attempts.pop(key, None)
                except Exception:
                    log.exception("Watcher failed for %s", path)

    def stop(self) -> None:
        self._stop.set()
        if self._observer is not None:
            self._observer.stop()
            self._observer.join(timeout=5)
            self._observer = None
        if self._worker is not None and self._worker is not threading.current_thread():
            self._worker.join(timeout=5)
        self._worker = None
