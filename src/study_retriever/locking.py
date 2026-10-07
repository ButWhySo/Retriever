from __future__ import annotations

import os
import threading
import time
from pathlib import Path
from types import TracebackType
from typing import BinaryIO

_thread_locks_guard = threading.Lock()
_thread_locks: dict[str, threading.Lock] = {}


class InterProcessLock:
    """Small cross-platform exclusive file lock using only the Python standard library."""

    def __init__(self, path: Path, *, timeout: float = 120.0, poll: float = 0.05):
        self.path = path
        self.timeout = timeout
        self.poll = poll
        self._fh: BinaryIO | None = None
        self._owner: int | None = None
        key = os.path.normcase(str(path.resolve()))
        with _thread_locks_guard:
            self._thread_lock = _thread_locks.setdefault(key, threading.Lock())

    def acquire(self) -> None:
        deadline = time.monotonic() + self.timeout
        if self._owner == threading.get_ident():
            raise RuntimeError("lock is not reentrant")
        if not self._thread_lock.acquire(timeout=self.timeout):
            raise TimeoutError(f"Timed out waiting for lock: {self.path}")
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            fh = self.path.open("a+b")
            if os.name == "nt":
                # msvcrt.locking locks bytes from the current file position. Keep one
                # byte in the file so every process contends on the same region.
                fh.seek(0, os.SEEK_END)
                if fh.tell() == 0:
                    fh.write(b"\0")
                    fh.flush()
            while True:
                try:
                    fh.seek(0)
                    if os.name == "nt":
                        import msvcrt

                        msvcrt.locking(fh.fileno(), msvcrt.LK_NBLCK, 1)
                    else:
                        import fcntl

                        fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    self._fh = fh
                    self._owner = threading.get_ident()
                    return
                except (BlockingIOError, OSError):
                    if time.monotonic() >= deadline:
                        fh.close()
                        raise TimeoutError(f"Timed out waiting for lock: {self.path}")
                    time.sleep(self.poll)
        except BaseException:
            self._thread_lock.release()
            raise

    def release(self) -> None:
        fh = self._fh
        if fh is None:
            return
        try:
            fh.seek(0)
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(fh.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
        finally:
            fh.close()
            self._fh = None
            self._owner = None
            self._thread_lock.release()

    def __enter__(self) -> "InterProcessLock":
        self.acquire()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.release()
