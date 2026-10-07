from __future__ import annotations

import multiprocessing as mp
import threading
import time
from pathlib import Path

from study_retriever.locking import InterProcessLock


def _hold_lock(path: str, ready: mp.Queue, hold_seconds: float) -> None:
    with InterProcessLock(Path(path), timeout=5):
        ready.put(True)
        time.sleep(hold_seconds)


def test_interprocess_lock_serializes_processes(tmp_path: Path) -> None:
    lock_path = tmp_path / "write.lock"
    ctx = mp.get_context("spawn")
    ready = ctx.Queue()
    proc = ctx.Process(target=_hold_lock, args=(str(lock_path), ready, 0.35))
    proc.start()
    assert ready.get(timeout=3) is True
    start = time.monotonic()
    with InterProcessLock(lock_path, timeout=3):
        waited = time.monotonic() - start
    proc.join(timeout=3)
    assert proc.exitcode == 0
    assert waited >= 0.20


def test_interprocess_lock_serializes_threads_on_one_instance(tmp_path: Path) -> None:
    lock = InterProcessLock(tmp_path / "write.lock", timeout=3)
    held = threading.Event()
    release = threading.Event()
    acquired = threading.Event()

    def hold() -> None:
        with lock:
            held.set()
            assert release.wait(3)

    def contend() -> None:
        with lock:
            acquired.set()

    first = threading.Thread(target=hold)
    second = threading.Thread(target=contend)
    first.start()
    assert held.wait(3)
    second.start()
    assert not acquired.wait(0.1)
    release.set()
    first.join(3)
    second.join(3)
    assert not first.is_alive() and not second.is_alive()
    assert acquired.is_set()
