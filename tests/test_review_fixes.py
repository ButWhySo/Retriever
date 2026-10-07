# pyright: reportArgumentType=false, reportAttributeAccessIssue=false, reportOptionalMemberAccess=false
from __future__ import annotations

import threading
from pathlib import Path

from study_retriever.cli import _print
from study_retriever.config import AppConfig, Paths, save_config
from study_retriever.runtime import Runtime


def make_runtime(tmp_path: Path) -> Runtime:
    home = tmp_path / "home"
    paths = Paths(home)
    save_config(paths, AppConfig(vector_backend="hashing", watcher_enabled=False, chunk_chars=500, overlap_chars=60))
    return Runtime(home=home, start_watcher=False)


def test_watcher_style_events_inside_excluded_dirs_are_not_indexed(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    (corpus / "node_modules" / "pkg").mkdir(parents=True)
    (corpus / "notes").mkdir()
    vendored = corpus / "node_modules" / "pkg" / "README.md"
    vendored.write_text("vendored dependency documentation\n", encoding="utf-8")
    note = corpus / "notes" / "real.md"
    note.write_text("real study note about gradient descent\n", encoding="utf-8")

    rt = make_runtime(tmp_path)
    try:
        assert rt.indexer.index_file(vendored, corpus)[0] == "excluded"
        assert rt.indexer.index_file(note, corpus)[0] == "indexed"
        assert rt.catalog.source_by_path(vendored) is None
        assert rt.catalog.source_by_path(note) is not None
    finally:
        rt.close()


def test_embedding_runs_outside_the_in_process_lock(tmp_path: Path) -> None:
    """A slow embedding must not block searches that need the runtime lock."""
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    doc = corpus / "big.md"
    doc.write_text("# Big\n" + "optimizer momentum adam\n" * 40, encoding="utf-8")

    rt = make_runtime(tmp_path)
    started, release = threading.Event(), threading.Event()
    real_embed = rt.vectors.embed

    def slow_embed(chunks):
        started.set()
        assert release.wait(timeout=10)
        return real_embed(chunks)

    rt.vectors.embed = slow_embed  # type: ignore[method-assign]
    worker = threading.Thread(target=lambda: rt.indexer.index_file(doc, corpus))
    try:
        worker.start()
        assert started.wait(timeout=10)
        acquired = rt.lock.acquire(timeout=1)
        assert acquired, "runtime lock was held while embedding"
        rt.lock.release()
    finally:
        release.set()
        worker.join(timeout=20)
        rt.close()


def test_reindex_and_rebuild_keep_catalog_and_vectors_consistent(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    doc = corpus / "a.md"
    doc.write_text("alpha beta gamma\n" * 80, encoding="utf-8")

    rt = make_runtime(tmp_path)
    try:
        rt.add_root(str(corpus), sync_now=True)
        before = rt.status()
        assert before["vector_count_matches_chunks"]

        doc.write_text("delta epsilon zeta\n" * 30, encoding="utf-8")
        assert rt.indexer.index_file(doc, corpus)[0] == "indexed"
        after = rt.status()
        assert after["vector_count_matches_chunks"]
        assert after["chunks"] < before["chunks"]

        result = rt.indexer.rebuild_vectors()
        assert result["vectors"] == after["chunks"]
        assert rt.status()["vector_count_matches_chunks"]
    finally:
        rt.close()


def test_cli_json_output_survives_non_cp1252_text(capsysbinary) -> None:
    _print({"symbol": "θ", "citation": "page 3 · slide 4"})
    out = capsysbinary.readouterr().out.decode("utf-8")
    assert "θ" in out and "·" in out


def test_editing_one_paragraph_re_embeds_only_changed_chunks(tmp_path: Path) -> None:
    """Delta sync: unchanged chunks keep their vectors (stable content-derived keys)."""
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    doc = corpus / "notes.md"
    paragraphs = [f"## Topic {i}\n" + (f"unique content number {i} about subject {i}. " * 12) for i in range(12)]
    doc.write_text("\n\n".join(paragraphs), encoding="utf-8")

    rt = make_runtime(tmp_path)
    embedded: list[int] = []
    real_embed = rt.vectors.embed

    def counting_embed(chunks):
        embedded.append(len(chunks))
        return real_embed(chunks)

    rt.vectors.embed = counting_embed  # type: ignore[method-assign]
    try:
        rt.add_root(str(corpus), sync_now=True)
        first_total = sum(embedded)
        total_chunks = rt.status()["chunks"]
        assert first_total == total_chunks and total_chunks > 6

        embedded.clear()
        paragraphs[9] = "## Topic 9\n" + "completely rewritten paragraph about something else. " * 12
        doc.write_text("\n\n".join(paragraphs), encoding="utf-8")
        assert rt.indexer.index_file(doc, corpus)[0] == "indexed"
        assert 0 < sum(embedded) < total_chunks // 2
        assert rt.status()["vector_count_matches_chunks"]
    finally:
        rt.close()


def test_repair_vectors_diffs_key_sets_instead_of_counts(tmp_path: Path) -> None:
    """An orphan plus a missing vector cancel out in a count check; key-set anti-entropy still fixes both."""
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "a.md").write_text("alpha beta gamma delta\n" * 60, encoding="utf-8")

    rt = make_runtime(tmp_path)
    try:
        rt.add_root(str(corpus), sync_now=True)
        keys = sorted(rt.catalog.all_vector_keys())
        rt.vectors.delete_keys([keys[0]])  # missing vector
        orphan = max(keys) + 12345
        rt.vectors._conn.execute("INSERT INTO vectors(key, vec) VALUES(?, ?)", (orphan, b"\0" * 4096))
        rt.vectors._conn.commit()
        assert rt.vectors.count() == len(keys)  # counts agree, sets do not
        assert rt.vectors.keys() != set(keys)

        result = rt.indexer.repair_vectors()
        assert result == {"orphans_removed": 1, "vectors_restored": 1}
        assert rt.vectors.keys() == set(keys)
        assert rt.indexer.repair_vectors() == {"orphans_removed": 0, "vectors_restored": 0}
    finally:
        rt.close()


def test_exclude_globs_filter_layer_removes_matching_files(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "keep.md").write_text("keep this note about transformers\n", encoding="utf-8")
    (corpus / "L3-presentation.md").write_text("duplicate of keep with other name\n", encoding="utf-8")

    home = tmp_path / "home"
    paths = Paths(home)
    save_config(paths, AppConfig(vector_backend="hashing", watcher_enabled=False, exclude_globs=["*-presentation.md"]))
    rt = Runtime(home=home, start_watcher=False)
    try:
        rt.add_root(str(corpus), sync_now=True)
        assert rt.catalog.source_by_path(corpus / "keep.md") is not None
        assert rt.catalog.source_by_path(corpus / "L3-presentation.md") is None
        assert rt.indexer.index_file(corpus / "L3-presentation.md", corpus)[0] == "excluded"
    finally:
        rt.close()


def test_malformed_file_is_not_reparsed_until_it_changes(tmp_path: Path, monkeypatch) -> None:
    import study_retriever.indexer as indexer_module

    corpus = tmp_path / "corpus"
    corpus.mkdir()
    bad = corpus / "broken.docx"
    bad.write_bytes(b"not a real docx")

    rt = make_runtime(tmp_path)
    calls = []
    real_parse = indexer_module.parse_document

    def counting_parse(path):
        calls.append(path)
        return real_parse(path)

    monkeypatch.setattr(indexer_module, "parse_document", counting_parse)
    monkeypatch.setattr(indexer_module.time, "sleep", lambda _s: None)
    try:
        assert rt.indexer.index_file(bad, corpus)[0] == "error"
        first = len(calls)
        assert first >= 1
        assert rt.indexer.index_file(bad, corpus)[0] == "error"
        assert len(calls) == first  # permanent failure is remembered, no re-parse
        bad.write_bytes(b"still not a docx but different")
        assert rt.indexer.index_file(bad, corpus)[0] == "error"
        assert len(calls) > first  # changed file is tried again
    finally:
        rt.close()


def test_status_reports_watcher_heartbeat(tmp_path: Path) -> None:
    rt = make_runtime(tmp_path)
    try:
        assert rt.status()["watcher_heartbeat_age_s"] is None
        rt.watcher._heartbeat()
        age = rt.status()["watcher_heartbeat_age_s"]
        assert age is not None and age < 5
    finally:
        rt.close()


def test_root_deny_list_blocks_profile_and_credentials_but_allows_study_folders(tmp_path: Path) -> None:
    import pytest

    from study_retriever.runtime import check_root_allowed

    home = Path.home()
    for bad in (home, home.parent, Path(home.anchor), home / ".ssh", home / "AppData" / "Roaming" / "Mozilla"):
        with pytest.raises(PermissionError):
            check_root_allowed(bad)
    check_root_allowed(tmp_path)  # pytest temp dir lives under AppData\Local\Temp and must stay allowed
    study = tmp_path / "Study" / "notes"
    study.mkdir(parents=True)
    check_root_allowed(study)


def test_mcp_root_changes_are_opt_in(tmp_path: Path, monkeypatch) -> None:
    import pytest
    from mcp.server.mcpserver.exceptions import ToolError

    from study_retriever import mcp_server

    folder = tmp_path / "docs"
    folder.mkdir()
    (folder / "a.md").write_text("some indexed text\n", encoding="utf-8")
    rt = make_runtime(tmp_path)
    monkeypatch.setattr(mcp_server, "_rt", lambda: rt)
    monkeypatch.delenv("STUDY_RETRIEVER_ALLOW_MCP_ROOT_CHANGES", raising=False)
    try:
        with pytest.raises(ToolError, match="disabled"):
            mcp_server.add_study_root(str(folder))
        with pytest.raises(ToolError, match="disabled"):
            mcp_server.remove_study_root(str(folder))
        monkeypatch.setenv("STUDY_RETRIEVER_ALLOW_MCP_ROOT_CHANGES", "1")
        assert mcp_server.add_study_root(str(folder)).root.endswith("docs")
    finally:
        rt.close()


def test_nul_query_and_oversized_inputs_fail_safely(tmp_path: Path, monkeypatch) -> None:
    import pytest

    import study_retriever.parsers as parsers
    from study_retriever.config import AppConfig as Cfg

    rt = make_runtime(tmp_path)
    try:
        assert rt.catalog.lexical_search("\x00", 5) == []
        assert rt.search.hybrid_search("find \x00 this", top_k=3) == []
    finally:
        rt.close()

    big = tmp_path / "big.md"
    big.write_text("x" * 5000, encoding="utf-8")
    monkeypatch.setattr(parsers, "_limits", lambda: Cfg(max_text_bytes=1000))
    with pytest.raises(ValueError, match="text-parser limit"):
        parsers.parse_document(big)

    svg = tmp_path / "bomb.svg"
    svg.write_text('<!DOCTYPE s [<!ENTITY a "aaaa">]><svg xmlns="http://www.w3.org/2000/svg"><text>&a;</text></svg>', encoding="utf-8")
    with pytest.raises(ValueError, match="entity"):
        parsers.parse_document(svg)


def test_pdf_render_is_clamped_to_a_pixel_budget(tmp_path: Path) -> None:
    import struct

    import fitz

    corpus = tmp_path / "corpus"
    corpus.mkdir()
    doc = fitz.open()
    doc.new_page(width=14400, height=14400)  # crafted 200-inch page
    doc.save(corpus / "huge.pdf")
    doc.close()

    rt = make_runtime(tmp_path)
    try:
        rt.add_root(str(corpus), sync_now=True)
        png = rt.search.render_pdf_page(path=str(corpus / "huge.pdf"), page=1, dpi=300)
        width, height = struct.unpack(">II", png[16:24])
        assert max(width, height) <= 4100
    finally:
        rt.close()


def test_disk_space_guard_refuses_writes_when_drive_is_nearly_full(tmp_path: Path, monkeypatch) -> None:
    import shutil as _shutil
    from collections import namedtuple

    import pytest

    corpus = tmp_path / "corpus"
    corpus.mkdir()
    doc = corpus / "a.md"
    doc.write_text("some study note\n", encoding="utf-8")
    rt = make_runtime(tmp_path)
    usage = namedtuple("usage", "total used free")
    try:
        monkeypatch.setattr(_shutil, "disk_usage", lambda _p: usage(10**12, 10**12 - 5 * 2**20, 5 * 2**20))
        assert rt.indexer.index_file(doc, corpus)[0] == "disk_full"
        assert rt.catalog.source_by_path(doc) is None
        assert rt.status()["free_disk_mb"] == 5
        with pytest.raises(RuntimeError, match="free disk space"):
            rt.indexer.rebuild_vectors()
        monkeypatch.undo()
        assert rt.indexer.index_file(doc, corpus)[0] == "indexed"
    finally:
        rt.close()


def test_search_degrades_to_lexical_when_dense_side_fails(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "a.md").write_text("gradient descent updates parameters along the negative gradient\n", encoding="utf-8")
    rt = make_runtime(tmp_path)
    try:
        rt.add_root(str(corpus), sync_now=True)

        def boom(*_a, **_k):
            raise RuntimeError("ONNX bad allocation")

        rt.vectors.search = boom  # type: ignore[method-assign]
        hits = rt.search.hybrid_search("negative gradient", top_k=3)
        assert hits and rt.search.degraded is True
        assert all(h.dense_rank is None for h in hits)
    finally:
        rt.close()


def test_model_is_unloaded_when_idle_and_reloaded_on_demand(tmp_path: Path, monkeypatch) -> None:
    import time

    import fastembed
    import numpy as np

    from study_retriever.vector_store import FastEmbedUSearchStore

    loads = []

    class FakeModel:
        embedding_size = 8

        def __init__(self, **_kw):
            loads.append(1)

        def query_embed(self, _q):
            yield np.ones(8, dtype=np.float32)

        def passage_embed(self, texts, batch_size=16):
            for _ in texts:
                yield np.ones(8, dtype=np.float32)

        @classmethod
        def get_embedding_size(cls, _name):
            return 8

    monkeypatch.setattr(fastembed, "TextEmbedding", FakeModel)
    paths = Paths(tmp_path / "home")
    store = FastEmbedUSearchStore(paths, AppConfig(model_idle_unload_seconds=1))
    assert loads == []  # dimension came from metadata; no model load at start-up
    assert store.search("hello", 3) == []
    assert len(loads) == 1 and store._model is not None
    deadline = time.time() + 6
    while store._model is not None and time.time() < deadline:
        time.sleep(0.1)
    assert store._model is None  # unloaded after idling
    store.search("again", 3)
    assert len(loads) == 2  # transparently reloaded


def test_watcher_queue_is_bounded_and_counts_drops(tmp_path: Path) -> None:
    import queue

    rt = make_runtime(tmp_path)
    try:
        rt.watcher._queue = queue.Queue(maxsize=2)
        for i in range(5):
            rt.watcher._enqueue(str(tmp_path / f"f{i}.md"), False)
        assert rt.watcher._queue.qsize() == 2 and rt.watcher.dropped == 3
    finally:
        rt.close()


def test_long_sync_returns_running_and_completes_in_background(tmp_path: Path, monkeypatch) -> None:
    import threading
    import time

    rt = make_runtime(tmp_path)
    release = threading.Event()
    calls = []

    def slow_sync(force=False):
        calls.append(force)
        release.wait(timeout=10)
        return {"total": {"indexed": 1, "unchanged": 0, "deleted": 0, "errors": 0, "chunks_added": 2}, "roots": {}}

    monkeypatch.setattr(rt.indexer, "sync_all", slow_sync)
    try:
        first = rt.start_sync(wait=0.2)
        assert first["status"] == "running" and rt.status()["sync_running"] is True
        assert rt.start_sync(wait=0.1)["status"] == "running" and len(calls) == 1  # no second sync started
        release.set()
        deadline = time.time() + 5
        while rt.status()["sync_running"] and time.time() < deadline:
            time.sleep(0.05)
        done = rt.start_sync(wait=5)
        assert done["status"] == "completed" and done["total"]["indexed"] == 1
    finally:
        release.set()
        rt.close()
