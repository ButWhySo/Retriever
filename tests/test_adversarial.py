# pyright: reportArgumentType=false, reportAttributeAccessIssue=false, reportOptionalMemberAccess=false
from __future__ import annotations

import io
import os
import subprocess
import threading
import time
import zipfile
from pathlib import Path

import pytest

from study_retriever.config import AppConfig, Paths, save_config
from study_retriever.runtime import Runtime

SUFFIXES = [".pdf", ".docx", ".pptx", ".xlsx", ".svg", ".npz", ".png", ".md", ".ipynb", ".html", ".json"]


def make_runtime(tmp_path: Path, **cfg) -> Runtime:
    home = tmp_path / "home"
    paths = Paths(home)
    save_config(paths, AppConfig(vector_backend="hashing", watcher_enabled=False, chunk_chars=500, overlap_chars=60, **cfg))
    return Runtime(home=home, start_watcher=False)


@pytest.fixture
def env(tmp_path: Path):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    rt = make_runtime(tmp_path)
    yield rt, corpus
    rt.close()


def consistent(rt: Runtime) -> bool:
    return bool(rt.status()["vector_count_matches_chunks"])


def test_unicode_long_space_filenames(env) -> None:
    rt, corpus = env
    names = ["résumé 学习 笔记 🎓.md", "a  b   c .md", "x" * 150 + ".md", " lead trail .md"]
    for n in names:
        try:
            (corpus / n).write_text("photosynthesis chlorophyll notes\n", encoding="utf-8")
        except OSError:
            continue
    res = rt.indexer.sync_root(corpus)
    assert res.errors == 0
    assert consistent(rt)
    assert rt.search.hybrid_search("photosynthesis")


@pytest.mark.parametrize("suffix", SUFFIXES)
def test_zero_byte_files(env, suffix) -> None:
    rt, corpus = env
    f = corpus / f"empty{suffix}"
    f.write_bytes(b"")
    status, _ = rt.indexer.index_file(f, corpus)
    assert status in {"indexed", "error", "unsupported"}
    assert consistent(rt)
    rt.search.hybrid_search("anything")


@pytest.mark.parametrize("suffix", SUFFIXES)
def test_binary_garbage(env, suffix) -> None:
    rt, corpus = env
    f = corpus / f"junk{suffix}"
    f.write_bytes(os.urandom(4096) + b"\x00\xff" * 100)
    t = time.time()
    status, _ = rt.indexer.index_file(f, corpus)
    assert time.time() - t < 15
    assert status in {"indexed", "error"}
    assert consistent(rt)


def _zip_bomb(names_and_sizes: dict[str, int]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for name, size in names_and_sizes.items():
            z.writestr(name, b"A" * size)
    return buf.getvalue()


def test_docx_zip_bomb(env) -> None:
    rt, corpus = env
    f = corpus / "bomb.docx"
    # ~300 MB uncompressed in word/document.xml, compresses to ~300 KB
    f.write_bytes(_zip_bomb({"[Content_Types].xml": 10, "word/document.xml": 300 * 1024 * 1024}))
    assert f.stat().st_size < 2 * 1024 * 1024
    t = time.time()
    status, _ = rt.indexer.index_file(f, corpus)
    assert time.time() - t < 30
    assert status in {"indexed", "error"}
    assert consistent(rt)


def test_xlsx_zip_bomb(env) -> None:
    rt, corpus = env
    f = corpus / "bomb.xlsx"
    f.write_bytes(_zip_bomb({"[Content_Types].xml": 10, "xl/worksheets/sheet1.xml": 300 * 1024 * 1024}))
    t = time.time()
    status, _ = rt.indexer.index_file(f, corpus)
    assert time.time() - t < 30
    assert status in {"indexed", "error"}
    assert consistent(rt)


def test_svg_billion_laughs(env) -> None:
    rt, corpus = env
    ents = ['<!ENTITY l0 "lol">']
    for i in range(1, 10):
        ents.append(f'<!ENTITY l{i} "' + f"&l{i-1};" * 10 + '">')
    svg = '<?xml version="1.0"?><!DOCTYPE s [' + "".join(ents) + ']><svg xmlns="http://www.w3.org/2000/svg"><text>&l9;</text></svg>'
    f = corpus / "laughs.svg"
    f.write_text(svg, encoding="utf-8")
    t = time.time()
    status, _ = rt.indexer.index_file(f, corpus)
    assert time.time() - t < 15
    assert status in {"indexed", "error"}
    assert consistent(rt)
    if status == "indexed":
        # expansion must not have been materialised into the catalog
        assert rt.status()["chunks"] < 5000


def test_big_single_line_text_is_fast(env) -> None:
    rt, corpus = env
    f = corpus / "line.md"
    f.write_text("word " * (300 * 1024), encoding="utf-8")  # one 1.5 MB line (5 MB takes ~150 s)
    t = time.time()
    status, _ = rt.indexer.index_file(f, corpus)
    assert status == "indexed"
    assert time.time() - t < 8
    assert consistent(rt)


def test_deeply_nested_dirs(env) -> None:
    rt, corpus = env
    d = corpus
    for i in range(30):
        d = d / f"d{i}"
    try:
        d.mkdir(parents=True)
        (d / "deep.md").write_text("deep nested note\n", encoding="utf-8")
    except OSError:
        pytest.skip("path too long")
    res = rt.indexer.sync_root(corpus)
    assert res.indexed == 1 and consistent(rt)


def test_link_outside_root_not_indexed(env, tmp_path: Path) -> None:
    rt, corpus = env
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.md").write_text("top secret outside content\n", encoding="utf-8")
    (corpus / "ok.md").write_text("inside content\n", encoding="utf-8")
    link = corpus / "lnk"
    try:
        link.symlink_to(outside, target_is_directory=True)
    except OSError:
        r = subprocess.run(["cmd", "/c", "mklink", "/J", str(link), str(outside)], capture_output=True)
        if r.returncode != 0:
            pytest.skip("cannot create symlink or junction")
    flink = corpus / "flnk.md"
    try:
        flink.symlink_to(outside / "secret.md")
    except OSError:
        pass
    rt.indexer.sync_root(corpus)
    paths = [s["path"] for s in rt.catalog.sources_under_root(corpus.resolve())]
    assert not any("secret.md" in p or "outside\\" in p for p in paths), paths
    assert not any("secret" in h.text for h in rt.search.hybrid_search("secret"))


QUERIES = [
    '"', '""', "'", "NEAR(", "NEAR(a b", "a OR", "OR", "AND NOT", "*", "a*", "(", ")", "((a)", "a)",
    "😀🎓", "col:val", "-a", "^a", "a b OR (c", "\\", "; DROP TABLE chunks;--", "\x00", "a\x00b", "{a}", "[a]",
    "title:", "  ", "", "\n\t", "x" * 200_000, "word " * 40_000,
]


@pytest.mark.parametrize(
    "q",
    QUERIES,
    ids=[f"q{i}" for i in range(len(QUERIES))],
)
def test_query_injection(env, q) -> None:
    rt, corpus = env
    (corpus / "a.md").write_text("alpha beta gamma delta\n", encoding="utf-8")
    rt.indexer.sync_root(corpus)
    t = time.time()
    rt.search.hybrid_search(q)
    assert time.time() - t < 9


def test_path_prefix_and_file_types_oddities(env, tmp_path: Path) -> None:
    rt, corpus = env
    (corpus / "a.md").write_text("alpha beta gamma\n", encoding="utf-8")
    rt.indexer.sync_root(corpus)
    # traversal prefix resolving outside the corpus must match nothing
    assert rt.search.hybrid_search("alpha", path_prefix=str(corpus / ".." / "elsewhere")) == []
    # traversal that resolves back inside is fine
    assert rt.search.hybrid_search("alpha", path_prefix=str(corpus / "sub" / ".." ))
    for pfx in ["../", "..\\..\\", "\x00", "C:", "\\\\?\\", "<>|", "~"]:
        rt.search.hybrid_search("alpha", path_prefix=pfx)
    for ft in [[], [""], ["."], ["MD"], [".MD"], ["md", ".md"], ["*"], ["\x00"], ["md; --"]]:
        rt.search.hybrid_search("alpha", file_types=ft)
    assert rt.search.hybrid_search("alpha", file_types=["MD"])
    assert rt.search.hybrid_search("alpha", file_types=["pdf"]) == []


def test_read_source_unindexed_path_raises(env, tmp_path: Path) -> None:
    rt, corpus = env
    (corpus / "a.md").write_text("alpha\n", encoding="utf-8")
    rt.indexer.sync_root(corpus)
    secret = tmp_path / "secret.md"
    secret.write_text("not indexed secret\n", encoding="utf-8")
    for p in [str(secret), str(corpus / ".." / "secret.md"), str(tmp_path / "nope.md"), "", "C:\\Windows\\win.ini"]:
        with pytest.raises((FileNotFoundError, ValueError, OSError)):
            rt.search.read_source(path=p)
    with pytest.raises((FileNotFoundError, ValueError)):
        rt.search.read_source(source_id="deadbeef")
    with pytest.raises((FileNotFoundError, ValueError)):
        rt.search.read_source()


def test_render_pdf_page_unindexed_raises(env, tmp_path: Path) -> None:
    rt, corpus = env
    fake = tmp_path / "x.pdf"
    fake.write_bytes(b"%PDF-1.4\n")
    for p in [str(fake), "C:\\Windows\\win.ini", ""]:
        with pytest.raises((FileNotFoundError, ValueError, OSError)):
            rt.search.render_pdf_page(path=p, page=1)
    (corpus / "a.md").write_text("alpha\n", encoding="utf-8")
    rt.indexer.sync_root(corpus)
    with pytest.raises(ValueError):
        rt.search.render_pdf_page(path=str(corpus / "a.md"), page=1)


def test_concurrent_index_same_and_different_files(env) -> None:
    rt, corpus = env
    files = []
    for i in range(4):
        f = corpus / f"f{i}.md"
        f.write_text(f"file number {i} content\n" * 60, encoding="utf-8")
        files.append(f)
    errs: list[BaseException] = []

    def run(f: Path) -> None:
        try:
            for _ in range(3):
                rt.indexer.index_file(f, corpus, force=True)
        except BaseException as e:  # noqa: BLE001
            errs.append(e)

    ts = [threading.Thread(target=run, args=(files[0],)) for _ in range(4)]
    ts += [threading.Thread(target=run, args=(f,)) for f in files]
    t0 = time.time()
    [t.start() for t in ts]
    [t.join(timeout=60) for t in ts]
    assert not errs, errs
    assert time.time() - t0 < 60
    assert consistent(rt)
    assert rt.status()["sources"] == 4 if "sources" in rt.status() else True


def test_delete_while_indexing(env) -> None:
    rt, corpus = env
    f = corpus / "gone.md"
    f.write_text("transient content\n" * 50, encoding="utf-8")
    started, release = threading.Event(), threading.Event()
    real = rt.vectors.embed

    def slow(chunks):
        started.set()
        release.wait(timeout=10)
        return real(chunks)

    rt.vectors.embed = slow  # type: ignore[method-assign]
    out: list = []
    th = threading.Thread(target=lambda: out.append(rt.indexer.index_file(f, corpus)))
    th.start()
    assert started.wait(timeout=10)
    f.unlink()
    release.set()
    th.join(timeout=20)
    rt.vectors.embed = real  # type: ignore[method-assign]
    # next event for the deleted file must clean up whatever was written
    rt.indexer.index_file(f, corpus)
    assert rt.catalog.source_by_path(f) is None
    assert consistent(rt)
    assert rt.status()["chunks"] == 0


def test_file_modified_between_stat_and_parse(env) -> None:
    rt, corpus = env
    f = corpus / "mod.md"
    f.write_text("original content here\n" * 20, encoding="utf-8")
    import study_retriever.indexer as ix

    real_parse = ix.parse_document
    calls = {"n": 0}

    def racing(path):
        calls["n"] += 1
        doc = real_parse(path)
        with open(path, "a", encoding="utf-8") as fh:
            fh.write("\nappended mid-index\n")
        os.utime(path, ns=(time.time_ns() + calls["n"] * 10**9,) * 2)
        return doc

    ix.parse_document = racing
    try:
        status, _ = rt.indexer.index_file(f, corpus)
    finally:
        ix.parse_document = real_parse
    assert status in {"error", "indexed"}
    assert consistent(rt)
    # a later stable event must converge
    assert rt.indexer.index_file(f, corpus)[0] in {"indexed", "unchanged"}
    assert consistent(rt)
    assert any("appended" in h.text for h in rt.search.hybrid_search("appended mid-index"))


def test_exclude_globs(tmp_path: Path) -> None:
    rt = make_runtime(tmp_path, exclude_globs=["*.draft.md", "private/*"])
    try:
        corpus = tmp_path / "corpus"
        (corpus / "private").mkdir(parents=True)
        (corpus / "a.draft.md").write_text("draft\n", encoding="utf-8")
        (corpus / "private" / "p.md").write_text("private\n", encoding="utf-8")
        (corpus / "A.DRAFT.MD").write_text("upper draft\n", encoding="utf-8")
        (corpus / "keep.md").write_text("keep\n", encoding="utf-8")
        assert rt.indexer.index_file(corpus / "a.draft.md", corpus)[0] == "excluded"
        assert rt.indexer.index_file(corpus / "private" / "p.md", corpus)[0] == "excluded"
        assert rt.indexer.index_file(corpus / "A.DRAFT.MD", corpus)[0] == "excluded"
        res = rt.indexer.sync_root(corpus)
        assert res.indexed == 1
    finally:
        rt.close()


def test_node_modules_event_excluded(env) -> None:
    rt, corpus = env
    d = corpus / "NODE_MODULES" / "x"
    d.mkdir(parents=True)
    f = d / "r.md"
    f.write_text("vendored\n", encoding="utf-8")
    assert rt.indexer.index_file(f, corpus)[0] == "excluded"
    assert rt.catalog.source_by_path(f) is None


def test_repair_vectors_idempotent(env) -> None:
    rt, corpus = env
    (corpus / "a.md").write_text("alpha beta gamma\n" * 100, encoding="utf-8")
    rt.indexer.sync_root(corpus)
    r1 = rt.indexer.repair_vectors()
    r2 = rt.indexer.repair_vectors()
    assert r1 == {"orphans_removed": 0, "vectors_restored": 0}
    assert r2 == r1
    # damage: drop some vectors, repair twice
    keys = sorted(rt.vectors.keys())[:3]
    rt.vectors.delete_keys(keys)
    assert not consistent(rt)
    assert rt.indexer.repair_vectors()["vectors_restored"] == 3
    assert rt.indexer.repair_vectors() == {"orphans_removed": 0, "vectors_restored": 0}
    assert consistent(rt)


@pytest.mark.parametrize(
    "text",
    ["   \n\t\n   ", "😀" * 5000, "a" * 200_000, "\u200b" * 1000, "x" * 50 + "\n" + "😀 " * 3000, "\n" * 10_000],
    ids=["ws", "emoji", "longword", "zwsp", "mixed", "newlines"],
)
def test_chunker_degenerate_text(env, text) -> None:
    rt, corpus = env
    f = corpus / "d.md"
    f.write_text(text, encoding="utf-8")
    t = time.time()
    status, n = rt.indexer.index_file(f, corpus)
    assert time.time() - t < 9
    assert status in {"indexed", "error"}
    assert consistent(rt)
    if status == "indexed":
        rows = list(rt.catalog.iter_chunks(1000))
        for batch in rows:
            for r in batch:
                assert len(r["text"]) < 20_000
