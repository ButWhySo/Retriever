from __future__ import annotations

import argparse
import asyncio
import json
import os
import signal
import statistics
import sys
import tempfile
import threading
import time
from pathlib import Path

from .config import DEFAULT_MODEL, Paths, load_config, save_config
from .runtime import Runtime, lower_process_priority


def _print(value) -> None:
    sys.stdout.buffer.write((json.dumps(value, indent=2, ensure_ascii=False, default=str) + "\n").encode("utf-8"))
    sys.stdout.buffer.flush()


def _runtime(args) -> Runtime:
    home = Path(args.home).expanduser().resolve() if getattr(args, "home", None) else None
    return Runtime(home=home, start_watcher=False)


def _validate_plugin_package(root: Path) -> dict[str, object]:
    root = root.resolve()
    required = [
        root / "plugin.json",
        root / "mcp.json",
        root / ".codex-plugin" / "plugin.json",
        root / ".mcp.json",
        root / "scripts" / "mcp-launch.ps1",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError(f"plugin package files missing: {missing}")

    plugin = json.loads((root / "plugin.json").read_text(encoding="utf-8"))
    portable = json.loads((root / "mcp.json").read_text(encoding="utf-8"))
    compat = json.loads((root / ".mcp.json").read_text(encoding="utf-8"))
    if plugin.get("$schema") != "https://agent-plugins.org/schemas/1.0.0/plugin.schema.json":
        raise RuntimeError("portable plugin.json schema is invalid")
    if portable.get("$schema") != "https://agent-plugins.org/schemas/1.0.0/mcp.schema.json":
        raise RuntimeError("portable mcp.json schema is invalid")

    server = portable.get("mcpServers", {}).get("study_retriever", {})
    if server.get("type") != "stdio" or server.get("command") != "powershell.exe":
        raise RuntimeError("portable Study Retriever MCP server declaration is invalid")
    cwd = server.get("cwd")
    valid_cwd = (
        cwd is None
        or cwd == "${PLUGIN_ROOT}"
        or (isinstance(cwd, str) and (cwd.startswith("./") or cwd.startswith("${PLUGIN_ROOT}/") or cwd.startswith("${PLUGIN_DATA}/")))
    )
    if not valid_cwd:
        raise RuntimeError(f"portable MCP cwd is invalid for Agent Plugins: {cwd!r}")
    args = server.get("args") or []
    if not args or args[-1] != "${PLUGIN_ROOT}/scripts/mcp-launch.ps1":
        raise RuntimeError("portable MCP launch script must use ${PLUGIN_ROOT}")

    compat_server = compat.get("mcpServers", {}).get("study_retriever", {})
    if compat_server.get("command") != "powershell.exe":
        raise RuntimeError("Codex compatibility MCP server declaration is invalid")
    if compat_server.get("cwd") != ".":
        raise RuntimeError("Codex compatibility MCP cwd must remain plugin-root relative")
    return {
        "root": str(root),
        "portable_manifest": True,
        "portable_mcp": True,
        "compat_manifest": True,
        "launcher_present": True,
    }


def cmd_add_root(args) -> int:
    rt = _runtime(args)
    try:
        _print(rt.add_root(args.path, sync_now=not args.no_sync))
    finally:
        rt.close()
    return 0


def cmd_remove_root(args) -> int:
    rt = _runtime(args)
    try:
        _print(rt.remove_root(args.path, delete_indexed=not args.keep_index))
    finally:
        rt.close()
    return 0


def cmd_sync(args) -> int:
    lower_process_priority()
    rt = _runtime(args)
    try:
        _print(rt.indexer.sync_all(force=args.force))
    finally:
        rt.close()
    return 0


def cmd_status(args) -> int:
    rt = _runtime(args)
    try:
        _print(rt.status())
    finally:
        rt.close()
    return 0


def cmd_watch(args) -> int:
    """Keep configured roots synchronized continuously until terminated."""
    lower_process_priority()
    home = Path(args.home).expanduser().resolve() if getattr(args, "home", None) else None
    rt = Runtime(home=home, start_watcher=True, reconcile=False)
    stop = threading.Event()

    def request_stop(_signum=None, _frame=None):
        stop.set()

    for sig_name in ("SIGINT", "SIGTERM", "SIGBREAK"):
        sig = getattr(signal, sig_name, None)
        if sig is not None:
            try:
                signal.signal(sig, request_stop)
            except (ValueError, OSError):
                pass

    try:
        sync = rt.indexer.sync_all()
        _print({"watching": True, "sync": sync, "status": rt.status()})
        while not stop.wait(1.0):
            pass
    finally:
        rt.close()
    return 0


def cmd_search(args) -> int:
    rt = _runtime(args)
    try:
        hits = rt.search.hybrid_search(args.query, top_k=args.top_k, path_prefix=args.path_prefix)
        _print([
            {
                "id": h.chunk_id, "score": h.score, "title": h.title, "citation": h.citation(),
                "path": h.path, "url": h.source_uri, "dense_rank": h.dense_rank,
                "lexical_rank": h.lexical_rank, "text": h.text,
            }
            for h in hits
        ])
    finally:
        rt.close()
    return 0


def cmd_topic(args) -> int:
    rt = _runtime(args)
    try:
        _print(rt.search.topic_bundle(args.query, max_chunks=args.max_chunks, max_chars=args.max_chars))
    finally:
        rt.close()
    return 0


def cmd_rebuild(args) -> int:
    rt = _runtime(args)
    try:
        _print(rt.indexer.rebuild_vectors())
    finally:
        rt.close()
    return 0


def _run_live_self_test(rt: Runtime) -> dict[str, object]:
    marker = "studyretriever-live-self-test-9f51"
    with tempfile.TemporaryDirectory(prefix="study-retriever-self-test-") as td:
        source = Path(td) / "retrieval-self-test.md"
        source.write_text(
            "# Hybrid Retrieval Self Test\n\n"
            f"{marker} combines semantic vector retrieval with exact lexical search and source provenance.\n",
            encoding="utf-8",
        )
        try:
            indexed = rt.add_root(str(source), sync_now=True)
            hits = rt.search.hybrid_search("semantic vector retrieval lexical provenance", top_k=10)
            hit = next((h for h in hits if Path(h.path) == source.resolve()), None)
            if hit is None:
                raise RuntimeError("self-test document was indexed but not retrieved")
            if hit.dense_rank is None:
                raise RuntimeError("dense retrieval did not return the self-test document")
            if hit.lexical_rank is None:
                raise RuntimeError("lexical retrieval did not return the self-test document")
            fetched = rt.search.fetch(hit.chunk_id)
            if not fetched or marker not in fetched["content"]:
                raise RuntimeError("fetch did not return indexed self-test content")
            result = {
                "indexed": indexed["sync"],
                "dense_rank": hit.dense_rank,
                "lexical_rank": hit.lexical_rank,
                "fetch_verified": True,
            }
        finally:
            rt.remove_root(str(source), delete_indexed=True)
    after = rt.status()
    if any(Path(root).resolve() == source.resolve() for root in after["roots"]):
        raise RuntimeError("self-test cleanup left its temporary root configured")
    if rt.catalog.sources_under_root(source):
        raise RuntimeError("self-test cleanup left temporary source entries")
    if not after["vector_count_matches_chunks"]:
        raise RuntimeError("self-test cleanup left catalog/vector counts inconsistent")
    return result


async def _mcp_wire_self_test_async(home: Path | None) -> dict[str, object]:
    """Exercise the installed MCP server through a real stdio subprocess."""
    from mcp import Client, StdioServerParameters

    data_home = Paths(home).home
    env = {"STUDY_RETRIEVER_HOME": str(data_home), "STUDY_RETRIEVER_ALLOW_MCP_ROOT_CHANGES": "1"}
    if os.environ.get("PYTHONPATH"):
        env["PYTHONPATH"] = os.environ["PYTHONPATH"]
    params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "study_retriever.mcp_server"],
        env=env,
    )
    required = {
        "search", "fetch", "search_advanced", "fetch_context", "curate_topic", "read_source",
        "render_pdf_page", "index_status", "sync_index", "add_study_root",
        "remove_study_root", "rebuild_vector_index",
    }
    marker = "studyretriever-mcp-wire-test-64fd2a"
    with tempfile.TemporaryDirectory(prefix="study-retriever-mcp-wire-") as td:
        source = Path(td) / "mcp-wire-test.md"
        source.write_text(
            "# MCP Wire Test\n\n"
            f"{marker} validates stdio MCP discovery, neural retrieval, structured output, fetch, and cleanup. "
            "Scaled dot product attention maps queries to keys and values for semantic retrieval.\n",
            encoding="utf-8",
        )
        added = False
        try:
            async with Client(params, read_timeout_seconds=90.0) as client:
                listed = await client.list_tools()
                names = {tool.name for tool in listed.tools}
                missing = sorted(required - names)
                if missing:
                    raise RuntimeError(f"MCP server missing tools: {missing}")
                for tool in listed.tools:
                    if tool.name in required - {"render_pdf_page"} and tool.output_schema is None:
                        raise RuntimeError(f"MCP tool lacks structured output schema: {tool.name}")

                add_result = await client.call_tool("add_study_root", {"path": str(source), "sync_now": True})
                if add_result.is_error:
                    raise RuntimeError(f"MCP add_study_root failed: {add_result.content}")
                added = True

                search_result = await client.call_tool("search", {"query": marker})
                if search_result.is_error or not search_result.structured_content:
                    raise RuntimeError(f"MCP search failed: {search_result.content}")
                results = search_result.structured_content.get("results", [])
                standard_hit = None
                fetch_result = None
                for candidate in results:
                    if set(candidate) != {"id", "title", "url"}:
                        raise RuntimeError(
                            f"MCP standard search result shape is incompatible: {sorted(candidate)}"
                        )
                    candidate_fetch = await client.call_tool("fetch", {"id": candidate["id"]})
                    if candidate_fetch.is_error or not candidate_fetch.structured_content:
                        continue
                    metadata = candidate_fetch.structured_content.get("metadata") or {}
                    if Path(metadata.get("path", "")) == source.resolve():
                        standard_hit = candidate
                        fetch_result = candidate_fetch
                        break
                if standard_hit is None or fetch_result is None:
                    raise RuntimeError("MCP standard search/fetch did not resolve the indexed test document")
                fetched = fetch_result.structured_content
                if marker not in fetched.get("text", ""):
                    raise RuntimeError("MCP standard fetch returned wrong content")
                if not fetched.get("url") or not fetched.get("metadata"):
                    raise RuntimeError("MCP standard fetch lacks citation URL or provenance metadata")

                advanced_result = await client.call_tool(
                    "search_advanced",
                    {"query": "scaled dot product attention queries keys values", "top_k": 10,
                     "path_prefix": str(source.parent)},
                )
                if advanced_result.is_error or not advanced_result.structured_content:
                    raise RuntimeError(f"MCP advanced search failed: {advanced_result.content}")
                detailed = advanced_result.structured_content.get("results", [])
                detail_hit = next((x for x in detailed if Path(x.get("path", "")) == source.resolve()), None)
                if detail_hit is None:
                    raise RuntimeError("MCP advanced search did not retrieve the document indexed through MCP")
                if detail_hit.get("dense_rank") is None or detail_hit.get("lexical_rank") is None:
                    raise RuntimeError(
                        "MCP advanced search did not verify both dense and lexical retrieval arms: "
                        f"dense_rank={detail_hit.get('dense_rank')}, lexical_rank={detail_hit.get('lexical_rank')}"
                    )

                context_result = await client.call_tool(
                    "fetch_context", {"id": detail_hit["id"], "neighbor_radius": 1}
                )
                if context_result.is_error or not context_result.structured_content:
                    raise RuntimeError(f"MCP fetch_context failed: {context_result.content}")
                if marker not in context_result.structured_content.get("content", ""):
                    raise RuntimeError("MCP fetch_context returned wrong content")

                status_result = await client.call_tool("index_status", {})
                if status_result.is_error or not status_result.structured_content:
                    raise RuntimeError(f"MCP index_status failed: {status_result.content}")
                if not status_result.structured_content.get("vector_count_matches_chunks"):
                    raise RuntimeError("MCP status reports vector/catalog count mismatch")

                remove_result = await client.call_tool(
                    "remove_study_root", {"path": str(source), "delete_indexed": True}
                )
                if remove_result.is_error:
                    raise RuntimeError(f"MCP remove_study_root failed: {remove_result.content}")
                added = False

                return {
                    "protocol_version": str(client.protocol_version),
                    "server": client.server_info.name if client.server_info else "Study Retriever",
                    "tool_count": len(listed.tools),
                    "structured_output": True,
                    "dense_and_lexical_search": True,
                    "fetch_verified": True,
                    "cleanup_verified": True,
                }
        finally:
            if added:
                # Best-effort cleanup after a mid-test MCP failure. Original study files are untouched.
                cleanup = Runtime(home=data_home, start_watcher=False)
                try:
                    cleanup.remove_root(str(source), delete_indexed=True)
                finally:
                    cleanup.close()


def _run_mcp_wire_self_test(home: Path | None) -> dict[str, object]:
    return asyncio.run(_mcp_wire_self_test_async(home))


def _run_watcher_self_test(home: Path | None) -> dict[str, object]:
    """Verify real watchdog events create, modify, and delete indexed content."""
    data_home = Paths(home).home
    marker_a = "studyretriever-watcher-create-2b6f"
    marker_b = "studyretriever-watcher-modify-4d19"
    with tempfile.TemporaryDirectory(prefix="study-retriever-watcher-") as td:
        root = Path(td).resolve()
        source = root / "watcher.md"
        rt = Runtime(home=data_home, start_watcher=True)
        added = False
        try:
            rt.add_root(str(root), sync_now=True)
            added = True
            if rt.watcher._observer is None:
                raise RuntimeError("filesystem watcher did not start")

            source.write_text(f"# Watcher Test\n\n{marker_a}\n", encoding="utf-8")
            deadline = time.monotonic() + 20.0
            while time.monotonic() < deadline:
                row = rt.catalog.source_by_path(source)
                if row and row.get("status") == "ready":
                    break
                time.sleep(0.1)
            else:
                raise RuntimeError("watcher did not index a newly created file")

            before = rt.catalog.source_by_path(source)
            source.write_text(f"# Watcher Test\n\n{marker_b}\n", encoding="utf-8")
            deadline = time.monotonic() + 20.0
            while time.monotonic() < deadline:
                row = rt.catalog.source_by_path(source)
                if row and before and row.get("sha256") != before.get("sha256"):
                    break
                time.sleep(0.1)
            else:
                raise RuntimeError("watcher did not re-index a modified file")

            hits = rt.search.hybrid_search(marker_b, top_k=10)
            if not any(Path(h.path) == source for h in hits):
                raise RuntimeError("watcher-updated content was not searchable")

            source.unlink()
            deadline = time.monotonic() + 20.0
            while time.monotonic() < deadline:
                if rt.catalog.source_by_path(source) is None:
                    break
                time.sleep(0.1)
            else:
                raise RuntimeError("watcher did not remove a deleted file from the index")

            return {
                "started": True,
                "create_event": True,
                "modify_event": True,
                "delete_event": True,
            }
        finally:
            if added:
                rt.remove_root(str(root), delete_indexed=True)
            rt.close()


def cmd_doctor(args) -> int:
    checks: dict[str, object] = {}
    home = Path(args.home).expanduser().resolve() if args.home else None
    rt: Runtime | None = None
    full = bool(getattr(args, "full", False))
    run_mcp = bool(args.mcp or args.wire or full)
    run_warmup = bool(args.warmup or full)
    run_self_test = bool(args.self_test or full)
    run_wire = bool(args.wire or full)
    run_watcher = bool(getattr(args, "watcher", False) or full)
    try:
        if getattr(args, "plugin_root", None):
            checks["plugin_package"] = _validate_plugin_package(Path(args.plugin_root))
        if run_mcp:
            from mcp.server import MCPServer  # noqa: F401

            from . import mcp_server  # noqa: F401
            checks["mcp_import"] = True
        rt = Runtime(home=home, start_watcher=False)
        checks["catalog"] = rt.catalog.stats()
        checks["vector_backend"] = rt.vectors.description
        checks["vector_count"] = rt.vectors.count()
        checks["vector_count_matches_chunks"] = rt.vectors.count() == rt.catalog.stats()["chunks"]
        if run_warmup:
            start = time.perf_counter()
            rt.vectors.warmup()
            checks["embedding_warmup_ms"] = round((time.perf_counter() - start) * 1000, 2)
        if run_self_test:
            checks["live_self_test"] = _run_live_self_test(rt)
        if rt is not None:
            rt.close()
            rt = None
        if run_watcher:
            checks["watcher_self_test"] = _run_watcher_self_test(home)
        if run_wire:
            checks["mcp_stdio_wire_test"] = _run_mcp_wire_self_test(home)
        checks["ok"] = bool(checks["vector_count_matches_chunks"])
    except Exception as exc:
        checks["ok"] = False
        checks["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        if rt is not None:
            rt.close()
    _print(checks)
    return 0 if checks.get("ok") else 2

def cmd_desktop_doctor(args) -> int:
    home = Path.home()
    market_file = home / ".agents" / "plugins" / "marketplace.json"
    config_file = home / ".codex" / "config.toml"
    result: dict[str, object] = {
        "marketplace_file": str(market_file),
        "config_file": str(config_file),
    }
    try:
        if not market_file.is_file():
            raise RuntimeError(f"personal marketplace is missing: {market_file}")
        market = json.loads(market_file.read_text(encoding="utf-8-sig"))
        market_name = str(market.get("name") or "")
        entry = next((p for p in market.get("plugins", []) if p.get("name") == "study-retriever"), None)
        if not entry:
            raise RuntimeError("study-retriever is not registered in personal marketplace")
        source_value = (entry.get("source") or {}).get("path")
        if not isinstance(source_value, str) or not source_value.startswith("./"):
            raise RuntimeError(f"Study Retriever marketplace source.path is invalid: {source_value!r}")
        source_root = (home / source_value[2:]).resolve()
        result["marketplace"] = market_name
        result["source_root"] = str(source_root)
        result["source_package"] = _validate_plugin_package(source_root)

        plugin_key = f"study-retriever@{market_name}"
        config_state: dict[str, object] = {"plugin_key": plugin_key, "present": config_file.is_file()}
        if config_file.is_file():
            import tomllib
            cfg = tomllib.loads(config_file.read_text(encoding="utf-8-sig"))
            plugin_cfg = (cfg.get("plugins") or {}).get(plugin_key)
            if isinstance(plugin_cfg, dict):
                config_state["enabled"] = plugin_cfg.get("enabled")
                mcp_cfg = (plugin_cfg.get("mcp_servers") or {}).get("study_retriever")
                if isinstance(mcp_cfg, dict):
                    config_state["mcp_enabled"] = mcp_cfg.get("enabled")
        result["config"] = config_state

        cache_base = home / ".codex" / "plugins" / "cache" / market_name / "study-retriever"
        cache_packages = []
        if cache_base.is_dir():
            candidates = [p.parent for p in cache_base.rglob("plugin.json") if p.parent.name != ".codex-plugin"]
            for candidate in sorted(set(candidates)):
                try:
                    cache_packages.append({"path": str(candidate), "valid": True, "details": _validate_plugin_package(candidate)})
                except Exception as exc:
                    cache_packages.append({"path": str(candidate), "valid": False, "error": f"{type(exc).__name__}: {exc}"})
        result["cache_root"] = str(cache_base)
        result["cache_packages"] = cache_packages
        result["cache_materialized"] = bool(cache_packages)
        result["ok"] = bool(cache_packages and all(item["valid"] for item in cache_packages))
        if not cache_packages:
            result["next"] = "Restart ChatGPT Desktop, open Plugins, refresh/install Study Retriever, then rerun desktop-doctor."
    except Exception as exc:
        result["ok"] = False
        result["error"] = f"{type(exc).__name__}: {exc}"
    _print(result)
    return 0 if result.get("ok") else 2


def cmd_set_model(args) -> int:
    paths = Paths(Path(args.home).expanduser().resolve() if args.home else None)
    cfg = load_config(paths)
    cfg.model_name = args.model
    save_config(paths, cfg)
    _print({"model": cfg.model_name, "next": "Run rebuild-vectors to rebuild dense vectors with this model."})
    return 0


def cmd_benchmark_models(args) -> int:
    try:
        from fastembed import TextEmbedding
    except ImportError as exc:
        raise SystemExit(f"fastembed is not installed: {exc}")
    paths = Paths(Path(args.home).expanduser().resolve() if args.home else None)
    candidates = args.models or [
        "snowflake/snowflake-arctic-embed-xs",
        "BAAI/bge-small-en-v1.5",
        "sentence-transformers/all-MiniLM-L6-v2",
    ]
    sample_docs = [
        "Scaled dot-product attention computes similarities between queries and keys, scales by square root of key dimension, applies softmax, then weights values.",
        "Virtual memory translates virtual addresses to physical addresses using page tables and a translation lookaside buffer.",
        "Kruskal's algorithm sorts graph edges by weight and repeatedly adds edges that do not create a cycle.",
        "Maximum likelihood estimation chooses parameters that maximize the probability of observed training data.",
        "Gradient descent updates parameters opposite the gradient of the objective function using a chosen step size.",
    ] * max(1, args.repeats)
    queries = ["why divide attention scores by sqrt dk", "TLB address translation", "Kruskal minimum spanning tree"]
    results = []
    for name in candidates:
        model = TextEmbedding(name, cache_dir=str(paths.model_cache), threads=min(2, max(1, (os.cpu_count() or 2) - 1)))
        list(model.passage_embed(sample_docs[:2], batch_size=2))
        start = time.perf_counter()
        vectors = list(model.passage_embed(sample_docs, batch_size=min(128, len(sample_docs))))
        elapsed = time.perf_counter() - start
        latencies = []
        for query in queries:
            q0 = time.perf_counter()
            next(iter(model.query_embed(query)))
            latencies.append((time.perf_counter() - q0) * 1000)
        results.append({
            "model": name,
            "dimension": len(vectors[0]),
            "documents": len(sample_docs),
            "documents_per_second": round(len(sample_docs) / elapsed, 2),
            "query_median_ms": round(statistics.median(latencies), 2),
            "query_max_ms": round(max(latencies), 2),
        })
    results.sort(key=lambda r: (r["query_median_ms"], -r["documents_per_second"]))
    _print({"results": results, "current_default": DEFAULT_MODEL})
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="study-retriever", description="Local hybrid study-material retrieval")
    p.add_argument("--home", help="Override persistent data directory")
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("add-root", help="Add and index a study folder or file")
    s.add_argument("path")
    s.add_argument("--no-sync", action="store_true")
    s.set_defaults(func=cmd_add_root)

    s = sub.add_parser("remove-root", help="Stop indexing a study root")
    s.add_argument("path")
    s.add_argument("--keep-index", action="store_true")
    s.set_defaults(func=cmd_remove_root)

    s = sub.add_parser("sync", help="Incrementally synchronize all roots")
    s.add_argument("--force", action="store_true", help="Reparse/re-embed every supported file")
    s.set_defaults(func=cmd_sync)

    s = sub.add_parser("status", help="Show index status")
    s.set_defaults(func=cmd_status)

    s = sub.add_parser("watch", help="Continuously watch configured roots and index changes")
    s.set_defaults(func=cmd_watch)

    s = sub.add_parser("search", help="Run hybrid semantic + BM25 search")
    s.add_argument("query")
    s.add_argument("-k", "--top-k", type=int, default=15)
    s.add_argument("--path-prefix")
    s.set_defaults(func=cmd_search)

    s = sub.add_parser("topic", help="Retrieve a source-grounded topic bundle")
    s.add_argument("query")
    s.add_argument("--max-chunks", type=int, default=30)
    s.add_argument("--max-chars", type=int, default=45_000)
    s.set_defaults(func=cmd_topic)

    s = sub.add_parser("rebuild-vectors", help="Rebuild vector index from catalog chunks")
    s.set_defaults(func=cmd_rebuild)

    s = sub.add_parser("doctor", help="Verify catalog, vector model, live retrieval, and a real stdio MCP subprocess")
    s.add_argument("--warmup", action="store_true", help="Load the configured neural embedding model and run one query embedding")
    s.add_argument("--mcp", action="store_true", help="Import the MCP SDK and registered Study Retriever server")
    s.add_argument("--self-test", action="store_true", help="Index, retrieve, fetch, and clean up a real temporary document")
    s.add_argument("--wire", action="store_true", help="Launch the MCP server as a stdio subprocess and exercise real tool calls")
    s.add_argument("--watcher", action="store_true", help="Verify real filesystem create/modify/delete events update the index")
    s.add_argument("--full", action="store_true", help="Run warmup, retrieval, watcher, MCP import, and real stdio MCP wire tests")
    s.add_argument("--plugin-root", help="Validate the exact Desktop plugin package at this path")
    s.set_defaults(func=cmd_doctor)

    s = sub.add_parser("desktop-doctor", help="Inspect personal marketplace, Desktop cache, and plugin enablement")
    s.set_defaults(func=cmd_desktop_doctor)

    s = sub.add_parser("set-model", help="Set FastEmbed model; rebuild vectors afterward")
    s.add_argument("model")
    s.set_defaults(func=cmd_set_model)

    s = sub.add_parser("benchmark-models", help="Benchmark lightweight FastEmbed models on this CPU")
    s.add_argument("--models", nargs="*")
    s.add_argument("--repeats", type=int, default=20)
    s.set_defaults(func=cmd_benchmark_models)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
