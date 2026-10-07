from __future__ import annotations

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _interface(plugin: dict, *, portable: bool) -> dict:
    if portable:
        return plugin["extensions"]["com.openai"]["interface"]
    return plugin["interface"]


def test_plugin_manifests_are_complete_and_local_stdio() -> None:
    from study_retriever import __version__

    plugin = json.loads((ROOT / "plugin.json").read_text(encoding="utf-8"))
    mcp = json.loads((ROOT / "mcp.json").read_text(encoding="utf-8"))
    compat = json.loads((ROOT / ".mcp.json").read_text(encoding="utf-8"))
    overlay = json.loads((ROOT / ".codex-plugin" / "plugin.json").read_text(encoding="utf-8"))

    assert plugin["$schema"] == "https://agent-plugins.org/schemas/1.0.0/plugin.schema.json"
    assert plugin["name"] == "study-retriever"
    assert plugin["version"] == __version__
    assert overlay["version"] == __version__
    assert plugin["author"]["name"]
    assert overlay["author"]["name"]
    assert overlay["mcpServers"] == "./.mcp.json"

    for manifest, portable in ((plugin, True), (overlay, False)):
        interface = _interface(manifest, portable=portable)
        assert 1 <= len(interface["displayName"]) <= 30
        assert 1 <= len(interface["shortDescription"]) <= 30
        assert 1 <= len(interface["longDescription"]) <= 4000
        assert 1 <= len(interface["developerName"]) <= 80
        assert 1 <= len(interface["capabilities"]) <= 20
        assert len(interface.get("defaultPrompt", [])) <= 3
        assert all(1 <= len(prompt) <= 128 and "\n" not in prompt for prompt in interface.get("defaultPrompt", []))

    portable_server = mcp["mcpServers"]["study_retriever"]
    assert portable_server["type"] == "stdio"
    assert portable_server["command"] == "powershell.exe"
    assert portable_server["args"][-1] == "${PLUGIN_ROOT}/scripts/mcp-launch.ps1"
    assert portable_server["cwd"] == "${PLUGIN_ROOT}"

    # Codex compatibility loader resolves relative cwd against installed plugin root.
    compat_server = compat["mcpServers"]["study_retriever"]
    assert compat_server["type"] == "stdio"
    assert compat_server["command"] == "powershell.exe"
    assert compat_server["args"][-1] == "scripts/mcp-launch.ps1"
    assert compat_server["cwd"] == "."

    claude = json.loads((ROOT / ".claude-plugin" / "plugin.json").read_text(encoding="utf-8"))
    claude_mcp = json.loads((ROOT / "claude.mcp.json").read_text(encoding="utf-8"))
    assert claude["version"] == __version__
    assert claude["mcpServers"] == "./claude.mcp.json"
    assert claude_mcp["mcpServers"]["study_retriever"]["args"][-1] == "${CLAUDE_PLUGIN_ROOT}/scripts/mcp-launch.ps1"

    assert (ROOT / "skills" / "study-retrieval" / "SKILL.md").is_file()
    assert (ROOT / "scripts" / "install.ps1").is_file()
    assert (ROOT / "scripts" / "mcp-launch.ps1").is_file()
    assert (ROOT / "scripts" / "indexer-launch.ps1").is_file()


def test_installer_is_self_contained_and_runs_full_live_doctor() -> None:
    script = (ROOT / "scripts" / "install.ps1").read_text(encoding="utf-8")
    project = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert 'requires-python = ">=3.11,<3.14"' in project
    assert '"-3.13", "-3.12", "-3.11"' in script
    assert '"-3.14"' not in script
    assert "function Test-CompatiblePython" in script
    assert '$ErrorActionPreference = "SilentlyContinue"' in script
    assert "Get-Command $name -All" in script
    assert "if (Test-CompatiblePython -Command $py.Source -Prefix @($version))" in script
    assert "if (-not (Test-CompatiblePython -Command $venvPython))" in script
    assert '< (3,14)' in script
    assert "doctor --full" in script
    assert "Study Retriever Background Indexer" in script
    assert "study_retriever.cli watch" in (ROOT / "scripts" / "indexer-launch.ps1").read_text(encoding="utf-8")
    assert 'HF_HUB_DOWNLOAD_TIMEOUT = "60"' in script
    assert 'HF_HUB_DISABLE_TELEMETRY = "1"' in script
    assert 'DO_NOT_TRACK = "1"' in script
    assert '--no-build-isolation' in script
    assert 'release.json' in script
    assert 'releaseMetadata.wheel.file' in script
    assert 'releaseMetadata.wheel.sha256' in script
    assert 'Bundled Study Retriever wheel SHA-256 mismatch' in script
    assert 'Installed Study Retriever wheel version does not match release.json' in script
    assert 'Plugin manifest version' in script
    assert '$launcher.Prefix[0] -m venv $venv' in script
    assert './.codex/plugins/study-retriever' in script
    assert 'installation = "INSTALLED_BY_DEFAULT"' in script
    assert "python-3.13.16-amd64.exe" in script
    assert "fb4f9f5d438b2396da0086dc70b935c530cb578e37adc6d354f7ad2037fee83b" in script
    assert "Python installer SHA-256 mismatch" in script
    assert "SkipModelWarmup" not in script
    for wrapper in ("install.cmd", "update.cmd", "manage.cmd", "verify.cmd", "uninstall.cmd"):
        assert (ROOT / wrapper).is_file()
    verify = (ROOT / "verify.cmd").read_text(encoding="utf-8")
    assert "doctor --full" in verify
    assert '--plugin-root "%~dp0"' in verify
    assert "Assert-PluginPackage" in script
    assert "Validating exact Desktop plugin package" in script
    assert "PYTHONPATH" not in (ROOT / "manage.cmd").read_text(encoding="utf-8")
    assert "PYTHONPATH" not in (ROOT / "verify.cmd").read_text(encoding="utf-8")
    assert "PYTHONPATH" not in (ROOT / "scripts" / "mcp-launch.ps1").read_text(encoding="utf-8")
    assert "PYTHONPATH" not in (ROOT / "scripts" / "indexer-launch.ps1").read_text(encoding="utf-8")
    uninstall = (ROOT / "scripts" / "uninstall.ps1").read_text(encoding="utf-8")
    assert ".codex\\plugins\\cache" in uninstall
    assert 'Join-Path $marketCache.FullName "study-retriever"' in uninstall
    assert "runtime" in uninstall
    assert "Study Retriever Background Indexer" in uninstall


def test_company_knowledge_search_fetch_contracts_are_exact() -> None:
    import ast

    from study_retriever.api_models import FetchResponse, SearchResponse, SearchResult

    tree = ast.parse((ROOT / "src" / "study_retriever" / "mcp_server.py").read_text(encoding="utf-8"))
    functions = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
    assert [arg.arg for arg in functions["search"].args.args] == ["query"]
    assert [arg.arg for arg in functions["fetch"].args.args] == ["id"]
    assert set(SearchResult.model_fields) == {"id", "title", "url"}
    assert set(SearchResponse.model_fields) == {"results"}
    assert set(FetchResponse.model_fields) == {"id", "title", "text", "url", "metadata"}


def test_release_metadata_and_update_pipeline_are_consistent() -> None:
    import hashlib

    from study_retriever import __version__

    if not (ROOT / "release.json").is_file():
        pytest.skip("release.json is generated by scripts/build_release.py")
    release = json.loads((ROOT / "release.json").read_text(encoding="utf-8"))
    assert release["schema"] == 1
    assert release["name"] == "study-retriever"
    assert release["version"] == __version__
    wheel = ROOT / release["wheel"]["file"]
    if wheel.is_file():
        assert hashlib.sha256(wheel.read_bytes()).hexdigest() == release["wheel"]["sha256"]

    updater = (ROOT / "scripts" / "update.ps1").read_text(encoding="utf-8")
    assert "Another Study Retriever update is already running" in updater
    assert "Test-SafeZip" in updater
    assert "Update ZIP SHA-256 mismatch" in updater
    assert "Creating rollback copy" in updater
    assert "previous release was restored" in updater
    assert "update-in-progress" in updater
    assert "Restart ChatGPT Desktop" in updater
    assert "-SkipMarketplace" in updater

    launcher = (ROOT / "scripts" / "mcp-launch.ps1").read_text(encoding="utf-8")
    assert "update-in-progress" in launcher
    assert "exit 75" in launcher

    build = (ROOT / "scripts" / "build_release.py").read_text(encoding="utf-8")
    assert "latest.json" in build
    assert "Study-Retriever-v" in build


def test_package_doctor_validates_exact_desktop_manifests() -> None:
    from study_retriever.cli import _validate_plugin_package

    result = _validate_plugin_package(ROOT)
    assert result["portable_manifest"] is True
    assert result["portable_mcp"] is True
    assert result["compat_manifest"] is True
    assert result["launcher_present"] is True


def test_desktop_doctor_checks_marketplace_source_cache_and_config(tmp_path, monkeypatch) -> None:
    import shutil
    from types import SimpleNamespace

    from study_retriever.cli import cmd_desktop_doctor

    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))

    def copy_minimal(dst: Path) -> None:
        for rel in (
            "plugin.json",
            "mcp.json",
            ".mcp.json",
            ".codex-plugin/plugin.json",
            "scripts/mcp-launch.ps1",
        ):
            target = dst / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / rel, target)

    source = tmp_path / ".codex" / "plugins" / "study-retriever"
    copy_minimal(source)
    cache = tmp_path / ".codex" / "plugins" / "cache" / "plugins-cli" / "study-retriever" / "local"
    copy_minimal(cache)

    market = tmp_path / ".agents" / "plugins" / "marketplace.json"
    market.parent.mkdir(parents=True)
    market.write_text(json.dumps({
        "name": "plugins-cli",
        "plugins": [{
            "name": "study-retriever",
            "source": {"source": "local", "path": "./.codex/plugins/study-retriever"},
            "policy": {"installation": "INSTALLED_BY_DEFAULT", "authentication": "ON_INSTALL"},
            "category": "Productivity",
        }],
    }), encoding="utf-8")
    config = tmp_path / ".codex" / "config.toml"
    config.write_text(
        '[plugins."study-retriever@plugins-cli"]\n'
        'enabled = true\n'
        '[plugins."study-retriever@plugins-cli".mcp_servers.study_retriever]\n'
        'enabled = true\n',
        encoding="utf-8",
    )

    assert cmd_desktop_doctor(SimpleNamespace()) == 0
