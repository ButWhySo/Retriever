from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DIST = ROOT / "dist"
WHEELHOUSE = ROOT / "wheelhouse"

EXCLUDED_DIRS = {
    ".git",
    ".venv",
    ".pytest_cache",
    "__pycache__",
    "build",
    "dist",
    ".claude",
    ".swarm",
    ".claude-flow",
    ".agents",
    ".benchmarks",
}
EXCLUDED_FILES = {"ruvector.db", "CLAUDE.md"}
EXCLUDED_SUFFIXES = {".pyc", ".pyo"}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def version() -> str:
    text = (ROOT / "src" / "study_retriever" / "__init__.py").read_text(encoding="utf-8")
    match = re.search(r'^__version__\s*=\s*["\']([^"\']+)["\']', text, re.MULTILINE)
    if not match:
        raise RuntimeError("Could not read __version__")
    value = match.group(1)
    if not re.fullmatch(r"\d+\.\d+\.\d+", value):
        raise RuntimeError(f"Release version must be stable semver X.Y.Z, got {value!r}")
    return value


def sync_manifest_versions(ver: str) -> None:
    for rel in ("plugin.json", ".codex-plugin/plugin.json", ".claude-plugin/plugin.json"):
        path = ROOT / rel
        obj = json.loads(path.read_text(encoding="utf-8"))
        obj["version"] = ver
        path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def run(*args: str) -> None:
    subprocess.run(args, cwd=ROOT, check=True)


def build_wheel(ver: str) -> Path:
    WHEELHOUSE.mkdir(parents=True, exist_ok=True)
    for old in WHEELHOUSE.glob(f"study_retriever-{ver}-*.whl"):
        old.unlink()
    run(
        sys.executable,
        "-m",
        "pip",
        "wheel",
        ".",
        "--no-deps",
        "--no-build-isolation",
        "--wheel-dir",
        str(WHEELHOUSE),
    )
    wheels = list(WHEELHOUSE.glob(f"study_retriever-{ver}-py3-none-any.whl"))
    if len(wheels) != 1:
        raise RuntimeError(f"Expected exactly one Study Retriever {ver} wheel, got {wheels}")
    return wheels[0]


def write_release_metadata(ver: str, wheel: Path) -> Path:
    metadata = {
        "schema": 1,
        "name": "study-retriever",
        "version": ver,
        "channel": "stable",
        "python": {"min": "3.11", "max_exclusive": "3.14"},
        "wheel": {
            "file": f"wheelhouse/{wheel.name}",
            "sha256": sha256(wheel),
        },
    }
    path = ROOT / "release.json"
    path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return path


def should_package(path: Path) -> bool:
    rel = path.relative_to(ROOT)
    if any(part in EXCLUDED_DIRS or part.endswith(".egg-info") for part in rel.parts):
        return False
    if path.suffix in EXCLUDED_SUFFIXES or path.name in EXCLUDED_FILES:
        return False
    return path.is_file()


def build_zip(ver: str) -> tuple[Path, Path, Path]:
    DIST.mkdir(parents=True, exist_ok=True)
    archive = DIST / f"Study-Retriever-v{ver}-Windows.zip"
    checksum = archive.with_suffix(archive.suffix + ".sha256")
    archive.unlink(missing_ok=True)
    checksum.unlink(missing_ok=True)
    files = sorted(p for p in ROOT.rglob("*") if should_package(p))
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        for path in files:
            rel = Path("study-retriever") / path.relative_to(ROOT)
            zf.write(path, rel.as_posix())
    digest = sha256(archive)
    checksum.write_text(f"{digest}  {archive.name}\n", encoding="ascii")
    feed = DIST / "latest.json"
    feed.write_text(json.dumps({
        "name": "study-retriever",
        "version": ver,
        "url": archive.name,
        "sha256": digest,
    }, indent=2) + "\n", encoding="utf-8")
    return archive, checksum, feed


def verify_zip(archive: Path, ver: str) -> None:
    with zipfile.ZipFile(archive) as zf:
        bad = zf.testzip()
        if bad:
            raise RuntimeError(f"ZIP CRC verification failed at {bad}")
        names = set(zf.namelist())
        required = {
            "study-retriever/release.json",
            "study-retriever/install.cmd",
            "study-retriever/update.cmd",
            "study-retriever/scripts/install.ps1",
            "study-retriever/scripts/update.ps1",
            "study-retriever/.codex-plugin/plugin.json",
            "study-retriever/.claude-plugin/plugin.json",
            "study-retriever/.claude-plugin/marketplace.json",
            "study-retriever/claude.mcp.json",
            "study-retriever/mcp.json",
        }
        missing = sorted(required - names)
        if missing:
            raise RuntimeError(f"Release ZIP missing required files: {missing}")
        release = json.loads(zf.read("study-retriever/release.json"))
        if release.get("version") != ver:
            raise RuntimeError("ZIP release.json version mismatch")


def main() -> int:
    parser = argparse.ArgumentParser(description="Build and validate a Study Retriever Windows release")
    parser.add_argument("--skip-tests", action="store_true")
    args = parser.parse_args()

    ver = version()
    sync_manifest_versions(ver)
    wheel = build_wheel(ver)
    write_release_metadata(ver, wheel)

    if not args.skip_tests:
        run(sys.executable, "-m", "compileall", "-q", "src", "tests", "scripts")
        run(sys.executable, "-m", "pytest", "-q")

    archive, checksum, feed = build_zip(ver)
    verify_zip(archive, ver)
    print(json.dumps({
        "version": ver,
        "wheel": str(wheel),
        "wheel_sha256": sha256(wheel),
        "archive": str(archive),
        "archive_sha256": sha256(archive),
        "checksum": str(checksum),
        "update_feed": str(feed),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
