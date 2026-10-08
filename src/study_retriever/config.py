from __future__ import annotations

import json
import os
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

DEFAULT_MODEL = "snowflake/snowflake-arctic-embed-xs"
SUPPORTED_TEXT_SUFFIXES = {
    ".txt", ".md", ".markdown", ".rst", ".tex", ".py", ".c", ".h", ".cpp", ".hpp",
    ".cc", ".java", ".kt", ".kts", ".go", ".rs", ".js", ".jsx", ".ts", ".tsx", ".css",
    ".scss", ".sql", ".sh", ".bash", ".zsh", ".ps1", ".bat", ".cmd", ".toml", ".ini",
    ".cfg", ".conf", ".yaml", ".yml", ".json", ".xml", ".csv", ".log", ".lean", ".typ", ".qmd",
}
SUPPORTED_BINARY_SUFFIXES = {".pdf", ".docx", ".pptx", ".xlsx", ".xlsm", ".ipynb", ".html", ".htm", ".png", ".jpg", ".jpeg", ".bmp", ".gif", ".tif", ".tiff", ".svg", ".npz"}
SUPPORTED_SUFFIXES = SUPPORTED_TEXT_SUFFIXES | SUPPORTED_BINARY_SUFFIXES


def default_home() -> Path:
    override = os.environ.get("STUDY_RETRIEVER_HOME")
    if override:
        return Path(override).expanduser().resolve()
    if sys.platform == "win32":
        base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
        return (base / "StudyRetriever").resolve()
    if sys.platform == "darwin":
        return (Path.home() / "Library" / "Application Support" / "StudyRetriever").resolve()
    base = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share"))
    return (base / "study-retriever").resolve()


@dataclass(slots=True)
class AppConfig:
    model_name: str = DEFAULT_MODEL
    vector_backend: str = "fastembed-usearch"  # fastembed-usearch | hashing
    chunk_chars: int = 1800
    overlap_chars: int = 220
    embedding_batch_size: int = 16
    dense_candidate_count: int = 50
    lexical_candidate_count: int = 50
    rrf_k: int = 60
    watcher_enabled: bool = True
    watcher_debounce_seconds: float = 1.2
    max_file_bytes: int = 256 * 1024 * 1024
    excluded_dir_names: list[str] = field(default_factory=lambda: [
        ".git", ".svn", ".hg", ".venv", "venv", "node_modules", "__pycache__", ".idea", ".vscode",
        ".study-data", "dist", "build", ".next", ".cache",
    ])
    roots: list[str] = field(default_factory=list)
    # Filter layer (ch. 13): fnmatch patterns on the path relative to its root, e.g. "*-presentation.pdf".
    exclude_globs: list[str] = field(default_factory=list)
    # Prompt-injection hardening: an LLM tool call must not be able to widen what gets indexed unless the user opts in.
    allow_mcp_root_changes: bool = True
    max_text_bytes: int = 32 * 1024 * 1024  # text/HTML/SVG are read whole; bigger files fail closed
    max_ocr_pages: int = 200
    # Optional cross-encoder rerank of the top hybrid hits (off by default; adds a model and query latency).
    reranker_model: str | None = "Xenova/ms-marco-MiniLM-L-6-v2"  # None disables
    rerank_top_k: int = 10
    rerank_chars: int = 600
    min_free_disk_mb: int = 1024  # refuse index writes below this (a full drive can corrupt SQLite/vector files)
    model_idle_unload_seconds: int = 300  # drop the embedding model from RAM when idle; 0 disables
    max_office_uncompressed_bytes: int = 256 * 1024 * 1024  # zip-bomb guard for DOCX/PPTX/XLSX

    def validate(self) -> None:
        if self.chunk_chars < 400:
            raise ValueError("chunk_chars must be >= 400")
        if not 0 <= self.overlap_chars < self.chunk_chars:
            raise ValueError("overlap_chars must be >= 0 and < chunk_chars")
        if self.embedding_batch_size < 1:
            raise ValueError("embedding_batch_size must be >= 1")
        if self.dense_candidate_count < 1 or self.lexical_candidate_count < 1:
            raise ValueError("candidate counts must be >= 1")
        if self.rrf_k < 1:
            raise ValueError("rrf_k must be >= 1")
        if self.watcher_debounce_seconds < 0.1:
            raise ValueError("watcher_debounce_seconds must be >= 0.1")
        if self.max_file_bytes < 1:
            raise ValueError("max_file_bytes must be >= 1")
        if self.rerank_top_k < 1 or self.rerank_chars < 100:
            raise ValueError("rerank_top_k must be >= 1 and rerank_chars >= 100")
        if self.min_free_disk_mb < 0 or self.model_idle_unload_seconds < 0:
            raise ValueError("min_free_disk_mb and model_idle_unload_seconds must be >= 0")
        if not all(isinstance(p, str) and p for p in self.exclude_globs):
            raise ValueError("exclude_globs must be non-empty strings")
        if self.vector_backend not in {"fastembed-usearch", "hashing"}:
            raise ValueError("vector_backend must be fastembed-usearch or hashing")


class Paths:
    def __init__(self, home: Path | None = None):
        self.home = (home or default_home()).expanduser().resolve()
        self.config = self.home / "config.json"
        self.db = self.home / "catalog.sqlite3"
        self.model_cache = self.home / "models"
        self.logs = self.home / "logs"
        self.home.mkdir(parents=True, exist_ok=True)
        self.model_cache.mkdir(parents=True, exist_ok=True)
        self.logs.mkdir(parents=True, exist_ok=True)


def load_config(paths: Paths) -> AppConfig:
    if not paths.config.exists():
        cfg = AppConfig()
        save_config(paths, cfg)
        return cfg
    data: dict[str, Any] = json.loads(paths.config.read_text(encoding="utf-8"))
    allowed = {f.name for f in AppConfig.__dataclass_fields__.values()}
    cfg = AppConfig(**{k: v for k, v in data.items() if k in allowed})
    cfg.validate()
    return cfg


def save_config(paths: Paths, config: AppConfig) -> None:
    config.validate()
    paths.config.parent.mkdir(parents=True, exist_ok=True)
    tmp = paths.config.with_suffix(".tmp")
    tmp.write_text(json.dumps(asdict(config), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, paths.config)
