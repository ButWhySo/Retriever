from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SearchResult(StrictModel):
    id: str
    title: str
    url: str


class SearchResponse(StrictModel):
    results: list[SearchResult]


class FetchResponse(StrictModel):
    id: str
    title: str
    text: str
    url: str
    metadata: dict[str, Any] | None = None


class DetailedSearchResult(StrictModel):
    id: str
    source_id: str
    title: str
    url: str
    text: str
    snippet: str
    citation: str
    path: str
    section: str
    locator: dict[str, Any]
    score: float
    dense_rank: int | None = None
    lexical_rank: int | None = None


class DetailedSearchResponse(StrictModel):
    results: list[DetailedSearchResult]
    degraded: bool = False  # True when the dense side failed and results are lexical-only


class DetailedFetchResponse(StrictModel):
    id: str
    source_id: str
    title: str
    url: str
    path: str
    section: str
    locator: dict[str, Any]
    citation: str
    content: str
    neighbor_chunk_ids: list[str]


class TopicSource(StrictModel):
    source_id: str
    title: str
    path: str
    url: str


class TopicBundleResponse(StrictModel):
    query: str
    chunk_count: int
    characters: int
    sources: list[TopicSource]
    content: str


class SourceBlock(StrictModel):
    section: str
    locator: dict[str, Any]
    text: str


class SourceReadResponse(StrictModel):
    source_id: str
    title: str
    path: str
    url: str
    metadata: dict[str, Any]
    blocks: list[SourceBlock]
    content: str
    truncated: bool


class IndexStats(StrictModel):
    version: str
    roots: list[str]
    sources: int
    chunks: int
    errors: int
    vector_backend: str
    vectors: int
    model: str
    home: str
    vector_count_matches_chunks: bool
    source_bytes: int
    db_path: str
    error_sources: list[dict[str, str]]
    watcher_heartbeat_age_s: float | None = None
    free_disk_mb: int | None = None
    sync_running: bool = False


class SyncCounts(StrictModel):
    indexed: int = 0
    unchanged: int = 0
    deleted: int = 0
    errors: int = 0
    chunks_added: int = 0


class SyncResponse(StrictModel):
    total: SyncCounts
    roots: dict[str, dict[str, Any]]
    status: str = "completed"  # "running" when the sync outlived the tool-call wait; see index_status.sync_running


class RootResponse(StrictModel):
    root: str
    sync: dict[str, Any] | None = None
    deleted_sources: int | None = None
    deleted_chunks: int | None = None


class RebuildResponse(StrictModel):
    vectors: int
    backend: str
