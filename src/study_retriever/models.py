from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


def format_citation(path: str, section: str, locator: dict[str, Any]) -> str:
    parts = [path]
    if page := locator.get("page"):
        parts.append(f"page {page}")
    if slide := locator.get("slide"):
        parts.append(f"slide {slide}")
    if paragraph := locator.get("paragraph"):
        parts.append(f"paragraph {paragraph}")
    if table := locator.get("table"):
        parts.append(f"table {table}")
    if sheet := locator.get("sheet"):
        row_start = locator.get("row_start")
        row_end = locator.get("row_end")
        suffix = f" rows {row_start}-{row_end}" if row_start and row_end else ""
        parts.append(f"sheet {sheet}{suffix}")
    if cell := locator.get("cell"):
        parts.append(f"cell {cell}")
    if line_start := locator.get("line_start"):
        line_end = locator.get("line_end", line_start)
        parts.append(f"lines {line_start}-{line_end}")
    if section:
        parts.append(section)
    return " · ".join(parts)


@dataclass(slots=True)
class Block:
    text: str
    section: str = ""
    locator: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ParsedDocument:
    title: str
    blocks: list[Block]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class Chunk:
    chunk_id: str
    source_id: str
    vector_key: int
    ordinal: int
    text: str
    embedding_text: str
    title: str
    section: str
    path: str
    source_uri: str
    locator: dict[str, Any]


@dataclass(slots=True)
class SearchHit:
    chunk_id: str
    source_id: str
    score: float
    text: str
    title: str
    section: str
    path: str
    source_uri: str
    locator: dict[str, Any]
    dense_rank: int | None = None
    lexical_rank: int | None = None

    def citation(self) -> str:
        return format_citation(self.path, self.section, self.locator)
