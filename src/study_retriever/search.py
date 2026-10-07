from __future__ import annotations

import logging
from collections import defaultdict
from pathlib import Path

from .catalog import Catalog
from .config import AppConfig
from .models import SearchHit
from .parsers import parse_document
from .vector_store import VectorStore

MAX_RENDER_PIXELS = 4000
log = logging.getLogger(__name__)


class SearchEngine:
    def __init__(self, catalog: Catalog, vectors: VectorStore, config: AppConfig):
        self.catalog = catalog
        self.vectors = vectors
        self.config = config
        self.degraded = False

    @staticmethod
    def _matches_filters(row: dict, path_prefix: str | None, file_types: list[str] | None) -> bool:
        if path_prefix:
            try:
                if not Path(row["path"]).resolve().is_relative_to(Path(path_prefix).expanduser().resolve()):
                    return False
            except (OSError, ValueError):
                return False
        if file_types:
            wanted = {x.lower() if x.startswith(".") else "." + x.lower() for x in file_types}
            if Path(row["path"]).suffix.lower() not in wanted:
                return False
        return True

    def _resolve_source(self, source_id: str | None, path: str | None) -> dict:
        source = self.catalog.source_by_id(source_id) if source_id else None
        if source is None and path:
            source = self.catalog.source_by_path(Path(path))
        if source is None:
            raise FileNotFoundError("Indexed source not found")
        source_path = Path(source["path"])
        if not source_path.is_file():
            raise FileNotFoundError(f"Indexed source no longer exists: {source_path}")
        return source

    def hybrid_search(
        self,
        query: str,
        *,
        top_k: int = 15,
        path_prefix: str | None = None,
        file_types: list[str] | None = None,
    ) -> list[SearchHit]:
        query = query.strip()
        if not query:
            return []
        top_k = max(1, min(int(top_k), 100))
        multiplier = 12 if (path_prefix or file_types) else 4
        floor = 300 if (path_prefix or file_types) else 0
        dense_limit = max(self.config.dense_candidate_count, top_k * multiplier, floor)
        lexical_limit = max(self.config.lexical_candidate_count, top_k * multiplier, floor)

        try:
            dense_raw = self.vectors.search(query, dense_limit)
            self.degraded = False
        except Exception:  # model load failure, ONNX OOM, damaged index: keep answering from BM25
            log.exception("dense search failed; returning lexical-only results")
            dense_raw = []
            self.degraded = True
        key_map = self.catalog.vector_key_map(key for key, _score in dense_raw)
        dense_ids = [key_map[key] for key, _score in dense_raw if key in key_map]
        lexical_raw = self.catalog.lexical_search(query, lexical_limit)
        lexical_ids = [cid for cid, _score in lexical_raw]

        rrf: dict[str, float] = defaultdict(float)
        dense_rank: dict[str, int] = {}
        lexical_rank: dict[str, int] = {}
        k = self.config.rrf_k
        for rank, cid in enumerate(dense_ids, start=1):
            if cid not in dense_rank:
                dense_rank[cid] = rank
                rrf[cid] += 1.0 / (k + rank)
        for rank, cid in enumerate(lexical_ids, start=1):
            if cid not in lexical_rank:
                lexical_rank[cid] = rank
                rrf[cid] += 1.0 / (k + rank)

        ordered = sorted(
            rrf,
            key=lambda cid: (-rrf[cid], lexical_rank.get(cid, 10**9), dense_rank.get(cid, 10**9)),
        )
        rows = self.catalog.get_chunks(ordered)
        hits: list[SearchHit] = []
        for cid in ordered:
            row = rows.get(cid)
            if not row or not self._matches_filters(row, path_prefix, file_types):
                continue
            hits.append(
                SearchHit(
                    chunk_id=cid,
                    source_id=row["source_id"],
                    score=rrf[cid],
                    text=row["text"],
                    title=row["title"],
                    section=row["section"],
                    path=row["path"],
                    source_uri=row["source_uri"],
                    locator=row["locator"],
                    dense_rank=dense_rank.get(cid),
                    lexical_rank=lexical_rank.get(cid),
                )
            )
            if len(hits) >= top_k:
                break
        return hits

    def fetch(self, chunk_id: str, *, neighbor_radius: int = 1) -> dict | None:
        row = self.catalog.get_chunk(chunk_id)
        if not row:
            return None
        neighbors = self.catalog.neighbors(
            row["source_id"], int(row["ordinal"]), max(0, min(neighbor_radius, 5))
        )
        return {
            "id": row["chunk_id"],
            "source_id": row["source_id"],
            "title": row["title"],
            "url": row["source_uri"],
            "path": row["path"],
            "section": row["section"],
            "locator": row["locator"],
            "citation": self._locator_label(row),
            "content": "\n\n".join(n["text"] for n in neighbors),
            "neighbor_chunk_ids": [n["chunk_id"] for n in neighbors],
        }

    @staticmethod
    def _locator_label(row: dict) -> str:
        loc = row["locator"]
        parts = [row["path"]]
        if loc.get("page"):
            parts.append(f"page {loc['page']}")
        if loc.get("slide"):
            parts.append(f"slide {loc['slide']}")
        if loc.get("paragraph"):
            parts.append(f"paragraph {loc['paragraph']}")
        if loc.get("table"):
            parts.append(f"table {loc['table']}")
        if loc.get("sheet"):
            rows = ""
            if loc.get("row_start") is not None:
                rows = f" rows {loc['row_start']}-{loc.get('row_end', loc['row_start'])}"
            parts.append(f"sheet {loc['sheet']}{rows}")
        if loc.get("cell"):
            parts.append(f"cell {loc['cell']}")
        if loc.get("line_start"):
            parts.append(f"lines {loc['line_start']}-{loc.get('line_end', loc['line_start'])}")
        if row["section"]:
            parts.append(row["section"])
        return " · ".join(parts)

    def topic_bundle(
        self,
        query: str,
        *,
        max_chunks: int = 30,
        max_chars: int = 45_000,
        per_source: int = 8,
        path_prefix: str | None = None,
    ) -> dict:
        max_chunks = max(1, min(max_chunks, 80))
        max_chars = max(2_000, min(max_chars, 90_000))  # stay under client tool-output caps
        per_source = max(1, min(per_source, 20))
        hits = self.hybrid_search(query, top_k=min(100, max_chunks * 3), path_prefix=path_prefix)
        selected: list[dict] = []
        seen: set[str] = set()
        by_source: dict[str, int] = defaultdict(int)
        total_chars = 0

        for hit in hits:
            if len(selected) >= max_chunks or total_chars >= max_chars:
                break
            hit_row = self.catalog.get_chunk(hit.chunk_id)
            if not hit_row:
                continue
            neighbors = self.catalog.neighbors(hit.source_id, int(hit_row["ordinal"]), radius=1)
            for row in neighbors:
                if row["chunk_id"] in seen or by_source[row["source_id"]] >= per_source:
                    continue
                text = row["text"]
                if total_chars + len(text) > max_chars and selected:
                    continue
                selected.append(row)
                seen.add(row["chunk_id"])
                by_source[row["source_id"]] += 1
                total_chars += len(text)
                if len(selected) >= max_chunks:
                    break

        sources: dict[str, dict] = {}
        content_blocks: list[str] = []
        for row in selected:
            sid = row["source_id"]
            if sid not in sources:
                sources[sid] = {
                    "source_id": sid,
                    "title": row["title"],
                    "path": row["path"],
                    "url": row["source_uri"],
                }
            content_blocks.append(f"[SOURCE {row['chunk_id']}] {self._locator_label(row)}\n{row['text']}")
        return {
            "query": query,
            "chunk_count": len(selected),
            "characters": total_chars,
            "sources": list(sources.values()),
            "content": "\n\n---\n\n".join(content_blocks),
        }

    @staticmethod
    def _block_matches(
        loc: dict,
        *,
        page: int | None,
        slide: int | None,
        cell: int | None,
        sheet: str | None,
        paragraph: int | None,
        table: int | None,
        row_start: int | None,
        row_end: int | None,
    ) -> bool:
        if page is not None and loc.get("page") != page:
            return False
        if slide is not None and loc.get("slide") != slide:
            return False
        if cell is not None and loc.get("cell") != cell:
            return False
        if sheet is not None and str(loc.get("sheet", "")).casefold() != sheet.casefold():
            return False
        if paragraph is not None and loc.get("paragraph") != paragraph:
            return False
        if table is not None and loc.get("table") != table:
            return False
        if row_start is not None or row_end is not None:
            block_start = loc.get("row_start")
            block_end = loc.get("row_end")
            if block_end is None:
                block_end = block_start
            if block_start is None or block_end is None:
                return False
            wanted_start = row_start if row_start is not None else -1
            wanted_end = row_end if row_end is not None else 2**63 - 1
            if int(block_end) < wanted_start or int(block_start) > wanted_end:
                return False
        return True

    def read_source(
        self,
        *,
        source_id: str | None = None,
        path: str | None = None,
        page: int | None = None,
        slide: int | None = None,
        cell: int | None = None,
        sheet: str | None = None,
        paragraph: int | None = None,
        table: int | None = None,
        row_start: int | None = None,
        row_end: int | None = None,
        line_start: int | None = None,
        line_end: int | None = None,
        max_chars: int = 60_000,
    ) -> dict:
        max_chars = max(1_000, min(int(max_chars), 250_000))
        source = self._resolve_source(source_id, path)
        source_path = Path(source["path"])
        doc = parse_document(source_path)
        blocks: list[dict] = []
        for block in doc.blocks:
            loc = block.locator
            if not self._block_matches(
                loc,
                page=page,
                slide=slide,
                cell=cell,
                sheet=sheet,
                paragraph=paragraph,
                table=table,
                row_start=row_start,
                row_end=row_end,
            ):
                continue
            text = block.text
            out_loc = dict(loc)
            if line_start is not None or line_end is not None:
                base = int(loc.get("line_start", 1))
                lines = text.splitlines()
                wanted_start = max(base, int(line_start if line_start is not None else base))
                wanted_end = int(line_end if line_end is not None else base + len(lines) - 1)
                first = max(0, wanted_start - base)
                last = min(len(lines), wanted_end - base + 1)
                if first >= last:
                    continue
                text = "\n".join(lines[first:last])
                out_loc["line_start"] = wanted_start
                out_loc["line_end"] = min(wanted_end, base + len(lines) - 1)
            blocks.append({"section": block.section, "locator": out_loc, "text": text})

        text = "\n\n".join(b["text"] for b in blocks)
        original_chars = len(text)
        truncated = original_chars > max_chars
        if truncated:
            text = text[:max_chars]
        return {
            "source_id": source["source_id"],
            "title": doc.title,
            "path": str(source_path),
            "url": source_path.as_uri(),
            "metadata": doc.metadata,
            "blocks": blocks if original_chars <= 12_000 else [],
            "content": text,
            "truncated": truncated,
        }

    def render_pdf_page(
        self,
        *,
        source_id: str | None = None,
        path: str | None = None,
        page: int,
        dpi: int = 144,
    ) -> bytes:
        """Render one indexed PDF page to PNG for diagram/scan/figure inspection."""
        import fitz

        source = self._resolve_source(source_id, path)
        source_path = Path(source["path"])
        if source_path.suffix.lower() != ".pdf":
            raise ValueError("render_pdf_page only accepts indexed PDF sources")
        dpi = max(72, min(int(dpi), 300))
        with fitz.open(source_path) as doc:
            if page < 1 or page > doc.page_count:
                raise ValueError(f"Page must be between 1 and {doc.page_count}")
            loaded = doc.load_page(page - 1)
            longest = max(loaded.rect.width, loaded.rect.height, 1.0)
            dpi = max(1, min(dpi, int(MAX_RENDER_PIXELS * 72 / longest)))  # crafted huge MediaBox must not allocate GBs
            matrix = fitz.Matrix(dpi / 72.0, dpi / 72.0)
            pix = loaded.get_pixmap(matrix=matrix, alpha=False)
            return pix.tobytes("png")
