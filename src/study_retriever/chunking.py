from __future__ import annotations

import bisect
import hashlib
import re
import uuid
from pathlib import Path

from .models import Chunk, ParsedDocument

_NAMESPACE = uuid.UUID("7b590b24-4792-4c76-9bfb-627f57eea4d8")


def source_id_for(path: Path) -> str:
    normalized = str(path.expanduser().resolve()).casefold()
    return str(uuid.uuid5(_NAMESPACE, normalized))


def _choose_break(text: str, start: int, target_end: int, minimum: int) -> int:
    if target_end >= len(text):
        return len(text)
    window_start = max(start + minimum, target_end - 320)
    for pattern in ("\n\n", "\n", ". ", "; ", ", ", " "):
        pos = text.rfind(pattern, window_start, target_end + 1)
        if pos >= window_start:
            return pos + len(pattern)
    return target_end


def _split_text(text: str, target: int, overlap: int) -> list[tuple[int, int, str]]:
    if len(text) <= target:
        return [(0, len(text), text.strip())]
    parts: list[tuple[int, int, str]] = []
    start = 0
    while start < len(text):
        end = _choose_break(text, start, min(len(text), start + target), max(120, target // 3))
        if end <= start:
            end = min(len(text), start + target)
        chunk = text[start:end].strip()
        if chunk:
            actual_start = start + (len(text[start:end]) - len(text[start:end].lstrip()))
            parts.append((actual_start, end, chunk))
        if end >= len(text):
            break
        start = max(start + 1, end - overlap)
    return parts


def chunk_document(
    path: Path,
    file_sha256: str,
    doc: ParsedDocument,
    chunk_chars: int,
    overlap_chars: int,
) -> list[Chunk]:
    sid = source_id_for(path)
    resolved = path.resolve()
    source_uri = resolved.as_uri()
    resolved_str = str(resolved)
    chunks: list[Chunk] = []
    ordinal = 0
    occurrences: dict[str, int] = {}
    for block in doc.blocks:
        # Newline offsets once per block: per-chunk slicing made huge single-block files quadratic.
        newlines = [m.start() for m in re.finditer("\n", block.text)] if "line_start" in block.locator else []
        for char_start, char_end, text in _split_text(block.text, chunk_chars, overlap_chars):
            locator = dict(block.locator)
            locator["char_start"] = char_start
            locator["char_end"] = char_end
            if "line_start" in block.locator:
                base = int(block.locator.get("line_start", 1))
                locator["line_start"] = base + bisect.bisect_left(newlines, char_start)
                locator["line_end"] = base + bisect.bisect_left(newlines, char_end)
            context_parts = [doc.title]
            if block.section and block.section.casefold() != doc.title.casefold():
                context_parts.append(block.section)
            context_parts.append(text)
            embedding_text = "\n\n".join(part for part in context_parts if part).strip()
            # Identity depends on what is embedded (context + text), not on the whole file or position, so an
            # edit re-embeds only the chunks whose content changed (delta sync) and unchanged ones keep their vectors.
            text_hash = hashlib.sha256(embedding_text.encode("utf-8")).hexdigest()[:20]
            occurrence = occurrences.get(text_hash, 0)
            occurrences[text_hash] = occurrence + 1
            cid = str(uuid.uuid5(_NAMESPACE, f"{sid}:{text_hash}:{occurrence}"))
            vector_key = int.from_bytes(hashlib.blake2b(cid.encode("ascii"), digest_size=8).digest(), "big") & 0x7FFFFFFFFFFFFFFF
            vector_key = vector_key or 1
            chunks.append(Chunk(
                chunk_id=cid,
                source_id=sid,
                vector_key=vector_key,
                ordinal=ordinal,
                text=text,
                embedding_text=embedding_text,
                title=doc.title,
                section=block.section,
                path=resolved_str,
                source_uri=source_uri,
                locator=locator,
            ))
            ordinal += 1
    return chunks
