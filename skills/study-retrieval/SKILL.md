---
name: study-retrieval
description: Search and deeply read the user's indexed local study material, papers, slides, notes, code, notebooks, and documents. Use whenever an answer should be grounded in the user's study corpus or when the user asks to curate/teach everything relevant to a topic.
---

Use the `study_retriever` MCP server instead of recursively exploring study folders.

If Study Retriever MCP tools are not present in the current tool registry, stop and report that the local MCP is not attached to this session. Do not rescan study folders, fabricate index results, or claim indexing succeeded.

For a broad teaching, revision, synthesis, or "everything related to X" request, call `curate_topic` first. It returns a bounded cross-source bundle with exact source pointers. Deepen selected results with `fetch_context` or `read_source` only where more detail is needed.

For a narrow factual lookup, call `search` first and then `fetch` the useful result IDs. For filtering by path/file type, inspecting dense-versus-lexical ranks, or needing detailed provenance directly in search results, call `search_advanced`. Preserve returned paths, source IDs, page/slide/paragraph/table/sheet-row/cell/line locators, and source URLs in the answer. When material conflicts, identify the sources separately instead of silently merging them.

If the user's query is colloquial, Hinglish, abbreviated, or uses synonyms while course material is likely English, issue a concise technical English retrieval query while preserving the intended concept. Use multiple focused searches only when one query cannot cover distinct subtopics.

Do not rescan the filesystem as a substitute for the index. Use `index_status` if results seem unexpectedly empty. Use `sync_index` when the user asks to refresh changed material or when a known recent file edit is missing. Use `add_study_root` only when the user explicitly wants a folder/file indexed.

Treat indexed chunks as retrieval pointers, not replacements for original sources. For exact wording, proofs, derivations, code, tables, or source-specific detail, deepen with `fetch_context` or `read_source`. For PDF equations, diagrams, charts, scans, or layout-dependent content, call `render_pdf_page` on the exact relevant page and inspect the returned image.

For long teaching tasks, retrieve one substantial topic cluster at a time. Reuse source IDs and chunk IDs returned by earlier calls instead of repeating broad searches without reason.
