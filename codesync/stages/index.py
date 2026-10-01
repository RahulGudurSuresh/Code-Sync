"""Build the Ask-the-Code library for one document (work packages 3 to 6):

    parsed.json -> chunks (verbatim, with heading paths, notes, references)
                -> figure and table descriptions ADDED by the models
                -> embeddings -> LanceDB rows

Descriptions are cached per element (by its page and position, which
survive a re-parse) and prompt version in descriptions.json, so
re-indexing after a parser or chunker change costs no model calls. The
original text is never sent to a model for rewriting.
"""
from __future__ import annotations

import time
from collections.abc import Callable

from .. import config, library, prompts, store
from ..models.chat import chat_json, image_part
from ..schema import DocumentMeta, ParsedDocument
from .chunk import Chunk, build_chunks

Progress = Callable[[str], None]
DESCRIBE_VERSION = "ask-2026-10-01.1"


def _place_key(e) -> str:
    """Identity of a figure or table that survives a re-parse (element ids
    are renumbered every time): kind, page and position."""
    return f"{e.kind}:p{e.page}:" + ",".join(str(round(v)) for v in e.bbox)


def _descriptions(meta: DocumentMeta, parsed: ParsedDocument,
                  chunks: list[Chunk], vlm, llm, say: Progress) -> dict:
    """{element_id: description} for every figure and table chunk."""
    key = f"{DESCRIBE_VERSION}|{vlm.name}|{llm.name}"
    cached = library.load_json(meta.doc_id, "descriptions.json") or {}
    by_place: dict[str, str] = cached.get("descriptions", {}) if cached.get("key") == key else {}
    fig_dir = library.doc_dir(meta.doc_id) / "figures"
    by_id = {e.id: e for e in parsed.elements}
    done: dict[str, str] = {}
    for c in chunks:
        if c.kind in ("figure", "table"):
            d = by_place.get(_place_key(by_id[c.element_ids[0]]))
            if d is not None:
                done[c.element_ids[0]] = d

    figs = [c for c in chunks if c.kind == "figure" and c.element_ids[0] not in done]
    if figs:
        say(f"describing {len(figs)} figures ({vlm.name})")
        requests = []
        for c in figs:
            e = by_id[c.element_ids[0]]
            cap = f', captioned "{e.text}"' if e.text else ""
            requests.append({"messages": [
                {"role": "system", "content": prompts.DESCRIBE_FIGURE_SYSTEM},
                {"role": "user", "content": [
                    image_part(fig_dir / e.image, config.LLM_IMG_TOKENS),
                    {"type": "text", "text": prompts.DESCRIBE_FIGURE_USER.format(
                        doc_name=meta.name, ref=c.ref, caption=cap)}]}],
                "max_tokens": 600})
        for c, (answer, _raw) in zip(figs, chat_json(vlm, requests)):
            done[c.element_ids[0]] = str((answer or {}).get("description", "")).strip() \
                if isinstance(answer, dict) else ""

    tables = [c for c in chunks if c.kind == "table" and c.element_ids[0] not in done]
    if tables:
        say(f"describing {len(tables)} tables ({llm.name})")
        requests = []
        for c in tables:
            t = c.table or {}
            rows = "\n".join(" | ".join(r) for r in (t.get("rows") or [])[:6])
            requests.append({"messages": [
                {"role": "system", "content": prompts.DESCRIBE_TABLE_SYSTEM},
                {"role": "user", "content": prompts.DESCRIBE_TABLE_USER.format(
                    doc_name=meta.name, section=" > ".join(c.heading_path) or "-",
                    caption=t.get("caption") or c.text or "(none)",
                    columns=" | ".join(t.get("columns") or []), rows=rows or "(none)",
                    notes=("Notes: " + " ".join(c.notes)) if c.notes else "")}],
                "max_tokens": 400})
        for c, (answer, _raw) in zip(tables, chat_json(llm, requests)):
            done[c.element_ids[0]] = str((answer or {}).get("description", "")).strip() \
                if isinstance(answer, dict) else ""

    for eid, d in done.items():
        by_place[_place_key(by_id[eid])] = d
    library.save_json(meta.doc_id, "descriptions.json", {"key": key, "descriptions": by_place})
    return done


def index_document(doc_id: str, llm, vlm, embedder, progress: Progress | None = None) -> dict:
    """Parse (cached), chunk, describe, embed and store one document."""
    say = progress or (lambda _m: None)
    meta = library.load_meta(doc_id)
    t0 = time.time()

    parsed = library.load_parsed(doc_id)
    if parsed is None:
        from ..pdfparse import parse_pdf
        say(f"[{meta.label}] parsing")
        d = library.doc_dir(doc_id)
        parsed = parse_pdf(d / "source.pdf", doc_id, d / "figures",
                           progress=lambda m: say(f"[{meta.label}] {m}"))
        library.save_parsed(parsed)
    t1 = time.time()

    say(f"[{meta.label}] chunking")
    chunks, stats = build_chunks(meta, parsed)
    descriptions = _descriptions(meta, parsed, chunks, vlm, llm,
                                 lambda m: say(f"[{meta.label}] {m}"))
    for c in chunks:
        if c.kind in ("figure", "table"):
            c.description = descriptions.get(c.element_ids[0], "")
    library.save_json(doc_id, "chunks.json",
                      {"stats": stats, "chunks": [c.model_dump() for c in chunks]})
    t2 = time.time()

    say(f"[{meta.label}] embedding {len(chunks)} chunks")
    vectors = embedder.embed([c.embed_text for c in chunks], as_query=False)
    n = store.replace_document(chunks, vectors)
    t3 = time.time()

    meta.counts["chunks"] = n
    meta.counts["chunking"] = stats
    meta.counts["indexed_key"] = DESCRIBE_VERSION
    meta.timings.update(index_chunk_s=round(t2 - t1, 1), index_embed_s=round(t3 - t2, 1))
    library.save_meta(meta)
    say(f"[{meta.label}] indexed {n} chunks in {time.time() - t0:.0f}s")
    return {"chunks": n, "stats": stats, "elapsed_s": round(time.time() - t0, 1)}
