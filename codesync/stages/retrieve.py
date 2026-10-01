"""Find the passages a question needs (work package 7).

    embed the question -> vector top 30 + keyword top 30 -> merge
    -> rerank to the best few -> one hop of references -> definitions

A plain top-5 by similarity finds the paragraph and misses the table it
points to; the reference hop and the definitions are what make an answer
complete.
"""
from __future__ import annotations

import re
import time

from .. import store
from ..models.chat import engine
from .chunk import Chunk

K_VECTOR = 30
K_KEYWORD = 30
K_RERANK = 40            # candidates sent to the reranker
K_FINAL = 6              # passages kept by rank ...
RERANK_FLOOR = 0.35      # ... plus any above this score, up to K_MAX
K_MAX = 8
K_REFS = 6               # referenced chunks pulled in after the final set
K_DEFS = 3               # definitions of terms used in the question/passages


def _rrf(*ranked: list[tuple[Chunk, float]], k: int = 60) -> list[Chunk]:
    """Reciprocal-rank fusion of several ranked lists into one order."""
    score: dict[str, float] = {}
    seen: dict[str, Chunk] = {}
    for lst in ranked:
        for rank, (c, _s) in enumerate(lst):
            score[c.id] = score.get(c.id, 0.0) + 1.0 / (k + rank + 1)
            seen.setdefault(c.id, c)
    return [seen[i] for i in sorted(score, key=score.__getitem__, reverse=True)]


TABLE_ROWS = 12          # rows of a table shown in a passage


def _tokens(text: str) -> set[str]:
    return {t for t in re.findall(r"[a-z0-9][a-z0-9.\-/]*", text.lower()) if len(t) >= 2}


def table_rows_for(table: dict, question: str | None, limit: int = TABLE_ROWS) -> list[list[str]]:
    """The rows of a table worth showing for a question: a table is data, so
    the rows that mention something the question mentions (a material, a
    grade, a size, a temperature) come first, in table order; the first rows
    fill the rest. Without a question: the first rows."""
    rows = table.get("rows") or []
    if not question or len(rows) <= limit:
        return rows[:limit]
    want = _tokens(question)
    scored = []
    for i, r in enumerate(rows):
        have = _tokens(" ".join(r))
        scored.append((len(want & have), i))
    picked = sorted((i for n, i in sorted(scored, key=lambda x: (-x[0], x[1]))[:limit] if n > 0))
    for i in range(len(rows)):                 # pad with leading rows
        if len(picked) >= limit:
            break
        if i not in picked:
            picked.append(i)
    return [rows[i] for i in sorted(picked)]


def _passage_text(c: Chunk, limit: int = 1_800, question: str | None = None) -> str:
    """What the reranker and the answer model read for a chunk."""
    head = " > ".join(c.heading_path)
    parts = [f"{head}\n" if head else "", c.text]
    if c.notes:
        parts.append("\nNotes: " + " ".join(c.notes))
    if c.description:
        parts.append("\nDescription: " + c.description)
    if c.table and c.table.get("rows"):
        cols = " | ".join(c.table.get("columns") or [])
        shown = table_rows_for(c.table, question)
        rows = "\n".join(" | ".join(r) for r in shown)
        more = len(c.table["rows"]) - len(shown)
        tail = f"\n({more} more rows not shown)" if more > 0 else ""
        parts.append(f"\nTable columns: {cols}\n{rows}{tail}")
    return "".join(parts)[:limit]


# Codes define their own verbs ("may: ... permission"); those are worth
# pulling in only when the question itself is about the word
_VERB_TERMS = {"may", "shall", "should", "must", "will", "can"}


def _definitions_for(texts: list[str], scope: list[str] | None, limit: int) -> list[Chunk]:
    """Definition chunks whose term appears in any of `texts` (the first
    of which is the question)."""
    question = texts[0].lower() if texts else ""
    hay = " ".join(texts).lower()
    hits: list[tuple[int, Chunk]] = []
    for d in store.definitions(scope):
        term = d.ref.replace("Definition:", "").strip().lower()
        if len(term) < 3:
            continue
        where = question if term in _VERB_TERMS else hay
        if re.search(r"\b" + re.escape(term) + r"\b", where):
            hits.append((len(term), d))
    hits.sort(key=lambda h: -h[0])              # longest terms first
    return [d for _n, d in hits[:limit]]


def retrieve(question: str, embedder, doc_ids: list[str] | None = None,
             code: str | None = None, edition: str | None = None) -> dict:
    """{"passages": [{"tag", "chunk", "why", "score"}], "timings": {...}}"""
    timings: dict[str, float] = {}
    t = time.time()
    qvec = embedder.embed([question], as_query=True)[0]
    timings["embed_s"] = round(time.time() - t, 2)

    t = time.time()
    by_vector = store.vector_search(qvec, K_VECTOR, doc_ids, code, edition)
    by_keyword = store.keyword_search(question, K_KEYWORD, doc_ids, code, edition)
    candidates = _rrf(by_vector, by_keyword)[:K_RERANK]
    timings["search_s"] = round(time.time() - t, 2)
    vec_rank = {c.id: i + 1 for i, (c, _s) in enumerate(by_vector)}
    kw_rank = {c.id: i + 1 for i, (c, _s) in enumerate(by_keyword)}

    final: list[tuple[Chunk, float]] = []
    if candidates:
        t = time.time()
        res = engine().rerank.remote(question, [_passage_text(c, question=question) for c in candidates])
        timings["rerank_s"] = round(time.time() - t, 2)
        ranked = sorted(zip(candidates, res["scores"]), key=lambda x: -x[1])
        final = [(c, s) for i, (c, s) in enumerate(ranked)
                 if i < K_FINAL or (s >= RERANK_FLOOR and i < K_MAX)]

    chosen = {c.id for c, _s in final}
    passages = [{"chunk": c, "score": round(s, 3), "why": "search",
                 "vector_rank": vec_rank.get(c.id), "keyword_rank": kw_rank.get(c.id)}
                for c, s in final]

    # one hop: what the chosen passages point at — tables and figures
    # first (they hold the values), then paragraphs, in passage order
    wanted: list[str] = []
    for kinds in (("table", "fig"), ("para", "eq", "appendix", "chapter")):
        for c, _s in final:
            for r in c.refs:
                if r.kind in kinds and r.chunk_id and r.chunk_id not in chosen \
                        and r.chunk_id not in wanted:
                    wanted.append(r.chunk_id)
    for cid, c in store.get(wanted[:K_REFS]).items():
        chosen.add(cid)
        passages.append({"chunk": c, "score": None, "why": "referenced",
                         "vector_rank": vec_rank.get(cid), "keyword_rank": kw_rank.get(cid)})

    # definitions of terms the question or passages use
    for d in _definitions_for([question] + [c.text for c, _s in final], doc_ids, K_DEFS):
        if d.id not in chosen:
            chosen.add(d.id)
            passages.append({"chunk": d, "score": None, "why": "definition",
                             "vector_rank": vec_rank.get(d.id), "keyword_rank": kw_rank.get(d.id)})

    for i, p in enumerate(passages, 1):
        p["tag"] = f"P{i}"
    return {"passages": passages, "timings": timings,
            "candidates": len(candidates), "vector_hits": len(by_vector),
            "keyword_hits": len(by_keyword)}
