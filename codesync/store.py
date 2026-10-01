"""The chunk store: LanceDB, one table of chunks across all documents
(work package 6). Text, fields and vector live in one row; search is by
vector, by keywords, or both, with filters on code and edition.

No server: the data is files under data/lancedb, like everything else.
"""
from __future__ import annotations

import json
from pathlib import Path

import lancedb
import numpy as np
import pyarrow as pa
from lancedb.index import FTS

from . import config
from .stages.chunk import Chunk

DB_DIR = config.DATA_DIR / "lancedb"
TABLE = "chunks"

SCHEMA = pa.schema([
    pa.field("id", pa.string()),
    pa.field("doc_id", pa.string()),
    pa.field("code", pa.string()),
    pa.field("edition", pa.string()),
    pa.field("kind", pa.string()),
    pa.field("ref", pa.string()),
    pa.field("page", pa.int32()),
    pa.field("page_label", pa.string()),
    pa.field("heading_path", pa.string()),      # " > " joined
    pa.field("text", pa.string()),              # verbatim
    pa.field("search_text", pa.string()),       # what the keyword index sees
    pa.field("chunk_json", pa.string()),        # the whole Chunk, for display
    pa.field("vector", pa.list_(pa.float32(), config.EMBED_DIM)),
])


def _db():
    DB_DIR.mkdir(parents=True, exist_ok=True)
    return lancedb.connect(str(DB_DIR))


def _names(db) -> list[str]:
    return list(db.list_tables().tables)


def _table():
    db = _db()
    if TABLE in _names(db):
        return db.open_table(TABLE)
    return db.create_table(TABLE, schema=SCHEMA)


def _open():
    """The table, or None before the first document is indexed."""
    db = _db()
    return db.open_table(TABLE) if TABLE in _names(db) else None


def search_text(c: Chunk) -> str:
    """Keyword index content: reference, heading path, text, notes and the
    description. The reference is repeated bare ("104.1.2") and with the
    dots stripped so "104 1 2" style tokenisation still hits it."""
    bits = [c.ref, c.ref.replace(".", " "), " > ".join(c.heading_path),
            c.context, c.text, " ".join(c.notes), c.description]
    if c.table:
        bits.append(c.table.get("caption", ""))
        bits.append(" ".join(c.table.get("columns", [])))
    return " ".join(b for b in bits if b)


def replace_document(chunks: list[Chunk], vectors: np.ndarray) -> int:
    """Store a document's chunks, replacing any earlier rows for it."""
    if not chunks:
        return 0
    assert vectors.shape == (len(chunks), config.EMBED_DIM), vectors.shape
    tbl = _table()
    doc_id = chunks[0].doc_id
    tbl.delete(f"doc_id = '{doc_id}'")
    rows = [{
        "id": c.id, "doc_id": c.doc_id, "code": c.code, "edition": c.edition,
        "kind": c.kind, "ref": c.ref, "page": c.page,
        "page_label": c.page_label, "heading_path": " > ".join(c.heading_path),
        "text": c.text, "search_text": search_text(c),
        "chunk_json": c.model_dump_json(),
        "vector": vectors[i].astype(np.float32).tolist(),
    } for i, c in enumerate(chunks)]
    tbl.add(rows)
    # the keyword index is rebuilt over the whole table after every load
    tbl.create_index("search_text", config=FTS(), replace=True)
    return len(rows)


def delete_document(doc_id: str) -> None:
    tbl = _open()
    if tbl is not None:
        tbl.delete(f"doc_id = '{doc_id}'")


def count(doc_id: str | None = None) -> int:
    tbl = _open()
    if tbl is None:
        return 0
    return tbl.count_rows(f"doc_id = '{doc_id}'" if doc_id else None)


def indexed_documents() -> dict[str, int]:
    """{doc_id: number of chunks} for every document in the store."""
    tbl = _open()
    if tbl is None:
        return {}
    t = tbl.search().select(["doc_id"]).limit(10_000_000).to_arrow()
    out: dict[str, int] = {}
    for d in t.column("doc_id").to_pylist():
        out[d] = out.get(d, 0) + 1
    return out


def _where(doc_ids: list[str] | None, code: str | None, edition: str | None) -> str | None:
    parts = []
    if doc_ids:
        ids = ", ".join(f"'{d}'" for d in doc_ids)
        parts.append(f"doc_id IN ({ids})")
    if code:
        parts.append(f"code = '{code}'")
    if edition:
        parts.append(f"edition = '{edition}'")
    return " AND ".join(parts) or None


def _rows_to_chunks(rows: list[dict], score_key: str) -> list[tuple[Chunk, float]]:
    out = []
    for r in rows:
        c = Chunk.model_validate_json(r["chunk_json"])
        out.append((c, float(r.get(score_key, 0.0))))
    return out


def vector_search(vector: np.ndarray, k: int = 30, doc_ids: list[str] | None = None,
                  code: str | None = None, edition: str | None = None
                  ) -> list[tuple[Chunk, float]]:
    """Nearest chunks by cosine similarity (vectors are unit length)."""
    tbl = _open()
    if tbl is None:
        return []
    q = tbl.search(vector.astype(np.float32).tolist(),
                   vector_column_name="vector").metric("cosine").limit(k)
    where = _where(doc_ids, code, edition)
    if where:
        q = q.where(where, prefilter=True)
    rows = q.to_list()
    # LanceDB returns a distance; similarity = 1 - distance
    return [(c, 1.0 - d) for c, d in _rows_to_chunks(rows, "_distance")]


def keyword_search(query: str, k: int = 30, doc_ids: list[str] | None = None,
                   code: str | None = None, edition: str | None = None
                   ) -> list[tuple[Chunk, float]]:
    """Full-text (BM25) search over search_text."""
    tbl = _open()
    if tbl is None:
        return []
    q = tbl.search(query, query_type="fts", fts_columns="search_text").limit(k)
    where = _where(doc_ids, code, edition)
    if where:
        q = q.where(where, prefilter=True)
    try:
        rows = q.to_list()
    except Exception:                                    # noqa: BLE001
        return []                                        # no index yet
    return _rows_to_chunks(rows, "_score")


def get(chunk_ids: list[str]) -> dict[str, Chunk]:
    tbl = _open()
    if tbl is None or not chunk_ids:
        return {}
    ids = ", ".join(f"'{i}'" for i in chunk_ids)
    rows = tbl.search().where(f"id IN ({ids})").limit(len(chunk_ids)).to_list()
    return {r["id"]: Chunk.model_validate_json(r["chunk_json"]) for r in rows}


def definitions(doc_ids: list[str] | None = None) -> list[Chunk]:
    """Every definition chunk in scope (for term matching at answer time)."""
    tbl = _open()
    if tbl is None:
        return []
    where = "kind = 'definition'"
    scope = _where(doc_ids, None, None)
    if scope:
        where += f" AND {scope}"
    rows = tbl.search().where(where).limit(5000).to_list()
    return [Chunk.model_validate_json(r["chunk_json"]) for r in rows]


def dump(path: Path, doc_id: str) -> None:
    """Plain JSON copy of a document's rows (without vectors), for a look."""
    rows = _table().search().where(f"doc_id = '{doc_id}'").limit(100000).to_list()
    path.write_text(json.dumps([{k: v for k, v in r.items() if k != "vector"} for r in rows],
                               indent=1, ensure_ascii=False), encoding="utf-8")
