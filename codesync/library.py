"""On-disk store: one folder per ingested code, one per comparison.

    data/library/<doc_id>/      source.pdf  meta.json  parsed.json
                                figures/    figure_readings.json
                                requirements.json  embeddings.npz
    data/comparisons/<cmp_id>/  meta.json  pairs.json  findings.json
                                report.md  report.xlsx

Plain files on purpose: every intermediate result can be opened and read,
and a stage is skipped when its output already exists for the same inputs.
"""
from __future__ import annotations

import hashlib
import json
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from . import config
from .schema import (ComparisonMeta, DocumentMeta, Finding, ParsedDocument,
                     Requirement)


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:24] or "doc"


def _write(path: Path, text: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)                  # atomic: readers never see half a file


# ---- documents -------------------------------------------------------------
def doc_dir(doc_id: str) -> Path:
    return config.LIBRARY_DIR / doc_id


def parse_pages(spec: str) -> list[int]:
    """"23-25,28-47,119" -> 1-based page numbers."""
    pages: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        a, _, b = part.partition("-")
        pages.extend(range(int(a), int(b or a) + 1))
    return pages


def add_document(pdf_path: Path, name: str, label: str, code: str = "",
                 edition: str = "", pages: str = "") -> DocumentMeta:
    """Register a PDF. Re-adding the same file returns the existing entry.
    `pages` ("23-25,28-47") stores only those pages, keeping a map back to
    the original page numbers so citations stay right."""
    import fitz

    data = Path(pdf_path).read_bytes()
    page_map: list[int] = []
    if pages:
        page_map = parse_pages(pages)
        src = fitz.open(stream=data, filetype="pdf")
        cut = fitz.open()
        for p in page_map:
            cut.insert_pdf(src, from_page=p - 1, to_page=p - 1)
        data = cut.tobytes(garbage=3, deflate=True)
    sha = hashlib.sha256(data).hexdigest()
    for meta in list_documents():
        if meta.sha256 == sha:
            return meta
    doc_id = f"{_slug(label)}-{sha[:8]}"
    d = doc_dir(doc_id)
    (d / "figures").mkdir(parents=True, exist_ok=True)
    (d / "source.pdf").write_bytes(data)
    meta = DocumentMeta(doc_id=doc_id, name=name, label=label,
                        filename=Path(pdf_path).name, sha256=sha, code=code,
                        edition=edition, page_map=page_map, created_at=now())
    save_meta(meta)
    return meta


def save_meta(meta: DocumentMeta) -> None:
    _write(doc_dir(meta.doc_id) / "meta.json", meta.model_dump_json(indent=2))


def load_meta(doc_id: str) -> DocumentMeta:
    return DocumentMeta.model_validate_json(
        (doc_dir(doc_id) / "meta.json").read_text(encoding="utf-8"))


def list_documents() -> list[DocumentMeta]:
    out = []
    for p in sorted(config.LIBRARY_DIR.glob("*/meta.json")):
        out.append(DocumentMeta.model_validate_json(
            p.read_text(encoding="utf-8")))
    return out


def delete_document(doc_id: str) -> None:
    shutil.rmtree(doc_dir(doc_id), ignore_errors=True)


def save_parsed(parsed: ParsedDocument) -> None:
    _write(doc_dir(parsed.doc_id) / "parsed.json", parsed.model_dump_json())


def load_parsed(doc_id: str) -> ParsedDocument | None:
    p = doc_dir(doc_id) / "parsed.json"
    return ParsedDocument.model_validate_json(
        p.read_text(encoding="utf-8")) if p.exists() else None


def save_json(doc_id: str, name: str, obj) -> None:
    _write(doc_dir(doc_id) / name, json.dumps(obj, indent=1,
                                              ensure_ascii=False))


def load_json(doc_id: str, name: str):
    p = doc_dir(doc_id) / name
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def save_requirements(doc_id: str, reqs: list[Requirement], key: str) -> None:
    save_json(doc_id, "requirements.json", {
        "key": key, "requirements": [r.model_dump() for r in reqs]})


def load_requirements(doc_id: str, key: str | None = None
                      ) -> list[Requirement] | None:
    """Stored requirements, or None if absent (or made with another key)."""
    data = load_json(doc_id, "requirements.json")
    if data is None or (key is not None and data.get("key") != key):
        return None
    return [Requirement.model_validate(r) for r in data["requirements"]]


def save_embeddings(doc_id: str, ids: list[str], doc_vecs: np.ndarray,
                    query_vecs: np.ndarray, key: str) -> None:
    np.savez(doc_dir(doc_id) / "embeddings.npz", ids=np.array(ids),
             doc=doc_vecs, query=query_vecs, key=np.array(key))


def load_embeddings(doc_id: str, key: str):
    p = doc_dir(doc_id) / "embeddings.npz"
    if not p.exists():
        return None
    z = np.load(p)
    if str(z["key"]) != key:
        return None
    return list(z["ids"]), z["doc"], z["query"]


# ---- comparisons -----------------------------------------------------------
def cmp_dir(cmp_id: str) -> Path:
    return config.COMPARISON_DIR / cmp_id


def new_comparison(doc_a: str, doc_b: str, governing: str) -> ComparisonMeta:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    meta = ComparisonMeta(cmp_id=f"cmp-{stamp}", doc_a=doc_a, doc_b=doc_b,
                          governing=governing, created_at=now())
    cmp_dir(meta.cmp_id).mkdir(parents=True, exist_ok=True)
    save_comparison(meta)
    return meta


def save_comparison(meta: ComparisonMeta) -> None:
    _write(cmp_dir(meta.cmp_id) / "meta.json", meta.model_dump_json(indent=2))


def load_comparison(cmp_id: str) -> ComparisonMeta:
    return ComparisonMeta.model_validate_json(
        (cmp_dir(cmp_id) / "meta.json").read_text(encoding="utf-8"))


def list_comparisons() -> list[ComparisonMeta]:
    return [ComparisonMeta.model_validate_json(p.read_text(encoding="utf-8"))
            for p in sorted(config.COMPARISON_DIR.glob("*/meta.json"))]


def save_findings(cmp_id: str, findings: list[Finding]) -> None:
    _write(cmp_dir(cmp_id) / "findings.json", json.dumps(
        [f.model_dump() for f in findings], indent=1, ensure_ascii=False))


def load_findings(cmp_id: str) -> list[Finding]:
    p = cmp_dir(cmp_id) / "findings.json"
    if not p.exists():
        return []
    return [Finding.model_validate(f)
            for f in json.loads(p.read_text(encoding="utf-8"))]


def save_cmp_json(cmp_id: str, name: str, obj) -> None:
    _write(cmp_dir(cmp_id) / name, json.dumps(obj, indent=1,
                                              ensure_ascii=False))


def load_reviews(cmp_id: str) -> dict:
    """Engineer decisions per finding, {finding_id: {decision, note, by,
    at}}. Kept apart from findings.json so a re-run of the comparison does
    not overwrite what a person decided."""
    p = cmp_dir(cmp_id) / "reviews.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def save_reviews(cmp_id: str, reviews: dict) -> None:
    save_cmp_json(cmp_id, "reviews.json", reviews)
