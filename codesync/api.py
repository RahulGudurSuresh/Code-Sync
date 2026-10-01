"""HTTP API for the frontend.

    uvicorn codesync.api:app --port 8000      (or: codesync serve)

Uploads and comparisons return immediately with a job; the frontend polls
`/jobs/{id}` (or the document / comparison itself, whose `status` and
`stage` fields are updated as the pipeline moves).
"""
from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import (FileResponse, PlainTextResponse,
                               RedirectResponse, StreamingResponse)
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import __version__, config, library, pipeline
from .jobs import runner

app = FastAPI(title="CodeSync Analyzer API", version=__version__)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"],
                   allow_headers=["*"])


def _doc(doc_id: str):
    try:
        return library.load_meta(doc_id)
    except FileNotFoundError:
        raise HTTPException(404, f"document {doc_id!r} not found") from None


def _cmp(cmp_id: str):
    try:
        return library.load_comparison(cmp_id)
    except FileNotFoundError:
        raise HTTPException(404, f"comparison {cmp_id!r} not found") from None


def _safe_name(name: str) -> str:
    if Path(name).name != name:
        raise HTTPException(400, "invalid file name")
    return name


# ---- service ---------------------------------------------------------------
@app.get("/health")
def health() -> dict:
    return {
        "status": "ok", "version": __version__,
        "engine": f"modal:{config.MODAL_APP}/{config.MODAL_CLASS}",
        "embedding": config.EMBED_MODEL,
        "figure_reader": config.FIGURE_READER,
        "local_reader_available": config.LLAMA_SERVER.exists(),
        "documents": len(library.list_documents()),
        "comparisons": len(library.list_comparisons()),
    }


@app.get("/jobs")
def list_jobs() -> list[dict]:
    return runner.list()


@app.get("/jobs/{job_id}")
def get_job(job_id: str) -> dict:
    job = runner.get(job_id)
    if job is None:
        raise HTTPException(404, f"job {job_id!r} not found")
    return job


# ---- documents -------------------------------------------------------------
@app.post("/documents")
async def upload_document(file: UploadFile = File(...),
                          label: str = Form(...), name: str = Form(""),
                          code: str = Form(""), edition: str = Form(""),
                          pages: str = Form(""),
                          ingest: bool = Form(True)) -> dict:
    """Register a code PDF and (by default) start ingesting it."""
    if not (file.filename or "").lower().endswith(".pdf"):
        raise HTTPException(400, "only PDF files are accepted")
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / Path(file.filename).name
        path.write_bytes(await file.read())
        try:
            meta = library.add_document(path, name or path.stem, label,
                                        code=code, edition=edition, pages=pages)
        except ValueError as e:
            raise HTTPException(400, f"bad page range: {e}") from None
    job = None
    if ingest and meta.status != "ready":
        job = runner.submit("ingest", meta.doc_id, lambda progress:
                            pipeline.ingest([meta.doc_id], progress=progress))
    return {"document": meta, "job": job}


@app.post("/documents/{doc_id}/ingest")
def ingest_document(doc_id: str) -> dict:
    meta = _doc(doc_id)
    return runner.submit("ingest", meta.doc_id, lambda progress:
                         pipeline.ingest([meta.doc_id], progress=progress))


@app.get("/documents")
def list_documents() -> list:
    return library.list_documents()


@app.get("/documents/{doc_id}")
def get_document(doc_id: str):
    return _doc(doc_id)


@app.delete("/documents/{doc_id}")
def delete_document(doc_id: str) -> dict:
    _doc(doc_id)
    library.delete_document(doc_id)
    return {"deleted": doc_id}


@app.get("/documents/{doc_id}/elements")
def get_elements(doc_id: str, kind: str | None = None) -> dict:
    _doc(doc_id)
    parsed = library.load_parsed(doc_id)
    if parsed is None:
        raise HTTPException(409, "document has not been parsed yet")
    elements = [e for e in parsed.elements if kind in (None, e.kind)]
    return {"pages": parsed.pages, "elements": elements}


@app.get("/documents/{doc_id}/requirements")
def get_requirements(doc_id: str) -> list:
    _doc(doc_id)
    reqs = library.load_requirements(doc_id)
    if reqs is None:
        raise HTTPException(409, "requirements have not been extracted yet")
    return reqs


@app.get("/documents/{doc_id}/figures/{name}")
def get_figure(doc_id: str, name: str) -> FileResponse:
    path = library.doc_dir(_doc(doc_id).doc_id) / "figures" / _safe_name(name)
    if not path.exists():
        raise HTTPException(404, "figure not found")
    return FileResponse(path, media_type="image/png")


@app.get("/documents/{doc_id}/pdf")
def get_pdf(doc_id: str) -> FileResponse:
    meta = _doc(doc_id)
    return FileResponse(library.doc_dir(doc_id) / "source.pdf",
                        media_type="application/pdf", filename=meta.filename)


# ---- comparisons -----------------------------------------------------------
class ComparisonRequest(BaseModel):
    doc_a: str
    doc_b: str
    governing: str | None = None       # doc id that takes precedence


@app.post("/comparisons")
def start_comparison(req: ComparisonRequest) -> dict:
    _doc(req.doc_a)
    _doc(req.doc_b)
    if req.doc_a == req.doc_b:
        raise HTTPException(400, "choose two different documents")
    governing = req.governing or req.doc_a
    if governing not in (req.doc_a, req.doc_b):
        raise HTTPException(400, "governing must be doc_a or doc_b")
    cmp = library.new_comparison(req.doc_a, req.doc_b, governing)
    job = runner.submit("compare", cmp.cmp_id, lambda progress:
                        pipeline.compare(cmp.cmp_id, progress=progress))
    return {"comparison": cmp, "job": job}


@app.get("/comparisons")
def list_comparisons() -> list:
    return library.list_comparisons()


@app.get("/comparisons/{cmp_id}")
def get_comparison(cmp_id: str):
    return _cmp(cmp_id)


@app.get("/comparisons/{cmp_id}/findings")
def get_findings(cmp_id: str, category: str | None = None,
                 type: str | None = None, severity: str | None = None,
                 needs_review: bool | None = None) -> list:
    _cmp(cmp_id)
    return [f for f in library.load_findings(cmp_id)
            if category in (None, f.category) and type in (None, f.type)
            and severity in (None, f.severity)
            and needs_review in (None, f.needs_review)]


@app.get("/comparisons/{cmp_id}/report.md")
def get_report_md(cmp_id: str) -> PlainTextResponse:
    path = library.cmp_dir(_cmp(cmp_id).cmp_id) / "report.md"
    if not path.exists():
        raise HTTPException(409, "report is not ready")
    return PlainTextResponse(path.read_text(encoding="utf-8"),
                             media_type="text/markdown")


@app.get("/comparisons/{cmp_id}/report.xlsx")
def get_report_xlsx(cmp_id: str) -> FileResponse:
    path = library.cmp_dir(_cmp(cmp_id).cmp_id) / "report.xlsx"
    if not path.exists():
        raise HTTPException(409, "report is not ready")
    return FileResponse(
        path, filename=f"{cmp_id}.xlsx", media_type="application/vnd."
        "openxmlformats-officedocument.spreadsheetml.sheet")


# ---- engineer review of findings -------------------------------------------
class ReviewDecision(BaseModel):
    decision: Literal["accepted", "rejected", ""] = ""
    note: str = ""
    by: str = ""


@app.get("/comparisons/{cmp_id}/reviews")
def get_reviews(cmp_id: str) -> dict:
    """{finding_id: {decision, note, by, at}} as saved by reviewers."""
    _cmp(cmp_id)
    return library.load_reviews(cmp_id)


@app.put("/comparisons/{cmp_id}/reviews/{finding_id}")
def put_review(cmp_id: str, finding_id: str, review: ReviewDecision) -> dict:
    _cmp(cmp_id)
    if not any(f.id == finding_id for f in library.load_findings(cmp_id)):
        raise HTTPException(404, f"finding {finding_id!r} not found")
    reviews = library.load_reviews(cmp_id)
    reviews[finding_id] = {**review.model_dump(), "at": library.now()}
    library.save_reviews(cmp_id, reviews)
    return reviews[finding_id]


# ---- the built frontend ----------------------------------------------------
# `cd frontend && npm run build` writes frontend/dist; it is served at /app.
# In development, `npm run dev` serves it with a proxy to this API instead.
_DIST = config.ROOT / "frontend" / "dist"
if _DIST.exists():
    app.mount("/app/assets", StaticFiles(directory=_DIST / "assets"),
              name="app-assets")

    @app.get("/app/{path:path}", include_in_schema=False)
    def spa(path: str) -> FileResponse:
        """The built single-page app. Any path that is not a file falls
        back to index.html so a refreshed deep link still loads."""
        target = (_DIST / path).resolve()
        if path and target.is_file() and _DIST.resolve() in target.parents:
            return FileResponse(target)
        return FileResponse(_DIST / "index.html")

    @app.get("/", include_in_schema=False)
    def root() -> RedirectResponse:
        return RedirectResponse("/app/")


# ---- Ask the Code ------------------------------------------------------------
class AskRequest(BaseModel):
    question: str
    doc_ids: list[str] | None = None   # scope; None = the whole library
    code: str | None = None
    edition: str | None = None


@app.post("/documents/{doc_id}/index")
def index_document(doc_id: str) -> dict:
    """Build the Ask-the-Code library for a document (chunks, descriptions,
    embeddings, store rows). Runs as a job."""
    meta = _doc(doc_id)
    from .models.chat import ModalLLM, ModalVLM
    from .models.embedder import Embedder
    from .stages.index import index_document as run_index
    return runner.submit("index", meta.doc_id, lambda progress: run_index(
        meta.doc_id, ModalLLM(), ModalVLM(), Embedder(), progress))


@app.get("/ask/scope")
def ask_scope() -> list[dict]:
    """Documents that can be asked about, with their chunk counts."""
    from . import store
    counts = store.indexed_documents()
    return [{"doc_id": m.doc_id, "label": m.label, "name": m.name, "code": m.code,
             "edition": m.edition, "chunks": counts.get(m.doc_id, 0)}
            for m in library.list_documents() if counts.get(m.doc_id)]


@app.post("/ask/warm")
def ask_warm() -> dict:
    """Start the GPU engine booting (the Ask page calls this when it opens,
    so the first question does not pay the whole cold start)."""
    from .models.chat import ModalLLM
    ModalLLM().warm()
    return {"warming": True}


@app.post("/ask")
def ask(req: AskRequest):
    """Server-sent events: status, sources, answer deltas, then done."""
    import json as _json

    from .models.embedder import Embedder
    from .stages.answer import check_citations, stream_answer
    from .stages.retrieve import retrieve

    question = req.question.strip()
    if not question:
        raise HTTPException(400, "empty question")

    def event(kind: str, data) -> str:
        return f"event: {kind}\ndata: {_json.dumps(data, ensure_ascii=False)}\n\n"

    def gen():
        yield event("status", {"stage": "searching"})
        try:
            found = retrieve(question, Embedder(), req.doc_ids, req.code, req.edition)
        except Exception as e:                           # noqa: BLE001
            yield event("error", {"message": f"search failed: {type(e).__name__}: {e}"})
            return
        passages = found["passages"]
        yield event("sources", {
            "passages": [{**{k: v for k, v in p.items() if k != "chunk"},
                          "chunk": p["chunk"].model_dump()} for p in passages],
            "timings": found["timings"], "candidates": found["candidates"]})
        if not passages:
            yield event("delta", {"text": "Not found in the passages supplied: the library has nothing close to this question."})
            yield event("done", {"check": {"ok": True, "cited": [], "bad_tags": [], "bad_quotes": [], "not_found": True}})
            return
        yield event("status", {"stage": "answering"})
        answer = ""
        meta: dict = {}
        try:
            for piece in stream_answer(question, passages):
                if "delta" in piece:
                    answer += piece["delta"]
                    yield event("delta", {"text": piece["delta"]})
                if piece.get("done"):
                    meta = piece
        except Exception as e:                           # noqa: BLE001
            yield event("error", {"message": f"answer failed: {type(e).__name__}: {e}"})
            return
        check = check_citations(answer, passages)
        yield event("done", {"check": check, "usage": {k: meta.get(k) for k in (
            "finish_reason", "prompt_tokens", "completion_tokens", "stream_fallback")}})

    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache",
                                      "X-Accel-Buffering": "no"})
