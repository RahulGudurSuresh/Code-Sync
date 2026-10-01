"""End-to-end orchestration: ingest documents, compare two of them.

Stages run across all documents before moving on (parse everything, read
every figure, extract, then embed). The Modal container is asked to boot as
soon as a job starts, so its cold start overlaps the local parsing, and it
stays warm while work keeps arriving.
"""
from __future__ import annotations

import hashlib
import json
import time
import traceback
from collections.abc import Callable

from . import config, library
from .models.chat import LocalVLM, ModalLLM, ModalVLM
from .models.embedder import Embedder
from .pdfparse import parse_pdf
from .schema import ComparisonMeta, DocumentMeta
from .stages import report
from .stages.extract import extract_requirements, extraction_key
from .stages.figures import read_figures
from .stages.judge import build_findings, judge_pairs, review_matches
from .stages.match import candidate_pairs, embed_requirements

Progress = Callable[[str], None]


def _stage(meta: DocumentMeta, stage: str, say: Progress) -> None:
    meta.stage = stage
    library.save_meta(meta)
    say(f"[{meta.label}] {stage}")


def _readings_key(llm_name: str, readings: list[dict]) -> str:
    digest = hashlib.sha256(json.dumps(
        readings, sort_keys=True).encode("utf-8")).hexdigest()[:12]
    return f"{extraction_key(llm_name)}|figures:{digest}"


def ingest(doc_ids: list[str], figure_reader: str = config.FIGURE_READER,
           llm: ModalLLM | None = None, progress: Progress | None = None,
           embed: bool = True) -> list[DocumentMeta]:
    """Bring documents to "ready": parsed, figures read, requirements
    extracted and embedded. Every stage is skipped when its output is
    already on disk for the same inputs."""
    say = progress or (lambda _m: None)
    llm = llm or ModalLLM()
    llm.warm()                     # boot the engine while we parse
    metas = [library.load_meta(d) for d in doc_ids]
    live = list(metas)

    def guard(meta: DocumentMeta, fn: Callable[[], None]) -> None:
        try:
            fn()
        except Exception as e:                           # noqa: BLE001
            meta.status, meta.stage = "failed", ""
            meta.error = f"{type(e).__name__}: {e}"
            library.save_meta(meta)
            say(f"[{meta.label}] FAILED: {meta.error}\n"
                + traceback.format_exc(limit=4))
            live.remove(meta)

    # 1. parse ---------------------------------------------------------------
    parsed = {}
    for meta in list(live):
        def step(meta=meta):
            p = library.load_parsed(meta.doc_id)
            if p is None:
                _stage(meta, "parsing", say)
                t = time.time()
                d = library.doc_dir(meta.doc_id)
                p = parse_pdf(d / "source.pdf", meta.doc_id, d / "figures",
                              progress=lambda m: say(f"[{meta.label}] {m}"))
                library.save_parsed(p)
                meta.timings["parse_s"] = round(time.time() - t, 1)
            parsed[meta.doc_id] = p
            kinds = [pg.kind for pg in p.pages]
            meta.pages = len(p.pages)
            meta.counts.update(
                clauses=sum(e.kind == "clause" for e in p.elements),
                tables=sum(e.kind == "table" for e in p.elements),
                figures=sum(e.kind == "figure" for e in p.elements),
                pages_text=kinds.count("text"),
                pages_drawing=kinds.count("drawing"),
                pages_scanned=kinds.count("scanned"))
            meta.status = "parsed"
            library.save_meta(meta)
        guard(meta, step)

    # 2. figures -------------------------------------------------------------
    local = figure_reader == "local"
    reader = {"modal": ModalVLM, "local": LocalVLM,
              "llm": lambda: llm}[figure_reader]()
    tokens = config.VLM_IMG_TOKENS if local else config.LLM_IMG_TOKENS
    readings = {}
    for meta in list(live):
        def step(meta=meta):
            if meta.counts.get("figures"):
                _stage(meta, f"reading figures ({reader.name})", say)
            t = time.time()
            readings[meta.doc_id] = read_figures(
                meta, parsed[meta.doc_id], reader, tokens,
                progress=lambda m: say(f"[{meta.label}] {m}"))
            if time.time() - t > 1:
                meta.timings["figures_s"] = round(time.time() - t, 1)
            meta.counts["figure_reader"] = reader.name
            meta.status = "figures"
            library.save_meta(meta)
        guard(meta, step)
    if local:
        from .models.llama_server import local_models
        local_models.release()

    # 3. requirements --------------------------------------------------------
    reqs = {}
    for meta in list(live):
        def step(meta=meta):
            key = _readings_key(llm.name, readings[meta.doc_id])
            r = library.load_requirements(meta.doc_id, key)
            if r is None:
                _stage(meta, "extracting requirements", say)
                t = time.time()
                r, stats = extract_requirements(
                    meta, parsed[meta.doc_id], readings[meta.doc_id], llm,
                    progress=lambda m: say(f"[{meta.label}] {m}"))
                library.save_requirements(meta.doc_id, r, key)
                meta.timings["extract_s"] = round(time.time() - t, 1)
                meta.counts["extraction"] = stats
            reqs[meta.doc_id] = r
            meta.counts["requirements"] = len(r)
            meta.counts["set_aside"] = sum(x.excluded for x in r)
            meta.status = "extracted"
            library.save_meta(meta)
        guard(meta, step)

    # 4. embeddings ----------------------------------------------------------
    if embed:
        embedder = Embedder()
        for meta in list(live):
            def step(meta=meta):
                _stage(meta, "embedding", say)
                t = time.time()
                # the same subset compare() uses, so its cache key matches
                embed_requirements(
                    meta.doc_id,
                    [r for r in reqs[meta.doc_id] if not r.excluded],
                    embedder)
                if time.time() - t > 1:
                    meta.timings["embed_s"] = round(time.time() - t, 1)
                meta.status, meta.stage, meta.error = "ready", "", ""
                library.save_meta(meta)
            guard(meta, step)
    return metas


def compare(cmp_id: str, figure_reader: str = config.FIGURE_READER,
            progress: Progress | None = None) -> ComparisonMeta:
    say = progress or (lambda _m: None)
    cmp = library.load_comparison(cmp_id)
    llm = ModalLLM()
    llm.warm()

    def stage(name: str) -> None:
        cmp.stage = name
        library.save_comparison(cmp)
        say(f"[compare] {name}")

    try:
        cmp.status = "running"
        t_all = time.time()
        stage("ingesting documents")
        metas = {m.doc_id: m for m in ingest(
            [cmp.doc_a, cmp.doc_b], figure_reader, llm, progress)}
        for m in metas.values():
            if m.status != "ready":
                raise RuntimeError(f"{m.label} is not ready: {m.error}")
        meta_a, meta_b = metas[cmp.doc_a], metas[cmp.doc_b]
        governing = metas[cmp.governing]
        all_a = library.load_requirements(cmp.doc_a)
        all_b = library.load_requirements(cmp.doc_b)
        # drawing statements the two readings disagreed on stay out of the
        # comparison; they are listed in the report for manual reading
        reqs_a = [r for r in all_a if not r.excluded]
        reqs_b = [r for r in all_b if not r.excluded]

        stage("matching")
        embedder = Embedder()              # cache hits: no call is made
        emb_a = embed_requirements(cmp.doc_a, reqs_a, embedder)
        emb_b = embed_requirements(cmp.doc_b, reqs_b, embedder)
        pairs, _sim = candidate_pairs(emb_a, emb_b)

        stage("judging")
        t = time.time()
        judged, jstats = judge_pairs(meta_a, meta_b, reqs_a, reqs_b, pairs,
                                     llm, progress)
        cmp.timings["judge_s"] = round(time.time() - t, 1)

        stage("reviewing matches")
        t = time.time()
        rstats = review_matches(judged, meta_a, meta_b, reqs_a, reqs_b,
                                governing, llm, progress)
        cmp.timings["review_s"] = round(time.time() - t, 1)
        findings = build_findings(meta_a, meta_b, reqs_a, reqs_b, judged,
                                  governing)

        stage("writing report")
        summary = report.summarize(findings, all_a, all_b)
        summary.update(candidate_pairs=len(pairs), judge=jstats,
                       review=rstats, llm=llm.stats,
                       relations={r: sum(p["relation"] == r for p in judged)
                                  for r in sorted({p["relation"]
                                                   for p in judged})})
        library.save_cmp_json(cmp_id, "pairs.json", judged)
        library.save_findings(cmp_id, findings)
        d = library.cmp_dir(cmp_id)
        (d / "report.md").write_text(report.markdown_report(
            meta_a, meta_b, governing, findings, summary, all_a + all_b),
            encoding="utf-8")
        report.excel_report(d / "report.xlsx", meta_a, meta_b, governing,
                            findings, all_a, all_b, summary)
        cmp.summary = summary
        cmp.timings["total_s"] = round(time.time() - t_all, 1)
        cmp.status, cmp.stage, cmp.error = "done", "", ""
    except Exception as e:                               # noqa: BLE001
        cmp.status, cmp.stage = "failed", ""
        cmp.error = f"{type(e).__name__}: {e}"
        say(f"[compare] FAILED: {cmp.error}\n" + traceback.format_exc(limit=6))
    finally:
        library.save_comparison(cmp)
    return cmp
