"""Command line front end.

    codesync setup
    codesync add <pdf> --label BP --name "BP WBU-GP 04-20-1"
    codesync list
    codesync ingest <doc_id> [<doc_id> ...]
    codesync compare <doc_a> <doc_b> --governing <doc_a>
    codesync serve [--port 8000]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from . import config, library


def _say(msg: str) -> None:
    print(f"{time.strftime('%H:%M:%S')} {msg}", flush=True)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="codesync", description=__doc__,
                                 formatter_class=argparse.RawTextHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("add", help="register a code PDF")
    p.add_argument("pdf", type=Path)
    p.add_argument("--label", required=True, help="short tag, e.g. BP")
    p.add_argument("--name", help="display name (default: file name)")
    p.add_argument("--code", default="", help='the standard, e.g. "ASME B31.1"')
    p.add_argument("--edition", default="", help='its edition, e.g. "2007"')
    p.add_argument("--pages", default="",
                   help='keep only these pages, e.g. "23-25,28-47"')

    sub.add_parser("list", help="list documents and comparisons")

    p = sub.add_parser("ingest", help="parse, read figures, extract, embed")
    p.add_argument("doc_ids", nargs="+")
    p.add_argument("--figure-reader", choices=("modal", "llm", "local"),
                   default=config.FIGURE_READER)

    p = sub.add_parser("compare", help="compare two ingested documents")
    p.add_argument("doc_a")
    p.add_argument("doc_b")
    p.add_argument("--governing", help="doc id that takes precedence "
                                       "(default: doc_a)")
    p.add_argument("--figure-reader", choices=("modal", "llm", "local"),
                   default=config.FIGURE_READER)

    p = sub.add_parser("index", help="build the Ask-the-Code library for documents")
    p.add_argument("doc_ids", nargs="+")

    p = sub.add_parser("ask", help="ask a question of the indexed codes")
    p.add_argument("question")
    p.add_argument("--doc", action="append", default=None, help="limit to document ids")
    p.add_argument("--no-answer", action="store_true", help="show the passages only")

    sub.add_parser("setup", help="optional: llama.cpp + GGUF for the local "
                                 "developer-fallback drawing reader")

    p = sub.add_parser("serve", help="run the HTTP API")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)

    args = ap.parse_args(argv)

    if args.cmd == "add":
        meta = library.add_document(args.pdf, args.name or args.pdf.stem,
                                    args.label, code=args.code,
                                    edition=args.edition, pages=args.pages)
        print(meta.doc_id)
    elif args.cmd == "list":
        for m in library.list_documents():
            print(f"{m.doc_id:22s} {m.status:10s} {m.label:6s} {m.name}")
        for c in library.list_comparisons():
            print(f"{c.cmp_id:22s} {c.status:10s} {c.doc_a} vs {c.doc_b}")
    elif args.cmd == "ingest":
        from . import pipeline
        metas = pipeline.ingest(args.doc_ids, args.figure_reader,
                                progress=_say)
        for m in metas:
            print(json.dumps({"doc_id": m.doc_id, "status": m.status,
                              "counts": m.counts, "timings": m.timings,
                              "error": m.error}, indent=1))
        return 0 if all(m.status == "ready" for m in metas) else 1
    elif args.cmd == "compare":
        from . import pipeline
        cmp = library.new_comparison(args.doc_a, args.doc_b,
                                     args.governing or args.doc_a)
        cmp = pipeline.compare(cmp.cmp_id, args.figure_reader, progress=_say)
        print(json.dumps({"cmp_id": cmp.cmp_id, "status": cmp.status,
                          "summary": cmp.summary, "timings": cmp.timings,
                          "error": cmp.error}, indent=1))
        if cmp.status == "done":
            print(f"report: {library.cmp_dir(cmp.cmp_id) / 'report.md'}")
        return 0 if cmp.status == "done" else 1
    elif args.cmd == "index":
        from .models.chat import ModalLLM, ModalVLM
        from .models.embedder import Embedder
        from .stages.index import index_document
        llm, vlm, emb = ModalLLM(), ModalVLM(), Embedder()
        llm.warm()
        for d in args.doc_ids:
            print(json.dumps(index_document(d, llm, vlm, emb, progress=_say), indent=1))
    elif args.cmd == "ask":
        from .models.embedder import Embedder
        from .stages.answer import check_citations, stream_answer
        from .stages.retrieve import retrieve
        t = time.time()
        found = retrieve(args.question, Embedder(), args.doc)
        print(f"-- {len(found['passages'])} passages ({found['candidates']} candidates) "
              f"in {time.time() - t:.1f}s {found['timings']}")
        for p in found["passages"]:
            c = p["chunk"]
            print(f"[{p['tag']}] {p['why']:10s} score={p['score']} v#{p['vector_rank']} k#{p['keyword_rank']} "
                  f"| {c.code} {c.edition} {c.ref} p.{c.page_label or c.page} | {c.text[:110]!r}")
        if not args.no_answer and found["passages"]:
            print("--")
            answer = ""
            for piece in stream_answer(args.question, found["passages"]):
                if "delta" in piece:
                    answer += piece["delta"]
                    print(piece["delta"], end="", flush=True)
            print()
            print("--", json.dumps(check_citations(answer, found["passages"])))
    elif args.cmd == "setup":
        from . import setup_local
        return setup_local.main()
    elif args.cmd == "serve":
        import uvicorn
        uvicorn.run("codesync.api:app", host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    sys.exit(main())
