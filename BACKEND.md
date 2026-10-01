# CodeSync Analyzer: backend and web app

Two things, on one library of parsed codes:

1. **Compare** two engineering codes and report where they conflict, where
   they agree, and what each says that the other does not. Built and
   verified on the two test codes in `codes/test_codes/` (BP WBU-GP 04-20-1
   section 11.6 against PIP STF05501).
2. **Ask the code**: a question in plain words, answered only from the
   passages found in the indexed codes, with the paragraph and page cited,
   equations quoted as written and tables shown as data. Built and verified
   on a 28-page sample of ASME B31.1-2007 (a scanned book).

The web app in `frontend/` is the screen on top of both.

## What runs where

| Stage | What it does | Model | Runs on |
|---|---|---|---|
| Parse | Text layer to a clause tree with citations (`11.6 a.3.c.ii`); page routing; layout, OCR, tables, equations | PyMuPDF, PP-DocLayoutV3, PP-OCRv5, PaddleOCR table pipeline, PP-FormulaNet | Local CPU |
| Read figures | Each drawing sheet to the requirements and arrangements drawn on it | Qwen3.5-9B (vLLM) | Modal engine |
| Extract | Clauses and tables to atomic requirements, each with a verbatim quote | Qwen3.8-27B NVFP4 (vLLM) | Modal engine |
| Check drawings | Every drawing statement re-read against its sheet | Qwen3.8-27B (it is a vision model too) | Modal engine |
| Embed | Requirement to vector, once as a query and once as a document | Qwen3-Embedding-8B, full precision (transformers) | Modal engine |
| Match | Top candidates per requirement in both directions | none | Local CPU |
| Judge | Relation of each candidate pair | Qwen3.8-27B | Modal engine |
| Review | Second look at every claimed match, with the clause text around it | Qwen3.8-27B | Modal engine |
| Report | `findings.json`, `report.md`, `report.xlsx` | none | Local CPU |

Ask the code adds:

| Stage | What it does | Model | Runs on |
|---|---|---|---|
| Chunk | Parsed elements to one chunk per clause, with heading path, lead-in, notes, equations (LaTeX), cross-references as links, tables as grids | none | Local CPU |
| Describe | A description ADDED beside each figure and table (never instead of its text) | Qwen3.5-9B for figures, Qwen3.8-27B for tables | Modal engine |
| Store | Chunk text, fields and vector in LanceDB (`data/lancedb`), with a BM25 index | Qwen3-Embedding-8B | Modal engine + local |
| Retrieve | Vector top 30 + keyword top 30, fused, reranked to 6-8; then one hop along the references and the definitions of the terms used | Qwen3-Reranker-0.6B | Modal engine |
| Answer | Written from the passages only, streamed token by token, every `[Pn]` citation and every quotation checked afterwards | Qwen3.8-27B | Modal engine |

"Modal engine" is one container on one RTX PRO 6000 (96 GB) holding all
four models: the 27B gets 40% of the card, the 9B 30%, the embedder about
17 GB, the reranker about 1.5 GB, and about 10 GB stays free. Measured at
boot: 84 GiB in use with nothing running. No GPU is needed on the machine
that runs the backend; the parsing models use the CPU.

## Setup

```
python -m venv .venv --system-site-packages
.venv\Scripts\pip install -e .[dev]
.venv\Scripts\modal deploy modal_app\codesync_modal.py
```

The venv reuses the system Python's PaddleOCR, PyMuPDF and torch (hence
`--system-site-packages`); on this machine the install adds only Modal,
FastAPI and the PaddleX OCR extras. A from-scratch install on another
machine has not been tried.

The Modal app (`codesync-llm`) is separate from the DTSPL apps: own name, own
weights volume (`codesync-qwen38-nvfp4-weights`, now holding all three
models, about 55 GB), own compile cache (`codesync-vllm-compile`). It uses
the DTSPL recipe: vLLM 0.27.1 in-process, GPU memory snapshot after a
warm-up, sleep mode never invoked, two-minute scaledown. It is deployed in
the `gsrahul1998` Modal profile. The first boot after a deploy loads and
compiles all three models (about 5 minutes with warm caches).

There is an optional local drawing reader for development without Modal
spend (`FIGURE_READER=local`, `codesync setup` fetches llama.cpp and the
Qwen3.5-9B GGUF, about 8.5 GB). It is a developer fallback, not a supported
mode: the judging still needs Modal, and it is slower.

## Running it

Web app (build once, then the backend serves it):

```
cd frontend && npm install && npm run build && cd ..
.venv\Scripts\codesync serve                        # http://127.0.0.1:8000/  ->  /app/
```

For frontend development, `cd frontend && npm run dev` serves it at
http://127.0.0.1:5173/app/ with the API proxied to port 8000.

Pages:

| Page | What it does |
|---|---|
| Library (`/app/`) | Upload a PDF with a short label; watch it move through parsing, drawings, extraction and embedding. |
| Document (`/app/documents/<id>`) | The clause tree with citations and page numbers, every extracted requirement with its verbatim quote and check status, and each drawing sheet with what was read from it. |
| Comparisons (`/app/comparisons`) | Pick A and B and which one governs; start the run; list past runs. |
| Comparison (`/app/comparisons/<id>`) | Live stage progress while running. When done: summary tiles; tabs for conflicts, similar, only-in-A, only-in-B, set-aside drawing statements, and the rendered report. Each finding shows both requirements with clause, page and basis, the drawing crop when one side is a drawing, the reason, the action under the governing document, and flags. |
| Ask (`/app/ask`) | Pick the codes to search, type a complete question. Stages (searching, answering) and timings are shown; the answer streams in with `[P1]` links to the passage cards; a banner says whether every citation and quotation checked out. Each passage card: code, edition, paragraph, printed page (linked into the PDF), heading path, verbatim text with equations rendered, notes, the table as a grid beside its image, the model's description (marked as such), and what it refers to. The question is kept in the URL (`?q=...&doc=...`) so an answer can be linked to. A document joins the search with "Index for Ask" on its page. |

The engineer's review lives on the finding cards: **Accept** / **Reject** and a
resolution note (RFI reference, decision). Decisions are saved per finding
(`PUT /comparisons/{id}/reviews/{finding_id}`) in `reviews.json`, apart
from `findings.json`, so a re-run of the comparison does not overwrite them.
Tabs are kept in the URL (`?tab=aside`), so a view can be linked to.

Command line:

```
codesync add "codes\test_codes\WBU-GP-04-20-1_platform_xtract.pdf" --label BP --name "BP WBU-GP 04-20-1"
codesync add "codes\test_codes\PIP STF05501 Fixed Ladders and Cages.pdf" --label PIP
codesync list
codesync compare <bp doc id> <pip doc id> --governing <bp doc id>

codesync add "codes\ASME B31.1-2007.pdf" --label B31.1 --code "ASME B31.1" --edition 2007 --pages 23-25,28-47,119-123
codesync index <b31.1 doc id>                 # parse, chunk, describe, embed, store
codesync ask "How do I calculate the minimum wall thickness of straight pipe under internal pressure?" --doc <b31.1 doc id>
codesync ask "..." --no-answer                 # the passages only
```

`--pages` cuts a sample out of a big PDF and keeps the map back to the
original pages. `--code` and `--edition` go on every chunk, so an answer can
say which edition it comes from and a search can be limited to one.

HTTP API (`codesync serve`, then `http://127.0.0.1:8000/docs` for the
interactive schema):

| Method and path | Purpose |
|---|---|
| `POST /documents` (multipart: `file`, `label`, `name`) | Register a PDF and start ingesting it. Returns the document and a job. |
| `GET /documents`, `GET /documents/{id}` | Status, stage, counts and timings |
| `GET /documents/{id}/elements?kind=clause` | Parsed clause tree, tables, figures |
| `GET /documents/{id}/requirements` | Extracted requirements |
| `GET /documents/{id}/figures/{name}`, `GET /documents/{id}/pdf` | Drawing crops and the source file |
| `POST /comparisons` (`{"doc_a", "doc_b", "governing"}`) | Start a comparison. Returns the comparison and a job. |
| `GET /comparisons/{id}` | Status, stage, summary counts |
| `GET /comparisons/{id}/findings?category=conflict&severity=high&needs_review=true` | Findings, filterable |
| `GET /comparisons/{id}/report.md`, `report.xlsx` | Reports |
| `GET /jobs/{id}` | Progress messages for polling |
| `GET /comparisons/{id}/reviews`, `PUT /comparisons/{id}/reviews/{finding_id}` | Engineer decisions per finding: `{"decision": "accepted" or "rejected" or "", "note": "..."}` |
| `POST /documents/{id}/index` | Build the Ask-the-Code chunks, descriptions, embeddings and store rows for a document (a job) |
| `GET /ask/scope` | The indexed documents with code, edition and chunk count |
| `POST /ask/warm` | Start the engine booting (the Ask page calls it on load) |
| `POST /ask` (`{"question", "doc_ids"}`) | Server-sent events: `status`, `sources` (the passages, before the answer), `delta` (answer text as it is generated), `done` (the citation check and token counts), `error` |
| `GET /app/...` | The built web app (needs `npm run build` first) |

Jobs run one at a time, because every job needs the one local GPU.

## What a finding is

| Category | Type | Meaning |
|---|---|---|
| conflict | contradiction | Same property of the same component, cannot both be satisfied |
| conflict | more_stringent | Both can be satisfied, one is stricter. Flagged for review when the stricter one is not the governing document. |
| similar | equivalent | Same requirement in substance |
| unique | only_in_a, only_in_b | No counterpart found in the other document |

Each conflict carries the governing document, an action sentence, a severity
and the model's reason. The governing document is chosen per comparison.

## How it guards against wrong findings

- **Clause citations come from the parser, not the model.** The LLM is given
  clause text tagged with references and must cite a tag; a requirement whose
  quote is not found verbatim in the source is marked `quote_verified: false`.
- **The judge must name the component and property on each side** before it
  may call a match, and a match it marked as "different component" is
  discarded.
- **Every claimed match is reviewed on its own** with the surrounding clause
  text. On the test codes the review rejects one to four of the six to nine
  claims per run (for example grating bar spacing against cage bar spacing).
- **A drawing statement is used only if two models agree on it.** The 9B
  vision model reads the sheet; the 27B then checks each statement against
  the image. Disputed statements are kept in the report under "set aside" and
  take no part in the comparison. On PIP STF05501 this set aside 51 of 155
  (34 read differently by the 27B, 17 it could not find on the sheet).
- **Runs of conflicting values are flagged.** Three or more values for the
  same property of the same component on one sheet, or three or more copies
  of the same sentence with only the numbers changed, mean a bare dimension
  was attached to the wrong label or the reader looped.

## How Ask the code keeps to the text

- **Nothing is rewritten.** Chunks are built from the parsed elements word
  for word; the chunker checks that every element lands in exactly one
  chunk and reports any that did not. Model-written descriptions of figures
  and tables are stored in their own field and shown marked as such.
- **Tables are data.** A table chunk carries its grid (columns and rows)
  and its image. When a question names a material, grade or temperature,
  the rows mentioning those come first in what the models read, so a value
  is looked up, not guessed. Notes under a table travel with it.
- **Equations are quoted, not computed.** Display equations are stored as
  LaTeX with their printed number; the clause that introduces one keeps
  the "where ..." nomenclature with it, even across a table printed in
  between or a column or page break.
- **Cross-references are links.** "see para. 102.4.5", "Table 104.1.2(A)",
  "eq. (3A)", "Appendix A" are extracted from each chunk and resolved to
  the chunk they name when it is in the library; the retrieval follows
  them one hop. Definitions ("arc welding: ...") are chunks of their own
  and are pulled in when the question or a passage uses the term.
- **Every chunk knows where it is**: code, edition, heading path, printed
  page number, the clause's own reference.
- **The answer is checked.** Only the supplied passages may be cited; every
  `[Pn]` and every quotation in the answer is verified against them and
  the result is shown with the answer. The prompt tells the model to say
  "Not found in the passages supplied" rather than fill a gap.
- **Search is keyword and vector, then a reranker.** A paragraph number or
  an exact phrase is found by BM25; a question in other words by the
  embedding; the reranker (a yes/no judge, P(yes) as the score) orders the
  fused candidates. On the sample the reranker separates sharply: 0.99 for
  the right table, under 0.2 for the clauses around it.

## Measured on the test codes

Clean run on 2026-10-01 with all three models in the Modal engine:

| Step | Time |
|---|---|
| Parsing both codes, including OCR of the six drawing sheets (local CPU) | 63 s |
| Reading the six sheets (9B on Modal) | 56 s |
| Extracting requirements, both codes | 43 s |
| Checking 139 drawing statements against their sheets | about 2 min (see below) |
| Embedding both codes | under 20 s |
| Comparison (415 candidate pairs judged, 7 reviewed) | 1 min 40 s |
| Total, upload to report, with the engine warm | about 6 min |

The same run with the local vision model took 11 min; the six sheets alone
were 5 min 48 s.

Result: 1 contradiction (BP asks for side-step ladders, PIP offers a
step-through ladder as a standard detail), 2 stricter-in-one-code findings
(PIP's double-bar OSHA gate against BP's "self-closing gates"), 2
equivalents, 96 requirements only in BP, 98 only in PIP, 51 drawing
statements set aside.

During that run Modal preempted the container in the middle of the drawing
check (its own infrastructure reclaiming the GPU). Modal restarted the same
call on a new container 14 minutes later and the run finished correctly;
only the in-flight batch was redone. The clock time above excludes that gap.

Cold starts: the first boot after a deploy loads and compiles all four
models in about 4-5 min with warm caches (12 min the very first time, with
the downloads). After scale-to-zero, the next call came back in 26 s
including its work, so the snapshot restores well. Re-running a
comparison on ingested documents takes about 2 min 20 s.

### Ask the code, on the ASME B31.1-2007 sample (28 scanned pages)

| Step | Time |
|---|---|
| Parse, first time: layout, OCR of 24 noisy pages, 9 tables, 43 equations (local CPU) | 14 min |
| Parse again after a parser change (model outputs are cached per page) | 6 s |
| Chunk: 224 chunks (146 clauses, 61 definitions, 9 tables, 8 figures), 0 elements lost | under 1 s |
| Describe 8 figures and 9 tables, embed 224 chunks (engine warm) | about 2 min |
| A question, engine warm: search 4-7 s (embed, two searches, rerank 40 candidates), answer 4-10 s | 8-16 s |
| The same with the engine asleep | 60-75 s |

Two questions, checked by hand against the book: the wall thickness
question returned 104.1.2(A) with eqs. (3), (3A), (4), (4A), the full
(A.1)-(A.7) nomenclature, the footnote on SF, the NOTE on rounding P, and
the references to 102.4.x and Table 104.1.2(A); the answer cited them and
the check passed. "y for ferritic steel at 1000F" returned Table 104.1.2(A)
first (0.99) and the answer read 0.7 from the right row and column.

## Known limits

- **Verdicts on borderline pairs vary between runs.** The side-step finding
  and the gate findings were stable across the last four runs; one low-
  severity pair flipped between "equivalent" and "stricter".
- **Bare dimensions on drawings are the weak point.** Both vision models
  misattribute a dimension line that has no wording. The checks above keep
  these out of findings, but they also mean some real drawing requirements
  never reach the comparison. They are listed, not lost.
- **The second reading is conservative.** Some statements it sets aside are
  correct (it disputed one over the word "inch"). The 9B reader on vLLM
  occasionally loops ("distance from the sixth rung to the seventh is 3",
  "... seventh to the eighth ..."); the shape filter and the second reading
  keep those out, at the cost of tokens.
- **Scanned two-column codes parse well enough to answer from, not
  perfectly.** On the 28-page B31.1 sample: headings, paragraph numbers,
  (A)/(A.1) sub-paragraphs, definitions, equations with their numbers,
  tables and figures come through; mirrored margins, column breaks and
  rotated pages are handled. What is still rough: a few sub-paragraph
  references on the figure-notes pages ("B.4.1.4.B.4.2.1"), the odd OCR
  misread in a term ("post'llxld heat treatment"), and the table recogniser
  dropping a column on a dense table (the image is shown beside the grid
  so a reader can check). The full 338-page book has not been run.
- **Large data tables.** An allowable-stress table (Appendix A) has
  hundreds of rows; the question-matched row selection covers the lookup,
  but the recogniser's accuracy on such tables is untested.
- **Thresholds were calibrated on one document pair** (`TOP_K`,
  `MIN_SIMILARITY` in `codesync/config.py`).
- **No authentication on the API**, and jobs are tracked in memory, so a
  server restart forgets queued jobs (document and comparison state is on
  disk and survives).

## Layout

```
codesync/
  config.py          paths, model choices, thresholds
  schema.py          Element, Requirement, Finding, ...
  prompts.py         every prompt, with version strings that key the caches
  pdfparse/          lines.py  numbering.py  vision.py  parser.py
  models/            chat.py (Modal clients)  embedder.py  llama_server.py (optional local reader)
  stages/            figures.py  extract.py  match.py  judge.py  report.py
                     chunk.py  index.py  retrieve.py  answer.py   (Ask the code)
  store.py           LanceDB chunk store (vector + keyword search)
  library.py         on-disk store
  pipeline.py        ingest() and compare()
  jobs.py  api.py  cli.py  setup_local.py
modal_app/codesync_modal.py
frontend/            Vite + React + TypeScript + Tailwind; `src/api.ts` mirrors schema.py
tests/               29 tests, no model needed (pytest)
data/library/<doc_id>/       source.pdf meta.json parsed.json figures/
                             ocr/ (cached OCR lines, layout, table and formula outputs per page)
                             figure_readings.json requirements.json embeddings.npz
                             chunks.json descriptions.json
data/comparisons/<cmp_id>/   meta.json pairs.json findings.json report.md report.xlsx
data/lancedb/                the chunk store, all documents in one table
tools/llama/                 llama.cpp, only for the optional local reader
```

Every stage is skipped when its output is already on disk for the same
inputs, so re-running a comparison on ingested documents takes about two
minutes. Changing the embedding file or a prompt version redoes exactly the
work that depends on it.
