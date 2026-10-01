# Bottlenecks and risks

Everything identified while building and testing the backend, in one place.
Status as of 2026-10-01, after the first Ask-the-Code build. "Open" means
nothing in the code handles it yet.

## 1. Sharing one GPU between four models

| | |
|---|---|
| Risk | The LLM engine (vLLM) claims its share of the card once at start and never exceeds it. The vision engine does the same. The embedder takes memory as it goes. If a dynamic allocation spikes (a very long table, a batch of large drawings) after the fixed shares are claimed, the process runs out of memory and the whole container dies mid-job. |
| Why it is manageable | The stages run one after another (drawings, extraction, checks, embedding, judging), so their working-memory peaks never coincide. Only the weights sit on the card together. |
| Mitigation in place | Fixed budgets per model (`modal_app/codesync_modal.py`): LLM 40%, vision 30%, embedder about 17 GB, reranker 1.5 GB, about 10 GB left free. Measured at boot: 84 GiB of 95 in use with nothing running. Embedding runs in batches bounded by a token budget (not a count), and the embedder and reranker hand their working memory back to the driver after every call. A dead vLLM engine ends the container so Modal starts a fresh one. |
| What was learned | 22% was not enough for the 9B: its weights plus the image-encoder's profiling activations filled the share and vLLM refused to start ("No available memory for the cache blocks"). The 27B's 40% leaves it 6.4 GiB of working space, about 200,000 tokens in flight, two to three times today's batches. Two crashes on 2026-10-01 came from the free 10 GB: (a) embedding 229 code clauses in count-based batches of 32 needed far more activation memory than 229 short requirement statements had; (b) after a big embed, PyTorch kept its reserved pool and the 9B's vision encoder found 95 MB free when six drawing sheets arrived ("OOM in embed_multimodal"), which killed that engine for the life of the container. Both are fixed as above. |
| Open | The budgets were checked on the test codes and a 28-page sample, not on a large document. A crash mid-job today loses the running stage (see 4). If concurrency is raised for big codes, the 27B share must grow with it. |

## 2. Modal cold starts and idle time are the cost, not memory

| | |
|---|---|
| Fact | Billing is per second of container life. Cold starts (35 s to 5 min observed, 12 min on the first boot after a deploy) and the 2-minute idle tail after each burst are billed like work. In the clean test run the LLM did about 3 minutes of work inside a much longer billed window. |
| Mitigation in place | GPU memory snapshot after warm-up; the pipeline starts the container booting before it is needed; all models on one container so nothing waits on a local step. |
| Measured 2026-10-01 | Three-model boot: about 5 min with warm caches, 12 min the first time. After scale-to-zero the next call returned in 26 s including its work, so the larger snapshot restores well. |
| Preemption | Modal can reclaim the GPU mid-job ("Container terminated due to preemption"). It happened once during the clean run: the same call was restarted on a new container 14 min later and the run finished correctly. Expect this on long jobs; it costs time, not results, as long as calls stay small and idempotent (they are). |
| Open | Every `modal deploy` invalidates the snapshot; the next one or two boots pay the full time. Snapshot restore is placement-sensitive (quick reboots land on the same worker; long gaps miss). A container that fails during startup is retried by Modal, and each failed attempt is billed (about 4 min each while the memory shares were being tuned). |

## 3. Big codes take hours, and the lever is parallelism

| | |
|---|---|
| Fact | A document is cut into small requests, so a 1,500-page code means tens of thousands of requests, not bigger ones. Measured LLM throughput is about 540 tokens per second at 48 concurrent requests. Extrapolated from the 16-page test: extraction a few hours per code, judging 5 to 12 hours for a pair, tens of dollars per comparison. Loose estimates. |
| Why memory does not fix it | The GPU's compute saturates long before the working space is full. More memory buys nothing past that point. |
| Open | `max_containers=1` and `LLM_BATCH=48` cap throughput. Raising both lets several containers work in parallel at roughly the same total cost. Not done. |

## 4. No progress saved inside a stage

| | |
|---|---|
| Fact | Results are written when a stage finishes (all figures of a document are saved per sheet, but extraction, judging and review are saved at the end). Fine for a 2-minute stage; a crash at hour 9 of judging loses hour 9 and the eight before it. |
| Open | Checkpoint per batch and resume from the last saved batch. |

## 5. Report volume at scale

| | |
|---|---|
| Fact | About 1% of candidate pairs ended as findings on the test pair (442 pairs, 4 to 5 findings). At big-code scale that is hundreds to thousands of findings. |
| Open | Grouping by topic, ranking by severity, and folding duplicates beyond the current "also stated at" merge. Unique lists include "info" statements (design basis, explanations) that a reviewer may not want. |

## 6. Matching thresholds were set on one small pair

| | |
|---|---|
| Fact | `TOP_K=5` and `MIN_SIMILARITY=0.50` were calibrated on BP 11.6 vs PIP STF05501 (98 vs 136 requirements). Finding the right counterpart among 20,000 requirements is a harder search; recall will fall and the floor may need to move. |
| Open | Re-calibrate on a large pair with a hand-checked sample. |

## 7. Drawings: the weak point of reading

| | |
|---|---|
| Fact | Both vision models attach a bare dimension line (one with no wording) to the nearest label. On one sheet the local model produced nine different "unsupported length" values; the 27B produced six. Callouts with words are read reliably. |
| Mitigation in place | Runs of three or more values for one property of one component on a sheet are flagged and kept out. Every drawing statement is re-read against the sheet by the 27B, and only statements both readings agree on enter the comparison (24 of 136 set aside on PIP). Set-aside statements are listed in the report for manual reading. |
| Open | The second reading is conservative: it set aside at least one correct statement over the word "inch". With the 9B on vLLM the set-aside rate rose to 51 of 155 (it produced more statements, including a loop of near-identical rung-spacing sentences on one sheet; a shape filter now catches such runs before the check). Real drawing requirements can therefore miss the comparison. A weld-symbol misread or a hole diameter read as a rung diameter can still pass if both readings agree. Two readings by the same 27B (if the 9B were dropped) are untested as a safeguard. |

## 8. Verdicts on borderline pairs vary between runs

| | |
|---|---|
| Fact | Batched inference is not bit-for-bit deterministic. The side-step and gate findings were stable over four runs; one low-severity pair flipped between "equivalent" and "stricter". |
| Mitigation in place | Judge must name component and property before calling a match; every match is reviewed separately; candidates that come back without a verdict are re-asked. |
| Open | Findings need a stable identity across re-runs (by requirement pair, not by position) so reviewer decisions survive a re-run. |

## 9. Parsing scanned, two-column codes

| | |
|---|---|
| Fact | A 28-page sample of ASME B31.1-2007 (definitions, design conditions, 104.1-104.8, Appendix A) was parsed and answered from. Pages with a noisy embedded text layer are re-OCR'd (threshold `REOCR_NOISE`, 0.3% odd characters; 24 of the 28 pages crossed it). Headings, paragraph numbers, (A)/(A.1) sub-paragraphs, definitions one per term, equations with their printed numbers and their "where" blocks, tables and figures come through. Mirrored margins (the text block shifts 22 pt between odd and even pages), two-column reading order, column breaks mid-paragraph, rotated landscape pages and footnotes are handled. |
| Still rough | A few sub-paragraph references on the figure-notes pages ("104.3.1 B.4.1.4.B.4.2.1"); OCR misreads inside words ("post'llxld", "comnponent", "BLanks"); line-end hyphens stay in the text ("tem-peratures": the models read through them and the citation check ignores them, but a dictionary-free parser cannot tell them from "side-step"); the "(E)" item of a list printed just above a heading can land under the heading. |
| Cost | First parse of the sample: 14 min on the local CPU (OCR about 25 s per page, layout 2 s, table recognition and formula recognition several seconds each). Every model output is now cached per page under `data/library/<id>/ocr/`, so a re-parse after a parser change takes 6 s. The full 338-page book would be about 2.5 hours the first time. A 1,500-page code: a working day, or a cloud CPU worker. |
| Open | Run the complete B31.1; a cloud CPU worker for parsing big codes; a word list to de-hyphenate safely. |

## 10. Large data tables

| | |
|---|---|
| Fact (comparison) | Each table goes to the LLM as one request (text plus image when the text came from OCR). An allowable-stress table with hundreds of values per page would overflow the request or produce hundreds of near-identical requirements. |
| Fact (Ask the code) | A table is stored as data: columns and rows from the table recogniser, its image beside it, a model description added separately. When a question names a material, grade, size or temperature, the matching rows are the ones shown to the reranker and the answer model, so an Appendix A lookup does not depend on the first twelve rows. Verified on Table 104.1.2(A): "y for ferritic steel at 1000F" answered 0.7 from the right row and column. |
| Risk | The table recogniser (SLANeXt, CPU) dropped one header cell on that same dense table, shifting a header row by a column; the values were still read correctly because the first columns lined up. On an Appendix A page (30+ columns of stresses) a shifted header would give a wrong value with a confident citation. The image is shown next to the grid so a reader can check, but the answer model does not see the image. |
| Open | Validate table grids (every row the same width as the header; numeric columns numeric); send the table image to the 27B along with the grid for lookups; test on a real Appendix A page. |

## 11. Multi-code projects and precedence

| | |
|---|---|
| Fact | A comparison is one pair with one governing document. A project with five applicable codes needs an order of precedence and several pairwise comparisons; neither is modelled. |
| Open | Project entity, precedence chain, and a merged "resolved requirement set" as the output the designer works from. |

## 12. API and jobs

| | |
|---|---|
| Fact | No authentication. Jobs are tracked in memory and run one at a time; a server restart forgets queued jobs (document and comparison state on disk survives). The local orchestrator must stay running for the whole job, which for a big code is hours. |
| Open | Auth, durable job queue, and moving orchestration off the desktop for long runs. |

## 13. The GPU image is fragile to dependency changes

| | |
|---|---|
| Fact | Adding packages to the GPU image can silently change pip's resolution of vLLM and torch (DTSPL saw vLLM 0.22 resolved instead of 0.27). The image is pinned and audited by `modal run modal_app/codesync_modal.py::imgcheck`. |
| Mitigation in place | The embedder runs on transformers, which is already in the image; no package was added for the three-model container. |
| Open | Any future package addition must go through `imgcheck` first. |

## 14. Data leaves the machine

| | |
|---|---|
| Fact | Clause text, tables and drawing crops of every uploaded code go to Modal for the LLM, the vision reader and the embedder. Nothing else leaves. A client who forbids cloud processing needs a different deployment with a proper GPU box; it is not a switch in this one. The optional local drawing reader (`FIGURE_READER=local`) is a developer fallback, not that deployment: judging still runs on Modal. |

## 15. Embedding index is locked to its model

| | |
|---|---|
| Fact | Vectors from different embedding files are not comparable. Changing `EMBED_MODEL` means re-embedding every document; the cache key does this automatically but the cost is a full re-run of that stage. |
| Note | KaLM-Embedding-Gemma3-12B is on hold; its licence ("tencent-kalm-embedding-community") has not been read. |

## 16. Shared Modal account

| | |
|---|---|
| Fact | `codesync-llm` lives in the same Modal profile as the DTSPL apps (`gsrahul1998`), so credits and billing are shared and both draw on the same GPU quota. |

## 17. Prompt examples leak into answers

| | |
|---|---|
| Fact | A worked example with a real dimension in the drawing prompt was copied into answers about sheets that did not show it. |
| Mitigation in place | Prompts carry no concrete values. Prompt versions key the caches so a prompt change redoes exactly the dependent work. |

## 18. Ask the code: what can still go wrong in an answer

| | |
|---|---|
| Retrieval misses | The right clause is not among the 6-8 passages (a question in words the code does not use, or the clause split across chunks). The answer then says "Not found in the passages supplied" or answers from a neighbour. The user sees the passages, so a miss is visible. A question set with known answers (WP2 of the build plan) is the only way to measure the rate; not built yet. |
| Wrong reading of a right passage | The model reads a value from the wrong row, or paraphrases a condition away. The citation check catches invented quotations and citations, not misreadings. Mitigation: equations are quoted as LaTeX, table rows are matched to the question, and the passage is one click away. |
| Cross-references outside the library | "see para. 102.3.2" resolves only when that paragraph is indexed. On an extract most references are unresolved (105 of 187 on the sample); on the full book most would resolve. Unresolved references are shown as text, not links. |
| Edition | Every chunk carries code and edition and the answer names them; but nothing stops a user indexing two editions and searching both at once. The scope chips make it explicit; a default of "latest edition only" is not implemented. |
| Latency | Warm: 8-16 s per question (4-7 s search, the rest generation). Asleep: 60-75 s for the first question (snapshot restore plus the first search); the Ask page starts the boot on load. Keeping the engine always on is about $73 a day (see 2); the business decision is open. |
| Cost per question | About 2-5 s of GPU time: a few cents. The idle tail after a question (2 min scaledown) costs more than the question. |
