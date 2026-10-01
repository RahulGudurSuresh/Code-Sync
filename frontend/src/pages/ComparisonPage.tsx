import { useCallback, useEffect, useMemo, useState } from "react";
import { Link, useParams } from "react-router-dom";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { ArrowLeft, Check, Download, FileSpreadsheet, FileText, Flag, Image as ImageIcon, X } from "lucide-react";
import { api, isBusy, type Comparison, type Document, type Finding, type Job, type Requirement, type Review } from "../api";
import { usePolling, useTab } from "../hooks";
import { Badge, Button, Card, Empty, ErrorNote, fmtSeconds, StatusPill, Tabs, Tile, type Tone } from "../components/ui";
import { ModalityBadge } from "./DocumentPage";

type Tab = "conflicts" | "similar" | "unique_a" | "unique_b" | "aside" | "report";
type FigureIndex = Record<string, { doc: string; image: string; ref: string }>;

export default function ComparisonPage() {
  const { id = "" } = useParams();
  const { data: cmp, error } = usePolling(() => api.comparison(id), (c) => isBusy(c.status), 3000, [id]);
  const [docs, setDocs] = useState<Record<string, Document>>({});
  const [findings, setFindings] = useState<Finding[] | null>(null);
  const [reviews, setReviews] = useState<Record<string, Review>>({});
  const [figures, setFigures] = useState<FigureIndex>({});
  const [reqs, setReqs] = useState<{ a: Requirement[]; b: Requirement[] } | null>(null);
  const [job, setJob] = useState<Job | null>(null);
  const [tab, setTab] = useTab<Tab>("conflicts", ["conflicts", "similar", "unique_a", "unique_b", "aside", "report"]);
  const [loadError, setLoadError] = useState<string | null>(null);

  // documents named in the comparison
  useEffect(() => {
    if (!cmp) return;
    Promise.all([api.document(cmp.doc_a), api.document(cmp.doc_b)])
      .then(([a, b]) => setDocs({ [a.doc_id]: a, [b.doc_id]: b }))
      .catch((e) => setLoadError(String(e)));
  }, [cmp?.doc_a, cmp?.doc_b]);

  // live job message while running
  useEffect(() => {
    if (!cmp || !isBusy(cmp.status)) return;
    const t = window.setInterval(() => api.jobs().then((js) => setJob(js.find((j) => j.target === id) ?? null)).catch(() => undefined), 2000);
    return () => window.clearInterval(t);
  }, [cmp?.status, id]);

  // results once done
  useEffect(() => {
    if (!cmp || cmp.status !== "done") return;
    api.findings(id).then(setFindings).catch((e) => setLoadError(String(e)));
    api.reviews(id).then(setReviews).catch(() => undefined);
    Promise.all([api.requirements(cmp.doc_a), api.requirements(cmp.doc_b)]).then(([a, b]) => setReqs({ a, b })).catch(() => undefined);
    Promise.all([cmp.doc_a, cmp.doc_b].map((d) => api.elements(d, "figure").then((r) => r.elements.map((e) => [e.id, { doc: d, image: e.image ?? "", ref: e.ref }] as const))))
      .then((lists) => setFigures(Object.fromEntries(lists.flat())))
      .catch(() => undefined);
  }, [cmp?.status, id]);

  const saveReview = useCallback(async (fid: string, review: Review) => {
    const saved = await api.saveReview(id, fid, review);
    setReviews((r) => ({ ...r, [fid]: saved }));
  }, [id]);

  const groups = useMemo(() => ({
    conflicts: findings?.filter((f) => f.category === "conflict") ?? [],
    similar: findings?.filter((f) => f.category === "similar") ?? [],
    unique_a: findings?.filter((f) => f.type === "only_in_a") ?? [],
    unique_b: findings?.filter((f) => f.type === "only_in_b") ?? [],
  }), [findings]);

  if (!cmp) return <ErrorNote error={error} />;
  const A = docs[cmp.doc_a], B = docs[cmp.doc_b];
  const labA = A?.label ?? "A", labB = B?.label ?? "B";
  const s = cmp.summary ?? {};
  const aside = [...(reqs?.a ?? []), ...(reqs?.b ?? [])].filter((r) => r.excluded);

  return (
    <div className="space-y-5">
      <Link to="/comparisons" className="inline-flex items-center gap-1 text-sm text-slate-500 hover:text-slate-800"><ArrowLeft className="h-4 w-4" /> Comparisons</Link>
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <h1 className="flex items-center gap-2 text-xl font-semibold text-slate-900">
            <Badge tone="blue">{labA}</Badge><span className="text-slate-400">vs</span><Badge tone="slate">{labB}</Badge>
            <span className="ml-2 text-base font-normal text-slate-500">governing: {docs[cmp.governing]?.label ?? cmp.governing}</span>
          </h1>
          <div className="mt-1 text-sm text-slate-500">{A?.name} · {B?.name}</div>
          <div className="mt-2 flex items-center gap-3"><StatusPill status={cmp.status} stage={cmp.stage} />{job && isBusy(cmp.status) && <span className="text-xs text-slate-500">{job.message}</span>}</div>
          {cmp.error && <div className="mt-2"><ErrorNote error={cmp.error} /></div>}
        </div>
        {cmp.status === "done" && (
          <div className="flex gap-2">
            <a href={api.reportMdUrl(id)} target="_blank" rel="noreferrer"><Button kind="secondary"><FileText className="h-4 w-4" /> report.md</Button></a>
            <a href={api.reportXlsxUrl(id)}><Button kind="secondary"><FileSpreadsheet className="h-4 w-4" /> report.xlsx</Button></a>
          </div>
        )}
      </div>

      {cmp.status === "done" && (
        <div className="grid grid-cols-2 gap-3 md:grid-cols-4 lg:grid-cols-7">
          <Tile label="Contradictions" value={s.contradictions ?? 0} tone="red" />
          <Tile label="One code stricter" value={s.more_stringent ?? 0} tone="amber" />
          <Tile label="Similar" value={s.similar ?? 0} tone="emerald" />
          <Tile label={`Only in ${labA}`} value={s.unique_a ?? 0} tone="sky" />
          <Tile label={`Only in ${labB}`} value={s.unique_b ?? 0} tone="sky" />
          <Tile label="Needs review" value={s.needs_review ?? 0} tone="amber" />
          <Tile label="Set aside" value={(s.set_aside_a ?? 0) + (s.set_aside_b ?? 0)} tone="violet" hint="Drawing statements the two readings disagreed on" />
        </div>
      )}
      <ErrorNote error={loadError} />

      {isBusy(cmp.status) && <Progress job={job} />}

      {cmp.status === "done" && findings && (
        <>
          <Tabs<Tab> value={tab} onChange={setTab} tabs={[
            { id: "conflicts", label: "Conflicts", count: groups.conflicts.length },
            { id: "similar", label: "Similar", count: groups.similar.length },
            { id: "unique_a", label: `Only in ${labA}`, count: groups.unique_a.length },
            { id: "unique_b", label: `Only in ${labB}`, count: groups.unique_b.length },
            { id: "aside", label: "Set aside", count: aside.length },
            { id: "report", label: "Report" },
          ]} />
          {tab === "conflicts" && <FindingList findings={groups.conflicts} labA={labA} labB={labB} figures={figures} reviews={reviews} onReview={saveReview} filters />}
          {tab === "similar" && <FindingList findings={groups.similar} labA={labA} labB={labB} figures={figures} reviews={reviews} onReview={saveReview} />}
          {tab === "unique_a" && <UniqueList findings={groups.unique_a} side="a" figures={figures} />}
          {tab === "unique_b" && <UniqueList findings={groups.unique_b} side="b" figures={figures} />}
          {tab === "aside" && <AsideTable rows={aside} docs={docs} figures={figures} />}
          {tab === "report" && <Report id={id} timings={cmp.timings} />}
        </>
      )}
    </div>
  );
}

function Progress({ job }: { job: Job | null }) {
  const steps = ["parsing", "reading figures", "extracting", "checking", "embedding", "matching", "judging", "reviewing", "writing report"];
  const log = job?.log ?? [];
  return (
    <Card className="p-4">
      <div className="text-sm font-medium text-slate-800">Running</div>
      <p className="mt-1 text-xs text-slate-500">Documents are parsed locally; drawings, requirements, embeddings and verdicts come from the Modal engine. A cold engine adds a minute or two at the start.</p>
      <ol className="mt-3 flex flex-wrap gap-2 text-xs">
        {steps.map((st) => {
          const seen = log.some((l) => l.includes(st));
          const current = (job?.message ?? "").includes(st);
          return <li key={st} className={`rounded-full px-2.5 py-1 ${current ? "bg-blue-600 text-white" : seen ? "bg-emerald-100 text-emerald-800" : "bg-slate-100 text-slate-500"}`}>{st}</li>;
        })}
      </ol>
      {log.length > 0 && <pre className="mt-3 max-h-48 overflow-auto rounded-lg bg-slate-900 p-3 text-xs text-slate-100">{log.slice(-12).join("\n")}</pre>}
    </Card>
  );
}

const SEVERITY_TONE: Record<string, Tone> = { high: "red", medium: "amber", low: "slate", "": "slate" };

function FindingList({ findings, labA, labB, figures, reviews, onReview, filters = false }: {
  findings: Finding[]; labA: string; labB: string; figures: FigureIndex; reviews: Record<string, Review>;
  onReview: (fid: string, r: Review) => Promise<void>; filters?: boolean;
}) {
  const [sev, setSev] = useState<"all" | "high" | "medium" | "low">("all");
  const [onlyReview, setOnlyReview] = useState(false);
  const [decision, setDecision] = useState<"all" | "open" | "accepted" | "rejected">("all");
  const rows = findings.filter((f) => (sev === "all" || f.severity === sev) && (!onlyReview || f.needs_review) && (
    decision === "all" || (decision === "open" ? !(reviews[f.id]?.decision) : reviews[f.id]?.decision === decision)));
  if (findings.length === 0) return <Empty>Nothing in this category.</Empty>;
  return (
    <div className="space-y-3">
      {filters && (
        <div className="flex flex-wrap items-center gap-2 text-xs">
          {(["all", "high", "medium", "low"] as const).map((v) => <button key={v} onClick={() => setSev(v)} className={`rounded-lg px-2.5 py-1 font-medium ${sev === v ? "bg-slate-900 text-white" : "bg-slate-100 text-slate-600"}`}>{v === "all" ? "All severities" : v}</button>)}
          <span className="mx-1 text-slate-300">|</span>
          {(["all", "open", "accepted", "rejected"] as const).map((v) => <button key={v} onClick={() => setDecision(v)} className={`rounded-lg px-2.5 py-1 font-medium ${decision === v ? "bg-slate-900 text-white" : "bg-slate-100 text-slate-600"}`}>{v === "all" ? "All decisions" : v}</button>)}
          <label className="ml-2 flex items-center gap-1 text-slate-600"><input type="checkbox" checked={onlyReview} onChange={(e) => setOnlyReview(e.target.checked)} /> needs review only</label>
          <span className="ml-auto text-slate-500">{rows.length} of {findings.length}</span>
        </div>
      )}
      {rows.map((f) => <FindingCard key={f.id} f={f} labA={labA} labB={labB} figures={figures} review={reviews[f.id]} onReview={onReview} />)}
    </div>
  );
}

function FindingCard({ f, labA, labB, figures, review, onReview }: {
  f: Finding; labA: string; labB: string; figures: FigureIndex; review?: Review; onReview: (fid: string, r: Review) => Promise<void>;
}) {
  const [note, setNote] = useState(review?.note ?? "");
  const [saving, setSaving] = useState(false);
  useEffect(() => setNote(review?.note ?? ""), [review?.note]);
  const decide = async (decision: Review["decision"]) => {
    setSaving(true);
    try { await onReview(f.id, { decision: review?.decision === decision ? "" : decision, note, by: "" }); } finally { setSaving(false); }
  };
  const saveNote = async () => {
    setSaving(true);
    try { await onReview(f.id, { decision: review?.decision ?? "", note, by: "" }); } finally { setSaving(false); }
  };
  const typeLabel = { contradiction: "Contradiction", more_stringent: `Stricter in ${f.stricter}`, equivalent: "Equivalent", only_in_a: "", only_in_b: "" }[f.type];
  const typeTone: Tone = f.type === "contradiction" ? "red" : f.type === "more_stringent" ? "amber" : "emerald";
  const border = review?.decision === "accepted" ? "border-emerald-300" : review?.decision === "rejected" ? "border-slate-300 opacity-70" : f.needs_review ? "border-amber-300" : "border-slate-200";

  return (
    <Card className={`border ${border}`}>
      <div className="flex flex-wrap items-center gap-2 border-b border-slate-100 px-4 py-2">
        <span className="font-mono text-xs text-slate-500">{f.id}</span>
        <Badge tone={typeTone}>{typeLabel}</Badge>
        {f.severity && <Badge tone={SEVERITY_TONE[f.severity]}>{f.severity}</Badge>}
        {f.topic && <span className="text-sm font-medium text-slate-800">{f.topic}</span>}
        {f.needs_review && <Badge tone="amber" title={f.review_reason}><Flag className="mr-1 h-3 w-3" /> needs review</Badge>}
        {f.drawing_check === "confirmed" && <Badge tone="violet" title="Drawing statement confirmed by a second reading"><ImageIcon className="mr-1 h-3 w-3" /> drawing checked</Badge>}
        {review?.decision && <Badge tone={review.decision === "accepted" ? "emerald" : "slate"}>{review.decision}</Badge>}
        <span className="ml-auto text-xs text-slate-400">similarity {f.similarity?.toFixed(2)}</span>
      </div>
      <div className="grid gap-4 px-4 py-3 md:grid-cols-2">
        <Side label={labA} r={f.a} figures={figures} />
        <Side label={labB} r={f.b} figures={figures} also={f.also_b} />
      </div>
      {(f.rationale || f.action) && (
        <div className="space-y-2 border-t border-slate-100 px-4 py-3 text-sm">
          {f.rationale && <p className="text-slate-600"><span className="font-medium text-slate-800">Why: </span>{f.rationale}</p>}
          {f.action && <p className="rounded-lg bg-blue-50 px-3 py-2 text-blue-900"><span className="font-medium">Action ({f.governing} governs): </span>{f.action}</p>}
          {f.needs_review && <p className="text-xs text-amber-800">{f.review_reason}</p>}
        </div>
      )}
      <div className="flex flex-wrap items-center gap-2 border-t border-slate-100 bg-slate-50/60 px-4 py-2">
        <Button kind={review?.decision === "accepted" ? "primary" : "secondary"} onClick={() => decide("accepted")} disabled={saving}><Check className="h-4 w-4" /> Accept</Button>
        <Button kind={review?.decision === "rejected" ? "danger" : "secondary"} onClick={() => decide("rejected")} disabled={saving}><X className="h-4 w-4" /> Reject</Button>
        <input value={note} onChange={(e) => setNote(e.target.value)} onBlur={() => note !== (review?.note ?? "") && saveNote()} placeholder="Resolution note / RFI reference" className="min-w-64 flex-1 rounded-lg border border-slate-300 bg-white px-2 py-1.5 text-sm" />
        {review?.at && <span className="text-xs text-slate-400">saved {new Date(review.at).toLocaleString()}</span>}
      </div>
    </Card>
  );
}

function Side({ label, r, figures, also }: { label: string; r: Requirement | null; figures: FigureIndex; also?: string[] }) {
  if (!r) return null;
  const fig = r.from_figure ? figures[r.element_id] : undefined;
  return (
    <div className="space-y-1.5">
      <div className="flex flex-wrap items-center gap-2 text-xs">
        <Badge tone="blue">{label}</Badge>
        <span className="font-mono text-slate-600">{r.ref}</span>
        <span className="text-slate-400">p.{r.page}</span>
        <ModalityBadge r={r} />
      </div>
      <p className="text-sm text-slate-900">{r.statement}</p>
      {r.quote && !r.from_figure && <p className="text-xs text-slate-500">“{r.quote}”</p>}
      {also && also.length > 0 && <p className="text-xs text-slate-500">Also stated at: {also.join("; ")}</p>}
      {fig?.image && (
        <a href={api.figureUrl(fig.doc, fig.image)} target="_blank" rel="noreferrer" className="block">
          <img src={api.figureUrl(fig.doc, fig.image)} alt={fig.ref} className="mt-1 max-h-40 rounded-lg border border-slate-200 bg-slate-50 object-contain" loading="lazy" />
        </a>
      )}
    </div>
  );
}

function UniqueList({ findings, side, figures }: { findings: Finding[]; side: "a" | "b"; figures: FigureIndex }) {
  const [q, setQ] = useState("");
  const rows = findings.filter((f) => `${f[side]?.ref} ${f[side]?.statement} ${f[side]?.subject}`.toLowerCase().includes(q.toLowerCase()));
  if (findings.length === 0) return <Empty>Nothing in this category.</Empty>;
  return (
    <div className="space-y-3">
      <div className="flex items-center gap-2"><input value={q} onChange={(e) => setQ(e.target.value)} placeholder="Search" className="w-72 rounded-lg border border-slate-300 px-2 py-1.5 text-sm" /><span className="text-xs text-slate-500">{rows.length} of {findings.length}</span></div>
      <Card>
        <table className="w-full text-sm">
          <thead className="bg-slate-50 text-left text-xs uppercase tracking-wide text-slate-500"><tr><th className="px-3 py-2">ID</th><th className="px-3 py-2">Clause</th><th className="px-3 py-2">Basis</th><th className="px-3 py-2">Requirement</th><th className="px-3 py-2">Subject</th></tr></thead>
          <tbody className="divide-y divide-slate-100">
            {rows.map((f) => {
              const r = f[side]!;
              const fig = r.from_figure ? figures[r.element_id] : undefined;
              return (
                <tr key={f.id}>
                  <td className="px-3 py-2 font-mono text-xs text-slate-500">{f.id}</td>
                  <td className="px-3 py-2 font-mono text-xs">{r.ref}<div className="text-slate-400">p.{r.page}</div>{fig?.image && <a className="text-blue-700 hover:underline" href={api.figureUrl(fig.doc, fig.image)} target="_blank" rel="noreferrer">sheet</a>}</td>
                  <td className="px-3 py-2"><ModalityBadge r={r} /></td>
                  <td className="px-3 py-2">{r.statement}</td>
                  <td className="px-3 py-2 text-xs text-slate-600">{r.subject}<div className="text-slate-400">{r.parameter}</div></td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </Card>
    </div>
  );
}

function AsideTable({ rows, docs, figures }: { rows: Requirement[]; docs: Record<string, Document>; figures: FigureIndex }) {
  if (rows.length === 0) return <Empty>No drawing statements were set aside.</Empty>;
  return (
    <div className="space-y-3">
      <p className="text-sm text-slate-600">Read from a drawing by the vision model and disputed by the second reading. They took no part in the comparison; read them from the sheet.</p>
      <Card>
        <table className="w-full text-sm">
          <thead className="bg-slate-50 text-left text-xs uppercase tracking-wide text-slate-500"><tr><th className="px-3 py-2">Document</th><th className="px-3 py-2">Location</th><th className="px-3 py-2">First reading</th><th className="px-3 py-2">Second reading</th></tr></thead>
          <tbody className="divide-y divide-slate-100">
            {rows.map((r) => {
              const fig = figures[r.element_id];
              return (
                <tr key={r.id}>
                  <td className="px-3 py-2"><Badge tone="blue">{docs[r.doc_id]?.label ?? r.doc_id}</Badge></td>
                  <td className="px-3 py-2 font-mono text-xs">{r.ref}<div className="text-slate-400">p.{r.page}</div>{fig?.image && <a className="text-blue-700 hover:underline" href={api.figureUrl(fig.doc, fig.image)} target="_blank" rel="noreferrer">sheet</a>}</td>
                  <td className="px-3 py-2">{r.statement}</td>
                  <td className="px-3 py-2 text-xs text-violet-800"><Badge tone="violet">{r.drawing_check}</Badge> {r.check_note}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </Card>
    </div>
  );
}

function Report({ id, timings }: { id: string; timings: Record<string, number> }) {
  const [md, setMd] = useState<string | null>(null);
  const [err, setErr] = useState<string | null>(null);
  useEffect(() => { api.reportMd(id).then(setMd).catch((e) => setErr(String(e))); }, [id]);
  return (
    <div className="space-y-3">
      <div className="flex flex-wrap items-center gap-3 text-xs text-slate-500">
        <span>judging {fmtSeconds(timings.judge_s)}</span><span>review {fmtSeconds(timings.review_s)}</span><span>total {fmtSeconds(timings.total_s)}</span>
        <a className="ml-auto inline-flex items-center gap-1 text-blue-700 hover:underline" href={api.reportMdUrl(id)} target="_blank" rel="noreferrer"><Download className="h-3 w-3" /> report.md</a>
        <a className="inline-flex items-center gap-1 text-blue-700 hover:underline" href={api.reportXlsxUrl(id)}><Download className="h-3 w-3" /> report.xlsx</a>
      </div>
      <ErrorNote error={err} />
      {md && <Card className="report p-6"><ReactMarkdown remarkPlugins={[remarkGfm]}>{md}</ReactMarkdown></Card>}
    </div>
  );
}

export type { Comparison };
