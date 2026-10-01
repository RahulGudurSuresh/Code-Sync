import { useEffect, useMemo, useRef, useState } from "react";
import { useSearchParams } from "react-router-dom";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";
import rehypeKatex from "rehype-katex";
import "katex/dist/katex.min.css";
import { BookOpen, CheckCircle2, ExternalLink, Loader2, Search, Send, ShieldAlert } from "lucide-react";
import { api, type AskScope, type Chunk, type Passage } from "../api";
import { Badge, Button, Card, Empty, ErrorNote, type Tone } from "../components/ui";

type Stage = "idle" | "searching" | "answering" | "done" | "error";

interface Check { ok: boolean; cited: string[]; bad_tags: string[]; bad_quotes: string[]; not_found: boolean; quotes?: number }

export default function Ask() {
  const [scope, setScope] = useState<AskScope[]>([]);
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const [question, setQuestion] = useState("");
  const [stage, setStage] = useState<Stage>("idle");
  const [passages, setPassages] = useState<Passage[]>([]);
  const [answer, setAnswer] = useState("");
  const [check, setCheck] = useState<Check | null>(null);
  const [timings, setTimings] = useState<Record<string, number>>({});
  const [error, setError] = useState<string | null>(null);
  const [active, setActive] = useState<string | null>(null);
  const started = useRef(0);
  // the question lives in the URL (?q=...&doc=...) so an answer can be
  // linked to and reopened; a link with a question runs it on load
  const [params, setParams] = useSearchParams();
  const autoRun = useRef(params.get("q") ?? "");

  useEffect(() => {
    api.askScope().then((s) => {
      setScope(s);
      const docs = params.getAll("doc").filter((d) => s.some((x) => x.doc_id === d));
      setSelected(new Set(docs.length ? docs : s.map((d) => d.doc_id)));
      if (autoRun.current) { setQuestion(autoRun.current); void run(autoRun.current, docs.length ? docs : s.map((d) => d.doc_id)); }
    }).catch((e) => setError(String(e)));
    api.askWarm().catch(() => undefined);     // start the GPU booting now, not at the first question
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const ask = () => run(question, [...selected]);

  const run = async (text: string, docs: string[]) => {
    const q = text.trim();
    if (!q || stage === "searching" || stage === "answering") return;
    setParams({ q, doc: docs }, { replace: true });
    setStage("searching"); setPassages([]); setAnswer(""); setCheck(null); setError(null); setTimings({}); setActive(null);
    started.current = performance.now();
    try {
      await api.ask(q, docs, (kind, data) => {
        if (kind === "status") setStage(data.stage === "answering" ? "answering" : "searching");
        else if (kind === "sources") { setPassages(data.passages); setTimings(data.timings ?? {}); }
        else if (kind === "delta") setAnswer((a) => a + data.text);
        else if (kind === "done") { setCheck(data.check); setStage("done"); setTimings((t) => ({ ...t, total_s: Math.round(performance.now() - started.current) / 1000 })); }
        else if (kind === "error") { setError(data.message); setStage("error"); }
      });
      setStage((s) => (s === "answering" || s === "searching" ? "done" : s));
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
      setStage("error");
    }
  };

  // [P3] in the answer becomes a link to the source card; a $$...$$ the
  // model wrote mid-sentence becomes a display block (it is an equation
  // quoted from the code, so it deserves the room)
  const linked = useMemo(() => answer
    .replace(/\[(P\d+)\]/g, (_m, t) => `[${t}](#src-${t})`)
    .replace(/\$\$([^$]+)\$\$/g, (_m, tex) => `\n\n$$\n${tex.trim()}\n$$\n\n`), [answer]);
  const byTag = Object.fromEntries(passages.map((p) => [p.tag, p]));

  return (
    <div className="space-y-5">
      <div>
        <h1 className="text-xl font-semibold text-slate-900">Ask the code</h1>
        <p className="mt-1 text-sm text-slate-500">Answers come only from the passages found in the indexed codes, with the paragraph and page cited. Equations are quoted, never calculated.</p>
      </div>

      <Card className="p-4">
        <div className="flex flex-wrap items-center gap-2 text-xs">
          <span className="text-slate-500">Search in:</span>
          {scope.length === 0 && <span className="text-slate-400">no document is indexed yet (Library → Index for Ask)</span>}
          {scope.map((d) => {
            const on = selected.has(d.doc_id);
            return (
              <button key={d.doc_id} onClick={() => setSelected((s) => { const n = new Set(s); if (on) n.delete(d.doc_id); else n.add(d.doc_id); return n; })}
                className={`rounded-lg px-2.5 py-1 font-medium ring-1 ${on ? "bg-slate-900 text-white ring-slate-900" : "bg-white text-slate-600 ring-slate-300"}`}
                title={`${d.name} · ${d.chunks} chunks`}>
                {d.code || d.label}{d.edition ? ` ${d.edition}` : ""}
              </button>
            );
          })}
        </div>
        <div className="mt-3 flex gap-2">
          <input value={question} onChange={(e) => setQuestion(e.target.value)} onKeyDown={(e) => e.key === "Enter" && ask()}
            placeholder="How do I calculate the minimum wall thickness for straight pipe under internal pressure?"
            className="flex-1 rounded-lg border border-slate-300 px-3 py-2 text-sm" />
          <Button onClick={ask} disabled={!question.trim() || stage === "searching" || stage === "answering" || selected.size === 0}>
            {stage === "searching" || stage === "answering" ? <Loader2 className="h-4 w-4 animate-spin" /> : <Send className="h-4 w-4" />} Ask
          </Button>
        </div>
        <div className="mt-2 flex flex-wrap items-center gap-3 text-xs text-slate-500">
          <StageChip stage={stage} />
          {timings.embed_s !== undefined && <span>search {((timings.embed_s ?? 0) + (timings.search_s ?? 0) + (timings.rerank_s ?? 0)).toFixed(1)} s</span>}
          {timings.total_s !== undefined && <span>total {timings.total_s.toFixed(1)} s</span>}
          {stage === "searching" && <span>A cold engine can take a minute or more on the first question.</span>}
        </div>
      </Card>
      <ErrorNote error={error} />

      {(answer || passages.length > 0) && (
        <div className="grid gap-4 lg:grid-cols-[3fr_2fr]">
          <Card className="p-5">
            <div className="mb-3 flex items-center gap-2 text-sm font-medium text-slate-800"><BookOpen className="h-4 w-4" /> Answer</div>
            {check && <CheckBanner check={check} />}
            <div className="report prose-sm max-w-none text-slate-900">
              <ReactMarkdown remarkPlugins={[remarkGfm, remarkMath]} rehypePlugins={[rehypeKatex]} components={{
                a: ({ href, children }) => {
                  const tag = (href ?? "").replace("#src-", "");
                  if (byTag[tag]) {
                    return <button onClick={() => { setActive(tag); document.getElementById(`src-${tag}`)?.scrollIntoView({ behavior: "smooth", block: "center" }); }}
                      className="mx-0.5 rounded bg-blue-50 px-1 font-mono text-xs text-blue-700 ring-1 ring-blue-200 hover:bg-blue-100">{tag}</button>;
                  }
                  return <a href={href} target="_blank" rel="noreferrer">{children}</a>;
                },
              }}>{linked}</ReactMarkdown>
              {stage === "answering" && <span className="ml-1 inline-block h-4 w-2 animate-pulse bg-slate-400 align-middle" />}
            </div>
          </Card>
          <div className="space-y-3">
            <div className="flex items-center gap-2 text-sm font-medium text-slate-800"><Search className="h-4 w-4" /> Passages ({passages.length})</div>
            {passages.length === 0 && stage === "searching" && <Empty>Searching…</Empty>}
            {passages.map((p) => <SourceCard key={p.tag} p={p} active={active === p.tag} onPick={() => setActive(p.tag)} />)}
          </div>
        </div>
      )}
    </div>
  );
}

function StageChip({ stage }: { stage: Stage }) {
  const map: Record<Stage, [string, Tone]> = { idle: ["ready", "slate"], searching: ["searching", "amber"], answering: ["answering", "amber"], done: ["done", "emerald"], error: ["error", "red"] };
  const [label, tone] = map[stage];
  return <Badge tone={tone}>{label}</Badge>;
}

function CheckBanner({ check }: { check: Check }) {
  if (check.ok) {
    return <div className="mb-3 flex items-center gap-2 rounded-lg bg-emerald-50 px-3 py-2 text-xs text-emerald-800"><CheckCircle2 className="h-4 w-4" />
      {check.not_found && check.cited.length === 0 ? "The model reports the passages do not answer this." : `Citations verified: ${check.cited.length} passage${check.cited.length === 1 ? "" : "s"} cited${check.quotes ? `, ${check.quotes} quote${check.quotes === 1 ? "" : "s"} matched word for word` : ""}.`}
    </div>;
  }
  return <div className="mb-3 rounded-lg bg-amber-50 px-3 py-2 text-xs text-amber-900">
    <div className="flex items-center gap-2 font-medium"><ShieldAlert className="h-4 w-4" /> Check the answer against the passages.</div>
    {check.bad_tags.length > 0 && <div className="mt-1">Cites passages that were not supplied: {check.bad_tags.join(", ")}.</div>}
    {check.bad_quotes.length > 0 && <div className="mt-1">Quotes not found word for word in the passages: {check.bad_quotes.map((q) => `“${q.slice(0, 80)}”`).join("; ")}.</div>}
    {check.bad_tags.length === 0 && check.bad_quotes.length === 0 && <div className="mt-1">No passage was cited.</div>}
  </div>;
}

const WHY_TONE: Record<string, Tone> = { search: "blue", referenced: "sky", definition: "violet" };

function SourceCard({ p, active, onPick }: { p: Passage; active: boolean; onPick: () => void }) {
  const c: Chunk = p.chunk;
  const [open, setOpen] = useState(false);
  const short = preview(c.text);
  const long = short.length < c.text.length;
  const page = c.page_label ? `page ${c.page_label}` : `PDF p.${c.page}`;
  return (
    <Card className={`p-3 ${active ? "ring-2 ring-blue-400" : ""}`}>
      <div id={`src-${p.tag}`} className="flex flex-wrap items-center gap-2 text-xs" onClick={onPick}>
        <span className="rounded bg-slate-900 px-1.5 py-0.5 font-mono text-white">{p.tag}</span>
        <Badge tone={WHY_TONE[p.why] ?? "slate"}>{p.why}</Badge>
        <span className="font-medium text-slate-800">{c.code} {c.edition}</span>
        <span className="font-mono text-slate-700">{c.ref}</span>
        <a href={`${api.pdfUrl(c.doc_id)}#page=${c.page}`} target="_blank" rel="noreferrer" className="inline-flex items-center gap-1 text-blue-700 hover:underline">{page} <ExternalLink className="h-3 w-3" /></a>
        {p.score !== null && p.score !== undefined && <span className="ml-auto text-slate-400" title="reranker relevance">{p.score.toFixed(2)}</span>}
      </div>
      {c.heading_path.length > 0 && <div className="mt-1 text-xs text-slate-500">{c.heading_path.join(" › ")}</div>}
      {c.context && <div className="mt-1 text-xs italic text-slate-500">{c.context}</div>}
      <div className="mt-1 whitespace-pre-line text-sm text-slate-900">
        <ReactMarkdown remarkPlugins={[remarkMath]} rehypePlugins={[rehypeKatex]}>{formatChunkText(open || !long ? c.text : short + "…")}</ReactMarkdown>
      </div>
      {long && <button onClick={() => setOpen((o) => !o)} className="mt-1 text-xs text-blue-700 hover:underline">{open ? "less" : "show all"}</button>}
      {c.notes.length > 0 && <div className="mt-2 rounded bg-slate-50 p-2 text-xs text-slate-700">{c.notes.map((n, i) => <p key={i}>{n}</p>)}</div>}
      {c.table && c.table.rows && c.table.rows.length > 0 && (
        <div className="mt-2 max-h-64 overflow-auto rounded border border-slate-200">
          <table className="w-full text-xs">
            <thead className="bg-slate-50"><tr>{c.table.columns.map((h, i) => <th key={i} className="px-2 py-1 text-left font-medium text-slate-700">{h}</th>)}</tr></thead>
            <tbody>{c.table.rows.map((r, i) => <tr key={i} className="border-t border-slate-100">{r.map((v, j) => <td key={j} className="px-2 py-1">{v}</td>)}</tr>)}</tbody>
          </table>
        </div>
      )}
      {c.image && <a href={api.figureUrl(c.doc_id, c.image)} target="_blank" rel="noreferrer"><img src={api.figureUrl(c.doc_id, c.image)} alt={c.ref} className="mt-2 max-h-56 rounded border border-slate-200 bg-slate-50 object-contain" loading="lazy" /></a>}
      {c.description && <div className="mt-2 text-xs text-slate-600"><span className="font-medium">Description (added by the model): </span>{c.description}</div>}
      {c.refs.length > 0 && <div className="mt-2 text-xs text-slate-500">Refers to: {c.refs.map((r) => r.text).slice(0, 6).join("; ")}</div>}
    </Card>
  );
}

const EQUATION = /\[EQUATION(?:\s+\(?([\w]+)\)?)?:\s*\$\$(.*?)\$\$\]/gs;

/** The parser stores display equations inline as [EQUATION (3): $$...$$]; show them as display math with the number as a tag. */
function formatChunkText(text: string): string {
  return text.replace(EQUATION, (_m, n, latex) => `\n\n$$\n${latex}${n ? `\\qquad\\text{(${n})}` : ""}\n$$\n\n`);
}

/** The first ~420 characters of a chunk, never cutting through an equation marker (the rest shows on "show all"). */
function preview(text: string, limit = 420): string {
  if (text.length <= limit) return text;
  let cut = limit;
  for (const m of text.matchAll(EQUATION)) {
    const start = m.index ?? 0, end = start + m[0].length;
    if (start < cut && cut < end) cut = end;
  }
  return text.slice(0, cut);
}
