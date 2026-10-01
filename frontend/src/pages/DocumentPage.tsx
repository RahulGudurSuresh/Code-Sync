import { useEffect, useMemo, useState } from "react";
import { Link, useParams } from "react-router-dom";
import { ArrowLeft, ExternalLink, MessageSquareText, RefreshCw } from "lucide-react";
import { api, isBusy, type Document, type Element, type PageInfo, type Requirement } from "../api";
import { usePolling, useTab } from "../hooks";
import { Badge, Button, Card, Empty, ErrorNote, fmtSeconds, StatusPill, Tabs, Tile, type Tone } from "../components/ui";

type Tab = "clauses" | "requirements" | "drawings";

export default function DocumentPage() {
  const { id = "" } = useParams();
  const { data: doc, error, refresh } = usePolling(() => api.document(id), (d) => isBusy(d.status), 3000, [id]);
  const [tab, setTab] = useTab<Tab>("clauses", ["clauses", "requirements", "drawings"]);
  const [elements, setElements] = useState<{ pages: PageInfo[]; elements: Element[] } | null>(null);
  const [reqs, setReqs] = useState<Requirement[] | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);

  useEffect(() => {
    if (!doc) return;
    setLoadError(null);
    if (doc.status !== "new") api.elements(id).then(setElements).catch((e) => setLoadError(String(e)));
    if (["extracted", "ready"].includes(doc.status)) api.requirements(id).then(setReqs).catch((e) => setLoadError(String(e)));
  }, [id, doc?.status]);

  if (!doc) return <ErrorNote error={error} />;
  const c = doc.counts as Record<string, number | undefined>;
  const figures = elements?.elements.filter((e) => e.kind === "figure") ?? [];

  return (
    <div className="space-y-5">
      <Link to="/" className="inline-flex items-center gap-1 text-sm text-slate-500 hover:text-slate-800"><ArrowLeft className="h-4 w-4" /> Library</Link>
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <div className="flex items-center gap-2"><Badge tone="blue">{doc.label}</Badge><h1 className="text-xl font-semibold text-slate-900">{doc.name}</h1></div>
          <div className="mt-1 flex items-center gap-3 text-sm text-slate-500">
            <span>{doc.filename}</span><span>{doc.pages} pages</span>
            <StatusPill status={doc.status} stage={doc.stage} />
          </div>
          {doc.error && <ErrorNote error={doc.error} />}
        </div>
        <div className="flex gap-2">
          <a href={api.pdfUrl(id)} target="_blank" rel="noreferrer"><Button kind="secondary"><ExternalLink className="h-4 w-4" /> Open PDF</Button></a>
          <Button kind="secondary" onClick={() => api.reingest(id).then(refresh)} disabled={isBusy(doc.status)} title="Re-run the stages whose inputs changed"><RefreshCw className="h-4 w-4" /> Re-ingest</Button>
          <Button kind="secondary" onClick={() => api.index(id).then(refresh)} disabled={isBusy(doc.status)} title="Chunk, describe, embed and store this document for Ask the Code"><MessageSquareText className="h-4 w-4" /> Index for Ask</Button>
        </div>
      </div>

      <div className="grid grid-cols-2 gap-3 md:grid-cols-6">
        <Tile label="Clauses" value={c.clauses ?? "–"} />
        <Tile label="Drawings" value={c.figures ?? "–"} />
        <Tile label="Requirements" value={c.requirements ?? "–"} hint="In the comparison" />
        <Tile label="Set aside" value={c.set_aside ?? "–"} tone="violet" hint="Drawing statements the two readings disagreed on" />
        <Tile label="Ask chunks" value={c.chunks ?? "–"} tone="sky" hint="Passages indexed for Ask the Code" />
        <Tile label="Ingest time" value={fmtSeconds(Object.values(doc.timings).reduce((a, b) => a + b, 0)) || "–"} hint={JSON.stringify(doc.timings)} />
      </div>

      <ErrorNote error={loadError} />
      <Tabs<Tab> value={tab} onChange={setTab} tabs={[
        { id: "clauses", label: "Clauses", count: elements?.elements.filter((e) => e.kind === "clause").length },
        { id: "requirements", label: "Requirements", count: reqs?.length },
        { id: "drawings", label: "Drawings", count: figures.length },
      ]} />

      {tab === "clauses" && (elements ? <ClauseTree doc={doc} elements={elements.elements} /> : <Empty>Not parsed yet.</Empty>)}
      {tab === "requirements" && (reqs ? <RequirementTable reqs={reqs} /> : <Empty>Requirements are extracted after parsing; check back when the document is ready.</Empty>)}
      {tab === "drawings" && (figures.length ? <Drawings doc={doc} figures={figures} reqs={reqs ?? []} /> : <Empty>No drawing sheets in this document.</Empty>)}
    </div>
  );
}

function ClauseTree({ doc, elements }: { doc: Document; elements: Element[] }) {
  return (
    <Card className="divide-y divide-slate-100">
      {elements.map((e) => {
        const front = Boolean(e.extra.front_matter);
        return (
          <div key={e.id} className={`flex gap-3 px-4 py-1.5 text-sm ${front ? "text-slate-400" : ""}`} style={{ paddingLeft: `${16 + e.depth * 18}px` }}>
            <span className="w-10 shrink-0 text-right text-xs text-slate-400">p.{e.page}</span>
            {e.kind === "clause" ? (
              <>
                <span className={`shrink-0 font-mono text-xs ${e.is_heading ? "text-slate-900" : "text-slate-500"}`}>{e.ref}</span>
                <span className={e.is_heading ? "font-semibold text-slate-900" : ""}>{e.text}</span>
              </>
            ) : (
              <span className="flex items-center gap-2">
                <Badge tone={e.kind === "figure" ? "violet" : "sky"}>{e.kind}</Badge>
                <span className="font-mono text-xs text-slate-500">{e.ref}</span>
                {e.image && <a className="text-blue-700 hover:underline" href={api.figureUrl(doc.doc_id, e.image)} target="_blank" rel="noreferrer">view</a>}
                {e.text && <span className="text-slate-600">{e.text.slice(0, 160)}</span>}
              </span>
            )}
          </div>
        );
      })}
    </Card>
  );
}

const MODALITY_TONE: Record<string, Tone> = { shall: "red", "shall not": "red", should: "amber", "should not": "amber", may: "sky", detail: "violet", info: "slate" };

export function ModalityBadge({ r }: { r: Requirement }) {
  const m = r.from_figure ? "drawing" : r.modality;
  return <Badge tone={MODALITY_TONE[r.modality] ?? "slate"}>{m}</Badge>;
}

function RequirementTable({ reqs }: { reqs: Requirement[] }) {
  const [q, setQ] = useState("");
  const [source, setSource] = useState<"all" | "text" | "drawing" | "aside">("all");
  const rows = useMemo(() => reqs.filter((r) => {
    if (source === "text" && r.from_figure) return false;
    if (source === "drawing" && !r.from_figure) return false;
    if (source === "aside" && !r.excluded) return false;
    const hay = `${r.ref} ${r.statement} ${r.subject} ${r.parameter} ${r.value}`.toLowerCase();
    return hay.includes(q.toLowerCase());
  }), [reqs, q, source]);
  return (
    <div className="space-y-3">
      <div className="flex flex-wrap items-center gap-2">
        <input value={q} onChange={(e) => setQ(e.target.value)} placeholder="Search requirements" className="w-72 rounded-lg border border-slate-300 px-2 py-1.5 text-sm" />
        {(["all", "text", "drawing", "aside"] as const).map((s) => (
          <button key={s} onClick={() => setSource(s)} className={`rounded-lg px-2.5 py-1 text-xs font-medium ${source === s ? "bg-slate-900 text-white" : "bg-slate-100 text-slate-600"}`}>
            {{ all: "All", text: "From clauses", drawing: "From drawings", aside: "Set aside" }[s]}
          </button>
        ))}
        <span className="text-xs text-slate-500">{rows.length} of {reqs.length}</span>
      </div>
      <Card>
        <table className="w-full text-sm">
          <thead className="bg-slate-50 text-left text-xs uppercase tracking-wide text-slate-500">
            <tr><th className="px-3 py-2">Clause</th><th className="px-3 py-2">Basis</th><th className="px-3 py-2">Requirement</th><th className="px-3 py-2">Subject / parameter</th><th className="px-3 py-2">Check</th></tr>
          </thead>
          <tbody className="divide-y divide-slate-100">
            {rows.map((r) => (
              <tr key={r.id} className={r.excluded ? "bg-violet-50/40 text-slate-500" : ""}>
                <td className="px-3 py-2 font-mono text-xs">{r.ref}<div className="text-slate-400">p.{r.page}</div></td>
                <td className="px-3 py-2"><ModalityBadge r={r} /></td>
                <td className="px-3 py-2">
                  {r.statement}
                  {r.quote && !r.from_figure && <div className="mt-1 text-xs text-slate-500">“{r.quote}”{!r.quote_verified && <span className="ml-1 text-amber-700">(quote not found verbatim)</span>}</div>}
                  {r.check_note && <div className="mt-1 text-xs text-violet-700">{r.check_note}</div>}
                </td>
                <td className="px-3 py-2 text-xs text-slate-600">{r.subject}<div className="text-slate-400">{r.parameter}{r.value ? ` = ${r.value}` : ""}</div></td>
                <td className="px-3 py-2 text-xs">{r.from_figure ? <Badge tone={r.drawing_check === "confirmed" ? "emerald" : "violet"}>{r.drawing_check || "–"}</Badge> : r.quote_verified ? <Badge tone="emerald">quoted</Badge> : <Badge tone="amber">unverified</Badge>}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </Card>
    </div>
  );
}

function Drawings({ doc, figures, reqs }: { doc: Document; figures: Element[]; reqs: Requirement[] }) {
  return (
    <div className="grid gap-4 lg:grid-cols-2">
      {figures.map((f) => {
        const mine = reqs.filter((r) => r.element_id === f.id);
        return (
          <Card key={f.id} className="overflow-hidden">
            <div className="flex items-center justify-between border-b border-slate-100 px-4 py-2">
              <div className="text-sm font-medium text-slate-800">{f.ref}</div>
              <span className="text-xs text-slate-500">{mine.filter((r) => !r.excluded).length} used, {mine.filter((r) => r.excluded).length} set aside</span>
            </div>
            {f.image && (
              <a href={api.figureUrl(doc.doc_id, f.image)} target="_blank" rel="noreferrer">
                <img src={api.figureUrl(doc.doc_id, f.image)} alt={f.ref} className="max-h-96 w-full object-contain bg-slate-50" loading="lazy" />
              </a>
            )}
            <ul className="max-h-72 divide-y divide-slate-100 overflow-auto text-sm">
              {mine.map((r) => (
                <li key={r.id} className={`flex gap-2 px-4 py-1.5 ${r.excluded ? "text-slate-400" : ""}`}>
                  <Badge tone={r.excluded ? "violet" : "emerald"}>{r.drawing_check || "?"}</Badge>
                  <span>{r.statement}</span>
                </li>
              ))}
            </ul>
          </Card>
        );
      })}
    </div>
  );
}
