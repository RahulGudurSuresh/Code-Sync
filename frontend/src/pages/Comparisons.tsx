import { useEffect, useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { Play } from "lucide-react";
import { api, isBusy, type Document } from "../api";
import { usePolling } from "../hooks";
import { Badge, Button, Card, Empty, ErrorNote, fmtDate, StatusPill } from "../components/ui";

export default function Comparisons() {
  const { data: cmps, error, refresh } = usePolling(api.comparisons, (d) => d.some((c) => isBusy(c.status)));
  const [docs, setDocs] = useState<Document[]>([]);
  useEffect(() => { api.documents().then(setDocs).catch(() => undefined); }, []);
  const byId = Object.fromEntries(docs.map((d) => [d.doc_id, d]));

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-xl font-semibold text-slate-900">Comparisons</h1>
        <p className="mt-1 text-sm text-slate-500">Pick two codes and say which one governs. The result lists what conflicts, what is stricter in one of them, what agrees, and what each says alone.</p>
      </div>
      <NewComparison docs={docs} onStarted={refresh} />
      <ErrorNote error={error} />
      {cmps && cmps.length === 0 && <Empty>No comparisons yet.</Empty>}
      {cmps && cmps.length > 0 && (
        <Card>
          <table className="w-full text-sm">
            <thead className="bg-slate-50 text-left text-xs uppercase tracking-wide text-slate-500">
              <tr><th className="px-4 py-2">Comparison</th><th className="px-4 py-2">Documents</th><th className="px-4 py-2">Governing</th><th className="px-4 py-2">Status</th><th className="px-4 py-2">Conflicts</th><th className="px-4 py-2">Similar</th><th className="px-4 py-2">Unique</th><th className="px-4 py-2">Started</th></tr>
            </thead>
            <tbody className="divide-y divide-slate-100">
              {[...cmps].reverse().map((c) => {
                const s = c.summary ?? {};
                const lab = (id: string) => byId[id]?.label ?? id;
                return (
                  <tr key={c.cmp_id} className="hover:bg-slate-50">
                    <td className="px-4 py-2"><Link to={`/comparisons/${c.cmp_id}`} className="font-mono text-xs font-medium text-blue-700 hover:underline">{c.cmp_id}</Link></td>
                    <td className="px-4 py-2"><Badge tone="blue">{lab(c.doc_a)}</Badge> <span className="text-slate-400">vs</span> <Badge tone="slate">{lab(c.doc_b)}</Badge></td>
                    <td className="px-4 py-2">{lab(c.governing)}</td>
                    <td className="px-4 py-2"><StatusPill status={c.status} stage={c.stage} />{c.error && <div className="text-xs text-red-700">{c.error}</div>}</td>
                    <td className="px-4 py-2">{s.conflicts ?? ""}</td>
                    <td className="px-4 py-2">{s.similar ?? ""}</td>
                    <td className="px-4 py-2">{s.unique_a !== undefined ? `${s.unique_a} / ${s.unique_b}` : ""}</td>
                    <td className="px-4 py-2 text-xs text-slate-500">{fmtDate(c.created_at)}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </Card>
      )}
    </div>
  );
}

function NewComparison({ docs, onStarted }: { docs: Document[]; onStarted: () => void }) {
  const navigate = useNavigate();
  const [a, setA] = useState("");
  const [b, setB] = useState("");
  const [gov, setGov] = useState<"a" | "b">("a");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const ready = docs.filter((d) => d.status === "ready");
  const start = async () => {
    setBusy(true); setError(null);
    try {
      const res = await api.startComparison(a, b, gov === "a" ? a : b);
      onStarted();
      navigate(`/comparisons/${res.comparison.cmp_id}`);
    } catch (e) { setError(e instanceof Error ? e.message : String(e)); } finally { setBusy(false); }
  };
  const Select = ({ value, onChange, exclude }: { value: string; onChange: (v: string) => void; exclude: string }) => (
    <select value={value} onChange={(e) => onChange(e.target.value)} className="w-full rounded-lg border border-slate-300 bg-white px-2 py-1.5 text-sm">
      <option value="">Choose a document</option>
      {docs.map((d) => <option key={d.doc_id} value={d.doc_id} disabled={d.doc_id === exclude || d.status !== "ready"}>{d.label} — {d.name}{d.status !== "ready" ? ` (${d.status})` : ""}</option>)}
    </select>
  );
  return (
    <Card className="p-4">
      <div className="grid gap-3 md:grid-cols-[1fr_1fr_auto_auto]">
        <label className="text-sm"><span className="mb-1 block text-xs font-medium text-slate-500">Document A</span><Select value={a} onChange={setA} exclude={b} /></label>
        <label className="text-sm"><span className="mb-1 block text-xs font-medium text-slate-500">Document B</span><Select value={b} onChange={setB} exclude={a} /></label>
        <div className="text-sm">
          <span className="mb-1 block text-xs font-medium text-slate-500">Governing</span>
          <div className="flex rounded-lg ring-1 ring-slate-300">
            {(["a", "b"] as const).map((g) => <button key={g} onClick={() => setGov(g)} className={`px-3 py-1.5 text-sm ${gov === g ? "bg-slate-900 text-white" : "text-slate-600"} ${g === "a" ? "rounded-l-lg" : "rounded-r-lg"}`}>{g.toUpperCase()}</button>)}
          </div>
        </div>
        <div className="flex items-end"><Button onClick={start} disabled={busy || !a || !b || a === b}><Play className="h-4 w-4" /> Run comparison</Button></div>
      </div>
      {ready.length < 2 && <p className="mt-2 text-xs text-slate-500">Two documents must be ready before a comparison can run.</p>}
      <div className="mt-2"><ErrorNote error={error} /></div>
    </Card>
  );
}
