import { useState } from "react";
import { Link } from "react-router-dom";
import { Upload } from "lucide-react";
import { api, isBusy, type Document } from "../api";
import { usePolling } from "../hooks";
import { Badge, Button, Card, Empty, ErrorNote, fmtDate, StatusPill } from "../components/ui";

export default function Library() {
  const { data: docs, error, refresh } = usePolling(api.documents, (d) => d.some((x) => isBusy(x.status)));
  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-xl font-semibold text-slate-900">Code library</h1>
        <p className="mt-1 text-sm text-slate-500">Upload the codes that apply to a project. Each one is parsed, its drawings read, and its requirements extracted once; comparisons reuse that work.</p>
      </div>
      <UploadForm onDone={refresh} />
      <ErrorNote error={error} />
      {docs && docs.length === 0 && <Empty>No documents yet. Upload a PDF above.</Empty>}
      {docs && docs.length > 0 && <DocumentTable docs={docs} />}
    </div>
  );
}

function UploadForm({ onDone }: { onDone: () => void }) {
  const [file, setFile] = useState<File | null>(null);
  const [label, setLabel] = useState("");
  const [name, setName] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const submit = async () => {
    if (!file || !label.trim()) return;
    setBusy(true);
    setError(null);
    try {
      await api.upload(file, label.trim(), name.trim() || file.name.replace(/\.pdf$/i, ""));
      setFile(null);
      setLabel("");
      setName("");
      onDone();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  };

  return (
    <Card className="p-4">
      <div className="grid gap-3 md:grid-cols-[1fr_140px_1fr_auto]">
        <label className="block text-sm">
          <span className="mb-1 block text-xs font-medium text-slate-500">PDF</span>
          <input type="file" accept="application/pdf" onChange={(e) => setFile(e.target.files?.[0] ?? null)} className="block w-full text-sm text-slate-700 file:mr-3 file:rounded-lg file:border-0 file:bg-slate-100 file:px-3 file:py-1.5 file:text-sm file:font-medium file:text-slate-700 hover:file:bg-slate-200" />
        </label>
        <label className="block text-sm">
          <span className="mb-1 block text-xs font-medium text-slate-500">Label (short, e.g. BP)</span>
          <input value={label} onChange={(e) => setLabel(e.target.value)} placeholder="BP" className="w-full rounded-lg border border-slate-300 px-2 py-1.5 text-sm" />
        </label>
        <label className="block text-sm">
          <span className="mb-1 block text-xs font-medium text-slate-500">Name (optional)</span>
          <input value={name} onChange={(e) => setName(e.target.value)} placeholder="BP WBU-GP 04-20-1 Civil Engineering" className="w-full rounded-lg border border-slate-300 px-2 py-1.5 text-sm" />
        </label>
        <div className="flex items-end">
          <Button onClick={submit} disabled={busy || !file || !label.trim()}><Upload className="h-4 w-4" /> {busy ? "Uploading" : "Upload and ingest"}</Button>
        </div>
      </div>
      <div className="mt-2"><ErrorNote error={error} /></div>
    </Card>
  );
}

function DocumentTable({ docs }: { docs: Document[] }) {
  return (
    <Card>
      <table className="w-full text-sm">
        <thead className="bg-slate-50 text-left text-xs uppercase tracking-wide text-slate-500">
          <tr>
            <th className="px-4 py-2">Label</th>
            <th className="px-4 py-2">Document</th>
            <th className="px-4 py-2">Pages</th>
            <th className="px-4 py-2">Status</th>
            <th className="px-4 py-2">Clauses</th>
            <th className="px-4 py-2">Drawings</th>
            <th className="px-4 py-2">Requirements</th>
            <th className="px-4 py-2">Added</th>
          </tr>
        </thead>
        <tbody className="divide-y divide-slate-100">
          {docs.map((d) => {
            const c = d.counts as Record<string, number | undefined>;
            return (
              <tr key={d.doc_id} className="hover:bg-slate-50">
                <td className="px-4 py-2"><Badge tone="blue">{d.label}</Badge></td>
                <td className="px-4 py-2">
                  <Link to={`/documents/${d.doc_id}`} className="font-medium text-blue-700 hover:underline">{d.name}</Link>
                  <div className="text-xs text-slate-500">{d.filename}</div>
                  {d.error && <div className="mt-1 text-xs text-red-700">{d.error}</div>}
                </td>
                <td className="px-4 py-2">{d.pages || ""}</td>
                <td className="px-4 py-2"><StatusPill status={d.status} stage={d.stage} /></td>
                <td className="px-4 py-2">{c.clauses ?? ""}</td>
                <td className="px-4 py-2">{c.figures ?? ""}</td>
                <td className="px-4 py-2">
                  {c.requirements ?? ""}
                  {c.set_aside ? <span className="ml-1 text-xs text-slate-500">({c.set_aside} set aside)</span> : null}
                </td>
                <td className="px-4 py-2 text-xs text-slate-500">{fmtDate(d.created_at)}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </Card>
  );
}
