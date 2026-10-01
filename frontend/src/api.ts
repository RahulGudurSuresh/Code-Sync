// Typed client for the CodeSync backend (codesync/api.py). The shapes mirror
// codesync/schema.py; keep them in step.

export type DocStatus = "new" | "parsed" | "figures" | "extracted" | "ready" | "failed";

export interface Document {
  doc_id: string;
  name: string;
  label: string;
  filename: string;
  pages: number;
  status: DocStatus;
  stage: string;
  error: string;
  counts: Record<string, unknown>;
  timings: Record<string, number>;
  created_at: string;
}

export interface PageInfo {
  page: number;
  kind: "text" | "drawing" | "scanned" | "toc" | "empty";
  label: string;
}

export interface Element {
  id: string;
  kind: "clause" | "table" | "figure" | "equation";
  page: number;
  number: string;
  ref: string;
  parent: string | null;
  depth: number;
  text: string;
  is_heading: boolean;
  heading_trail: string[];
  source: string;
  image: string | null;
  extra: Record<string, unknown>;
}

export interface Requirement {
  id: string;
  doc_id: string;
  element_id: string;
  ref: string;
  page: number;
  statement: string;
  subject: string;
  parameter: string;
  value: string;
  modality: string;
  quote: string;
  quote_verified: boolean;
  references: string[];
  from_figure: boolean;
  drawing_check: string;
  check_note: string;
  excluded: boolean;
}

export type Category = "conflict" | "similar" | "unique";

export interface Finding {
  id: string;
  category: Category;
  type: "contradiction" | "more_stringent" | "equivalent" | "only_in_a" | "only_in_b";
  topic: string;
  a: Requirement | null;
  b: Requirement | null;
  similarity: number | null;
  also_b: string[];
  rationale: string;
  governing: string;
  stricter: string;
  action: string;
  severity: "high" | "medium" | "low" | "";
  drawing_check: string;
  needs_review: boolean;
  review_reason: string;
}

export interface Comparison {
  cmp_id: string;
  doc_a: string;
  doc_b: string;
  governing: string;
  status: "queued" | "running" | "done" | "failed";
  stage: string;
  error: string;
  summary: Record<string, any>;
  timings: Record<string, number>;
  created_at: string;
}

export interface Job {
  id: string;
  kind: string;
  target: string;
  status: "queued" | "running" | "done" | "failed";
  message: string;
  log: string[];
  error: string;
  created_at: string;
  finished_at: string;
}

export interface Review {
  decision: "accepted" | "rejected" | "";
  note: string;
  by: string;
  at?: string;
}

const BASE = import.meta.env.VITE_API_BASE ?? "";

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(BASE + path, init);
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      detail = body.detail ?? detail;
    } catch {
      /* not JSON */
    }
    throw new Error(`${res.status}: ${detail}`);
  }
  return res.json() as Promise<T>;
}

export const api = {
  health: () => request<Record<string, unknown>>("/health"),
  documents: () => request<Document[]>("/documents"),
  document: (id: string) => request<Document>(`/documents/${id}`),
  elements: (id: string, kind?: string) =>
    request<{ pages: PageInfo[]; elements: Element[] }>(`/documents/${id}/elements${kind ? `?kind=${kind}` : ""}`),
  requirements: (id: string) => request<Requirement[]>(`/documents/${id}/requirements`),
  figureUrl: (id: string, name: string) => `${BASE}/documents/${id}/figures/${encodeURIComponent(name)}`,
  pdfUrl: (id: string) => `${BASE}/documents/${id}/pdf`,
  upload: (file: File, label: string, name: string) => {
    const fd = new FormData();
    fd.append("file", file);
    fd.append("label", label);
    fd.append("name", name);
    return request<{ document: Document; job: Job | null }>("/documents", { method: "POST", body: fd });
  },
  reingest: (id: string) => request<Job>(`/documents/${id}/ingest`, { method: "POST" }),
  deleteDocument: (id: string) => request<{ deleted: string }>(`/documents/${id}`, { method: "DELETE" }),
  comparisons: () => request<Comparison[]>("/comparisons"),
  comparison: (id: string) => request<Comparison>(`/comparisons/${id}`),
  startComparison: (doc_a: string, doc_b: string, governing: string) =>
    request<{ comparison: Comparison; job: Job }>("/comparisons", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ doc_a, doc_b, governing }),
    }),
  findings: (id: string) => request<Finding[]>(`/comparisons/${id}/findings`),
  reportMd: async (id: string) => {
    const res = await fetch(`${BASE}/comparisons/${id}/report.md`);
    if (!res.ok) throw new Error(`${res.status}: report not ready`);
    return res.text();
  },
  reportMdUrl: (id: string) => `${BASE}/comparisons/${id}/report.md`,
  reportXlsxUrl: (id: string) => `${BASE}/comparisons/${id}/report.xlsx`,
  reviews: (id: string) => request<Record<string, Review>>(`/comparisons/${id}/reviews`),
  saveReview: (id: string, findingId: string, review: Review) =>
    request<Review>(`/comparisons/${id}/reviews/${findingId}`, {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(review),
    }),
  jobs: () => request<Job[]>("/jobs"),
  askScope: () => request<AskScope[]>("/ask/scope"),
  askWarm: () => request<{ warming: boolean }>("/ask/warm", { method: "POST" }),
  index: (docId: string) => request<Job>(`/documents/${docId}/index`, { method: "POST" }),
  ask: (question: string, doc_ids: string[], onEvent: (kind: string, data: any) => void) => askStream(question, doc_ids, onEvent),
};

export const isBusy = (status: string) => status !== "ready" && status !== "failed" && status !== "done";

// ---- Ask the Code ------------------------------------------------------------
export interface AskScope { doc_id: string; label: string; name: string; code: string; edition: string; chunks: number }

export interface ChunkRef { kind: string; target: string; text: string; chunk_id: string | null }

export interface Chunk {
  id: string; doc_id: string; code: string; edition: string; kind: string; ref: string;
  page: number; page_label: string; heading_path: string[]; context: string; text: string;
  notes: string[]; latex: string[]; table: { caption: string; columns: string[]; rows: string[][] } | null;
  image: string | null; description: string; refs: ChunkRef[]; element_ids: string[];
}

export interface Passage { tag: string; chunk: Chunk; why: string; score: number | null; vector_rank: number | null; keyword_rank: number | null }

/** POST /ask and read the server-sent events as they arrive. */
export async function askStream(question: string, doc_ids: string[], onEvent: (kind: string, data: any) => void): Promise<void> {
  const res = await fetch(`${BASE}/ask`, {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question, doc_ids }),
  });
  if (!res.ok || !res.body) throw new Error(`${res.status}: ${res.statusText}`);
  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    let cut: number;
    while ((cut = buffer.indexOf("\n\n")) >= 0) {
      const frame = buffer.slice(0, cut);
      buffer = buffer.slice(cut + 2);
      let kind = "message";
      const dataLines: string[] = [];
      for (const line of frame.split("\n")) {
        if (line.startsWith("event:")) kind = line.slice(6).trim();
        else if (line.startsWith("data:")) dataLines.push(line.slice(5).trim());
      }
      if (dataLines.length) onEvent(kind, JSON.parse(dataLines.join("\n")));
    }
  }
}
