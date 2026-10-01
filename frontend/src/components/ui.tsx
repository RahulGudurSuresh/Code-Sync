import type { ReactNode } from "react";
import { AlertTriangle, CheckCircle2, Loader2, XCircle } from "lucide-react";

export function Badge({ children, tone = "slate", title }: { children: ReactNode; tone?: Tone; title?: string }) {
  return (
    <span title={title} className={`inline-flex items-center rounded-md px-1.5 py-0.5 text-xs font-medium ${TONES[tone]}`}>
      {children}
    </span>
  );
}

export type Tone = "slate" | "red" | "amber" | "emerald" | "sky" | "violet" | "blue";

const TONES: Record<Tone, string> = {
  slate: "bg-slate-100 text-slate-700 ring-1 ring-slate-200",
  red: "bg-red-50 text-red-700 ring-1 ring-red-200",
  amber: "bg-amber-50 text-amber-800 ring-1 ring-amber-200",
  emerald: "bg-emerald-50 text-emerald-700 ring-1 ring-emerald-200",
  sky: "bg-sky-50 text-sky-700 ring-1 ring-sky-200",
  violet: "bg-violet-50 text-violet-700 ring-1 ring-violet-200",
  blue: "bg-blue-600 text-white",
};

export function StatusPill({ status, stage }: { status: string; stage?: string }) {
  const busy = !["ready", "failed", "done"].includes(status);
  const tone: Tone = status === "failed" ? "red" : busy ? "amber" : "emerald";
  return (
    <span className="inline-flex items-center gap-1.5">
      <Badge tone={tone}>
        {busy ? <Loader2 className="mr-1 h-3 w-3 animate-spin" /> : status === "failed" ? <XCircle className="mr-1 h-3 w-3" /> : <CheckCircle2 className="mr-1 h-3 w-3" />}
        {status}
      </Badge>
      {busy && stage && <span className="text-xs text-slate-500">{stage}</span>}
    </span>
  );
}

export function Card({ children, className = "" }: { children: ReactNode; className?: string }) {
  return <div className={`rounded-xl border border-slate-200 bg-white shadow-sm ${className}`}>{children}</div>;
}

export function Tile({ label, value, tone = "slate", hint }: { label: string; value: ReactNode; tone?: Tone; hint?: string }) {
  const ring = { slate: "border-slate-200", red: "border-red-200", amber: "border-amber-200", emerald: "border-emerald-200", sky: "border-sky-200", violet: "border-violet-200", blue: "border-blue-200" }[tone];
  return (
    <div className={`rounded-xl border ${ring} bg-white px-4 py-3 shadow-sm`} title={hint}>
      <div className="text-xs font-medium uppercase tracking-wide text-slate-500">{label}</div>
      <div className="mt-1 text-2xl font-semibold text-slate-900">{value}</div>
    </div>
  );
}

export function Button({
  children, onClick, kind = "primary", disabled, type = "button", title,
}: { children: ReactNode; onClick?: () => void; kind?: "primary" | "secondary" | "ghost" | "danger"; disabled?: boolean; type?: "button" | "submit"; title?: string }) {
  const styles = {
    primary: "bg-blue-600 text-white hover:bg-blue-700 disabled:bg-blue-300",
    secondary: "bg-white text-slate-700 ring-1 ring-slate-300 hover:bg-slate-50 disabled:text-slate-400",
    ghost: "text-slate-600 hover:bg-slate-100",
    danger: "bg-white text-red-700 ring-1 ring-red-200 hover:bg-red-50",
  }[kind];
  return (
    <button type={type} title={title} onClick={onClick} disabled={disabled} className={`inline-flex items-center gap-1.5 rounded-lg px-3 py-1.5 text-sm font-medium transition ${styles} disabled:cursor-not-allowed`}>
      {children}
    </button>
  );
}

export function Tabs<T extends string>({ tabs, value, onChange }: { tabs: { id: T; label: string; count?: number }[]; value: T; onChange: (t: T) => void }) {
  return (
    <div className="flex flex-wrap gap-1 border-b border-slate-200">
      {tabs.map((t) => (
        <button key={t.id} onClick={() => onChange(t.id)} className={`-mb-px border-b-2 px-3 py-2 text-sm font-medium ${value === t.id ? "border-blue-600 text-blue-700" : "border-transparent text-slate-500 hover:text-slate-800"}`}>
          {t.label}
          {t.count !== undefined && <span className="ml-1.5 rounded-full bg-slate-100 px-1.5 py-0.5 text-xs text-slate-600">{t.count}</span>}
        </button>
      ))}
    </div>
  );
}

export function Empty({ children }: { children: ReactNode }) {
  return <div className="rounded-xl border border-dashed border-slate-300 p-8 text-center text-sm text-slate-500">{children}</div>;
}

export function ErrorNote({ error }: { error: string | null }) {
  if (!error) return null;
  return (
    <div className="flex items-start gap-2 rounded-lg border border-red-200 bg-red-50 px-3 py-2 text-sm text-red-800">
      <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0" />
      <span>{error}</span>
    </div>
  );
}

export const fmtSeconds = (s?: number) => (s === undefined ? "" : s >= 90 ? `${(s / 60).toFixed(1)} min` : `${Math.round(s)} s`);
export const fmtDate = (iso: string) => (iso ? new Date(iso).toLocaleString() : "");
