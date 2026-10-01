import { Link, NavLink, Route, Routes } from "react-router-dom";
import { Activity, FileStack, GitCompareArrows, MessageSquareText } from "lucide-react";
import { useEffect, useState } from "react";
import { api } from "./api";
import Library from "./pages/Library";
import DocumentPage from "./pages/DocumentPage";
import Comparisons from "./pages/Comparisons";
import ComparisonPage from "./pages/ComparisonPage";
import Ask from "./pages/Ask";

export default function App() {
  const [health, setHealth] = useState<Record<string, unknown> | null>(null);
  const [down, setDown] = useState(false);
  useEffect(() => {
    api.health().then((h) => { setHealth(h); setDown(false); }).catch(() => setDown(true));
  }, []);

  const nav = ({ isActive }: { isActive: boolean }) =>
    `flex items-center gap-1.5 rounded-lg px-3 py-1.5 text-sm font-medium ${isActive ? "bg-slate-900 text-white" : "text-slate-600 hover:bg-slate-100"}`;

  return (
    <div className="min-h-screen">
      <header className="sticky top-0 z-10 border-b border-slate-200 bg-white/90 backdrop-blur">
        <div className="mx-auto flex max-w-7xl items-center gap-6 px-4 py-3">
          <Link to="/" className="flex items-center gap-2">
            <span className="grid h-8 w-8 place-items-center rounded-lg bg-blue-600 text-white"><GitCompareArrows className="h-4 w-4" /></span>
            <span className="text-base font-semibold tracking-tight text-slate-900">CodeSync Analyzer</span>
          </Link>
          <nav className="flex items-center gap-1">
            <NavLink to="/" end className={nav}><FileStack className="h-4 w-4" /> Library</NavLink>
            <NavLink to="/comparisons" className={nav}><GitCompareArrows className="h-4 w-4" /> Comparisons</NavLink>
            <NavLink to="/ask" className={nav}><MessageSquareText className="h-4 w-4" /> Ask</NavLink>
          </nav>
          <div className="ml-auto flex items-center gap-2 text-xs text-slate-500" title={health ? JSON.stringify(health, null, 1) : ""}>
            <Activity className={`h-4 w-4 ${down ? "text-red-500" : "text-emerald-500"}`} />
            {down ? "backend unreachable" : health ? `engine ${String(health.engine).replace("modal:", "")}` : "connecting"}
          </div>
        </div>
      </header>
      <main className="mx-auto max-w-7xl px-4 py-6">
        <Routes>
          <Route path="/" element={<Library />} />
          <Route path="/documents/:id" element={<DocumentPage />} />
          <Route path="/comparisons" element={<Comparisons />} />
          <Route path="/comparisons/:id" element={<ComparisonPage />} />
          <Route path="/ask" element={<Ask />} />
        </Routes>
      </main>
    </div>
  );
}
