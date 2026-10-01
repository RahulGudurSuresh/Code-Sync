import { useCallback, useEffect, useRef, useState } from "react";
import { useSearchParams } from "react-router-dom";

/** A tab selection kept in the URL (?tab=...) so views can be linked to. */
export function useTab<T extends string>(fallback: T, allowed: readonly T[]): [T, (t: T) => void] {
  const [params, setParams] = useSearchParams();
  const raw = params.get("tab") as T | null;
  const tab = raw && allowed.includes(raw) ? raw : fallback;
  const setTab = (t: T) => setParams((p) => { p.set("tab", t); return p; }, { replace: true });
  return [tab, setTab];
}

/** Fetch once, and keep re-fetching every `intervalMs` while `busy(data)`
 *  says the thing is still in progress (a document ingesting, a comparison
 *  running). */
export function usePolling<T>(load: () => Promise<T>, busy: (data: T) => boolean, intervalMs = 3000, deps: unknown[] = []) {
  const [data, setData] = useState<T | null>(null);
  const [error, setError] = useState<string | null>(null);
  const timer = useRef<number | null>(null);
  const alive = useRef(true);

  const refresh = useCallback(async () => {
    try {
      const d = await load();
      if (!alive.current) return;
      setData(d);
      setError(null);
      if (busy(d)) timer.current = window.setTimeout(refresh, intervalMs);
    } catch (e) {
      if (!alive.current) return;
      setError(e instanceof Error ? e.message : String(e));
      timer.current = window.setTimeout(refresh, intervalMs * 2);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, deps);

  useEffect(() => {
    alive.current = true;
    refresh();
    return () => {
      alive.current = false;
      if (timer.current) window.clearTimeout(timer.current);
    };
  }, [refresh]);

  return { data, error, refresh };
}
