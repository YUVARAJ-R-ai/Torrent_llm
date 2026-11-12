import { useCallback, useEffect, useRef, useState } from "react";

import type { BackendStatus, ReadPath } from "../../shared/types";

/** The backend's status, live. */
export function useBackend(): BackendStatus | null {
  const [status, setStatus] = useState<BackendStatus | null>(null);
  useEffect(() => {
    let alive = true;
    void window.tllm.backend.status().then((s) => alive && setStatus(s));
    const stop = window.tllm.backend.onStatus((s) => setStatus(s));
    return () => {
      alive = false;
      stop();
    };
  }, []);
  return status;
}

export interface Polled<T> {
  data: T | null;
  error: string | null;
  loading: boolean;
  refresh: () => void;
}

/**
 * GET ``path`` every ``intervalMs`` while ``enabled``. Keeps the last good data when a
 * poll fails (a device restarting should not blank the screen) and reports the error.
 */
export function usePoll<T>(
  path: ReadPath,
  { intervalMs = 2000, enabled = true, query }: { intervalMs?: number; enabled?: boolean; query?: Record<string, string | number> } = {},
): Polled<T> {
  const [data, setData] = useState<T | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const queryKey = JSON.stringify(query ?? {});
  const tick = useRef(0);

  const load = useCallback(async () => {
    const mine = ++tick.current;
    setLoading(true);
    try {
      const value = await window.tllm.api.get<T>(path, JSON.parse(queryKey));
      if (mine === tick.current) {
        setData(value);
        setError(null);
      }
    } catch (e) {
      if (mine === tick.current) setError(e instanceof Error ? e.message : String(e));
    } finally {
      if (mine === tick.current) setLoading(false);
    }
  }, [path, queryKey]);

  useEffect(() => {
    if (!enabled) return;
    void load();
    const timer = setInterval(() => void load(), intervalMs);
    return () => clearInterval(timer);
  }, [enabled, intervalMs, load]);

  return { data, error, loading, refresh: () => void load() };
}

export function errorText(e: unknown): string {
  const text = e instanceof Error ? e.message : String(e);
  // Electron prefixes IPC errors with this; people do not need to see it.
  return text.replace(/^Error invoking remote method '[^']+': (Error: )?/, "");
}
