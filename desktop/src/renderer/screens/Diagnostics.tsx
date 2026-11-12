/**
 * Diagnostics: Debugging information and troubleshooting aid for failed starts and connectivity issues.
 * Works in every backend phase. Shows device runtime status, live logs with copy/refresh,
 * and key troubleshooting instructions.
 */

import { Copy, RefreshCw } from "lucide-react";
import { useEffect, useRef, useState } from "react";

import type { ScreenProps } from "../App";
import { Button, Card, ErrorNote, PageTitle, Pill, type Tone } from "../components/ui";
import { errorText } from "../lib/hooks";

const PHASE: Record<string, { tone: Tone; text: string }> = {
  missing: { tone: "neutral", text: "Not installed" },
  installing: { tone: "warn", text: "Installing" },
  stopped: { tone: "neutral", text: "Stopped" },
  starting: { tone: "warn", text: "Starting" },
  running: { tone: "good", text: "Running" },
  stopping: { tone: "warn", text: "Stopping" },
  error: { tone: "critical", text: "Problem" },
};

/** ``snapshot`` followed by the part of ``live`` it does not already end with. */
export function withLiveTail(snapshot: string[], live: string[]): string[] {
  for (let overlap = Math.min(snapshot.length, live.length); overlap > 0; overlap--) {
    let same = true;
    for (let i = 0; i < overlap; i++) {
      if (snapshot[snapshot.length - overlap + i] !== live[i]) {
        same = false;
        break;
      }
    }
    if (same) return [...snapshot, ...live.slice(overlap)];
  }
  return [...snapshot, ...live];
}

export default function DiagnosticsScreen({ status }: ScreenProps) {
  const [logs, setLogs] = useState<string[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [copied, setCopied] = useState(false);
  const logBoxRef = useRef<HTMLDivElement | null>(null);
  const copiedTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  // Lines that streamed in while a snapshot was being fetched: kept unless the snapshot
  // already ends with them, so nothing is lost or doubled.
  const liveDuringFetch = useRef<string[]>([]);
  const fetching = useRef(false);
  useEffect(
    () => () => {
      if (copiedTimer.current) clearTimeout(copiedTimer.current);
    },
    [],
  );

  const fetchLogs = async () => {
    setBusy(true);
    setError(null);
    liveDuringFetch.current = [];
    fetching.current = true;
    try {
      const lines = await window.tllm.backend.logs();
      setLogs(withLiveTail(lines, liveDuringFetch.current).slice(-500));
    } catch (err) {
      setError(errorText(err));
    } finally {
      fetching.current = false;
      setBusy(false);
    }
  };

  useEffect(() => {
    const unsub = window.tllm.backend.onLog((line) => {
      if (fetching.current) liveDuringFetch.current.push(line);
      setLogs((prev) => [...prev, line].slice(-500));
    });

    void fetchLogs();

    return () => {
      unsub();
    };
  }, []);

  useEffect(() => {
    if (logBoxRef.current) {
      logBoxRef.current.scrollTop = logBoxRef.current.scrollHeight;
    }
  }, [logs]);

  const handleCopy = async () => {
    setBusy(true);
    setError(null);
    try {
      await window.tllm.system.copy(logs.join("\n"));
      setCopied(true);
      if (copiedTimer.current) clearTimeout(copiedTimer.current);
      copiedTimer.current = setTimeout(() => setCopied(false), 2000);
    } catch (err) {
      setError(errorText(err));
    } finally {
      setBusy(false);
    }
  };

  const phaseInfo = PHASE[status.phase] ?? PHASE.error;

  return (
    <div className="mx-auto max-w-4xl space-y-4">
      <PageTitle title="Diagnostics">For when something does not work.</PageTitle>

      <Card title="Device">
        <dl className="grid grid-cols-1 gap-x-6 gap-y-3 sm:grid-cols-2">
          <div>
            <dt className="text-xs" style={{ color: "var(--text-muted)" }}>
              Phase
            </dt>
            <dd className="mt-0.5">
              <Pill tone={phaseInfo.tone}>{phaseInfo.text}</Pill>
            </dd>
          </div>
          <div>
            <dt className="text-xs" style={{ color: "var(--text-muted)" }}>
              Process ID
            </dt>
            <dd className="mt-0.5 text-sm font-medium tabular-nums">
              {status.pid ?? "—"}
            </dd>
          </div>
          <div>
            <dt className="text-xs" style={{ color: "var(--text-muted)" }}>
              URL
            </dt>
            <dd className="mt-0.5 text-sm font-medium">{status.url || "—"}</dd>
          </div>
          <div>
            <dt className="text-xs" style={{ color: "var(--text-muted)" }}>
              Executable
            </dt>
            <dd className="mt-0.5 text-xs mono break-all">
              {status.executable || "not found"}
            </dd>
          </div>
          <div className="sm:col-span-2">
            <dt className="text-xs" style={{ color: "var(--text-muted)" }}>
              Message
            </dt>
            <dd className="mt-0.5 text-sm" style={{ color: "var(--text-secondary)" }}>
              {status.message || "—"}
            </dd>
          </div>
        </dl>
      </Card>

      <Card
        title="Log"
        actions={
          <div className="flex items-center gap-2">
            {copied && (
              <span className="text-xs font-medium" style={{ color: "var(--status-good)" }}>
                Copied
              </span>
            )}
            <Button onClick={handleCopy} disabled={busy || logs.length === 0}>
              <Copy size={14} aria-hidden /> Copy log
            </Button>
            <Button onClick={fetchLogs} disabled={busy}>
              <RefreshCw size={14} aria-hidden /> Refresh
            </Button>
          </div>
        }
      >
        {error && (
          <div className="mb-2">
            <ErrorNote>{error}</ErrorNote>
          </div>
        )}
        <div
          ref={logBoxRef}
          className="mono text-xs overflow-y-auto rounded p-3 select-text whitespace-pre-wrap break-all"
          style={{
            maxHeight: "24rem",
            background: "var(--surface-2)",
            color: "var(--text-primary)",
            borderColor: "var(--border)",
          }}
        >
          {logs.length > 0 ? logs.join("\n") : "No logs available."}
        </div>
      </Card>

      <Card title="Help">
        <ul
          className="list-disc space-y-2 pl-5 text-sm"
          style={{ color: "var(--text-secondary)" }}
        >
          <li>
            <strong style={{ color: "var(--text-primary)" }}>Firewall:</strong> Allow
            incoming TCP 9100 so other devices can reach you directly.
          </li>
          <li>
            <strong style={{ color: "var(--text-primary)" }}>Hotspot isolation:</strong> If
            phones or guest Wi-Fi block direct connections, use the invite link;
            devices still connect through relays.
          </li>
          <li>
            <strong style={{ color: "var(--text-primary)" }}>Model download:</strong> In an
            invite swarm, a device without the model fetches it from the others and checks
            every file. Otherwise it downloads from the internet the first time.
          </li>
          <li>
            <strong style={{ color: "var(--text-primary)" }}>
              Leave and rejoin the swarm:
            </strong>{" "}
            If a connection stalls or fails, leave the swarm on Home and rejoin to
            start fresh.
          </li>
        </ul>
      </Card>
    </div>
  );
}
