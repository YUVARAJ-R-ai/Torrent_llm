/**
 * Activity: Live audit log of swarm events (devices joining, leaving, stage handovers).
 * Polls GET /events every 2s, deduping by t+text and keeping the 200 newest events.
 */

import { useEffect, useRef, useState } from "react";

import type { ScreenProps } from "../App";
import type { PoolEvent } from "../../shared/types";
import { Button, Card, Empty, ErrorNote, PageTitle, Pill, type Tone } from "../components/ui";
import { ago } from "../lib/format";
import { errorText } from "../lib/hooks";

/** Maps event kind to semantic tone: joined/up=good, down=critical, stage=accent, state=warn, else neutral. */
function eventTone(kind: string): Tone {
  if (kind === "joined" || kind === "up") return "good";
  if (kind === "down") return "critical";
  if (kind === "stage") return "accent";
  if (kind === "state") return "warn";
  return "neutral";
}

export default function ActivityScreen({ status, go }: ScreenProps) {
  const running = status.phase === "running";
  const [events, setEvents] = useState<PoolEvent[]>([]);
  const [error, setError] = useState<string | null>(null);
  const lastT = useRef<number>(0);
  // Re-render now and then so "12s ago" keeps moving between new events.
  const [, setTick] = useState(0);
  useEffect(() => {
    const timer = setInterval(() => setTick((n) => n + 1), 15_000);
    return () => clearInterval(timer);
  }, []);

  useEffect(() => {
    if (!running) return;
    let alive = true;

    const poll = async () => {
      try {
        const incoming = await window.tllm.api.get<PoolEvent[]>("/events", {
          since: lastT.current,
        });
        if (!alive) return;
        setError(null);
        if (Array.isArray(incoming) && incoming.length > 0) {
          for (const ev of incoming) {
            if (typeof ev.t === "number" && ev.t > lastT.current) {
              lastT.current = ev.t;
            }
          }
          setEvents((prev) => {
            const combined = [...prev, ...incoming];
            combined.sort((a, b) => b.t - a.t);
            const seen = new Set<string>();
            const deduped: PoolEvent[] = [];
            for (const ev of combined) {
              const key = `${ev.t}:${ev.text}`;
              if (!seen.has(key)) {
                seen.add(key);
                deduped.push(ev);
              }
            }
            return deduped.slice(0, 200);
          });
        }
      } catch (err) {
        if (alive) {
          setError(errorText(err));
        }
      }
    };

    void poll();
    const timer = setInterval(() => void poll(), 2000);
    return () => {
      alive = false;
      clearInterval(timer);
    };
  }, [running]);

  if (!running) {
    return (
      <div className="mx-auto max-w-4xl space-y-4">
        <PageTitle title="Activity">What has happened in the swarm lately.</PageTitle>
        <Card>
          <Empty title="Not in a swarm">
            <p className="mb-3">Start or join a swarm on Home to see activity.</p>
            <Button kind="primary" onClick={() => go("home")}>
              Go to Home
            </Button>
          </Empty>
        </Card>
      </div>
    );
  }

  return (
    <div className="mx-auto max-w-4xl space-y-4">
      <PageTitle title="Activity">What has happened in the swarm lately.</PageTitle>

      {error && <ErrorNote>{error}</ErrorNote>}

      <Card>
        {events.length === 0 ? (
          <Empty title="Nothing has happened yet.">
            Devices joining, leaving or taking over a stage will show up here.
          </Empty>
        ) : (
          <div className="space-y-1">
            {events.map((e, idx) => (
              <div
                key={`${e.t}-${e.text}-${idx}`}
                className="flex items-center gap-3 py-2 border-b text-sm last:border-b-0"
                style={{ borderColor: "var(--gridline)" }}
              >
                <span
                  className="w-16 shrink-0 text-xs tabular-nums"
                  style={{ color: "var(--text-muted)" }}
                >
                  {ago(e.t)}
                </span>
                <span className="shrink-0 w-20">
                  <Pill tone={eventTone(e.kind)}>{e.kind}</Pill>
                </span>
                <span
                  className="min-w-0 flex-1 break-words"
                  style={{ color: "var(--text-primary)" }}
                >
                  {e.text}
                </span>
              </div>
            ))}
          </div>
        )}
      </Card>
    </div>
  );
}
