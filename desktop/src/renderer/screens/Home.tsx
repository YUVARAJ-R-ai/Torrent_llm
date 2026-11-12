/**
 * Home: is this device in a swarm, and is the swarm healthy? The one place to start,
 * stop, or install. Reference screen: the other screens follow its patterns
 * (usePoll for data, Card/Stat/Pill for layout, errors shown inline, never blank).
 */

import { Download, Play, Square } from "lucide-react";
import { useState } from "react";

import type { ScreenProps } from "../App";
import type { NodeView, PoolView } from "../../shared/types";
import { Button, Card, Empty, ErrorNote, PageTitle, Pill, Stat, type Tone } from "../components/ui";
import { errorText, usePoll } from "../lib/hooks";

const PHASE: Record<string, { tone: Tone; text: string }> = {
  missing: { tone: "neutral", text: "Not installed" },
  installing: { tone: "warn", text: "Installing" },
  stopped: { tone: "neutral", text: "Stopped" },
  starting: { tone: "warn", text: "Starting" },
  running: { tone: "good", text: "Running" },
  stopping: { tone: "warn", text: "Stopping" },
  error: { tone: "critical", text: "Problem" },
};

const HEALTH: Record<string, { tone: Tone; text: string }> = {
  healthy: { tone: "good", text: "Healthy: every stage has its spare copies" },
  degraded: { tone: "warn", text: "Degraded: some stage has only one copy" },
  down: { tone: "critical", text: "Down: some stage has no copy yet" },
};

export default function HomeScreen({ status, go }: ScreenProps) {
  const running = status.phase === "running";
  const pool = usePoll<PoolView>("/pool", { enabled: running });
  const node = usePoll<NodeView>("/node", { enabled: running, intervalMs: 5000 });
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function act(action: () => Promise<unknown>) {
    setBusy(true);
    setError(null);
    try {
      await action();
    } catch (e) {
      setError(errorText(e));
    } finally {
      setBusy(false);
    }
  }

  const phase = PHASE[status.phase] ?? PHASE.error;
  const online = pool.data?.members.filter((m) => m.liveness === "alive").length ?? 0;
  const ready = pool.data?.ready_copies ?? [];
  const health = pool.data ? HEALTH[pool.data.health] : null;

  return (
    <div className="mx-auto max-w-4xl space-y-4">
      <PageTitle title="Home">Lend this computer to a swarm and use the model you run together.</PageTitle>

      <Card
        title="This device"
        actions={
          status.phase === "missing" ? (
            <Button kind="primary" disabled={busy} onClick={() => act(() => window.tllm.backend.install())}>
              <Download size={14} aria-hidden /> Install
            </Button>
          ) : running || status.phase === "starting" ? (
            <Button kind="danger" disabled={busy} onClick={() => act(() => window.tllm.backend.stop())}>
              <Square size={14} aria-hidden /> Leave swarm
            </Button>
          ) : (
            <Button kind="primary" disabled={busy || status.phase === "installing"} onClick={() => go("swarm")}>
              <Play size={14} aria-hidden /> Join or create a swarm
            </Button>
          )
        }
      >
        <div className="flex flex-wrap items-center gap-x-6 gap-y-2">
          <Pill tone={phase.tone}>{phase.text}</Pill>
          {node.data && (
            <span className="text-sm" style={{ color: "var(--text-secondary)" }}>
              {node.data.public ? "Public swarm" : `Swarm “${node.data.swarm_name}”`} · {node.data.label}
            </span>
          )}
        </div>
        {status.message && status.phase !== "running" && (
          <p className="mt-2 text-sm" style={{ color: "var(--text-secondary)" }}>
            {status.message}
          </p>
        )}
        {error && <div className="mt-2"><ErrorNote>{error}</ErrorNote></div>}
      </Card>

      {running ? (
        <>
          <Card title="The swarm">
            {pool.error && !pool.data && <ErrorNote>{pool.error}</ErrorNote>}
            {pool.data && (
              <>
                <div className="grid grid-cols-2 gap-4 sm:grid-cols-4">
                  <Stat label="Devices online" value={online} />
                  <Stat label="Stages" value={pool.data.pool.stages.length} hint={pool.data.pool.model_id} />
                  <Stat
                    label="Copies ready"
                    value={ready.length ? Math.min(...ready) : 0}
                    hint={`of ${pool.data.pool.replicas} wanted per stage`}
                  />
                  <Stat
                    label="Reachable"
                    value={node.data ? node.data.reachability : "…"}
                    hint={node.data?.relays.length ? `via ${node.data.relays.length} relay(s)` : undefined}
                  />
                </div>
                {health && (
                  <div className="mt-3">
                    <Pill tone={health.tone}>{health.text}</Pill>
                  </div>
                )}
              </>
            )}
          </Card>
          <div className="flex flex-wrap gap-2">
            <Button onClick={() => go("chat")}>Ask the model</Button>
            <Button onClick={() => go("swarm")}>Invite friends</Button>
            <Button onClick={() => go("peers")}>See who is in</Button>
          </div>
        </>
      ) : (
        status.phase !== "starting" && (
          <Card>
            <Empty title="Not in a swarm">
              Create a swarm and send friends the invite link, or join one with a link you were sent.
              Everything stays on your computers; nothing goes through a server.
            </Empty>
          </Card>
        )
      )}
    </div>
  );
}
