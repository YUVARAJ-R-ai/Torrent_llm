import type { ScreenProps } from "../App";
import type { Member, PoolView, StageStatus } from "../../shared/types";
import { Button, Card, Empty, ErrorNote, PageTitle, Pill, type Tone } from "../components/ui";
import { shortId } from "../lib/format";
import { usePoll } from "../lib/hooks";

const HEALTH: Record<string, { tone: Tone; text: string }> = {
  healthy: { tone: "good", text: "Healthy: every stage has its spare copies" },
  degraded: { tone: "warn", text: "Degraded: some stage has only one copy" },
  down: { tone: "critical", text: "Down: some stage has no copy yet" },
};

export default function ModelScreen({ status, go }: ScreenProps) {
  const running = status.phase === "running";
  const pool = usePoll<PoolView>("/pool", { enabled: running, intervalMs: 2000 });

  if (!running) {
    return (
      <div className="mx-auto max-w-4xl space-y-4">
        <PageTitle title="Model">{pool.data?.pool.model_id || "—"}</PageTitle>
        <Card>
          <Empty title="The device is not running">
            <Button onClick={() => go("home")}>Go to Home</Button>
          </Empty>
        </Card>
      </div>
    );
  }

  const health = pool.data ? HEALTH[pool.data.health] ?? { tone: "neutral", text: pool.data.health } : null;

  return (
    <div className="mx-auto max-w-4xl space-y-4">
      <PageTitle title="Model">{pool.data?.pool.model_id || "—"}</PageTitle>

      {pool.error && (
        <div className="mb-3">
          <ErrorNote>{pool.error}</ErrorNote>
        </div>
      )}

      {pool.data && (
        (() => {
          const poolData = pool.data;
          return (
            <>
              <Card title="Status">
                <div className="flex flex-wrap items-center justify-between gap-3">
                  {health && <Pill tone={health.tone}>{health.text}</Pill>}
                  <span className="text-sm" style={{ color: "var(--text-secondary)" }}>
                    Wire format: {poolData.pool.codec}
                  </span>
                </div>
              </Card>

              <Card title="Stages">
                <div className="divide-y" style={{ borderColor: "var(--gridline)" }}>
                  {poolData.pool.stages.map((stage) => {
                    const stageNum = stage.stage + 1;
                    const stageMembers: { member: Member; status: StageStatus }[] = [];
                    for (const m of poolData.members) {
                      const hosted = m.hosting?.find((h) => h.stage === stage.stage);
                      if (hosted) {
                        stageMembers.push({ member: m, status: hosted.status });
                      }
                    }

                    // Count what the device itself counts (api.py /pool): ready copies on
                    // devices that are alive, so this agrees with the health shown on Home.
                    const readyCount = stageMembers.filter(
                      (sm) => sm.status === "ready" && sm.member.liveness === "alive",
                    ).length;
                    const replicasWanted = poolData.pool.replicas;
                    const pillTone: Tone =
                      readyCount >= replicasWanted ? "good" : readyCount >= 1 ? "warn" : "critical";

                    return (
                      <div key={stage.stage} className="py-3 first:pt-0 last:pb-0">
                        <div className="flex flex-wrap items-center justify-between gap-2">
                          <span className="text-sm font-medium">
                            Stage {stageNum}: layers {stage.start}-{stage.end}
                          </span>
                          <Pill tone={pillTone}>
                            {readyCount} of {replicasWanted} ready
                          </Pill>
                        </div>
                        <div className="mt-2 flex flex-wrap items-center gap-2">
                          {stageMembers.map(({ member, status: memberStatus }) => {
                            const alive = member.liveness === "alive";
                            const isReady = memberStatus === "ready" && alive;
                            const note = !alive
                              ? "offline"
                              : memberStatus === "ready"
                                ? "ready"
                                : memberStatus;
                            return (
                              <span
                                key={member.name}
                                className="inline-flex items-center rounded px-2.5 py-1 text-xs font-medium"
                                style={{
                                  background: isReady ? "var(--series-1)" : "transparent",
                                  border: `1px solid ${isReady ? "transparent" : "var(--border)"}`,
                                  color: isReady ? "var(--accent-text)" : "var(--text-secondary)",
                                }}
                                title={note}
                              >
                                {member.label || shortId(member.name)}
                                {!isReady && <span className="ml-1 opacity-75">({note})</span>}
                                {isReady && <span className="sr-only"> (ready)</span>}
                              </span>
                            );
                          })}
                          {stageMembers.length === 0 && (
                            <span className="text-xs" style={{ color: "var(--text-muted)" }}>
                              No copies hosted yet
                            </span>
                          )}
                        </div>
                      </div>
                    );
                  })}
                </div>
              </Card>
            </>
          );
        })()
      )}
    </div>
  );
}
