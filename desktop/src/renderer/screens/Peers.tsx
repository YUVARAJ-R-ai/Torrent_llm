import type { ScreenProps } from "../App";
import type { ConnectionView, PoolView } from "../../shared/types";
import { Button, Card, Empty, ErrorNote, PageTitle, Pill, Table, Td, type Tone } from "../components/ui";
import { bandwidth, bytes, ms, shortId } from "../lib/format";
import { usePoll } from "../lib/hooks";

export default function PeersScreen({ status, go }: ScreenProps) {
  const running = status.phase === "running";
  const pool = usePoll<PoolView>("/pool", { enabled: running, intervalMs: 2000 });
  const peers = usePoll<ConnectionView[]>("/peers", { enabled: running, intervalMs: 2000 });

  if (!running) {
    return (
      <div className="mx-auto max-w-4xl space-y-4">
        <PageTitle title="Peers">Devices in this swarm, as this device sees them.</PageTitle>
        <Card>
          <Empty title="The device is not running">
            <Button onClick={() => go("home")}>Go to Home</Button>
          </Empty>
        </Card>
      </div>
    );
  }

  const members = pool.data?.members ?? [];
  const otherMembers = members.filter((m) => !m.is_self);
  const connections = peers.data ?? [];

  return (
    <div className="mx-auto max-w-4xl space-y-4">
      <PageTitle title="Peers">Devices in this swarm, as this device sees them.</PageTitle>

      <Card title="Members">
        {pool.error && (
          <div className="mb-3">
            <ErrorNote>{pool.error}</ErrorNote>
          </div>
        )}
        {pool.data && (
          <>
            {members.length > 0 && (
              <Table head={["Device", "Hardware", "Holds", "Status", "Lent", "RTT"]}>
                {members.map((m) => {
                  const tone: Tone =
                    m.liveness === "alive"
                      ? "good"
                      : m.liveness === "suspect"
                        ? "warn"
                        : m.liveness === "down"
                          ? "critical"
                          : "neutral";
                  const statusText =
                    m.liveness === "alive"
                      ? "Online"
                      : m.liveness === "suspect"
                        ? "Not answering"
                        : m.liveness === "down"
                          ? "Gone"
                          : m.liveness;
                  const ownerUsing = m.state === "draining" || m.state === "away";
                  const holdsText =
                    m.hosting && m.hosting.length > 0
                      ? [...m.hosting]
                          .sort((a, b) => a.stage - b.stage)
                          .map((h) => `${h.stage + 1}${h.status === "loading" ? " (loading)" : ""}`)
                          .join(", ")
                      : "—";

                  return (
                    <tr key={m.name}>
                      <Td>
                        <div>
                          {m.label}{" "}
                          {m.is_self && (
                            <span className="text-xs" style={{ color: "var(--text-muted)" }}>
                              (this device)
                            </span>
                          )}
                        </div>
                        <div className="mono text-xs" style={{ color: "var(--text-muted)" }}>
                          {shortId(m.name)}
                        </div>
                      </Td>
                      <Td>{[m.device, m.os].filter(Boolean).join(" · ") || "—"}</Td>
                      <Td>{holdsText}</Td>
                      <Td>
                        <div className="flex flex-wrap items-center gap-1.5">
                          <Pill tone={tone}>{statusText}</Pill>
                          {ownerUsing && (
                            <span className="text-xs" style={{ color: "var(--text-secondary)" }}>
                              · owner using it
                            </span>
                          )}
                        </div>
                      </Td>
                      <Td right>{`${m.memory_mb} MB`}</Td>
                      <Td right>{ms(m.rtt_ms)}</Td>
                    </tr>
                  );
                })}
              </Table>
            )}
            {otherMembers.length === 0 && (
              <div className={members.length > 0 ? "mt-4" : ""}>
                <Empty title="Nobody else yet.">Share the invite from the Swarm screen.</Empty>
              </div>
            )}
          </>
        )}
      </Card>

      <Card title="Connections">
        {peers.error && (
          <div className="mb-3">
            <ErrorNote>{peers.error}</ErrorNote>
          </div>
        )}
        {peers.data && (
          <>
            {connections.length === 0 ? (
              <Empty title="No active connections." />
            ) : (
              <Table head={["Peer", "Direction", "Route", "RTT", "Bandwidth", "Sent / Received"]}>
                {connections.map((c) => (
                  <tr key={c.peer_id}>
                    <Td>
                      <div>{c.label || shortId(c.peer_id)}</div>
                      <div className="mono text-xs" style={{ color: "var(--text-muted)" }}>
                        {shortId(c.peer_id)}
                      </div>
                    </Td>
                    <Td>{c.inbound ? "Inbound" : "Outbound"}</Td>
                    <Td>{c.via_relay ? `via relay ${shortId(c.via_relay)}` : "direct"}</Td>
                    <Td right>{ms(c.rtt_ms)}</Td>
                    <Td right>{bandwidth(c.bandwidth_bps)}</Td>
                    <Td right>{`${bytes(c.bytes_sent)} / ${bytes(c.bytes_received)}`}</Td>
                  </tr>
                ))}
              </Table>
            )}
          </>
        )}
      </Card>
    </div>
  );
}
