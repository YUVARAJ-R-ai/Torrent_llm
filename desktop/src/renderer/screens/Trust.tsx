import { useState } from "react";

import type { ScreenProps } from "../App";
import type { TrustView } from "../../shared/types";
import { Button, Card, Empty, ErrorNote, PageTitle, Pill, Table, Td, type Tone } from "../components/ui";
import { ms, shortId } from "../lib/format";
import { errorText, usePoll } from "../lib/hooks";

export default function TrustScreen({ status, go }: ScreenProps) {
  const running = status.phase === "running";
  const trust = usePoll<TrustView[]>("/trust", { enabled: running, intervalMs: 5000 });
  const [banningPeerId, setBanningPeerId] = useState<string | null>(null);
  const [banReason, setBanReason] = useState("");
  const [banError, setBanError] = useState<string | null>(null);
  const [actionError, setActionError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  if (!running) {
    return (
      <div className="mx-auto max-w-4xl space-y-4">
        <PageTitle title="Trust">
          This device checks a share of other devices' answers. Devices caught returning wrong results are banned here, on this device only.
        </PageTitle>
        <Card>
          <Empty title="The device is not running">
            <Button onClick={() => go("home")}>Go to Home</Button>
          </Empty>
        </Card>
      </div>
    );
  }

  async function handleBan(peerId: string) {
    const trimmed = banReason.trim();
    if (!trimmed || trimmed.length < 1 || trimmed.length > 200) {
      setBanError("Reason is required (1-200 characters).");
      return;
    }
    setBusy(true);
    setActionError(null);
    try {
      await window.tllm.api.post("/trust/ban", { peer_id: peerId, reason: trimmed });
      setBanningPeerId(null);
      setBanReason("");
      setBanError(null);
      trust.refresh();
    } catch (e) {
      setActionError(errorText(e));
    } finally {
      setBusy(false);
    }
  }

  async function handleUnban(peerId: string) {
    setBusy(true);
    setActionError(null);
    try {
      await window.tllm.api.post("/trust/unban", { peer_id: peerId });
      trust.refresh();
    } catch (e) {
      setActionError(errorText(e));
    } finally {
      setBusy(false);
    }
  }

  const rows = trust.data ?? [];

  return (
    <div className="mx-auto max-w-4xl space-y-4">
      <PageTitle title="Trust">
        This device checks a share of other devices' answers. Devices caught returning wrong results are banned here, on this device only.
      </PageTitle>

      <Card>
        {trust.error && (
          <div className="mb-3">
            <ErrorNote>{trust.error}</ErrorNote>
          </div>
        )}
        {actionError && (
          <div className="mb-3">
            <ErrorNote>{actionError}</ErrorNote>
          </div>
        )}
        {trust.data && (
          <>
            {rows.length === 0 ? (
              <Empty title="No checks yet." />
            ) : (
              <Table head={["Device", "Checks", "Failed", "Score", "Gave / Took", "Status", "Action"]}>
                {rows.map((t) => {
                  const statusTone: Tone = t.banned ? "critical" : "good";
                  const statusText = t.banned
                    ? t.reason && t.reason.trim()
                      ? `Banned: ${t.reason.trim()}`
                      : "Banned"
                    : "Trusted";

                  return (
                    <tr key={t.peer_id}>
                      <Td>
                        <div>{t.label || shortId(t.peer_id)}</div>
                        <div className="mono text-xs" style={{ color: "var(--text-muted)" }}>
                          {shortId(t.peer_id)}
                        </div>
                      </Td>
                      <Td right>{t.checks}</Td>
                      <Td right>{t.failures}</Td>
                      <Td right>{`${Math.round(t.score <= 1 ? t.score * 100 : t.score)}%`}</Td>
                      <Td right>{`${ms(t.served_ms)} / ${ms(t.used_ms)}`}</Td>
                      <Td>
                        <Pill tone={statusTone}>{statusText}</Pill>
                      </Td>
                      <Td>
                        {t.banned ? (
                          <Button kind="secondary" disabled={busy} onClick={() => handleUnban(t.peer_id)}>
                            Unban
                          </Button>
                        ) : banningPeerId === t.peer_id ? (
                          <div className="flex flex-col gap-1">
                            <div className="flex items-center gap-2">
                              <input
                                type="text"
                                value={banReason}
                                onChange={(e) => {
                                  setBanReason(e.target.value);
                                  if (banError) setBanError(null);
                                }}
                                onKeyDown={(e) => {
                                  if (e.key === "Enter") {
                                    e.preventDefault();
                                    void handleBan(t.peer_id);
                                  }
                                }}
                                placeholder="Reason (1-200 chars)"
                                maxLength={200}
                                aria-label="Ban reason"
                                className="rounded border px-2 py-1 text-xs"
                                style={{
                                  background: "var(--surface-2)",
                                  borderColor: "var(--border)",
                                  color: "var(--text-primary)",
                                }}
                                autoFocus
                              />
                              <Button kind="danger" disabled={busy} onClick={() => handleBan(t.peer_id)}>
                                Ban
                              </Button>
                              <Button
                                kind="ghost"
                                disabled={busy}
                                onClick={() => {
                                  setBanningPeerId(null);
                                  setBanReason("");
                                  setBanError(null);
                                }}
                              >
                                Cancel
                              </Button>
                            </div>
                            {banError && <ErrorNote>{banError}</ErrorNote>}
                          </div>
                        ) : (
                          <Button
                            kind="danger"
                            disabled={busy}
                            onClick={() => {
                              setBanningPeerId(t.peer_id);
                              setBanReason("");
                              setBanError(null);
                            }}
                          >
                            Ban
                          </Button>
                        )}
                      </Td>
                    </tr>
                  );
                })}
              </Table>
            )}
          </>
        )}
      </Card>
    </div>
  );
}
