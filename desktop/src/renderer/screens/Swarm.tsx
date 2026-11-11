/**
 * Swarm: manage swarm membership.
 * When stopped/error: join an existing swarm, create a new one, or join the public swarm.
 * When running: invite friends (link, QR code, install commands) and view swarm details.
 */

import { Check, Copy, Eye, EyeOff, LogOut, Plus, Share2, Users } from "lucide-react";
import QRCode from "qrcode";
import { useEffect, useRef, useState } from "react";

import type { ScreenProps } from "../App";
import type { InviteView, NodeView } from "../../shared/types";
import { Button, Card, Empty, ErrorNote, PageTitle, Pill, type Tone } from "../components/ui";
import { shortId } from "../lib/format";
import { errorText, usePoll } from "../lib/hooks";

const REACHABILITY: Record<string, { tone: Tone; text: string }> = {
  public: { tone: "good", text: "Reachable from the internet" },
  private: { tone: "warn", text: "Behind a router; reachable through relays" },
  unknown: { tone: "neutral", text: "Checking…" },
};

export default function SwarmScreen({ status, go }: ScreenProps) {
  const running = status.phase === "running";
  const node = usePoll<NodeView>("/node", { enabled: running, intervalMs: 5000 });

  // Creation & Join form state
  const [swarmName, setSwarmName] = useState("My swarm");
  const [inviteLink, setInviteLink] = useState("");
  const [linkError, setLinkError] = useState<string | null>(null);

  // Invite secret state (only fetched on demand)
  const [inviteData, setInviteData] = useState<InviteView | null>(null);
  const [qrUrl, setQrUrl] = useState<string | null>(null);
  const [publicSwarmMsg, setPublicSwarmMsg] = useState<string | null>(null);
  const [copiedKey, setCopiedKey] = useState<string | null>(null);

  // Async action state
  const [busy, setBusy] = useState(false);
  const [actionError, setActionError] = useState<string | null>(null);

  // Timers
  const autoHideTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const copyTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  // The invite is a secret: forget it as soon as the device leaves the swarm, so a later
  // swarm never shows an old link without "Show invite" being pressed again.
  useEffect(() => {
    if (!running) hideInvite();
    // hideInvite only touches refs and setters
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [running]);

  // Clear timers and the invite secret on unmount
  useEffect(() => {
    return () => {
      if (autoHideTimerRef.current) {
        clearTimeout(autoHideTimerRef.current);
        autoHideTimerRef.current = null;
      }
      if (copyTimerRef.current) {
        clearTimeout(copyTimerRef.current);
        copyTimerRef.current = null;
      }
      setInviteData(null);
      setQrUrl(null);
    };
  }, []);

  async function triggerCopy(key: string, text: string) {
    try {
      await window.tllm.system.copy(text);
    } catch (e) {
      setActionError(`Could not copy: ${errorText(e)}`);
      return;
    }
    setCopiedKey(key);
    if (copyTimerRef.current) clearTimeout(copyTimerRef.current);
    copyTimerRef.current = setTimeout(() => {
      setCopiedKey(null);
    }, 2000);
  }

  function hideInvite() {
    if (autoHideTimerRef.current) {
      clearTimeout(autoHideTimerRef.current);
      autoHideTimerRef.current = null;
    }
    setInviteData(null);
    setQrUrl(null);
  }

  async function handleShowInvite() {
    setBusy(true);
    setActionError(null);
    setPublicSwarmMsg(null);
    try {
      const data = await window.tllm.api.get<InviteView>("/swarm/invite");
      setInviteData(data);
      try {
        const url = await QRCode.toDataURL(data.link, { margin: 1, width: 200 });
        setQrUrl(url);
      } catch {
        // QR rendering failure is non-fatal
      }
      if (autoHideTimerRef.current) clearTimeout(autoHideTimerRef.current);
      // Auto-hide secret after 5 minutes
      autoHideTimerRef.current = setTimeout(() => {
        setInviteData(null);
        setQrUrl(null);
      }, 5 * 60 * 1000);
    } catch (e) {
      const msg = errorText(e);
      if (msg.toLowerCase().includes("public")) {
        setPublicSwarmMsg("The public swarm has no invite: anyone can join with torrent-peer public.");
      } else {
        setActionError(msg);
      }
    } finally {
      setBusy(false);
    }
  }

  async function handleCreate() {
    const name = swarmName.trim();
    if (!name) {
      setActionError("Give the swarm a name.");
      return;
    }
    setBusy(true);
    setActionError(null);
    try {
      await window.tllm.backend.start({ mode: "create", swarmName: name });
      go("home");
    } catch (e) {
      setActionError(errorText(e));
    } finally {
      setBusy(false);
    }
  }

  async function handleJoin() {
    const trimmed = inviteLink.trim();
    if (!trimmed.startsWith("tllm://join/") || trimmed.length >= 4096) {
      setLinkError("That is not an invite link. It starts with tllm://join/");
      return;
    }
    setLinkError(null);
    setBusy(true);
    setActionError(null);
    try {
      await window.tllm.backend.start({ mode: "join", invite: trimmed });
      go("home");
    } catch (e) {
      setActionError(errorText(e));
    } finally {
      setBusy(false);
    }
  }

  async function handleJoinPublic() {
    setBusy(true);
    setActionError(null);
    try {
      await window.tllm.backend.start({ mode: "public" });
      go("home");
    } catch (e) {
      setActionError(errorText(e));
    } finally {
      setBusy(false);
    }
  }

  async function handleLeave() {
    setBusy(true);
    setActionError(null);
    try {
      await window.tllm.backend.stop();
    } catch (e) {
      setActionError(errorText(e));
    } finally {
      setBusy(false);
    }
  }

  // Phase missing: install hint only
  if (status.phase === "missing") {
    return (
      <div className="mx-auto max-w-4xl space-y-4">
        <PageTitle title="Swarm">Connect your device to a network of models.</PageTitle>
        <Card>
          <Empty title="Torrent-peer is not installed">
            Install it from the Home screen first.
            <div className="mt-4 flex justify-center">
              <Button kind="primary" onClick={() => go("home")}>
                Go to Home
              </Button>
            </div>
          </Empty>
        </Card>
      </div>
    );
  }

  // Running state
  if (running) {
    // 2: node.data?.public is known from /node poll
    const isPublic = Boolean(node.data?.public || publicSwarmMsg);
    const reachability = node.data
      ? REACHABILITY[node.data.reachability] ?? REACHABILITY.unknown
      : REACHABILITY.unknown;

    return (
      <div className="mx-auto max-w-4xl space-y-4">
        <PageTitle title="Swarm">Manage your swarm membership and invite others.</PageTitle>

        {actionError && <ErrorNote>{actionError}</ErrorNote>}

        <Card title="Invite friends">
          {isPublic ? (
            <p className="text-sm" style={{ color: "var(--text-secondary)" }}>
              The public swarm has no invite: anyone can join with torrent-peer public.
            </p>
          ) : !inviteData ? (
            <div className="space-y-2">
              <p className="text-sm" style={{ color: "var(--text-secondary)" }}>
                Invite links contain the secret key to join your private swarm. Treat them like passwords.
              </p>
              <Button kind="primary" disabled={busy} onClick={handleShowInvite}>
                <Eye size={14} aria-hidden /> Show invite
              </Button>
            </div>
          ) : (
            <div className="space-y-4">
              <div>
                <label htmlFor="invite-textarea" className="mb-1 block text-xs font-medium" style={{ color: "var(--text-secondary)" }}>
                  Invite link (secret)
                </label>
                <textarea
                  id="invite-textarea"
                  readOnly
                  rows={3}
                  value={inviteData.link}
                  className="mono w-full rounded-md border p-2 text-xs"
                  style={{
                    background: "var(--surface-2)",
                    borderColor: "var(--border)",
                    color: "var(--text-primary)",
                  }}
                />
              </div>

              <div className="flex flex-wrap items-center gap-2">
                <Button kind="primary" onClick={() => triggerCopy("link", inviteData.link)}>
                  {copiedKey === "link" ? <Check size={14} aria-hidden /> : <Copy size={14} aria-hidden />}
                  {copiedKey === "link" ? "Copied" : "Copy link"}
                </Button>
                <Button onClick={hideInvite}>
                  <EyeOff size={14} aria-hidden /> Hide
                </Button>
              </div>

              {qrUrl && (
                <div className="pt-2">
                  <div className="mb-1 text-xs font-medium" style={{ color: "var(--text-secondary)" }}>
                    Scan with camera
                  </div>
                  <img
                    src={qrUrl}
                    alt="QR code of the invite link"
                    className="h-[200px] w-[200px] rounded border"
                    style={{ borderColor: "var(--border)" }}
                  />
                </div>
              )}

              <div className="space-y-3 pt-2">
                <div className="text-xs font-semibold" style={{ color: "var(--text-muted)" }}>
                  Install commands
                </div>

                <div className="space-y-2 text-xs">
                  <div>
                    <div className="mb-1 font-medium" style={{ color: "var(--text-secondary)" }}>
                      Linux / macOS
                    </div>
                    <div className="flex items-center gap-2">
                      <code
                        className="mono flex-1 overflow-x-auto rounded border p-1.5"
                        style={{
                          background: "var(--surface-2)",
                          borderColor: "var(--border)",
                          color: "var(--text-primary)",
                        }}
                      >
                        {inviteData.commands.linux_macos}
                      </code>
                      <Button onClick={() => triggerCopy("linux", inviteData.commands.linux_macos)}>
                        {copiedKey === "linux" ? "Copied" : "Copy"}
                      </Button>
                    </div>
                  </div>

                  <div>
                    <div className="mb-1 font-medium" style={{ color: "var(--text-secondary)" }}>
                      Windows
                    </div>
                    <div className="flex items-center gap-2">
                      <code
                        className="mono flex-1 overflow-x-auto rounded border p-1.5"
                        style={{
                          background: "var(--surface-2)",
                          borderColor: "var(--border)",
                          color: "var(--text-primary)",
                        }}
                      >
                        {inviteData.commands.windows}
                      </code>
                      <Button onClick={() => triggerCopy("windows", inviteData.commands.windows)}>
                        {copiedKey === "windows" ? "Copied" : "Copy"}
                      </Button>
                    </div>
                  </div>

                  <div>
                    <div className="mb-1 font-medium" style={{ color: "var(--text-secondary)" }}>
                      Already installed
                    </div>
                    <div className="flex items-center gap-2">
                      <code
                        className="mono flex-1 overflow-x-auto rounded border p-1.5"
                        style={{
                          background: "var(--surface-2)",
                          borderColor: "var(--border)",
                          color: "var(--text-primary)",
                        }}
                      >
                        {inviteData.commands.installed}
                      </code>
                      <Button onClick={() => triggerCopy("installed", inviteData.commands.installed)}>
                        {copiedKey === "installed" ? "Copied" : "Copy"}
                      </Button>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          )}
        </Card>

        <Card
          title="This swarm"
          actions={
            <Button kind="danger" disabled={busy} onClick={handleLeave}>
              <LogOut size={14} aria-hidden /> Leave swarm
            </Button>
          }
        >
          {node.error && !node.data && <ErrorNote>{node.error}</ErrorNote>}
          {node.data && (
            <div className="space-y-4 text-sm">
              <div className="flex flex-wrap items-center justify-between gap-2 border-b pb-3" style={{ borderColor: "var(--border)" }}>
                <div>
                  <div className="font-semibold" style={{ color: "var(--text-primary)" }}>
                    {node.data.public ? "Public swarm" : node.data.swarm_name || "—"}
                  </div>
                  <div className="text-xs" style={{ color: "var(--text-secondary)" }}>
                    {node.data.label} · <span className="mono">{shortId(node.data.peer_id)}</span>
                  </div>
                </div>
                <div className="flex items-center gap-3">
                  <Pill tone={reachability.tone}>{reachability.text}</Pill>
                  <span className="text-xs" style={{ color: "var(--text-muted)" }}>
                    {node.data.relays.length} relay(s)
                  </span>
                </div>
              </div>

              {node.data.serving_relay && (
                <p className="text-xs font-medium" style={{ color: "var(--status-good)" }}>
                  This device relays for others
                </p>
              )}

              <div>
                <div className="mb-1 text-xs font-medium" style={{ color: "var(--text-muted)" }}>
                  Addresses
                </div>
                {node.data.addresses.length > 0 ? (
                  <ul className="mono space-y-0.5 text-xs" style={{ color: "var(--text-secondary)" }}>
                    {node.data.addresses.map((addr, i) => (
                      <li key={i}>{addr}</li>
                    ))}
                  </ul>
                ) : (
                  <div className="text-xs" style={{ color: "var(--text-muted)" }}>
                    No addresses announced
                  </div>
                )}
              </div>
            </div>
          )}
        </Card>
      </div>
    );
  }

  // Busy phases: the device is changing state, so no choices may be pressed.
  if (status.phase === "starting" || status.phase === "installing" || status.phase === "stopping") {
    return (
      <div className="mx-auto max-w-4xl space-y-4">
        <PageTitle title="Swarm">Connect your device to a network of models.</PageTitle>
        <Card>
          <Empty
            title={
              status.phase === "installing"
                ? "Installing Torrent-LLM…"
                : status.phase === "stopping"
                  ? "Leaving the swarm…"
                  : "Starting the device…"
            }
          >
            {status.message || "This takes a moment. The Home screen shows the progress."}
          </Empty>
        </Card>
      </div>
    );
  }

  // Phase stopped or error: three choices as Cards
  return (
    <div className="mx-auto max-w-4xl space-y-4">
      <PageTitle title="Swarm">Join or create a swarm to share model inference.</PageTitle>

      {actionError && <ErrorNote>{actionError}</ErrorNote>}

      <Card title="Create a swarm">
        <p className="mb-3 text-sm" style={{ color: "var(--text-secondary)" }}>
          Start a new private swarm and become the initial coordinator.
        </p>
        <div className="space-y-3">
          <div>
            <label htmlFor="swarm-name-input" className="mb-1 block text-xs font-medium" style={{ color: "var(--text-secondary)" }}>
              Swarm name
            </label>
            <input
              id="swarm-name-input"
              type="text"
              maxLength={80}
              value={swarmName}
              onChange={(e) => setSwarmName(e.target.value)}
              className="w-full rounded-md border px-3 py-1.5 text-sm"
              style={{
                background: "var(--surface-2)",
                borderColor: "var(--border)",
                color: "var(--text-primary)",
              }}
            />
          </div>
          <Button kind="primary" disabled={busy} onClick={handleCreate}>
            <Plus size={14} aria-hidden /> Create and start
          </Button>
        </div>
      </Card>

      <Card title="Join with a link">
        <p className="mb-3 text-sm" style={{ color: "var(--text-secondary)" }}>
          Paste an invitation link received from a swarm administrator.
        </p>
        <div className="space-y-3">
          <div>
            <label htmlFor="invite-link-input" className="mb-1 block text-xs font-medium" style={{ color: "var(--text-secondary)" }}>
              Invite link
            </label>
            <textarea
              id="invite-link-input"
              placeholder="tllm://join/…"
              rows={2}
              value={inviteLink}
              onChange={(e) => {
                setInviteLink(e.target.value);
                if (linkError) setLinkError(null);
              }}
              className="mono w-full rounded-md border px-3 py-1.5 text-sm"
              style={{
                background: "var(--surface-2)",
                borderColor: "var(--border)",
                color: "var(--text-primary)",
              }}
            />
            {linkError && (
              <div className="mt-1">
                <ErrorNote>{linkError}</ErrorNote>
              </div>
            )}
          </div>
          <Button kind="primary" disabled={busy} onClick={handleJoin}>
            <Share2 size={14} aria-hidden /> Join
          </Button>
        </div>
      </Card>

      <Card title="Public swarm">
        <p className="mb-3 text-sm" style={{ color: "var(--text-secondary)" }}>
          Anyone can join; this device checks others' answers.
        </p>
        <Button kind="primary" disabled={busy} onClick={handleJoinPublic}>
          <Users size={14} aria-hidden /> Join the public swarm
        </Button>
      </Card>
    </div>
  );
}
