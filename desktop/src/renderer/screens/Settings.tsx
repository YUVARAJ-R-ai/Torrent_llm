/**
 * Settings: configure device parameters, ports, network bootstrap nodes, and theme.
 * Loads values from window.tllm.settings.get() and saves changes via settings.set().
 */

import { useEffect, useState } from "react";

import type { ScreenProps } from "../App";
import type { AppSettings } from "../../shared/types";
import { Button, Card, ErrorNote, PageTitle } from "../components/ui";
import { errorText } from "../lib/hooks";

// Finding 1: BOOTSTRAP_PATTERN matching /^tcp:\/\/[^\s]+:(\d{1,5})$/ to allow IPv6 hosts
const BOOTSTRAP_PATTERN = /^tcp:\/\/[^\s]+:(\d{1,5})$/;

export default function SettingsScreen({ status }: ScreenProps) {
  const [loaded, setLoaded] = useState(false);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [label, setLabel] = useState("");
  const [memoryMb, setMemoryMb] = useState("0");
  const [controlPort, setControlPort] = useState("9000");
  const [listenPort, setListenPort] = useState("9100");
  const [bootstrapText, setBootstrapText] = useState("");
  const [upnp, setUpnp] = useState(true);
  const [relay, setRelay] = useState(false);
  const [theme, setTheme] = useState<"system" | "light" | "dark">("system");
  const [executable, setExecutable] = useState("");

  // Validation errors
  const [controlPortError, setControlPortError] = useState<string | null>(null);
  const [peerPortError, setPeerPortError] = useState<string | null>(null);
  const [bootstrapError, setBootstrapError] = useState<string | null>(null);
  const [memoryError, setMemoryError] = useState<string | null>(null);

  // Async state
  const [busy, setBusy] = useState(false);
  const [saveError, setSaveError] = useState<string | null>(null);
  const [saved, setSaved] = useState(false);

  useEffect(() => {
    let alive = true;
    void window.tllm.settings.get().then((s) => {
      if (!alive) return;
      setLabel(s.label ?? "");
      setMemoryMb(String(s.memoryMb ?? 0));
      setControlPort(String(s.controlPort ?? 9000));
      setListenPort(String(s.listenPort ?? 9100));
      setBootstrapText((s.bootstrap ?? []).join("\n"));
      setUpnp(s.upnp ?? true);
      setRelay(s.relay ?? false);
      setTheme(s.theme ?? "system");
      setExecutable(s.executable ?? "");
      setLoaded(true);
    }).catch((e) => {
      if (alive) setLoadError(errorText(e));
    });
    return () => {
      alive = false;
    };
  }, []);

  function handleThemeChange(newTheme: "system" | "light" | "dark") {
    setTheme(newTheme);
    setSaved(false);
    if (newTheme === "system") {
      document.documentElement.removeAttribute("data-theme");
    } else {
      document.documentElement.setAttribute("data-theme", newTheme);
    }
  }

  async function handleSave(e?: React.FormEvent) {
    if (e) e.preventDefault();
    setSaved(false);
    setSaveError(null);

    // Validate memory
    const mem = parseInt(memoryMb, 10);
    if (Number.isNaN(mem) || mem < 0) {
      setMemoryError("Memory must be 0 or a positive number.");
      return;
    }
    setMemoryError(null);

    // Validate control port
    const cp = parseInt(controlPort, 10);
    if (Number.isNaN(cp) || cp < 1024 || cp > 65535) {
      setControlPortError("Control panel port must be an integer between 1024 and 65535.");
      return;
    }
    setControlPortError(null);

    // Validate peer port
    const lp = parseInt(listenPort, 10);
    if (Number.isNaN(lp) || lp < 1024 || lp > 65535) {
      setPeerPortError("Peer port must be an integer between 1024 and 65535.");
      return;
    }

    // Finding 4: Control port and peer port may not be equal
    if (cp === lp) {
      setPeerPortError("Peer port must differ from the control panel port.");
      return;
    }
    setPeerPortError(null);

    // Validate bootstrap lines
    const lines = bootstrapText
      .split("\n")
      .map((l) => l.trim())
      .filter((l) => l.length > 0);

    for (const line of lines) {
      const match = line.match(BOOTSTRAP_PATTERN);
      if (!match) {
        setBootstrapError(`Invalid bootstrap address "${line}". Must match tcp://host:port`);
        return;
      }
      const port = parseInt(match[1], 10);
      if (port < 1 || port > 65535) {
        setBootstrapError(`Port out of range in "${line}". Must be 1-65535.`);
        return;
      }
    }
    setBootstrapError(null);

    setBusy(true);
    try {
      const patch: Partial<AppSettings> = {
        label: label.trim(),
        memoryMb: mem,
        controlPort: cp,
        listenPort: lp,
        bootstrap: lines,
        upnp,
        relay,
        theme,
        executable: executable.trim(),
      };
      await window.tllm.settings.set(patch);
      setSaved(true);
    } catch (err) {
      setSaveError(errorText(err));
    } finally {
      setBusy(false);
    }
  }

  if (!loaded) {
    return (
      <div className="mx-auto max-w-4xl space-y-4">
        <PageTitle title="Settings">Adjust device configuration and network settings.</PageTitle>
        <Card>
          {loadError ? (
            <ErrorNote>Could not read the settings: {loadError}</ErrorNote>
          ) : (
            <div className="p-4 text-sm" style={{ color: "var(--text-muted)" }}>
              Loading settings…
            </div>
          )}
        </Card>
      </div>
    );
  }

  return (
    <div className="mx-auto max-w-4xl space-y-4">
      <PageTitle title="Settings">Adjust device configuration and network settings.</PageTitle>

      {status.phase === "running" && (
        <p className="text-xs" style={{ color: "var(--text-muted)" }}>
          Changes apply the next time the device starts.
        </p>
      )}

      {saveError && <ErrorNote>{saveError}</ErrorNote>}

      {/* Finding 5: Form with onSubmit, noValidate for custom error display, and Save Button with type="submit" */}
      <form onSubmit={handleSave} noValidate className="space-y-4">
        <Card title="Device identification">
          <div className="space-y-3">
            <div>
              <label htmlFor="device-name" className="mb-1 block text-xs font-medium" style={{ color: "var(--text-secondary)" }}>
                Device name
              </label>
              <input
                id="device-name"
                type="text"
                maxLength={64}
                value={label}
                onChange={(e) => {
                  setLabel(e.target.value);
                  setSaved(false);
                }}
                className="w-full rounded-md border px-3 py-1.5 text-sm"
                style={{
                  background: "var(--surface-2)",
                  borderColor: "var(--border)",
                  color: "var(--text-primary)",
                }}
              />
            </div>

            <div>
              <label htmlFor="memory-mb" className="mb-1 block text-xs font-medium" style={{ color: "var(--text-secondary)" }}>
                Memory to lend in MB
              </label>
              <input
                id="memory-mb"
                type="number"
                min={0}
                value={memoryMb}
                onChange={(e) => {
                  setMemoryMb(e.target.value);
                  setSaved(false);
                }}
                className="w-full rounded-md border px-3 py-1.5 text-sm"
                style={{
                  background: "var(--surface-2)",
                  borderColor: "var(--border)",
                  color: "var(--text-primary)",
                }}
              />
              <p className="mt-1 text-xs" style={{ color: "var(--text-muted)" }}>
                0 = let the device decide
              </p>
              {memoryError && (
                <div className="mt-1">
                  <ErrorNote>{memoryError}</ErrorNote>
                </div>
              )}
            </div>
          </div>
        </Card>

        <Card title="Networking & Ports">
          <div className="space-y-3">
            <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
              <div>
                <label htmlFor="control-port" className="mb-1 block text-xs font-medium" style={{ color: "var(--text-secondary)" }}>
                  Control panel port
                </label>
                <input
                  id="control-port"
                  type="number"
                  min={1024}
                  max={65535}
                  value={controlPort}
                  onChange={(e) => {
                    setControlPort(e.target.value);
                    setControlPortError(null);
                    setSaved(false);
                  }}
                  className="w-full rounded-md border px-3 py-1.5 text-sm"
                  style={{
                    background: "var(--surface-2)",
                    borderColor: "var(--border)",
                    color: "var(--text-primary)",
                  }}
                />
                {controlPortError && (
                  <div className="mt-1">
                    <ErrorNote>{controlPortError}</ErrorNote>
                  </div>
                )}
              </div>

              <div>
                <label htmlFor="peer-port" className="mb-1 block text-xs font-medium" style={{ color: "var(--text-secondary)" }}>
                  Peer port
                </label>
                <input
                  id="peer-port"
                  type="number"
                  min={1024}
                  max={65535}
                  value={listenPort}
                  onChange={(e) => {
                    setListenPort(e.target.value);
                    setPeerPortError(null);
                    setSaved(false);
                  }}
                  className="w-full rounded-md border px-3 py-1.5 text-sm"
                  style={{
                    background: "var(--surface-2)",
                    borderColor: "var(--border)",
                    color: "var(--text-primary)",
                  }}
                />
                {peerPortError && (
                  <div className="mt-1">
                    <ErrorNote>{peerPortError}</ErrorNote>
                  </div>
                )}
              </div>
            </div>

            <div>
              <label htmlFor="bootstrap-addresses" className="mb-1 block text-xs font-medium" style={{ color: "var(--text-secondary)" }}>
                Extra bootstrap addresses
              </label>
              <textarea
                id="bootstrap-addresses"
                rows={3}
                value={bootstrapText}
                onChange={(e) => {
                  setBootstrapText(e.target.value);
                  setBootstrapError(null);
                  setSaved(false);
                }}
                placeholder="tcp://203.0.113.7:9100"
                className="mono w-full rounded-md border px-3 py-1.5 text-sm"
                style={{
                  background: "var(--surface-2)",
                  borderColor: "var(--border)",
                  color: "var(--text-primary)",
                }}
              />
              <p className="mt-1 text-xs" style={{ color: "var(--text-muted)" }}>
                One per line, each matching tcp://&lt;host&gt;:&lt;port&gt;
              </p>
              {bootstrapError && (
                <div className="mt-1">
                  <ErrorNote>{bootstrapError}</ErrorNote>
                </div>
              )}
            </div>

            <div className="space-y-2 pt-1">
              <label className="flex items-center gap-2 text-sm" style={{ color: "var(--text-primary)" }}>
                <input
                  type="checkbox"
                  checked={upnp}
                  onChange={(e) => {
                    setUpnp(e.target.checked);
                    setSaved(false);
                  }}
                  className="rounded border"
                  style={{ borderColor: "var(--border)" }}
                />
                <span>Ask the router to forward the port (UPnP)</span>
              </label>

              <label className="flex items-center gap-2 text-sm" style={{ color: "var(--text-primary)" }}>
                <input
                  type="checkbox"
                  checked={relay}
                  onChange={(e) => {
                    setRelay(e.target.checked);
                    setSaved(false);
                  }}
                  className="rounded border"
                  style={{ borderColor: "var(--border)" }}
                />
                <span>Relay for devices that cannot be reached</span>
              </label>
            </div>
          </div>
        </Card>

        <Card title="Appearance & Paths">
          <div className="space-y-3">
            <div>
              <label htmlFor="theme-select" className="mb-1 block text-xs font-medium" style={{ color: "var(--text-secondary)" }}>
                Theme
              </label>
              <select
                id="theme-select"
                value={theme}
                onChange={(e) => handleThemeChange(e.target.value as "system" | "light" | "dark")}
                className="rounded-md border px-3 py-1.5 text-sm"
                style={{
                  background: "var(--surface-2)",
                  borderColor: "var(--border)",
                  color: "var(--text-primary)",
                }}
              >
                <option value="system">System</option>
                <option value="light">Light</option>
                <option value="dark">Dark</option>
              </select>
            </div>

            <div>
              <label htmlFor="executable-path" className="mb-1 block text-xs font-medium" style={{ color: "var(--text-secondary)" }}>
                torrent-peer path
              </label>
              <input
                id="executable-path"
                type="text"
                placeholder="Optional; leave blank to find automatically"
                value={executable}
                onChange={(e) => {
                  setExecutable(e.target.value);
                  setSaved(false);
                }}
                className="w-full rounded-md border px-3 py-1.5 text-sm"
                style={{
                  background: "var(--surface-2)",
                  borderColor: "var(--border)",
                  color: "var(--text-primary)",
                }}
              />
            </div>
          </div>
        </Card>

        <div className="flex items-center gap-3">
          <Button type="submit" kind="primary" disabled={busy}>
            Save
          </Button>
          {saved && (
            <span className="text-sm font-medium" style={{ color: "var(--status-good)" }}>
              Saved.
            </span>
          )}
        </div>
      </form>
    </div>
  );
}
