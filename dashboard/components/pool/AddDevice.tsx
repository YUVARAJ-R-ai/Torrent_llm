"use client";

import { useSyncExternalStore } from "react";

const subscribe = () => () => undefined;

/**
 * How to add a device, shown when this page is served by a device in a real pool.
 *
 * The address to join is read from the browser's own address bar, since that is
 * exactly how this device was reached. It is read with useSyncExternalStore so the
 * static page and the live one agree while hydrating, instead of an effect that
 * would flash the wrong text first.
 */
export default function AddDevice() {
  const host = useSyncExternalStore(
    subscribe,
    () => window.location.host,
    () => "",
  );
  const code = { color: "var(--text-primary)" };
  return (
    <div
      className="rounded-xl px-4 py-3 text-xs leading-relaxed"
      style={{
        background: "var(--page-plane)",
        border: "1px dashed var(--border)",
        color: "var(--text-secondary)",
      }}
    >
      <div
        className="mb-1 text-sm font-medium"
        style={{ color: "var(--text-primary)" }}
      >
        Add another device
      </div>
      On another laptop on this network, run{" "}
      <code style={code}>scripts/join.sh</code> (Windows:{" "}
      <code style={code}>scripts\join.ps1</code>). It finds this pool by itself.
      {host && (
        <>
          {" "}
          If the network hides devices from each other, point it here:{" "}
          <code style={code}>scripts/join.sh --join {host}</code>
        </>
      )}
    </div>
  );
}
