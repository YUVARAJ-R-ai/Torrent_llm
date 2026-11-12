/** Small formatting helpers shared by the screens. Pure; tested in format.test.ts. */

export function bytes(n: number | null | undefined): string {
  if (n == null || !Number.isFinite(n)) return "—";
  const units = ["B", "KB", "MB", "GB", "TB"];
  let value = Math.abs(n);
  let unit = 0;
  while (value >= 1024 && unit < units.length - 1) {
    value /= 1024;
    unit += 1;
  }
  const text = value >= 100 || unit === 0 ? value.toFixed(0) : value.toFixed(1);
  return `${n < 0 ? "-" : ""}${text} ${units[unit]}`;
}

export function ms(n: number | null | undefined): string {
  if (n == null || !Number.isFinite(n)) return "—";
  if (n < 1000) return `${Math.round(n)} ms`;
  return `${(n / 1000).toFixed(n < 10_000 ? 2 : 1)} s`;
}

export function bandwidth(bps: number | null | undefined): string {
  if (bps == null || !Number.isFinite(bps)) return "—";
  const bits = bps * 8;
  if (bits >= 1e9) return `${(bits / 1e9).toFixed(1)} Gbit/s`;
  if (bits >= 1e6) return `${(bits / 1e6).toFixed(1)} Mbit/s`;
  return `${Math.max(0, Math.round(bits / 1e3))} kbit/s`;
}

export function shortId(peerId: string | null | undefined): string {
  return peerId ? peerId.slice(0, 8) : "—";
}

export function ago(unixSeconds: number, now: number = Date.now() / 1000): string {
  const s = Math.max(0, Math.round(now - unixSeconds));
  if (s < 60) return `${s}s ago`;
  if (s < 3600) return `${Math.floor(s / 60)}m ago`;
  return `${Math.floor(s / 3600)}h ago`;
}

export function ratio(uncompressed: number, sent: number): string {
  if (!sent || !uncompressed) return "—";
  return `${(uncompressed / sent).toFixed(1)}x`;
}
