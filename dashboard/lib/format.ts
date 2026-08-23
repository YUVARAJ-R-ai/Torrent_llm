/**
 * Number formatting for the dashboard.
 *
 * The payload sizes here span roughly six orders of magnitude — 8 bytes for a
 * decode-step token id, tens of megabytes for a long-context prefill — so a
 * single fixed unit is unreadable at one end or the other. Everything below
 * picks a unit per value rather than per column.
 */

/** Bytes with a binary unit chosen per value: 8 B, 4.0 KiB, 16.0 MiB. */
export function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KiB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MiB`;
}

/** Milliseconds, dropping decimals once the value is large enough not to need them. */
export function formatMs(ms: number): string {
  if (ms < 10) return `${ms.toFixed(1)} ms`;
  if (ms < 1000) return `${Math.round(ms)} ms`;
  return `${(ms / 1000).toFixed(2)} s`;
}

/** A ratio as a multiplier: 5.67×. */
export function formatRatio(ratio: number): string {
  return `${ratio.toFixed(2)}×`;
}

/** A 0–1 fraction as a percentage. */
export function formatPercent(fraction: number): string {
  return `${Math.round(fraction * 100)}%`;
}

/**
 * Short label for a hop, e.g. "hop 0".
 *
 * Hops are identified by index rather than address in chart labels: the address
 * is a host:port that says nothing about position in the chain, and the chain
 * order is the thing being read off the axis.
 */
export function hopLabel(hop: number): string {
  return `hop ${hop}`;
}
