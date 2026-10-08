/**
 * Turns one measured generation into a sequence of animation steps.
 *
 * The API answers after the whole generation has run, so the demo replays it:
 * every step below comes from a real hop record, in the order it happened.
 * Only the pacing is changed. Real hops on a simulated home link take tens of
 * milliseconds, which is too fast to follow, so each step gets a floor and the
 * first pass (the prefill, where the whole prompt crosses) is slowed further so
 * there is time to read it.
 *
 * Positions on the stage: link 0 joins the prompt to the first device, link k
 * joins device k-1 to device k, and the last link joins the final device to the
 * answer.
 */

import type { GenerateResult, HopMetrics } from "@/lib/api";
import { formatBytes } from "@/lib/format";

export type Step =
  | {
      kind: "send";
      link: number;
      label: string;
      seconds: number;
      bytes: number;
      betweenDevices: boolean;
      networkMs: number;
    }
  | { kind: "compute"; node: number; seconds: number; computeMs: number }
  | { kind: "token"; text: string; seconds: number };

/** How much slower than real time each kind of pass plays. */
const PREFILL_SLOWDOWN = 4;
const DECODE_SLOWDOWN = 1.5;
/** Floors, in seconds, so even a 1 ms step is visible. */
const PREFILL_FLOOR = { send: 0.7, compute: 0.45 };
const DECODE_FLOOR = { send: 0.22, compute: 0.14 };

function pace(
  realMs: number,
  prefill: boolean,
  kind: "send" | "compute",
): number {
  const floor = prefill ? PREFILL_FLOOR[kind] : DECODE_FLOOR[kind];
  const slowdown = prefill ? PREFILL_SLOWDOWN : DECODE_SLOWDOWN;
  return Math.max(floor, (realMs / 1000) * slowdown);
}

/** Split the flat hop list into one group per forward pass through the chain. */
function passes(hops: HopMetrics[], numShards: number): HopMetrics[][] {
  const out: HopMetrics[][] = [];
  for (let i = 0; i < hops.length; i += numShards) {
    out.push(hops.slice(i, i + numShards));
  }
  return out;
}

export function buildTimeline(
  result: GenerateResult,
  numShards: number,
): Step[] {
  const steps: Step[] = [];

  passes(result.hops, numShards).forEach((pass, passIndex) => {
    const prefill = pass[0].phase === "prefill";
    const last = pass[pass.length - 1];

    // Hop 0 carries token ids, so its size is best said as a count.
    const ids = pass[0].seq_len;
    steps.push({
      kind: "send",
      link: 0,
      label: `${ids} token${ids === 1 ? "" : "s"}`,
      seconds: pace(pass[0].transport_ms / 2, prefill, "send"),
      bytes: pass[0].sent_bytes,
      betweenDevices: false,
      networkMs: pass[0].transport_ms / 2,
    });

    pass.forEach((hop, k) => {
      if (k > 0) {
        // What device k-1 produced is what device k receives.
        const networkMs = pass[k - 1].transport_ms / 2 + hop.transport_ms / 2;
        steps.push({
          kind: "send",
          link: k,
          label: formatBytes(hop.sent_bytes),
          seconds: pace(networkMs, prefill, "send"),
          bytes: hop.sent_bytes,
          betweenDevices: true,
          networkMs,
        });
      }
      steps.push({
        kind: "compute",
        node: k,
        seconds: pace(hop.compute_ms, prefill, "compute"),
        computeMs: hop.compute_ms,
      });
    });

    steps.push({
      kind: "send",
      link: pass.length,
      label: formatBytes(last.received_bytes),
      seconds: pace(last.transport_ms / 2, prefill, "send"),
      bytes: last.received_bytes,
      betweenDevices: false,
      networkMs: last.transport_ms / 2,
    });

    const text = result.tokens[passIndex];
    if (text !== undefined) {
      steps.push({ kind: "token", text, seconds: prefill ? 0.3 : 0.05 });
    }
  });

  return steps;
}
