/**
 * Turns one measured pool run into a sequence of animation steps.
 *
 * Like the two-device demo, this replays a run that already happened: every step
 * comes from the pass and hop records the pool returned, in the order they
 * happened, and only the pacing is slowed so it can be followed.
 *
 * What is new is that the route can change in the middle. When a device was lost
 * the run's `events` say which one and when, and the passes after that carry a
 * different route. The timeline shows the failed attempt, the lost device going
 * dark, and then the same answer carrying on through the new route.
 */

import type { PassInfo, PoolGenerateResult } from "@/lib/pool";
import { formatBytes } from "@/lib/format";

/** A place on the stage that a payload can travel between. */
export type Place =
  | { type: "you" }
  | { type: "answer" }
  | { type: "copy"; member: string; stage: number };

export const placeKey = (place: Place): string =>
  place.type === "copy" ? `${place.member}#${place.stage}` : place.type;

export type Step =
  | {
      kind: "send";
      from: Place;
      to: Place;
      label: string;
      seconds: number;
      bytes: number;
      networkMs: number;
      betweenDevices: boolean;
      failed?: boolean;
    }
  | { kind: "compute"; at: Place; seconds: number; computeMs: number }
  | { kind: "token"; text: string; seconds: number }
  | { kind: "lost"; member: string; stage: number; seconds: number }
  | { kind: "waiting"; stage: number; seconds: number }
  | { kind: "recovered"; route: string; seconds: number };

const PREFILL_SLOWDOWN = 4;
const DECODE_SLOWDOWN = 1.5;
const PREFILL_FLOOR = { send: 0.7, compute: 0.45 };
const DECODE_FLOOR = { send: 0.22, compute: 0.14 };
/** Most seconds all the decode steps may take together; long answers speed up. */
const DECODE_BUDGET_S = 8;

function pace(
  realMs: number,
  prefill: boolean,
  kind: "send" | "compute",
): number {
  const floor = prefill ? PREFILL_FLOOR[kind] : DECODE_FLOOR[kind];
  const slowdown = prefill ? PREFILL_SLOWDOWN : DECODE_SLOWDOWN;
  return Math.max(floor, (realMs / 1000) * slowdown);
}

const copy = (member: string, stage: number): Place => ({
  type: "copy",
  member,
  stage,
});

export function buildTimeline(result: PoolGenerateResult): Step[] {
  const steps: Step[] = [];
  const stagesOf = (pass: PassInfo) =>
    pass.route.map(([stage, member]) => copy(member, stage));

  result.passes.forEach((pass, i) => {
    // Things that happened while this pass was being attempted.
    for (const event of result.events.filter((e) => e.at_pass === i)) {
      if (event.kind === "waiting") {
        steps.push({ kind: "waiting", stage: event.stage ?? 0, seconds: 1.2 });
      }
      if (event.kind === "failover" && event.member !== null) {
        const stage = event.stage ?? 0;
        // The pass that was running when the device went. Its route is the one
        // before this pass, which is the one that failed.
        const before = i > 0 ? stagesOf(result.passes[i - 1]) : [];
        const previous: Place =
          stage > 0 && before[stage - 1] ? before[stage - 1] : { type: "you" };
        steps.push({
          kind: "send",
          from: previous,
          to: copy(event.member, stage),
          label: "no answer",
          seconds: 0.9,
          bytes: 0,
          networkMs: 0,
          betweenDevices: false,
          failed: true,
        });
        steps.push({ kind: "lost", member: event.member, stage, seconds: 1.1 });
      }
    }

    const prefill = pass.phase === "prefill";
    const places = stagesOf(pass);

    pass.hops.forEach((hop, k) => {
      const from: Place = k === 0 ? { type: "you" } : places[k - 1];
      const to = places[k];
      const legMs = hop.transport_ms / 2;
      steps.push({
        kind: "send",
        from,
        to,
        label:
          k === 0
            ? `${pass.seq_len} token${pass.seq_len === 1 ? "" : "s"}`
            : formatBytes(hop.sent_bytes),
        seconds: pace(legMs, prefill, "send"),
        bytes: hop.sent_bytes,
        networkMs: legMs,
        betweenDevices: k > 0,
      });
      steps.push({
        kind: "compute",
        at: to,
        seconds: pace(hop.compute_ms, prefill, "compute"),
        computeMs: hop.compute_ms,
      });
    });

    const last = pass.hops[pass.hops.length - 1];
    steps.push({
      kind: "send",
      from: places[places.length - 1],
      to: { type: "answer" },
      label: "1 token",
      seconds: pace(last.transport_ms / 2, prefill, "send"),
      bytes: last.received_bytes,
      networkMs: last.transport_ms / 2,
      betweenDevices: false,
    });

    // Every pass that completes yields one token, a recovery pass included: it
    // re-sends the text so far and then produces the next token like any other.
    // The pass that ends the answer picks a stop token, which has no text.
    const text = result.tokens[i];
    if (text !== undefined) {
      steps.push({ kind: "token", text, seconds: prefill ? 0.3 : 0.05 });
    }

    for (const event of result.events.filter((e) => e.at_pass === i + 1)) {
      if (event.kind === "recovered") {
        steps.push({ kind: "recovered", route: event.detail, seconds: 0.8 });
      }
    }
  });

  // Decode passes share a time budget, so a long answer never drags.
  const firstToken = steps.findIndex((s) => s.kind === "token");
  const decode = firstToken < 0 ? [] : steps.slice(firstToken + 1);
  const total = decode.reduce((sum, s) => sum + s.seconds, 0);
  if (total > DECODE_BUDGET_S) {
    const scale = DECODE_BUDGET_S / total;
    for (const step of decode) {
      // Never squash the failover moments: they are what the viewer came to see.
      if (
        step.kind === "lost" ||
        step.kind === "recovered" ||
        (step.kind === "send" && step.failed)
      ) {
        continue;
      }
      step.seconds *= scale;
    }
  }
  return steps;
}
