import { ArrowRight, TriangleAlert } from "lucide-react";

import { formatMs } from "@/lib/format";
import { stageColor } from "@/components/pool/Cards";
import type { PoolGenerateResult } from "@/lib/pool";

interface Leg {
  route: [number, string][];
  passes: number;
  /** Tokens re-sent to rebuild the cache, when this leg began with a recovery. */
  resent: number | null;
  lostBefore: { member: string; stage: number } | null;
}

/** Group consecutive passes that took the same route into legs. */
function legsOf(result: PoolGenerateResult): Leg[] {
  const legs: Leg[] = [];
  result.passes.forEach((pass, i) => {
    const key = pass.route.map(([, m]) => m).join(">");
    const last = legs[legs.length - 1];
    if (
      last &&
      last.route.map(([, m]) => m).join(">") === key &&
      !pass.recovery
    ) {
      last.passes += 1;
      return;
    }
    const failure = result.events.find(
      (e) => e.kind === "failover" && e.at_pass === i,
    );
    legs.push({
      route: pass.route,
      passes: 1,
      resent: pass.recovery ? pass.seq_len : null,
      lostBefore:
        failure && failure.member
          ? { member: failure.member, stage: failure.stage ?? 0 }
          : null,
    });
  });
  return legs;
}

/**
 * The route an answer actually took, in words and chips.
 *
 * The animation shows this happening; this says it afterwards and stays on screen,
 * which is what a reader wants once the motion has stopped. A failover shows as a
 * break in the route with what was lost and what it cost.
 */
export default function Journey({
  result,
  nameOf,
}: {
  result: PoolGenerateResult;
  nameOf: (member: string) => string;
}) {
  const legs = legsOf(result);
  const t = result.totals;
  return (
    <section
      className="rounded-2xl px-4 py-3.5"
      style={{
        background: "var(--surface-1)",
        border: "1px solid var(--border)",
      }}
    >
      <div className="mb-2.5 flex items-baseline justify-between gap-3">
        <h2 className="text-xs font-medium">How that answer travelled</h2>
        <span
          className="tabular text-[11px]"
          style={{ color: "var(--text-muted)" }}
        >
          {result.passes.length} steps ·{" "}
          {formatMs(t.compute_ms + t.transport_ms)} of work
        </span>
      </div>
      <ol className="flex flex-col gap-2">
        {legs.map((leg, i) => (
          <li key={i} className="flex flex-col gap-1.5">
            {leg.lostBefore && (
              <div
                className="flex items-start gap-2 rounded-lg px-2.5 py-1.5 text-xs"
                style={{
                  background: "var(--page-plane)",
                  border: "1px solid var(--status-critical)",
                  color: "var(--text-primary)",
                }}
              >
                <TriangleAlert
                  size={13}
                  className="mt-0.5 shrink-0"
                  style={{ color: "var(--status-critical)" }}
                />
                <span>
                  {nameOf(leg.lostBefore.member)} stopped answering.
                  {leg.resent !== null && (
                    <>
                      {" "}
                      The text so far ({leg.resent} tokens) was sent again to
                      the new route to rebuild its memory of the conversation,
                      then the answer carried on.
                    </>
                  )}
                </span>
              </div>
            )}
            <div className="flex flex-wrap items-center gap-1.5">
              {leg.route.map(([stage, member], k) => (
                <span key={k} className="flex items-center gap-1.5">
                  {k > 0 && (
                    <ArrowRight
                      size={12}
                      style={{ color: "var(--text-muted)" }}
                      aria-hidden
                    />
                  )}
                  <span
                    className="flex items-center gap-1.5 rounded-md px-2 py-1 text-xs"
                    style={{
                      background: "var(--page-plane)",
                      border: "1px solid var(--border)",
                    }}
                  >
                    <span
                      className="size-1.5 rounded-full"
                      style={{ background: stageColor(stage) }}
                    />
                    {nameOf(member)}
                    <span style={{ color: "var(--text-muted)" }}>
                      stage {stage + 1}
                    </span>
                  </span>
                </span>
              ))}
              <span
                className="tabular text-[11px]"
                style={{ color: "var(--text-muted)" }}
              >
                {leg.passes} {leg.passes === 1 ? "step" : "steps"}
              </span>
            </div>
          </li>
        ))}
      </ol>
    </section>
  );
}
