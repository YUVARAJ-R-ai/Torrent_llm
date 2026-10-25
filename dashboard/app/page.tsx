"use client";

/**
 * The pool: many devices lending what they have, behaving like one.
 *
 * Read left to right. A question comes in, passes through every stage of the
 * model (a column each), and an answer comes out. Each column holds every device
 * that has that stage loaded, so a question can go through any one of them. When a
 * device disappears, the question goes through another and the answer still
 * arrives. When the pool wants another copy and has a spare device, one slides
 * across from the bench.
 *
 * Everything shown is real: the cards come from the devices' own reports, and the
 * animation replays the pass and hop records the pool returned for the run.
 */

import { AnimatePresence, LayoutGroup, motion } from "motion/react";
import { Play, Sparkles, UserRound, Zap } from "lucide-react";
import { useEffect, useMemo, useRef, useState, type RefObject } from "react";

import AnimatedBeam from "@/components/AnimatedBeam";
import Nav from "@/components/Nav";
import Feed from "@/components/pool/Feed";
import {
  BenchTile,
  CopyCard,
  MissingCopy,
  stageColor,
} from "@/components/pool/Cards";
import { formatBytes, formatMs } from "@/lib/format";
import {
  memberUrl,
  PoolError,
  usePool,
  type LabDevice,
  type PoolGenerateResult,
  type PoolMember,
  type PoolView,
} from "@/lib/pool";
import { buildTimeline, placeKey } from "@/lib/poolReplay";

type Phase = "idle" | "running" | "playing" | "done";

interface BeamState {
  from: string;
  to: string;
  pulse: number;
  label: string;
  duration: number;
  failed: boolean;
}

interface Totals {
  betweenDevicesBytes: number;
  networkMs: number;
  computeMs: number;
  tokens: number;
  lost: number;
}

const ZERO: Totals = {
  betweenDevicesBytes: 0,
  networkMs: 0,
  computeMs: 0,
  tokens: 0,
  lost: 0,
};

/** Long enough to run for several seconds, so a device can be lost during it. */
const CRASH_QUESTION = "Tell me a story about a robot who learns to paint.";

const sleep = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms));

export default function PoolPage() {
  const pool = usePool();
  const { view, lab, events, error } = pool;

  const [prompt, setPrompt] = useState("What is the capital of France?");
  const [phase, setPhase] = useState<Phase>("idle");
  const [result, setResult] = useState<PoolGenerateResult | null>(null);
  const [runError, setRunError] = useState<string | null>(null);
  const [asked, setAsked] = useState("");

  // While a run replays, the stage is frozen as it was when the run began, so a
  // device that died during the run is still there to be shown dying. It
  // thaws when the replay ends and the cards then glide to where the pool
  // actually put things.
  const [frozen, setFrozen] = useState<PoolView | null>(null);
  const [beams, setBeams] = useState<Record<string, BeamState>>({});
  const [busyKey, setBusyKey] = useState<string | null>(null);
  const [lost, setLost] = useState<Set<string>>(new Set());
  const [tokens, setTokens] = useState<string[]>([]);
  const [totals, setTotals] = useState<Totals>(ZERO);
  const [note, setNote] = useState<string | null>(null);
  const runId = useRef(0);
  // The newest view, for timers that outlive a render. Kept in step by an
  // effect: writing a ref while rendering is not allowed.
  const viewRef = useRef<PoolView | null>(null);
  useEffect(() => {
    viewRef.current = view;
  }, [view]);

  const shown = frozen ?? view;
  const labByName = useMemo(
    () => new Map((lab ?? []).map((d) => [d.name, d])),
    [lab],
  );

  // --- where things are on the stage ---------------------------------------

  const stageRef = useRef<HTMLDivElement>(null);
  const [nodes] = useState(() => new NodeRegistry());
  const refFor = (key: string) => nodes.ref(key);
  const register = (key: string) => nodes.callback(key);

  const layout = useMemo(() => {
    if (!shown) return null;
    const copies: {
      member: PoolMember;
      hosted: PoolMember["hosting"][number];
    }[][] = shown.pool.stages.map(() => []);
    const holding = new Set<string>();
    for (const member of shown.members) {
      const usable =
        member.liveness !== "down" &&
        (member.state === "active" || member.state === "draining");
      if (!usable) continue;
      for (const hosted of member.hosting) {
        if (hosted.status === "failed") continue;
        copies[hosted.stage]?.push({ member, hosted });
        holding.add(member.name);
      }
    }
    const bench = shown.members.filter((m) => !holding.has(m.name));
    const known = new Set(shown.members.map((m) => m.name));
    const ghosts = (lab ?? []).filter((d) => !known.has(d.name));
    return { copies, bench, ghosts };
  }, [shown, lab]);

  const remeasure = useMemo(
    () =>
      shown
        ? shown.members
            .map(
              (m) =>
                `${m.name}:${m.hosting.map((h) => h.stage).join("")}:${m.liveness}:${m.state}`,
            )
            .join("|")
        : "",
    [shown],
  );

  // --- replaying a run ---------------------------------------------------------

  async function play(run: PoolGenerateResult, startView: PoolView) {
    const id = ++runId.current;
    setFrozen(startView);
    setBeams({});
    setBusyKey(null);
    setLost(new Set());
    setTokens([]);
    setTotals(ZERO);
    setNote(null);
    setPhase("playing");

    for (const step of buildTimeline(run)) {
      // A newer run cancels this one between steps.
      if (runId.current !== id) return;
      if (step.kind === "send") {
        const from = placeKey(step.from);
        const to = placeKey(step.to);
        setBusyKey(null);
        setBeams((prev) => ({
          ...prev,
          [`${from}>${to}`]: {
            from,
            to,
            pulse: (prev[`${from}>${to}`]?.pulse ?? 0) + 1,
            label: step.label,
            duration: step.seconds,
            failed: !!step.failed,
          },
        }));
        setTotals((t) => ({
          ...t,
          networkMs: t.networkMs + step.networkMs,
          betweenDevicesBytes:
            t.betweenDevicesBytes + (step.betweenDevices ? step.bytes : 0),
        }));
      } else if (step.kind === "compute") {
        setBusyKey(placeKey(step.at));
        setTotals((t) => ({ ...t, computeMs: t.computeMs + step.computeMs }));
      } else if (step.kind === "token") {
        setBusyKey(null);
        setTokens((prev) => [...prev, step.text]);
        setTotals((t) => ({ ...t, tokens: t.tokens + 1 }));
      } else if (step.kind === "lost") {
        setBusyKey(null);
        setLost((prev) => new Set(prev).add(step.member));
        setTotals((t) => ({ ...t, lost: t.lost + 1 }));
        setNote(
          `${nameOfIn(startView, step.member)} stopped answering. Finding another copy…`,
        );
      } else if (step.kind === "waiting") {
        setNote(`Waiting for a device to take on stage ${step.stage + 1}…`);
      } else if (step.kind === "recovered") {
        setBeams({});
        setNote(
          `Carrying on through ${step.route
            .split(" -> ")
            .map((m) => nameOfIn(startView, m))
            .join(" and ")}`,
        );
      }
      await sleep(step.seconds * 1000);
    }

    if (runId.current === id) {
      setBusyKey(null);
      setNote(null);
      setFrozen(null);
      setLost(new Set());
      setPhase("done");
    }
  }

  async function run(withCrash: boolean) {
    const current = viewRef.current;
    // A crash needs an answer long enough to still be under way when it
    // happens, so that run asks its own question rather than the typed one.
    const question = withCrash ? CRASH_QUESTION : prompt;
    if (!question.trim() || phase === "running" || !current) return;
    runId.current++;
    setRunError(null);
    setAsked(question);
    setPhase("running");
    setBeams({});
    setTokens([]);
    setResult(null);

    let via: string | undefined;
    let victim: PoolMember | undefined;
    if (withCrash) {
      // The router prefers the fastest copy, so that is the likeliest to be
      // carrying the request. Ask a different device to take the question, or
      // crashing the one in charge would end the request instead of testing it.
      const holders = current.members
        .filter(
          (m) =>
            m.liveness === "alive" &&
            m.hosting.some((h) => h.status === "ready"),
        )
        .sort((a, b) => b.speed - a.speed);
      victim = holders[0];
      const requester =
        current.members.find(
          (m) =>
            m.liveness === "alive" &&
            m.state === "active" &&
            m.name !== victim?.name &&
            m.hosting.length === 0,
        ) ??
        current.members.find(
          (m) => m.liveness === "alive" && m.name !== victim?.name,
        );
      if (requester) via = memberUrl(requester);
    }

    const timer = withCrash
      ? setTimeout(() => {
          // Whichever device is busiest right now, other than the requester.
          const now = viewRef.current;
          const busiest = now?.members
            .filter((m) => m.liveness === "alive" && memberUrl(m) !== via)
            .sort(
              (a, b) =>
                b.hosting.reduce((s, h) => s + h.inflight, 0) -
                a.hosting.reduce((s, h) => s + h.inflight, 0),
            )[0];
          const target =
            busiest && busiest.hosting.some((h) => h.inflight > 0)
              ? busiest
              : victim;
          if (target)
            void pool.labAction(target.name, "kill").catch(() => undefined);
        }, 1300)
      : undefined;

    try {
      const output = await pool.generate(question, {
        via,
        ...(withCrash ? { maxNewTokens: 90, systemPrompt: "" } : {}),
      });
      clearTimeout(timer);
      setResult(output);
      await play(output, current);
    } catch (e) {
      clearTimeout(timer);
      setPhase("idle");
      setRunError(
        e instanceof PoolError || e instanceof Error ? e.message : String(e),
      );
    }
  }

  // --- the owner's controls ------------------------------------------------------

  const actionsFor = (
    member: PoolMember | null,
    device: LabDevice | null | undefined,
  ) => ({
    onOwner: member
      ? () => {
          const away = member.state === "draining" || member.state === "away";
          void pool
            .setOwnerBusy(member, !away)
            .then(pool.refresh, () => undefined);
        }
      : undefined,
    onStop: device
      ? () =>
          void pool
            .labAction(device.name, "stop")
            .then(pool.refresh, () => undefined)
      : undefined,
    onKill: device
      ? () =>
          void pool
            .labAction(device.name, "kill")
            .then(pool.refresh, () => undefined)
      : undefined,
    onStart: device
      ? () =>
          void pool
            .labAction(device.name, "start")
            .then(pool.refresh, () => undefined)
      : undefined,
  });

  const healthText = useMemo(() => {
    if (!shown) return null;
    const need = shown.pool.replicas;
    const thin = shown.ready_copies.findIndex((n) => n < need);
    if (shown.health === "healthy")
      return `Every stage is on ${need} devices. Any one can go.`;
    const missing = shown.ready_copies.findIndex((n) => n === 0);
    if (missing >= 0)
      return `No device holds stage ${missing + 1}, so questions cannot be answered.`;
    return `Stage ${thin + 1} is on one device only. Losing it would stop the pool.`;
  }, [shown]);

  const nowSec = view?.now ?? 0;
  const crashAvailable = !!lab && lab.some((d) => d.running);

  return (
    <main className="mx-auto flex w-full max-w-6xl flex-col gap-5 px-4 py-8 sm:px-6">
      <header className="flex flex-wrap items-end justify-between gap-3">
        <div>
          <h1 className="text-xl font-semibold tracking-tight">
            Torrent-LLM pool
          </h1>
          <p
            className="mt-1 text-sm"
            style={{ color: "var(--text-secondary)" }}
          >
            Devices lending what they have, working as one. Any of them can
            answer.
          </p>
        </div>
        <Nav current="/" />
      </header>

      {/* ---- health ---- */}
      <section
        className="flex flex-wrap items-center gap-x-4 gap-y-1 rounded-xl px-4 py-2.5 text-sm"
        style={{
          background: "var(--surface-1)",
          border: "1px solid var(--border)",
        }}
      >
        {shown ? (
          <>
            <span className="flex items-center gap-2 font-medium">
              <span
                className="size-2.5 rounded-full"
                style={{
                  background:
                    shown.health === "healthy"
                      ? "var(--status-good)"
                      : shown.health === "degraded"
                        ? "var(--status-warn)"
                        : "var(--status-critical)",
                }}
              />
              {shown.health === "healthy"
                ? "Healthy"
                : shown.health === "degraded"
                  ? "Degraded"
                  : "Down"}
            </span>
            <span style={{ color: "var(--text-secondary)" }}>{healthText}</span>
            <span
              className="tabular ml-auto text-xs"
              style={{ color: "var(--text-muted)" }}
            >
              {shown.pool.model_id} ·{" "}
              {shown.members.filter((m) => m.liveness === "alive").length}{" "}
              devices online · talking to {shown.self}
            </span>
          </>
        ) : (
          <span
            style={{
              color: error ? "var(--status-critical)" : "var(--text-secondary)",
            }}
          >
            {error ?? "Looking for the pool…"}
            {error && (
              <span
                className="ml-2 text-xs"
                style={{ color: "var(--text-muted)" }}
              >
                Start it with scripts/pool.sh
              </span>
            )}
          </span>
        )}
      </section>

      {/* ---- the stage ---- */}
      <LayoutGroup>
        <section
          className="rounded-2xl px-4 pt-5 pb-5 sm:px-6"
          style={{
            background: "var(--surface-1)",
            border: "1px solid var(--border)",
          }}
        >
          <AnimatePresence>
            {note && (
              <motion.div
                initial={{ opacity: 0, y: -6 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0 }}
                className="mb-4 rounded-lg px-3 py-2 text-xs"
                style={{
                  background: "var(--page-plane)",
                  border: "1px solid var(--status-warn)",
                  color: "var(--text-primary)",
                }}
              >
                {note}
              </motion.div>
            )}
          </AnimatePresence>

          <div
            ref={stageRef}
            className="relative flex flex-col items-stretch gap-10 md:flex-row md:gap-6"
          >
            <div className="z-10 flex items-center justify-center md:w-16">
              <Terminal
                icon={<UserRound size={18} />}
                title="You"
                refCb={register("you")}
              />
            </div>

            {layout?.copies.map((list, stage) => {
              const info = shown!.pool.stages[stage];
              const wanted = shown!.pool.replicas;
              const missing = Math.max(0, wanted - list.length);
              const tone =
                list.length >= wanted
                  ? "good"
                  : list.length >= 1
                    ? "warn"
                    : "bad";
              return (
                <div
                  key={stage}
                  className="z-10 flex min-w-0 flex-1 flex-col gap-2.5"
                >
                  <div className="flex items-baseline justify-between gap-2 px-0.5">
                    <div className="flex items-center gap-2 text-xs font-medium">
                      <span
                        className="size-2 rounded-full"
                        style={{ background: stageColor(stage) }}
                      />
                      Stage {stage + 1}
                      <span
                        className="tabular font-normal"
                        style={{ color: "var(--text-muted)" }}
                      >
                        layers {info.start}–{info.end - 1}
                      </span>
                    </div>
                    <span
                      className="tabular text-[11px]"
                      style={{
                        color:
                          tone === "good"
                            ? "var(--status-good)"
                            : tone === "warn"
                              ? "var(--status-warn)"
                              : "var(--status-critical)",
                      }}
                    >
                      {list.length > wanted
                        ? `${list.length} copies`
                        : `${list.length} of ${wanted} copies`}
                    </span>
                  </div>
                  <AnimatePresence mode="popLayout">
                    {list.map(({ member, hosted }) => {
                      const key = `${member.name}#${stage}`;
                      return (
                        <CopyCard
                          key={key}
                          ref={register(key)}
                          layoutId={key}
                          member={member}
                          stage={stage}
                          layers={`${info.start}–${info.end - 1}`}
                          hosted={hosted}
                          lab={labByName.get(member.name)}
                          busy={busyKey === key}
                          lost={lost.has(member.name)}
                          actions={actionsFor(
                            member,
                            labByName.get(member.name),
                          )}
                        />
                      );
                    })}
                    {Array.from({ length: missing }, (_, i) => (
                      <MissingCopy
                        key={`missing-${stage}-${i}`}
                        stage={stage}
                        anyDevices={(layout?.bench ?? []).some(
                          (m) => m.liveness === "alive" && m.state === "active",
                        )}
                      />
                    ))}
                  </AnimatePresence>
                </div>
              );
            })}

            <div className="z-10 flex items-center justify-center md:w-16">
              <Terminal
                icon={<Sparkles size={18} />}
                title="Answer"
                refCb={register("answer")}
              />
            </div>

            {Object.entries(beams).map(([id, beam]) => (
              <AnimatedBeam
                key={id}
                containerRef={stageRef}
                fromRef={refFor(beam.from)}
                toRef={refFor(beam.to)}
                fit="edges"
                remeasureKey={remeasure}
                pulse={beam.pulse}
                duration={beam.duration}
                label={beam.label}
                gradientStartColor={
                  beam.failed ? "var(--status-critical)" : "var(--series-1)"
                }
                gradientStopColor={
                  beam.failed ? "var(--status-critical)" : "var(--series-2)"
                }
              />
            ))}
          </div>

          {/* ---- the bench ---- */}
          <div
            className="mt-6 border-t pt-4"
            style={{ borderColor: "var(--border)" }}
          >
            <div className="mb-2.5 flex items-baseline justify-between">
              <h2 className="text-xs font-medium">Standing by</h2>
              <span
                className="text-[11px]"
                style={{ color: "var(--text-muted)" }}
              >
                Spare, away, or offline. Hover a device for controls.
              </span>
            </div>
            <div className="grid grid-cols-1 gap-2.5 sm:grid-cols-2 lg:grid-cols-3">
              <AnimatePresence mode="popLayout">
                {layout?.bench.map((member) => (
                  <BenchTile
                    key={`dev-${member.name}`}
                    layoutId={`dev-${member.name}`}
                    member={member}
                    lab={labByName.get(member.name)}
                    actions={actionsFor(member, labByName.get(member.name))}
                  />
                ))}
                {layout?.ghosts.map((device) => (
                  <BenchTile
                    key={`dev-${device.name}`}
                    layoutId={`dev-${device.name}`}
                    member={null}
                    lab={device}
                    actions={actionsFor(null, device)}
                  />
                ))}
              </AnimatePresence>
              {layout && layout.bench.length + layout.ghosts.length === 0 && (
                <p className="text-xs" style={{ color: "var(--text-muted)" }}>
                  Every device is working. Add one to see it take a share.
                </p>
              )}
            </div>
          </div>
        </section>
      </LayoutGroup>

      {/* ---- ask ---- */}
      <form
        autoComplete="off"
        className="flex flex-col gap-2 sm:flex-row"
        onSubmit={(event) => {
          event.preventDefault();
          void run(false);
        }}
      >
        <input
          value={prompt}
          onChange={(event) => setPrompt(event.target.value)}
          placeholder="Ask the pool something"
          className="min-w-0 flex-1 rounded-xl px-4 py-3 text-sm outline-none focus:ring-2"
          style={{
            background: "var(--surface-1)",
            border: "1px solid var(--border)",
            color: "var(--text-primary)",
          }}
        />
        <button
          type="submit"
          disabled={phase === "running" || !view || view.health === "down"}
          className="flex items-center justify-center gap-2 rounded-xl px-5 py-3 text-sm font-medium text-white transition-[transform,opacity] active:scale-[0.98] disabled:opacity-50"
          style={{ background: "var(--series-1)" }}
        >
          <Play size={15} />
          {phase === "running" ? "Asking…" : "Ask"}
        </button>
        {crashAvailable && (
          <button
            type="button"
            disabled={
              phase === "running" ||
              phase === "playing" ||
              !view ||
              view.health === "down"
            }
            onClick={() => void run(true)}
            title="Asks for a long answer, then crashes a device partway through it"
            className="flex items-center justify-center gap-2 rounded-xl px-4 py-3 text-sm transition-transform active:scale-[0.98] disabled:opacity-50"
            style={{
              border: "1px solid var(--status-critical)",
              color: "var(--status-critical)",
            }}
          >
            <Zap size={15} />
            Ask, then crash a device
          </button>
        )}
      </form>

      {runError && (
        <div
          className="rounded-xl px-4 py-3 text-sm"
          style={{
            border: "1px solid var(--status-critical)",
            color: "var(--status-critical)",
          }}
        >
          {runError}
        </div>
      )}

      <div className="grid gap-5 lg:grid-cols-[1fr_20rem]">
        <div className="flex flex-col gap-4">
          <section
            className="min-h-24 rounded-2xl px-5 py-4 text-lg leading-relaxed"
            style={{
              background: "var(--surface-1)",
              border: "1px solid var(--border)",
            }}
          >
            {asked && (
              <div
                className="mb-1 text-sm"
                style={{ color: "var(--text-muted)" }}
              >
                {asked}
              </div>
            )}
            {tokens.map((token, i) => (
              <motion.span
                key={i}
                initial={{ opacity: 0, filter: "blur(4px)" }}
                animate={{ opacity: 1, filter: "blur(0px)" }}
                transition={{ duration: 0.35 }}
              >
                {token}
              </motion.span>
            ))}
            {(phase === "running" || phase === "playing") && (
              <motion.span
                className="ml-0.5 inline-block h-5 w-0.5 translate-y-1"
                style={{ background: "var(--series-1)" }}
                animate={{ opacity: [1, 0, 1] }}
                transition={{ duration: 1, repeat: Infinity }}
              />
            )}
            {phase === "idle" && !result && (
              <span className="text-sm" style={{ color: "var(--text-muted)" }}>
                The answer appears here, one token at a time.
              </span>
            )}
          </section>

          <AnimatePresence>
            {phase !== "idle" && phase !== "running" && (
              <motion.section
                initial={{ opacity: 0, y: 8 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0 }}
                className="grid grid-cols-2 gap-3 sm:grid-cols-5"
              >
                <Stat
                  label="Sent between devices"
                  value={formatBytes(totals.betweenDevicesBytes)}
                />
                <Stat
                  label="Time on the network"
                  value={formatMs(totals.networkMs)}
                />
                <Stat
                  label="Time computing"
                  value={formatMs(totals.computeMs)}
                />
                <Stat label="Tokens" value={String(totals.tokens)} />
                <Stat
                  label="Devices lost"
                  value={String(totals.lost)}
                  alert={totals.lost > 0}
                />
              </motion.section>
            )}
          </AnimatePresence>
          {phase === "playing" && (
            <p className="text-xs" style={{ color: "var(--text-muted)" }}>
              Replaying the measured run, slowed down so each transfer is
              visible.
            </p>
          )}
        </div>

        <section
          className="rounded-2xl px-4 py-3.5"
          style={{
            background: "var(--surface-1)",
            border: "1px solid var(--border)",
          }}
        >
          <h2 className="mb-2.5 text-xs font-medium">
            What the pool has been doing
          </h2>
          <Feed events={events} now={nowSec} />
        </section>
      </div>
    </main>
  );
}

function nameOfIn(view: PoolView, member: string): string {
  return view.members.find((m) => m.name === member)?.label || member;
}

function Terminal({
  icon,
  title,
  refCb,
}: {
  icon: React.ReactNode;
  title: string;
  refCb: (el: HTMLElement | null) => void;
}) {
  return (
    <div className="flex flex-row items-center gap-2 md:flex-col">
      <div
        ref={refCb}
        className="flex size-12 items-center justify-center rounded-full"
        style={{
          background: "var(--page-plane)",
          border: "1px solid var(--border)",
          color: "var(--text-secondary)",
        }}
      >
        {icon}
      </div>
      <span className="text-xs" style={{ color: "var(--text-muted)" }}>
        {title}
      </span>
    </div>
  );
}

function Stat({
  label,
  value,
  alert,
}: {
  label: string;
  value: string;
  alert?: boolean;
}) {
  return (
    <div
      className="rounded-xl px-4 py-3"
      style={{
        background: "var(--surface-1)",
        border: `1px solid ${alert ? "var(--status-critical)" : "var(--border)"}`,
      }}
    >
      <div className="text-xs" style={{ color: "var(--text-muted)" }}>
        {label}
      </div>
      <div
        className="tabular mt-1 text-lg font-semibold"
        style={{ color: alert ? "var(--status-critical)" : undefined }}
      >
        {value}
      </div>
    </div>
  );
}

/**
 * Where each element of the stage is, looked up by name.
 *
 * Beams need a reference to the elements they join, and those elements come and
 * go as devices move. This keeps one stable reference per name that always
 * points at whatever element currently has it, so a beam need not be rebuilt
 * when a card is replaced.
 */
class NodeRegistry {
  private els = new Map<string, HTMLElement>();
  private refs = new Map<string, RefObject<HTMLElement | null>>();
  private callbacks = new Map<string, (el: HTMLElement | null) => void>();

  ref(key: string): RefObject<HTMLElement | null> {
    let ref = this.refs.get(key);
    if (!ref) {
      const els = this.els;
      ref = {
        get current() {
          return els.get(key) ?? null;
        },
      } as RefObject<HTMLElement | null>;
      this.refs.set(key, ref);
    }
    return ref;
  }

  callback(key: string): (el: HTMLElement | null) => void {
    let cb = this.callbacks.get(key);
    if (!cb) {
      cb = (el) => {
        if (el) this.els.set(key, el);
        else this.els.delete(key);
      };
      this.callbacks.set(key, cb);
    }
    return cb;
  }
}
