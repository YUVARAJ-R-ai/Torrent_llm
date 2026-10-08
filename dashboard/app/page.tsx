"use client";

/**
 * The showcase view: one language model split across two devices.
 *
 * Everything on screen comes from a real run. The prompt goes to the API, the
 * API drives the shard chain, and the per-hop records that come back are
 * replayed here as an animation: tokens into the first device, the hidden
 * state across the link to the second, scores back out, one decode step at a
 * time. Only the pacing is slowed so it can be followed; see lib/replay.ts.
 *
 * The dense testing view this replaced lives at /details.
 */

import { AnimatePresence, motion } from "motion/react";
import { Laptop, MessageSquare, Play, RotateCcw, Sparkles } from "lucide-react";
import Link from "next/link";
import {
  forwardRef,
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";

import AnimatedBeam from "@/components/AnimatedBeam";
import { ApiError, api, type GenerateResult, type Topology } from "@/lib/api";
import { formatBytes, formatMs } from "@/lib/format";
import { buildTimeline, type Step } from "@/lib/replay";

/** A ceiling, not a target: the chain stops as soon as the answer ends. */
const MAX_NEW_TOKENS = 40;

type Phase = "idle" | "waiting" | "playing" | "done";

interface Totals {
  betweenDevicesBytes: number;
  networkMs: number;
  computeMs: number;
  tokens: number;
}

const ZERO: Totals = {
  betweenDevicesBytes: 0,
  networkMs: 0,
  computeMs: 0,
  tokens: 0,
};

/** "[0:14)" from the API, shown as "Layers 0–13". */
function layerLabel(range: string): string {
  const match = /\[(\d+):(\d+)\)/.exec(range);
  if (!match) return range;
  return `Layers ${match[1]}–${Number(match[2]) - 1}`;
}

export default function Demo() {
  const [topology, setTopology] = useState<Topology | null>(null);
  const [error, setError] = useState<string | null>(null);

  const [prompt, setPrompt] = useState("What is the capital of France?");
  const [phase, setPhase] = useState<Phase>("idle");
  const [result, setResult] = useState<GenerateResult | null>(null);
  // The question as sent, so editing the box afterwards does not relabel
  // the answer on screen.
  const [asked, setAsked] = useState("");

  // Replay state: which link is lit, which node is busy, what has arrived.
  const [pulses, setPulses] = useState<number[]>([]);
  const [labels, setLabels] = useState<string[]>([]);
  const [durations, setDurations] = useState<number[]>([]);
  const [busy, setBusy] = useState<number | null>(null);
  const [lastCompute, setLastCompute] = useState<(number | null)[]>([]);
  const [tokens, setTokens] = useState<string[]>([]);
  const [totals, setTotals] = useState<Totals>(ZERO);
  const runId = useRef(0);

  // State is only set once the request settles, never synchronously, so this
  // is safe to call from an effect.
  const loadTopology = useCallback(
    () =>
      api.topology().then(
        (t) => {
          setTopology(t);
          setError(null);
        },
        (e) => {
          setTopology(null);
          setError(e instanceof ApiError ? e.message : String(e));
        },
      ),
    [],
  );

  useEffect(() => {
    void loadTopology();
  }, [loadTopology]);

  const numShards = topology?.shards.length ?? 2;
  const numLinks = numShards + 1;

  function resetStage() {
    setPulses(Array(numLinks).fill(0));
    setDurations(Array(numLinks).fill(0.6));
    setLabels(Array(numLinks).fill(""));
    setBusy(null);
    setLastCompute(Array(numShards).fill(null));
    setTokens([]);
    setTotals(ZERO);
  }

  async function play(run: GenerateResult) {
    const id = ++runId.current;
    resetStage();
    setPhase("playing");

    const steps: Step[] = buildTimeline(run, numShards);
    for (const step of steps) {
      // A newer run (or a reset) cancels this one between steps.
      if (runId.current !== id) return;

      if (step.kind === "send") {
        setBusy(null);
        setLabels((prev) =>
          prev.map((l, i) => (i === step.link ? step.label : l)),
        );
        setDurations((prev) =>
          prev.map((d, i) => (i === step.link ? step.seconds : d)),
        );
        setPulses((prev) => prev.map((p, i) => (i === step.link ? p + 1 : p)));
        setTotals((t) => ({
          ...t,
          networkMs: t.networkMs + step.networkMs,
          betweenDevicesBytes:
            t.betweenDevicesBytes + (step.betweenDevices ? step.bytes : 0),
        }));
      } else if (step.kind === "compute") {
        setBusy(step.node);
        setLastCompute((prev) =>
          prev.map((c, i) => (i === step.node ? step.computeMs : c)),
        );
        setTotals((t) => ({ ...t, computeMs: t.computeMs + step.computeMs }));
      } else {
        setBusy(null);
        setTokens((prev) => [...prev, step.text]);
        setTotals((t) => ({ ...t, tokens: t.tokens + 1 }));
      }
      await new Promise((resolve) => setTimeout(resolve, step.seconds * 1000));
    }

    if (runId.current === id) {
      setBusy(null);
      setPhase("done");
    }
  }

  async function run() {
    if (!prompt.trim() || phase === "waiting") return;
    runId.current++;
    resetStage();
    setError(null);
    setAsked(prompt);
    setPhase("waiting");
    try {
      const output = await api.generate(prompt, MAX_NEW_TOKENS, true, true);
      setResult(output);
      await play(output);
    } catch (e) {
      setPhase("idle");
      setError(e instanceof ApiError ? e.message : String(e));
    }
  }

  // Refs for the beams: prompt, each device, answer.
  const stageRef = useRef<HTMLDivElement>(null);
  const promptRef = useRef<HTMLDivElement>(null);
  const answerRef = useRef<HTMLDivElement>(null);
  const deviceRefs = useRef<(HTMLDivElement | null)[]>([]);
  // Stable objects per node, so the beams measure once per layout change and
  // not on every render. Each reads its element lazily, since the elements
  // only exist after the first render.
  const nodeRefs = useMemo(
    () =>
      Array.from({ length: numShards + 2 }, (_, index) => ({
        get current(): HTMLElement | null {
          if (index === 0) return promptRef.current;
          if (index === numShards + 1) return answerRef.current;
          return deviceRefs.current[index - 1] ?? null;
        },
      })),
    [numShards],
  );

  const captions = Array.from({ length: numLinks }, (_, i) =>
    i === 0 ? "prompt" : i === numLinks - 1 ? "scores" : "hidden state",
  );

  return (
    <main className="mx-auto flex w-full max-w-4xl flex-col gap-6 px-4 py-10 sm:px-6">
      <header className="flex flex-wrap items-end justify-between gap-3">
        <div>
          <h1 className="text-xl font-semibold tracking-tight">Torrent-LLM</h1>
          <p
            className="mt-1 text-sm"
            style={{ color: "var(--text-secondary)" }}
          >
            One language model, split across two devices.
          </p>
        </div>
        <Link
          href="/details"
          className="text-xs underline-offset-2 hover:underline"
          style={{ color: "var(--text-muted)" }}
        >
          Detailed metrics
        </Link>
      </header>

      {/* ---- the stage ---- */}
      <section
        className="rounded-2xl px-4 pt-6 pb-10 sm:px-8"
        style={{
          background: "var(--surface-1)",
          border: "1px solid var(--border)",
        }}
      >
        <div className="mb-8 flex flex-wrap items-center justify-between gap-2 text-xs">
          <span style={{ color: "var(--text-secondary)" }}>
            {topology ? topology.model_id : "Connecting to the chain…"}
          </span>
          {topology?.link && (
            <span
              className="tabular rounded-full px-2.5 py-1"
              style={{
                background: "var(--page-plane)",
                color: "var(--text-secondary)",
              }}
            >
              Simulated network · {topology.link.bandwidth_mbps} Mbps ·{" "}
              {topology.link.latency_ms} ms
            </span>
          )}
        </div>

        <div
          ref={stageRef}
          className="relative flex flex-col items-center justify-between gap-20 sm:flex-row sm:gap-6"
        >
          <Endpoint
            ref={promptRef}
            icon={<MessageSquare size={18} />}
            title="You"
          />

          {(topology?.shards ?? [null, null]).map((shard, i) => (
            <Device
              key={i}
              ref={(el) => {
                deviceRefs.current[i] = el;
              }}
              name={`Device ${String.fromCharCode(65 + i)}`}
              detail={shard ? layerLabel(shard.layers) : "…"}
              hardware={shard?.device.toUpperCase()}
              busy={busy === i}
              computeMs={lastCompute[i] ?? null}
            />
          ))}

          <Endpoint
            ref={answerRef}
            icon={<Sparkles size={18} />}
            title="Answer"
          />

          {Array.from({ length: numLinks }, (_, i) => (
            <AnimatedBeam
              key={`${i}-${numShards}`}
              containerRef={stageRef}
              fromRef={nodeRefs[i]}
              toRef={nodeRefs[i + 1]}
              pulse={pulses[i] ?? 0}
              duration={durations[i] ?? 0.6}
              label={labels[i]}
              caption={captions[i]}
            />
          ))}
        </div>
      </section>

      {/* ---- prompt ---- */}
      {/* autoComplete off stops Firefox restoring the button's disabled state
          on reload, which made the server and client HTML disagree. */}
      <form
        autoComplete="off"
        className="flex flex-col gap-2 sm:flex-row"
        onSubmit={(event) => {
          event.preventDefault();
          void run();
        }}
      >
        <input
          value={prompt}
          onChange={(event) => setPrompt(event.target.value)}
          placeholder="Type a prompt"
          className="min-w-0 flex-1 rounded-xl px-4 py-3 text-sm outline-none focus:ring-2"
          style={{
            background: "var(--surface-1)",
            border: "1px solid var(--border)",
            color: "var(--text-primary)",
          }}
        />
        <button
          type="submit"
          disabled={phase === "waiting" || !topology}
          className="flex items-center justify-center gap-2 rounded-xl px-5 py-3 text-sm font-medium text-white transition-[transform,opacity] active:scale-[0.98] disabled:opacity-50"
          style={{ background: "var(--series-1)" }}
        >
          <Play size={15} />
          {phase === "waiting" ? "Running…" : "Run"}
        </button>
        {phase === "done" && result && (
          <button
            type="button"
            onClick={() => void play(result)}
            className="flex items-center justify-center gap-2 rounded-xl px-4 py-3 text-sm transition-transform active:scale-[0.98]"
            style={{
              border: "1px solid var(--border)",
              color: "var(--text-secondary)",
            }}
          >
            <RotateCcw size={15} />
            Replay
          </button>
        )}
      </form>

      {error && (
        <div
          className="rounded-xl px-4 py-3 text-sm"
          style={{
            border: "1px solid var(--status-critical)",
            color: "var(--status-critical)",
          }}
        >
          {error}
          {!topology && (
            <span
              className="mt-1 block text-xs"
              style={{ color: "var(--text-secondary)" }}
            >
              Start everything with <code>scripts/demo.sh</code>, then{" "}
              <button className="underline" onClick={() => void loadTopology()}>
                retry
              </button>
              .
            </span>
          )}
        </div>
      )}

      {/* ---- output ---- */}
      <section
        className="min-h-24 rounded-2xl px-5 py-4 text-lg leading-relaxed"
        style={{
          background: "var(--surface-1)",
          border: "1px solid var(--border)",
        }}
      >
        {asked && (
          <div className="mb-1 text-sm" style={{ color: "var(--text-muted)" }}>
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
        {(phase === "waiting" || phase === "playing") && (
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

      {/* ---- numbers ---- */}
      <AnimatePresence>
        {phase !== "idle" && (
          <motion.section
            initial={{ opacity: 0, y: 8 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0 }}
            className="grid grid-cols-2 gap-3 sm:grid-cols-4"
          >
            <Stat
              label="Sent between devices"
              value={formatBytes(totals.betweenDevicesBytes)}
            />
            <Stat
              label="Time on the network"
              value={formatMs(totals.networkMs)}
            />
            <Stat label="Time computing" value={formatMs(totals.computeMs)} />
            <Stat label="Tokens" value={String(totals.tokens)} />
          </motion.section>
        )}
      </AnimatePresence>

      {phase === "playing" && (
        <p
          className="text-center text-xs"
          style={{ color: "var(--text-muted)" }}
        >
          Replaying the measured run, slowed down so each transfer is visible.
        </p>
      )}
    </main>
  );
}

const Endpoint = forwardRef<
  HTMLDivElement,
  { icon: React.ReactNode; title: string }
>(function Endpoint({ icon, title }, ref) {
  return (
    <div className="z-10 flex flex-col items-center gap-2">
      <div
        ref={ref}
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
});

const Device = forwardRef<
  HTMLDivElement,
  {
    name: string;
    detail: string;
    hardware?: string;
    busy: boolean;
    computeMs: number | null;
  }
>(function Device({ name, detail, hardware, busy, computeMs }, ref) {
  return (
    <div className="z-10 flex flex-col items-center gap-2">
      <motion.div
        ref={ref}
        className="relative flex w-36 flex-col items-center gap-1.5 rounded-2xl px-4 py-4"
        // The ring is a CSS transition rather than a motion value: motion
        // cannot interpolate a shadow whose colour is a CSS variable.
        style={{
          background: "var(--page-plane)",
          border: "1px solid var(--border)",
          boxShadow: busy
            ? "0 0 0 2px var(--series-1), 0 12px 32px -10px var(--series-1)"
            : "0 0 0 0 transparent, 0 0 0 0 transparent",
          transition: "box-shadow 180ms ease-out",
        }}
        animate={{ scale: busy ? 1.05 : 1 }}
        transition={{ type: "spring", stiffness: 300, damping: 22 }}
      >
        <Laptop
          size={22}
          style={{ color: busy ? "var(--series-1)" : "var(--text-secondary)" }}
        />
        <span className="text-sm font-medium">{name}</span>
        <span
          className="tabular text-xs"
          style={{ color: "var(--text-secondary)" }}
        >
          {detail}
        </span>
        {hardware && (
          <span
            className="rounded px-1.5 py-0.5 text-[10px] tracking-wide"
            style={{
              background: "var(--surface-1)",
              color: "var(--text-muted)",
            }}
          >
            {hardware}
          </span>
        )}
      </motion.div>
      <span
        className="tabular h-4 text-xs"
        style={{ color: "var(--text-muted)" }}
      >
        {busy
          ? "computing…"
          : computeMs !== null
            ? `${formatMs(computeMs)} compute`
            : ""}
      </span>
    </div>
  );
});

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div
      className="rounded-xl px-4 py-3"
      style={{
        background: "var(--surface-1)",
        border: "1px solid var(--border)",
      }}
    >
      <div className="text-xs" style={{ color: "var(--text-muted)" }}>
        {label}
      </div>
      <div className="tabular mt-1 text-lg font-semibold">{value}</div>
    </div>
  );
}
