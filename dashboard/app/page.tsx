"use client";

/**
 * Torrent-LLM chain dashboard (issue #27).
 *
 * A testing and visualization aid, explicitly not the project's final
 * interface. It exists so the chain's behaviour is legible while the parts that
 * matter (the compressor, #8/#9/#12) get built.
 *
 * The organising principle throughout: **prefill and cached decode are never
 * plotted on the same axis.** Their payloads differ by roughly three orders of
 * magnitude, so one always vanishes next to the other, and the whole point of
 * docs/bandwidth-regimes.md is that averaging the two regimes together produces
 * a wrong number that looks reasonable.
 */

import { useCallback, useEffect, useState } from "react";

import HopTable from "@/components/HopTable";
import HorizontalBars, { type BarRow } from "@/components/HorizontalBars";
import Panel from "@/components/Panel";
import {
  ApiError,
  api,
  type CacheComparison,
  type GenerateResult,
  type HopMetrics,
  type Topology,
} from "@/lib/api";
import {
  formatBytes,
  formatMs,
  formatPercent,
  formatRatio,
  hopLabel,
} from "@/lib/format";

export default function Dashboard() {
  const [topology, setTopology] = useState<Topology | null>(null);
  const [topologyError, setTopologyError] = useState<string | null>(null);

  const [prompt, setPrompt] = useState("The capital of France is");
  const [maxTokens, setMaxTokens] = useState(8);
  const [running, setRunning] = useState(false);
  const [runError, setRunError] = useState<string | null>(null);
  const [result, setResult] = useState<GenerateResult | null>(null);
  const [comparison, setComparison] = useState<CacheComparison | null>(null);

  const loadTopology = useCallback(async () => {
    setTopologyError(null);
    try {
      setTopology(await api.topology());
    } catch (error) {
      setTopology(null);
      setTopologyError(
        error instanceof ApiError ? error.message : String(error),
      );
    }
  }, []);

  useEffect(() => {
    void loadTopology();
  }, [loadTopology]);

  async function run(compare: boolean) {
    setRunning(true);
    setRunError(null);
    try {
      if (compare) {
        const output = await api.compareCache(prompt, maxTokens);
        setComparison(output);
        setResult(output.cached);
      } else {
        setComparison(null);
        setResult(await api.generate(prompt, maxTokens, true));
      }
    } catch (error) {
      setRunError(error instanceof ApiError ? error.message : String(error));
    } finally {
      setRunning(false);
    }
  }

  return (
    <main className="mx-auto max-w-5xl px-6 py-10">
      <header className="mb-8">
        <h1
          className="text-lg font-semibold"
          style={{ color: "var(--text-primary)" }}
        >
          Torrent-LLM chain
        </h1>
        <p className="mt-1 text-xs" style={{ color: "var(--text-secondary)" }}>
          Testing and visualization for the layer-sharded inference chain. Not
          the final interface.
        </p>
      </header>

      <div className="flex flex-col gap-5">
        <TopologyPanel
          topology={topology}
          error={topologyError}
          onRetry={loadTopology}
        />

        <Panel
          title="Run a generation"
          note="Compare mode runs the same prompt twice — once with the server-side KV cache and once without — so the bandwidth difference is measured rather than assumed."
        >
          <div className="flex flex-wrap items-end gap-3">
            <label className="flex min-w-64 flex-1 flex-col gap-1">
              <span className="text-xs" style={{ color: "var(--text-muted)" }}>
                prompt
              </span>
              <input
                value={prompt}
                onChange={(event) => setPrompt(event.target.value)}
                className="rounded-md px-3 py-2 text-sm outline-none"
                style={{
                  background: "var(--page-plane)",
                  color: "var(--text-primary)",
                  border: "1px solid var(--border)",
                }}
              />
            </label>
            <label className="flex w-28 flex-col gap-1">
              <span className="text-xs" style={{ color: "var(--text-muted)" }}>
                new tokens
              </span>
              <input
                type="number"
                min={1}
                max={512}
                value={maxTokens}
                onChange={(event) =>
                  setMaxTokens(Number(event.target.value) || 1)
                }
                className="tabular rounded-md px-3 py-2 text-sm outline-none"
                style={{
                  background: "var(--page-plane)",
                  color: "var(--text-primary)",
                  border: "1px solid var(--border)",
                }}
              />
            </label>
            <button
              onClick={() => void run(false)}
              disabled={running || !topology}
              className="rounded-md px-4 py-2 text-sm font-medium disabled:opacity-50"
              style={{ background: "var(--series-1)", color: "#ffffff" }}
            >
              {running ? "running…" : "Generate"}
            </button>
            <button
              onClick={() => void run(true)}
              disabled={running || !topology}
              className="rounded-md px-4 py-2 text-sm font-medium disabled:opacity-50"
              style={{
                background: "transparent",
                color: "var(--text-primary)",
                border: "1px solid var(--border)",
              }}
            >
              Compare cache on/off
            </button>
          </div>

          {runError && (
            <p
              className="mt-3 text-xs"
              style={{ color: "var(--status-critical)" }}
            >
              {runError}
            </p>
          )}

          {result && (
            <div
              className="mt-4 rounded-md p-3 text-sm"
              style={{
                background: "var(--page-plane)",
                border: "1px solid var(--border)",
              }}
            >
              <span style={{ color: "var(--text-muted)" }}>{result.prompt}</span>
              <span style={{ color: "var(--text-primary)" }}>
                {result.completion}
              </span>
            </div>
          )}
        </Panel>

        {comparison && <CachePanel comparison={comparison} />}
        {result && <BandwidthPanel hops={result.hops} />}
        {result && <LatencyPanel hops={result.hops} />}
        {result && (
          <Panel
            title="Every hop"
            note="The same figures as above, in full. Also the accessible view of the charts."
          >
            <HopTable hops={result.hops} />
          </Panel>
        )}
      </div>
    </main>
  );
}

function TopologyPanel({
  topology,
  error,
  onRetry,
}: {
  topology: Topology | null;
  error: string | null;
  onRetry: () => void;
}) {
  return (
    <Panel
      title="Topology"
      note={
        topology
          ? `${topology.model_id} · ${topology.num_layers} layers across ${topology.num_shards} shards · codec "${topology.codec}" · ${topology.dtype}`
          : "What each node reports hosting, asked of the nodes themselves rather than read off the config."
      }
      action={
        <button
          onClick={onRetry}
          className="rounded-md px-2.5 py-1 text-xs"
          style={{
            color: "var(--text-secondary)",
            border: "1px solid var(--border)",
          }}
        >
          refresh
        </button>
      }
    >
      {error ? (
        <div className="text-xs" style={{ color: "var(--status-critical)" }}>
          <p>{error}</p>
          <p className="mt-2" style={{ color: "var(--text-secondary)" }}>
            Start the chain with{" "}
            <code>torrent-shard --config configs/local-2shard.yaml --index 0</code>{" "}
            (and <code>--index 1</code>), then{" "}
            <code>torrent-api --config configs/local-2shard.yaml</code>.
          </p>
        </div>
      ) : !topology ? (
        <p className="text-xs" style={{ color: "var(--text-muted)" }}>
          loading…
        </p>
      ) : (
        <div className="flex flex-wrap gap-2">
          {topology.shards.map((shard, index) => (
            <div key={shard.address} className="flex items-center gap-2">
              <div
                className="rounded-lg px-3 py-2"
                style={{
                  background: "var(--page-plane)",
                  border: "1px solid var(--border)",
                }}
              >
                <div
                  className="text-xs font-medium tabular"
                  style={{ color: "var(--text-primary)" }}
                >
                  shard {shard.shard} · layers {shard.layers}
                </div>
                <div
                  className="mt-0.5 text-xs tabular"
                  style={{ color: "var(--text-muted)" }}
                >
                  {shard.address} · {shard.device} · {shard.dtype} · hidden{" "}
                  {shard.hidden_size}
                </div>
              </div>
              {index < topology.shards.length - 1 && (
                // Hidden once the cards wrap: a horizontal arrow pointing at a
                // card that is now below it reads as wrong. The cards carry
                // their own shard index, so chain order survives without it.
                <span
                  aria-hidden
                  className="hidden text-sm md:inline"
                  style={{ color: "var(--text-muted)" }}
                >
                  →
                </span>
              )}
            </div>
          ))}
        </div>
      )}
    </Panel>
  );
}

function CachePanel({ comparison }: { comparison: CacheComparison }) {
  const rows: BarRow[] = [
    {
      label: "cache off",
      sublabel: "resends the whole prefix",
      segments: [
        {
          series: 2,
          value: comparison.uncached.total_sent_bytes,
          label: "bytes sent",
        },
      ],
      valueLabel: formatBytes(comparison.uncached.total_sent_bytes),
    },
    {
      label: "cache on",
      sublabel: "sends one new token",
      segments: [
        {
          series: 1,
          value: comparison.cached.total_sent_bytes,
          label: "bytes sent",
        },
      ],
      valueLabel: formatBytes(comparison.cached.total_sent_bytes),
    },
  ];

  return (
    <Panel
      title="KV cache on vs off"
      note={`Same prompt, ${comparison.tokens_generated} tokens, both ways. The two differ only in how much they recompute — never in what they compute.`}
    >
      <div className="mb-5 flex flex-wrap gap-8">
        <div>
          <div
            className="text-3xl font-semibold"
            style={{ color: "var(--text-primary)" }}
          >
            {formatRatio(comparison.bandwidth_reduction)}
          </div>
          <div className="text-xs" style={{ color: "var(--text-secondary)" }}>
            less data on the wire
          </div>
        </div>
        <div>
          <div
            className="text-3xl font-semibold"
            style={{ color: "var(--text-primary)" }}
          >
            {formatRatio(
              comparison.uncached.total_wall_ms /
                comparison.cached.total_wall_ms,
            )}
          </div>
          <div className="text-xs" style={{ color: "var(--text-secondary)" }}>
            faster end to end
          </div>
        </div>
        <div>
          <div
            className="flex items-center gap-1.5 text-sm font-medium"
            style={{
              color: comparison.same_output
                ? "var(--success-text)"
                : "var(--status-critical)",
            }}
          >
            {/* Icon + label, never color alone. */}
            <span aria-hidden>{comparison.same_output ? "✓" : "✕"}</span>
            {comparison.same_output
              ? "identical output"
              : "OUTPUTS DIVERGED — bug"}
          </div>
          <div className="text-xs" style={{ color: "var(--text-secondary)" }}>
            {comparison.same_output
              ? "both paths produced the same text"
              : "the two paths must never disagree"}
          </div>
        </div>
      </div>

      <HorizontalBars rows={rows} format={formatBytes} />
    </Panel>
  );
}

function BandwidthPanel({ hops }: { hops: HopMetrics[] }) {
  const prefill = hops.filter((hop) => hop.phase === "prefill");
  const decode = hops.filter((hop) => hop.phase === "decode");

  const toRows = (subset: HopMetrics[]): BarRow[] => {
    // One row per hop index, summed across steps: a 16-token generation makes
    // 16 decode passes over the same two hops, and 32 near-identical rows
    // communicate less than 2.
    const byHop = new Map<number, number>();
    for (const hop of subset) {
      byHop.set(hop.hop, (byHop.get(hop.hop) ?? 0) + hop.sent_bytes);
    }
    return [...byHop.entries()]
      .sort(([a], [b]) => a - b)
      .map(([hop, bytes]) => ({
        label: hopLabel(hop),
        sublabel: hop === 0 ? "token ids" : "activations",
        segments: [{ series: 1 as const, value: bytes, label: "bytes sent" }],
        valueLabel: formatBytes(bytes),
      }));
  };

  return (
    <Panel
      title="Bytes on the wire, by hop"
      note="Prefill and decode are charted separately and to their own scales, on purpose: their payloads differ by orders of magnitude, so a shared axis would flatten one to nothing. The ratio between them is the interesting number, not their relative bar lengths."
    >
      <div className="grid gap-8 md:grid-cols-2">
        <div>
          <h3
            className="mb-3 text-xs font-medium"
            style={{ color: "var(--text-secondary)" }}
          >
            Prefill — the whole prompt crosses each hop
          </h3>
          <HorizontalBars rows={toRows(prefill)} format={formatBytes} />
        </div>
        <div>
          <h3
            className="mb-3 text-xs font-medium"
            style={{ color: "var(--text-secondary)" }}
          >
            Decode — one position per step, summed over all steps
          </h3>
          {decode.length > 0 ? (
            <HorizontalBars rows={toRows(decode)} format={formatBytes} />
          ) : (
            <p className="text-xs" style={{ color: "var(--text-muted)" }}>
              no decode hops — generate more than one token to see this
            </p>
          )}
        </div>
      </div>
    </Panel>
  );
}

function LatencyPanel({ hops }: { hops: HopMetrics[] }) {
  const byHop = new Map<number, { compute: number; transport: number }>();
  for (const hop of hops) {
    const entry = byHop.get(hop.hop) ?? { compute: 0, transport: 0 };
    entry.compute += hop.compute_ms;
    entry.transport += hop.transport_ms;
    byHop.set(hop.hop, entry);
  }

  const rows: BarRow[] = [...byHop.entries()]
    .sort(([a], [b]) => a - b)
    .map(([hop, split]) => {
      const total = split.compute + split.transport;
      return {
        label: hopLabel(hop),
        sublabel: `${formatPercent(total ? split.transport / total : 0)} transport`,
        segments: [
          { series: 1 as const, value: split.compute, label: "compute" },
          { series: 2 as const, value: split.transport, label: "transport" },
        ],
        valueLabel: formatMs(total),
      };
    });

  return (
    <Panel
      title="Where each hop's time went"
      note="Transport is the round trip minus the shard's own reported compute, so it includes serialisation as well as the network. Its share is what decides whether compressing a hop could shorten it at all — a hop that spends 2% of its time on the wire cannot be made faster by sending fewer bytes."
    >
      <HorizontalBars
        rows={rows}
        format={(value) => formatMs(value)}
        legend={[
          { series: 1, label: "shard compute" },
          { series: 2, label: "transport" },
        ]}
      />
    </Panel>
  );
}
