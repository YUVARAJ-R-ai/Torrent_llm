/**
 * Talking to a pool of devices.
 *
 * Every device serves the same HTTP API, so the dashboard is not tied to any one
 * of them: it asks whichever answers, and when that one goes away it moves on to
 * another. Losing a device, even the one this page happens to be talking to,
 * therefore does not blank the screen. That is the point of the project, so the
 * page behaves the same way.
 *
 * The lab (src/torrent_llm/pool/lab.py) is separate and optional. It exists only
 * where devices are simulated on one machine, and it is what lets the page
 * switch a pretend laptop on, off, or crash it. With real devices there is no
 * lab and those buttons simply are not there.
 */

import { useCallback, useEffect, useRef, useState } from "react";

const SEEDS = (
  process.env.NEXT_PUBLIC_POOL_SEEDS ??
  "http://127.0.0.1:9101,http://127.0.0.1:9102,http://127.0.0.1:9103"
).split(",");
export const LAB_BASE =
  process.env.NEXT_PUBLIC_LAB_BASE ?? "http://127.0.0.1:8100";

export type Liveness = "alive" | "suspect" | "down";
export type DeviceState = "active" | "draining" | "away" | "left";

export interface HostedStage {
  stage: number;
  port: number;
  status: "loading" | "ready" | "failed";
  inflight: number;
  served: number;
}

export interface PoolMember {
  name: string;
  host: string;
  control_port: number;
  label: string;
  os: string;
  hostname: string;
  device: string;
  memory_mb: number;
  slots: number;
  speed: number;
  link: { latency_ms: number; bandwidth_mbps: number } | null;
  state: DeviceState;
  hosting: HostedStage[];
  liveness: Liveness;
  is_self: boolean;
  rtt_ms: number | null;
}

export interface PoolStageInfo {
  stage: number;
  start: number;
  end: number;
}

export interface PoolView {
  self: string;
  pool: {
    model_id: string;
    num_layers: number;
    replicas: number;
    shard_mb: number;
    stages: PoolStageInfo[];
  };
  members: PoolMember[];
  ready_copies: number[];
  health: "healthy" | "degraded" | "down";
  now: number;
}

export interface ActivityEvent {
  t: number;
  kind: string;
  text: string;
  member: string | null;
}

export interface HopInfo {
  stage: number;
  member: string;
  address: string;
  sent_bytes: number;
  received_bytes: number;
  compute_ms: number;
  transport_ms: number;
  wall_ms: number;
}

export interface PassInfo {
  index: number;
  step: number;
  phase: "prefill" | "decode";
  recovery: boolean;
  seq_len: number;
  route: [number, string][];
  hops: HopInfo[];
  token_id: number | null;
}

export interface RunEvent {
  kind: "failover" | "waiting" | "recovered";
  at_pass: number;
  step: number;
  stage: number | null;
  member: string | null;
  detail: string;
  t: number;
}

export interface PoolGenerateResult {
  prompt: string;
  completion: string;
  tokens: string[];
  requester: string;
  passes: PassInfo[];
  events: RunEvent[];
  totals: {
    tokens: number;
    failovers: number;
    sent_bytes: number;
    compute_ms: number;
    transport_ms: number;
  };
}

export interface LabDevice {
  name: string;
  label: string;
  device: string;
  memory_mb: number;
  speed: number;
  link: { latency_ms: number; bandwidth_mbps: number } | null;
  port: number;
  url: string;
  running: boolean;
  pid: number | null;
  last_event: string;
}

export class PoolError extends Error {
  constructor(
    message: string,
    readonly status: number,
  ) {
    super(message);
    this.name = "PoolError";
  }
}

async function getJson<T>(
  url: string,
  init?: RequestInit,
  timeoutMs = 2500,
): Promise<T> {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const response = await fetch(url, {
      ...init,
      signal: controller.signal,
      headers: { "Content-Type": "application/json", ...init?.headers },
    });
    if (!response.ok) {
      let detail = response.statusText;
      try {
        const body = await response.json();
        detail =
          typeof body.detail === "string"
            ? body.detail
            : JSON.stringify(body.detail);
      } catch {
        // Not JSON; the status text will do.
      }
      throw new PoolError(detail, response.status);
    }
    return (await response.json()) as T;
  } finally {
    clearTimeout(timer);
  }
}

/** A device's own URL, from what the pool says about it. */
export function memberUrl(member: PoolMember): string {
  return `http://${member.host}:${member.control_port}`;
}

export interface PoolState {
  view: PoolView | null;
  /** The device this page is currently talking to. */
  gateway: string | null;
  events: ActivityEvent[];
  lab: LabDevice[] | null;
  /** Set when no device can be reached at all. */
  error: string | null;
}

/**
 * Keeps a live picture of the pool.
 *
 * Polls whichever device answers, and on failure tries the others it has heard
 * of before giving up. The lab is polled alongside, and its absence is normal.
 */
export function usePool(intervalMs = 1000) {
  const [state, setState] = useState<PoolState>({
    view: null,
    gateway: null,
    events: [],
    lab: null,
    error: null,
  });
  const base = useRef<string | null>(null);
  const known = useRef<string[]>(SEEDS);
  const lastEvent = useRef(0);

  const poll = useCallback(async () => {
    // Start from the device that worked last time, then everyone else.
    const order = [base.current, ...known.current].filter(
      (u, i, all): u is string => !!u && all.indexOf(u) === i,
    );
    let view: PoolView | null = null;
    for (const url of order) {
      try {
        view = await getJson<PoolView>(`${url}/pool`);
        base.current = url;
        break;
      } catch {
        // Try the next device.
      }
    }

    if (!view) {
      for (const url of SEEDS.filter((u) => !order.includes(u))) {
        try {
          view = await getJson<PoolView>(`${url}/pool`);
          base.current = url;
          break;
        } catch {
          // Try the next seed.
        }
      }
    }

    let events: ActivityEvent[] = [];
    if (view && base.current) {
      // Only devices believed alive: asking a dead one's address would fill the
      // console with refused connections. The seeds come back in if nothing
      // else answers (see the fallback in poll).
      known.current = [
        ...new Set([
          base.current,
          ...view.members.filter((m) => m.liveness === "alive").map(memberUrl),
        ]),
      ];
      try {
        events = await getJson<ActivityEvent[]>(
          `${base.current}/events?since=${lastEvent.current}`,
        );
        if (events.length)
          lastEvent.current = Math.max(...events.map((e) => e.t));
      } catch {
        // The feed is a nicety.
      }
    }

    let lab: LabDevice[] | null = null;
    try {
      lab = (
        await getJson<{ devices: LabDevice[] }>(
          `${LAB_BASE}/lab`,
          undefined,
          1200,
        )
      ).devices;
    } catch {
      // No lab: real devices, or it is not running.
    }

    setState((prev) => ({
      view: view ?? prev.view,
      gateway: view ? view.self : null,
      events: events.length
        ? [...prev.events, ...events].slice(-60)
        : prev.events,
      lab,
      error: view ? null : "No device in the pool can be reached.",
    }));
  }, []);

  useEffect(() => {
    let stopped = false;
    let timer: ReturnType<typeof setTimeout>;
    const loop = async () => {
      await poll();
      if (!stopped) timer = setTimeout(loop, intervalMs);
    };
    void loop();
    return () => {
      stopped = true;
      clearTimeout(timer);
    };
  }, [poll, intervalMs]);

  const generate = useCallback(
    async (
      prompt: string,
      options: {
        maxNewTokens?: number;
        systemPrompt?: string;
        /** Ask this device specifically (its URL), instead of whichever answers. */
        via?: string;
      } = {},
    ) => {
      const order = options.via
        ? [options.via]
        : [base.current, ...known.current].filter(
            (u, i, all): u is string => !!u && all.indexOf(u) === i,
          );
      let lastError: unknown = new PoolError(
        "No device in the pool can be reached.",
        0,
      );
      for (const url of order) {
        try {
          return await getJson<PoolGenerateResult>(
            `${url}/generate`,
            {
              method: "POST",
              body: JSON.stringify({
                prompt,
                max_new_tokens: options.maxNewTokens ?? 40,
                ...(options.systemPrompt !== undefined
                  ? { system_prompt: options.systemPrompt }
                  : {}),
              }),
            },
            120_000,
          );
        } catch (error) {
          // A device that answered with an error is the pool's honest answer
          // (e.g. a stage has no copies); only a device that did not answer at
          // all is worth working around.
          if (error instanceof PoolError) throw error;
          lastError = error;
        }
      }
      throw lastError;
    },
    [],
  );

  const setOwnerBusy = useCallback(
    (member: PoolMember, busy: boolean) =>
      getJson(`${memberUrl(member)}/admin/owner`, {
        method: "POST",
        body: JSON.stringify({ busy }),
      }),
    [],
  );

  const labAction = useCallback(
    (name: string, action: "start" | "stop" | "kill") =>
      getJson(
        `${LAB_BASE}/lab/devices/${name}/${action}`,
        { method: "POST" },
        15_000,
      ),
    [],
  );

  return { ...state, generate, setOwnerBusy, labAction, refresh: poll };
}
