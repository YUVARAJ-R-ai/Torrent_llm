/**
 * A fake window.tllm for renderer tests. Every screen test installs one with the
 * data it needs; calls are recorded so tests can assert what the screen asked for.
 */

import { vi } from "vitest";

import {
  type AppSettings,
  type BackendStatus,
  DEFAULT_SETTINGS,
  type GenerateResult,
  type TllmBridge,
} from "../../shared/types";

export const RUNNING: BackendStatus = {
  phase: "running",
  message: "Running.",
  executable: "/x/torrent-peer",
  url: "http://127.0.0.1:9000",
  pid: 42,
  mode: "create",
  startedAt: 0,
};

export const PEER_A = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
export const PEER_B = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";

export interface MockOptions {
  status?: BackendStatus;
  /** Responses for api.get, by path. A function may throw to simulate an error. */
  get?: Record<string, unknown | (() => unknown)>;
  post?: Record<string, unknown | ((body: unknown) => unknown)>;
  settings?: Partial<AppSettings>;
  /** Tokens api.generate streams before resolving with ``result``. */
  tokens?: string[];
  result?: Partial<GenerateResult>;
  logs?: string[];
}

export function installBridge(options: MockOptions = {}) {
  let settings: AppSettings = { ...DEFAULT_SETTINGS, ...options.settings };
  const status = options.status ?? RUNNING;
  const resolve = (table: Record<string, unknown> | undefined, key: string, arg?: unknown) => {
    if (!table || !(key in table)) return Promise.reject(new Error(`no mock for ${key}`));
    const value = table[key];
    try {
      return Promise.resolve(typeof value === "function" ? (value as (a: unknown) => unknown)(arg) : value);
    } catch (e) {
      return Promise.reject(e);
    }
  };
  const bridge: TllmBridge = {
    backend: {
      status: vi.fn(async () => status),
      start: vi.fn(async () => status),
      stop: vi.fn(async () => ({ ...status, phase: "stopped" as const })),
      install: vi.fn(async () => status),
      logs: vi.fn(async () => options.logs ?? []),
      onStatus: vi.fn(() => () => {}),
      onLog: vi.fn(() => () => {}),
    },
    api: {
      get: vi.fn((path: string) => resolve(options.get, path)) as TllmBridge["api"]["get"],
      post: vi.fn((path: string, body?: unknown) => resolve(options.post, path, body)) as TllmBridge["api"]["post"],
      generate: vi.fn(async (_req, onToken) => {
        (options.tokens ?? []).forEach((t, i) => onToken(t, i));
        return {
          prompt: "",
          completion: (options.tokens ?? []).join(""),
          tokens: options.tokens ?? [],
          requester: PEER_A,
          passes: [],
          events: [],
          totals: { tokens: 0, failovers: 0, sent_bytes: 0, uncompressed_bytes: 0, compute_ms: 0, transport_ms: 0 },
          ...options.result,
        } as GenerateResult;
      }),
    },
    settings: {
      get: vi.fn(async () => settings),
      set: vi.fn(async (patch: Partial<AppSettings>) => {
        settings = { ...settings, ...patch };
        return settings;
      }),
    },
    system: {
      openExternal: vi.fn(async () => {}),
      copy: vi.fn(async () => {}),
      platform: "linux",
    },
  };
  (window as unknown as { tllm: TllmBridge }).tllm = bridge;
  return bridge;
}
