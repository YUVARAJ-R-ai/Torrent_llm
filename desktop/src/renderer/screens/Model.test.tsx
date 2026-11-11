// @vitest-environment jsdom
import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import type { PoolView } from "../../shared/types";
import { installBridge, PEER_A, PEER_B, RUNNING } from "../test/mockBridge";
import ModelScreen from "./Model";

const MOCK_POOL: PoolView = {
  self: PEER_A,
  pool: {
    model_id: "llama-3-8b",
    num_layers: 32,
    replicas: 2,
    shard_mb: 2048,
    codec: "q4_k",
    stages: [
      { stage: 0, start: 0, end: 16 },
      { stage: 1, start: 16, end: 32 },
    ],
  },
  members: [
    {
      name: PEER_A,
      label: "Device A",
      os: "linux",
      hostname: "host-a",
      device: "cuda",
      memory_mb: 8192,
      slots: 1,
      speed: 1,
      state: "active",
      hosting: [{ stage: 0, port: 9001, status: "ready", inflight: 0, served: 10, busy_ms: 100 }],
      asked: 0,
      asked_tokens: 0,
      addrs: [],
      liveness: "alive",
      is_self: true,
      rtt_ms: 15,
    },
    {
      name: PEER_B,
      label: "Device B",
      os: "macos",
      hostname: "host-b",
      device: "m3",
      memory_mb: 4096,
      slots: 1,
      speed: 1,
      state: "active",
      hosting: [{ stage: 0, port: 9002, status: "loading", inflight: 0, served: 0, busy_ms: 0 }],
      asked: 0,
      asked_tokens: 0,
      addrs: [],
      liveness: "alive",
      is_self: false,
      rtt_ms: 40,
    },
  ],
  ready_copies: [1, 0],
  health: "degraded",
  now: 1000,
};

describe("ModelScreen", () => {
  afterEach(cleanup);

  it("shows not-running empty state when phase is not running", () => {
    installBridge();
    const go = vi.fn();
    render(<ModelScreen status={{ ...RUNNING, phase: "stopped" }} go={go} />);

    expect(screen.getByText("The device is not running")).toBeTruthy();
    const btn = screen.getByRole("button", { name: /go to home/i });
    fireEvent.click(btn);
    expect(go).toHaveBeenCalledWith("home");
  });

  it("renders model details, health, codec and stage chips with correct styles (Finding 1)", async () => {
    installBridge({
      get: {
        "/pool": MOCK_POOL,
      },
    });

    render(<ModelScreen status={RUNNING} go={vi.fn()} />);

    expect(await screen.findByText("llama-3-8b")).toBeTruthy();
    expect(screen.getByText("Degraded: some stage has only one copy")).toBeTruthy();
    expect(screen.getByText("Wire format: q4_k")).toBeTruthy();

    expect(screen.getByText("Stage 1: layers 0-16")).toBeTruthy();
    expect(screen.getByText("Stage 2: layers 16-32")).toBeTruthy();
    expect(screen.getByText("1 of 2 ready")).toBeTruthy();
    expect(screen.getByText("0 of 2 ready")).toBeTruthy();

    // Finding 1: Assert that a ready chip is filled with var(--series-1) and a loading chip is outlined (transparent)
    const readyChip = screen.getByText("Device A");
    expect(readyChip.style.background).toBe("var(--series-1)");

    const loadingChip = screen.getByText("Device B");
    expect(["transparent", "none"]).toContain(loadingChip.style.background);
    expect(loadingChip.style.backgroundColor).toBe("transparent");
  });

  it("shows em dash in subtitle when model_id is missing (Finding 4)", async () => {
    installBridge({
      get: {
        "/pool": {
          ...MOCK_POOL,
          pool: {
            ...MOCK_POOL.pool,
            model_id: "",
          },
        },
      },
    });

    render(<ModelScreen status={RUNNING} go={vi.fn()} />);

    expect(await screen.findByText("—")).toBeTruthy();
  });

  it("shows error note on failing poll", async () => {
    installBridge({
      get: {
        "/pool": () => {
          throw new Error("Failed to load model info");
        },
      },
    });

    render(<ModelScreen status={RUNNING} go={vi.fn()} />);

    expect(await screen.findByText("Failed to load model info")).toBeTruthy();
  });
});

describe("manager regressions", () => {
  it("counts only devices that are alive and says why a chip is not ready", async () => {
    const pool = {
      self: PEER_A,
      pool: { model_id: "m", num_layers: 4, replicas: 2, shard_mb: 1, codec: "bf16", stages: [{ stage: 0, start: 0, end: 4 }] },
      members: [
        { name: PEER_A, label: "alpha", os: "", hostname: "", device: "cpu", memory_mb: 1, slots: 1, speed: 1, state: "active", hosting: [{ stage: 0, port: 0, status: "ready", inflight: 0, served: 0, busy_ms: 0 }], asked: 0, asked_tokens: 0, addrs: [], liveness: "alive", is_self: true, rtt_ms: null },
        { name: PEER_B, label: "beta", os: "", hostname: "", device: "cpu", memory_mb: 1, slots: 1, speed: 1, state: "active", hosting: [{ stage: 0, port: 0, status: "ready", inflight: 0, served: 0, busy_ms: 0 }], asked: 0, asked_tokens: 0, addrs: [], liveness: "down", is_self: false, rtt_ms: null },
      ],
      ready_copies: [1],
      health: "degraded",
      now: 0,
    };
    installBridge({ get: { "/pool": pool } });
    render(<ModelScreen status={RUNNING} go={vi.fn()} />);
    expect(await screen.findByText(/1 of 2 ready/)).toBeTruthy();
    expect(screen.getByText(/\(offline\)/)).toBeTruthy();
  });
});
