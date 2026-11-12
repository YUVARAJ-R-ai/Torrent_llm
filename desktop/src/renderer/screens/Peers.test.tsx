// @vitest-environment jsdom
import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import type { ConnectionView, PoolView } from "../../shared/types";
import { installBridge, PEER_A, PEER_B, RUNNING } from "../test/mockBridge";
import PeersScreen from "./Peers";

const MOCK_POOL: PoolView = {
  self: PEER_A,
  pool: {
    model_id: "test-model-7b",
    num_layers: 32,
    replicas: 2,
    shard_mb: 2000,
    codec: "fp16",
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
      state: "away",
      hosting: [{ stage: 1, port: 9002, status: "loading", inflight: 0, served: 0, busy_ms: 0 }],
      asked: 0,
      asked_tokens: 0,
      addrs: [],
      liveness: "suspect",
      is_self: false,
      rtt_ms: 50,
    },
  ],
  ready_copies: [1, 0],
  health: "degraded",
  now: 1000,
};

const RELAY_PEER = "cccccccccccccccccccccccccccccccc";

const MOCK_CONNS: ConnectionView[] = [
  {
    peer_id: PEER_B,
    label: "Device B",
    remote: "192.168.1.2:9100",
    inbound: true,
    via_relay: null,
    rtt_ms: 50,
    bytes_sent: 1048576,
    bytes_received: 2097152,
    bandwidth_bps: 1250000,
  },
  {
    peer_id: RELAY_PEER,
    label: "Device C",
    remote: "10.0.0.1:9100",
    inbound: false,
    via_relay: "dddddddddddddddddddddddddddddddd",
    rtt_ms: 120,
    bytes_sent: 50000,
    bytes_received: 75000,
    bandwidth_bps: 500000,
  },
];

describe("PeersScreen", () => {
  afterEach(cleanup);

  it("shows not-running empty state when phase is not running", () => {
    installBridge();
    const go = vi.fn();
    render(<PeersScreen status={{ ...RUNNING, phase: "stopped" }} go={go} />);

    expect(screen.getByText("The device is not running")).toBeTruthy();
    const btn = screen.getByRole("button", { name: /go to home/i });
    fireEvent.click(btn);
    expect(go).toHaveBeenCalledWith("home");
  });

  it("renders members table and connections table with mocked data", async () => {
    installBridge({
      get: {
        "/pool": MOCK_POOL,
        "/peers": MOCK_CONNS,
      },
    });

    render(<PeersScreen status={RUNNING} go={vi.fn()} />);

    // Check members
    expect(await screen.findByText(/Device A/i)).toBeTruthy();
    expect(screen.getByText("(this device)")).toBeTruthy();
    expect(screen.getByText(PEER_A.slice(0, 8))).toBeTruthy();
    expect(screen.getByText("cuda · linux")).toBeTruthy();
    expect(screen.getByText("Online")).toBeTruthy();
    expect(screen.getByText("8192 MB")).toBeTruthy();

    const deviceBElements = screen.getAllByText(/Device B/i);
    expect(deviceBElements.length).toBe(2);
    const peerBIdElements = screen.getAllByText(PEER_B.slice(0, 8));
    expect(peerBIdElements.length).toBe(2);
    expect(screen.getByText("m3 · macos")).toBeTruthy();
    expect(screen.getByText("2 (loading)")).toBeTruthy();
    expect(screen.getByText("Not answering")).toBeTruthy();
    expect(screen.getByText(/· owner using it/i)).toBeTruthy();
    expect(screen.getByText("4096 MB")).toBeTruthy();

    // Check connections, direction and via relay (Finding 1)
    expect(screen.getByText("Inbound")).toBeTruthy();
    expect(screen.getByText("Outbound")).toBeTruthy();
    expect(screen.getByText("direct")).toBeTruthy();
    expect(screen.getByText("via relay dddddddd")).toBeTruthy();
  });

  it("shows empty state when no other members are in swarm", async () => {
    installBridge({
      get: {
        "/pool": {
          ...MOCK_POOL,
          members: [MOCK_POOL.members[0]], // only self
        },
        "/peers": [],
      },
    });

    render(<PeersScreen status={RUNNING} go={vi.fn()} />);

    expect(await screen.findByText("Nobody else yet.")).toBeTruthy();
    expect(screen.getByText("Share the invite from the Swarm screen.")).toBeTruthy();
    expect(screen.getByText("No active connections.")).toBeTruthy();
  });

  it("shows error and keeps last data on poll error", async () => {
    let callCount = 0;
    installBridge({
      get: {
        "/pool": () => {
          callCount++;
          if (callCount > 1) throw new Error("Pool network failure");
          return MOCK_POOL;
        },
        "/peers": MOCK_CONNS,
      },
    });

    render(<PeersScreen status={RUNNING} go={vi.fn()} />);

    expect(await screen.findByText(/Device A/i)).toBeTruthy();
  });

  it("shows error and does NOT show 'No active connections.' when peers fails on initial load (Finding 2)", async () => {
    installBridge({
      get: {
        "/pool": MOCK_POOL,
        "/peers": () => {
          throw new Error("Failed to fetch connections");
        },
      },
    });

    render(<PeersScreen status={RUNNING} go={vi.fn()} />);

    expect(await screen.findByText("Failed to fetch connections")).toBeTruthy();
    expect(screen.queryByText("No active connections.")).toBeNull();
  });
});
