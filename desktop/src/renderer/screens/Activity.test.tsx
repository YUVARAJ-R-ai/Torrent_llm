// @vitest-environment jsdom

import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import type { BackendStatus, PoolEvent } from "../../shared/types";
import { installBridge, RUNNING } from "../test/mockBridge";
import ActivityScreen from "./Activity";

afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
  vi.useRealTimers();
});

describe("ActivityScreen", () => {
  it("renders events newest first and dedupes repeated ones across polls", async () => {
    vi.useFakeTimers();

    const now = Math.floor(Date.now() / 1000);
    const mockEvents: PoolEvent[] = [
      { t: now - 30, kind: "joined", text: "Device alpha joined the swarm", member: "p1" },
      { t: now - 5, kind: "stage", text: "Stage 2 handover complete", member: "p2" },
      { t: now - 15, kind: "down", text: "Device beta unreachable", member: "p3" },
      { t: now - 10, kind: "state", text: "Device gamma draining", member: "p4" },
    ];

    const bridge = installBridge({
      get: {
        "/events": () => [...mockEvents],
      },
    });

    render(<ActivityScreen status={RUNNING} go={vi.fn()} />);

    // Wait for initial poll to render
    await vi.waitFor(() => {
      expect(screen.getByText("Stage 2 handover complete")).toBeTruthy();
    });

    // Check first poll was called with since: 0
    expect(bridge.api.get).toHaveBeenNthCalledWith(1, "/events", { since: 0 });

    // Verify all 4 event texts are in exact newest-first order
    const eventTexts = screen.getAllByText(/^(Device|Stage) .*/);
    expect(eventTexts.map((el) => el.textContent)).toEqual([
      "Stage 2 handover complete",
      "Device gamma draining",
      "Device beta unreachable",
      "Device alpha joined the swarm",
    ]);

    // Second poll fires at 2s interval with the same list
    await vi.advanceTimersByTimeAsync(2000);
    await vi.waitFor(() => {
      expect(bridge.api.get).toHaveBeenCalledTimes(2);
    });

    // Second poll was called with since: max t (now - 5)
    expect(bridge.api.get).toHaveBeenNthCalledWith(2, "/events", { since: now - 5 });

    // Events should be deduped (only 1 occurrence each, not duplicated)
    expect(screen.getAllByText("Device alpha joined the swarm").length).toBe(1);
    expect(screen.getAllByText("Stage 2 handover complete").length).toBe(1);
  });

  it("caps events at the newest 200 items", async () => {
    const events: PoolEvent[] = Array.from({ length: 250 }, (_, i) => ({
      t: 1000 + i,
      kind: "state",
      text: `Event number ${i}`,
      member: null,
    }));

    installBridge({
      get: {
        "/events": () => [...events],
      },
    });

    render(<ActivityScreen status={RUNNING} go={vi.fn()} />);

    expect(await screen.findByText("Event number 249")).toBeTruthy();
    const rows = screen.getAllByText(/Event number \d+/);
    expect(rows.length).toBe(200);

    // Oldest 50 events are dropped
    expect(screen.queryByText("Event number 0")).toBeNull();
    expect(screen.queryByText("Event number 49")).toBeNull();
    // 50th through 249th events are kept
    expect(screen.getByText("Event number 50")).toBeTruthy();
  });

  it("renders Empty when no events exist", async () => {
    installBridge({
      get: {
        "/events": [],
      },
    });

    render(<ActivityScreen status={RUNNING} go={vi.fn()} />);

    expect(await screen.findByText("Nothing has happened yet.")).toBeTruthy();
    expect(
      screen.getByText("Devices joining, leaving or taking over a stage will show up here."),
    ).toBeTruthy();
  });

  it("renders Empty and Go to Home when not running", () => {
    const stoppedStatus: BackendStatus = {
      ...RUNNING,
      phase: "stopped",
      message: "Stopped.",
    };
    const go = vi.fn();

    render(<ActivityScreen status={stoppedStatus} go={go} />);

    expect(screen.getByText("Not in a swarm")).toBeTruthy();
    const btn = screen.getByRole("button", { name: /go to home/i });
    fireEvent.click(btn);
    expect(go).toHaveBeenCalledWith("home");
  });
});
