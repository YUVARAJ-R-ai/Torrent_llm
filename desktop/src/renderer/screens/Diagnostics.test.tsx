// @vitest-environment jsdom

import { act, cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import type { BackendStatus } from "../../shared/types";
import { installBridge, RUNNING } from "../test/mockBridge";
import DiagnosticsScreen from "./Diagnostics";

afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
});

describe("DiagnosticsScreen", () => {
  it("shows the logs from backend.logs and merges with live onLog", async () => {
    let logCallback: (line: string) => void = () => {};
    let resolveLogs!: (lines: string[]) => void;
    const logsPromise = new Promise<string[]>((res) => {
      resolveLogs = res;
    });

    const bridge = installBridge();
    bridge.backend.logs = vi.fn(() => logsPromise);
    bridge.backend.onLog = vi.fn((cb) => {
      logCallback = cb;
      return () => {};
    });

    render(<DiagnosticsScreen status={RUNNING} go={vi.fn()} />);

    // Live log arrives before initial backend.logs resolves
    act(() => {
      logCallback("early live line");
    });

    expect(await screen.findByText(/early live line/)).toBeTruthy();

    // Now backend.logs resolves
    await act(async () => {
      resolveLogs(["initial line 1", "initial line 2"]);
    });

    // Both initial and early live lines must be present (not dropped)
    expect(await screen.findByText(/initial line 1/)).toBeTruthy();
    expect(screen.getByText(/initial line 2/)).toBeTruthy();
    expect(screen.getByText(/early live line/)).toBeTruthy();
  });

  it("appends live onLog lines after initial fetch", async () => {
    let logCallback: (line: string) => void = () => {};
    const bridge = installBridge({
      logs: ["initial line"],
    });
    bridge.backend.onLog = vi.fn((cb) => {
      logCallback = cb;
      return () => {};
    });

    render(<DiagnosticsScreen status={RUNNING} go={vi.fn()} />);

    expect(await screen.findByText(/initial line/)).toBeTruthy();

    act(() => {
      logCallback("incoming live log message");
    });

    expect(await screen.findByText(/incoming live log message/)).toBeTruthy();
  });

  it("calls system.copy with joined text and shows Copied note", async () => {
    const bridge = installBridge({
      logs: ["line 1", "line 2", "line 3"],
    });

    render(<DiagnosticsScreen status={RUNNING} go={vi.fn()} />);

    expect(await screen.findByText(/line 1/)).toBeTruthy();

    const copyBtn = screen.getByRole("button", { name: /copy log/i });
    fireEvent.click(copyBtn);

    expect(bridge.system.copy).toHaveBeenCalledWith("line 1\nline 2\nline 3");
    expect(await screen.findByText("Copied")).toBeTruthy();
  });

  it("shows ErrorNote when copy fails", async () => {
    const bridge = installBridge({
      logs: ["some log line"],
    });
    bridge.system.copy = vi.fn().mockRejectedValue(new Error("Clipboard permission denied"));

    render(<DiagnosticsScreen status={RUNNING} go={vi.fn()} />);

    expect(await screen.findByText(/some log line/)).toBeTruthy();

    const copyBtn = screen.getByRole("button", { name: /copy log/i });
    fireEvent.click(copyBtn);

    const alert = await screen.findByRole("alert");
    expect(alert.textContent).toContain("Clipboard permission denied");
  });

  it("renders properly when status.phase is error with Problem pill", () => {
    const errorStatus: BackendStatus = {
      ...RUNNING,
      phase: "error",
      message: "Fatal: port 9100 is already in use by another process.",
      executable: "",
    };

    render(<DiagnosticsScreen status={errorStatus} go={vi.fn()} />);

    // Renders Problem pill (not raw "error")
    expect(screen.getByText("Problem")).toBeTruthy();
    expect(screen.getByText("Fatal: port 9100 is already in use by another process.")).toBeTruthy();
    expect(screen.getByText("not found")).toBeTruthy();
  });

  it("shows troubleshooting tips in Help card", () => {
    render(<DiagnosticsScreen status={RUNNING} go={vi.fn()} />);

    expect(screen.getByText(/Firewall:/)).toBeTruthy();
    expect(screen.getByText(/Hotspot isolation:/)).toBeTruthy();
    expect(screen.getByText(/Model download:/)).toBeTruthy();
    expect(screen.getByText(/Leave and rejoin the swarm:/)).toBeTruthy();
  });
});

describe("withLiveTail", () => {
  it("adds only what the snapshot does not already end with, keeping real repeats", async () => {
    const { withLiveTail } = await import("./Diagnostics");
    expect(withLiveTail(["a", "b"], ["b", "c"])).toEqual(["a", "b", "c"]);
    expect(withLiveTail(["a", "b"], ["c"])).toEqual(["a", "b", "c"]);
    expect(withLiveTail(["x", "x"], ["x", "x", "x"])).toEqual(["x", "x", "x"]);
    expect(withLiveTail([], ["a"])).toEqual(["a"]);
    expect(withLiveTail(["a"], [])).toEqual(["a"]);
  });
});
