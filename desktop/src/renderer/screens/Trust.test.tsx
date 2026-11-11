// @vitest-environment jsdom
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import type { TrustView } from "../../shared/types";
import { installBridge, PEER_A, PEER_B, RUNNING } from "../test/mockBridge";
import TrustScreen from "./Trust";

const PEER_C = "cccccccccccccccccccccccccccccccc";

const MOCK_TRUST: TrustView[] = [
  {
    peer_id: PEER_A,
    label: "Device A",
    checks: 10,
    failures: 0,
    score: 1.0,
    banned: false,
    reason: "",
    served_ms: 1500,
    used_ms: 3000,
  },
  {
    peer_id: PEER_B,
    label: "Device B",
    checks: 8,
    failures: 3,
    score: 0.67,
    banned: true,
    reason: "Bad output",
    served_ms: 500,
    used_ms: 200,
  },
  {
    peer_id: PEER_C,
    label: "Device C",
    checks: 5,
    failures: 4,
    score: 0.2,
    banned: true,
    reason: "",
    served_ms: 100,
    used_ms: 50,
  },
];

describe("TrustScreen", () => {
  afterEach(cleanup);

  it("shows not-running empty state when phase is not running", () => {
    installBridge();
    const go = vi.fn();
    render(<TrustScreen status={{ ...RUNNING, phase: "stopped" }} go={go} />);

    expect(screen.getByText("The device is not running")).toBeTruthy();
    const btn = screen.getByRole("button", { name: /go to home/i });
    fireEvent.click(btn);
    expect(go).toHaveBeenCalledWith("home");
  });

  it("renders rows with mocked data and checks empty reason renders 'Banned' without colon (Finding 3)", async () => {
    installBridge({
      get: {
        "/trust": MOCK_TRUST,
      },
    });

    render(<TrustScreen status={RUNNING} go={vi.fn()} />);

    expect(await screen.findByText("Device A")).toBeTruthy();
    expect(screen.getByText("100%")).toBeTruthy();
    expect(screen.getByText("1.50 s / 3.00 s")).toBeTruthy();
    expect(screen.getByText("Trusted")).toBeTruthy();

    expect(screen.getByText("Device B")).toBeTruthy();
    expect(screen.getByText("67%")).toBeTruthy();
    expect(screen.getByText("500 ms / 200 ms")).toBeTruthy();
    expect(screen.getByText("Banned: Bad output")).toBeTruthy();

    // Finding 3: empty banned reason renders 'Banned' without trailing colon
    expect(screen.getByText("Device C")).toBeTruthy();
    expect(screen.getByText("Banned")).toBeTruthy();
    expect(screen.queryByText("Banned:")).toBeNull();
  });

  it("shows empty state when data is empty", async () => {
    installBridge({
      get: {
        "/trust": [],
      },
    });

    render(<TrustScreen status={RUNNING} go={vi.fn()} />);

    expect(await screen.findByText("No checks yet.")).toBeTruthy();
  });

  it("shows error and does NOT show 'No checks yet.' when trust fails on initial load (Finding 2)", async () => {
    installBridge({
      get: {
        "/trust": () => {
          throw new Error("Trust store unavailable");
        },
      },
    });

    render(<TrustScreen status={RUNNING} go={vi.fn()} />);

    expect(await screen.findByText("Trust store unavailable")).toBeTruthy();
    expect(screen.queryByText("No checks yet.")).toBeNull();
  });

  it("handles ban flow with Enter key and posts to /trust/ban (Finding 3)", async () => {
    const postMock = vi.fn().mockResolvedValue({ banned: true });
    installBridge({
      get: {
        "/trust": [MOCK_TRUST[0]], // Device A (unbanned)
      },
      post: {
        "/trust/ban": postMock,
      },
    });

    render(<TrustScreen status={RUNNING} go={vi.fn()} />);

    expect(await screen.findByText("Device A")).toBeTruthy();

    // Click Ban to open inline form
    const banBtn = screen.getByRole("button", { name: /^ban$/i });
    fireEvent.click(banBtn);

    const input = screen.getByPlaceholderText(/Reason \(1-200 chars\)/i);
    expect(input).toBeTruthy();

    // Type reason and press Enter (Finding 3 keyboard accessibility)
    fireEvent.change(input, { target: { value: "Repeated verification failures" } });
    fireEvent.keyDown(input, { key: "Enter" });

    await waitFor(() => {
      expect(postMock).toHaveBeenCalledWith({
        peer_id: PEER_A,
        reason: "Repeated verification failures",
      });
    });
  });

  it("rejects ban without reason and does not post", async () => {
    const postMock = vi.fn();
    installBridge({
      get: {
        "/trust": [MOCK_TRUST[0]], // Device A
      },
      post: {
        "/trust/ban": postMock,
      },
    });

    render(<TrustScreen status={RUNNING} go={vi.fn()} />);

    expect(await screen.findByText("Device A")).toBeTruthy();

    // Click Ban to open input
    const banBtn = screen.getByRole("button", { name: /^ban$/i });
    fireEvent.click(banBtn);

    // Click submit Ban without entering reason
    const submitBtn = screen.getByRole("button", { name: /^ban$/i });
    fireEvent.click(submitBtn);

    expect(await screen.findByText(/Reason is required/i)).toBeTruthy();
    expect(postMock).not.toHaveBeenCalled();
  });

  it("handles unban flow and posts to /trust/unban", async () => {
    const postMock = vi.fn().mockResolvedValue({ banned: false });
    installBridge({
      get: {
        "/trust": [MOCK_TRUST[1]], // Device B (banned)
      },
      post: {
        "/trust/unban": postMock,
      },
    });

    render(<TrustScreen status={RUNNING} go={vi.fn()} />);

    expect(await screen.findByText("Device B")).toBeTruthy();

    const unbanBtn = screen.getByRole("button", { name: /^unban$/i });
    fireEvent.click(unbanBtn);

    await waitFor(() => {
      expect(postMock).toHaveBeenCalledWith({
        peer_id: PEER_B,
      });
    });
  });
});
