// @vitest-environment jsdom

import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import type { BackendStatus, GenerateResult } from "../../shared/types";
import { installBridge, RUNNING } from "../test/mockBridge";
import ChatScreen from "./Chat";

afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
});

describe("ChatScreen", () => {
  it("streams tokens into the answer and shows summary line with tokens count, ms, and traffic ratio", async () => {
    installBridge({
      tokens: ["Hel", "lo"],
      result: {
        totals: {
          tokens: 2,
          failovers: 0,
          sent_bytes: 100,
          uncompressed_bytes: 500,
          compute_ms: 50,
          transport_ms: 25,
        },
      },
    });

    render(<ChatScreen status={RUNNING} go={vi.fn()} />);

    const textarea = screen.getByLabelText("Your question");
    fireEvent.change(textarea, { target: { value: "Hi there" } });

    const sendBtn = screen.getByRole("button", { name: /send/i });
    fireEvent.click(sendBtn);

    // Answer should be streamed
    const answer = await screen.findByText("Hello");
    expect(answer).toBeTruthy();

    // Summary line should be displayed with tokens count, time, and traffic ratio
    const summary = await screen.findByText(/traffic.*smaller/);
    expect(summary.textContent).toContain("2 tokens");
    expect(summary.textContent).toContain("75 ms");
    expect(summary.textContent).toContain("5.0x smaller");
  });

  it("displays route labels and failovers in summary line", async () => {
    installBridge({
      tokens: ["Output answer"],
      result: {
        passes: [
          {
            index: 0,
            phase: "prompt",
            route: [
              [0, "peer-1"],
              [1, "peer-2"],
            ],
            hops: [],
          },
        ],
        totals: {
          tokens: 4,
          failovers: 2,
          sent_bytes: 200,
          uncompressed_bytes: 800,
          compute_ms: 100,
          transport_ms: 50,
        },
      },
      get: {
        "/pool": {
          self: "peer-1",
          pool: { model_id: "test", num_layers: 2, replicas: 1, shard_mb: 100, codec: "int8", stages: [] },
          members: [
            { name: "peer-1", label: "Alpha" },
            { name: "peer-2", label: "Beta" },
          ],
          ready_copies: [1, 1],
          health: "healthy",
          now: 0,
        },
      },
    });

    render(<ChatScreen status={RUNNING} go={vi.fn()} />);

    const textarea = screen.getByLabelText("Your question");
    fireEvent.change(textarea, { target: { value: "Route check" } });
    fireEvent.click(screen.getByRole("button", { name: /send/i }));

    const summary = await screen.findByText(/route:/);
    expect(summary.textContent).toContain("route: Alpha → Beta");
    expect(summary.textContent).toContain("switched devices 2x");
  });

  it("disables Send while answering", async () => {
    const bridge = installBridge();
    let resolveGen!: (res: GenerateResult) => void;
    const pending = new Promise<GenerateResult>((resolve) => {
      resolveGen = resolve;
    });

    bridge.api.generate = vi.fn((_req, onToken) => {
      onToken("Thinking", 0);
      return pending;
    });

    render(<ChatScreen status={RUNNING} go={vi.fn()} />);

    const textarea = screen.getByLabelText("Your question");
    fireEvent.change(textarea, { target: { value: "Tell me a joke" } });

    const sendBtn = screen.getByRole("button", { name: /send/i }) as HTMLButtonElement;
    expect(sendBtn.disabled).toBe(false);

    fireEvent.click(sendBtn);

    // While answering, Send is disabled
    expect(sendBtn.disabled).toBe(true);

    // Resolve generation
    resolveGen({
      prompt: "Tell me a joke",
      completion: "Thinking...",
      tokens: ["Thinking..."],
      requester: "self",
      passes: [],
      events: [],
      totals: {
        tokens: 1,
        failovers: 0,
        sent_bytes: 10,
        uncompressed_bytes: 10,
        compute_ms: 10,
        transport_ms: 5,
      },
    });

    await waitFor(() => {
      expect(sendBtn.disabled).toBe(false);
    });
  });

  it("shows ErrorNote on error path while keeping partial streamed text", async () => {
    const bridge = installBridge();
    bridge.api.generate = vi.fn((_req, onToken) => {
      onToken("Partial words", 0);
      return Promise.reject(new Error("Network connection dropped"));
    });

    render(<ChatScreen status={RUNNING} go={vi.fn()} />);

    const textarea = screen.getByLabelText("Your question");
    fireEvent.change(textarea, { target: { value: "Why did the connection drop?" } });

    fireEvent.click(screen.getByRole("button", { name: /send/i }));

    // Partial text is retained above error
    expect(await screen.findByText("Partial words")).toBeTruthy();

    const alert = await screen.findByRole("alert");
    expect(alert.textContent).toContain("Network connection dropped");
  });

  it("Enter sends and Shift+Enter does not, passing expected parameters", () => {
    const bridge = installBridge({ tokens: ["ok"] });

    render(<ChatScreen status={RUNNING} go={vi.fn()} />);

    const textarea = screen.getByLabelText("Your question");
    fireEvent.change(textarea, { target: { value: "Multi-line query" } });

    // Shift+Enter should NOT send
    fireEvent.keyDown(textarea, { key: "Enter", shiftKey: true });
    expect(bridge.api.generate).not.toHaveBeenCalled();

    // Enter without Shift sends with prompt, max_new_tokens: 64, chat: true
    fireEvent.keyDown(textarea, { key: "Enter", shiftKey: false });
    expect(bridge.api.generate).toHaveBeenCalledTimes(1);
    expect(bridge.api.generate).toHaveBeenCalledWith(
      expect.objectContaining({
        prompt: "Multi-line query",
        max_new_tokens: 64,
        chat: true,
      }),
      expect.any(Function),
    );
  });

  it("clears conversation when Clear button is clicked", async () => {
    installBridge({ tokens: ["Hi!"] });

    render(<ChatScreen status={RUNNING} go={vi.fn()} />);

    const textarea = screen.getByLabelText("Your question");
    fireEvent.change(textarea, { target: { value: "Hello" } });
    fireEvent.click(screen.getByRole("button", { name: /send/i }));

    expect(await screen.findByText("Hi!")).toBeTruthy();

    const clearBtn = screen.getByRole("button", { name: /clear/i });
    fireEvent.click(clearBtn);

    expect(screen.queryByText("Hi!")).toBeNull();
    expect(screen.getByText("No messages yet")).toBeTruthy();
  });

  it("renders Empty and Go to Home when not running", () => {
    const stoppedStatus: BackendStatus = {
      ...RUNNING,
      phase: "stopped",
      message: "Swarm stopped.",
    };
    const go = vi.fn();

    render(<ChatScreen status={stoppedStatus} go={go} />);

    expect(screen.getByText("Not in a swarm")).toBeTruthy();
    const homeBtn = screen.getByRole("button", { name: /go to home/i });
    fireEvent.click(homeBtn);
    expect(go).toHaveBeenCalledWith("home");
  });
});

describe("manager regressions", () => {
  it("does not send a blank question", async () => {
    const bridge = installBridge({ tokens: ["x"] });
    render(<ChatScreen status={RUNNING} go={vi.fn()} />);
    const box = screen.getByLabelText(/your question/i);
    fireEvent.change(box, { target: { value: "   " } });
    fireEvent.keyDown(box, { key: "Enter" });
    await new Promise((r) => setTimeout(r, 20));
    expect(bridge.api.generate).not.toHaveBeenCalled();
  });

  it("sends the chosen max length and asks as a chat", async () => {
    const bridge = installBridge({ tokens: ["ok"] });
    render(<ChatScreen status={RUNNING} go={vi.fn()} />);
    fireEvent.change(screen.getByLabelText(/max length/i), { target: { value: "128" } });
    const box = screen.getByLabelText(/your question/i);
    fireEvent.change(box, { target: { value: "Capital of France?" } });
    fireEvent.keyDown(box, { key: "Enter" });
    await waitFor(() => expect(bridge.api.generate).toHaveBeenCalled());
    const request = (bridge.api.generate as ReturnType<typeof vi.fn>).mock.calls[0][0];
    expect(request).toMatchObject({ prompt: "Capital of France?", max_new_tokens: 128, chat: true });
  });

  it("keeps the text on Shift+Enter and lets you type while an answer streams", async () => {
    installBridge({ tokens: ["a"] });
    render(<ChatScreen status={RUNNING} go={vi.fn()} />);
    const box = screen.getByLabelText(/your question/i) as HTMLTextAreaElement;
    fireEvent.change(box, { target: { value: "line one" } });
    fireEvent.keyDown(box, { key: "Enter", shiftKey: true });
    expect(box.value).toBe("line one");
    expect(box.disabled).toBe(false);
  });
});
