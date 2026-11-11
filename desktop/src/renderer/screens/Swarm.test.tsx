// @vitest-environment jsdom
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import type { BackendStatus, InviteView, NodeView } from "../../shared/types";
import SwarmScreen from "./Swarm";
import { installBridge, RUNNING } from "../test/mockBridge";

const STOPPED: BackendStatus = {
  phase: "stopped",
  message: "Stopped.",
  executable: "/x/torrent-peer",
  url: "http://127.0.0.1:9000",
  pid: null,
  mode: null,
  startedAt: null,
};

const MISSING: BackendStatus = {
  ...STOPPED,
  phase: "missing",
  executable: "",
};

const MOCK_NODE: NodeView = {
  peer_id: "0123456789abcdef0123456789abcdef",
  label: "Test Node",
  swarm_id: "swarm-uuid",
  swarm_name: "Community Swarm",
  public: false,
  addresses: ["192.168.1.50:9100"],
  reachability: "public",
  mapped: "192.168.1.50:9100",
  relays: ["relay1"],
  serving_relay: true,
  connections: 3,
  version: "0.1.0",
};

const MOCK_INVITE: InviteView = {
  link: "tllm://join/secret-token-1234",
  commands: {
    linux_macos: "curl -sL https://example.com/install.sh | sh",
    windows: "irm https://example.com/install.ps1 | iex",
    installed: "torrent-peer join tllm://join/secret-token-1234",
  },
};

afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
});

describe("SwarmScreen", () => {
  it("renders three choices when not running", () => {
    installBridge({ status: STOPPED });
    const go = vi.fn();
    render(<SwarmScreen status={STOPPED} go={go} />);

    expect(screen.getByText("Create a swarm")).toBeDefined();
    expect(screen.getByText("Join with a link")).toBeDefined();
    expect(screen.getByText("Public swarm")).toBeDefined();
    expect(screen.getByRole("button", { name: /Create and start/i })).toBeDefined();
    expect(screen.getByRole("button", { name: /^Join$/i })).toBeDefined();
    expect(screen.getByRole("button", { name: /Join the public swarm/i })).toBeDefined();
  });

  it("shows install hint when phase is missing", () => {
    installBridge({ status: MISSING });
    const go = vi.fn();
    render(<SwarmScreen status={MISSING} go={go} />);

    expect(screen.getByText("Torrent-peer is not installed")).toBeDefined();
    expect(screen.queryByText("Create a swarm")).toBeNull();

    const homeBtn = screen.getByRole("button", { name: /Go to Home/i });
    fireEvent.click(homeBtn);
    expect(go).toHaveBeenCalledWith("home");
  });

  it("creates a swarm with custom name and navigates home", async () => {
    const bridge = installBridge({ status: STOPPED });
    const go = vi.fn();
    render(<SwarmScreen status={STOPPED} go={go} />);

    const input = screen.getByLabelText(/Swarm name/i);
    fireEvent.change(input, { target: { value: "Super Swarm" } });

    const createBtn = screen.getByRole("button", { name: /Create and start/i });
    fireEvent.click(createBtn);

    await waitFor(() => {
      expect(bridge.backend.start).toHaveBeenCalledWith({
        mode: "create",
        swarmName: "Super Swarm",
      });
      expect(go).toHaveBeenCalledWith("home");
    });
  });

  it("validates invite link client-side before joining", async () => {
    const bridge = installBridge({ status: STOPPED });
    const go = vi.fn();
    render(<SwarmScreen status={STOPPED} go={go} />);

    const textarea = screen.getByLabelText(/Invite link/i);
    const joinBtn = screen.getByRole("button", { name: /^Join$/i });

    // Invalid link
    fireEvent.change(textarea, { target: { value: "https://example.com/not-invite" } });
    fireEvent.click(joinBtn);

    expect(screen.getByText("That is not an invite link. It starts with tllm://join/")).toBeDefined();
    expect(bridge.backend.start).not.toHaveBeenCalled();

    // Valid link
    fireEvent.change(textarea, { target: { value: "   tllm://join/secret123   " } });
    fireEvent.click(joinBtn);

    await waitFor(() => {
      expect(bridge.backend.start).toHaveBeenCalledWith({
        mode: "join",
        invite: "tllm://join/secret123",
      });
      expect(go).toHaveBeenCalledWith("home");
    });
  });

  it("joins the public swarm and navigates home", async () => {
    const bridge = installBridge({ status: STOPPED });
    const go = vi.fn();
    render(<SwarmScreen status={STOPPED} go={go} />);

    const publicBtn = screen.getByRole("button", { name: /Join the public swarm/i });
    fireEvent.click(publicBtn);

    await waitFor(() => {
      expect(bridge.backend.start).toHaveBeenCalledWith({ mode: "public" });
      expect(go).toHaveBeenCalledWith("home");
    });
  });

  it("when running: does NOT fetch invite until 'Show invite' is clicked, then shows secret, QR, copy", async () => {
    const bridge = installBridge({
      status: RUNNING,
      get: {
        "/node": MOCK_NODE,
        "/swarm/invite": MOCK_INVITE,
      },
    });
    const go = vi.fn();
    render(<SwarmScreen status={RUNNING} go={go} />);

    // Swarm info card displayed
    await screen.findByText("Community Swarm");
    expect(screen.getByText(/Test Node/)).toBeDefined();
    expect(screen.getByText("Reachable from the internet")).toBeDefined();
    expect(screen.getByText("This device relays for others")).toBeDefined();

    // Invite not fetched yet
    expect(bridge.api.get).not.toHaveBeenCalledWith("/swarm/invite");
    const showInviteBtn = screen.getByRole("button", { name: /Show invite/i });
    expect(showInviteBtn).toBeDefined();

    // Click Show invite
    fireEvent.click(showInviteBtn);

    await waitFor(() => {
      expect(bridge.api.get).toHaveBeenCalledWith("/swarm/invite");
    });

    // Link textarea and QR code appear
    const textarea = await screen.findByDisplayValue("tllm://join/secret-token-1234");
    expect(textarea).toBeDefined();
    const qrImg = await screen.findByAltText("QR code of the invite link");
    expect(qrImg).toBeDefined();

    // Copy link
    const copyBtn = screen.getByRole("button", { name: /Copy link/i });
    fireEvent.click(copyBtn);
    expect(bridge.system.copy).toHaveBeenCalledWith("tllm://join/secret-token-1234");

    // Copy install command
    const copyCmdBtns = screen.getAllByRole("button", { name: /^Copy$/i });
    expect(copyCmdBtns.length).toBe(3);
    fireEvent.click(copyCmdBtns[0]);
    expect(bridge.system.copy).toHaveBeenCalledWith(MOCK_INVITE.commands.linux_macos);

    // Hide button
    const hideBtn = screen.getByRole("button", { name: /Hide/i });
    fireEvent.click(hideBtn);
    expect(screen.queryByDisplayValue("tllm://join/secret-token-1234")).toBeNull();
    expect(screen.getByRole("button", { name: /Show invite/i })).toBeDefined();
  });

  it("handles public swarm error fallback when invite endpoint returns public error", async () => {
    installBridge({
      status: RUNNING,
      get: {
        "/node": { ...MOCK_NODE, public: false },
        "/swarm/invite": () => {
          throw new Error("This is a public swarm; no invite exists");
        },
      },
    });
    render(<SwarmScreen status={RUNNING} go={vi.fn()} />);

    const showInviteBtn = await screen.findByRole("button", { name: /Show invite/i });
    fireEvent.click(showInviteBtn);

    await screen.findByText("The public swarm has no invite: anyone can join with torrent-peer public.");
  });

  it("Finding 2: renders public swarm note directly when node.data.public is true", async () => {
    installBridge({
      status: RUNNING,
      get: {
        "/node": { ...MOCK_NODE, public: true, swarm_name: "" },
      },
    });
    render(<SwarmScreen status={RUNNING} go={vi.fn()} />);

    await screen.findByText("The public swarm has no invite: anyone can join with torrent-peer public.");
    expect(screen.queryByRole("button", { name: /Show invite/i })).toBeNull();
  });

  it("allows leaving the swarm", async () => {
    const bridge = installBridge({
      status: RUNNING,
      get: { "/node": MOCK_NODE },
    });
    render(<SwarmScreen status={RUNNING} go={vi.fn()} />);

    const leaveBtn = await screen.findByRole("button", { name: /Leave swarm/i });
    fireEvent.click(leaveBtn);

    await waitFor(() => {
      expect(bridge.backend.stop).toHaveBeenCalled();
    });
  });

  it("Finding 3: cleans up timers and invite data on unmount", async () => {
    installBridge({
      status: RUNNING,
      get: {
        "/node": MOCK_NODE,
        "/swarm/invite": MOCK_INVITE,
      },
    });
    const { unmount } = render(<SwarmScreen status={RUNNING} go={vi.fn()} />);
    const showInviteBtn = await screen.findByRole("button", { name: /Show invite/i });
    fireEvent.click(showInviteBtn);
    await screen.findByDisplayValue("tllm://join/secret-token-1234");

    // Unmounting must not throw or cause React warnings
    unmount();
  });
});

describe("manager regressions", () => {
  it("does not offer choices while the device is starting, installing or stopping", async () => {
    for (const phase of ["starting", "installing", "stopping"] as const) {
      installBridge();
      const { unmount } = render(<SwarmScreen status={{ ...RUNNING, phase }} go={vi.fn()} />);
      expect(screen.queryByRole("button", { name: /create and start/i })).toBeNull();
      expect(screen.queryByRole("button", { name: /^join$/i })).toBeNull();
      unmount();
    }
  });

  it("refuses an empty swarm name", async () => {
    const bridge = installBridge({ status: { ...RUNNING, phase: "stopped" } });
    render(<SwarmScreen status={{ ...RUNNING, phase: "stopped" }} go={vi.fn()} />);
    const input = screen.getByLabelText(/swarm name/i);
    fireEvent.change(input, { target: { value: "   " } });
    fireEvent.click(screen.getByRole("button", { name: /create and start/i }));
    expect(await screen.findByText(/give the swarm a name/i)).toBeTruthy();
    expect(bridge.backend.start).not.toHaveBeenCalled();
  });

  it("forgets a shown invite when the device leaves the swarm", async () => {
    installBridge({
      get: {
        "/swarm/invite": { link: "tllm://join/SECRET", commands: { linux_macos: "a", windows: "b", installed: "c" } },
        "/node": { peer_id: "x", label: "me", swarm_id: "s", swarm_name: "S", public: false, addresses: [], reachability: "unknown", mapped: "", relays: [], serving_relay: false, connections: 0, version: "0" },
      },
    });
    const { rerender } = render(<SwarmScreen status={RUNNING} go={vi.fn()} />);
    fireEvent.click(await screen.findByRole("button", { name: /show invite/i }));
    expect((await screen.findAllByDisplayValue(/tllm:\/\/join\/SECRET/)).length).toBeGreaterThan(0);
    rerender(<SwarmScreen status={{ ...RUNNING, phase: "stopped" }} go={vi.fn()} />);
    rerender(<SwarmScreen status={RUNNING} go={vi.fn()} />);
    await waitFor(() => expect(screen.queryByDisplayValue(/SECRET/)).toBeNull());
  });

  it("says so when copying fails instead of claiming it worked", async () => {
    const bridge = installBridge({
      get: {
        "/swarm/invite": { link: "tllm://join/SECRET", commands: { linux_macos: "a", windows: "b", installed: "c" } },
        "/node": { peer_id: "x", label: "me", swarm_id: "s", swarm_name: "S", public: false, addresses: [], reachability: "unknown", mapped: "", relays: [], serving_relay: false, connections: 0, version: "0" },
      },
    });
    (bridge.system.copy as ReturnType<typeof vi.fn>).mockRejectedValue(new Error("no clipboard"));
    render(<SwarmScreen status={RUNNING} go={vi.fn()} />);
    fireEvent.click(await screen.findByRole("button", { name: /show invite/i }));
    fireEvent.click((await screen.findAllByRole("button", { name: /copy link/i }))[0]);
    expect(await screen.findByText(/could not copy/i)).toBeTruthy();
  });
});
