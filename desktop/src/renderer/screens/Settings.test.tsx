// @vitest-environment jsdom
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import type { BackendStatus } from "../../shared/types";
import SettingsScreen from "./Settings";
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

afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
  document.documentElement.removeAttribute("data-theme");
});

describe("SettingsScreen", () => {
  it("loads and displays existing values", async () => {
    installBridge({
      settings: {
        label: "Node Prime",
        memoryMb: 4096,
        controlPort: 9005,
        listenPort: 9105,
        bootstrap: ["tcp://192.168.1.10:9100"],
        upnp: false,
        relay: true,
        theme: "dark",
        executable: "/opt/torrent-peer",
      },
    });

    render(<SettingsScreen status={STOPPED} go={vi.fn()} />);

    expect(await screen.findByDisplayValue("Node Prime")).toBeDefined();
    expect(screen.getByDisplayValue("4096")).toBeDefined();
    expect(screen.getByDisplayValue("9005")).toBeDefined();
    expect(screen.getByDisplayValue("9105")).toBeDefined();
    expect(screen.getByDisplayValue("tcp://192.168.1.10:9100")).toBeDefined();
    expect(screen.getByDisplayValue("/opt/torrent-peer")).toBeDefined();

    const upnpCheckbox = screen.getByLabelText(/Ask the router to forward the port/i) as HTMLInputElement;
    expect(upnpCheckbox.checked).toBe(false);

    const relayCheckbox = screen.getByLabelText(/Relay for devices that cannot be reached/i) as HTMLInputElement;
    expect(relayCheckbox.checked).toBe(true);

    const themeSelect = screen.getByLabelText(/Theme/i) as HTMLSelectElement;
    expect(themeSelect.value).toBe("dark");
  });

  it("shows error and does not save if control port or peer port is invalid", async () => {
    const bridge = installBridge();
    render(<SettingsScreen status={STOPPED} go={vi.fn()} />);

    await screen.findByDisplayValue("9000");

    // Invalid control port (< 1024)
    const controlInput = screen.getByLabelText(/Control panel port/i);
    fireEvent.change(controlInput, { target: { value: "80" } });

    const saveBtn = screen.getByRole("button", { name: /Save/i });
    fireEvent.click(saveBtn);

    expect(screen.getByText("Control panel port must be an integer between 1024 and 65535.")).toBeDefined();
    expect(bridge.settings.set).not.toHaveBeenCalled();

    // Fix control port, set invalid peer port (> 65535)
    fireEvent.change(controlInput, { target: { value: "9000" } });
    const peerInput = screen.getByLabelText(/Peer port/i);
    fireEvent.change(peerInput, { target: { value: "70000" } });
    fireEvent.click(saveBtn);

    expect(screen.getByText("Peer port must be an integer between 1024 and 65535.")).toBeDefined();
    expect(bridge.settings.set).not.toHaveBeenCalled();
  });

  it("Finding 4: shows error and does not save when control port equals peer port", async () => {
    const bridge = installBridge();
    render(<SettingsScreen status={STOPPED} go={vi.fn()} />);

    await screen.findByDisplayValue("9000");
    const peerInput = screen.getByLabelText(/Peer port/i);
    fireEvent.change(peerInput, { target: { value: "9000" } });

    const saveBtn = screen.getByRole("button", { name: /Save/i });
    fireEvent.click(saveBtn);

    expect(screen.getByText("Peer port must differ from the control panel port.")).toBeDefined();
    expect(bridge.settings.set).not.toHaveBeenCalled();
  });

  it("Finding 1: bootstrap pattern accepts IPv6 hosts and rejects invalid format or port range", async () => {
    const bridge = installBridge();
    render(<SettingsScreen status={STOPPED} go={vi.fn()} />);

    await screen.findByDisplayValue("9000");
    const bootstrapInput = screen.getByLabelText(/Extra bootstrap addresses/i);
    const saveBtn = screen.getByRole("button", { name: /Save/i });

    // Invalid protocol
    fireEvent.change(bootstrapInput, { target: { value: "http://10.0.0.1:9000" } });
    fireEvent.click(saveBtn);
    expect(screen.getByText(/Invalid bootstrap address/i)).toBeDefined();
    expect(bridge.settings.set).not.toHaveBeenCalled();

    // Invalid port 0
    fireEvent.change(bootstrapInput, { target: { value: "tcp://10.0.0.1:0" } });
    fireEvent.click(saveBtn);
    expect(screen.getByText(/Port out of range in/i)).toBeDefined();
    expect(bridge.settings.set).not.toHaveBeenCalled();

    // Valid IPv6 bootstrap address
    fireEvent.change(bootstrapInput, { target: { value: "tcp://[2001:db8::1]:9100" } });
    fireEvent.click(saveBtn);

    await waitFor(() => {
      expect(bridge.settings.set).toHaveBeenCalledWith(
        expect.objectContaining({
          bootstrap: ["tcp://[2001:db8::1]:9100"],
        }),
      );
    });
  });

  it("valid save calls settings.set and shows Saved indicator", async () => {
    const bridge = installBridge();
    render(<SettingsScreen status={STOPPED} go={vi.fn()} />);

    await screen.findByDisplayValue("9000");

    const nameInput = screen.getByLabelText(/Device name/i);
    fireEvent.change(nameInput, { target: { value: "Updated Device" } });

    const memInput = screen.getByLabelText(/Memory to lend in MB/i);
    fireEvent.change(memInput, { target: { value: "8192" } });

    const saveBtn = screen.getByRole("button", { name: /Save/i });
    fireEvent.click(saveBtn);

    await waitFor(() => {
      expect(bridge.settings.set).toHaveBeenCalledWith(
        expect.objectContaining({
          label: "Updated Device",
          memoryMb: 8192,
        }),
      );
      expect(screen.getByText("Saved.")).toBeDefined();
    });
  });

  it("Finding 5: form submit on Enter triggers save", async () => {
    const bridge = installBridge();
    render(<SettingsScreen status={STOPPED} go={vi.fn()} />);

    const nameInput = await screen.findByLabelText(/Device name/i);
    fireEvent.change(nameInput, { target: { value: "Saved With Enter" } });

    // Submit the form
    fireEvent.submit(nameInput.closest("form")!);

    await waitFor(() => {
      expect(bridge.settings.set).toHaveBeenCalledWith(
        expect.objectContaining({
          label: "Saved With Enter",
        }),
      );
    });
  });

  it("theme change sets data-theme on document.documentElement", async () => {
    installBridge();
    render(<SettingsScreen status={STOPPED} go={vi.fn()} />);

    const themeSelect = (await screen.findByLabelText(/Theme/i)) as HTMLSelectElement;

    fireEvent.change(themeSelect, { target: { value: "dark" } });
    expect(document.documentElement.getAttribute("data-theme")).toBe("dark");

    fireEvent.change(themeSelect, { target: { value: "light" } });
    expect(document.documentElement.getAttribute("data-theme")).toBe("light");

    fireEvent.change(themeSelect, { target: { value: "system" } });
    expect(document.documentElement.getAttribute("data-theme")).toBeNull();
  });

  it("shows note that changes apply on restart only when running", async () => {
    installBridge();
    const { rerender } = render(<SettingsScreen status={RUNNING} go={vi.fn()} />);

    expect(await screen.findByText("Changes apply the next time the device starts.")).toBeDefined();

    rerender(<SettingsScreen status={STOPPED} go={vi.fn()} />);
    expect(screen.queryByText("Changes apply the next time the device starts.")).toBeNull();
  });
});

describe("manager regressions", () => {
  it("shows an error instead of loading forever when settings cannot be read", async () => {
    const bridge = installBridge();
    (bridge.settings.get as ReturnType<typeof vi.fn>).mockRejectedValue(new Error("disk full"));
    render(<SettingsScreen status={RUNNING} go={vi.fn()} />);
    expect(await screen.findByText(/could not read the settings: disk full/i)).toBeTruthy();
  });
});
