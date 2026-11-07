// @vitest-environment node
import { EventEmitter } from "node:events";

import { describe, expect, it, vi } from "vitest";

import { DEFAULT_SETTINGS } from "../shared/types";
import { BackendManager, type BackendDeps } from "./backend";

/** A stopped child that never exits, so stop() must fall back to killing it. */
function fakeChild(pid: number | undefined) {
  const child = new EventEmitter() as EventEmitter & Record<string, unknown>;
  Object.assign(child, { pid, exitCode: null, stdout: new EventEmitter(), stderr: new EventEmitter(), kill: vi.fn() });
  return child;
}

function deps(child: ReturnType<typeof fakeChild>, killGroup = vi.fn()): BackendDeps {
  return {
    health: async () => false,
    leave: async () => {},
    exists: () => true,
    readToken: () => null,
    spawn: vi.fn(() => child) as unknown as BackendDeps["spawn"],
    killGroup,
    platform: "linux",
    resourcesPath: "/r",
    now: () => 0,
  };
}

describe("stopping never signals pid 0 or 1", () => {
  for (const pid of [0, 1, undefined, -5]) {
    it(`pid ${pid}`, async () => {
      vi.useFakeTimers();
      const killGroup = vi.fn();
      const child = fakeChild(pid);
      const manager = new BackendManager(deps(child, killGroup), () => ({ ...DEFAULT_SETTINGS, executable: "/x" }));
      await manager.start({ mode: "public" });
      const stopping = manager.stop();
      await vi.advanceTimersByTimeAsync(11_000);
      await stopping;
      expect(killGroup).not.toHaveBeenCalled();
      vi.useRealTimers();
    });
  }

  it("a real pid is killed as a group, through the injected function", async () => {
    vi.useFakeTimers();
    const killGroup = vi.fn();
    const child = fakeChild(4321);
    const manager = new BackendManager(deps(child, killGroup), () => ({ ...DEFAULT_SETTINGS, executable: "/x" }));
    await manager.start({ mode: "public" });
    const stopping = manager.stop();
    await vi.advanceTimersByTimeAsync(11_000);
    await stopping;
    expect(killGroup).toHaveBeenCalledWith(4321);
    vi.useRealTimers();
  });
});
