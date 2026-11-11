// @vitest-environment node
import { EventEmitter } from "node:events";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";

import { afterEach, describe, expect, it, vi } from "vitest";

import { type AppSettings, DEFAULT_SETTINGS, type StartOptions } from "../shared/types";
import { BackendManager, buildArgs, redact, realDeps, type BackendDeps } from "./backend";
import { installedExecutable } from "./paths";

function fakeChild(pid?: number) {
  const assignedPid = arguments.length === 0 ? 4321 : pid;
  const child = new EventEmitter() as EventEmitter & Record<string, unknown>;
  Object.assign(child, {
    pid: assignedPid,
    exitCode: null,
    stdout: new EventEmitter(),
    stderr: new EventEmitter(),
    kill: vi.fn(),
  });
  child.on("exit", (code) => {
    child.exitCode = code ?? 0;
  });
  return child;
}

function makeDeps(child: ReturnType<typeof fakeChild>, overrides: Partial<BackendDeps> = {}): BackendDeps {
  return {
    health: vi.fn(async () => true),
    leave: vi.fn(async () => {}),
    exists: vi.fn(() => true),
    readToken: vi.fn(() => "mock-token-123456789012345678901234567890"),
    spawn: vi.fn(() => child) as unknown as BackendDeps["spawn"],
    killGroup: vi.fn(),
    platform: "linux",
    resourcesPath: "/resources",
    now: vi.fn(() => 1000),
    ...overrides,
  };
}

describe("backend", () => {
  describe("redact", () => {
    it("hides tllm://join/ links anywhere in a string", () => {
      expect(redact("No secrets here")).toBe("No secrets here");
      expect(redact("tllm://join/INVITE123_abc-456")).toBe("tllm://join/<hidden>");
      expect(redact("connecting to tllm://join/SECRET_abc at port 9000")).toBe(
        "connecting to tllm://join/<hidden> at port 9000",
      );
      expect(
        redact("primary: tllm://join/FIRST_123 and secondary: tllm://join/SECOND_456!"),
      ).toBe("primary: tllm://join/<hidden> and secondary: tllm://join/<hidden>!");
    });
  });

  describe("buildArgs", () => {
    const baseSettings: AppSettings = {
      ...DEFAULT_SETTINGS,
      label: "my-node",
      memoryMb: 0,
      controlPort: 9000,
      listenPort: 9100,
      bootstrap: [],
      upnp: true,
      relay: false,
    };

    it("builds args for create mode with trimmed swarm name up to 80 chars", () => {
      const opts: StartOptions = {
        mode: "create",
        swarmName: "   " + "s".repeat(100) + "   ",
      };
      const args = buildArgs(opts, baseSettings, null);
      expect(args[0]).toBe("create");
      expect(args).toContain("--swarm-name");
      const nameIndex = args.indexOf("--swarm-name");
      expect(args[nameIndex + 1]).toBe("s".repeat(80));

      // Without swarmName
      const argsNoName = buildArgs({ mode: "create" }, baseSettings, null);
      expect(argsNoName).not.toContain("--swarm-name");
    });

    it("builds args for join mode, validating invite link prefix", () => {
      const validOpts: StartOptions = {
        mode: "join",
        invite: "   tllm://join/SWARM_SECRET_123   ",
      };
      const args = buildArgs(validOpts, baseSettings, null);
      expect(args[0]).toBe("join");
      expect(args[1]).toBe("tllm://join/SWARM_SECRET_123");

      expect(() => buildArgs({ mode: "join" }, baseSettings, null)).toThrow(
        "Paste the invite link first (it starts with tllm://join/).",
      );
      expect(() => buildArgs({ mode: "join", invite: "https://example.com" }, baseSettings, null)).toThrow(
        "Paste the invite link first (it starts with tllm://join/).",
      );
    });

    it("builds args for public mode without invite or swarm name", () => {
      const args = buildArgs({ mode: "public" }, baseSettings, null);
      expect(args[0]).toBe("public");
      expect(args).not.toContain("--swarm-name");
    });

    it("handles settings flags: memoryMb (omitted when 0), ports, bootstrap, upnp, relay", () => {
      const settingsWithFlags: AppSettings = {
        ...baseSettings,
        label: "  device-42  ",
        memoryMb: 4096.9,
        controlPort: 9001,
        listenPort: 9101,
        bootstrap: ["tcp://10.0.0.1:9100", "   ", "tcp://10.0.0.2:9100"],
        upnp: false,
        relay: true,
      };

      const args = buildArgs({ mode: "public" }, settingsWithFlags, "/path/to/pool.yaml");

      expect(args).toContain("--name");
      expect(args[args.indexOf("--name") + 1]).toBe("device-42");

      expect(args).toContain("--memory-mb");
      expect(args[args.indexOf("--memory-mb") + 1]).toBe("4096");

      expect(args).toContain("--port");
      expect(args[args.indexOf("--port") + 1]).toBe("9001");
      expect(args).toContain("--listen-port");
      expect(args[args.indexOf("--listen-port") + 1]).toBe("9101");

      const bootstrapIndices: number[] = [];
      args.forEach((arg, i) => {
        if (arg === "--bootstrap") bootstrapIndices.push(i);
      });
      expect(bootstrapIndices).toHaveLength(2);
      expect(args[bootstrapIndices[0] + 1]).toBe("tcp://10.0.0.1:9100");
      expect(args[bootstrapIndices[1] + 1]).toBe("tcp://10.0.0.2:9100");

      expect(args).toContain("--no-upnp");
      expect(args).toContain("--relay");
      expect(args).toContain("--pool");
      expect(args[args.indexOf("--pool") + 1]).toBe("/path/to/pool.yaml");
    });

    it("omits --memory-mb when memoryMb is 0 and omits --no-upnp/--relay when upnp is true and relay is false", () => {
      const args = buildArgs({ mode: "public" }, baseSettings, null);
      expect(args).not.toContain("--memory-mb");
      expect(args).not.toContain("--no-upnp");
      expect(args).not.toContain("--relay");
      expect(args).not.toContain("--pool");
    });
  });

  describe("BackendManager", () => {
    afterEach(() => {
      vi.useRealTimers();
      vi.restoreAllMocks();
    });

    it("initializes with missing phase if executable is not found", () => {
      const child = fakeChild();
      const deps = makeDeps(child, { exists: () => false });
      const manager = new BackendManager(deps, () => DEFAULT_SETTINGS);

      expect(manager.current().phase).toBe("missing");
      expect(manager.current().message).toContain("Torrent-LLM is not installed on this computer yet.");
      expect(manager.locate()).toBe("");
      expect(manager.poolFile()).toBeNull();
      expect(manager.adminToken()).toBe("mock-token-123456789012345678901234567890");
    });

    it("locate() prefers configured executable, then TORRENT_PEER env, then installed executable", () => {
      const child = fakeChild();
      const deps = makeDeps(child, { exists: (p) => p.includes("exists") });

      // Configured executable
      const m1 = new BackendManager(deps, () => ({ ...DEFAULT_SETTINGS, executable: "/custom/exists" }));
      expect(m1.locate()).toBe("/custom/exists");

      // Environment variable TORRENT_PEER
      const oldEnv = process.env.TORRENT_PEER;
      process.env.TORRENT_PEER = "/env/exists";
      try {
        const m2 = new BackendManager(deps, () => ({ ...DEFAULT_SETTINGS, executable: "" }));
        expect(m2.locate()).toBe("/env/exists");
      } finally {
        if (oldEnv !== undefined) process.env.TORRENT_PEER = oldEnv;
        else delete process.env.TORRENT_PEER;
      }

      // Installed executable fallback
      const installedPath = installedExecutable({ platform: deps.platform });
      const depsWithInstalled = makeDeps(child, { exists: (p) => p === installedPath });
      const m3 = new BackendManager(depsWithInstalled, () => ({ ...DEFAULT_SETTINGS, executable: "" }));
      expect(m3.locate()).toBe(installedPath);
    });

    it("poolFile() returns path when pool.yaml exists", () => {
      const child = fakeChild();
      const deps = makeDeps(child, { exists: (f) => f.includes("pool.yaml") });
      const manager = new BackendManager(deps, () => DEFAULT_SETTINGS);
      expect(manager.poolFile()).toContain("pool.yaml");
    });

    it("start() rejects and phase is 'missing' when executable is missing", async () => {
      const child = fakeChild();
      const deps = makeDeps(child, { exists: () => false });
      const manager = new BackendManager(deps, () => DEFAULT_SETTINGS);

      await expect(manager.start({ mode: "public" })).rejects.toThrow(
        "Torrent-LLM is not installed on this computer yet.",
      );
      expect(manager.current().phase).toBe("missing");
    });

    it("start() transitions from 'starting' to 'running' once health() resolves true", async () => {
      const child = fakeChild(5001);
      const deps = makeDeps(child, { health: vi.fn(async () => true) });
      const manager = new BackendManager(deps, () => ({ ...DEFAULT_SETTINGS, executable: "/bin/torrent-peer" }));

      const statuses: string[] = [];
      manager.on("status", (s) => statuses.push(s.phase));

      const status = await manager.start({ mode: "public" });
      expect(status.phase).toBe("starting");
      expect(status.pid).toBe(5001);
      expect(status.executable).toBe("/bin/torrent-peer");

      await vi.waitFor(() => {
        expect(manager.current().phase).toBe("running");
      });

      expect(manager.current().phase).toBe("running");
      expect(manager.current().message).toBe("Running.");
      expect(statuses).toEqual(["starting", "running"]);
    });

    it("start() transitions to error when health() times out after START_TIMEOUT_MS", async () => {
      vi.useFakeTimers();
      const child = fakeChild(5002);
      let currentTime = 1000;
      const deps = makeDeps(child, {
        health: vi.fn(async () => false),
        now: () => currentTime,
      });
      const manager = new BackendManager(deps, () => ({ ...DEFAULT_SETTINGS, executable: "/bin/torrent-peer" }));

      await manager.start({ mode: "public" });
      expect(manager.current().phase).toBe("starting");

      // Advance time beyond 15 minutes
      currentTime += 16 * 60 * 1000;
      await vi.advanceTimersByTimeAsync(16 * 60 * 1000);

      expect(manager.current().phase).toBe("error");
      expect(manager.current().message).toContain("The device did not come up in time.");
    });

    it("start() throws when already running", async () => {
      const child = fakeChild(5003);
      const deps = makeDeps(child);
      const manager = new BackendManager(deps, () => ({ ...DEFAULT_SETTINGS, executable: "/bin/torrent-peer" }));

      await manager.start({ mode: "public" });
      await expect(manager.start({ mode: "public" })).rejects.toThrow(
        "The device is already running. Stop it first.",
      );
    });

    it("redacts stdout and stderr lines, logs them, and emits on 'log'", async () => {
      const child = fakeChild(5004);
      const deps = makeDeps(child);
      const manager = new BackendManager(deps, () => ({ ...DEFAULT_SETTINGS, executable: "/bin/torrent-peer" }));

      const receivedLogs: string[] = [];
      manager.on("log", (line) => receivedLogs.push(line));

      await manager.start({ mode: "public" });

      (child.stdout as EventEmitter).emit(
        "data",
        Buffer.from("Connected to swarm with tllm://join/INVITE_ABC123 successfully\n"),
      );
      (child.stderr as EventEmitter).emit(
        "data",
        Buffer.from("Warning: peer tllm://join/SECRET_XYZ disconnected\n"),
      );

      expect(receivedLogs).toContain("Connected to swarm with tllm://join/<hidden> successfully");
      expect(receivedLogs).toContain("Warning: peer tllm://join/<hidden> disconnected");
      expect(manager.recentLogs()).toEqual(expect.arrayContaining(receivedLogs));
    });

    it("buffers partial lines so invite links split across 2 and 3 chunks are redacted and never appear in clear", async () => {
      const child = fakeChild(5004);
      const deps = makeDeps(child);
      const manager = new BackendManager(deps, () => ({ ...DEFAULT_SETTINGS, executable: "/bin/torrent-peer" }));

      const receivedLogs: string[] = [];
      manager.on("log", (line) => receivedLogs.push(line));

      await manager.start({ mode: "public" });

      // 1. Chunk split across 2 chunks: 'tllm://join/ABCD' and 'EFGH\n'
      (child.stdout as EventEmitter).emit("data", Buffer.from("join: tllm://join/ABCD"));
      // Before newline arrives, incomplete line is buffered and NOT yet emitted
      expect(receivedLogs.some((l) => l.includes("ABCD"))).toBe(false);
      expect(receivedLogs.some((l) => l.includes("EFGH"))).toBe(false);

      (child.stdout as EventEmitter).emit("data", Buffer.from("EFGH\n"));
      // Completed line is emitted with invite redacted
      expect(receivedLogs.some((l) => l.includes("EFGH"))).toBe(false);
      expect(receivedLogs.some((l) => l.includes("ABCD"))).toBe(false);
      expect(manager.recentLogs().some((l) => l.includes("EFGH"))).toBe(false);
      expect(manager.recentLogs().some((l) => l.includes("ABCD"))).toBe(false);
      expect(receivedLogs).toContain("join: tllm://join/<hidden>");

      // 2. Chunk split across 3 chunks: 'secret: tllm://', 'join/PART1_', 'PART2\n'
      (child.stderr as EventEmitter).emit("data", Buffer.from("secret: tllm://"));
      (child.stderr as EventEmitter).emit("data", Buffer.from("join/PART1_"));
      (child.stderr as EventEmitter).emit("data", Buffer.from("PART2\n"));

      expect(receivedLogs.some((l) => l.includes("PART1_"))).toBe(false);
      expect(receivedLogs.some((l) => l.includes("PART2"))).toBe(false);
      expect(manager.recentLogs().some((l) => l.includes("PART1_"))).toBe(false);
      expect(manager.recentLogs().some((l) => l.includes("PART2"))).toBe(false);
      expect(receivedLogs).toContain("secret: tllm://join/<hidden>");

      // 3. Trailing remainder without newline is flushed on child exit
      (child.stdout as EventEmitter).emit("data", Buffer.from("trailing secret tllm://join/LAST_CHUNK"));
      child.emit("exit", 0, null);

      expect(receivedLogs.some((l) => l.includes("LAST_CHUNK"))).toBe(false);
      expect(manager.recentLogs().some((l) => l.includes("LAST_CHUNK"))).toBe(false);
      expect(receivedLogs).toContain("trailing secret tllm://join/<hidden>");
    });

    it("sets phase to 'error' when child exits unexpectedly while not stopping", async () => {
      const child = fakeChild(5005);
      const deps = makeDeps(child);
      const manager = new BackendManager(deps, () => ({ ...DEFAULT_SETTINGS, executable: "/bin/torrent-peer" }));

      await manager.start({ mode: "public" });
      (child.stdout as EventEmitter).emit("data", Buffer.from("Last log line before crash\n"));

      child.emit("exit", 1, null);

      expect(manager.current().phase).toBe("error");
      expect(manager.current().message).toContain("exit 1");
      expect(manager.current().message).toContain("Last log line before crash");
      expect(manager.current().pid).toBeNull();
    });

    it("sets phase to 'error' when child emits error event", async () => {
      const child = fakeChild(5006);
      const deps = makeDeps(child);
      const manager = new BackendManager(deps, () => ({ ...DEFAULT_SETTINGS, executable: "/bin/torrent-peer" }));

      await manager.start({ mode: "public" });
      child.emit("error", new Error("EACCES permission denied"));

      expect(manager.current().phase).toBe("error");
      expect(manager.current().message).toBe("Could not start torrent-peer: EACCES permission denied");
      expect(manager.current().pid).toBeNull();
    });

    it("stop() calls leave with token and ends 'stopped' when child exits", async () => {
      const child = fakeChild(5007);
      const deps = makeDeps(child, {
        readToken: () => "valid-admin-token",
        leave: vi.fn(async () => {}),
      });
      const manager = new BackendManager(deps, () => ({
        ...DEFAULT_SETTINGS,
        executable: "/bin/torrent-peer",
        controlPort: 9000,
      }));

      await manager.start({ mode: "public" });

      const stopPromise = manager.stop();
      expect(manager.current().phase).toBe("stopping");

      await vi.waitFor(() => {
        expect(deps.leave).toHaveBeenCalledWith(9000, "valid-admin-token");
      });

      child.emit("exit", 0, null);
      const result = await stopPromise;

      expect(result.phase).toBe("stopped");
      expect(result.message).toBe("Stopped.");
      expect(result.url).toBe("");
      expect(result.pid).toBeNull();
    });

    it("stop() catches leave error and still completes when child exits", async () => {
      const child = fakeChild(5008);
      const deps = makeDeps(child, {
        leave: vi.fn(async () => {
          throw new Error("Network error leaving");
        }),
      });
      const manager = new BackendManager(deps, () => ({ ...DEFAULT_SETTINGS, executable: "/bin/torrent-peer" }));

      await manager.start({ mode: "public" });
      const stopPromise = manager.stop();

      await vi.waitFor(() => {
        expect(deps.leave).toHaveBeenCalled();
      });

      child.emit("exit", 0, null);
      const result = await stopPromise;
      expect(result.phase).toBe("stopped");
    });

    it("stop() returns immediately when no child is running", async () => {
      const child = fakeChild();
      const deps = makeDeps(child);
      const manager = new BackendManager(deps, () => ({ ...DEFAULT_SETTINGS, executable: "/bin/torrent-peer" }));

      const status = await manager.stop();
      expect(status.phase).toBe("stopped");
    });

    it("stop() kills after grace period on posix via deps.killGroup", async () => {
      vi.useFakeTimers();
      const child = fakeChild(8888);
      const killGroup = vi.fn();
      const deps = makeDeps(child, {
        killGroup,
        platform: "linux",
      });
      const manager = new BackendManager(deps, () => ({ ...DEFAULT_SETTINGS, executable: "/bin/torrent-peer" }));

      await manager.start({ mode: "public" });
      const stopPromise = manager.stop();

      // At 9_999 ms, grace period has not elapsed: killGroup must NOT have been called
      await vi.advanceTimersByTimeAsync(9_999);
      expect(killGroup).not.toHaveBeenCalled();

      // At 10_000+ ms, grace period elapses: killGroup is called
      await vi.advanceTimersByTimeAsync(1);
      await stopPromise;

      expect(killGroup).toHaveBeenCalledWith(8888);
    });

    it("falls back to child.kill('SIGKILL') when killGroup throws on posix", async () => {
      vi.useFakeTimers();
      const child = fakeChild(8888);
      const killGroup = vi.fn(() => {
        throw new Error("killGroup failed");
      });
      const deps = makeDeps(child, {
        killGroup,
        platform: "linux",
      });
      const manager = new BackendManager(deps, () => ({ ...DEFAULT_SETTINGS, executable: "/bin/torrent-peer" }));

      await manager.start({ mode: "public" });
      const stopPromise = manager.stop();

      await vi.advanceTimersByTimeAsync(10_000);
      await stopPromise;

      expect(killGroup).toHaveBeenCalledWith(8888);
      expect(child.kill).toHaveBeenCalledWith("SIGKILL");
    });

    it("BackendManager.kill ignores invalid pids (<= 1 or non-integers)", async () => {
      vi.useFakeTimers();
      for (const badPid of [0, 1, -1, undefined, NaN, 1.5]) {
        const child = fakeChild(badPid as number);
        const killGroup = vi.fn();
        const deps = makeDeps(child, {
          killGroup,
          platform: "linux",
        });
        const manager = new BackendManager(deps, () => ({ ...DEFAULT_SETTINGS, executable: "/bin/torrent-peer" }));

        await manager.start({ mode: "public" });
        const stopPromise = manager.stop();

        await vi.advanceTimersByTimeAsync(10_000);
        await stopPromise;

        expect(killGroup).not.toHaveBeenCalled();
        expect(child.kill).not.toHaveBeenCalled();
      }
    });

    it("stop() kills after grace period on win32 via spawn taskkill", async () => {
      vi.useFakeTimers();
      const child = fakeChild(9999);
      const spawnMock = vi.fn(() => child) as unknown as BackendDeps["spawn"];
      const deps = makeDeps(child, {
        spawn: spawnMock,
        platform: "win32",
      });
      const manager = new BackendManager(deps, () => ({ ...DEFAULT_SETTINGS, executable: "C:\\torrent-peer.exe" }));

      await manager.start({ mode: "public" });
      const stopPromise = manager.stop();

      await vi.advanceTimersByTimeAsync(11_000);
      await stopPromise;

      expect(spawnMock).toHaveBeenCalledWith(
        "taskkill",
        ["/T", "/F", "/PID", "9999"],
        { windowsHide: true },
      );
    });

    it("dispose() triggers stopping and kills child if active", async () => {
      const child = fakeChild(7777);
      const killGroup = vi.fn();
      const deps = makeDeps(child, { killGroup, platform: "linux" });
      const manager = new BackendManager(deps, () => ({ ...DEFAULT_SETTINGS, executable: "/bin/torrent-peer" }));

      manager.dispose(); // No child running: does nothing
      expect(killGroup).not.toHaveBeenCalled();

      // Start child and await it before disposing
      await manager.start({ mode: "public" });
      manager.dispose();
      expect(killGroup).toHaveBeenCalledWith(7777);
    });

    describe("install", () => {
      it("throws when child is already running", async () => {
        const child = fakeChild();
        const deps = makeDeps(child);
        const manager = new BackendManager(deps, () => ({ ...DEFAULT_SETTINGS, executable: "/bin/torrent-peer" }));

        await manager.start({ mode: "public" });
        await expect(manager.install()).rejects.toThrow("Stop the device before reinstalling.");
      });

      it("runs sh on posix from resourcesPath/install and handles exit code 0", async () => {
        const installChild = fakeChild();
        const spawnMock = vi.fn(() => installChild) as unknown as BackendDeps["spawn"];
        const deps = makeDeps(fakeChild(), {
          spawn: spawnMock,
          platform: "linux",
          resourcesPath: "/custom/res",
        });
        const manager = new BackendManager(deps, () => DEFAULT_SETTINGS);

        const installPromise = manager.install();
        expect(manager.current().phase).toBe("installing");
        expect(spawnMock).toHaveBeenCalledWith(
          "sh",
          ["/custom/res/install/install.sh"],
          expect.objectContaining({ windowsHide: true }),
        );

        (installChild.stdout as EventEmitter).emit("data", Buffer.from("Downloading 100%...\n"));
        installChild.emit("exit", 0);

        const result = await installPromise;
        expect(result.phase).toBe("stopped");
        expect(result.message).toBe("Installed. Choose how to join and press Start.");
      });

      it("runs powershell.exe on win32 from resourcesPath/install and handles non-zero exit code", async () => {
        const installChild = fakeChild();
        const spawnMock = vi.fn(() => installChild) as unknown as BackendDeps["spawn"];
        const deps = makeDeps(fakeChild(), {
          spawn: spawnMock,
          platform: "win32",
          resourcesPath: "C:\\custom\\res",
        });
        const manager = new BackendManager(deps, () => DEFAULT_SETTINGS);

        const installPromise = manager.install();
        expect(spawnMock).toHaveBeenCalledWith(
          "powershell.exe",
          ["-NoProfile", "-ExecutionPolicy", "Bypass", "-File", path.join("C:\\custom\\res", "install", "install.ps1")],
          expect.objectContaining({ windowsHide: true }),
        );

        installChild.emit("exit", 2);

        const result = await installPromise;
        expect(result.phase).toBe("error");
        expect(result.message).toContain("The installer failed (exit 2).");
      });

      it("handles spawn error during install", async () => {
        const installChild = fakeChild();
        const spawnMock = vi.fn(() => installChild) as unknown as BackendDeps["spawn"];
        const deps = makeDeps(fakeChild(), {
          spawn: spawnMock,
          platform: "linux",
        });
        const manager = new BackendManager(deps, () => DEFAULT_SETTINGS);

        const installPromise = manager.install();
        installChild.emit("error", new Error("sh not found in PATH"));

        const result = await installPromise;
        expect(result.phase).toBe("error");
        expect(result.message).toContain("The installer failed (exit -1).");
      });
    });
  });

  describe("realDeps safety guards", () => {
    it("refuses to signal process groups with pid <= 1 or non-integers", () => {
      const deps = realDeps("/res");
      expect(() => deps.killGroup(0)).toThrow("refusing to signal process group 0");
      expect(() => deps.killGroup(1)).toThrow("refusing to signal process group 1");
      expect(() => deps.killGroup(-1)).toThrow("refusing to signal process group -1");
      expect(() => deps.killGroup(1.5)).toThrow("refusing to signal process group 1.5");
      expect(() => deps.killGroup(NaN)).toThrow("refusing to signal process group NaN");
    });

    it("provides now, platform, exists and killGroup safety checks", () => {
      const deps = realDeps("/res");
      expect(deps.resourcesPath).toBe("/res");
      expect(deps.platform).toBe(process.platform);
      expect(typeof deps.now()).toBe("number");
      expect(deps.exists("/non/existent/path/for/sure")).toBe(false);
    });

    it("provides readToken helper reading >= 32 char token and rejecting short or missing files", () => {
      const tmpHome = fs.mkdtempSync(path.join(os.tmpdir(), "tllm-readtoken-test-"));
      const oldHome = process.env.TORRENT_LLM_HOME;
      process.env.TORRENT_LLM_HOME = tmpHome;
      try {
        const deps = realDeps("/res");
        // 1. Missing token file
        expect(deps.readToken()).toBeNull();

        // 2. Short token (< 32 chars)
        const tokenFile = path.join(tmpHome, "admin-token");
        fs.writeFileSync(tokenFile, "short-token\n", "utf8");
        expect(deps.readToken()).toBeNull();

        // 3. Valid token (>= 32 chars, trimmed)
        const validToken = "a".repeat(40);
        fs.writeFileSync(tokenFile, `  ${validToken}  \n`, "utf8");
        expect(deps.readToken()).toBe(validToken);
      } finally {
        if (oldHome !== undefined) process.env.TORRENT_LLM_HOME = oldHome;
        else delete process.env.TORRENT_LLM_HOME;
        fs.rmSync(tmpHome, { recursive: true, force: true });
      }
    });
  });
});

describe("manager regressions: output decoding", () => {
  async function started() {
    const child = fakeChild(4242);
    const manager = new BackendManager(makeDeps(child), () => ({ ...DEFAULT_SETTINGS, executable: "/x" }));
    await manager.start({ mode: "public" });
    return { child, manager };
  }

  it("keeps a multi-byte character that arrives split across two chunks", async () => {
    const { child, manager } = await started();
    const bytes = Buffer.from("café ✓\n", "utf8");
    const cut = bytes.indexOf(0xa9); // inside "é"
    (child.stdout as EventEmitter).emit("data", bytes.subarray(0, cut));
    (child.stdout as EventEmitter).emit("data", bytes.subarray(cut));
    expect(manager.recentLogs()).toContain("café ✓");
  });

  it("bounds a line that never ends and never leaks part of an invite link", async () => {
    const { child, manager } = await started();
    const link = `tllm://join/${"A".repeat(300)}`;
    const filler = "x".repeat(70 * 1024 - 100);
    (child.stdout as EventEmitter).emit("data", Buffer.from(filler + link + "y".repeat(5000)));
    const text = manager.recentLogs().join("\n");
    expect(text).not.toContain("A".repeat(50));
  });

  it("safeCut never cuts inside a link", async () => {
    const { safeCut } = await import("./backend");
    const text = `abc tllm://join/${"B".repeat(40)} def`;
    const inside = text.indexOf("B") + 10;
    expect(safeCut(text, inside)).toBe(text.indexOf("tllm://"));
    expect(safeCut(text, 2)).toBe(2);
  });
});
