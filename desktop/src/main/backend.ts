/**
 * Runs the device (torrent-peer) for the app: finds it, installs it if missing,
 * starts it in the chosen swarm, watches it, and stops it cleanly.
 *
 * The app never needs a terminal: everything the one-command installer does, the
 * app can do with the installer script it ships with (extraResources/install).
 * Invite links are secrets, so they are passed to the process as an argument but
 * redacted from every log line the app keeps or shows.
 */

import { type ChildProcess, spawn } from "node:child_process";
import { EventEmitter } from "node:events";
import fs from "node:fs";
import path from "node:path";
import { StringDecoder } from "node:string_decoder";

import type { AppSettings, BackendStatus, StartOptions } from "../shared/types";
import { adminTokenPath, installedAppDir, installedExecutable } from "./paths";

const LOG_LINES = 2000;
const START_TIMEOUT_MS = 15 * 60 * 1000; // first start downloads and loads the model
const STOP_GRACE_MS = 10_000;
const INVITE = /tllm:\/\/join\/[A-Za-z0-9_-]+/g;
/** Longest invite link (net/invite.py MAX_LINK_CHARS). */
const MAX_LINK_CHARS = 4096;
/** Characters held while waiting for a newline before they are logged anyway. */
const MAX_PENDING_CHARS = 64 * 1024;

/** A cut point at or before ``at`` that is not inside an invite link. */
export function safeCut(text: string, at: number): number {
  const start = text.lastIndexOf("tllm://join/", at);
  if (start < 0) return at;
  let end = start + "tllm://join/".length;
  while (end < text.length && /[A-Za-z0-9_-]/.test(text[end])) end += 1;
  return end > at ? start : at;
}

/** Hide invite links (secrets) in anything shown or stored. */
export function redact(line: string): string {
  return line.replace(INVITE, "tllm://join/<hidden>");
}

/** The torrent-peer command line for a start request. Pure, for tests. */
export function buildArgs(options: StartOptions, settings: AppSettings, poolFile: string | null): string[] {
  const args: string[] = [options.mode];
  if (options.mode === "join") {
    if (!options.invite || !options.invite.trim().startsWith("tllm://join/")) {
      throw new Error("Paste the invite link first (it starts with tllm://join/).");
    }
    args.push(options.invite.trim());
  }
  if (options.mode === "create" && options.swarmName?.trim()) {
    args.push("--swarm-name", options.swarmName.trim().slice(0, 80));
  }
  if (poolFile) args.push("--pool", poolFile);
  if (settings.label.trim()) args.push("--name", settings.label.trim().slice(0, 64));
  if (settings.memoryMb > 0) args.push("--memory-mb", String(Math.floor(settings.memoryMb)));
  args.push("--port", String(settings.controlPort), "--listen-port", String(settings.listenPort));
  for (const address of settings.bootstrap) {
    if (address.trim()) args.push("--bootstrap", address.trim());
  }
  if (!settings.upnp) args.push("--no-upnp");
  if (settings.relay) args.push("--relay");
  return args;
}

export interface BackendDeps {
  /** Returns true when http://127.0.0.1:<port>/health answers. */
  health(port: number): Promise<boolean>;
  /** Asks the device to leave the swarm (POST /admin/leave). */
  leave(port: number, token: string): Promise<void>;
  exists(file: string): boolean;
  readToken(): string | null;
  spawn: typeof spawn;
  /** SIGKILL the process group ``pid`` leads (POSIX). Injected so tests never send real signals. */
  killGroup(pid: number): void;
  platform: NodeJS.Platform;
  resourcesPath: string;
  now(): number;
}

export class BackendManager extends EventEmitter {
  private status: BackendStatus = {
    phase: "stopped",
    message: "",
    executable: "",
    url: "",
    pid: null,
    mode: null,
    startedAt: null,
  };
  private child: ChildProcess | null = null;
  private logs: string[] = [];
  private stopping = false;
  private streams = {
    stdout: { decoder: new StringDecoder("utf8"), pending: "" },
    stderr: { decoder: new StringDecoder("utf8"), pending: "" },
  };

  constructor(
    private readonly deps: BackendDeps,
    private readonly getSettings: () => AppSettings,
  ) {
    super();
    this.refreshPresence();
  }

  // --- reading -----------------------------------------------------------------

  current(): BackendStatus {
    return { ...this.status };
  }

  recentLogs(): string[] {
    return [...this.logs];
  }

  adminToken(): string | null {
    return this.deps.readToken();
  }

  /** The executable to run, or "" if none is found. */
  locate(): string {
    const configured = this.getSettings().executable.trim();
    if (configured) return this.deps.exists(configured) ? configured : "";
    const fromEnv = process.env.TORRENT_PEER;
    if (fromEnv && this.deps.exists(fromEnv)) return fromEnv;
    const installed = installedExecutable({ platform: this.deps.platform });
    return this.deps.exists(installed) ? installed : "";
  }

  /** The pool file of the installed copy, so every device uses the same settings. */
  poolFile(): string | null {
    const file = path.join(installedAppDir({ platform: this.deps.platform }), "configs", "pool.yaml");
    return this.deps.exists(file) ? file : null;
  }

  // --- changing --------------------------------------------------------------------

  async start(options: StartOptions): Promise<BackendStatus> {
    if (this.child) throw new Error("The device is already running. Stop it first.");
    const executable = this.locate();
    if (!executable) {
      this.set({ phase: "missing", message: "Torrent-LLM is not installed on this computer yet.", executable: "" });
      throw new Error(this.status.message);
    }
    const settings = this.getSettings();
    const args = buildArgs(options, settings, this.poolFile());
    this.logs = [];
    this.resetStreams();
    this.log(`$ torrent-peer ${redact(args.join(" "))}`);
    const child = this.deps.spawn(executable, args, {
      env: { ...process.env, PYTHONUNBUFFERED: "1", PYTHONIOENCODING: "utf-8" },
      windowsHide: true,
      detached: this.deps.platform !== "win32", // own process group, so stop() can end it all
      stdio: ["ignore", "pipe", "pipe"],
    });
    this.child = child;
    this.stopping = false;
    this.set({
      phase: "starting",
      message: "Starting the device (the first start downloads the model)...",
      executable,
      url: `http://127.0.0.1:${settings.controlPort}`,
      pid: child.pid ?? null,
      mode: options.mode,
      startedAt: this.deps.now(),
    });
    child.stdout?.on("data", (chunk: Buffer) => this.feedStream("stdout", chunk));
    child.stderr?.on("data", (chunk: Buffer) => this.feedStream("stderr", chunk));
    child.on("error", (error) => {
      this.flushStreams();
      this.child = null;
      this.set({ phase: "error", message: `Could not start torrent-peer: ${error.message}`, pid: null });
    });
    child.on("exit", (code, signal) => {
      this.flushStreams();
      this.child = null;
      if (this.stopping) {
        this.set({ phase: "stopped", message: "Stopped.", pid: null, url: "" });
      } else {
        const tail = this.logs.slice(-3).join(" ").slice(-300);
        this.set({
          phase: "error",
          message: `The device stopped unexpectedly (${signal ?? `exit ${code}`}). ${tail}`,
          pid: null,
        });
      }
    });
    void this.waitUntilHealthy(settings.controlPort, child);
    return this.current();
  }

  async stop(): Promise<BackendStatus> {
    const child = this.child;
    if (!child) return this.current();
    this.stopping = true;
    this.set({ phase: "stopping", message: "Leaving the swarm..." });
    const token = this.deps.readToken();
    const port = this.getSettings().controlPort;
    if (token) {
      try {
        await this.deps.leave(port, token);
      } catch {
        // The process may already be on its way out; the kill below covers it.
      }
    }
    const exited = await this.waitForExit(child, STOP_GRACE_MS);
    if (!exited) this.kill(child);
    return this.current();
  }

  /** Runs the bundled installer (no swarm argument: install only). */
  async install(): Promise<BackendStatus> {
    if (this.child) throw new Error("Stop the device before reinstalling.");
    const dir = path.join(this.deps.resourcesPath, "install");
    const [command, args] =
      this.deps.platform === "win32"
        ? ["powershell.exe", ["-NoProfile", "-ExecutionPolicy", "Bypass", "-File", path.join(dir, "install.ps1")]]
        : ["sh", [path.join(dir, "install.sh")]];
    this.logs = [];
    this.resetStreams();
    this.set({ phase: "installing", message: "Installing Torrent-LLM (this downloads about 3 GB)..." });
    const child = this.deps.spawn(command, args, { windowsHide: true, stdio: ["ignore", "pipe", "pipe"] });
    child.stdout?.on("data", (chunk: Buffer) => this.feedStream("stdout", chunk));
    child.stderr?.on("data", (chunk: Buffer) => this.feedStream("stderr", chunk));
    const code = await new Promise<number | null>((resolve) => {
      child.on("exit", (c) => {
        this.flushStreams();
        resolve(c);
      });
      child.on("error", (e) => {
        this.flushStreams();
        this.log(`installer failed to start: ${e.message}`);
        resolve(-1);
      });
    });
    if (code === 0) {
      this.refreshPresence();
      this.set({ message: "Installed. Choose how to join and press Start." });
    } else {
      this.set({ phase: "error", message: `The installer failed (exit ${code}). See Diagnostics for its output.` });
    }
    return this.current();
  }

  dispose(): void {
    if (this.child) {
      this.stopping = true;
      this.kill(this.child);
    }
  }

  // --- internals -------------------------------------------------------------------

  private refreshPresence(): void {
    const executable = this.locate();
    this.status = {
      ...this.status,
      executable,
      phase: executable ? "stopped" : "missing",
      message: executable ? "" : "Torrent-LLM is not installed on this computer yet.",
    };
  }

  private set(patch: Partial<BackendStatus>): void {
    this.status = { ...this.status, ...patch };
    this.emit("status", this.current());
  }

  private feedStream(stream: "stdout" | "stderr", chunk: Buffer | string): void {
    const state = this.streams[stream];
    // StringDecoder keeps a multi-byte character that is split across two chunks whole.
    state.pending += typeof chunk === "string" ? chunk : state.decoder.write(chunk);
    const parts = state.pending.split("\n");
    state.pending = parts.pop() ?? "";
    for (const part of parts) this.logLine(part);
    if (state.pending.length > MAX_PENDING_CHARS) {
      // A child printing without newlines must not grow memory forever. Cut where no
      // invite link can be split, so the hidden part of a link is never left behind.
      const cut = safeCut(state.pending, state.pending.length - MAX_LINK_CHARS);
      this.logLine(state.pending.slice(0, cut));
      state.pending = state.pending.slice(cut);
    }
  }

  private flushStreams(): void {
    for (const stream of ["stdout", "stderr"] as const) {
      const state = this.streams[stream];
      const rest = state.pending + state.decoder.end();
      state.pending = "";
      state.decoder = new StringDecoder("utf8");
      if (rest) this.logLine(rest);
    }
  }

  /** Forget a previous process's half-lines before a new one starts. */
  private resetStreams(): void {
    for (const stream of ["stdout", "stderr"] as const) {
      this.streams[stream] = { decoder: new StringDecoder("utf8"), pending: "" };
    }
  }

  private logLine(raw: string): void {
    const line = raw.endsWith("\r") ? raw.slice(0, -1) : raw;
    if (line.trim()) this.log(line);
  }

  private log(line: string): void {
    const clean = redact(line);
    this.logs.push(clean);
    if (this.logs.length > LOG_LINES) this.logs.splice(0, this.logs.length - LOG_LINES);
    this.emit("log", clean);
  }

  private async waitUntilHealthy(port: number, child: ChildProcess): Promise<void> {
    const deadline = this.deps.now() + START_TIMEOUT_MS;
    while (this.child === child && this.deps.now() < deadline) {
      if (await this.deps.health(port).catch(() => false)) {
        if (this.child === child && this.status.phase === "starting") {
          this.set({ phase: "running", message: "Running." });
        }
        return;
      }
      await new Promise((r) => setTimeout(r, 1000));
    }
    if (this.child === child && this.status.phase === "starting") {
      this.set({ phase: "error", message: "The device did not come up in time. See Diagnostics." });
    }
  }

  private waitForExit(child: ChildProcess, ms: number): Promise<boolean> {
    if (child.exitCode !== null) return Promise.resolve(true);
    return new Promise((resolve) => {
      const timer = setTimeout(() => resolve(false), ms);
      child.once("exit", () => {
        clearTimeout(timer);
        resolve(true);
      });
    });
  }

  private kill(child: ChildProcess): void {
    const pid = child.pid;
    // Never signal pid 0 or 1: kill(-1) reaches EVERY process of this user (it once took
    // down a whole desktop session from a test with a fake child), kill(-0) our own group.
    if (typeof pid !== "number" || !Number.isInteger(pid) || pid <= 1) return;
    if (this.deps.platform === "win32") {
      // The whole tree: torrent-peer may have started helpers.
      this.deps.spawn("taskkill", ["/T", "/F", "/PID", String(pid)], { windowsHide: true });
      return;
    }
    try {
      this.deps.killGroup(pid); // the process group made by detached: true
    } catch {
      try {
        child.kill("SIGKILL");
      } catch {
        // already gone
      }
    }
  }
}

/** Production dependencies. */
export function realDeps(resourcesPath: string): BackendDeps {
  return {
    async health(port) {
      const response = await fetch(`http://127.0.0.1:${port}/health`, { signal: AbortSignal.timeout(2000) });
      return response.ok;
    },
    async leave(port, token) {
      await fetch(`http://127.0.0.1:${port}/admin/leave`, {
        method: "POST",
        headers: { "x-pool-token": token },
        signal: AbortSignal.timeout(3000),
      });
    },
    exists: (file) => {
      try {
        return fs.statSync(file).isFile();
      } catch {
        return false;
      }
    },
    readToken: () => {
      try {
        const token = fs.readFileSync(adminTokenPath(), "utf8").trim();
        return token.length >= 32 ? token : null;
      } catch {
        return null;
      }
    },
    spawn,
    killGroup: (pid) => {
      if (!Number.isInteger(pid) || pid <= 1) throw new Error(`refusing to signal process group ${pid}`);
      process.kill(-pid, "SIGKILL");
    },
    platform: process.platform,
    resourcesPath,
    now: () => Date.now(),
  };
}
