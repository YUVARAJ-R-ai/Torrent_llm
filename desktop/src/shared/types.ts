/**
 * Types shared by the main process, the preload bridge and the renderer.
 *
 * The API shapes mirror the device's control panel (src/torrent_llm/pool/api.py and
 * api_swarm.py). Keep them in step: the renderer never talks to the device directly,
 * only through the bridge, which the main process backs with real HTTP calls.
 */

// --- the device's API ---------------------------------------------------------------

export type Liveness = "alive" | "suspect" | "down";
export type DeviceState = "active" | "draining" | "away" | "left";
export type StageStatus = "loading" | "ready" | "failed";

export interface HostedStage {
  stage: number;
  port: number;
  status: StageStatus;
  inflight: number;
  served: number;
  busy_ms: number;
}

export interface Member {
  name: string; // peer id in a swarm
  label: string;
  os: string;
  hostname: string;
  device: string;
  memory_mb: number;
  slots: number;
  speed: number;
  state: DeviceState;
  hosting: HostedStage[];
  asked: number;
  asked_tokens: number;
  addrs: string[];
  liveness: Liveness;
  is_self: boolean;
  rtt_ms: number | null;
}

export interface PoolView {
  self: string;
  pool: {
    model_id: string;
    num_layers: number;
    replicas: number;
    shard_mb: number;
    codec: string;
    stages: { stage: number; start: number; end: number }[];
  };
  members: Member[];
  ready_copies: number[];
  health: "healthy" | "degraded" | "down";
  now: number;
}

export interface NodeView {
  peer_id: string;
  label: string;
  swarm_id: string;
  swarm_name: string;
  public: boolean;
  addresses: string[];
  reachability: "public" | "private" | "unknown";
  mapped: string;
  relays: string[];
  serving_relay: boolean;
  connections: number;
  version: string;
}

export interface ConnectionView {
  peer_id: string;
  label: string;
  remote: string;
  inbound: boolean;
  via_relay: string | null;
  rtt_ms: number | null;
  bytes_sent: number;
  bytes_received: number;
  bandwidth_bps: number | null;
}

export interface TrustView {
  peer_id: string;
  label: string;
  checks: number;
  failures: number;
  score: number;
  banned: boolean;
  reason: string;
  served_ms: number;
  used_ms: number;
}

export interface PoolEvent {
  t: number;
  kind: "joined" | "down" | "up" | "state" | "stage" | string;
  text: string;
  member: string | null;
}

export interface InviteView {
  link: string;
  commands: { linux_macos: string; windows: string; installed: string };
}

export interface GenerateRequest {
  prompt: string;
  max_new_tokens?: number;
  chat?: boolean;
  temperature?: number;
}

export interface HopInfo {
  stage: number;
  member: string;
  sent_bytes: number;
  uncompressed_bytes: number;
  compute_ms: number;
  transport_ms: number;
}

export interface GenerateResult {
  prompt: string;
  completion: string;
  tokens: string[];
  requester: string;
  passes: { index: number; phase: string; route: [number, string][]; hops: HopInfo[] }[];
  events: { kind: string; member: string | null; detail: string; t: number }[];
  totals: {
    tokens: number;
    failovers: number;
    sent_bytes: number;
    uncompressed_bytes: number;
    compute_ms: number;
    transport_ms: number;
  };
}

// --- the backend process ----------------------------------------------------------

export type SwarmMode = "create" | "join" | "public";

export type BackendPhase =
  | "missing" // torrent-peer not found: offer to install
  | "installing"
  | "stopped"
  | "starting" // process up, control panel not answering yet (model loading)
  | "running"
  | "stopping"
  | "error";

export interface BackendStatus {
  phase: BackendPhase;
  /** Human sentence for the current phase, e.g. why it failed. */
  message: string;
  /** Path of the torrent-peer executable in use, or "" when missing. */
  executable: string;
  /** Control panel URL while running, e.g. http://127.0.0.1:9000. */
  url: string;
  pid: number | null;
  mode: SwarmMode | null;
  startedAt: number | null;
}

export interface StartOptions {
  mode: SwarmMode;
  /** Required for "join"; the tllm://join/... link. */
  invite?: string;
  /** Name of a new swarm (with "create"). */
  swarmName?: string;
}

export interface AppSettings {
  /** What this device is called in the swarm. */
  label: string;
  /** Memory to lend, in MB; 0 = let the device decide. */
  memoryMb: number;
  /** Control panel port (HTTP, this machine only). */
  controlPort: number;
  /** Peer connections port (TCP, encrypted). */
  listenPort: number;
  /** Explicit path to torrent-peer; "" = find it. */
  executable: string;
  /** Extra bootstrap addresses, e.g. tcp://203.0.113.7:9100. */
  bootstrap: string[];
  /** Ask the router to forward the port (UPnP). */
  upnp: boolean;
  /** Always offer to relay for unreachable devices. */
  relay: boolean;
  /** Last swarm joined, to offer "rejoin" on start. Never the invite secret itself. */
  lastSwarmName: string;
  theme: "system" | "light" | "dark";
}

export const DEFAULT_SETTINGS: AppSettings = {
  label: "",
  memoryMb: 0,
  controlPort: 9000,
  listenPort: 9100,
  executable: "",
  bootstrap: [],
  upnp: true,
  relay: false,
  lastSwarmName: "",
  theme: "system",
};

/** Paths the renderer may read through the bridge (GET). */
export const READ_PATHS = [
  "/health",
  "/pool",
  "/events",
  "/node",
  "/peers",
  "/trust",
  "/swarm/invite",
] as const;
export type ReadPath = (typeof READ_PATHS)[number];

/** Paths the renderer may post to through the bridge. */
export const WRITE_PATHS = [
  "/generate",
  "/trust/ban",
  "/trust/unban",
  "/admin/owner",
] as const;
export type WritePath = (typeof WRITE_PATHS)[number];

// --- the bridge (window.tllm) --------------------------------------------------------

export type Unsubscribe = () => void;

export interface TllmBridge {
  backend: {
    status(): Promise<BackendStatus>;
    start(options: StartOptions): Promise<BackendStatus>;
    stop(): Promise<BackendStatus>;
    install(): Promise<BackendStatus>;
    logs(): Promise<string[]>;
    onStatus(callback: (status: BackendStatus) => void): Unsubscribe;
    onLog(callback: (line: string) => void): Unsubscribe;
  };
  api: {
    /** GET a control-panel path; query is appended as ?k=v. Rejects with Error(message). */
    get<T>(path: ReadPath, query?: Record<string, string | number>): Promise<T>;
    post<T>(path: WritePath, body?: unknown): Promise<T>;
    /** Streams one answer; onToken gets each piece of text as it arrives. */
    generate(
      request: GenerateRequest,
      onToken: (text: string, index: number) => void,
    ): Promise<GenerateResult>;
  };
  settings: {
    get(): Promise<AppSettings>;
    set(patch: Partial<AppSettings>): Promise<AppSettings>;
  };
  system: {
    /** Opens http(s) URLs only. */
    openExternal(url: string): Promise<void>;
    copy(text: string): Promise<void>;
    platform: "win32" | "darwin" | "linux" | string;
  };
}

declare global {
  interface Window {
    tllm: TllmBridge;
  }
}
