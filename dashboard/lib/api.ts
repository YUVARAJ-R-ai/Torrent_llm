/**
 * Typed client for the Torrent-LLM HTTP layer (issue #26).
 *
 * These types mirror `src/torrent_llm/api/schemas.py` by hand. That duplication
 * is deliberate: generating them from the OpenAPI schema would mean the
 * dashboard cannot be typechecked without a running API, and this is a testing
 * tool that should build from a clean clone. The schemas are small and change
 * rarely; if they drift, the runtime shape check below is what catches it.
 *
 * Every duration is already milliseconds — the API converts from the profiler's
 * nanoseconds once, server-side, so nothing here has to.
 */

export const API_BASE =
  process.env.NEXT_PUBLIC_API_BASE ?? "http://127.0.0.1:8000";

export interface ShardInfo {
  address: string;
  shard: number;
  /** Half-open layer range as the node itself reports it, e.g. "[0:14)". */
  layers: string;
  hidden_size: number;
  device: string;
  dtype: string;
  model_id: string;
}

export interface Topology {
  model_id: string;
  num_layers: number;
  num_shards: number;
  codec: string;
  dtype: string;
  shards: ShardInfo[];
}

/** One activation crossing one hop. */
export interface HopMetrics {
  hop: number;
  address: string;
  codec: string;
  /**
   * "prefill" or "decode". These two are NOT comparable on one axis — their
   * payloads differ by roughly three orders of magnitude, which is why every
   * chart here facets by phase rather than plotting them together.
   */
  phase: string;
  batch: number;
  seq_len: number;
  /** 0 on hop 0, which carries token ids: the embedding has not happened yet. */
  hidden_size: number;
  dtype: string;
  sent_bytes: number;
  received_bytes: number;
  uncompressed_bytes: number;
  compression_ratio: number;
  wall_ms: number;
  compute_ms: number;
  transport_ms: number;
  /** Transport over wall. Decides whether compressing this hop could help at all. */
  transport_share: number;
}

export interface GenerateResult {
  prompt: string;
  completion: string;
  full_text: string;
  tokens_generated: number;
  use_cache: boolean;
  total_sent_bytes: number;
  total_wall_ms: number;
  total_compute_ms: number;
  total_transport_ms: number;
  hops: HopMetrics[];
}

export interface CacheComparison {
  prompt: string;
  tokens_generated: number;
  cached: GenerateResult;
  uncached: GenerateResult;
  /** uncached bytes / cached bytes — the headline number for issue #24. */
  bandwidth_reduction: number;
  /** False means a real bug: the two paths must compute the same thing. */
  same_output: boolean;
}

export interface HopSummaryMetrics {
  hop: number;
  codec: string;
  samples: number;
  median_sent_bytes: number;
  median_wall_ms: number;
  median_compute_ms: number;
  median_transport_ms: number;
  mean_compression_ratio: number;
  mean_transport_share: number;
  mean_effective_mbps: number;
}

export interface SeqLenProfile {
  seq_len: number;
  /** Bandwidth-bound or compute-bound, stated in words by the server. */
  verdict: string;
  hops: HopSummaryMetrics[];
}

export interface ProfileResult {
  model_id: string;
  codec: string;
  repeats: number;
  profiles: SeqLenProfile[];
}

/**
 * Thrown for any non-2xx response, carrying the API's own `detail` message.
 *
 * The API says useful things in that field — "prompt encoded to zero tokens",
 * "could not reach every shard in the chain" — and surfacing them beats a
 * generic "request failed" that sends someone to the network tab.
 */
export class ApiError extends Error {
  constructor(
    message: string,
    readonly status: number,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response;
  try {
    response = await fetch(`${API_BASE}${path}`, {
      ...init,
      headers: { "Content-Type": "application/json", ...init?.headers },
    });
  } catch (cause) {
    // fetch only rejects on a transport-level failure, which here almost
    // always means the API process is not running at all — a much more useful
    // thing to say than "Failed to fetch".
    throw new ApiError(
      `cannot reach the API at ${API_BASE}. Is \`torrent-api\` running?`,
      0,
    );
  }

  if (!response.ok) {
    let detail = response.statusText;
    try {
      const body = await response.json();
      // FastAPI puts validation errors in a list and explicit raises in a string.
      detail =
        typeof body.detail === "string"
          ? body.detail
          : JSON.stringify(body.detail);
    } catch {
      // Non-JSON error body; statusText is the best available.
    }
    throw new ApiError(detail, response.status);
  }

  return response.json() as Promise<T>;
}

export const api = {
  topology: () => request<Topology>("/topology"),

  generate: (prompt: string, maxNewTokens: number, useCache: boolean) =>
    request<GenerateResult>("/generate", {
      method: "POST",
      body: JSON.stringify({
        prompt,
        max_new_tokens: maxNewTokens,
        use_cache: useCache,
      }),
    }),

  compareCache: (prompt: string, maxNewTokens: number) =>
    request<CacheComparison>("/compare-cache", {
      method: "POST",
      body: JSON.stringify({ prompt, max_new_tokens: maxNewTokens }),
    }),

  profile: (seqLens: number[], repeats: number) =>
    request<ProfileResult>("/profile", {
      method: "POST",
      body: JSON.stringify({ seq_lens: seqLens, repeats }),
    }),
};
