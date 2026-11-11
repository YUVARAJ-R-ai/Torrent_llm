/**
 * The main process's client for the device's control panel. The renderer reaches it
 * only through the bridge, with paths from a fixed list, and never sees the admin
 * token: it is added here.
 */

import type { GenerateRequest, GenerateResult } from "../shared/types";
import { READ_PATHS, WRITE_PATHS } from "../shared/types";

export class ApiError extends Error {}

function headers(token: string | null): Record<string, string> {
  const h: Record<string, string> = { "content-type": "application/json" };
  if (token) {
    // The pool's owner routes and the swarm routes both accept the admin token.
    h["x-pool-token"] = token;
    h["x-admin-token"] = token;
  }
  return h;
}

async function detail(response: Response): Promise<string> {
  try {
    const body = (await response.json()) as { detail?: unknown };
    if (typeof body.detail === "string") return body.detail;
  } catch {
    // not JSON
  }
  return `${response.status} ${response.statusText}`;
}

export function checkReadPath(path: string): void {
  if (!(READ_PATHS as readonly string[]).includes(path)) throw new ApiError(`not an allowed path: ${path}`);
}

export function checkWritePath(path: string): void {
  if (!(WRITE_PATHS as readonly string[]).includes(path)) throw new ApiError(`not an allowed path: ${path}`);
}

export function withQuery(path: string, query?: Record<string, string | number>): string {
  if (!query) return path;
  const params = new URLSearchParams();
  for (const [k, v] of Object.entries(query)) params.set(k, String(v));
  const text = params.toString();
  return text ? `${path}?${text}` : path;
}

export async function apiGet<T>(base: string, path: string, token: string | null, query?: Record<string, string | number>): Promise<T> {
  checkReadPath(path);
  const response = await fetch(base + withQuery(path, query), {
    headers: headers(token),
    signal: AbortSignal.timeout(10_000),
  });
  if (!response.ok) throw new ApiError(await detail(response));
  return (await response.json()) as T;
}

export async function apiPost<T>(base: string, path: string, token: string | null, body?: unknown): Promise<T> {
  checkWritePath(path);
  const response = await fetch(base + path, {
    method: "POST",
    headers: headers(token),
    body: JSON.stringify(body ?? {}),
    signal: AbortSignal.timeout(path === "/generate" ? 600_000 : 10_000),
  });
  if (!response.ok) throw new ApiError(await detail(response));
  return (await response.json()) as T;
}

/** Parses one server-sent-events block into (event, data). */
export function parseSseBlock(block: string): { event: string; data: string } | null {
  let event = "message";
  const data: string[] = [];
  for (const line of block.split(/\r?\n/)) {
    if (line.startsWith("event:")) event = line.slice(6).trim();
    else if (line.startsWith("data:")) data.push(line.slice(5).replace(/^ /, ""));
  }
  return data.length ? { event, data: data.join("\n") } : null;
}

/** POST /generate/stream and call onToken per token; resolves with the final result. */
export async function apiGenerate(
  base: string,
  token: string | null,
  request: GenerateRequest,
  onToken: (text: string, index: number) => void,
): Promise<GenerateResult> {
  const response = await fetch(`${base}/generate/stream`, {
    method: "POST",
    headers: { ...headers(token), accept: "text/event-stream" },
    body: JSON.stringify(request),
    signal: AbortSignal.timeout(600_000),
  });
  if (!response.ok || !response.body) throw new ApiError(await detail(response));
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    let cut: number;
    while ((cut = buffer.search(/\r?\n\r?\n/)) >= 0) {
      const block = buffer.slice(0, cut);
      buffer = buffer.slice(cut).replace(/^\r?\n\r?\n/, "");
      const parsed = parseSseBlock(block);
      if (!parsed) continue;
      let payload: Record<string, unknown>;
      try {
        payload = JSON.parse(parsed.data) as Record<string, unknown>;
      } catch (e) {
        throw new ApiError(`malformed SSE event payload: ${e instanceof Error ? e.message : String(e)}`);
      }
      if (parsed.event === "token") onToken(String(payload.text ?? ""), Number(payload.index ?? 0));
      else if (parsed.event === "done") return payload as unknown as GenerateResult;
      else if (parsed.event === "error") throw new ApiError(String(payload.detail ?? "the answer failed"));
    }
  }
  throw new ApiError("the device closed the stream before the answer finished");
}
