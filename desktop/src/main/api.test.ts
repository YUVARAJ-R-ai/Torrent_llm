// @vitest-environment node
import { afterEach, describe, expect, it, vi } from "vitest";

import type { GenerateRequest } from "../shared/types";
import { READ_PATHS, WRITE_PATHS } from "../shared/types";
import {
  ApiError,
  apiGenerate,
  apiGet,
  apiPost,
  checkReadPath,
  checkWritePath,
  parseSseBlock,
  withQuery,
} from "./api";

describe("api", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
    vi.restoreAllMocks();
  });

  describe("checkReadPath and checkWritePath", () => {
    it("allows every defined read path and rejects anything else", () => {
      for (const p of READ_PATHS) {
        expect(() => checkReadPath(p)).not.toThrow();
      }
      expect(() => checkReadPath("/pool/../x")).toThrow(ApiError);
      expect(() => checkReadPath("/admin/leave")).toThrow(ApiError);
      expect(() => checkReadPath("/generate")).toThrow(ApiError);
      expect(() => checkReadPath("")).toThrow(ApiError);
    });

    it("allows every defined write path and rejects anything else", () => {
      for (const p of WRITE_PATHS) {
        expect(() => checkWritePath(p)).not.toThrow();
      }
      expect(() => checkWritePath("/pool")).toThrow(ApiError);
      expect(() => checkWritePath("/health")).toThrow(ApiError);
      expect(() => checkWritePath("/admin/leave")).toThrow(ApiError);
      expect(() => checkWritePath("")).toThrow(ApiError);
    });
  });

  describe("withQuery", () => {
    it("returns path unmodified if query is omitted or empty", () => {
      expect(withQuery("/pool")).toBe("/pool");
      expect(withQuery("/pool", {})).toBe("/pool");
    });

    it("encodes query parameters into the path", () => {
      const result = withQuery("/pool", { filter: "alive & ready", count: 5 });
      expect(result).toBe("/pool?filter=alive+%26+ready&count=5");
    });
  });

  describe("parseSseBlock", () => {
    it("parses single-line event and data", () => {
      const block = "event: token\ndata: {\"text\":\"hello\"}";
      expect(parseSseBlock(block)).toEqual({
        event: "token",
        data: "{\"text\":\"hello\"}",
      });
    });

    it("defaults event to message when event line is omitted", () => {
      const block = "data: {\"text\":\"hello\"}";
      expect(parseSseBlock(block)).toEqual({
        event: "message",
        data: "{\"text\":\"hello\"}",
      });
    });

    it("joins multi-line data with newlines", () => {
      const block = "event: chunk\ndata: line 1\ndata: line 2\ndata: line 3";
      expect(parseSseBlock(block)).toEqual({
        event: "chunk",
        data: "line 1\nline 2\nline 3",
      });
    });

    it("ignores comment lines starting with colon", () => {
      const block = ": keep-alive comment\nevent: ping\n: intermediate comment\ndata: pong\n: trailing";
      expect(parseSseBlock(block)).toEqual({
        event: "ping",
        data: "pong",
      });
    });

    it("handles data with and without space after colon", () => {
      expect(parseSseBlock("data:foo")).toEqual({ event: "message", data: "foo" });
      expect(parseSseBlock("data: foo")).toEqual({ event: "message", data: "foo" });
      expect(parseSseBlock("data:  two spaces")).toEqual({ event: "message", data: " two spaces" });
    });

    it("returns null when there are no data lines", () => {
      expect(parseSseBlock("")).toBeNull();
      expect(parseSseBlock(": only a comment")).toBeNull();
      expect(parseSseBlock("event: token")).toBeNull();
    });

    it("handles CRLF line endings", () => {
      const block = "event: token\r\ndata: {\"text\":\"crlf\"}\r\n";
      expect(parseSseBlock(block)).toEqual({
        event: "token",
        data: "{\"text\":\"crlf\"}",
      });
    });
  });

  describe("apiGet and apiPost with mocked fetch", () => {
    it("apiGet refuses bad path and does not call fetch", async () => {
      const fetchMock = vi.fn();
      vi.stubGlobal("fetch", fetchMock);

      await expect(apiGet("http://127.0.0.1:9000", "/admin/leave", null)).rejects.toThrow(ApiError);
      expect(fetchMock).not.toHaveBeenCalled();
    });

    it("apiPost refuses bad path and does not call fetch", async () => {
      const fetchMock = vi.fn();
      vi.stubGlobal("fetch", fetchMock);

      await expect(apiPost("http://127.0.0.1:9000", "/pool", null)).rejects.toThrow(ApiError);
      expect(fetchMock).not.toHaveBeenCalled();
    });

    it("apiGet sends both token headers when token is provided", async () => {
      const fetchMock = vi.fn().mockResolvedValue({
        ok: true,
        json: async () => ({ status: "ok" }),
      });
      vi.stubGlobal("fetch", fetchMock);

      const result = await apiGet<{ status: string }>("http://127.0.0.1:9000", "/pool", "secret-123");
      expect(result).toEqual({ status: "ok" });
      expect(fetchMock).toHaveBeenCalledWith(
        "http://127.0.0.1:9000/pool",
        expect.objectContaining({
          headers: {
            "content-type": "application/json",
            "x-pool-token": "secret-123",
            "x-admin-token": "secret-123",
          },
        }),
      );
    });

    it("apiGet omits token headers when token is null", async () => {
      const fetchMock = vi.fn().mockResolvedValue({
        ok: true,
        json: async () => ({ alive: true }),
      });
      vi.stubGlobal("fetch", fetchMock);

      await apiGet("http://127.0.0.1:9000", "/health", null, { check: 1 });
      expect(fetchMock).toHaveBeenCalledWith(
        "http://127.0.0.1:9000/health?check=1",
        expect.objectContaining({
          headers: {
            "content-type": "application/json",
          },
        }),
      );
    });

    it("apiGet maps { detail } error response to thrown ApiError message", async () => {
      vi.stubGlobal(
        "fetch",
        vi.fn().mockResolvedValue({
          ok: false,
          status: 403,
          statusText: "Forbidden",
          json: async () => ({ detail: "Token invalid or expired" }),
        }),
      );

      await expect(apiGet("http://127.0.0.1:9000", "/node", "bad-tok")).rejects.toThrow(
        new ApiError("Token invalid or expired"),
      );
    });

    it("apiGet maps non-JSON error to 'status statusText'", async () => {
      vi.stubGlobal(
        "fetch",
        vi.fn().mockResolvedValue({
          ok: false,
          status: 502,
          statusText: "Bad Gateway",
          json: async () => {
            throw new Error("Invalid JSON");
          },
        }),
      );

      await expect(apiGet("http://127.0.0.1:9000", "/pool", null)).rejects.toThrow(
        new ApiError("502 Bad Gateway"),
      );
    });

    it("apiPost sends method POST, body, and token headers", async () => {
      const fetchMock = vi.fn().mockResolvedValue({
        ok: true,
        json: async () => ({ banned: true }),
      });
      vi.stubGlobal("fetch", fetchMock);

      const payload = { peer_id: "peer-abc" };
      const res = await apiPost("http://127.0.0.1:9000", "/trust/ban", "tok-xyz", payload);
      expect(res).toEqual({ banned: true });

      expect(fetchMock).toHaveBeenCalledWith(
        "http://127.0.0.1:9000/trust/ban",
        expect.objectContaining({
          method: "POST",
          headers: {
            "content-type": "application/json",
            "x-pool-token": "tok-xyz",
            "x-admin-token": "tok-xyz",
          },
          body: JSON.stringify(payload),
        }),
      );
    });

    it("apiPost defaults body to empty object when body is undefined", async () => {
      const fetchMock = vi.fn().mockResolvedValue({
        ok: true,
        json: async () => ({}),
      });
      vi.stubGlobal("fetch", fetchMock);

      await apiPost("http://127.0.0.1:9000", "/admin/owner", null);
      expect(fetchMock).toHaveBeenCalledWith(
        "http://127.0.0.1:9000/admin/owner",
        expect.objectContaining({
          body: "{}",
        }),
      );
    });

    it("apiPost maps { detail } and non-JSON error properly", async () => {
      vi.stubGlobal(
        "fetch",
        vi.fn().mockResolvedValue({
          ok: false,
          status: 400,
          statusText: "Bad Request",
          json: async () => ({ detail: "Unknown peer id" }),
        }),
      );

      await expect(apiPost("http://127.0.0.1:9000", "/trust/unban", "tok", { peer: "x" })).rejects.toThrow(
        new ApiError("Unknown peer id"),
      );

      vi.stubGlobal(
        "fetch",
        vi.fn().mockResolvedValue({
          ok: false,
          status: 500,
          statusText: "Internal Server Error",
          json: async () => {
            throw new Error("HTML response");
          },
        }),
      );

      await expect(apiPost("http://127.0.0.1:9000", "/trust/unban", "tok", { peer: "x" })).rejects.toThrow(
        new ApiError("500 Internal Server Error"),
      );
    });
  });

  describe("apiGenerate", () => {
    it("handles tokens split across chunk boundaries and resolves with result", async () => {
      const encoder = new TextEncoder();
      const chunks = [
        encoder.encode("event: to"),
        encoder.encode("ken\ndata: {\"text\":\"Hel\",\"index\":0}\n\nevent: to"),
        encoder.encode("ken\ndata: {\"text\":\"lo\",\"index\":1}\n\nevent: do"),
        encoder.encode(
          "ne\ndata: {\"prompt\":\"test\",\"completion\":\"Hello\",\"tokens\":[\"Hel\",\"lo\"],\"requester\":\"self\",\"passes\":[],\"events\":[],\"totals\":{\"tokens\":2,\"failovers\":0,\"sent_bytes\":0,\"uncompressed_bytes\":0,\"compute_ms\":10,\"transport_ms\":5}}\n\n",
        ),
      ];

      const stream = new ReadableStream<Uint8Array>({
        start(controller) {
          for (const chunk of chunks) controller.enqueue(chunk);
          controller.close();
        },
      });

      const fetchMock = vi.fn().mockResolvedValue({
        ok: true,
        body: stream,
      });
      vi.stubGlobal("fetch", fetchMock);

      const tokens: { text: string; index: number }[] = [];
      const onToken = (text: string, index: number) => {
        tokens.push({ text, index });
      };

      const req: GenerateRequest = { prompt: "Say Hello" };
      const result = await apiGenerate("http://127.0.0.1:9000", "gen-token", req, onToken);

      expect(fetchMock).toHaveBeenCalledWith(
        "http://127.0.0.1:9000/generate/stream",
        expect.objectContaining({
          method: "POST",
          headers: {
            "content-type": "application/json",
            "x-pool-token": "gen-token",
            "x-admin-token": "gen-token",
            accept: "text/event-stream",
          },
          body: JSON.stringify(req),
        }),
      );

      expect(tokens).toEqual([
        { text: "Hel", index: 0 },
        { text: "lo", index: 1 },
      ]);
      expect(result.completion).toBe("Hello");
      expect(result.totals.tokens).toBe(2);
    });

    it("handles CRLF-delimited SSE stream events correctly", async () => {
      const encoder = new TextEncoder();
      const stream = new ReadableStream<Uint8Array>({
        start(controller) {
          controller.enqueue(encoder.encode("event: token\r\ndata: {\"text\":\"CRLF\",\"index\":0}\r\n\r\n"));
          controller.enqueue(
            encoder.encode(
              "event: done\r\ndata: {\"prompt\":\"p\",\"completion\":\"CRLF\",\"tokens\":[\"CRLF\"],\"requester\":\"s\",\"passes\":[],\"events\":[],\"totals\":{\"tokens\":1,\"failovers\":0,\"sent_bytes\":0,\"uncompressed_bytes\":0,\"compute_ms\":1,\"transport_ms\":1}}\r\n\r\n",
            ),
          );
          controller.close();
        },
      });

      vi.stubGlobal(
        "fetch",
        vi.fn().mockResolvedValue({
          ok: true,
          body: stream,
        }),
      );

      const tokens: string[] = [];
      const result = await apiGenerate("http://127.0.0.1:9000", null, { prompt: "p" }, (t) => tokens.push(t));
      expect(tokens).toEqual(["CRLF"]);
      expect(result.completion).toBe("CRLF");
    });

    it("handles multi-byte UTF-8 character split across chunk boundaries", async () => {
      // The euro sign '€' is 3 bytes in UTF-8: 0xE2, 0x82, 0xAC
      const encoder = new TextEncoder();
      const part1 = encoder.encode('event: token\ndata: {"text":"');
      const euroBytes = new Uint8Array([0xe2, 0x82, 0xac]);
      const part2 = encoder.encode('","index":0}\n\nevent: done\ndata: {"prompt":"p","completion":"€","tokens":["€"],"requester":"s","passes":[],"events":[],"totals":{"tokens":1,"failovers":0,"sent_bytes":0,"uncompressed_bytes":0,"compute_ms":1,"transport_ms":1}}\n\n');

      const chunk1 = new Uint8Array(part1.length + 2);
      chunk1.set(part1, 0);
      chunk1.set(euroBytes.slice(0, 2), part1.length);

      const chunk2 = new Uint8Array(1 + part2.length);
      chunk2.set(euroBytes.slice(2, 3), 0);
      chunk2.set(part2, 1);

      const stream = new ReadableStream<Uint8Array>({
        start(controller) {
          controller.enqueue(chunk1);
          controller.enqueue(chunk2);
          controller.close();
        },
      });

      vi.stubGlobal(
        "fetch",
        vi.fn().mockResolvedValue({
          ok: true,
          body: stream,
        }),
      );

      const tokens: string[] = [];
      const result = await apiGenerate("http://127.0.0.1:9000", null, { prompt: "p" }, (t) => tokens.push(t));
      expect(tokens).toEqual(["€"]);
      expect(result.completion).toBe("€");
    });

    it("rejects with ApiError when an SSE data payload contains malformed JSON", async () => {
      const encoder = new TextEncoder();
      const stream = new ReadableStream<Uint8Array>({
        start(controller) {
          controller.enqueue(encoder.encode("event: token\ndata: { malformed json\n\n"));
          controller.close();
        },
      });

      vi.stubGlobal(
        "fetch",
        vi.fn().mockResolvedValue({
          ok: true,
          body: stream,
        }),
      );

      await expect(
        apiGenerate("http://127.0.0.1:9000", null, { prompt: "test" }, () => {}),
      ).rejects.toThrow(ApiError);
    });

    it("rejects when an error event is received in the stream", async () => {
      const encoder = new TextEncoder();
      const stream = new ReadableStream<Uint8Array>({
        start(controller) {
          controller.enqueue(encoder.encode("event: error\ndata: {\"detail\":\"Model ran out of memory\"}\n\n"));
          controller.close();
        },
      });

      vi.stubGlobal(
        "fetch",
        vi.fn().mockResolvedValue({
          ok: true,
          body: stream,
        }),
      );

      await expect(
        apiGenerate("http://127.0.0.1:9000", null, { prompt: "test" }, () => {}),
      ).rejects.toThrow(new ApiError("Model ran out of memory"));
    });

    it("rejects when the stream ends without done event", async () => {
      const encoder = new TextEncoder();
      const stream = new ReadableStream<Uint8Array>({
        start(controller) {
          controller.enqueue(encoder.encode("event: token\ndata: {\"text\":\"hi\",\"index\":0}\n\n"));
          controller.close();
        },
      });

      vi.stubGlobal(
        "fetch",
        vi.fn().mockResolvedValue({
          ok: true,
          body: stream,
        }),
      );

      await expect(
        apiGenerate("http://127.0.0.1:9000", null, { prompt: "test" }, () => {}),
      ).rejects.toThrow(new ApiError("the device closed the stream before the answer finished"));
    });

    it("rejects when response is not ok", async () => {
      vi.stubGlobal(
        "fetch",
        vi.fn().mockResolvedValue({
          ok: false,
          status: 503,
          statusText: "Service Unavailable",
          json: async () => ({ detail: "Engine overloaded" }),
        }),
      );

      await expect(
        apiGenerate("http://127.0.0.1:9000", null, { prompt: "test" }, () => {}),
      ).rejects.toThrow(new ApiError("Engine overloaded"));
    });

    it("rejects when response.body is null", async () => {
      vi.stubGlobal(
        "fetch",
        vi.fn().mockResolvedValue({
          ok: true,
          status: 200,
          statusText: "OK",
          body: null,
          json: async () => ({}),
        }),
      );

      await expect(
        apiGenerate("http://127.0.0.1:9000", null, { prompt: "test" }, () => {}),
      ).rejects.toThrow(new ApiError("200 OK"));
    });
  });
});
