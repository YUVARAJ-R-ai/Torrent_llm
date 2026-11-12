/**
 * Chat: Ask the model that this swarm runs cooperatively.
 * Streams answers token-by-token, shows the routing and compression summary,
 * and handles failovers and network errors gracefully.
 */

import { Send, Trash2 } from "lucide-react";
import { useEffect, useRef, useState } from "react";

import type { ScreenProps } from "../App";
import type { GenerateResult, PoolView } from "../../shared/types";
import { Button, Card, Empty, ErrorNote, PageTitle } from "../components/ui";
import { ms, ratio, shortId } from "../lib/format";
import { errorText, usePoll } from "../lib/hooks";

interface ChatMessage {
  id: string;
  role: "user" | "assistant";
  text: string;
  error?: string | null;
  result?: GenerateResult | null;
}

/** Formats the routing, timing, and compression stats for a completed generation. */
function formatSummary(
  result: GenerateResult,
  getLabel: (peerId: string) => string,
): string {
  const tokenCount = result.totals?.tokens ? result.totals.tokens : (result.tokens?.length ?? 0);
  const computeMs = result.totals?.compute_ms ?? 0;
  const transportMs = result.totals?.transport_ms ?? 0;
  const timeStr = ms(computeMs + transportMs);

  const firstRoute = result.passes?.[0]?.route;
  const routeStr =
    firstRoute && firstRoute.length > 0
      ? firstRoute.map(([_, peerId]) => getLabel(peerId)).join(" → ")
      : "—";

  const uncompressed = result.totals?.uncompressed_bytes ?? 0;
  const sent = result.totals?.sent_bytes ?? 0;
  const trafficStr = ratio(uncompressed, sent);

  let summary = `${tokenCount} tokens · ${timeStr} · route: ${routeStr} · traffic ${trafficStr} smaller`;
  if (result.totals?.failovers && result.totals.failovers > 0) {
    summary += ` · switched devices ${result.totals.failovers}x`;
  }
  return summary;
}

export default function ChatScreen({ status, go }: ScreenProps) {
  // Typing stays possible while an answer streams; only Send waits. Focus comes back
  // to the box after each answer so the next question can be typed straight away.
  const questionRef = useRef<HTMLTextAreaElement | null>(null);

  const running = status.phase === "running";
  const pool = usePoll<PoolView>("/pool", { enabled: running });

  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [prompt, setPrompt] = useState("");
  const [maxLength, setMaxLength] = useState(64);
  const [busy, setBusy] = useState(false);
  const busyRef = useRef(false);

  const bottomRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    if (bottomRef.current?.scrollIntoView) {
      bottomRef.current.scrollIntoView({ behavior: "auto" });
    }
  }, [messages]);

  const getLabel = (peerId: string): string => {
    const member = pool.data?.members.find((m) => m.name === peerId);
    return member?.label || shortId(peerId);
  };

  const handleSend = async () => {
    const trimmed = prompt.trim();
    if (!trimmed || busyRef.current) return;

    busyRef.current = true;
    setBusy(true);

    const userMsg: ChatMessage = {
      id: `${Date.now()}-u`,
      role: "user",
      text: trimmed,
    };
    const answerId = `${Date.now()}-a`;
    const answerMsg: ChatMessage = {
      id: answerId,
      role: "assistant",
      text: "",
    };

    setMessages((prev) => [...prev, userMsg, answerMsg]);
    setPrompt("");

    try {
      const result = await window.tllm.api.generate(
        {
          prompt: trimmed,
          max_new_tokens: maxLength,
          chat: true,
        },
        (tokenText: string) => {
          setMessages((prev) =>
            prev.map((msg) =>
              msg.id === answerId ? { ...msg, text: msg.text + tokenText } : msg,
            ),
          );
        },
      );
      setMessages((prev) =>
        prev.map((msg) =>
          msg.id === answerId
            ? {
                ...msg,
                text: result.completion || msg.text,
                result,
              }
            : msg,
        ),
      );
    } catch (err) {
      const errMessage = errorText(err);
      setMessages((prev) =>
        prev.map((msg) =>
          msg.id === answerId ? { ...msg, error: errMessage } : msg,
        ),
      );
    } finally {
      busyRef.current = false;
      setBusy(false);
      questionRef.current?.focus();
    }
  };

  if (!running) {
    return (
      <div className="mx-auto max-w-4xl space-y-4">
        <PageTitle title="Chat">Ask the model this swarm runs together.</PageTitle>
        <Card>
          <Empty title="Not in a swarm">
            <p className="mb-3">
              Start or join a swarm on Home before chatting.
            </p>
            <Button kind="primary" onClick={() => go("home")}>
              Go to Home
            </Button>
          </Empty>
        </Card>
      </div>
    );
  }

  return (
    <div className="mx-auto max-w-4xl space-y-4">
      <div className="flex items-center justify-between gap-4">
        <PageTitle title="Chat">Ask the model this swarm runs together.</PageTitle>
        {messages.length > 0 && (
          <Button
            kind="ghost"
            disabled={busy}
            onClick={() => setMessages([])}
          >
            <Trash2 size={14} aria-hidden /> Clear
          </Button>
        )}
      </div>

      <Card>
        <div className="min-h-64 space-y-3">
          {messages.length === 0 ? (
            <Empty title="No messages yet">
              Type your question below to ask the distributed model.
            </Empty>
          ) : (
            messages.map((msg) => (
              <div key={msg.id} className="space-y-1">
                {msg.role === "user" ? (
                  <div className="flex justify-end">
                    <div
                      className="max-w-[85%] rounded-lg px-3 py-2 text-sm whitespace-pre-wrap"
                      style={{
                        background: "var(--surface-2)",
                        color: "var(--text-primary)",
                      }}
                    >
                      {msg.text}
                    </div>
                  </div>
                ) : (
                  <div className="flex flex-col items-start max-w-[85%]">
                    {msg.text ? (
                      <div
                        className="rounded-lg px-3 py-2 text-sm whitespace-pre-wrap"
                        style={{ color: "var(--text-primary)" }}
                      >
                        {msg.text}
                      </div>
                    ) : null}
                    {msg.error ? (
                      <ErrorNote>{msg.error}</ErrorNote>
                    ) : (
                      <>
                        {!msg.text && busy && msg === messages[messages.length - 1] && (
                          <div
                            className="rounded-lg px-3 py-2 text-sm whitespace-pre-wrap"
                            style={{ color: "var(--text-primary)" }}
                          >
                            …
                          </div>
                        )}
                        {msg.result && (
                          <div
                            className="mt-1 text-xs"
                            style={{ color: "var(--text-muted)" }}
                          >
                            {formatSummary(msg.result, getLabel)}
                          </div>
                        )}
                      </>
                    )}
                  </div>
                )}
              </div>
            ))
          )}
          <div ref={bottomRef} />
        </div>
      </Card>

      <Card>
        <form
          onSubmit={(e) => {
            e.preventDefault();
            void handleSend();
          }}
          className="space-y-3"
        >
          <div>
            <label
              htmlFor="chat-question"
              className="block text-xs font-medium mb-1"
              style={{ color: "var(--text-secondary)" }}
            >
              Your question
            </label>
            <textarea
              id="chat-question"
              placeholder="Your question…"
              rows={3}
              maxLength={4000}
              ref={questionRef}
              value={prompt}
              onChange={(e) => setPrompt(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter" && !e.shiftKey) {
                  e.preventDefault();
                  void handleSend();
                }
              }}
              className="w-full rounded-md border p-2 text-sm focus:outline-none"
              style={{
                background: "var(--surface-1)",
                borderColor: "var(--border)",
                color: "var(--text-primary)",
              }}
            />
          </div>

          <div className="flex items-center justify-between gap-3">
            <div className="flex items-center gap-2">
              <label
                htmlFor="max-length-select"
                className="text-xs"
                style={{ color: "var(--text-secondary)" }}
              >
                Max length
              </label>
              <select
                id="max-length-select"
                value={maxLength}
                disabled={busy}
                onChange={(e) => setMaxLength(Number(e.target.value))}
                className="rounded border px-2 py-1 text-sm"
                style={{
                  background: "var(--surface-1)",
                  borderColor: "var(--border)",
                  color: "var(--text-primary)",
                }}
              >
                <option value={32}>32 tokens</option>
                <option value={64}>64 tokens</option>
                <option value={128}>128 tokens</option>
                <option value={256}>256 tokens</option>
              </select>
            </div>

            <Button
              kind="primary"
              type="submit"
              disabled={busy}
            >
              <Send size={14} aria-hidden /> Send
            </Button>
          </div>
        </form>
      </Card>
    </div>
  );
}
