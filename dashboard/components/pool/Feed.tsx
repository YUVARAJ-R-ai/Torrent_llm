"use client";

import { AnimatePresence, motion } from "motion/react";

import type { ActivityEvent } from "@/lib/pool";

const KIND_COLOR: Record<string, string> = {
  joined: "var(--status-good)",
  up: "var(--status-good)",
  down: "var(--status-critical)",
  state: "var(--status-warn)",
  stage: "var(--series-1)",
};

function ago(seconds: number): string {
  if (seconds < 3) return "just now";
  if (seconds < 60) return `${Math.floor(seconds)}s ago`;
  return `${Math.floor(seconds / 60)}m ago`;
}

/** What the pool has been doing, newest first, in plain sentences. */
export default function Feed({
  events,
  now,
}: {
  events: ActivityEvent[];
  now: number;
}) {
  const shown = [...events].reverse().slice(0, 8);
  return (
    <ul className="flex flex-col gap-1.5 text-xs" aria-live="polite">
      <AnimatePresence initial={false}>
        {shown.map((event) => (
          <motion.li
            key={`${event.t}-${event.text}`}
            layout
            initial={{ opacity: 0, y: -6 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0 }}
            className="flex items-baseline gap-2"
          >
            <span
              className="mt-1 size-1.5 shrink-0 rounded-full"
              style={{
                background: KIND_COLOR[event.kind] ?? "var(--text-muted)",
              }}
            />
            <span style={{ color: "var(--text-primary)" }}>{event.text}</span>
            <span
              className="tabular ml-auto shrink-0"
              style={{ color: "var(--text-muted)" }}
            >
              {ago(now - event.t)}
            </span>
          </motion.li>
        ))}
      </AnimatePresence>
      {shown.length === 0 && (
        <li style={{ color: "var(--text-muted)" }}>
          Nothing has happened yet.
        </li>
      )}
    </ul>
  );
}
