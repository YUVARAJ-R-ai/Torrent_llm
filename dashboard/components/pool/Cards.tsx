"use client";

/**
 * The things on the pool stage: one card per copy of a stage, a tile for a device
 * that is holding nothing, and a placeholder for a copy the pool is still after.
 *
 * Cards carry a `layoutId`, so when the pool moves a stage from one device to
 * another (or a device drops out and its card goes to the bench) the card glides
 * there instead of blinking out and back in. That motion is what makes a handover
 * readable.
 */

import { motion } from "motion/react";
import { Gamepad2, Play, Power, Zap } from "lucide-react";
import { forwardRef, type ReactNode } from "react";

import type { HostedStage, LabDevice, PoolMember } from "@/lib/pool";

export const STAGE_COLORS = [
  "var(--series-1)",
  "var(--series-2)",
  "var(--series-3)",
];
export const stageColor = (stage: number): string =>
  STAGE_COLORS[stage] ?? "var(--text-secondary)";

type Tone = "good" | "warn" | "bad" | "idle";

const TONE_COLOR: Record<Tone, string> = {
  good: "var(--status-good)",
  warn: "var(--status-warn)",
  bad: "var(--status-critical)",
  idle: "var(--text-muted)",
};

/** What a device is doing, in a few words, and how worried to be about it. */
export function describeDevice(
  member: PoolMember,
  lab: LabDevice | null | undefined,
  hosted?: HostedStage,
): { text: string; tone: Tone } {
  if (member.liveness === "down") {
    if (member.state === "left") return { text: "Left the pool", tone: "bad" };
    return {
      text: lab?.last_event === "crashed" ? "Crashed" : "Offline",
      tone: "bad",
    };
  }
  if (member.liveness === "suspect")
    return { text: "Not responding…", tone: "warn" };
  if (member.state === "draining")
    return { text: "Handing over…", tone: "warn" };
  if (member.state === "away")
    return { text: "Owner is using it", tone: "idle" };
  if (hosted?.status === "loading")
    return { text: "Loading its share…", tone: "warn" };
  if (hosted?.status === "ready") return { text: "Serving", tone: "good" };
  return { text: "Spare, ready to help", tone: "idle" };
}

function Dot({ tone, pulse }: { tone: Tone; pulse?: boolean }) {
  return (
    <span className="relative flex size-2">
      {pulse && (
        <span
          className="absolute inline-flex size-full animate-ping rounded-full opacity-60"
          style={{ background: TONE_COLOR[tone] }}
        />
      )}
      <span
        className="relative inline-flex size-2 rounded-full"
        style={{ background: TONE_COLOR[tone] }}
      />
    </span>
  );
}

function Chip({ children }: { children: ReactNode }) {
  return (
    <span
      className="tabular rounded px-1.5 py-0.5 text-[10px] tracking-wide"
      style={{ background: "var(--surface-1)", color: "var(--text-muted)" }}
    >
      {children}
    </span>
  );
}

/** Memory, as the number of stages the owner has made room for. */
function Slots({ used, total }: { used: number; total: number }) {
  return (
    <div
      className="flex items-center gap-1.5"
      title={`${used} of ${total} stage slots in use`}
    >
      <div className="flex gap-0.5">
        {Array.from({ length: Math.max(total, 1) }, (_, i) => (
          <span
            key={i}
            className="h-1.5 w-4 rounded-full"
            style={{
              background:
                i < used ? "var(--text-secondary)" : "var(--gridline)",
            }}
          />
        ))}
      </div>
      <span
        className="tabular text-[10px]"
        style={{ color: "var(--text-muted)" }}
      >
        {used}/{total} slots
      </span>
    </div>
  );
}

function linkText(member: PoolMember): string | null {
  if (!member.link) return null;
  return `${Math.round(member.link.latency_ms)} ms · ${Math.round(member.link.bandwidth_mbps)} Mbps`;
}

interface Actions {
  onOwner?: () => void;
  onStop?: () => void;
  onKill?: () => void;
  onStart?: () => void;
}

function ActionButtons({
  member,
  lab,
  actions,
}: {
  member: PoolMember | null;
  lab?: LabDevice | null;
  actions: Actions;
}) {
  const owner =
    member && (member.state === "draining" || member.state === "away");
  const button = (
    label: string,
    onClick: (() => void) | undefined,
    icon: ReactNode,
    tone?: "danger",
  ) =>
    onClick && (
      <button
        key={label}
        type="button"
        title={label}
        aria-label={label}
        onClick={onClick}
        className="flex size-6 items-center justify-center rounded-md transition-colors hover:bg-[var(--gridline)] focus-visible:outline-2"
        style={{
          color:
            tone === "danger"
              ? "var(--status-critical)"
              : "var(--text-secondary)",
        }}
      >
        {icon}
      </button>
    );
  return (
    <div className="flex gap-0.5">
      {member &&
        member.liveness === "alive" &&
        button(
          owner ? "Owner is done, lend it again" : "Owner needs this device",
          actions.onOwner,
          <Gamepad2 size={13} />,
        )}
      {lab?.running &&
        button("Switch off", actions.onStop, <Power size={13} />)}
      {lab?.running &&
        button("Crash it", actions.onKill, <Zap size={13} />, "danger")}
      {lab &&
        !lab.running &&
        button("Switch on", actions.onStart, <Play size={13} />)}
    </div>
  );
}

/** One copy of one stage, living on one device. */
export const CopyCard = forwardRef<
  HTMLDivElement,
  {
    member: PoolMember;
    stage: number;
    layers: string;
    hosted: HostedStage;
    layoutId: string;
    lab?: LabDevice | null;
    busy?: boolean;
    lost?: boolean;
    actions: Actions;
  }
>(function CopyCard(
  { member, stage, layers, hosted, layoutId, lab, busy, lost, actions },
  ref,
) {
  const status = lost
    ? { text: "No answer", tone: "bad" as Tone }
    : describeDevice(member, lab, hosted);
  const color = stageColor(stage);
  return (
    <motion.div
      ref={ref}
      layout
      layoutId={layoutId}
      initial={{ opacity: 0, scale: 0.92 }}
      animate={{
        opacity: lost ? 0.55 : 1,
        scale: busy ? 1.03 : 1,
        filter: lost ? "grayscale(1)" : "grayscale(0)",
      }}
      exit={{ opacity: 0, scale: 0.9 }}
      transition={{ type: "spring", stiffness: 320, damping: 28 }}
      className="group relative w-full rounded-xl px-3.5 py-3"
      style={{
        background: "var(--page-plane)",
        // Longhand on purpose: mixing `border` with `borderLeft` makes the
        // animation library warn when the styles change.
        borderStyle: "solid",
        borderWidth: "1px 1px 1px 3px",
        borderColor: lost
          ? "var(--status-critical)"
          : `var(--border) var(--border) var(--border) ${color}`,
        // The ring is a CSS transition: motion cannot interpolate a shadow
        // whose colour is a variable.
        boxShadow: busy
          ? `0 0 0 2px ${color}, 0 12px 30px -12px ${color}`
          : "none",
        transition: "box-shadow 180ms ease-out",
      }}
    >
      <div className="flex items-start justify-between gap-2">
        <div className="min-w-0">
          <div className="truncate text-sm font-medium">
            {member.label || member.name}
          </div>
          <div
            className="tabular mt-0.5 text-xs"
            style={{ color: "var(--text-secondary)" }}
          >
            Layers {layers}
          </div>
        </div>
        <div className="flex shrink-0 flex-col items-end gap-1">
          <Chip>{member.device.toUpperCase()}</Chip>
          <div className="opacity-0 transition-opacity group-focus-within:opacity-100 group-hover:opacity-100">
            <ActionButtons member={member} lab={lab} actions={actions} />
          </div>
        </div>
      </div>
      <div
        className="mt-2 flex items-center gap-1.5 text-[11px]"
        style={{ color: "var(--text-secondary)" }}
      >
        <Dot tone={status.tone} pulse={status.tone === "warn" || busy} />
        <span>{busy ? "Computing…" : status.text}</span>
        {hosted.inflight > 0 && !busy && (
          <span className="tabular" style={{ color: "var(--text-muted)" }}>
            · {hosted.inflight} running
          </span>
        )}
      </div>
      <div className="mt-2 flex flex-wrap items-center justify-between gap-x-3 gap-y-1">
        <Slots used={member.hosting.length} total={member.slots} />
        {linkText(member) && (
          <span
            className="tabular text-[10px]"
            style={{ color: "var(--text-muted)" }}
          >
            {linkText(member)}
          </span>
        )}
      </div>
    </motion.div>
  );
});

/** A device that is not holding any stage right now: spare, away, or offline. */
export const BenchTile = forwardRef<
  HTMLDivElement,
  {
    member: PoolMember | null;
    lab?: LabDevice | null;
    layoutId: string;
    actions: Actions;
  }
>(function BenchTile({ member, lab, layoutId, actions }, ref) {
  const name = member?.label || lab?.label || "Device";
  const status = member
    ? describeDevice(member, lab)
    : {
        text: lab?.last_event === "crashed" ? "Crashed" : "Switched off",
        tone: "idle" as Tone,
      };
  const dim = !member || member.liveness !== "alive" || member.state === "away";
  return (
    <motion.div
      ref={ref}
      layout
      layoutId={layoutId}
      initial={{ opacity: 0, scale: 0.92 }}
      animate={{ opacity: dim ? 0.7 : 1, scale: 1 }}
      exit={{ opacity: 0, scale: 0.9 }}
      transition={{ type: "spring", stiffness: 320, damping: 28 }}
      className="group relative rounded-xl px-3 py-2.5"
      style={{
        background: "var(--page-plane)",
        border: "1px dashed var(--border)",
      }}
    >
      <div className="flex items-start justify-between gap-2">
        <div className="min-w-0">
          <div className="truncate text-sm">{name}</div>
          <div
            className="mt-1 flex items-center gap-1.5 text-[11px]"
            style={{ color: "var(--text-secondary)" }}
          >
            <Dot tone={status.tone} />
            <span>{status.text}</span>
          </div>
        </div>
        <div className="flex shrink-0 flex-col items-end gap-1">
          <Chip>{(member?.device ?? lab?.device ?? "").toUpperCase()}</Chip>
          <div className="opacity-0 transition-opacity group-focus-within:opacity-100 group-hover:opacity-100">
            <ActionButtons member={member} lab={lab} actions={actions} />
          </div>
        </div>
      </div>
      {member && (
        <div className="mt-2">
          <Slots used={member.hosting.length} total={member.slots} />
        </div>
      )}
    </motion.div>
  );
});

/** A copy the pool wants but no device is holding yet. */
export function MissingCopy({
  stage,
  anyDevices,
}: {
  stage: number;
  anyDevices: boolean;
}) {
  return (
    <motion.div
      layout
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      className="flex min-h-[4.25rem] w-full items-center justify-center rounded-xl px-3 text-center text-[11px]"
      style={{
        border: `1px dashed ${stageColor(stage)}`,
        color: "var(--text-muted)",
        opacity: 0.8,
      }}
    >
      {anyDevices
        ? "Waiting for a device to take this on"
        : "No device has room for this"}
    </motion.div>
  );
}
