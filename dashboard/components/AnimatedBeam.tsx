"use client";

/**
 * A line between two elements that a pulse of light travels along.
 *
 * Adapted from Magic UI's AnimatedBeam (MIT, https://magicui.design,
 * registry/magicui/animated-beam.tsx). Changes from the original:
 *
 * - The pulse plays once per change of `pulse` instead of looping forever, so
 *   each one stands for one real payload crossing the link.
 * - The gradient is placed in pixel space along the path itself rather than in
 *   percentages of the whole container. The original assumes the beam spans the
 *   container; ours join neighbours in a row, so a percentage sweep would spend
 *   most of its time off the visible segment.
 * - Straight paths only, which works for both the row and the stacked layout.
 * - A dot travels with the pulse, and an optional label above the midpoint
 *   says what is being sent.
 */

import { AnimatePresence, motion } from "motion/react";
import { useEffect, useId, useState, type RefObject } from "react";

interface Point {
  x: number;
  y: number;
}

export interface AnimatedBeamProps {
  containerRef: RefObject<HTMLElement | null>;
  fromRef: RefObject<HTMLElement | null>;
  toRef: RefObject<HTMLElement | null>;
  /** Changing this plays the pulse once. 0 means idle: no pulse at all. */
  pulse: number;
  /** Seconds the pulse takes to cross. */
  duration: number;
  /** Shown at the midpoint while a pulse is in flight. */
  label?: string;
  /** Always shown just under the midpoint: what this link carries. */
  caption?: string;
  pathColor?: string;
  gradientStartColor?: string;
  gradientStopColor?: string;
}

export default function AnimatedBeam({
  containerRef,
  fromRef,
  toRef,
  pulse,
  duration,
  label,
  caption,
  pathColor = "var(--baseline)",
  gradientStartColor = "var(--series-1)",
  gradientStopColor = "var(--series-2)",
}: AnimatedBeamProps) {
  // useId output contains characters that are not valid in a url(#...)
  // reference, so keep only the safe ones.
  const id = `beam-${useId().replace(/[^a-zA-Z0-9_-]/g, "")}`;
  const [ends, setEnds] = useState<{
    a: Point;
    b: Point;
    horizontal: boolean;
  } | null>(null);
  const [size, setSize] = useState({ width: 0, height: 0 });

  useEffect(() => {
    const update = () => {
      const container = containerRef.current;
      const from = fromRef.current;
      const to = toRef.current;
      if (!container || !from || !to) return;

      const box = container.getBoundingClientRect();
      const ra = from.getBoundingClientRect();
      const rb = to.getBoundingClientRect();
      const ca = {
        x: ra.left - box.left + ra.width / 2,
        y: ra.top - box.top + ra.height / 2,
      };
      const cb = {
        x: rb.left - box.left + rb.width / 2,
        y: rb.top - box.top + rb.height / 2,
      };

      // Start and end at the elements' edges, not their centres, so the line
      // never runs underneath a card.
      const horizontal = Math.abs(cb.x - ca.x) >= Math.abs(cb.y - ca.y);
      // Stacked, a node's caption sits under it in the line's path, so the
      // line has to start below the whole node, caption included.
      const below = horizontal
        ? ra
        : (from.parentElement?.getBoundingClientRect() ?? ra);
      const a = horizontal
        ? { x: ra.right - box.left, y: ca.y }
        : { x: ca.x, y: below.bottom - box.top + 4 };
      const b = horizontal
        ? { x: rb.left - box.left, y: cb.y }
        : { x: cb.x, y: rb.top - box.top };

      setSize({ width: box.width, height: box.height });
      setEnds({ a, b, horizontal });
    };

    const observer = new ResizeObserver(update);
    if (containerRef.current) observer.observe(containerRef.current);
    update();
    return () => observer.disconnect();
  }, [containerRef, fromRef, toRef]);

  if (!ends) return null;

  const { a, b, horizontal } = ends;
  const mid = { x: (a.x + b.x) / 2, y: (a.y + b.y) / 2 };
  const dx = b.x - a.x;
  const dy = b.y - a.y;
  const length = Math.hypot(dx, dy) || 1;
  const ux = dx / length;
  const uy = dy / length;
  // The lit band is a fixed share of the segment, so short and long links
  // read the same way.
  const band = length * 0.45;
  // The head of the lit band moves exactly with the packet dot, from one end
  // to the other; the trail behind it fades out with the packet on arrival.
  const start = a;
  const end = b;

  return (
    <>
      <svg
        fill="none"
        width={size.width}
        height={size.height}
        viewBox={`0 0 ${size.width} ${size.height}`}
        className="pointer-events-none absolute top-0 left-0 transform-gpu"
        aria-hidden
      >
        <path
          d={`M ${a.x},${a.y} L ${b.x},${b.y}`}
          stroke={pathColor}
          strokeWidth={2}
          strokeDasharray="4 6"
          strokeLinecap="round"
        />
        {pulse > 0 && (
          <motion.path
            key={`trail-${pulse}`}
            initial={{ opacity: 1 }}
            animate={{ opacity: [1, 1, 0] }}
            transition={{ duration: duration * 1.15, times: [0, 0.87, 1] }}
            d={`M ${a.x},${a.y} L ${b.x},${b.y}`}
            stroke={`url(#${id})`}
            strokeWidth={4}
            strokeLinecap="round"
          />
        )}
        {pulse > 0 && (
          // The packet itself: a dot riding the head of the beam, so a
          // transfer reads as something moving rather than just a colour.
          <motion.circle
            key={`packet-${pulse}`}
            r={4.5}
            style={{
              fill: gradientStartColor,
              filter: `drop-shadow(0 0 6px ${gradientStartColor})`,
            }}
            initial={{ cx: a.x, cy: a.y, opacity: 0 }}
            animate={{ cx: b.x, cy: b.y, opacity: [0, 1, 1, 0] }}
            transition={{
              duration,
              ease: [0.45, 0, 0.55, 1],
              opacity: { duration, times: [0, 0.1, 0.85, 1] },
            }}
          />
        )}
        <defs>
          <motion.linearGradient
            key={pulse}
            id={id}
            gradientUnits="userSpaceOnUse"
            initial={{
              x1: start.x,
              y1: start.y,
              x2: start.x - ux * band,
              y2: start.y - uy * band,
            }}
            animate={{
              x1: end.x,
              y1: end.y,
              x2: end.x - ux * band,
              y2: end.y - uy * band,
            }}
            transition={{ duration, ease: [0.45, 0, 0.55, 1] }}
          >
            {/* Colours go through style, not the stop-color attribute:
                presentation attributes do not resolve CSS variables. */}
            <stop style={{ stopColor: gradientStartColor, stopOpacity: 0 }} />
            <stop style={{ stopColor: gradientStartColor }} />
            <stop offset="32.5%" style={{ stopColor: gradientStopColor }} />
            <stop
              offset="100%"
              style={{ stopColor: gradientStopColor, stopOpacity: 0 }}
            />
          </motion.linearGradient>
        </defs>
      </svg>

      {caption && (
        <span
          // Under the line in a row; to its left when stacked.
          className={`pointer-events-none absolute z-10 text-[10px] tracking-wide whitespace-nowrap uppercase ${
            horizontal
              ? "-translate-x-1/2"
              : "-translate-x-full -translate-y-1/2"
          }`}
          style={{
            left: horizontal ? mid.x : mid.x - 14,
            top: horizontal ? mid.y + 16 : mid.y,
            color: "var(--text-muted)",
          }}
        >
          {caption}
        </span>
      )}

      <AnimatePresence>
        {label && pulse > 0 && (
          <motion.span
            key={`${pulse}-${label}`}
            // Beside the line, not on it, so the line stays visible end to
            // end: above it in a row, to its right when stacked.
            className={`tabular pointer-events-none absolute z-20 rounded-full px-2.5 py-1 text-[11px] font-medium whitespace-nowrap shadow-sm ${
              horizontal
                ? "-translate-x-1/2 -translate-y-full"
                : "-translate-y-1/2"
            }`}
            style={{
              left: horizontal ? mid.x : mid.x + 14,
              top: horizontal ? mid.y - 10 : mid.y,
              background: "var(--surface-1)",
              color: "var(--text-primary)",
              border: "1px solid var(--border)",
            }}
            initial={{ opacity: 0, y: 6 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -6 }}
            transition={{ duration: 0.18 }}
          >
            {label}
          </motion.span>
        )}
      </AnimatePresence>
    </>
  );
}
