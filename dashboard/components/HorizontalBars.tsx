"use client";

/**
 * Horizontal bar chart, single or stacked.
 *
 * Horizontal rather than vertical because every category here is a text label
 * ("hop 0 · prefill", "uncached") that would need rotating on a vertical axis,
 * and rotated axis labels are the most common way a small chart becomes
 * unreadable.
 *
 * Written as plain SVG rather than pulling in a chart library: two bar charts
 * do not justify a dependency, and the mark specs (4px rounded data-ends, a 2px
 * surface gap between stacked segments, recessive gridlines) are easier to hit
 * directly than to configure around.
 */

import { useEffect, useId, useRef, useState } from "react";

export interface BarSegment {
  /** Which categorical slot this segment uses. Assigned by role, never by rank. */
  series: 1 | 2;
  value: number;
  label: string;
}

export interface BarRow {
  label: string;
  /** Secondary line under the label — units, sample counts, whatever qualifies the row. */
  sublabel?: string;
  segments: BarSegment[];
  /** Direct label at the end of the bar. Selective: the total, not every segment. */
  valueLabel: string;
}

interface Props {
  rows: BarRow[];
  /** Formats a raw value for tooltips. */
  format: (value: number) => string;
  /**
   * Shared maximum across several charts. Pass this when two charts must be
   * read against each other; leave undefined to scale each chart to its own
   * data. Never use a shared max across phases — prefill and decode payloads
   * differ by three orders of magnitude and one would vanish.
   */
  max?: number;
  /** Legend entries, required whenever a row has more than one segment. */
  legend?: { series: 1 | 2; label: string }[];
}

const BAR_HEIGHT = 18;
const ROW_GAP = 30;
const LABEL_WIDTH = 132;
const VALUE_WIDTH = 92;
const SEGMENT_GAP = 2; // surface-colored gap between stacked segments
const TOP_PAD = 8;

export default function HorizontalBars({ rows, format, max, legend }: Props) {
  const gradientId = useId();
  const containerRef = useRef<HTMLDivElement>(null);
  // The viewBox is sized to the container's real pixel width so one SVG unit is
  // one pixel. The alternative -- a fixed viewBox with
  // preserveAspectRatio="none" -- stretches the coordinate system to fit, which
  // also stretches the glyphs: labels come out horizontally smeared at any
  // container width other than the viewBox width.
  const [width, setWidth] = useState(600);
  const [hover, setHover] = useState<{
    x: number;
    y: number;
    text: string;
  } | null>(null);

  useEffect(() => {
    const element = containerRef.current;
    if (!element) return;
    const observer = new ResizeObserver(([entry]) => {
      // Floor to avoid a subpixel width causing a re-render loop.
      setWidth(Math.max(320, Math.floor(entry.contentRect.width)));
    });
    observer.observe(element);
    return () => observer.disconnect();
  }, []);

  const rowTotal = (row: BarRow) =>
    row.segments.reduce((sum, s) => sum + s.value, 0);
  const dataMax = max ?? Math.max(...rows.map(rowTotal), 1);

  const height = TOP_PAD + rows.length * ROW_GAP;
  const plotStart = LABEL_WIDTH;

  return (
    <div ref={containerRef} className="relative w-full">
      {legend && legend.length > 0 && (
        <div className="mb-3 flex flex-wrap gap-x-4 gap-y-1">
          {legend.map((entry) => (
            <span
              key={entry.series}
              className="flex items-center gap-1.5 text-xs"
              style={{ color: "var(--text-secondary)" }}
            >
              <span
                aria-hidden
                className="inline-block h-2.5 w-2.5 rounded-[2px]"
                style={{ background: `var(--series-${entry.series})` }}
              />
              {entry.label}
            </span>
          ))}
        </div>
      )}

      <svg
        width={width}
        height={height}
        viewBox={`0 0 ${width} ${height}`}
        role="img"
        aria-label="Bar chart; the same figures are listed in the table below."
      >
        <defs>
          {/* Clip keeps the rounded data-end from overhanging the plot area on
              a full-width bar. */}
          <clipPath id={`${gradientId}-plot`}>
            <rect
              x={plotStart}
              y={0}
              width={width - plotStart - VALUE_WIDTH}
              height={height}
            />
          </clipPath>
        </defs>

        {rows.map((row, rowIndex) => {
          const y = TOP_PAD + rowIndex * ROW_GAP;
          const plotWidth = width - plotStart - VALUE_WIDTH;
          const total = rowTotal(row);
          let cursor = plotStart;

          return (
            <g key={row.label}>
              <text
                x={0}
                y={y + BAR_HEIGHT / 2 + 1}
                dominantBaseline="middle"
                fontSize={11}
                fill="var(--text-secondary)"
              >
                {row.label}
              </text>
              {row.sublabel && (
                <text
                  x={0}
                  y={y + BAR_HEIGHT / 2 + 12}
                  dominantBaseline="middle"
                  fontSize={9}
                  fill="var(--text-muted)"
                >
                  {row.sublabel}
                </text>
              )}

              {/* Track: shows the unused remainder so a short bar still reads as
                  a proportion rather than as a floating stub. */}
              <rect
                x={plotStart}
                y={y}
                width={plotWidth}
                height={BAR_HEIGHT}
                rx={4}
                fill="var(--gridline)"
                opacity={0.45}
              />

              <g clipPath={`url(#${gradientId}-plot)`}>
                {row.segments.map((segment, segmentIndex) => {
                  const isLast = segmentIndex === row.segments.length - 1;
                  const raw = (segment.value / dataMax) * plotWidth;
                  // Reserve the gap out of every segment but the last, so the
                  // stack still sums to the true total width. Named distinctly
                  // from the component's `width` state on purpose -- shadowing
                  // it here would be an easy thing to trip over later.
                  const segmentWidth = Math.max(
                    0,
                    isLast ? raw : raw - SEGMENT_GAP,
                  );
                  const x = cursor;
                  cursor += raw;

                  if (segmentWidth <= 0) return null;

                  return (
                    <rect
                      key={`${segment.series}-${segment.label}`}
                      x={x}
                      y={y}
                      width={segmentWidth}
                      height={BAR_HEIGHT}
                      rx={4}
                      fill={`var(--series-${segment.series})`}
                      onMouseEnter={(event) =>
                        setHover({
                          x: event.clientX,
                          y: event.clientY,
                          text: `${row.label} · ${segment.label}: ${format(segment.value)}`,
                        })
                      }
                      onMouseMove={(event) =>
                        setHover({
                          x: event.clientX,
                          y: event.clientY,
                          text: `${row.label} · ${segment.label}: ${format(segment.value)}`,
                        })
                      }
                      onMouseLeave={() => setHover(null)}
                    />
                  );
                })}
              </g>

              {/* Direct label: the row total, not a number on every segment. */}
              <text
                x={width - VALUE_WIDTH + 8}
                y={y + BAR_HEIGHT / 2 + 1}
                dominantBaseline="middle"
                fontSize={11}
                fill="var(--text-primary)"
                className="tabular"
              >
                {row.valueLabel}
              </text>

              {total === 0 && (
                <text
                  x={plotStart + 6}
                  y={y + BAR_HEIGHT / 2 + 1}
                  dominantBaseline="middle"
                  fontSize={10}
                  fill="var(--text-muted)"
                >
                  no data
                </text>
              )}
            </g>
          );
        })}
      </svg>

      {hover && (
        <div
          className="pointer-events-none fixed z-50 rounded-md px-2.5 py-1.5 text-xs shadow-lg"
          style={{
            left: hover.x + 12,
            top: hover.y + 12,
            background: "var(--surface-1)",
            color: "var(--text-primary)",
            border: "1px solid var(--border)",
          }}
        >
          {hover.text}
        </div>
      )}
    </div>
  );
}
