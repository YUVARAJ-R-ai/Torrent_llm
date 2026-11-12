/**
 * The few building blocks every screen uses. Screens compose these instead of
 * styling from scratch, so the app stays one visual system.
 */

import type { ButtonHTMLAttributes, ReactNode } from "react";

export function Card({ title, actions, children }: { title?: string; actions?: ReactNode; children: ReactNode }) {
  return (
    <section
      className="rounded-lg border p-4"
      style={{ background: "var(--surface-1)", borderColor: "var(--border)" }}
    >
      {(title || actions) && (
        <header className="mb-3 flex items-center justify-between gap-3">
          {title && (
            <h2 className="text-sm font-semibold" style={{ color: "var(--text-primary)" }}>
              {title}
            </h2>
          )}
          {actions}
        </header>
      )}
      {children}
    </section>
  );
}

export type Tone = "good" | "warn" | "critical" | "neutral" | "accent";

const TONE: Record<Tone, string> = {
  good: "var(--status-good)",
  warn: "var(--status-warn)",
  critical: "var(--status-critical)",
  neutral: "var(--text-muted)",
  accent: "var(--accent)",
};

export function Pill({ tone = "neutral", children }: { tone?: Tone; children: ReactNode }) {
  return (
    <span className="inline-flex items-center gap-1.5 text-xs" style={{ color: "var(--text-secondary)" }}>
      <span aria-hidden className="inline-block h-2 w-2 rounded-full" style={{ background: TONE[tone] }} />
      {children}
    </span>
  );
}

type ButtonKind = "primary" | "secondary" | "danger" | "ghost";

export function Button({
  kind = "secondary",
  className = "",
  ...props
}: ButtonHTMLAttributes<HTMLButtonElement> & { kind?: ButtonKind }) {
  const style: Record<ButtonKind, React.CSSProperties> = {
    primary: { background: "var(--accent)", color: "var(--accent-text)", borderColor: "transparent" },
    secondary: { background: "var(--surface-1)", color: "var(--text-primary)", borderColor: "var(--border)" },
    danger: { background: "var(--surface-1)", color: "var(--status-critical)", borderColor: "var(--border)" },
    ghost: { background: "transparent", color: "var(--text-secondary)", borderColor: "transparent" },
  };
  return (
    <button
      type="button"
      {...props}
      className={`inline-flex items-center gap-1.5 rounded-md border px-3 py-1.5 text-sm font-medium disabled:cursor-not-allowed disabled:opacity-50 ${className}`}
      style={{ ...style[kind], ...props.style }}
    />
  );
}

export function Stat({ label, value, hint }: { label: string; value: ReactNode; hint?: string }) {
  return (
    <div>
      <div className="text-xs" style={{ color: "var(--text-muted)" }}>
        {label}
      </div>
      <div className="mt-0.5 text-xl font-semibold tabular-nums">{value}</div>
      {hint && (
        <div className="text-xs" style={{ color: "var(--text-muted)" }}>
          {hint}
        </div>
      )}
    </div>
  );
}

export function Empty({ title, children }: { title: string; children?: ReactNode }) {
  return (
    <div className="py-10 text-center">
      <div className="text-sm font-medium">{title}</div>
      {children && (
        <div className="mx-auto mt-1 max-w-md text-sm" style={{ color: "var(--text-secondary)" }}>
          {children}
        </div>
      )}
    </div>
  );
}

export function ErrorNote({ children }: { children: ReactNode }) {
  return (
    <p role="alert" className="text-sm" style={{ color: "var(--status-critical)" }}>
      {children}
    </p>
  );
}

/** A plain table: header row, zebra-free, numbers right-aligned by the caller. */
export function Table({ head, children }: { head: ReactNode[]; children: ReactNode }) {
  return (
    <div className="overflow-x-auto">
      <table className="w-full text-sm">
        <thead>
          <tr style={{ color: "var(--text-muted)" }}>
            {head.map((h, i) => (
              <th key={i} className="border-b px-2 py-1.5 text-left text-xs font-medium" style={{ borderColor: "var(--gridline)" }}>
                {h}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>{children}</tbody>
      </table>
    </div>
  );
}

export function Td({ children, right = false, mono = false }: { children: ReactNode; right?: boolean; mono?: boolean }) {
  return (
    <td
      className={`border-b px-2 py-1.5 ${right ? "text-right tabular-nums" : ""} ${mono ? "mono text-xs" : ""}`}
      style={{ borderColor: "var(--gridline)" }}
    >
      {children}
    </td>
  );
}

export function PageTitle({ title, children }: { title: string; children?: ReactNode }) {
  return (
    <div className="mb-4">
      <h1 className="text-lg font-semibold">{title}</h1>
      {children && (
        <p className="mt-0.5 text-sm" style={{ color: "var(--text-secondary)" }}>
          {children}
        </p>
      )}
    </div>
  );
}
