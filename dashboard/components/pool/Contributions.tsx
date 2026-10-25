import type { PoolMember } from "@/lib/pool";

/** Compute a device has given to the pool, in seconds, since it started. */
const given = (m: PoolMember) =>
  m.hosting.reduce((sum, h) => sum + (h.busy_ms ?? 0), 0) / 1000;

/**
 * Who has given what, and who has used it.
 *
 * Each device reports the seconds it has spent computing for requests, and the
 * tokens of answers it has asked for. Set side by side they say whether a device
 * is carrying others or being carried, which is what a sharing scheme has to be
 * able to answer before it can be fair. This is accounting, not enforcement:
 * nothing is withheld from a device that has taken more than it gave.
 */
export default function Contributions({ members }: { members: PoolMember[] }) {
  const rows = members
    .map((m) => ({ m, gave: given(m), used: m.asked_tokens ?? 0 }))
    .filter((r) => r.gave > 0 || r.used > 0)
    .sort((a, b) => b.gave - a.gave);
  if (rows.length === 0) return null;
  const maxGave = Math.max(...rows.map((r) => r.gave), 0.001);
  const maxUsed = Math.max(...rows.map((r) => r.used), 1);

  return (
    <section
      className="rounded-2xl px-4 py-3.5"
      style={{
        background: "var(--surface-1)",
        border: "1px solid var(--border)",
      }}
    >
      <h2 className="text-xs font-medium">Who is carrying the pool</h2>
      <p
        className="mt-1 mb-3 text-[11px]"
        style={{ color: "var(--text-muted)" }}
      >
        Compute each device has given, and the answers it has asked for.
      </p>
      <ul className="flex flex-col gap-2.5">
        {rows.map(({ m, gave, used }) => (
          <li key={m.name} className="text-xs">
            <div className="mb-1 flex items-baseline justify-between gap-2">
              <span
                className="truncate"
                style={{ opacity: m.liveness === "down" ? 0.5 : 1 }}
              >
                {m.label || m.name}
              </span>
              <span
                className="tabular shrink-0"
                style={{ color: "var(--text-muted)" }}
              >
                gave {gave.toFixed(1)} s · asked {used} tokens
              </span>
            </div>
            <Bar value={gave / maxGave} color="var(--series-1)" label="given" />
            <Bar value={used / maxUsed} color="var(--series-2)" label="asked" />
          </li>
        ))}
      </ul>
    </section>
  );
}

function Bar({
  value,
  color,
  label,
}: {
  value: number;
  color: string;
  label: string;
}) {
  return (
    <div
      className="mt-0.5 h-1.5 overflow-hidden rounded-full"
      style={{ background: "var(--gridline)" }}
      role="img"
      aria-label={`${label}: ${(value * 100).toFixed(0)}% of the largest`}
    >
      <div
        className="h-full rounded-full transition-[width] duration-500"
        style={{
          width: `${Math.max(0, Math.min(1, value)) * 100}%`,
          background: color,
        }}
      />
    </div>
  );
}
