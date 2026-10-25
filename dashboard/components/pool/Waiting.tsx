/**
 * What to show before any device has answered.
 *
 * Two different situations look the same from here, so it names both: the pool is
 * still starting (the model takes a while to load on every device), or nothing is
 * running. It says what to run, since the commonest fix is to start it.
 */
export default function WaitingForPool({ error }: { error: string | null }) {
  return (
    <div
      className="flex flex-col items-center gap-3 rounded-xl px-4 py-10 text-center"
      style={{
        background: "var(--page-plane)",
        border: "1px dashed var(--border)",
      }}
    >
      <div className="flex gap-1.5" aria-hidden>
        {[0, 1, 2].map((i) => (
          <span
            key={i}
            className="size-2 animate-pulse rounded-full"
            style={{
              background: "var(--text-muted)",
              animationDelay: `${i * 160}ms`,
            }}
          />
        ))}
      </div>
      <div className="text-sm font-medium">
        {error ? "No device in the pool is answering" : "Looking for the pool…"}
      </div>
      <p
        className="max-w-md text-xs leading-relaxed"
        style={{ color: "var(--text-secondary)" }}
      >
        {error
          ? "It is either still starting (the model takes a minute to load on every device) or it is not running."
          : "Asking the devices who is in the pool."}
      </p>
      {error && (
        <code
          className="rounded-md px-3 py-1.5 text-xs"
          style={{
            background: "var(--surface-1)",
            color: "var(--text-primary)",
          }}
        >
          scripts/pool.sh
        </code>
      )}
    </div>
  );
}
