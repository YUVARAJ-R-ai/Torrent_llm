/** Shared chrome for a dashboard section: a titled surface with optional note. */
export default function Panel({
  title,
  note,
  children,
  action,
}: {
  title: string;
  note?: string;
  children: React.ReactNode;
  action?: React.ReactNode;
}) {
  return (
    <section
      className="rounded-xl p-5"
      style={{
        background: "var(--surface-1)",
        border: "1px solid var(--border)",
      }}
    >
      <div className="mb-4 flex items-start justify-between gap-4">
        <div>
          <h2
            className="text-sm font-semibold"
            style={{ color: "var(--text-primary)" }}
          >
            {title}
          </h2>
          {note && (
            <p
              className="mt-1 max-w-2xl text-xs leading-relaxed"
              style={{ color: "var(--text-secondary)" }}
            >
              {note}
            </p>
          )}
        </div>
        {action}
      </div>
      {children}
    </section>
  );
}
