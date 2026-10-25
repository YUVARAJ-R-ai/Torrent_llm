import Link from "next/link";

const PAGES = [
  { href: "/", label: "Pool" },
  { href: "/two-devices", label: "Two devices" },
  { href: "/details", label: "Details" },
];

/** The three views of the same project, as a quiet row of links. */
export default function Nav({ current }: { current: string }) {
  return (
    <nav className="flex gap-4 text-xs" aria-label="Views">
      {PAGES.map((page) =>
        page.href === current ? (
          <span
            key={page.href}
            aria-current="page"
            style={{ color: "var(--text-primary)" }}
          >
            {page.label}
          </span>
        ) : (
          <Link
            key={page.href}
            href={page.href}
            className="underline-offset-2 hover:underline"
            style={{ color: "var(--text-muted)" }}
          >
            {page.label}
          </Link>
        ),
      )}
    </nav>
  );
}
