import { Layers, Network, ShieldCheck } from "lucide-react";

const STEPS = [
  {
    icon: <Layers size={16} />,
    title: "The model is cut into stages",
    text: "Too big for most laptops, so each stage is a slice of its layers. A question passes through every stage, in order.",
  },
  {
    icon: <Network size={16} />,
    title: "Every stage lives on several devices",
    text: "Devices lend what they have. A question uses one device from each column, picked by how busy and how fast each is.",
  },
  {
    icon: <ShieldCheck size={16} />,
    title: "Losing a device is survivable",
    text: "If one disappears, even mid-answer, the question carries on through another copy and the pool rebuilds the missing one.",
  },
];

/**
 * Three sentences that say what the page is about, for someone seeing the
 * project for the first time. A native <details> so it can be folded away
 * without any state to keep in step.
 */
export default function Explainer() {
  return (
    <details
      open
      className="group rounded-xl px-4 py-3"
      style={{
        background: "var(--surface-1)",
        border: "1px solid var(--border)",
      }}
    >
      <summary
        className="flex cursor-pointer list-none items-center justify-between text-xs font-medium"
        style={{ color: "var(--text-secondary)" }}
      >
        What am I looking at?
        <span
          className="text-[11px] font-normal group-open:hidden"
          style={{ color: "var(--text-muted)" }}
        >
          Show
        </span>
        <span
          className="hidden text-[11px] font-normal group-open:inline"
          style={{ color: "var(--text-muted)" }}
        >
          Hide
        </span>
      </summary>
      <ol className="mt-3 grid gap-4 sm:grid-cols-3">
        {STEPS.map((step, i) => (
          <li key={step.title} className="flex gap-3">
            <span
              className="mt-0.5 flex size-7 shrink-0 items-center justify-center rounded-full"
              style={{
                background: "var(--page-plane)",
                color: "var(--text-secondary)",
              }}
              aria-hidden
            >
              {step.icon}
            </span>
            <div>
              <div className="text-sm font-medium">
                <span
                  className="tabular mr-1.5"
                  style={{ color: "var(--text-muted)" }}
                >
                  {i + 1}.
                </span>
                {step.title}
              </div>
              <p
                className="mt-0.5 text-xs leading-relaxed"
                style={{ color: "var(--text-secondary)" }}
              >
                {step.text}
              </p>
            </div>
          </li>
        ))}
      </ol>
    </details>
  );
}
