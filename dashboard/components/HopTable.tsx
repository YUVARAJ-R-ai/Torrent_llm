/**
 * The chart data as a table.
 *
 * Not optional decoration: several palette slots sit below 3:1 contrast on the
 * light surface, and the data-viz relief rule requires either visible direct
 * labels or a table view. It is also the only view that shows every number at
 * once, which is what you actually want when a result looks wrong.
 */

import type { HopMetrics } from "@/lib/api";
import { formatBytes, formatMs, formatPercent } from "@/lib/format";

export default function HopTable({ hops }: { hops: HopMetrics[] }) {
  return (
    <div className="overflow-x-auto">
      <table className="w-full text-left text-xs tabular">
        <thead>
          <tr style={{ color: "var(--text-muted)" }}>
            <th className="py-1.5 pr-4 font-medium">hop</th>
            <th className="py-1.5 pr-4 font-medium">phase</th>
            <th className="py-1.5 pr-4 font-medium">seq</th>
            <th className="py-1.5 pr-4 text-right font-medium">sent</th>
            <th className="py-1.5 pr-4 text-right font-medium">compute</th>
            <th className="py-1.5 pr-4 text-right font-medium">transport</th>
            <th className="py-1.5 text-right font-medium">transport share</th>
          </tr>
        </thead>
        <tbody style={{ color: "var(--text-secondary)" }}>
          {hops.map((hop, index) => (
            <tr
              key={index}
              style={{ borderTop: "1px solid var(--gridline)" }}
            >
              <td className="py-1.5 pr-4">{hop.hop}</td>
              <td className="py-1.5 pr-4">{hop.phase}</td>
              <td className="py-1.5 pr-4">{hop.seq_len}</td>
              <td className="py-1.5 pr-4 text-right">
                {formatBytes(hop.sent_bytes)}
              </td>
              <td className="py-1.5 pr-4 text-right">
                {formatMs(hop.compute_ms)}
              </td>
              <td className="py-1.5 pr-4 text-right">
                {formatMs(hop.transport_ms)}
              </td>
              <td className="py-1.5 text-right">
                {formatPercent(hop.transport_share)}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
