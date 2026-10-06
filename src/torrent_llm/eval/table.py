"""Rate-Distortion table reporting and formatting (Issue #10).

Formats ASCII tables displaying the trade-off between compression rank,
bandwidth reduction ratio, language modeling perplexity, downstream task accuracy,
and projected network speedup.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class RateDistortionRow:
    """One row in the Rate-Distortion evaluation table."""

    rank_label: str
    compression_ratio: float
    perplexity: float | None = None
    downstream_accuracy: float | None = None
    gsm8k_acc: float | None = None
    mmlu_acc: float | None = None
    speedup: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "rank": self.rank_label,
            "compression_ratio": round(self.compression_ratio, 2),
            "perplexity": round(self.perplexity, 2) if self.perplexity is not None else None,
            "accuracy": (
                round(self.downstream_accuracy * 100, 1)
                if self.downstream_accuracy is not None
                else None
            ),
            "gsm8k_acc": round(self.gsm8k_acc * 100, 1) if self.gsm8k_acc is not None else None,
            "mmlu_acc": round(self.mmlu_acc * 100, 1) if self.mmlu_acc is not None else None,
            "speedup": round(self.speedup, 2) if self.speedup is not None else None,
        }


def format_rate_distortion_table(
    rows: list[RateDistortionRow],
    *,
    model_id: str = "Qwen/Qwen3-0.6B",
    num_shards: int = 2,
    link_mbps: float = 100.0,
    rtt_ms: float = 20.0,
) -> str:
    """Format evaluation results into a clean ASCII rate-distortion table.

    Example layout:
    +-------------------------------------------------------------------------------+
    |                     Rate-Distortion Evaluation Summary                        |
    | Model: Qwen/Qwen3-0.6B | Shards: 2 | Link: 100.0 Mbps | RTT: 20.0 ms          |
    +-------+---------+------------+------------+------------+----------------------+
    | Rank  | Ratio   | Perplexity | GSM8K Acc  | MMLU Acc   | Speedup              |
    +-------+---------+------------+------------+------------+----------------------+
    | raw   |    1.0x |      14.23 |      60.0% |      80.0% | 1.00x (baseline)     |
    | 128   |    8.0x |      14.88 |      60.0% |      80.0% | 2.14x                |
    | 64    |   16.0x |      16.12 |      40.0% |      60.0% | 2.89x                |
    | 32    |   32.0x |      21.45 |      20.0% |      40.0% | 3.45x                |
    +-------+---------+------------+------------+------------+----------------------+
    """
    header_title = "Rate-Distortion Evaluation Summary"
    header_meta = (
        f"Model: {model_id} | Shards: {num_shards} | "
        f"Link: {link_mbps:.1f} Mbps | RTT: {rtt_ms:.1f} ms"
    )

    cols = [
        ("Rank", 7),
        ("Ratio", 9),
        ("Perplexity", 12),
        ("GSM8K Acc", 12),
        ("MMLU Acc", 12),
        ("Speedup", 22),
    ]

    total_width = sum(w for _, w in cols) + (len(cols) - 1) * 3 + 4
    border = "+" + "-" * (total_width - 2) + "+"

    lines = [
        border,
        f"| {header_title.center(total_width - 4)} |",
        f"| {header_meta.center(total_width - 4)} |",
    ]

    sep_line = "+" + "+".join("-" * (w + 2) for _, w in cols) + "+"
    lines.append(sep_line)

    header_cells = [f" {name:<{w}} " for name, w in cols]
    lines.append("|" + "|".join(header_cells) + "|")
    lines.append(sep_line)

    for row in rows:
        ratio_str = f"{row.compression_ratio:>6.1f}x"
        ppl_str = f"{row.perplexity:>10.2f}" if row.perplexity is not None else f"{'N/A':>10}"
        gsm_str = (
            f"{row.gsm8k_acc * 100:>9.1f}%" if row.gsm8k_acc is not None else f"{'N/A':>10}"
        )
        mmlu_str = (
            f"{row.mmlu_acc * 100:>9.1f}%" if row.mmlu_acc is not None else f"{'N/A':>10}"
        )

        if row.speedup is not None:
            if row.rank_label == "raw":
                speedup_str = f"{row.speedup:>5.2f}x (baseline)"
            else:
                speedup_str = f"{row.speedup:>5.2f}x"
        else:
            speedup_str = "N/A"

        row_cells = [
            f" {row.rank_label:<{cols[0][1]}} ",
            f" {ratio_str:<{cols[1][1]}} ",
            f" {ppl_str:<{cols[2][1]}} ",
            f" {gsm_str:<{cols[3][1]}} ",
            f" {mmlu_str:<{cols[4][1]}} ",
            f" {speedup_str:<{cols[5][1]}} ",
        ]
        lines.append("|" + "|".join(row_cells) + "|")

    lines.append(sep_line)
    return "\n".join(lines)
