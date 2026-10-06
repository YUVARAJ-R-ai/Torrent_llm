"""Unified CLI runner for RecursiveMAS multi-agent execution across sharded layers (Issue #11).

Runs Planner, Solver, and Critic roles communicating across network hops via RecursiveLink,
tracking wire bytes, compression ratios, and per-agent latency.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

from torrent_llm.agent import (
    MASRunResult,
    RecursiveLink,
    RecursiveMASOrchestrator,
)
from torrent_llm.codec import get_codec

logger = logging.getLogger("torrent_llm.mas")

DEFAULT_TASK = (
    "Design a fault-tolerant distributed tensor sharding strategy with "
    "low-rank activation compression for edge peer-to-peer LLM inference."
)


def build_arg_parser() -> argparse.ArgumentParser:
    """Build command line argument parser for multi-agent CLI."""
    parser = argparse.ArgumentParser(
        description="Run RecursiveMAS multi-agent reasoning across sharded model layers.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    task_group = parser.add_mutually_exclusive_group()
    task_group.add_argument(
        "--task",
        type=str,
        default=DEFAULT_TASK,
        help="Goal or task description to decompose and solve.",
    )
    task_group.add_argument(
        "--prompt",
        type=str,
        dest="task",
        help="Alias for --task.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen3-0.6B",
        help="HuggingFace model ID or local checkpoint path.",
    )
    parser.add_argument(
        "--codec",
        type=str,
        choices=["lowrank", "raw"],
        default="lowrank",
        help="Activation compression codec on the wire.",
    )
    parser.add_argument(
        "--codec-rank",
        type=int,
        default=128,
        help="Latent bottleneck rank for lowrank codec.",
    )
    parser.add_argument(
        "--shards",
        type=int,
        default=2,
        help="Number of network shards to partition model across.",
    )
    parser.add_argument(
        "--max-rounds",
        type=int,
        default=3,
        help="Maximum iterative refinement rounds between Solver and Critic.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=64,
        help="Maximum generated tokens per role step.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Optional path to export JSON execution summary.",
    )
    parser.add_argument(
        "--mock",
        "--offline",
        action="store_true",
        dest="offline",
        help="Run in offline mock mode without downloading weights or starting live gRPC servers.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable debug logging.",
    )
    return parser


def run_multiagent(args: argparse.Namespace) -> MASRunResult:
    """Execute the multi-agent pipeline from parsed CLI arguments."""
    codec_args: dict[str, Any] = {}
    if args.codec == "lowrank":
        codec_args["rank"] = args.codec_rank
    codec = get_codec(args.codec, **codec_args)

    runner = None
    tokenizer = None

    if not args.offline:
        try:
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(args.model)
        except Exception as e:
            logger.warning(
                "Could not load AutoTokenizer (%s); using offline tokenizer fallback.", e
            )
            tokenizer = None

    # Instantiate latent communication link and orchestrator
    link = RecursiveLink(
        runner=runner,
        codec=codec,
        rank=args.codec_rank,
        hidden_size=2048,
        tokenizer=tokenizer,
    )

    orchestrator = RecursiveMASOrchestrator(
        link=link,
        max_rounds=args.max_rounds,
        max_new_tokens=args.max_new_tokens,
    )

    print("=" * 80)
    print("RecursiveMAS Sharded Multi-Agent Run")
    print(
        f"Model: {args.model} | Codec: {args.codec} (rank={args.codec_rank}) | "
        f"Shards: {args.shards}"
    )
    print(f"Task: {args.task}")
    print("=" * 80)
    print()

    result = orchestrator.run(task=args.task)

    # Print step-by-step role execution logs
    for step in result.step_logs:
        role_label = step.role.upper()
        if step.role in ("solver", "critic"):
            role_label = f"{role_label} (Round {step.round_index})"
        print(
            f"[{step.step_number}: {role_label}]  Latency: {step.latency_s:.3f}s | "
            f"Wire Bytes: {step.wire_bytes:,}"
        )
        print(step.content)
        print("-" * 80)

    print()
    print(result.format_summary())

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(result.as_dict(), f, indent=2)
        print(f"\nExecution record saved to: {args.out}")

    return result


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    run_multiagent(args)


if __name__ == "__main__":
    main()
