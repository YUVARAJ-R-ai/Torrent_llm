#!/usr/bin/env python
"""Evaluate compression quality and Rate-Distortion frontier across ranks (Issue #10).

Measures:
1. Language modeling perplexity (chunked & sliding window)
2. Downstream task accuracy (GSM8K math reasoning & MMLU multiple-choice)
3. Layer-depth sensitivity
4. Projected network speedup under realistic network links

Usage:
    python scripts/eval_quality.py --model Qwen/Qwen3-0.6B --ranks raw,128,64,32 --benchmark all
    python scripts/eval_quality.py --benchmark perplexity --ranks raw,128,64
    python scripts/eval_quality.py --benchmark mmlu-sample --ranks raw,64
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import torch

from torrent_llm.codec import get_codec
from torrent_llm.eval import (
    RateDistortionRow,
    SimpleCharTokenizer,
    compute_perplexity,
    evaluate_downstream,
    format_rate_distortion_table,
    get_synthetic_corpus,
)
from torrent_llm.profile import (
    KNOWN_HIDDEN_SIZES,
    prefill_budget,
)
from torrent_llm.runner import ChainRunner, TopologyConfig
from torrent_llm.shard import load_shard, num_layers_of
from torrent_llm.transport import serve

logger = logging.getLogger("torrent_llm.eval")


def parse_ranks(raw_ranks: str) -> list[str | int]:
    """Parse comma-separated rank specifications into strings or integers."""
    parsed: list[str | int] = []
    for r in raw_ranks.split(","):
        cleaned = r.strip()
        if not cleaned:
            continue
        if cleaned.lower() == "raw":
            parsed.append("raw")
        elif cleaned.isdigit():
            parsed.append(int(cleaned))
        else:
            raise ValueError(f"Invalid rank specifier: {cleaned!r}. Expected integer or 'raw'.")
    return parsed or ["raw", 128, 64]


def build_local_chain(
    model_id: str,
    num_shards: int,
    dtype: str = "float32",
    device: str = "cpu",
    codec_name: str = "raw",
    codec_args: dict[str, Any] | None = None,
) -> tuple[TopologyConfig, list[Any]]:
    """Start num_shards servers locally and return their TopologyConfig."""
    codec_args = codec_args or {}
    layers = num_layers_of(model_id)
    planning = TopologyConfig.from_dict(
        {
            "model_id": model_id,
            "num_layers": layers,
            "dtype": dtype,
            "nodes": [{"address": "127.0.0.1:0"} for _ in range(num_shards)],
        }
    )

    servers, addresses = [], []
    for spec in planning.shard_plan():
        runtime = load_shard(model_id, spec, device=device, dtype=getattr(torch, dtype))
        codec_obj = get_codec(codec_name, **codec_args)
        server, port = serve(runtime, codec_obj, host="127.0.0.1", model_id=model_id)
        servers.append(server)
        addresses.append(f"127.0.0.1:{port}")

    config_dict: dict[str, Any] = {
        "model_id": model_id,
        "num_layers": layers,
        "dtype": dtype,
        "nodes": [{"address": a} for a in addresses],
    }
    if codec_args:
        config_dict["codec"] = {"name": codec_name, **codec_args}
    else:
        config_dict["codec"] = codec_name

    return TopologyConfig.from_dict(config_dict), servers


@contextmanager
def local_chain_runner(
    model_id: str,
    num_shards: int,
    dtype: str = "float32",
    device: str = "cpu",
    codec_name: str = "raw",
    codec_args: dict[str, Any] | None = None,
) -> Generator[ChainRunner, None, None]:
    """Context manager spinning up local shards and yielding a ChainRunner."""
    config, servers = build_local_chain(
        model_id,
        num_shards,
        dtype=dtype,
        device=device,
        codec_name=codec_name,
        codec_args=codec_args,
    )
    try:
        with ChainRunner(config) as runner:
            yield runner
    finally:
        for s in servers:
            s.stop(grace=None)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Torrent-LLM Compression-Quality Evaluation Harness (Issue #10)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model",
        default="Qwen/Qwen3-0.6B",
        help="HuggingFace model ID or local path",
    )
    parser.add_argument(
        "--ranks",
        default="raw,128,64,32",
        help="Comma-separated list of ranks to evaluate (e.g., 'raw,256,128,64,32')",
    )
    parser.add_argument(
        "--benchmark",
        default="all",
        choices=["perplexity", "gsm8k-sample", "mmlu-sample", "all"],
        help="Benchmark suite to evaluate",
    )
    parser.add_argument(
        "--dataset",
        default=None,
        help="Optional path to custom text or JSONL dataset for evaluation",
    )
    parser.add_argument(
        "--out",
        default="runs/eval_quality.jsonl",
        help="Output destination path for jsonl evaluation records",
    )
    parser.add_argument(
        "--shards",
        type=int,
        default=2,
        help="Number of pipeline shards",
    )
    parser.add_argument(
        "--dtype",
        default="float32",
        choices=["float32", "float16", "bfloat16"],
        help="Computation dtype",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Device to place shards on (e.g. cpu, cuda:0)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=5,
        help="Max samples to evaluate per downstream benchmark",
    )
    parser.add_argument(
        "--max-chunk-size",
        type=int,
        default=256,
        help="Max chunk size for perplexity forward passes",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=None,
        help="Stride for sliding-window perplexity (None for non-overlapping)",
    )
    parser.add_argument(
        "--link-mbps",
        type=float,
        default=100.0,
        help="Simulated network link bandwidth (Mbps) for speedup projection",
    )
    parser.add_argument(
        "--rtt-ms",
        type=float,
        default=20.0,
        help="Simulated link round-trip time (ms) for speedup projection",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Preconfigured topology YAML file (skips spinning up fresh local shards)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose debug logging",
    )
    return parser


def load_tokenizer(model_id: str) -> Any:
    """Attempt to load AutoTokenizer; fallback to SimpleCharTokenizer offline."""
    try:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_id)
        logger.info("Loaded AutoTokenizer for %s", model_id)
        return tokenizer
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "AutoTokenizer failed for %s (%s); falling back to SimpleCharTokenizer",
            model_id,
            exc,
        )
        return SimpleCharTokenizer()


def compute_projected_speedup(
    model_id: str,
    rank: str | int,
    link_mbps: float,
    rtt_ms: float,
    dtype: str = "float32",
    seq_len: int = 256,
) -> tuple[float, float]:
    """Calculate compression ratio and projected network speedup factor."""
    hidden_size = KNOWN_HIDDEN_SIZES.get(model_id, 1024)
    if rank == "raw":
        ratio = 1.0
        return ratio, 1.0

    rank_int = int(rank)
    ratio = hidden_size / rank_int if rank_int > 0 else 1.0

    base_budget = prefill_budget(
        seq_len=seq_len,
        hidden_size=hidden_size,
        dtype=dtype,
        link_mbps=link_mbps,
        rtt_ms=rtt_ms,
        compute_ms=2.0,
    )
    comp_budget = base_budget.with_compression(ratio, codec_ms=0.5)
    speedup = comp_budget.speedup_from(base_budget)
    return ratio, speedup


def run_evaluation(args: argparse.Namespace) -> int:
    ranks = parse_ranks(args.ranks)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    tokenizer = load_tokenizer(args.model)

    # Prepare corpus for perplexity
    corpus: list[str] = []
    if args.dataset and Path(args.dataset).exists():
        p = Path(args.dataset)
        if p.suffix == ".jsonl":
            lines = [json.loads(line) for line in p.read_text().splitlines() if line.strip()]
            corpus = [item.get("text", "") for item in lines if "text" in item]
        else:
            corpus = [p.read_text()]
    else:
        corpus = get_synthetic_corpus()

    rows: list[RateDistortionRow] = []

    print(
        f"\nStarting Quality Evaluation Harness for {args.model}\n"
        f"  Ranks to evaluate: {ranks}\n"
        f"  Benchmarks: {args.benchmark}\n"
        f"  Shards: {args.shards} | Link: {args.link_mbps} Mbps | RTT: {args.rtt_ms} ms\n"
    )

    for rank_spec in ranks:
        rank_label = str(rank_spec)
        print(f"--- Evaluating Rank [{rank_label}] ---", file=sys.stderr)

        codec_name = "raw" if rank_spec == "raw" else "lowrank"
        codec_args = {} if rank_spec == "raw" else {"rank": int(rank_spec), "seed": 42}

        ratio, speedup = compute_projected_speedup(
            model_id=args.model,
            rank=rank_spec,
            link_mbps=args.link_mbps,
            rtt_ms=args.rtt_ms,
            dtype=args.dtype,
        )

        ppl_val: float | None = None
        gsm8k_acc: float | None = None
        mmlu_acc: float | None = None

        if args.config:
            # Use preconfigured runner from config
            config = TopologyConfig.from_file(args.config)
            runner_cm = ChainRunner(config)
        else:
            runner_cm = local_chain_runner(
                model_id=args.model,
                num_shards=args.shards,
                dtype=args.dtype,
                device=args.device,
                codec_name=codec_name,
                codec_args=codec_args,
            )

        with runner_cm as runner:
            # 1. Perplexity benchmark
            if args.benchmark in ("perplexity", "all"):
                ppl_res = compute_perplexity(
                    runner,
                    sequences=corpus,
                    tokenizer=tokenizer,
                    max_chunk_size=args.max_chunk_size,
                    stride=args.stride,
                )
                ppl_val = ppl_res.perplexity
                print(
                    f"  Perplexity: {ppl_val:.2f} "
                    f"(loss={ppl_res.loss:.3f}, tokens={ppl_res.total_tokens})",
                    file=sys.stderr,
                )

            # 2. Downstream benchmarks
            if args.benchmark in ("gsm8k-sample", "mmlu-sample", "all"):
                downstream_bench = (
                    args.benchmark
                    if args.benchmark in ("gsm8k-sample", "mmlu-sample")
                    else "all"
                )
                down_res = evaluate_downstream(
                    runner,
                    benchmark=downstream_bench,
                    tokenizer=tokenizer,
                    max_samples=args.max_samples,
                    dataset_file=args.dataset,
                )

                if "gsm8k-sample" in down_res:
                    gsm_res = down_res["gsm8k-sample"]
                    gsm8k_acc = gsm_res.accuracy
                    print(
                        f"  GSM8K Accuracy: {gsm8k_acc * 100:.1f}% "
                        f"({gsm_res.correct_samples}/{gsm_res.num_samples})",
                        file=sys.stderr,
                    )

                if "mmlu-sample" in down_res:
                    mmlu_res = down_res["mmlu-sample"]
                    mmlu_acc = mmlu_res.accuracy
                    print(
                        f"  MMLU Accuracy:  {mmlu_acc * 100:.1f}% "
                        f"({mmlu_res.correct_samples}/{mmlu_res.num_samples})",
                        file=sys.stderr,
                    )

        # Average downstream accuracy if multiple evaluated
        accs = [a for a in (gsm8k_acc, mmlu_acc) if a is not None]
        avg_acc = (sum(accs) / len(accs)) if accs else None

        row = RateDistortionRow(
            rank_label=rank_label,
            compression_ratio=ratio,
            perplexity=ppl_val,
            downstream_accuracy=avg_acc,
            gsm8k_acc=gsm8k_acc,
            mmlu_acc=mmlu_acc,
            speedup=speedup,
        )
        rows.append(row)

        # Write output record to JSONL
        record = {
            "timestamp": time.time(),
            "model_id": args.model,
            "rank": rank_spec,
            "compression_ratio": ratio,
            "speedup": speedup,
            "perplexity": ppl_val,
            "gsm8k_accuracy": gsm8k_acc,
            "mmlu_accuracy": mmlu_acc,
            "benchmark": args.benchmark,
            "num_shards": args.shards,
        }
        with open(out_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

    # Display Rate-Distortion ASCII Table
    table_str = format_rate_distortion_table(
        rows,
        model_id=args.model,
        num_shards=args.shards,
        link_mbps=args.link_mbps,
        rtt_ms=args.rtt_ms,
    )
    print("\n" + table_str + "\n")
    print(f"Results logged to: {out_path.resolve()}\n")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    return run_evaluation(args)


if __name__ == "__main__":
    sys.exit(main())
