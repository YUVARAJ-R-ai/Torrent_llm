#!/usr/bin/env python
"""Measure per-hop bandwidth and latency across a shard chain (issue #6).

Runs every shard in-process on loopback by default, which measures
serialisation, gRPC framing and compute honestly but gives a near-zero network
cost. That is the point of the ``--project`` output: loopback establishes the
compute side of the budget, and the projection carries it onto a real link.

    python scripts/profile_chain.py --model Qwen/Qwen3-0.6B --seq-lens 128,512,1024
    python scripts/profile_chain.py --config configs/lan-2machine.yaml   # real rig
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

from torrent_llm.codec import get_codec
from torrent_llm.profile import (
    KNOWN_HIDDEN_SIZES,
    HopProfiler,
    RunMetadata,
    bandwidth_verdict,
    format_table,
    new_run_id,
    prefill_budget,
    summarize_by_hop,
)
from torrent_llm.runner import ChainRunner, TopologyConfig
from torrent_llm.shard import load_shard, num_layers_of
from torrent_llm.transport import serve

#: Upper bound on synthetic profiling token ids. Far below any real
#: tokenizer's vocabulary, so a small-vocab model does not index past its
#: embedding table and fail with a bare "index out of range in self".
PROFILE_TOKEN_CEILING = 100


def build_local_chain(model_id: str, num_shards: int, dtype: str, device: str):
    """Start ``num_shards`` servers in this process and return a topology for them."""
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
        server, port = serve(runtime, get_codec("raw"), host="127.0.0.1", model_id=model_id)
        servers.append(server)
        addresses.append(f"127.0.0.1:{port}")
        print(f"  {spec} on 127.0.0.1:{port}", file=sys.stderr)

    config = TopologyConfig.from_dict(
        {
            "model_id": model_id,
            "num_layers": layers,
            "dtype": dtype,
            "nodes": [{"address": a} for a in addresses],
        }
    )
    return config, servers


def project(model_id: str, seq_len: int, link_mbps: float, rtt_ms: float, compute_ms: float):
    """Print what this hop would cost on a real link, at this and larger model sizes."""
    print(
        f"\nProjected prefill hop at seq_len={seq_len} on {link_mbps:.0f} Mbps, "
        f"{rtt_ms:.0f} ms RTT, {compute_ms:.1f} ms shard compute:\n"
    )
    header = (
        f"{'model':<32}{'hidden':>8}{'payload':>12}{'wire ms':>10}{'total ms':>10}{'wire %':>9}"
    )
    print(header)
    print("-" * len(header))
    for name, hidden in KNOWN_HIDDEN_SIZES.items():
        budget = prefill_budget(
            hidden_size=hidden,
            seq_len=seq_len,
            link_mbps=link_mbps,
            rtt_ms=rtt_ms,
            compute_ms=compute_ms,
        )
        print(
            f"{name:<32}{hidden:>8}"
            f"{budget.payload_bytes / 1024 / 1024:>10.1f} MiB"
            f"{budget.wire_ms:>10.1f}{budget.total_ms:>10.1f}"
            f"{budget.wire_share:>8.0%}"
            + ("  <- bandwidth-bound" if budget.is_bandwidth_bound else "")
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--config", help="use an existing topology instead of local servers")
    parser.add_argument("--shards", type=int, default=2)
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seq-lens", default="128,512,1024")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--out", default="runs/profile.jsonl")
    parser.add_argument("--link-mbps", type=float, default=100.0, help="link to project onto")
    parser.add_argument("--rtt-ms", type=float, default=30.0, help="RTT to project onto")
    parser.add_argument(
        "--project-compute-ms",
        type=float,
        default=None,
        help=(
            "shard compute to assume in the projection, instead of the measured "
            "value. Use this when profiling on CPU: CPU prefill is orders of "
            "magnitude slower than the GPU a real node would use, and feeding "
            "that number into the projection makes every hop look compute-bound"
        ),
    )
    parser.add_argument(
        "--decode-tokens",
        type=int,
        default=0,
        help=(
            "also run cached generation (issue #24) for this many new tokens after "
            "prefilling the largest --seq-lens context, to measure real decode-regime "
            "payloads instead of only projecting them. 0 (the default) skips this."
        ),
    )
    args = parser.parse_args(argv)

    seq_lens = [int(s) for s in args.seq_lens.split(",")]
    servers = []

    if args.config:
        config = TopologyConfig.from_file(args.config)
    else:
        print(f"starting {args.shards} shards of {args.model} locally", file=sys.stderr)
        config, servers = build_local_chain(args.model, args.shards, args.dtype, args.device)

    metadata = RunMetadata(
        run_id=new_run_id(),
        model_id=config.model_id,
        dtype=config.dtype,
        codec=config.codec,
        num_shards=len(config.nodes),
        note=f"seq_lens={seq_lens} repeats={args.repeats} device={args.device}",
    )

    try:
        with HopProfiler(path=args.out, metadata=metadata) as profiler:
            with ChainRunner(config, profiler=profiler) as runner:
                for seq_len in seq_lens:
                    ids = torch.randint(0, PROFILE_TOKEN_CEILING, (1, seq_len))
                    for _ in range(args.repeats):
                        # Ask the tail shard for one position of logits. The
                        # full seq x vocab tensor is not what we are profiling
                        # and on a large vocab it dwarfs every activation hop.
                        runner.forward(ids, logits_keep_last=1)
                    print(f"  seq_len={seq_len} done", file=sys.stderr)

                if args.decode_tokens > 0:
                    # One cached generation, run after the largest prefill
                    # context, so the decode steps measured here have a
                    # realistic amount of history behind them rather than
                    # decoding from an empty prompt. generate() tags its own
                    # records prefill/decode; only the decode ones are new
                    # here since the prefill step duplicates what the loop
                    # above already measured.
                    print(
                        f"  running {args.decode_tokens} cached decode steps "
                        f"after seq_len={seq_lens[-1]} context",
                        file=sys.stderr,
                    )
                    decode_prompt = torch.randint(0, PROFILE_TOKEN_CEILING, (1, seq_lens[-1]))
                    runner.generate(decode_prompt, max_new_tokens=args.decode_tokens + 1)
            records = profiler.records
    finally:
        for server in servers:
            server.stop(grace=None)

    for seq_len in seq_lens:
        subset = [r for r in records if r.seq_len == seq_len and r.phase == "prefill"]
        summaries = summarize_by_hop(subset)
        print(f"\n=== prefill, seq_len = {seq_len} ===")
        print(format_table(summaries))
        print(f"\nverdict: {bandwidth_verdict(summaries)}")

    if args.decode_tokens > 0:
        decode_records = [r for r in records if r.phase == "decode"]
        decode_summaries = summarize_by_hop(decode_records)
        print(f"\n=== cached decode, after seq_len = {seq_lens[-1]} context ===")
        print(format_table(decode_summaries))
        print(f"\nverdict: {bandwidth_verdict(decode_summaries)}")

    last = [r for r in records if r.seq_len == seq_lens[-1] and r.hop > 0]
    measured_ms = (sum(r.compute_ns for r in last) / len(last) / 1e6) if last else 0.0
    compute_ms = args.project_compute_ms if args.project_compute_ms is not None else measured_ms

    if args.device == "cpu" and args.project_compute_ms is None:
        print(
            f"\nWARNING: projecting with {measured_ms:.0f} ms of measured CPU compute. "
            "CPU prefill is orders of magnitude slower than a real node's GPU, so this "
            "makes every hop look compute-bound. Re-run with --device cuda, or pass "
            "--project-compute-ms with a realistic figure.",
            file=sys.stderr,
        )

    project(config.model_id, seq_lens[-1], args.link_mbps, args.rtt_ms, compute_ms)

    print(f"\nrecords written to {Path(args.out).resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
