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
                    ids = torch.randint(0, 1000, (1, seq_len))
                    for _ in range(args.repeats):
                        runner.forward(ids)
                    print(f"  seq_len={seq_len} done", file=sys.stderr)
            records = profiler.records
    finally:
        for server in servers:
            server.stop(grace=None)

    for seq_len in seq_lens:
        subset = [r for r in records if r.seq_len == seq_len]
        summaries = summarize_by_hop(subset)
        print(f"\n=== seq_len = {seq_len} ===")
        print(format_table(summaries))
        print(f"\nverdict: {bandwidth_verdict(summaries)}")

    last = [r for r in records if r.seq_len == seq_lens[-1] and r.hop > 0]
    compute_ms = (sum(r.compute_ns for r in last) / len(last) / 1e6) if last else 0.0
    project(config.model_id, seq_lens[-1], args.link_mbps, args.rtt_ms, compute_ms)

    print(f"\nrecords written to {Path(args.out).resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
