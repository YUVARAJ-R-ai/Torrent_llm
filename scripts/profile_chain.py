#!/usr/bin/env python
"""Measure per-hop bandwidth, latency, and compression across a shard chain (issue #6, #9).

Runs every shard in-process on loopback by default, which measures
serialisation, gRPC framing, and compute honestly but gives a near-zero network
cost. The projection carries these measurements onto a real link.

With ``--compare-codecs`` (issue #9), the harness runs both raw uncompressed
and low-rank compressed passes side-by-side, reporting:
- Before/after bytes on wire per hop
- Latency overhead of encode/decode
- Net latency and speedup under real link constraints (bandwidth-bound vs compute-bound)

Usage:
    python scripts/profile_chain.py --model Qwen/Qwen3-0.6B --seq-lens 128,512,1024
    python scripts/profile_chain.py --compare-codecs --codec-rank 128
    python scripts/profile_chain.py --config configs/local-2shard-lowrank.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import torch

from torrent_llm.codec import get_codec
from torrent_llm.profile import (
    KNOWN_HIDDEN_SIZES,
    HopProfiler,
    HopRecord,
    HopSummary,
    RunMetadata,
    bandwidth_verdict,
    format_table,
    new_run_id,
    prefill_budget,
    summarize_by_hop,
    transfer_ms,
)
from torrent_llm.runner import ChainRunner, TopologyConfig
from torrent_llm.shard import load_shard, num_layers_of
from torrent_llm.transport import serve

#: Upper bound on synthetic profiling token ids. Far below any real
#: tokenizer's vocabulary, so a small-vocab model does not index past its
#: embedding table and fail with a bare "index out of range in self".
PROFILE_TOKEN_CEILING = 100


def build_local_chain(
    model_id: str,
    num_shards: int,
    dtype: str,
    device: str,
    codec_name: str = "raw",
    codec_args: dict[str, Any] | None = None,
) -> tuple[TopologyConfig, list[Any]]:
    """Start ``num_shards`` servers in this process and return a topology for them."""
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
        print(f"  {spec} on 127.0.0.1:{port} [{codec_name}]", file=sys.stderr)

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

    config = TopologyConfig.from_dict(config_dict)
    return config, servers


def format_comparison_table(
    raw_summaries: list[HopSummary],
    lr_summaries: list[HopSummary],
    link_mbps: float,
    rtt_ms: float,
) -> str:
    """Side-by-side comparison table showing bandwidth reduction and net latency."""
    raw_by_hop = {s.hop: s for s in raw_summaries}
    lr_by_hop = {s.hop: s for s in lr_summaries}

    rows = []
    for hop in sorted(set(raw_by_hop) | set(lr_by_hop)):
        raw_s = raw_by_hop.get(hop)
        lr_s = lr_by_hop.get(hop)
        if not raw_s or not lr_s:
            continue

        raw_kib = raw_s.median_sent_bytes / 1024
        lr_kib = lr_s.median_sent_bytes / 1024
        reduction = (raw_kib / lr_kib) if lr_kib > 0 else 1.0

        codec_overhead_ms = lr_s.median_codec_overhead_ms
        raw_wire_ms = transfer_ms(int(raw_s.median_sent_bytes), link_mbps)
        lr_wire_ms = transfer_ms(int(lr_s.median_sent_bytes), link_mbps)

        raw_net_ms = raw_wire_ms + rtt_ms + raw_s.median_compute_ms
        lr_net_ms = lr_wire_ms + rtt_ms + lr_s.median_compute_ms + codec_overhead_ms

        speedup = raw_net_ms / lr_net_ms if lr_net_ms > 0 else 1.0
        is_bw_bound = raw_wire_ms > (rtt_ms + raw_s.median_compute_ms)
        regime = "BW-bound" if is_bw_bound else "Compute-bound"

        rows.append(
            {
                "hop": hop,
                "raw_KiB": round(raw_kib, 1),
                "comp_KiB": round(lr_kib, 1),
                "ratio": f"{reduction:.1f}x",
                "codec_ms": round(codec_overhead_ms, 3),
                "raw_net_ms": round(raw_net_ms, 2),
                "comp_net_ms": round(lr_net_ms, 2),
                "speedup": f"{speedup:.2f}x",
                "regime": regime,
            }
        )

    if not rows:
        return "(no common hops to compare)"

    cols = list(rows[0])
    widths = {c: max(len(c), *(len(str(r[c])) for r in rows)) for c in cols}
    header = "  ".join(c.rjust(widths[c]) for c in cols)
    rule = "  ".join("-" * widths[c] for c in cols)
    lines = ["  ".join(str(r[c]).rjust(widths[c]) for c in cols) for r in rows]
    return "\n".join([header, rule, *lines])


def project_comparison(
    seq_len: int,
    link_mbps: float,
    rtt_ms: float,
    compute_ms: float,
    ratio: float = 8.0,
    codec_ms: float = 0.5,
) -> None:
    """Print projected net latency comparison across models under stated link."""
    print(
        f"\nProjected prefill hop comparison at seq_len={seq_len} on {link_mbps:.0f} Mbps, "
        f"{rtt_ms:.0f} ms RTT, {compute_ms:.1f} ms compute "
        f"(ratio={ratio:.1f}x, codec={codec_ms:.2f} ms):\n"
    )
    header = (
        f"{'model':<26}{'raw payload':>12}{'raw total':>11}"
        f"{'comp payload':>13}{'comp total':>12}{'speedup':>9}{'regime':>15}"
    )
    print(header)
    print("-" * len(header))
    for name, hidden in KNOWN_HIDDEN_SIZES.items():
        base = prefill_budget(
            hidden_size=hidden,
            seq_len=seq_len,
            link_mbps=link_mbps,
            rtt_ms=rtt_ms,
            compute_ms=compute_ms,
        )
        comp = base.with_compression(ratio, codec_ms=codec_ms)
        speedup = comp.speedup_from(base)
        regime = "BW-bound" if base.is_bandwidth_bound else "Compute-bound"
        print(
            f"{name:<26}"
            f"{base.payload_bytes / 1024 / 1024:>10.1f} MiB"
            f"{base.total_ms:>9.1f} ms"
            f"{comp.payload_bytes / 1024 / 1024:>11.1f} MiB"
            f"{comp.total_ms:>10.1f} ms"
            f"{speedup:>8.2f}x"
            f"{regime:>15}"
        )


def run_profile_pass(
    config: TopologyConfig,
    servers: list[Any],
    seq_lens: list[int],
    repeats: int,
    decode_tokens: int,
    out_path: str | None,
    note: str,
) -> list[HopRecord]:
    """Execute one profiling pass over a topology."""
    metadata = RunMetadata(
        run_id=new_run_id(),
        model_id=config.model_id,
        dtype=config.dtype,
        codec=config.codec,
        num_shards=len(config.nodes),
        note=note,
    )
    try:
        with HopProfiler(path=out_path, metadata=metadata) as profiler:
            with ChainRunner(config, profiler=profiler) as runner:
                for seq_len in seq_lens:
                    ids = torch.randint(0, PROFILE_TOKEN_CEILING, (1, seq_len))
                    for _ in range(repeats):
                        runner.forward(ids, logits_keep_last=1)
                    print(f"  [{config.codec}] seq_len={seq_len} done", file=sys.stderr)

                if decode_tokens > 0:
                    print(
                        f"  [{config.codec}] running {decode_tokens} cached decode steps "
                        f"after seq_len={seq_lens[-1]} context",
                        file=sys.stderr,
                    )
                    decode_prompt = torch.randint(0, PROFILE_TOKEN_CEILING, (1, seq_lens[-1]))
                    runner.generate(decode_prompt, max_new_tokens=decode_tokens + 1)
        return profiler.records
    finally:
        for server in servers:
            server.stop(grace=None)


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
    parser.add_argument(
        "--codec", default="raw", choices=["raw", "lowrank"], help="codec to profile"
    )
    parser.add_argument(
        "--codec-rank", type=int, default=128, help="rank bottleneck for lowrank codec"
    )
    parser.add_argument(
        "--codec-seed", type=int, default=42, help="random seed for lowrank projections"
    )
    parser.add_argument(
        "--compare-codecs",
        action="store_true",
        help="run raw and lowrank side-by-side to measure bandwidth reduction and overhead",
    )
    parser.add_argument("--link-mbps", type=float, default=100.0, help="link to project onto")
    parser.add_argument("--rtt-ms", type=float, default=30.0, help="RTT to project onto")
    parser.add_argument(
        "--project-compute-ms",
        type=float,
        default=None,
        help="shard compute to assume in the projection, instead of the measured value.",
    )
    parser.add_argument(
        "--decode-tokens",
        type=int,
        default=0,
        help="also run cached generation for this many new tokens.",
    )
    args = parser.parse_args(argv)

    seq_lens = [int(s) for s in args.seq_lens.split(",")]

    if args.compare_codecs:
        print(f"\n--- Running Baseline Profile (RawCodec) on {args.model} ---", file=sys.stderr)
        raw_config, raw_servers = build_local_chain(
            args.model, args.shards, args.dtype, args.device, codec_name="raw"
        )
        raw_records = run_profile_pass(
            raw_config,
            raw_servers,
            seq_lens,
            args.repeats,
            args.decode_tokens,
            out_path=None,
            note="baseline raw comparison",
        )

        print(
            f"\n--- Running LowRank Profile (r={args.codec_rank}) on {args.model} ---",
            file=sys.stderr,
        )
        lr_args = {"rank": args.codec_rank, "seed": args.codec_seed}
        lr_config, lr_servers = build_local_chain(
            args.model,
            args.shards,
            args.dtype,
            args.device,
            codec_name="lowrank",
            codec_args=lr_args,
        )
        lr_records = run_profile_pass(
            lr_config,
            lr_servers,
            seq_lens,
            args.repeats,
            args.decode_tokens,
            out_path=args.out,
            note=f"lowrank rank={args.codec_rank} comparison",
        )

        for seq_len in seq_lens:
            raw_sub = [r for r in raw_records if r.seq_len == seq_len and r.phase == "prefill"]
            lr_sub = [r for r in lr_records if r.seq_len == seq_len and r.phase == "prefill"]
            raw_sum = summarize_by_hop(raw_sub)
            lr_sum = summarize_by_hop(lr_sub)
            title = f"raw vs lowrank r={args.codec_rank}, seq_len={seq_len}"
            print(f"\n=== Codec Comparison ({title}) ===")
            print(format_comparison_table(raw_sum, lr_sum, args.link_mbps, args.rtt_ms))

        # Project for paper
        last_hop_records = [r for r in lr_records if r.seq_len == seq_lens[-1] and r.hop > 0]
        measured_compute_ms = (
            sum(r.compute_ns for r in last_hop_records) / len(last_hop_records) / 1e6
            if last_hop_records
            else 0.0
        )
        measured_codec_ms = (
            sum(r.codec_overhead_ns for r in last_hop_records) / len(last_hop_records) / 1e6
            if last_hop_records
            else 0.5
        )
        compute_ms = (
            args.project_compute_ms if args.project_compute_ms is not None else measured_compute_ms
        )

        # Estimate empirical ratio on activation hop
        hidden_size = KNOWN_HIDDEN_SIZES.get(args.model, 1024)
        ratio = hidden_size / args.codec_rank if args.codec_rank else 8.0
        project_comparison(
            seq_lens[-1],
            args.link_mbps,
            args.rtt_ms,
            compute_ms,
            ratio=ratio,
            codec_ms=measured_codec_ms,
        )
        print(f"\nrecords written to {Path(args.out).resolve()}")
        return 0

    # Single codec profiling run
    codec_args = {}
    if args.codec == "lowrank":
        codec_args = {"rank": args.codec_rank, "seed": args.codec_seed}

    if args.config:
        config = TopologyConfig.from_file(args.config)
        servers = []
    else:
        print(
            f"starting {args.shards} shards of {args.model} locally [{args.codec}]", file=sys.stderr
        )
        config, servers = build_local_chain(
            args.model,
            args.shards,
            args.dtype,
            args.device,
            codec_name=args.codec,
            codec_args=codec_args,
        )

    records = run_profile_pass(
        config,
        servers,
        seq_lens,
        args.repeats,
        args.decode_tokens,
        out_path=args.out,
        note=f"seq_lens={seq_lens} repeats={args.repeats} device={args.device}",
    )

    for seq_len in seq_lens:
        subset = [r for r in records if r.seq_len == seq_len and r.phase == "prefill"]
        summaries = summarize_by_hop(subset)
        print(f"\n=== prefill, seq_len = {seq_len} ({config.codec}) ===")
        print(format_table(summaries))
        print(f"\nverdict: {bandwidth_verdict(summaries)}")

    if args.decode_tokens > 0:
        decode_records = [r for r in records if r.phase == "decode"]
        decode_summaries = summarize_by_hop(decode_records)
        print(f"\n=== cached decode, after seq_len = {seq_lens[-1]} context ({config.codec}) ===")
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

    ratio = 1.0
    if config.codec == "lowrank":
        hidden_size = KNOWN_HIDDEN_SIZES.get(config.model_id, 1024)
        ratio = hidden_size / args.codec_rank if args.codec_rank else 8.0
        project_comparison(seq_lens[-1], args.link_mbps, args.rtt_ms, compute_ms, ratio=ratio)
    else:
        project_comparison(seq_lens[-1], args.link_mbps, args.rtt_ms, compute_ms, ratio=1.0)

    print(f"\nrecords written to {Path(args.out).resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
