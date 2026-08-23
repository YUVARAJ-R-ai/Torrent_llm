"""Command-line entry points.

``torrent-shard`` starts one node; ``torrent-run`` drives a request through all
of them. Both read the same topology file, so moving from one machine to two is
an edit to the addresses and nothing else.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys

import torch

from torrent_llm.codec import get_codec
from torrent_llm.runner import ChainRunner, TopologyConfig
from torrent_llm.shard import load_shard
from torrent_llm.transport import serve


def _configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    )


def shard_main(argv: list[str] | None = None) -> int:
    """Serve one shard of the topology and block."""
    parser = argparse.ArgumentParser(description="Serve one Torrent-LLM shard.")
    parser.add_argument("--config", required=True, help="topology YAML")
    parser.add_argument("--index", type=int, required=True, help="which node in the topology")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)
    _configure_logging(args.verbose)

    config = TopologyConfig.from_file(args.config)
    if not 0 <= args.index < len(config.nodes):
        parser.error(f"--index must be in [0, {len(config.nodes)}), got {args.index}")

    node = config.nodes[args.index]
    spec = config.shard_plan()[args.index]

    runtime = load_shard(
        config.model_id,
        spec,
        device=node.device,
        dtype=getattr(torch, config.dtype),
        trust_remote_code=config.trust_remote_code,
    )
    server, port = serve(
        runtime,
        get_codec(config.codec, **config.codec_args),
        host=node.host,
        port=node.port,
        model_id=config.model_id,
    )
    print(f"{spec} serving on {node.host}:{port} ({runtime.dtype} on {node.device})")
    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        server.stop(grace=1.0)
    return 0


def run_main(argv: list[str] | None = None) -> int:
    """Drive a prompt through the chain."""
    parser = argparse.ArgumentParser(description="Run a request across a Torrent-LLM chain.")
    parser.add_argument("--config", required=True, help="topology YAML")
    parser.add_argument("--prompt", default="The capital of France is")
    parser.add_argument("--max-new-tokens", type=int, default=0, help="0 = prefill only")
    parser.add_argument("--describe", action="store_true", help="print the chain and exit")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)
    _configure_logging(args.verbose)

    config = TopologyConfig.from_file(args.config)
    with ChainRunner(config) as runner:
        if args.describe:
            print(json.dumps(runner.describe(), indent=2))
            return 0

        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            config.model_id, trust_remote_code=config.trust_remote_code
        )
        input_ids = tokenizer(args.prompt, return_tensors="pt").input_ids

        if args.max_new_tokens > 0:
            ids, passes = runner.generate(input_ids, max_new_tokens=args.max_new_tokens)
            print(tokenizer.decode(ids[0], skip_special_tokens=True))
            result = passes[-1]
        else:
            result = runner.forward(input_ids)
            top = result.logits[0, -1].argmax().item()
            print(f"{args.prompt}{tokenizer.decode([top])}")

        print(
            f"\n{len(result.hops)} hops | "
            f"{result.total_sent_bytes / 1024:.1f} KiB on the wire | "
            f"{result.total_wall_ns / 1e6:.1f} ms total "
            f"({result.total_compute_ns / 1e6:.1f} ms compute, "
            f"{result.total_transport_ns / 1e6:.1f} ms transport)"
        )
    return 0


if __name__ == "__main__":
    sys.exit(run_main())
