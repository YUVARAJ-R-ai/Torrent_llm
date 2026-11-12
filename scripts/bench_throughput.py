"""Throughput: requester routing vs peer-to-peer forwarding, one and many requests (#36).

Builds one pool on this machine: an asking device plus one overlay node per stage, all
on loopback, with every message between two devices paying a simulated link (latency
and bandwidth, both directions, the same rule for both modes). Then runs the same
prompts through both routing modes at each concurrency and records, per request,
tokens/s, per-token latency and bytes on the wire.

    python scripts/bench_throughput.py --model Qwen/Qwen2.5-0.5B --stages 3 \\
        --latency-ms 40 --mbps 20 --concurrency 1,2,4,8

``--model tiny`` uses a random small Llama, for a quick offline check. Results go to
``runs/throughput-<stamp>.jsonl`` (one line per request) and a table is printed.
Greedy decoding: both modes must produce the same tokens, which is checked.
"""

from __future__ import annotations

import argparse
import copy
import dataclasses
import json
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, LlamaConfig

from torrent_llm.link import LinkProfile
from torrent_llm.net.identity import Identity
from torrent_llm.net.node import Node, NodeConfig
from torrent_llm.pool.config import PoolConfig
from torrent_llm.pool.forwarding import Forwarder
from torrent_llm.pool.hosting import OVERLAY, ShardHost
from torrent_llm.pool.members import ACTIVE, READY, HostedStage, Member, MemberTable
from torrent_llm.pool.overlay import serve_shards
from torrent_llm.pool.runner import PoolRunner
from torrent_llm.shard import ShardRuntime


def shape_links(node: Node, link: LinkProfile) -> None:
    """Make every call this node sends to another device pay ``link`` both ways."""
    real = node.call

    def call(peer_id, method, header=None, body=b"", *, timeout):
        if peer_id == node.peer_id:
            return real(peer_id, method, header, body, timeout=timeout)
        time.sleep(link.one_way_s(len(body)))
        reply_header, reply_body = real(peer_id, method, header, body, timeout=timeout)
        time.sleep(link.one_way_s(len(reply_body)))
        return reply_header, reply_body

    node.call = call  # type: ignore[method-assign]


def wire_bytes(hops, routing: str) -> int:
    """Bytes that crossed a link in one pass. Requester routing: every hop out and back.
    Relay: into every stage, then the token home (a stage's output is the next one's input).
    """
    if routing == "requester":
        return sum(h.sent_bytes + h.received_bytes for h in hops)
    return sum(h.sent_bytes for h in hops) + (hops[-1].received_bytes if hops else 0)


def one_request(runner: PoolRunner, prompt: torch.Tensor, new_tokens: int):
    """Run one request; return its new tokens, the pool's result and its timings."""
    stamps: list[float] = []
    t0 = time.perf_counter()
    res = runner.generate(
        prompt, max_new_tokens=new_tokens, on_token=lambda _t: stamps.append(time.perf_counter())
    )
    wall = time.perf_counter() - t0
    new = res.ids[0, prompt.shape[1] :].tolist()
    row = {
        "new_tokens": len(new),
        "wall_s": wall,
        "tokens_per_s": len(new) / wall,
        "first_token_s": (stamps[0] - t0) if stamps else None,
        "token_gap_ms": [(b - a) * 1000 for a, b in zip(stamps, stamps[1:], strict=False)],
        "failovers": res.failovers,
    }
    return new, res, row


def build_model(name: str):
    if name == "tiny":
        torch.manual_seed(0)
        cfg = LlamaConfig(
            hidden_size=512,
            intermediate_size=1024,
            num_hidden_layers=6,
            num_attention_heads=8,
            num_key_value_heads=4,
            vocab_size=4096,
            max_position_embeddings=2048,
        )
        return AutoModelForCausalLM.from_config(cfg).eval()
    return AutoModelForCausalLM.from_pretrained(name, dtype=torch.float32).eval()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--model", default="tiny")
    ap.add_argument("--stages", type=int, default=3)
    ap.add_argument("--latency-ms", type=float, default=40.0, help="round trip per link")
    ap.add_argument("--mbps", type=float, default=20.0, help="bandwidth per link")
    ap.add_argument("--concurrency", default="1,2,4,8")
    ap.add_argument("--prompt-lens", default="16,128", help="short and long prompts")
    ap.add_argument("--new-tokens", type=int, default=16)
    ap.add_argument("--codec", default="raw")
    ap.add_argument("--out", type=Path, default=Path("runs"))
    args = ap.parse_args(argv)

    base = build_model(args.model)
    num_layers = base.config.num_hidden_layers
    vocab = base.config.vocab_size
    config = PoolConfig(
        model_id=args.model, num_layers=num_layers, stages=args.stages, codec=args.codec
    )
    link = LinkProfile(latency_ms=args.latency_ms, bandwidth_mbps=args.mbps)

    def node() -> Node:
        n = Node(NodeConfig(identity=Identity.generate(), listen_host="127.0.0.1"))
        n.start()
        return n

    asker = node()
    devices = [node() for _ in range(args.stages)]
    everyone = [asker, *devices]
    for a in everyone:
        for b in everyone:
            if a is not b:
                a.add_record(b.record())
    hosts, forwarders = [], []
    print(f"loading {args.stages} stages of {args.model} ...", file=sys.stderr)
    for stage, dev in enumerate(devices):
        host = ShardHost(
            config.specs(),
            lambda spec: ShardRuntime(copy.deepcopy(base), spec),
            model_id=args.model,
            transport=OVERLAY,
            codecs=config.codec_set(),
        )
        host.start(stage)
        while not all(h.status in (READY, "failed") for h in host.hosting()):
            time.sleep(0.1)
        if any(h.status != READY for h in host.hosting()):
            raise SystemExit(f"stage {stage} failed to load: {host.error(stage)}")
        serve_shards(dev, host)
        forwarders.append(Forwarder(dev, host, workers=16))
        hosts.append(host)
    for n in everyone:
        shape_links(n, link)

    def table() -> MemberTable:
        def member(n: Node, stages: tuple[int, ...]) -> Member:
            return Member(
                name=n.peer_id,
                host="h",
                control_port=1,
                slots=args.stages,
                state=ACTIVE,
                hosting=tuple(HostedStage(stage=s, port=0, status=READY) for s in stages),
                heartbeat=1,
            )

        t = MemberTable(asker.peer_id, suspect_after=1e9, down_after=1e9)  # nobody heartbeats here
        t.set_self(member(asker, ()))
        t.merge([member(d, (i,)) for i, d in enumerate(devices)])
        return t

    stamp = time.strftime("%Y%m%d-%H%M%S")
    args.out.mkdir(parents=True, exist_ok=True)
    out_path = args.out / f"throughput-{stamp}.jsonl"
    rows, answers = [], {}
    gen = torch.Generator().manual_seed(1)
    prompts = {
        int(n): torch.randint(0, vocab, (1, int(n)), generator=gen)
        for n in args.prompt_lens.split(",")
    }
    with out_path.open("w") as out:
        for routing in ("requester", "relay"):
            runner = PoolRunner(
                dataclasses.replace(config, routing=routing),
                table(),
                me=asker.peer_id,
                node=asker,
                max_busy_retries=1000,
            )
            for plen, prompt in prompts.items():
                for conc in (int(c) for c in args.concurrency.split(",")):
                    t0 = time.perf_counter()
                    with ThreadPoolExecutor(conc) as pool:
                        futures = [
                            pool.submit(one_request, runner, prompt, args.new_tokens)
                            for _ in range(conc)
                        ]
                        done = [f.result() for f in futures]
                    span = time.perf_counter() - t0
                    total = sum(len(new) for new, _, _ in done)
                    for i, (new, res, row) in enumerate(done):
                        if answers.setdefault(plen, new) != new:
                            raise SystemExit(f"{routing} gave different tokens for prompt {plen}")
                        row.update(
                            routing=routing,
                            prompt_len=plen,
                            concurrency=conc,
                            request=i,
                            wire_bytes=sum(wire_bytes(p.hops, routing) for p in res.passes),
                            aggregate_tokens_per_s=total / span,
                            link=link.describe(),
                            model=args.model,
                            stages=args.stages,
                            codec=args.codec,
                        )
                        out.write(json.dumps(row) + "\n")
                        rows.append(row)
                    print(f"  {routing:9s} prompt={plen:4d} conc={conc}: done", file=sys.stderr)

    print(f"\n{args.model}, {args.stages} stages, link {link.describe()}, codec {args.codec}")
    print(
        f"{'routing':10s}{'prompt':>7s}{'conc':>5s}{'tok/s/req':>11s}{'agg tok/s':>11s}"
        f"{'gap p50':>9s}{'gap p95':>9s}{'KiB/req':>9s}"
    )
    keys = sorted({(r["routing"], r["prompt_len"], r["concurrency"]) for r in rows})
    for routing, plen, conc in keys:
        sel = [
            r
            for r in rows
            if (r["routing"], r["prompt_len"], r["concurrency"]) == (routing, plen, conc)
        ]
        gaps = sorted(g for r in sel for g in r["token_gap_ms"]) or [0.0]
        p95 = gaps[min(len(gaps) - 1, int(0.95 * len(gaps)))]
        print(
            f"{routing:10s}{plen:7d}{conc:5d}"
            f"{statistics.mean(r['tokens_per_s'] for r in sel):11.2f}"
            f"{sel[0]['aggregate_tokens_per_s']:11.2f}"
            f"{statistics.median(gaps):9.1f}{p95:9.1f}"
            f"{statistics.mean(r['wire_bytes'] for r in sel) / 1024:9.1f}"
        )
    print(f"\nwrote {out_path}")
    for f in forwarders:
        f.close()
    for h in hosts:
        h.close()
    for n in everyone:
        n.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
