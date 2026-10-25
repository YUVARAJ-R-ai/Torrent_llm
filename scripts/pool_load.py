#!/usr/bin/env python
"""Ask a running pool many questions at once and report how the work was shared.

    python scripts/pool_load.py --requests 24 --concurrency 6

Each request goes to a different device in turn, the way separate people would
reach the pool through their own laptops. Prints throughput, answer latency, and
how many passes each device ran, so you can see whether the pool really spreads
work or piles it on one device.
"""

from __future__ import annotations

import argparse
import statistics
import threading
import time
from collections import Counter

import httpx

QUESTIONS = [
    "What is the capital of France?",
    "Who wrote Hamlet?",
    "What is 12 times 7?",
    "What is the largest planet?",
    "What colour is the sky?",
    "How many days are in a week?",
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", default="http://127.0.0.1:9102", help="any device in the pool")
    parser.add_argument("--requests", type=int, default=24)
    parser.add_argument("--concurrency", type=int, default=6)
    args = parser.parse_args()

    view = httpx.get(f"{args.seed}/pool", timeout=5).json()
    gateways = [
        f"http://{m['host']}:{m['control_port']}"
        for m in view["members"]
        if m["liveness"] == "alive"
    ]
    print(f"{len(gateways)} devices online; stage copies {view['ready_copies']}")

    lock = threading.Lock()
    latencies: list[float] = []
    work: Counter[str] = Counter()
    tokens = 0
    failures = 0
    slots = threading.Semaphore(args.concurrency)

    def ask(i: int) -> None:
        nonlocal tokens, failures
        with slots:
            gateway = gateways[i % len(gateways)]
            started = time.perf_counter()
            try:
                response = httpx.post(
                    f"{gateway}/generate",
                    json={"prompt": QUESTIONS[i % len(QUESTIONS)], "max_new_tokens": 24},
                    timeout=120,
                )
                response.raise_for_status()
                body = response.json()
            except httpx.HTTPError as exc:
                with lock:
                    failures += 1
                print(f"  request {i} failed: {exc}")
                return
            took = time.perf_counter() - started
            with lock:
                latencies.append(took)
                tokens += body["totals"]["tokens"]
                for p in body["passes"]:
                    for stage, member in p["route"]:
                        work[f"stage {stage + 1} on {member}"] += 1

    started = time.perf_counter()
    threads = [threading.Thread(target=ask, args=(i,)) for i in range(args.requests)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall = time.perf_counter() - started

    done = len(latencies)
    print(f"\n{done}/{args.requests} answered in {wall:.1f}s  ({failures} failed)")
    if not done:
        return 1
    print(f"  {done / wall:.2f} questions/s, {tokens / wall:.1f} tokens/s")
    print(
        f"  answer time: median {statistics.median(latencies):.2f}s, slowest {max(latencies):.2f}s"
    )
    print("\nforward passes by device and stage:")
    for key, count in sorted(work.items()):
        print(f"  {key:28} {count:4}  {'#' * (count * 40 // max(work.values()))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
