#!/usr/bin/env python
"""Does compression make answers faster on a slow link?

    python scripts/bench_compression.py --link-mbps 15 --link-ms 100

Starts two real devices on this machine, each holding one half of the model, with the
link to the second one throttled to a chosen bandwidth and latency. Every hidden state
therefore crosses that link, as it would between two laptops on a phone hotspot. It asks
the same long prompt several times with no compression and again with each codec, and
reports the time and the bytes.

The question it answers is narrower than "is compression good": at what link speed does
spending compute on encoding pay for itself in time saved on the wire? On a fast LAN it
should not, and the numbers should say so.

Needs about 5 GB of free RAM. Both devices run on the CPU so the comparison is not
decided by who has the GPU; use --device cuda to change that. Stops everything when done.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import statistics
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PROMPT = (
    "Summarise the following in one sentence. "
    + "The pool lends spare computers to anyone who asks. " * 12
)


def post(url: str, body: dict, timeout: float = 300.0) -> dict:
    request = urllib.request.Request(
        url,
        data=json.dumps(body).encode(),
        headers={"content-type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
        return json.loads(response.read())


def get(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=5) as response:  # noqa: S310
        return json.loads(response.read())


def start_pool(pool_file: Path, args, ports: tuple[int, int]) -> list[subprocess.Popen]:
    env = {**os.environ, "HF_HUB_OFFLINE": "1", "OMP_NUM_THREADS": str(args.threads)}
    procs = []
    for name, port, link in (("near", ports[0], False), ("far", ports[1], True)):
        cmd = [
            sys.executable,
            "-m",
            "torrent_llm.pool.peer",
            "--pool",
            str(pool_file),
            "--name",
            name,
            "--device",
            args.device,
            "--memory-mb",
            "1600",
            "--port",
            str(port),
            "--shard-port-base",
            str(port + 200),
            "--host",
            "127.0.0.1",
            "--no-discover",
            "--no-watch-owner",
        ]
        if link:
            cmd += [
                "--link-latency-ms",
                str(args.link_ms),
                "--link-bandwidth-mbps",
                str(args.link_mbps),
            ]
        else:
            cmd += ["--join", f"127.0.0.1:{ports[1]}"]
        procs.append(
            subprocess.Popen(  # noqa: S603
                cmd,
                cwd=REPO,
                env=env,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
        )
    return procs


def stop_pool(procs: list[subprocess.Popen]) -> None:
    for proc in procs:
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    for proc in procs:
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)


def wait_ready(base: str, timeout: float = 150.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            view = get(f"{base}/pool")
            if min(view["ready_copies"]) >= 1 and len(view["members"]) == 2:
                return
        except Exception:  # noqa: BLE001 - still starting
            pass
        time.sleep(1.0)
    raise SystemExit("the devices did not come up in time")


def measure(pool_file: Path, label: str, args, ports: tuple[int, int]) -> dict:
    procs = start_pool(pool_file, args, ports)
    base = f"http://127.0.0.1:{ports[0]}"
    try:
        wait_ready(base)
        post(f"{base}/generate", {"prompt": "warm up", "max_new_tokens": 2})  # first call is slow
        rows = []
        for _ in range(args.repeats):
            started = time.perf_counter()
            body = post(
                f"{base}/generate", {"prompt": PROMPT, "max_new_tokens": args.tokens, "chat": False}
            )
            wall = (time.perf_counter() - started) * 1000
            t = body["totals"]
            rows.append(
                {
                    "wall_ms": wall,
                    "transport_ms": t["transport_ms"],
                    "compute_ms": t["compute_ms"],
                    "sent_bytes": t["sent_bytes"],
                    "uncompressed_bytes": t["uncompressed_bytes"],
                    "tokens": t["tokens"],
                }
            )
        med = {k: statistics.median(r[k] for r in rows) for k in rows[0]}
        med["label"] = label
        return med
    finally:
        stop_pool(procs)
        time.sleep(2)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--link-mbps", type=float, default=15.0)
    parser.add_argument("--link-ms", type=float, default=100.0)
    parser.add_argument("--repeats", type=int, default=4)
    parser.add_argument("--tokens", type=int, default=8)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument(
        "--pools", nargs="+", default=["configs/pool.yaml", "configs/pool-compressed.yaml"]
    )
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    free = 0.0
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable"):
            free = int(line.split()[1]) / 1024**2
    if free < 6:
        print(f"only {free:.1f} GB of RAM free; need about 6. Close something and retry.")
        return 2

    results = []
    for i, pool in enumerate(args.pools):
        label = Path(pool).stem
        print(
            f"measuring {label} over {args.link_mbps:g} Mbps, {args.link_ms:g} ms ...", flush=True
        )
        results.append(measure(REPO / pool, label, args, (9400 + 20 * i, 9410 + 20 * i)))

    print(
        f"\n{'pool':22} {'time':>9} {'network':>9} {'compute':>9} {'wire bytes':>12} {'smaller':>8}"
    )
    for r in results:
        ratio = r["uncompressed_bytes"] / max(1, r["sent_bytes"])
        print(
            f"{r['label']:22} {r['wall_ms']:8.0f}ms {r['transport_ms']:8.0f}ms "
            f"{r['compute_ms']:8.0f}ms {r['sent_bytes'] / 1024:10.0f}Ki {ratio:7.1f}x"
        )
    base = results[0]
    for r in results[1:]:
        print(
            f"\n{r['label']} vs {base['label']}: {r['wall_ms'] / base['wall_ms']:.2f}x the time "
            f"({base['wall_ms'] - r['wall_ms']:+.0f} ms)"
        )
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(
            json.dumps(
                {
                    "link_mbps": args.link_mbps,
                    "link_ms": args.link_ms,
                    "device": args.device,
                    "results": results,
                },
                indent=2,
            )
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
