#!/usr/bin/env python
"""Is the pool working? Ask every device the same question and check they all agree.

    python scripts/pool_check.py http://192.168.1.20:9000

Run it once everyone has joined. It uses only the standard library, so it runs anywhere
Python does, including a laptop that is not part of the pool.

For every device that is up it asks the same question and checks:

* the answer is the same everywhere (the model is deterministic, so a difference means
  two devices are not running the same thing),
* the answer is right,
* a question asked through that device really was answered, and by which route,
* how long it took, and how much the hidden-state traffic was compressed.

It then asks a few questions at once, to see the pool share the work. Exits with a
non-zero status if anything is wrong, so it can be used in a script.
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
import urllib.error
import urllib.request

QUESTION = "What is the capital of France?"
EXPECT = "paris"
#: Ask as a question (the model's chat template). Off only for a model without one.
CHAT = True


def get(url: str, timeout: float = 5.0) -> dict:
    with urllib.request.urlopen(url, timeout=timeout) as response:  # noqa: S310
        return json.loads(response.read())


def ask(base: str, question: str, max_tokens: int = 24, timeout: float = 120.0) -> dict:
    request = urllib.request.Request(
        f"{base}/generate",
        data=json.dumps({"prompt": question, "max_new_tokens": max_tokens, "chat": CHAT}).encode(),
        headers={"content-type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
        return json.loads(response.read())


def device_url(member: dict) -> str:
    return f"http://{member['host']}:{member['control_port']}"


def check(seed: str, *, out=print) -> bool:
    """Run every check against the pool ``seed`` belongs to. Returns whether all passed."""
    ok = True

    def fail(message: str) -> None:
        nonlocal ok
        ok = False
        out(f"  FAIL  {message}")

    try:
        view = get(f"{seed}/pool")
    except (urllib.error.URLError, OSError, TimeoutError) as exc:
        out(f"Could not reach {seed}: {getattr(exc, 'reason', exc)}")
        out("Is torrent-peer running there? Is the address and port right?")
        return False

    pool = view["pool"]
    members = [m for m in view["members"] if m["liveness"] == "alive"]
    out(f"Pool: {pool['model_id']}, {len(members)} device(s) online, health {view['health']}")
    out(f"Stage copies ready: {view['ready_copies']} (each stage wants {pool['replicas']})")
    out(f"Compression: {pool.get('codec', 'raw')}")
    for m in view["members"]:
        held = ", ".join(f"stage {h['stage'] + 1}" for h in m["hosting"]) or "nothing"
        out(f"  {m['name']:16} {m['liveness']:8} {m['state']:9} {m['device']:6} holds {held}")
    if view["health"] == "down":
        fail("a stage has no copy, so no question can be answered")
        return False
    if len(members) < 2:
        out("  note: only one device, so there is nothing to compare yet")

    out(f"\nAsking each device: {QUESTION!r}")
    answers: dict[str, str] = {}
    for m in members:
        started = time.perf_counter()
        try:
            body = ask(device_url(m), QUESTION)
        except urllib.error.HTTPError as exc:
            fail(f"{m['name']}: {exc.code} {exc.read().decode()[:200]}")
            continue
        except (urllib.error.URLError, OSError, TimeoutError) as exc:
            fail(f"{m['name']}: no answer ({getattr(exc, 'reason', exc)})")
            continue
        took = time.perf_counter() - started
        route = " -> ".join(member for _, member in body["passes"][0]["route"])
        ratio = ""
        hidden = [h for p in body["passes"] for h in p["hops"] if h["stage"] > 0]
        if hidden and sum(h["sent_bytes"] for h in hidden):
            r = sum(h["uncompressed_bytes"] for h in hidden) / sum(h["sent_bytes"] for h in hidden)
            ratio = f", traffic {r:.1f}x smaller"
        answers[m["name"]] = body["completion"].strip()
        out(f"  {m['name']:16} {took:5.1f}s  via {route}{ratio}")
        out(f"  {'':16} {body['completion'].strip()!r}")
        if EXPECT not in body["completion"].lower():
            fail(f"{m['name']}: the answer does not mention Paris")

    if len(set(answers.values())) > 1:
        fail("devices gave different answers to the same question: " + str(answers))
    elif answers:
        out("  OK    every device gave the same answer")

    out("\nAsking five questions at once through different devices...")
    results: list[float] = []
    errors: list[str] = []
    lock = threading.Lock()

    def worker(i: int) -> None:
        target = device_url(members[i % len(members)])
        started = time.perf_counter()
        try:
            ask(target, QUESTION, max_tokens=12)
            with lock:
                results.append(time.perf_counter() - started)
        except (urllib.error.URLError, OSError, TimeoutError) as exc:
            with lock:
                errors.append(str(getattr(exc, "reason", exc)))

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(5)]
    wall = time.perf_counter()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall = time.perf_counter() - wall
    if errors:
        fail(f"{len(errors)} of 5 simultaneous questions failed: {errors[0]}")
    else:
        out(f"  OK    all five answered in {wall:.1f}s (slowest {max(results):.1f}s)")

    out("\n" + ("All checks passed." if ok else "Some checks failed; see FAIL lines above."))
    return ok


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("url", help="address of any device, e.g. http://192.168.1.20:9000")
    args = parser.parse_args()
    url = args.url if args.url.startswith("http") else f"http://{args.url}"
    return 0 if check(url.rstrip("/")) else 1


if __name__ == "__main__":
    sys.exit(main())
