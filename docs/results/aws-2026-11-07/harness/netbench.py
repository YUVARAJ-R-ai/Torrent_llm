"""Client for the real-network bench. Runs on a pool device against a pool device's HTTP API.

    python netbench.py wait    http://127.0.0.1:9000 --members 2
    python netbench.py speed   http://127.0.0.1:9000 --label pool --repeats 5 --tokens 16
    python netbench.py quality http://127.0.0.1:9000 --label pool

Prints one JSON object on stdout. Standard library only, except `quality`, which reads the
95 held-out questions from the repo's scripts/calibrate_codec.py so the numbers compare
with docs/compression.md.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
import urllib.request
from pathlib import Path

# Same long prompt as scripts/bench_compression.py, so prefill bytes dominate like there.
PROMPT = (
    "Summarise the following in one sentence. "
    + "The pool lends spare computers to anyone who asks. " * 12
)


def post(url: str, body: dict, timeout: float = 900.0) -> dict:
    request = urllib.request.Request(
        url, data=json.dumps(body).encode(), headers={"content-type": "application/json"}
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
        return json.loads(response.read())


def get(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=5) as response:  # noqa: S310
        return json.loads(response.read())


def wait(base: str, members: int, copies: int, timeout: float) -> dict:
    deadline = time.monotonic() + timeout
    view: dict = {}
    while time.monotonic() < deadline:
        try:
            view = get(f"{base}/pool")
            if len(view["members"]) >= members and min(view["ready_copies"]) >= copies:
                return {"ok": True, "ready_copies": view["ready_copies"], "members": len(view["members"])}
        except Exception:  # noqa: BLE001 - still starting
            pass
        time.sleep(1.0)
    return {"ok": False, "last_view": view}


def speed(base: str, label: str, repeats: int, tokens: int) -> dict:
    post(f"{base}/generate", {"prompt": "warm up", "max_new_tokens": 2})
    rows, texts = [], set()
    for _ in range(repeats):
        started = time.perf_counter()
        body = post(f"{base}/generate", {"prompt": PROMPT, "max_new_tokens": tokens, "chat": False})
        wall = (time.perf_counter() - started) * 1000
        texts.add(body["completion"])
        t = body["totals"]
        rows.append({"wall_ms": wall, **{k: t[k] for k in (
            "transport_ms", "compute_ms", "sent_bytes", "uncompressed_bytes", "tokens", "failovers")}})
    med = {k: statistics.median(r[k] for r in rows) for k in rows[0]}
    med["wall_ms_min"] = min(r["wall_ms"] for r in rows)
    med["wall_ms_max"] = max(r["wall_ms"] for r in rows)
    return {"label": label, "repeats": repeats, "distinct_outputs": len(texts), **med}


def quality(base: str, label: str, repo: Path, max_new: int) -> dict:
    sys.path.insert(0, str(repo / "scripts"))
    from calibrate_codec import EVAL_QUESTIONS  # noqa: PLC0415 - heavy import, only here

    right, wrong = 0, []
    started = time.perf_counter()
    for question, expected in EVAL_QUESTIONS:
        answer = post(f"{base}/generate", {"prompt": question, "max_new_tokens": max_new})["completion"]
        if expected.lower() in answer.lower():
            right += 1
        else:
            wrong.append([question, expected, answer])
    return {
        "label": label,
        "questions": len(EVAL_QUESTIONS),
        "right": right,
        "accuracy": right / len(EVAL_QUESTIONS),
        "seconds": time.perf_counter() - started,
        "wrong": wrong,
    }


def long(base: str, label: str, tokens: int) -> dict:
    """One long answer; reports the route each pass took and any failover events."""
    started = time.perf_counter()
    body = post(f"{base}/generate", {
        "prompt": "Count from one to two hundred in words, separated by commas.",
        "max_new_tokens": tokens})
    wall = time.perf_counter() - started
    routes = [tuple(h.get("member") for h in p["hops"]) for p in body["passes"]]
    changes = [i for i in range(1, len(routes)) if routes[i] != routes[i - 1]]
    return {
        "label": label,
        "wall_s": wall,
        "tokens": body["totals"]["tokens"],
        "failovers": body["totals"]["failovers"],
        "events": body["events"],
        "first_route": routes[0] if routes else None,
        "last_route": routes[-1] if routes else None,
        "route_changes_at_pass": changes,
        "completion": body["completion"],
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["wait", "speed", "quality", "long"])
    p.add_argument("base")
    p.add_argument("--label", default="")
    p.add_argument("--members", type=int, default=2)
    p.add_argument("--copies", type=int, default=1)
    p.add_argument("--timeout", type=float, default=600)
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--tokens", type=int, default=16)
    p.add_argument("--repo", default=str(Path.home() / "Torrent_llm"))
    a = p.parse_args()
    if a.cmd == "wait":
        out = wait(a.base, a.members, a.copies, a.timeout)
    elif a.cmd == "speed":
        out = speed(a.base, a.label, a.repeats, a.tokens)
    elif a.cmd == "long":
        out = long(a.base, a.label, a.tokens)
    else:
        out = quality(a.base, a.label, Path(a.repo), 14)
    print(json.dumps(out))
    return 0 if out.get("ok", True) else 1


if __name__ == "__main__":
    raise SystemExit(main())
