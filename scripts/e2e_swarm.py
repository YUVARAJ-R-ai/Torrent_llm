"""End-to-end: real ``torrent-peer`` processes form a swarm on this machine and answer.

Each device is its own process with its own home (identity, swarm secret, trust) and
its own ports, joined only through the invite link, exactly as friends would do:

1. ``create``: device A makes an invite swarm and prints the link.
2. ``join``: device B joins with that link. The pool spreads the stages over A and B.
3. ``join`` without the model: device C has an empty, offline model cache, so it must
   fetch the weights from A and B (``model_share.py``) before it can take a stage.
4. Every device is asked the same question; every answer must be the same as the whole
   model's answer on one machine (greedy decoding).
5. With ``--relay-only`` (needs the relay): device D does not accept direct connections
   (no listening port advertised); it must be reached through A acting as a relay.

    python scripts/e2e_swarm.py --model Qwen/Qwen3-0.6B

The invite link is a secret: it is passed between processes, never printed.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LINK = re.compile(r"tllm://join/\S+")
QUESTION = "What is the capital of France?"


class Device:
    """One ``torrent-peer`` process and what it printed."""

    def __init__(self, name: str, home: Path, args: list[str], env: dict[str, str]) -> None:
        self.name = name
        self.lines: list[str] = []
        self.log = (home / "device.log").open("w", encoding="utf-8")
        self.proc = subprocess.Popen(
            [sys.executable, "-m", "torrent_llm.pool.peer", *args],
            cwd=ROOT,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        threading.Thread(target=self._read, daemon=True).start()

    def _read(self) -> None:
        assert self.proc.stdout is not None
        for line in self.proc.stdout:
            self.lines.append(line)
            # The log keeps the run inspectable without ever holding the secret.
            self.log.write(LINK.sub("tllm://join/<secret>", line))
            self.log.flush()

    def wait_for(self, pattern: re.Pattern[str], timeout: float) -> str:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            for line in list(self.lines):
                found = pattern.search(line)
                if found:
                    return found.group(0)
            if self.proc.poll() is not None:
                raise RuntimeError(f"{self.name} exited ({self.proc.returncode}); see its log")
            time.sleep(0.2)
        raise TimeoutError(f"{self.name}: no {pattern.pattern!r} within {timeout}s")

    def stop(self) -> None:
        if self.proc.poll() is None:
            self.proc.terminate()  # SIGTERM: the device leaves the swarm gracefully
            try:
                self.proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait(timeout=5)
        self.log.close()


def ask(port: int, *, max_new_tokens: int, timeout: float) -> dict:
    """POST /generate, retrying while the pool is still placing stages (503)."""
    body = json.dumps({"prompt": QUESTION, "max_new_tokens": max_new_tokens}).encode()
    deadline = time.monotonic() + timeout
    last = ""
    while time.monotonic() < deadline:
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}/generate",
            data=body,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                return json.loads(resp.read())
        except (urllib.error.URLError, ConnectionError, TimeoutError) as exc:
            last = str(exc)
            time.sleep(2)
    raise TimeoutError(f"device on :{port} never answered: {last}")


def wait_for_relayed(port: int, timeout: float) -> bool:
    """Whether the device on ``port`` gets a connection that came through a relay."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/peers", timeout=10) as resp:
                if any(p.get("via_relay") for p in json.loads(resp.read())):
                    return True
        except (urllib.error.URLError, ConnectionError, TimeoutError, ValueError):
            pass
        time.sleep(2)
    return False


def reference(model_id: str, max_new_tokens: int) -> str:
    """The whole model's greedy answer on this machine, rendered the way devices do."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from torrent_llm.chat import DEFAULT_SYSTEM_PROMPT, render_chat, stop_token_ids

    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float32).eval()
    ids = torch.tensor([tok.encode(render_chat(tok, QUESTION, DEFAULT_SYSTEM_PROMPT))])
    stops = stop_token_ids(model_id, tok)
    out: list[int] = []
    with torch.no_grad():
        for _ in range(max_new_tokens):
            token = int(model(ids).logits[0, -1].argmax())
            if token in stops:
                break
            out.append(token)
            ids = torch.cat([ids, torch.tensor([[token]])], dim=1)
    return tok.decode(out)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--pool", default="configs/pool.yaml")
    ap.add_argument("--max-new-tokens", type=int, default=12)
    ap.add_argument("--base-port", type=int, default=47300)
    ap.add_argument("--skip-fetch", action="store_true", help="skip device C (model fetch)")
    ap.add_argument("--relay-only", action="store_true", help="add device D behind a relay")
    args = ap.parse_args(argv)

    work = Path(tempfile.mkdtemp(prefix="tllm-e2e-"))
    devices: list[Device] = []
    results: dict[str, bool] = {}

    def start(name: str, mode: list[str], n: int, extra_env: dict[str, str] | None = None):
        home = work / name
        home.mkdir()
        env = {**os.environ, "TORRENT_LLM_HOME": str(home), "PYTHONUNBUFFERED": "1"}
        env.update(extra_env or {})
        cli = [
            *mode,
            "--pool", args.pool,
            "--name", name,
            "--listen-port", str(args.base_port + n),
            "--port", str(args.base_port + 100 + n),
            "--no-upnp",
        ]  # fmt: skip
        dev = Device(name, home, cli, env)
        devices.append(dev)
        return dev

    try:
        print(f"work dir: {work}")
        want = reference(args.model, args.max_new_tokens)
        print(f"whole model says: {want!r}")

        # On one machine autonat cannot prove A is public, so A is told to relay.
        relay_flag = ["--relay"] if args.relay_only else []
        a = start("a", ["create", "--swarm-name", "e2e", *relay_flag], 0)
        link = a.wait_for(LINK, timeout=180)
        a.wait_for(re.compile(r"Control panel"), timeout=300)
        b = start("b", ["join", link], 1)
        b.wait_for(re.compile(r"Control panel"), timeout=300)
        joined = [("a", 0), ("b", 1)]

        if not args.skip_fetch:
            empty = work / "empty-hf"
            empty.mkdir()
            c = start("c", ["join", link], 2, {"HF_HOME": str(empty), "HF_HUB_OFFLINE": "1"})
            c.wait_for(re.compile(r"Model fetched and checked"), timeout=900)
            c.wait_for(re.compile(r"Control panel"), timeout=300)
            joined.append(("c", 2))
            results["c fetched the model from the swarm"] = True

        if args.relay_only:
            d = start("d", ["join", link, "--no-inbound"], 3)
            d.wait_for(re.compile(r"Control panel"), timeout=300)
            # Only relay addresses (or none yet): nothing direct is published.
            d.wait_for(
                re.compile(r"Reachable at: (through relays only|relay://\S+?(, relay://\S+?)*)$"),
                timeout=60,
            )
            joined.append(("d", 3))
            results["d publishes only relay addresses"] = True
            # Informational: D also dials out (as a device behind NAT can), so others
            # often reach it over its own connections and never need a circuit.
            relayed = wait_for_relayed(args.base_port + 103, timeout=60)
            print(f"d has a connection through the relay: {relayed}")

        for name, n in joined:
            reply = ask(args.base_port + 100 + n, max_new_tokens=args.max_new_tokens, timeout=600)
            got = reply["completion"]
            route = {m for p in reply["passes"] for _, m in p["route"]}
            ok = got == want
            results[f"{name} answered like the whole model"] = ok
            print(f"{name}: {got!r} via {len(route)} device(s) {'OK' if ok else 'MISMATCH'}")
    finally:
        for dev in devices:
            dev.stop()
        for dev in devices:
            tail = (work / dev.name / "device.log").read_text(encoding="utf-8").splitlines()
            print(f"--- {dev.name} (last lines; full log in {work / dev.name})")
            print("\n".join(tail[-5:]))

    print()
    for what, ok in results.items():
        print(f"  {'PASS' if ok else 'FAIL'}  {what}")
    return 0 if results and all(results.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
