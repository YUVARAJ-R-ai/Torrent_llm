#!/usr/bin/env python
"""A pool with no models in it, for working on the dashboard.

    python scripts/mock_pool.py            # then run the dashboard as usual

It serves the same HTTP API as the real devices (ports 9101 and up) and the lab
(port 8100), so the dashboard cannot tell the difference. Nothing is loaded and
almost nothing is computed, which makes it cheap enough to keep running while you
change the page. Answers are canned.

It is not a stand-in for testing the pool. The decisions that matter (who holds
which stage, which copies a request takes) come from the real ``placement`` and
``router`` modules, so the *behaviour* is the pool's; only the timing and the text
are made up. To see the real thing, use ``scripts/pool.sh``.
"""

from __future__ import annotations

import argparse
import random
import threading
import time
from collections import deque
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from torrent_llm.pool.config import LabConfig
from torrent_llm.pool.members import (
    ACTIVE,
    AWAY,
    DOWN,
    DRAINING,
    LOADING,
    READY,
    HostedStage,
    Member,
)
from torrent_llm.pool.placement import coverage, wanted_stages
from torrent_llm.pool.router import NoReplica, pick_route

BASE_PORT = 9101
LOAD_S = 1.6  # how long a stage takes to load
DEAD_AFTER_S = 1.4  # how long before others notice a crashed device
DRAIN_GRACE_S = 5.0

ANSWERS = {
    "france": "The capital of France is Paris.",
    "hamlet": "William Shakespeare wrote Hamlet.",
    "japan": "The capital of Japan is Tokyo.",
    "times": "That is 81.",
    "story": (
        "A small robot named Pip lived in a quiet workshop, surrounded by tins of paint it "
        "had never dared to open. One rainy afternoon it dipped a brush into the bluest tin "
        "and drew a single wobbly line across a blank canvas. The line looked like a river. "
        "Pip painted a bridge over it, then a village beside it, and by evening the whole "
        "workshop glowed with colour. From that day on, Pip painted every morning."
    ),
}


def answer_for(prompt: str) -> str:
    text = prompt.lower()
    for key, reply in ANSWERS.items():
        if key in text:
            return reply
    return "That is a good question, and the pool answered it across several devices."


class Device:
    def __init__(self, profile, index: int) -> None:
        self.profile = profile
        self.index = index
        self.port = BASE_PORT + index
        self.running = False
        self.state = ACTIVE
        self.crashed_at: float | None = None
        self.drain_started = 0.0
        self.last_event = "never started"
        self.hosting: dict[int, tuple[str, float]] = {}  # stage -> (status, ready_at)
        self.inflight = 0
        self.served = 0
        self.stage_ports = {}


class MockPool:
    def __init__(self, config: LabConfig, start: int) -> None:
        self.config = config
        self.pool = config.pool
        self.devices = {d.name: Device(d, i) for i, d in enumerate(config.devices)}
        self.events: deque[dict] = deque(maxlen=80)
        self.lock = threading.RLock()
        self._started = time.time()
        for device in list(self.devices.values())[:start]:
            self._switch_on(device, quiet=True)
        threading.Thread(target=self._loop, daemon=True).start()

    # --- the world ---------------------------------------------------------------

    def _say(self, kind: str, text: str, member: str | None = None) -> None:
        self.events.append({"t": time.time(), "kind": kind, "text": text, "member": member})

    def _switch_on(self, device: Device, quiet: bool = False) -> None:
        device.running = True
        device.state = ACTIVE
        device.crashed_at = None
        device.hosting = {}
        device.last_event = "started"
        if not quiet:
            self._say("joined", f"{device.profile.name} joined the pool", device.profile.name)

    def kill(self, name: str) -> None:
        with self.lock:
            device = self.devices[name]
            if device.running:
                device.running = False
                device.crashed_at = time.time()
                device.last_event = "crashed"

    def stop(self, name: str) -> None:
        with self.lock:
            device = self.devices[name]
            if device.running:
                device.running = False
                device.crashed_at = time.time() - 10  # said goodbye: others know at once
                device.last_event = "switched off"
                self._say("down", f"{name} left", name)

    def start(self, name: str) -> None:
        with self.lock:
            device = self.devices[name]
            if not device.running:
                self._switch_on(device)

    def set_owner(self, name: str, busy: bool) -> str:
        with self.lock:
            device = self.devices[name]
            if busy and device.state == ACTIVE:
                device.state = DRAINING
                device.drain_started = time.time()
                self._say(
                    "state", f"{name} is being used by its owner, handing its work over", name
                )
            elif not busy and device.state in (DRAINING, AWAY):
                device.state = ACTIVE
                self._say("state", f"{name} is lending its resources again", name)
            return device.state

    # --- table view ----------------------------------------------------------------

    def liveness(self, device: Device) -> str:
        if device.running:
            return "alive"
        if device.crashed_at is None:
            return DOWN
        quiet = time.time() - device.crashed_at
        return "alive" if quiet < 0.5 else ("suspect" if quiet < DEAD_AFTER_S else DOWN)

    def member(self, device: Device) -> Member:
        p = device.profile
        hosting = tuple(
            HostedStage(
                stage=s,
                port=50000 + device.index * 10 + s,
                status=status,
                inflight=device.inflight if status == READY else 0,
                served=device.served,
            )
            for s, (status, _) in sorted(device.hosting.items())
        )
        return Member(
            name=p.name,
            host="127.0.0.1",
            control_port=device.port,
            label=p.label or p.name,
            os="Linux",
            device=p.device if p.device != "auto" else "cpu",
            memory_mb=p.memory_mb,
            slots=p.slots(self.pool.shard_mb),
            speed=p.speed,
            link=p.link.describe() if p.link else None,
            state=device.state,
            hosting=hosting,
        )

    def known(self) -> list[Device]:
        """Devices the pool has heard of: running now, or crashed/stopped earlier."""
        return [d for d in self.devices.values() if d.running or d.crashed_at is not None]

    def usable(self) -> list[Member]:
        return [self.member(d) for d in self.known() if self.liveness(d) == "alive"]

    # --- the beat ------------------------------------------------------------------

    def _loop(self) -> None:
        while True:
            time.sleep(0.3)
            with self.lock:
                self._tick()

    def _tick(self) -> None:
        now = time.time()
        for d in self.devices.values():
            if not d.running and d.crashed_at is not None and self.liveness(d) == DOWN:
                if d.hosting:
                    self._say("down", f"{d.profile.name} went down", d.profile.name)
                d.hosting = {}
            for stage, (status, ready_at) in list(d.hosting.items()):
                if status == LOADING and now >= ready_at:
                    d.hosting[stage] = (READY, ready_at)
                    self._say(
                        "stage", f"{d.profile.name} now serves stage {stage + 1}", d.profile.name
                    )

        members = self.usable()
        wanted = wanted_stages(self.pool, members)
        for d in self.devices.values():
            if not d.running:
                continue
            want = set(wanted.get(d.profile.name, []))
            if d.state == ACTIVE:
                for stage in want - set(d.hosting):
                    d.hosting[stage] = (LOADING, now + LOAD_S)
            else:
                for stage in list(d.hosting):
                    others = [m for m in members if m.name != d.profile.name and m.state == ACTIVE]
                    ready = sum(1 for m in others if m.hosts(stage, ready_only=True))
                    could = any(m.slots - len(m.hosting) > 0 and not m.hosts(stage) for m in others)
                    safe = (
                        ready >= self.pool.replicas
                        or (ready >= 1 and not could)
                        or now - d.drain_started > DRAIN_GRACE_S
                    )
                    if safe and d.inflight == 0:
                        del d.hosting[stage]
                        self._say(
                            "stage",
                            f"{d.profile.name} stopped serving stage {stage + 1}",
                            d.profile.name,
                        )
                if not d.hosting and d.state == DRAINING:
                    d.state = AWAY
                    self._say("state", f"{d.profile.name} has released its memory", d.profile.name)

    # --- reading -------------------------------------------------------------------

    def view(self, me: Device) -> dict:
        rows = []
        usable = []
        for d in self.known():
            m = self.member(d)
            live = self.liveness(d)
            row = m.to_dict()
            row["liveness"] = live
            row["is_self"] = d is me
            row["rtt_ms"] = None
            rows.append(row)
            if live == "alive":
                usable.append(m)
        ready = coverage(self.pool, usable)
        low = min(ready, default=0)
        health = "healthy" if low >= self.pool.replicas else ("degraded" if low >= 1 else "down")
        return {
            "self": me.profile.name,
            "pool": {
                "model_id": self.pool.model_id,
                "num_layers": self.pool.num_layers,
                "replicas": self.pool.replicas,
                "shard_mb": self.pool.shard_mb,
                "stages": [
                    {"stage": s.index, "start": s.start, "end": s.end} for s in self.pool.specs()
                ],
            },
            "members": rows,
            "ready_copies": ready,
            "health": health,
            "now": time.time(),
        }

    def lab(self) -> dict:
        return {
            "model_id": self.pool.model_id,
            "replicas": self.pool.replicas,
            "stages": self.pool.stages,
            "devices": [
                {
                    "name": d.profile.name,
                    "label": d.profile.label or d.profile.name,
                    "device": d.profile.device,
                    "memory_mb": d.profile.memory_mb,
                    "speed": d.profile.speed,
                    "link": d.profile.link.describe() if d.profile.link else None,
                    "port": d.port,
                    "url": f"http://127.0.0.1:{d.port}",
                    "running": d.running,
                    "pid": 40000 + d.index if d.running else None,
                    "started_at": self._started,
                    "last_event": d.last_event,
                }
                for d in self.devices.values()
            ],
        }

    # --- answering -----------------------------------------------------------------

    def generate(self, me: Device, prompt: str, max_new_tokens: int) -> dict:
        words = answer_for(prompt).split(" ")
        tokens = [w + (" " if i < len(words) - 1 else "") for i, w in enumerate(words)]
        tokens = tokens[:max_new_tokens]
        prompt_tokens = 30 + len(prompt) // 4
        passes, events = [], []
        started = time.monotonic()
        route = self._route()
        recovery = False
        i = 0
        while i < len(tokens):
            time.sleep(0.06)
            with self.lock:
                dead = [r for r in route if not self.devices[r.member].running]
            if dead:
                lost = dead[0]
                events.append(
                    {
                        "kind": "failover",
                        "at_pass": len(passes),
                        "step": i,
                        "stage": lost.stage,
                        "member": lost.member,
                        "detail": "UNAVAILABLE",
                        "t": time.monotonic() - started,
                    }
                )
                # Like the real pool: a request that failed against a device knows
                # more than the heartbeat timer, so it stops using it at once.
                with self.lock:
                    self.devices[lost.member].crashed_at = time.time() - DEAD_AFTER_S
                time.sleep(0.3)
                route = self._route(wait=True)
                recovery = True
                continue
            seq = prompt_tokens + i if recovery or i == 0 else 1
            hops = []
            for r in route:
                d = self.devices[r.member]
                cpu = d.profile.device != "cuda"
                compute = (28.0 if cpu else 7.0) / max(d.profile.speed, 0.3) * (1 + seq / 80)
                link = d.profile.link
                latency = link.latency_ms if link else 0.0
                bytes_ = seq * 2048 if r.stage > 0 else seq * 8
                transport = (
                    latency + (bytes_ * 8 / ((link.bandwidth_mbps if link else 1000) * 1e6)) * 1000
                )
                hops.append(
                    {
                        "stage": r.stage,
                        "member": r.member,
                        "address": r.address,
                        "sent_bytes": bytes_,
                        "received_bytes": 8 if r.stage == len(route) - 1 else seq * 2048,
                        "compute_ms": round(compute, 2),
                        "transport_ms": round(transport, 2),
                        "wall_ms": round(compute + transport, 2),
                    }
                )
            passes.append(
                {
                    "index": len(passes),
                    "step": i,
                    "phase": "prefill" if seq > 1 else "decode",
                    "recovery": recovery,
                    "seq_len": seq,
                    "route": [[r.stage, r.member] for r in route],
                    "hops": hops,
                    "token_id": 1000 + i,
                }
            )
            if recovery:
                events.append(
                    {
                        "kind": "recovered",
                        "at_pass": len(passes),
                        "step": i,
                        "stage": None,
                        "member": None,
                        "detail": " -> ".join(r.member for r in route),
                        "t": time.monotonic() - started,
                    }
                )
                recovery = False
            i += 1
        all_hops = [h for p in passes for h in p["hops"]]
        return {
            "prompt": prompt,
            "completion": "".join(tokens),
            "tokens": tokens,
            "requester": me.profile.name,
            "passes": passes,
            "events": events,
            "totals": {
                "tokens": len(tokens),
                "failovers": sum(1 for e in events if e["kind"] == "failover"),
                "sent_bytes": sum(h["sent_bytes"] for h in all_hops),
                "compute_ms": round(sum(h["compute_ms"] for h in all_hops), 2),
                "transport_ms": round(sum(h["transport_ms"] for h in all_hops), 2),
            },
        }

    def _route(self, wait: bool = False):
        deadline = time.monotonic() + (6 if wait else 0)
        while True:
            try:
                return pick_route(self.pool, self.usable(), rng=random.Random())
            except NoReplica as exc:
                if time.monotonic() >= deadline:
                    raise HTTPException(status_code=503, detail=str(exc)) from exc
                time.sleep(0.2)


class GenerateIn(BaseModel):
    prompt: str
    max_new_tokens: int = 40
    system_prompt: str | None = None
    chat: bool = True


class OwnerIn(BaseModel):
    busy: bool


def build_app(mock: MockPool) -> FastAPI:
    app = FastAPI(title="Mock pool")
    app.add_middleware(
        CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
    )

    def me(request: Request) -> Device:
        port = request.url.port
        device = next((d for d in mock.devices.values() if d.port == port), None)
        if device is None or not device.running:
            raise HTTPException(status_code=503, detail="this device is not running")
        return device

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok"}

    @app.get("/pool")
    def pool(request: Request) -> dict:
        return mock.view(me(request))

    @app.get("/events")
    def events(request: Request, since: float = 0.0) -> list[dict]:
        me(request)
        return [e for e in mock.events if e["t"] > since]

    @app.post("/generate")
    def generate(request: Request, body: GenerateIn) -> dict:
        device = me(request)
        device.inflight += 1
        try:
            return mock.generate(device, body.prompt, body.max_new_tokens)
        finally:
            device.inflight -= 1

    @app.post("/admin/owner")
    def owner(request: Request, body: OwnerIn) -> dict:
        return {"state": mock.set_owner(me(request).profile.name, body.busy)}

    @app.get("/lab")
    def lab() -> dict:
        return mock.lab()

    for action in ("start", "stop", "kill"):

        def handler(name: str, _action: str = action) -> dict:
            if name not in mock.devices:
                raise HTTPException(status_code=404, detail=f"no simulated device named {name!r}")
            getattr(mock, _action)(name)
            return {"ok": True}

        app.post(f"/lab/devices/{{name}}/{action}")(handler)
    return app


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool", default="configs/pool-lab.yaml")
    parser.add_argument("--start", type=int, default=5)
    args = parser.parse_args()

    config = LabConfig.from_file(Path(args.pool))
    mock = MockPool(config, args.start)
    app = build_app(mock)
    ports = [d.port for d in mock.devices.values()] + [8100]
    servers = [
        uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=p, log_level="error"))
        for p in ports
    ]
    for server in servers[1:]:
        threading.Thread(target=server.run, daemon=True).start()
    print(
        f"mock pool: {args.start} devices up, ports {ports[0]}-{ports[-2]} and lab on 8100",
        flush=True,
    )
    try:
        servers[0].run()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
