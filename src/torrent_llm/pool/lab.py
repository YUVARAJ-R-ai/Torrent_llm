"""``torrent-lab``: a pool of pretend laptops on one machine.

Each simulated device is a separate operating-system process, running the same
``torrent-peer`` a real laptop would, with its own port, its own memory budget,
its own network link, and (where there is one) its own use of the GPU. They find
each other and talk over real sockets. What is simulated is only *where they
are*: nothing in the pool can tell they share a box except by the speed of the
wire, and the per-device link makes that unremarkable too.

The lab also lets you do to a device what the world does to laptops: switch it
on, switch it off cleanly, or kill it dead without a word. The dashboard drives
these through a small HTTP API.

None of this is part of the pool itself. A real deployment has no lab: people
start ``torrent-peer`` on their own machines.
"""

from __future__ import annotations

import argparse
import logging
import os
import signal
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from torrent_llm.pool.config import DeviceProfile, LabConfig

logger = logging.getLogger(__name__)

BASE_PORT = 9101


@dataclass
class LabDevice:
    """One simulated device: its profile, and the process playing it if it is on."""

    profile: DeviceProfile
    port: int
    process: subprocess.Popen | None = None
    started_at: float = 0.0
    log_path: Path | None = None
    #: What was last done to it, so the dashboard can tell "off" from "crashed".
    last_event: str = "never started"

    @property
    def running(self) -> bool:
        return self.process is not None and self.process.poll() is None

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}"


Command = Callable[[LabDevice, Sequence[str]], list[str]]


def peer_command(pool_file: Path, token: str | None, cors: Sequence[str]) -> Command:
    """How to start a real ``torrent-peer`` for a lab device."""

    def build(device: LabDevice, join: Sequence[str]) -> list[str]:
        cmd = [
            sys.executable,
            "-m",
            "torrent_llm.pool.peer",
            "--pool",
            str(pool_file),
            "--profile",
            device.profile.name,
            "--port",
            str(device.port),
        ]
        for address in join:
            cmd += ["--join", address]
        for origin in cors:
            cmd += ["--cors-origin", origin]
        if token:
            cmd += ["--token", token]
        return cmd

    return build


class Lab:
    """Starts, stops and kills simulated devices."""

    def __init__(
        self,
        config: LabConfig,
        command: Command,
        *,
        log_dir: Path,
        threads_per_device: int = 4,
    ) -> None:
        self.config = config
        self._command = command
        self._log_dir = log_dir
        self._threads = threads_per_device
        self._lock = threading.RLock()
        self.devices: dict[str, LabDevice] = {
            d.name: LabDevice(profile=d, port=BASE_PORT + i) for i, d in enumerate(config.devices)
        }
        log_dir.mkdir(parents=True, exist_ok=True)

    def _seeds(self, exclude: str) -> list[str]:
        """Addresses of running devices, for a starting device to introduce itself to."""
        return [
            f"127.0.0.1:{d.port}" for n, d in self.devices.items() if n != exclude and d.running
        ]

    def start(self, name: str) -> LabDevice:
        with self._lock:
            device = self._get(name)
            if device.running:
                return device
            log = self._log_dir / f"{name}.log"
            env = {
                **os.environ,
                # Several processes each spinning up every core fight each other
                # and make all of them slow. A laptop would not have the others'
                # cores, so give each device a few.
                "OMP_NUM_THREADS": str(self._threads),
                "MKL_NUM_THREADS": str(self._threads),
                "PYTHONUNBUFFERED": "1",
                "HF_HUB_OFFLINE": os.environ.get("HF_HUB_OFFLINE", "1"),
            }
            device.process = subprocess.Popen(  # noqa: S603
                self._command(device, self._seeds(name)),
                stdout=log.open("ab"),
                stderr=subprocess.STDOUT,
                env=env,
                start_new_session=True,
            )
            device.log_path = log
            device.started_at = time.time()
            device.last_event = "started"
            return device

    def stop(self, name: str, *, timeout: float = 8.0) -> LabDevice:
        """Switch off cleanly: the device says goodbye and the pool hears it."""
        with self._lock:
            device = self._get(name)
            if device.running:
                os.killpg(device.process.pid, signal.SIGTERM)
                try:
                    device.process.wait(timeout=timeout)
                except subprocess.TimeoutExpired:
                    os.killpg(device.process.pid, signal.SIGKILL)
            device.last_event = "switched off"
            return device

    def kill(self, name: str) -> LabDevice:
        """Kill it dead, with no goodbye: what a crash or a pulled plug looks like."""
        with self._lock:
            device = self._get(name)
            if device.running:
                os.killpg(device.process.pid, signal.SIGKILL)
                device.process.wait(timeout=5)
            device.last_event = "crashed"
            return device

    def shutdown(self) -> None:
        for name in list(self.devices):
            try:
                self.kill(name)
            except Exception:  # noqa: BLE001 - tear down what we can
                logger.exception("could not stop %s", name)

    def _get(self, name: str) -> LabDevice:
        if name not in self.devices:
            raise KeyError(name)
        return self.devices[name]

    def describe(self) -> list[dict[str, Any]]:
        out = []
        for d in self.devices.values():
            out.append(
                {
                    "name": d.profile.name,
                    "label": d.profile.label or d.profile.name,
                    "device": d.profile.device,
                    "memory_mb": d.profile.memory_mb,
                    "speed": d.profile.speed,
                    "link": d.profile.link.describe() if d.profile.link else None,
                    "port": d.port,
                    "url": d.url,
                    "running": d.running,
                    "pid": d.process.pid if d.running else None,
                    "started_at": d.started_at or None,
                    "last_event": d.last_event,
                }
            )
        return out


def wait_until_up(url: str, *, timeout: float = 90.0) -> bool:
    """Wait for a device's HTTP server to answer."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            if httpx.get(f"{url}/health", timeout=1.0).status_code == 200:
                return True
        except httpx.HTTPError:
            pass
        time.sleep(0.3)
    return False


def create_lab_app(lab: Lab, cors_origins: Sequence[str] = ()) -> FastAPI:
    app = FastAPI(title="Torrent-LLM lab", description=__doc__)
    if cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=list(cors_origins),
            allow_methods=["*"],
            allow_headers=["*"],
        )

    def device(name: str) -> LabDevice:
        try:
            return lab._get(name)
        except KeyError as exc:
            raise HTTPException(
                status_code=404, detail=f"no simulated device named {name!r}"
            ) from exc

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/lab")
    def lab_state() -> dict[str, Any]:
        pool = lab.config.pool
        return {
            "model_id": pool.model_id,
            "replicas": pool.replicas,
            "stages": pool.stages,
            "devices": lab.describe(),
        }

    @app.post("/lab/devices/{name}/start")
    def start(name: str) -> dict[str, Any]:
        device(name)
        lab.start(name)
        return {"ok": True}

    @app.post("/lab/devices/{name}/stop")
    def stop(name: str) -> dict[str, Any]:
        device(name)
        lab.stop(name)
        return {"ok": True}

    @app.post("/lab/devices/{name}/kill")
    def kill(name: str) -> dict[str, Any]:
        device(name)
        lab.kill(name)
        return {"ok": True}

    return app


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a pool of simulated devices on this machine.")
    parser.add_argument("--pool", required=True, help="pool YAML with a devices list")
    parser.add_argument("--start", type=int, default=None, help="how many devices to switch on")
    parser.add_argument("--port", type=int, default=8100, help="port of the lab's own API")
    parser.add_argument("--cors-origin", action="append", default=[])
    parser.add_argument("--log-dir", default="/tmp/torrent-pool")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)

    pool_file = Path(args.pool).resolve()
    config = LabConfig.from_file(pool_file)
    cors = args.cors_origin or ["http://localhost:3000", "http://127.0.0.1:3000"]
    lab = Lab(config, peer_command(pool_file, config.token, cors), log_dir=Path(args.log_dir))

    count = args.start if args.start is not None else (config.start or len(config.devices))
    names = [d.profile.name for d in list(lab.devices.values())[:count]]

    import uvicorn

    api = uvicorn.Server(
        uvicorn.Config(
            create_lab_app(lab, cors), host="127.0.0.1", port=args.port, log_level="warning"
        )
    )
    threading.Thread(target=api.run, name="lab-api", daemon=True).start()

    # The first device is the introduction point for the rest, so it must be up
    # before the others look for it. After that they come up together.
    first, rest = names[0], names[1:]
    print(f"switching on {first}", flush=True)
    lab.start(first)
    if not wait_until_up(lab.devices[first].url):
        print(f"{first} did not come up; see {lab.devices[first].log_path}", flush=True)
        lab.shutdown()
        return 1
    for name in rest:
        print(f"switching on {name}", flush=True)
        lab.start(name)

    print(f"lab ready: {len(names)} devices, API on http://127.0.0.1:{args.port}", flush=True)
    done = threading.Event()
    signal.signal(signal.SIGINT, lambda *_: done.set())
    signal.signal(signal.SIGTERM, lambda *_: done.set())
    done.wait()
    print("switching everything off", flush=True)
    lab.shutdown()
    api.should_exit = True
    return 0


if __name__ == "__main__":
    sys.exit(main())
