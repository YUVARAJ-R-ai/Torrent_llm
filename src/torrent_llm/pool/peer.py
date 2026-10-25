"""``torrent-peer``: run one device of a pool.

    torrent-peer --pool configs/pool-lab.yaml --profile gaming --port 9001
    torrent-peer --pool configs/pool-lab.yaml --profile thinkpad --port 9002 --join 127.0.0.1:9001

The second command needs no more than the address of anyone already in the pool.
It learns the rest from them, and they learn of it.
"""

from __future__ import annotations

import argparse
import logging
import os
import signal
import socket
import sys
import threading
from dataclasses import dataclass
from pathlib import Path

import torch
import uvicorn
import yaml
from fastapi.middleware.cors import CORSMiddleware

from torrent_llm.chat import DEFAULT_SYSTEM_PROMPT
from torrent_llm.pool.api import Tokenizer, create_peer_app
from torrent_llm.pool.config import DeviceProfile, PoolConfig
from torrent_llm.pool.daemon import PeerDaemon
from torrent_llm.pool.hosting import Loader
from torrent_llm.shard import ShardSpec, load_shard

logger = logging.getLogger(__name__)


@dataclass
class RunningPeer:
    """A device that is up: the daemon and the HTTP server in front of it."""

    daemon: PeerDaemon
    server: uvicorn.Server
    thread: threading.Thread
    port: int

    @property
    def url(self) -> str:
        return f"http://{self.daemon.host}:{self.port}"

    def close(self, *, graceful: bool = True) -> None:
        self.daemon.stop(graceful=graceful)
        self.server.should_exit = True
        self.thread.join(timeout=3.0)


def launch(
    config: PoolConfig,
    profile: DeviceProfile,
    loader: Loader,
    tokenizer: Tokenizer,
    *,
    host: str = "127.0.0.1",
    port: int = 0,
    bind: str | None = None,
    join: list[str] | None = None,
    token: str | None = None,
    cors_origins: list[str] | None = None,
    stop_token_ids: set[int] | None = None,
    system_prompt: str | None = DEFAULT_SYSTEM_PROMPT,
    **daemon_kwargs,
) -> RunningPeer:
    """Bring up one device and return once it is answering HTTP."""
    # Bind first so the port is known before the device announces itself. Asking
    # for port 0 gets a free one, which is what tests and local labs want.
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind((bind or host, port))
    bound = sock.getsockname()[1]

    daemon = PeerDaemon(
        config,
        profile,
        loader,
        host=host,
        control_port=bound,
        bind=bind,
        join=join,
        token=token,
        **daemon_kwargs,
    )
    holder: dict[str, RunningPeer] = {}

    def leave() -> None:
        holder["peer"].close(graceful=True)
        os._exit(0)

    app = create_peer_app(
        daemon,
        tokenizer,
        stop_token_ids=stop_token_ids,
        system_prompt=system_prompt,
        on_leave=leave,
    )
    if cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=cors_origins,
            allow_methods=["*"],
            allow_headers=["*"],
        )
    server = uvicorn.Server(uvicorn.Config(app, log_level="warning"))
    thread = threading.Thread(
        target=lambda: server.run(sockets=[sock]), name="peer-http", daemon=True
    )
    thread.start()
    while not server.started:
        if not thread.is_alive():
            raise RuntimeError("the device's HTTP server failed to start")
        threading.Event().wait(0.02)

    daemon.start()
    peer = RunningPeer(daemon=daemon, server=server, thread=thread, port=bound)
    holder["peer"] = peer
    return peer


def _profile_from_args(args: argparse.Namespace, raw: dict) -> DeviceProfile:
    base: dict = {}
    if args.profile:
        match = [d for d in raw.get("devices", []) if d["name"] == args.profile]
        if not match:
            names = [d["name"] for d in raw.get("devices", [])]
            raise SystemExit(f"no device named {args.profile!r} in the pool file; have {names}")
        base = dict(match[0])
    overrides = {
        "name": args.name,
        "label": args.label,
        "device": args.device,
        "memory_mb": args.memory_mb,
        "speed": args.speed,
        "watch_owner": args.watch_owner,
        "owner_busy_mb": args.owner_busy_mb,
    }
    base.update({k: v for k, v in overrides.items() if v is not None})
    if args.link_latency_ms is not None or args.link_bandwidth_mbps is not None:
        base["link"] = {
            "latency_ms": args.link_latency_ms or 0.0,
            "bandwidth_mbps": args.link_bandwidth_mbps or 100.0,
        }
    if "name" not in base:
        base["name"] = socket.gethostname()
    return DeviceProfile.from_dict(base)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run one device of a Torrent-LLM pool.")
    parser.add_argument("--pool", required=True, help="pool YAML: model, stages, replicas")
    parser.add_argument("--profile", help="take this device's description from the pool file")
    parser.add_argument("--name", help="this device's name (unique in the pool)")
    parser.add_argument("--label", help="what to call it on the dashboard")
    parser.add_argument("--device", help="cpu, cuda, mps or auto")
    parser.add_argument("--memory-mb", type=int, help="memory this device lends the pool")
    parser.add_argument("--speed", type=float, help="relative speed, to rank devices")
    parser.add_argument(
        "--watch-owner",
        action="store_true",
        default=None,
        help="back off automatically when other programs take much of the GPU",
    )
    parser.add_argument(
        "--owner-busy-mb",
        type=int,
        help="GPU memory other programs may hold before this counts as the owner using it",
    )
    parser.add_argument("--link-latency-ms", type=float)
    parser.add_argument("--link-bandwidth-mbps", type=float)
    parser.add_argument("--host", default="127.0.0.1", help="address others reach this device on")
    parser.add_argument("--bind", default=None, help="address to listen on (default: --host)")
    parser.add_argument("--port", type=int, default=9000, help="HTTP control port")
    parser.add_argument("--join", action="append", default=[], help="host:port of any pool device")
    parser.add_argument("--token", default=os.environ.get("TORRENT_POOL_TOKEN"))
    parser.add_argument("--cors-origin", action="append", default=[])
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    )

    raw = yaml.safe_load(Path(args.pool).read_text())
    config = PoolConfig.from_dict(raw)
    profile = _profile_from_args(args, raw)
    token = args.token or raw.get("token")

    from transformers import AutoTokenizer

    from torrent_llm.chat import stop_token_ids

    tokenizer = AutoTokenizer.from_pretrained(
        config.model_id, trust_remote_code=config.trust_remote_code
    )

    def loader(spec: ShardSpec):
        return load_shard(
            config.model_id,
            spec,
            device=profile.device,
            dtype=getattr(torch, config.dtype),
            trust_remote_code=config.trust_remote_code,
        )

    peer = launch(
        config,
        profile,
        loader,
        tokenizer,
        host=args.host,
        port=args.port,
        bind=args.bind,
        join=args.join,
        token=token,
        cors_origins=args.cors_origin,
        stop_token_ids=stop_token_ids(config.model_id, tokenizer),
    )
    link = profile.link.describe() if profile.link else "none"
    print(f"{profile.name} is up at {peer.url}  (device {profile.device}, link {link})", flush=True)

    done = threading.Event()
    signal.signal(signal.SIGINT, lambda *_: done.set())
    signal.signal(signal.SIGTERM, lambda *_: done.set())
    done.wait()
    peer.close(graceful=True)
    return 0


__all__ = ["RunningPeer", "launch", "main"]

if __name__ == "__main__":
    sys.exit(main())
