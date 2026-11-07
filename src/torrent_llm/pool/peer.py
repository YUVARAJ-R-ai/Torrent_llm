"""``torrent-peer``: run one device of a pool.

    torrent-peer --pool configs/pool-lab.yaml --profile gaming --port 9001
    torrent-peer --pool configs/pool-lab.yaml --profile thinkpad --port 9002 --join 127.0.0.1:9001

The second command needs no more than the address of anyone already in the pool.
It learns the rest from them, and they learn of it.
"""

from __future__ import annotations

import logging
import os
import signal
import socket
import sys
import threading
from dataclasses import dataclass

import torch
import uvicorn
from fastapi.middleware.cors import CORSMiddleware

from torrent_llm.chat import DEFAULT_SYSTEM_PROMPT
from torrent_llm.pool.api import Tokenizer, create_peer_app
from torrent_llm.pool.config import DeviceProfile, PoolConfig
from torrent_llm.pool.daemon import PeerDaemon
from torrent_llm.pool.doctor import FAIL, report, run_checks
from torrent_llm.pool.hosting import Loader
from torrent_llm.pool.settings import PeerSettings, build_parser, settings_from_args
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
    cors_regex: str | None = None,
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
    if cors_origins or cors_regex:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=cors_origins or [],
            allow_origin_regex=cors_regex,
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


def banner(settings: PeerSettings, peer: RunningPeer) -> str:
    """What a person needs to see once the device is up, in the order they need it."""
    sl = settings.slots
    lines = [
        "",
        f'  Torrent-LLM pool device "{settings.profile.name}" is running.',
        "",
        f"  Open this in a browser to see and use the pool:   {settings.url}",
        "",
        f"  Hardware : {settings.resolved_device}, lending {settings.profile.memory_mb} MB "
        f"({settings.memory_source}), room for {sl} stage{'s' if sl != 1 else ''}",
        f"  Model    : {settings.config.model_id}, loading in the background",
    ]
    if settings.discover:
        lines.append("  Others   : devices on this network will find this one by themselves")
    lines.append(
        f"  Or point another device here:   torrent-peer --join {settings.host}:{settings.port}"
    )
    if settings.token:
        lines.append("  Token    : on (other devices need the same one)")
    ports = f"TCP {settings.port}"
    if settings.shard_ports:
        ports += f", {settings.shard_ports[0]}-{settings.shard_ports[1]}"
    lines.append(f"  Ports    : {ports}" + ("  and UDP 9099" if settings.discover else ""))
    for note in settings.notes:
        lines.append(f"  NOTE     : {note}")
    if sl == 0:
        lines.append(
            f"  NOTE     : {settings.profile.memory_mb} MB is less than one stage "
            f"({settings.config.shard_mb} MB), so this device will join but hold nothing. "
            "Close other programs or pass --memory-mb."
        )
    lines += ["", "  Stop with Ctrl+C. The device says goodbye so the pool adjusts at once.", ""]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    )
    settings = settings_from_args(args)

    if args.check:
        checks = run_checks(settings)
        print(report(checks))
        return 1 if any(c.status == FAIL for c in checks) else 0

    if args.mode:
        from torrent_llm.pool import swarm_cli

        return swarm_cli.main(args, settings)

    config, profile = settings.config, settings.profile

    try:
        from transformers import AutoTokenizer

        from torrent_llm.chat import stop_token_ids

        tokenizer = AutoTokenizer.from_pretrained(
            config.model_id, trust_remote_code=config.trust_remote_code
        )
    except OSError as exc:
        print(
            f"\nCould not load {config.model_id}: {exc}\n\n"
            "The model is probably not downloaded on this machine. Connect to the internet "
            "and run scripts/setup.sh (or scripts\\setup.ps1 on Windows), then try again.\n"
            "To check everything at once: torrent-peer --check\n",
            file=sys.stderr,
        )
        return 1

    def loader(spec: ShardSpec):
        return load_shard(
            config.model_id,
            spec,
            device=profile.device,
            dtype=getattr(torch, config.dtype),
            trust_remote_code=config.trust_remote_code,
        )

    try:
        peer = launch(
            config,
            profile,
            loader,
            tokenizer,
            host=settings.host,
            port=settings.port,
            bind=settings.bind,
            join=settings.join,
            token=settings.token,
            cors_origins=settings.cors_origins,
            cors_regex=settings.cors_regex,
            stop_token_ids=stop_token_ids(config.model_id, tokenizer),
            discover=settings.discover,
            shard_port_base=settings.shard_port_base,
        )
    except OSError as exc:
        print(
            f"\nCould not start on port {settings.port}: {exc}\n\n"
            "Another program (or another torrent-peer) is probably using it. Stop that one, "
            "or choose another port with --port.\nTo check everything: torrent-peer --check\n",
            file=sys.stderr,
        )
        return 1

    print(banner(settings, peer), flush=True)

    done = threading.Event()
    signal.signal(signal.SIGINT, lambda *_: done.set())
    signal.signal(signal.SIGTERM, lambda *_: done.set())
    # Wait in short slices: on Windows a wait with no timeout cannot be interrupted by
    # Ctrl+C, which would leave the device impossible to stop from its own terminal.
    while not done.wait(0.5):
        pass
    print("Leaving the pool...", flush=True)
    peer.close(graceful=True)
    return 0


def start_pool_device(settings: PeerSettings, *, node, ui_lan: bool = False) -> RunningPeer:
    """Run the pool device on an overlay ``node`` (the swarm modes).

    The control panel listens on 127.0.0.1 (or on the LAN with ``ui_lan``) and its
    owner actions need the admin token from the data directory.
    """
    from transformers import AutoTokenizer

    from torrent_llm.chat import stop_token_ids
    from torrent_llm.pool.net import lan_address
    from torrent_llm.pool.swarm_cli import admin_token

    config, profile = settings.config, settings.profile
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            config.model_id, trust_remote_code=config.trust_remote_code
        )
    except OSError as exc:
        raise RuntimeError(
            f"could not load {config.model_id} ({exc}). The model is probably not "
            "downloaded: connect to the internet and run scripts/setup.sh "
            "(scripts\\setup.ps1 on Windows)."
        ) from exc

    def loader(spec: ShardSpec):
        return load_shard(
            config.model_id,
            spec,
            device=profile.device,
            dtype=getattr(torch, config.dtype),
            trust_remote_code=config.trust_remote_code,
        )

    host = lan_address() if ui_lan else "127.0.0.1"
    peer = launch(
        config,
        profile,
        loader,
        tokenizer,
        host=host,
        port=settings.port,
        bind="0.0.0.0" if ui_lan else "127.0.0.1",
        join=[],
        token=admin_token(),
        cors_origins=settings.cors_origins,
        cors_regex=settings.cors_regex,
        stop_token_ids=stop_token_ids(config.model_id, tokenizer),
        discover=False,
        node=node,
    )
    print(f"  Control panel: http://{host}:{peer.port}", flush=True)
    return peer


__all__ = ["RunningPeer", "launch", "main", "start_pool_device"]

if __name__ == "__main__":
    sys.exit(main())
