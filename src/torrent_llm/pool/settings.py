"""Everything ``torrent-peer`` decides before it starts, in one place.

Kept apart from the code that starts things so it can be tested with no network, no
model and no GPU, and so ``--check`` and the real start-up are guaranteed to agree
about what this device is about to do.

The defaults are for a real machine in front of a person: find the network address
automatically, lend a share of what is free, discover other devices by broadcast, and
use a fixed range of ports so a firewall can be opened for exactly them. A lab on one
machine overrides the few that differ.
"""

from __future__ import annotations

import argparse
import os
import socket
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import yaml

from torrent_llm.pool import net
from torrent_llm.pool.config import DeviceProfile, PoolConfig
from torrent_llm.shard import resolve_device

DEFAULT_CONTROL_PORT = 9000
DEFAULT_SHARD_PORT_BASE = 9200

#: Pages served by one device may call another device on the same network. Only
#: private and Tailscale ranges, never the open internet.
PRIVATE_ORIGIN_REGEX = (
    r"^https?://(localhost|127\.0\.0\.1|10\.\d+\.\d+\.\d+|192\.168\.\d+\.\d+|"
    r"172\.(1[6-9]|2\d|3[01])\.\d+\.\d+|100\.(6[4-9]|[7-9]\d|1[01]\d|12[0-7])\.\d+\.\d+)(:\d+)?$"
)


def default_pool_file() -> Path:
    """``configs/pool.yaml``, found from where the command is run or from the repository."""
    for candidate in (
        Path("configs/pool.yaml"),
        Path(__file__).resolve().parents[3] / "configs/pool.yaml",
    ):
        if candidate.exists():
            return candidate
    return Path("configs/pool.yaml")


@dataclass
class PeerSettings:
    """What this device is going to be."""

    pool_file: Path
    pool_raw: dict[str, Any]
    config: PoolConfig
    profile: DeviceProfile
    resolved_device: str
    host: str
    bind: str
    port: int
    shard_port_base: int
    join: list[str]
    discover: bool
    token: str | None
    cors_origins: list[str]
    cors_regex: str | None
    #: How memory was chosen, for the banner: "auto" or "set by you".
    memory_source: str = "auto"
    notes: list[str] = field(default_factory=list)

    @property
    def shard_ports(self) -> tuple[int, int] | None:
        if not self.shard_port_base:
            return None
        return (self.shard_port_base, self.shard_port_base + self.config.stages - 1)

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}"

    @property
    def slots(self) -> int:
        return self.profile.slots(self.config.shard_mb)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="torrent-peer",
        description=(
            "Run one device of a Torrent-LLM pool. With no options it finds this machine's "
            "address, lends a share of what is free, and finds the other devices on the "
            "network by itself."
        ),
    )
    parser.add_argument(
        "mode",
        nargs="?",
        choices=["create", "join", "public", "bootstrap"],
        help=(
            "create: start a new invite swarm and print its link; join LINK: join one; "
            "public: join the public swarm; bootstrap: run a relay/discovery node with no "
            "model. Without a mode, the device joins a pool on this network as before."
        ),
    )
    parser.add_argument("invite", nargs="?", help="the tllm://join/... link (with join)")
    parser.add_argument(
        "--pool", help="pool file: model, stages, replicas (default: configs/pool.yaml)"
    )
    parser.add_argument("--swarm-name", help="name of a new invite swarm (with create)")
    parser.add_argument(
        "--bootstrap",
        action="append",
        default=[],
        help="address of a swarm member, e.g. tcp://203.0.113.7:9100 (repeatable)",
    )
    parser.add_argument(
        "--listen-port",
        type=int,
        default=9100,
        help="TCP port for encrypted peer connections in a swarm (default 9100)",
    )
    parser.add_argument(
        "--no-upnp", action="store_true", help="do not ask the router to forward the port"
    )
    parser.add_argument(
        "--relay", action="store_true", help="always offer to relay for unreachable devices"
    )
    parser.add_argument(
        "--privacy",
        action="store_true",
        help=(
            "keep the first and last stage of the model on this device, so other devices "
            "never see your prompt's or answer's tokens (needs memory for two stages)"
        ),
    )
    parser.add_argument(
        "--ui-lan",
        action="store_true",
        help="also serve the read-only dashboard to this network (default: this machine only)",
    )
    parser.add_argument("--check", action="store_true", help="check this machine is ready and exit")
    parser.add_argument(
        "--name", help="this device's name, unique in the pool (default: the hostname)"
    )
    parser.add_argument("--label", help="what to call it on the dashboard")
    parser.add_argument("--device", help="cpu, cuda, mps or auto (default: auto)")
    parser.add_argument(
        "--memory-mb",
        default="auto",
        help="memory to lend the pool, in MB, or 'auto' for a share of what is free",
    )
    parser.add_argument("--speed", type=float, help="relative speed, only to rank devices")
    parser.add_argument(
        "--watch-owner",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="back off when other programs take the GPU (default: on for an NVIDIA GPU)",
    )
    parser.add_argument(
        "--owner-busy-mb", type=int, help="GPU memory others may hold before backing off"
    )
    parser.add_argument(
        "--max-requests",
        type=int,
        help="questions to start at once before refusing more (default: 8 on a GPU, 3 on a CPU)",
    )
    parser.add_argument(
        "--profile", help="take this device's description from the pool file (lab use)"
    )
    parser.add_argument("--link-latency-ms", type=float)
    parser.add_argument("--link-bandwidth-mbps", type=float)
    parser.add_argument(
        "--host",
        default="auto",
        help="address others reach this device on: auto, lan, tailscale, or an address",
    )
    parser.add_argument(
        "--bind", help="address to listen on (default: everywhere, unless --host is loopback)"
    )
    parser.add_argument(
        "--port", type=int, default=DEFAULT_CONTROL_PORT, help="control and web port"
    )
    parser.add_argument(
        "--shard-port-base",
        type=int,
        default=DEFAULT_SHARD_PORT_BASE,
        help="first port for model servers; stage N uses base+N (0 lets the system choose)",
    )
    parser.add_argument(
        "--join", action="append", default=[], help="host:port of any device in the pool"
    )
    parser.add_argument(
        "--discover",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="find devices on the local network by broadcast",
    )
    parser.add_argument(
        "--token", default=os.environ.get("TORRENT_POOL_TOKEN"), help="shared pool secret"
    )
    parser.add_argument("--cors-origin", action="append", default=[])
    parser.add_argument("--verbose", action="store_true")
    return parser


def settings_from_args(args: argparse.Namespace) -> PeerSettings:
    pool_file = Path(args.pool) if args.pool else default_pool_file()
    if not pool_file.exists():
        raise SystemExit(
            f"pool file not found: {pool_file}\nRun from the repository, or pass --pool."
        )
    raw = yaml.safe_load(pool_file.read_text())
    config = PoolConfig.from_dict(raw)
    config = _with_resolved_weights(config, pool_file)

    base: dict[str, Any] = {}
    if args.profile:
        match = [d for d in raw.get("devices", []) if d["name"] == args.profile]
        if not match:
            names = [d["name"] for d in raw.get("devices", [])]
            raise SystemExit(f"no device named {args.profile!r} in the pool file; have {names}")
        base = dict(match[0])
    for key, value in {
        "name": args.name,
        "label": args.label,
        "device": args.device,
        "speed": args.speed,
        "owner_busy_mb": args.owner_busy_mb,
        "max_requests": args.max_requests,
    }.items():
        if value is not None:
            base[key] = value
    base.setdefault("name", socket.gethostname())
    base.setdefault("device", "auto")
    if args.link_latency_ms is not None or args.link_bandwidth_mbps is not None:
        base["link"] = {
            "latency_ms": args.link_latency_ms or 0.0,
            "bandwidth_mbps": args.link_bandwidth_mbps or 100.0,
        }

    resolved = resolve_device(base["device"])
    notes: list[str] = []

    memory_source = "set by you"
    if args.memory_mb != "auto":
        base["memory_mb"] = int(args.memory_mb)
    elif "memory_mb" not in base:
        base["memory_mb"] = net.auto_memory_mb(resolved)
        memory_source = "a share of what is free"
    else:
        memory_source = "set in the pool file"

    if args.watch_owner is not None:
        base["watch_owner"] = args.watch_owner
    elif "watch_owner" not in base:
        # On a laptop with an NVIDIA GPU, the owner is the main reason to back off.
        base["watch_owner"] = resolved.startswith("cuda")

    host = net.choose_host(args.host)
    if host == "127.0.0.1" and args.host in ("auto", "lan"):
        notes.append(
            "no network address found, so only this machine can reach the device. "
            "Connect to Wi-Fi or Ethernet, or pass --host."
        )
    loopback = host.startswith("127.")
    bind = args.bind or ("127.0.0.1" if loopback else "0.0.0.0")  # noqa: S104

    return PeerSettings(
        pool_file=pool_file,
        pool_raw=raw,
        config=config,
        profile=DeviceProfile.from_dict(base),
        resolved_device=resolved,
        host=host,
        bind=bind,
        port=args.port,
        shard_port_base=args.shard_port_base,
        join=list(args.join),
        discover=args.discover,
        token=args.token or raw.get("token"),
        cors_origins=list(args.cors_origin),
        cors_regex=None if args.cors_origin else PRIVATE_ORIGIN_REGEX,
        memory_source=memory_source,
        notes=notes,
    )


def _with_resolved_weights(config: PoolConfig, pool_file: Path) -> PoolConfig:
    """Find the codec's weights file wherever the command was run from.

    The pool file names weights relative to the repository (``weights/x.safetensors``).
    A person may run the command from any directory, so look beside the pool file's
    repository as well as the current one, and say clearly when it is nowhere.
    """
    weights = config.codec_args.get("weights_path")
    if not weights or Path(weights).is_absolute() or Path(weights).exists():
        return config
    for root in (pool_file.resolve().parent.parent, Path(__file__).resolve().parents[3]):
        candidate = root / weights
        if candidate.exists():
            return replace(config, codec_args={**config.codec_args, "weights_path": str(candidate)})
    raise SystemExit(
        f"the pool file asks for codec weights {weights!r} but that file was not found.\n"
        "It ships in the repository's weights/ folder; run from the repository, or "
        "regenerate it with python scripts/calibrate_codec.py."
    )
