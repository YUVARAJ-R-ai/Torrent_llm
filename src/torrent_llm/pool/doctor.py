"""``torrent-peer --check``: is this machine ready to join a pool?

Setting up several real machines goes wrong in a small number of boring ways: the
model was never downloaded, a firewall blocks the ports, the network has no address,
there is not enough free memory, two devices picked the same name. Each of those looks
like "it does not work" from the outside, so this checks them one at a time and says
what to do about each, in plain words.

Each check is a small function that returns a :class:`Check`. None of them changes
anything on the machine.
"""

from __future__ import annotations

import importlib
import platform
import socket
import sys
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass

from torrent_llm.pool import net
from torrent_llm.pool.discovery import DISCOVERY_PORT
from torrent_llm.pool.settings import PeerSettings

OK, WARN, FAIL = "ok", "warn", "fail"


@dataclass(frozen=True)
class Check:
    name: str
    status: str
    detail: str
    #: What to do, when the status is not ok.
    fix: str = ""


def check_python() -> Check:
    version = sys.version_info
    text = f"Python {version.major}.{version.minor}.{version.micro}"
    if version < (3, 11):
        return Check(
            "Python",
            FAIL,
            text,
            "Python 3.11 or newer is needed. Run scripts/setup.sh (it installs 3.12 for you).",
        )
    return Check("Python", OK, text)


def check_packages() -> Check:
    missing = []
    for module in ("torch", "transformers", "grpc", "fastapi", "uvicorn", "yaml", "safetensors"):
        try:
            importlib.import_module(module)
        except ImportError:
            missing.append(module)
    if missing:
        return Check(
            "Packages",
            FAIL,
            "missing: " + ", ".join(missing),
            "Run scripts/setup.sh (Linux/macOS) or scripts\\setup.ps1 (Windows).",
        )
    return Check("Packages", OK, "everything the device needs is installed")


def check_stubs() -> Check:
    try:
        importlib.import_module("torrent_llm.transport.activation_pb2")
    except ImportError:
        return Check(
            "Network protocol",
            FAIL,
            "the generated gRPC files are missing",
            "Run: python -m torrent_llm.codegen   (scripts/setup.sh does this for you)",
        )
    return Check("Network protocol", OK, "gRPC files generated")


def check_hardware(settings: PeerSettings) -> Check:
    try:
        import torch

        if settings.resolved_device.startswith("cuda") and torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            return Check("Hardware", OK, f"GPU: {name}, {net.free_vram_mb()} MB free")
        if settings.resolved_device == "mps":
            return Check("Hardware", OK, "Apple GPU (MPS)")
    except Exception:  # noqa: BLE001 - a broken driver is a finding, not a crash
        return Check(
            "Hardware",
            WARN,
            "the GPU could not be read",
            "Update the graphics driver, or use --device cpu.",
        )
    return Check(
        "Hardware",
        OK,
        f"CPU only, {net.available_ram_mb()} MB of RAM free (slower than a GPU, but it works)",
    )


def check_model(settings: PeerSettings) -> Check:
    model = settings.config.model_id
    try:
        from huggingface_hub import snapshot_download

        snapshot_download(model, local_files_only=True)
    except Exception:  # noqa: BLE001 - not cached, or no hub library: either way, not usable offline
        return Check(
            "Model",
            FAIL,
            f"{model} is not downloaded on this machine",
            "Connect to the internet and run scripts/setup.sh, which downloads it (about 1.5 GB).",
        )
    return Check("Model", OK, f"{model} is downloaded")


def check_memory(settings: PeerSettings) -> Check:
    stage = settings.config.shard_mb
    lending = settings.profile.memory_mb
    if lending < stage:
        return Check(
            "Memory",
            WARN,
            f"lending {lending} MB, but one stage needs about {stage} MB, so this device would "
            "join but hold nothing",
            "Close other programs, or lend more with --memory-mb (if you have the room).",
        )
    return Check(
        "Memory",
        OK,
        f"lending {lending} MB ({settings.memory_source}): room for {settings.slots} "
        f"stage{'s' if settings.slots != 1 else ''}",
    )


def check_network(settings: PeerSettings) -> Check:
    if settings.host.startswith("127."):
        return Check(
            "Network address",
            WARN,
            "no network address found, so only this machine can reach the device",
            "Connect to Wi-Fi or Ethernet, or pass --host <address>.",
        )
    return Check("Network address", OK, f"other devices will reach this one at {settings.host}")


def check_ports(settings: PeerSettings) -> Check:
    busy = []
    wanted = [settings.port]
    if settings.shard_ports:
        wanted += list(range(settings.shard_ports[0], settings.shard_ports[1] + 1))
    for port in wanted:
        if not net.port_is_free(port, settings.bind):
            busy.append(port)
    if busy:
        return Check(
            "Ports",
            FAIL,
            "already in use: " + ", ".join(str(p) for p in busy),
            "Another program (or another copy of torrent-peer) has them. Stop it, or choose "
            "other ports with --port and --shard-port-base.",
        )
    span = (
        ""
        if not settings.shard_ports
        else f" and {settings.shard_ports[0]}-{settings.shard_ports[1]}"
    )
    return Check("Ports", OK, f"{settings.port}{span} are free")


def check_discovery(settings: PeerSettings) -> Check:
    if not settings.discover:
        return Check("Finding devices", OK, "broadcast discovery is off; use --join to connect")
    try:
        probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        probe.sendto(b"torrent-pool-check", ("255.255.255.255", DISCOVERY_PORT))
        probe.close()
    except OSError as exc:
        return Check(
            "Finding devices",
            WARN,
            f"could not broadcast on this network ({exc})",
            "Devices will not find each other automatically here. Give each device the "
            "address of one other with --join <address>:9000, or use Tailscale.",
        )
    return Check(
        "Finding devices",
        OK,
        f"broadcasting on the local network (UDP {DISCOVERY_PORT}); a device on a different "
        "network, or on Wi-Fi that isolates clients, needs --join",
    )


def check_firewall(settings: PeerSettings) -> Check:
    return Check(
        "Firewall",
        WARN if platform.system() == "Windows" else OK,
        "other devices must be able to reach this one",
        net.firewall_advice(settings.port, settings.shard_ports),
    )


def check_seeds(settings: PeerSettings) -> list[Check]:
    results = []
    for address in settings.join:
        try:
            with urllib.request.urlopen(f"http://{address}/health", timeout=3):  # noqa: S310
                results.append(Check(f"Reach {address}", OK, "answered"))
        except (urllib.error.URLError, OSError, TimeoutError) as exc:
            results.append(
                Check(
                    f"Reach {address}",
                    FAIL,
                    f"no answer ({getattr(exc, 'reason', exc)})",
                    "Is torrent-peer running there? Is it on the same network, and is its "
                    f"port open in its firewall? Try opening http://{address} in a browser.",
                )
            )
    return results


def run_checks(settings: PeerSettings) -> list[Check]:
    """Every check, in the order a person would think of them."""
    basics: list[Callable[[], Check]] = [check_python, check_packages, check_stubs]
    results = [check() for check in basics]
    # The rest need the packages the first ones look for.
    if all(c.status != FAIL for c in results):
        results += [
            check_hardware(settings),
            check_model(settings),
            check_memory(settings),
            check_network(settings),
            check_ports(settings),
            check_discovery(settings),
            check_firewall(settings),
            *check_seeds(settings),
        ]
    return results


SYMBOL = {OK: "OK  ", WARN: "WARN", FAIL: "FAIL"}


def report(checks: list[Check]) -> str:
    lines = []
    for c in checks:
        lines.append(f"[{SYMBOL[c.status]}] {c.name}: {c.detail}")
        if c.fix and c.status != OK or (c.fix and c.name == "Firewall"):
            lines.extend(f"        {row}" for row in c.fix.splitlines())
    failed = sum(c.status == FAIL for c in checks)
    warned = sum(c.status == WARN for c in checks)
    lines.append("")
    if failed:
        lines.append(
            f"{failed} problem{'s' if failed != 1 else ''} to fix before this device can join."
        )
    elif warned:
        lines.append("Ready, with the notes above.")
    else:
        lines.append("Ready to join.")
    return "\n".join(lines)
