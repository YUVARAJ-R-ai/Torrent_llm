"""``torrent-peer create | join LINK | public | bootstrap``: running a device in a swarm.

What each does, from the user's side:

* ``create`` makes an invite swarm, starts this device in it, and prints the link and
  the one command friends paste to join (Windows, macOS and Linux).
* ``join LINK`` joins the swarm in the link, from anywhere.
* ``public`` joins the public swarm (anyone may join; identities carry proof of work).
* ``bootstrap`` runs only the network part (relay + discovery) with no model, for a
  small always-on machine that helps others find and reach each other.

The local control panel (HTTP) listens on this machine only, guarded by an admin
token kept in the data directory, which the desktop app reads. ``--ui-lan`` also
shows the dashboard to the local network.
"""

from __future__ import annotations

import hashlib
import secrets
import signal
import sys
import threading
from dataclasses import dataclass
from pathlib import Path

from torrent_llm.net import paths
from torrent_llm.net.identity import PUBLIC_POW_BITS, Identity
from torrent_llm.net.invite import Invite
from torrent_llm.net.node import Node, NodeConfig
from torrent_llm.pool.config import PoolConfig
from torrent_llm.pool.swarm import SwarmServices, SwarmSpec, load_swarms, save_swarm

INSTALL_SH = "https://raw.githubusercontent.com/YUVARAJ-R-ai/Torrent_llm/main/scripts/install.sh"
INSTALL_PS1 = "https://raw.githubusercontent.com/YUVARAJ-R-ai/Torrent_llm/main/scripts/install.ps1"


def pool_hash(config: PoolConfig) -> str:
    """What an invite records about the pool settings, so a mismatch is caught at join."""
    return hashlib.sha256(config.signature.encode()).hexdigest()


def join_commands(link: str) -> dict[str, str]:
    """The one command a friend pastes, per operating system."""
    return {
        "linux_macos": f"curl -LsSf {INSTALL_SH} | sh -s -- '{link}'",
        "windows": (
            'powershell -ExecutionPolicy Bypass -c "& ([scriptblock]::Create((irm '
            f"{INSTALL_PS1}))) '{link}'\""
        ),
        "installed": f"torrent-peer join '{link}'",
    }


def admin_token(path: Path | None = None) -> str:
    """The local control panel's secret: made once, kept owner-readable only."""
    path = path or paths.admin_token_path()
    if path.exists():
        token = path.read_text(encoding="utf-8").strip()
        if len(token) >= 32:
            return token
    path.parent.mkdir(parents=True, exist_ok=True)
    token = secrets.token_hex(32)
    path.write_text(token, encoding="utf-8")
    try:
        path.chmod(0o600)
    except OSError:  # pragma: no cover - filesystems without modes
        pass
    return token


class SwarmError(RuntimeError):
    """Something the person must fix; the message says how."""


@dataclass
class SwarmDevice:
    """A device in a swarm: its node, its services, and (unless bootstrap) its pool."""

    node: Node
    services: SwarmServices
    spec: SwarmSpec
    peer: object | None = None  # RunningPeer, when the device runs the model

    def invite(self, config: PoolConfig) -> Invite | None:
        """A fresh invite for this swarm with this device's current addresses."""
        if self.spec.public or self.spec.psk is None:
            return None
        return Invite(
            swarm_id=self.spec.swarm_id,
            name=self.spec.name,
            psk=self.spec.psk,
            pool_hash=pool_hash(config),
            bootstrap=tuple(self.node.addresses()[:8]) or self.spec.bootstrap[:8],
            peers=(self.node.peer_id,),
        )

    def close(self) -> None:
        if self.peer is not None:
            self.peer.close(graceful=True)  # type: ignore[attr-defined]
        self.services.stop()
        self.node.close()


def resolve_spec(mode: str, invite: str | None, config: PoolConfig, args) -> SwarmSpec:
    """The swarm this run is for, from the mode and arguments."""
    swarms_dir = paths.swarms_dir()
    if mode == "join":
        if not invite:
            raise SwarmError("join needs the invite link: torrent-peer join 'tllm://join/...'")
        try:
            parsed = Invite.from_link(invite)
        except ValueError as exc:
            raise SwarmError(f"That invite link does not work: {exc}") from exc
        if parsed.pool_hash != pool_hash(config):
            raise SwarmError(
                "This invite is for a swarm running different settings (model, stages or "
                "codecs) from this device's pool file. Update the repository on both "
                "devices (git pull) so everyone uses the same configs/pool.yaml."
            )
        spec = SwarmSpec.from_invite(parsed)
        extra = tuple(args.bootstrap or ())
        if extra:
            spec = SwarmSpec(**{**spec.__dict__, "bootstrap": spec.bootstrap + extra})
        save_swarm(swarms_dir, spec)
        return spec
    if mode == "create":
        name = (args.swarm_name or "My swarm").strip()[:80] or "My swarm"
        # The invite's addresses are this device's own, known once its node is up
        # (SwarmDevice.invite); the stored spec keeps the secret across restarts.
        alphabet = "abcdefghijklmnopqrstuvwxyz0123456789"
        spec = SwarmSpec(
            swarm_id="".join(secrets.choice(alphabet) for _ in range(12)),
            name=name,
            psk=secrets.token_bytes(32),
            bootstrap=tuple(args.bootstrap or ()),
            pool_hash=pool_hash(config),
        )
        save_swarm(swarms_dir, spec)
        return spec
    if mode in ("public", "bootstrap"):
        if mode == "bootstrap" and invite:
            known = {s.swarm_id: s for s in load_swarms(swarms_dir)}
            try:
                parsed = Invite.from_link(invite)
            except ValueError as exc:
                raise SwarmError(f"That invite link does not work: {exc}") from exc
            return known.get(parsed.swarm_id) or SwarmSpec.from_invite(parsed)
        return SwarmSpec.public_swarm(tuple(args.bootstrap or ()))
    raise SwarmError(f"unknown mode {mode!r}")


def start_node(spec: SwarmSpec, *, listen_port: int, label: str, relay: bool) -> Node:
    print(
        "Preparing this device's identity"
        + (" (proof of work, a few seconds the first time)..." if spec.public else "..."),
        flush=True,
    )
    identity = Identity.load_or_create(
        paths.identity_path(), pow_bits=PUBLIC_POW_BITS if spec.public else 0
    )
    node = Node(
        NodeConfig(
            identity=identity,
            psk=spec.psk,
            swarm_id=spec.swarm_id,
            min_pow_bits=spec.min_pow_bits,
            listen_host="0.0.0.0",
            listen_port=listen_port,
            attrs={"label": label[:64], **({"relay": True} if relay else {})},
        )
    )
    try:
        node.start()
    except OSError as exc:
        raise SwarmError(
            f"Could not listen on port {listen_port}: {exc}. Another torrent-peer is probably "
            "running; stop it or pass --listen-port."
        ) from exc
    return node


def wait_for_ctrl_c() -> None:
    done = threading.Event()
    signal.signal(signal.SIGINT, lambda *_: done.set())
    signal.signal(signal.SIGTERM, lambda *_: done.set())
    # Short slices: on Windows an untimed wait cannot be interrupted by Ctrl+C.
    while not done.wait(0.5):
        pass


def print_invite(device: SwarmDevice, config: PoolConfig) -> None:
    invite = device.invite(config)
    if invite is None:
        return
    link = invite.to_link()
    commands = join_commands(link)
    print(
        "\n  Invite link (anyone with it can join; share it privately):\n\n"
        f"    {link}\n\n"
        "  Friends paste ONE of these into a terminal:\n\n"
        f"    Linux / macOS:  {commands['linux_macos']}\n\n"
        f"    Windows:        {commands['windows']}\n",
        flush=True,
    )


def main(args, settings) -> int:
    """Entry for the swarm modes; ``settings`` is ``PeerSettings`` from ``settings.py``."""
    try:
        spec = resolve_spec(args.mode, args.invite, settings.config, args)
        node = start_node(
            spec,
            listen_port=args.listen_port,
            label=settings.profile.label or settings.profile.name,
            relay=args.relay or args.mode == "bootstrap",
        )
    except SwarmError as exc:
        print(f"\n{exc}\n", file=sys.stderr)
        return 1
    services = SwarmServices(
        node, spec, use_upnp=not args.no_upnp, relay_always=args.relay or args.mode == "bootstrap"
    )
    services.start()
    device = SwarmDevice(node=node, services=services, spec=spec)

    if args.mode != "bootstrap":
        from torrent_llm.pool.peer import start_pool_device  # avoid an import cycle

        try:
            device.peer = start_pool_device(settings, node=node, device=device, ui_lan=args.ui_lan)
        except (OSError, RuntimeError) as exc:
            print(f"\nCould not start the device: {exc}\n", file=sys.stderr)
            device.close()
            return 1

    where = "the public swarm" if spec.public else f'the swarm "{spec.name}"'
    print(f"\n  This device ({node.peer_id[:8]}) is in {where}.", flush=True)
    print(f"  Reachable at: {', '.join(node.addresses()) or 'through relays only'}", flush=True)
    if args.mode == "create" or (args.mode == "join" and not spec.public):
        print_invite(device, settings.config)
    if args.mode == "bootstrap":
        print("  Running as a bootstrap node: relaying and helping devices find each other.")
    print("\n  Stop with Ctrl+C.\n", flush=True)
    wait_for_ctrl_c()
    print("Leaving the swarm...", flush=True)
    device.close()
    return 0
