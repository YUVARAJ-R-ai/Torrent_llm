"""Being part of a swarm: which one, how to reach it, and keeping reachable.

A device is in exactly one swarm at a time:

* an **invite swarm**, made by ``torrent-peer create`` and joined with the link it
  prints (``tllm://join/...``). The link carries the swarm secret, so only people
  given it can connect. Nothing is published anywhere.
* the **public swarm**, joined with ``torrent-peer public``. Anyone may join; every
  identity carries proof of work, and spot-checks catch devices that lie.

``SwarmServices`` runs everything that keeps a device connected, on top of its
:class:`~torrent_llm.net.node.Node`:

1. connect to the bootstrap addresses (invite link, ``--bootstrap``, ``--join``);
2. ask the router for a port mapping (UPnP), best effort;
3. ask connected peers to dial back (``autonat``): reachable devices offer to relay
   for others; unreachable ones reserve a slot on up to two relays, so they can
   still be reached from anywhere;
4. announce the swarm in the DHT and look for other members there, so the swarm
   holds together when the device that made the invite is gone.

It repeats 3-4 every ``refresh_s`` and whenever the device is alone.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import tempfile
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from torrent_llm.net import upnp
from torrent_llm.net.autonat import PRIVATE, PUBLIC, AutoNat
from torrent_llm.net.dht import Dht, topic_key
from torrent_llm.net.identity import PUBLIC_POW_BITS
from torrent_llm.net.invite import Invite
from torrent_llm.net.node import Node
from torrent_llm.net.records import PeerRecord, TcpAddress
from torrent_llm.net.relay import Relay
from torrent_llm.pool.discovery import DISCOVERY_PORT, Announcement, Discovery

logger = logging.getLogger(__name__)

PUBLIC_SWARM_ID = "public"
#: Bootstrap peers for the public swarm. Empty until someone runs well-known
#: bootstrap nodes; until then pass --bootstrap (any reachable member will do).
DEFAULT_PUBLIC_BOOTSTRAP: tuple[str, ...] = ()
_SAFE_NAME = re.compile(r"[^a-z0-9-]")


@dataclass(frozen=True)
class SwarmSpec:
    """Which swarm a device is in. ``psk`` is ``None`` for the public swarm."""

    swarm_id: str
    name: str
    psk: bytes | None
    bootstrap: tuple[str, ...] = ()
    peers: tuple[str, ...] = ()
    pool_hash: str = ""
    min_pow_bits: int = 0

    @property
    def public(self) -> bool:
        return self.psk is None

    @classmethod
    def from_invite(cls, invite: Invite) -> SwarmSpec:
        return cls(
            swarm_id=invite.swarm_id,
            name=invite.name,
            psk=invite.psk,
            bootstrap=tuple(invite.bootstrap),
            peers=tuple(invite.peers),
            pool_hash=invite.pool_hash,
        )

    @classmethod
    def public_swarm(cls, bootstrap: tuple[str, ...] = ()) -> SwarmSpec:
        return cls(
            swarm_id=PUBLIC_SWARM_ID,
            name="Public swarm",
            psk=None,
            bootstrap=tuple(bootstrap) or DEFAULT_PUBLIC_BOOTSTRAP,
            min_pow_bits=PUBLIC_POW_BITS,
        )

    def to_json(self) -> str:
        data = asdict(self)
        data["psk"] = self.psk.hex() if self.psk is not None else None
        return json.dumps(data, indent=1)

    @classmethod
    def from_json(cls, text: str) -> SwarmSpec:
        data = json.loads(text)
        data["psk"] = bytes.fromhex(data["psk"]) if data.get("psk") else None
        data["bootstrap"] = tuple(data.get("bootstrap", ()))
        data["peers"] = tuple(data.get("peers", ()))
        return cls(**data)


def swarm_tag(spec: SwarmSpec) -> str:
    """What LAN announcements carry to say which swarm they belong to. Derived from
    the secret, so it reveals neither the swarm id nor the secret, and two swarms on
    one network never mix."""
    secret = spec.psk or b""
    return "swarm-" + hashlib.sha256(b"tllm-lan" + spec.swarm_id.encode() + secret).hexdigest()[:16]


def save_swarm(directory: Path, spec: SwarmSpec) -> Path:
    """Remember a joined swarm (it holds the secret: owner-only permissions)."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{_SAFE_NAME.sub('-', spec.swarm_id.lower())}.json"
    fd, tmp = tempfile.mkstemp(dir=directory, prefix=".swarm-")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(spec.to_json())
        try:
            os.chmod(tmp, 0o600)
        except OSError:  # pragma: no cover - filesystems without modes
            pass
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
    return path


def load_swarms(directory: Path) -> list[SwarmSpec]:
    """Every remembered swarm, skipping damaged files."""
    out = []
    if not directory.is_dir():
        return out
    for path in sorted(directory.glob("*.json")):
        try:
            out.append(SwarmSpec.from_json(path.read_text(encoding="utf-8")))
        except (OSError, ValueError, TypeError, KeyError) as exc:
            logger.warning("ignoring damaged swarm file %s: %s", path.name, exc)
    return out


@dataclass
class Reachability:
    status: str = "unknown"  # public | private | unknown
    mapped: str = ""  # UPnP address, if any
    relays: list[str] = field(default_factory=list)
    serving_relay: bool = False
    checked_at: float = 0.0


class SwarmServices:
    """Keeps one node connected to its swarm. ``start()`` returns at once; the work
    happens on a background thread; ``stop()`` undoes port mappings and stops it."""

    def __init__(
        self,
        node: Node,
        spec: SwarmSpec,
        *,
        use_upnp: bool = True,
        refresh_s: float = 120.0,
        relay_always: bool = False,
        lan_discovery: bool = True,
        discovery_port: int = DISCOVERY_PORT,
        inbound: bool = True,
    ) -> None:
        #: ``False``: never publish a direct address, even one a check would confirm;
        #: be reached through relays only (``--no-inbound``).
        self.inbound = inbound
        self.node = node
        self.lan_discovery = lan_discovery
        self.discovery_port = discovery_port
        self.discovery: Discovery | None = None
        self.spec = spec
        self.use_upnp = use_upnp
        self.refresh_s = refresh_s
        self.relay = Relay(node, serve=relay_always)
        self.autonat = AutoNat(node)
        self.dht = Dht(node)
        self.reachability = Reachability(serving_relay=relay_always)
        self._mapping: upnp.Mapping | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._kick = threading.Event()

    # --- lifecycle -----------------------------------------------------------

    def start(self) -> None:
        self.relay.attach()
        self.autonat.attach()
        self.dht.attach()
        if self.lan_discovery:
            self._start_lan()
        self._thread = threading.Thread(target=self._run, name="swarm-services", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._kick.set()
        if self.discovery is not None:
            self.discovery.stop()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
        self.relay.detach()
        if self._mapping is not None:
            mapping = self._mapping
            try:
                self.node.run(
                    upnp.delete_port_mapping(mapping.gateway, external_port=mapping.external_port),
                    timeout=5.0,
                )
            except Exception as exc:  # noqa: BLE001 - best effort on the way out
                logger.debug("removing the port mapping failed: %s", exc)

    # --- the local network ----------------------------------------------------
    #
    # On a phone hotspot or one Wi-Fi, members find each other by broadcast as soon as
    # one of them has joined with the invite: no address typing at all. Announcements
    # carry the signed peer record; a listener checks the signature before dialling.

    def _announcement(self) -> Announcement:
        record = self.node.record()
        lan = next(
            (a for a in record.parsed_addresses() if isinstance(a, TcpAddress) and a.is_private),
            None,
        )
        return Announcement(
            name=self.node.peer_id,
            host=lan.host if lan else "",
            port=self.node.listen_port,
            pool=swarm_tag(self.spec),
            record=record.to_dict(),
        )

    def _start_lan(self) -> None:
        try:
            self.discovery = Discovery(self._announcement(), self._heard, port=self.discovery_port)
            self.discovery.start()
        except OSError as exc:
            logger.info("LAN discovery unavailable here: %s", exc)
            self.discovery = None

    def _heard(self, found: Announcement) -> None:
        if self.discovery is not None:
            self.discovery.me = self._announcement()  # keep our own up to date
        if found.record is None or found.name == self.node.peer_id:
            return
        try:
            record = PeerRecord.from_dict(found.record)
        except ValueError:
            return
        if record.peer_id != found.name or not self.node.add_record(record):
            return
        if not self.node.is_connected(record.peer_id):
            try:
                self.node.call(record.peer_id, "node.ping", timeout=5.0)
                logger.info("found %s on this network", record.peer_id[:8])
            except Exception as exc:  # noqa: BLE001 - it may have just left
                logger.debug("LAN peer %s unreachable: %s", record.peer_id[:8], exc)

    def refresh_now(self) -> None:
        """Re-check reachability and look for peers without waiting for the timer."""
        self._kick.set()

    # --- the work ------------------------------------------------------------

    def _run(self) -> None:
        self._connect_bootstrap()
        if self.use_upnp:
            self._map_port()
        while not self._stop.is_set():
            try:
                self._round()
            except Exception:  # noqa: BLE001 - one bad round must not end the device
                logger.exception("swarm maintenance failed")
            alone = not self.node.connected_peers()
            self._kick.wait(10.0 if alone else self.refresh_s)
            self._kick.clear()
            if alone and not self._stop.is_set():
                self._connect_bootstrap()

    def _connect_bootstrap(self) -> None:
        for address in self.spec.bootstrap:
            if self._stop.is_set():
                return
            try:
                peer = self.node.connect(address, timeout=5.0)
                logger.info("connected to bootstrap %s (%s)", address, peer[:8])
            except Exception as exc:  # noqa: BLE001 - a bootstrap may be down
                logger.info("bootstrap %s unreachable: %s", address, exc)

    def _map_port(self) -> None:
        try:
            mapping = self.node.run(upnp.map_port(self.node.listen_port), timeout=10.0)
        except Exception as exc:  # noqa: BLE001 - UPnP is best effort
            logger.debug("UPnP failed: %s", exc)
            mapping = None
        if mapping is not None:
            self._mapping = mapping
            self.reachability.mapped = mapping.address
            logger.info("router forwards %s to this device", mapping.address)

    def _round(self) -> None:
        peers = self.node.connected_peers()
        if peers:
            if self.inbound:
                extra = (self.reachability.mapped,) if self.reachability.mapped else ()
                self.autonat.check(peers[:3], extra)
                self.reachability.status = self.autonat.status
            else:
                self.reachability.status = PRIVATE
            self.reachability.checked_at = time.time()
            if self.reachability.status == PUBLIC:
                if not self.relay.serving:
                    self.relay.set_serving(True)
                    logger.info("this device is reachable and now relays for others")
                self.reachability.serving_relay = True
            else:
                candidates = [
                    r.peer_id
                    for r in self.node.records()
                    if r.attrs.get("relay") and r.peer_id != self.node.peer_id
                ]
                self.relay.set_candidates(candidates)
                if candidates:
                    self.reachability.relays = self.relay.reserve_on(candidates)
        key = topic_key(self.spec.swarm_id)
        try:
            self.dht.bootstrap()
            self.dht.provide(key)
            for record in self.dht.find_providers(key, limit=16):
                if record.peer_id != self.node.peer_id and not self.node.is_connected(
                    record.peer_id
                ):
                    self.node.add_record(record)
                    try:
                        self.node.call(record.peer_id, "node.ping", timeout=5.0)
                    except Exception as exc:  # noqa: BLE001 - it may be gone already
                        logger.debug("provider %s unreachable: %s", record.peer_id[:8], exc)
        except Exception as exc:  # noqa: BLE001 - the DHT is a helper, not a requirement
            logger.debug("DHT round failed: %s", exc)
