"""Finding other devices on the same network without being told where they are.

Every device announces itself with a small UDP broadcast every couple of seconds
and listens for everyone else's. Hearing an announcement means "there is a device
at this address"; from there the normal gossip takes over and the two devices
swap what they know. Nothing here decides who is in the pool: it only introduces
devices, so a wrong or missing announcement costs a delay, not correctness.

Broadcast works on one subnet: the same Wi-Fi, the same router, a phone hotspot.
It does not cross networks, and some campus or guest Wi-Fi blocks it entirely
("client isolation"). In those cases a device is told one address with ``--join``,
or the devices are joined over Tailscale, and everything else works the same.

Only the standard library is used, and the same code runs on Windows, Linux and
macOS.
"""

from __future__ import annotations

import json
import logging
import socket
import threading
from collections.abc import Callable, Sequence
from dataclasses import dataclass

logger = logging.getLogger(__name__)

DISCOVERY_PORT = 9099
#: Marks a datagram as ours, so unrelated traffic on the port is ignored.
MAGIC = "torrent-pool/1"


@dataclass(frozen=True)
class Announcement:
    """What a device says about itself in a broadcast."""

    name: str
    host: str
    port: int
    #: Identifies the pool (model and shape), so two groups on one network do not mix.
    pool: str

    @property
    def address(self) -> str:
        return f"{self.host}:{self.port}"

    def encode(self) -> bytes:
        return json.dumps(
            {
                "magic": MAGIC,
                "name": self.name,
                "host": self.host,
                "port": self.port,
                "pool": self.pool,
            }
        ).encode()

    @classmethod
    def decode(cls, data: bytes) -> Announcement | None:
        """The announcement in ``data``, or ``None`` if it is not one of ours."""
        try:
            raw = json.loads(data.decode())
            if raw.get("magic") != MAGIC:
                return None
            return cls(
                name=str(raw["name"]),
                host=str(raw["host"]),
                port=int(raw["port"]),
                pool=str(raw["pool"]),
            )
        except (ValueError, KeyError, TypeError, UnicodeDecodeError, AttributeError):
            return None


def _open_listener(port: int) -> socket.socket:
    """A UDP socket that several programs on one machine may listen on at once.

    Two devices on the same machine (a lab, or a test) each need to hear the other's
    announcements, so the port has to be shareable. ``SO_REUSEADDR`` does that on
    Windows and macOS, and ``SO_REUSEPORT`` is added where it exists (Linux, macOS).
    """
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    if hasattr(socket, "SO_REUSEPORT"):
        try:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
        except OSError:
            pass
    sock.bind(("", port))
    sock.settimeout(0.5)
    return sock


class Discovery:
    """Announces this device and reports the others it hears.

    Args:
        me: This device's announcement. Others see it; this device ignores it.
        on_found: Called with each announcement from another device of the same
            pool. May be called repeatedly for the same device, once per broadcast.
        port: The UDP port everyone uses.
        targets: Where announcements are sent. The default is the whole local
            network; tests point it at loopback.
        interval_s: Seconds between announcements.
    """

    def __init__(
        self,
        me: Announcement,
        on_found: Callable[[Announcement], None],
        *,
        port: int = DISCOVERY_PORT,
        targets: Sequence[tuple[str, int]] | None = None,
        interval_s: float = 2.0,
    ) -> None:
        self.me = me
        self._on_found = on_found
        self.port = port
        self.targets = list(targets) if targets is not None else [("255.255.255.255", port)]
        self.interval_s = interval_s
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []
        self._listener: socket.socket | None = None
        #: Set when announcing is impossible here (no network, blocked), so the
        #: device can say so instead of silently never being found.
        self.send_error: str | None = None
        self.heard = 0

    def start(self) -> None:
        self._listener = _open_listener(self.port)
        for target, name in (
            (self._listen, "discovery-listen"),
            (self._announce, "discovery-announce"),
        ):
            thread = threading.Thread(target=target, name=name, daemon=True)
            thread.start()
            self._threads.append(thread)

    def stop(self) -> None:
        self._stop.set()
        for thread in self._threads:
            thread.join(timeout=2.0)
        if self._listener is not None:
            self._listener.close()

    def _announce(self) -> None:
        sender = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sender.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        try:
            while not self._stop.is_set():
                payload = self.me.encode()
                failures = []
                for target in self.targets:
                    try:
                        sender.sendto(payload, target)
                    except OSError as exc:
                        failures.append(str(exc))
                self.send_error = (
                    "; ".join(failures) if len(failures) == len(self.targets) else None
                )
                self._stop.wait(self.interval_s)
        finally:
            sender.close()

    def _listen(self) -> None:
        assert self._listener is not None
        while not self._stop.is_set():
            try:
                data, _addr = self._listener.recvfrom(2048)
            except TimeoutError:
                continue
            except OSError:
                return
            found = Announcement.decode(data)
            if found is None or found.pool != self.me.pool:
                continue
            if (found.name, found.host, found.port) == (self.me.name, self.me.host, self.me.port):
                continue
            self.heard += 1
            try:
                self._on_found(found)
            except Exception:  # noqa: BLE001 - a bad callback must not end discovery
                logger.exception("handling an announcement failed")
