"""Signed peer records and the addresses in them.

A :class:`PeerRecord` says "peer X can be reached at these addresses", signed by
X. Anyone may pass records along (gossip, DHT, invites, LAN broadcast), and anyone
can check them, so a peer cannot be impersonated or have its addresses replaced by
someone else. ``seq`` only ever goes up, so a newer record replaces an older one
and an old one replayed later is ignored.

Address strings:

==============================  =============================================
``tcp://192.168.1.20:9100``     direct TCP, IPv4
``tcp://[2001:db8::1]:9100``    direct TCP, IPv6
``tcp://example.org:9100``      direct TCP, a DNS name (bootstrap lists)
``relay://<relay peer id>``     through a relay peer (``relay.py``)
==============================  =============================================
"""

from __future__ import annotations

import base64
import ipaddress
import json
import re
import time
from dataclasses import dataclass, field

from torrent_llm.net.identity import Identity, is_peer_id, peer_id_of, verify_signature

#: Addresses kept per record; more is a sign of abuse.
MAX_ADDRESSES = 16
#: Records issued further than this in the future are refused (clock skew allowance).
MAX_CLOCK_SKEW_S = 600
_HOSTNAME = re.compile(
    r"^(?=.{1,253}$)([a-zA-Z0-9]([a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?\.)*"
    r"[a-zA-Z0-9]([a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?$"
)


@dataclass(frozen=True)
class TcpAddress:
    host: str
    port: int

    def __str__(self) -> str:
        return (
            f"tcp://[{self.host}]:{self.port}"
            if ":" in self.host
            else f"tcp://{self.host}:{self.port}"
        )

    @property
    def is_private(self) -> bool:
        """True for loopback, link-local and private-range IP literals (not DNS names)."""
        raise NotImplementedError


@dataclass(frozen=True)
class RelayAddress:
    relay: str  # the relay's peer id

    def __str__(self) -> str:
        return f"relay://{self.relay}"


Address = TcpAddress | RelayAddress


def parse_address(text: str) -> Address:
    """Parse an address string; ``ValueError`` names what is wrong.

    Examples::

        parse_address("tcp://1.2.3.4:9100")     -> TcpAddress("1.2.3.4", 9100)
        parse_address("tcp://[::1]:80")          -> TcpAddress("::1", 80)
        parse_address("tcp://host.example:9")    -> TcpAddress("host.example", 9)
        parse_address("relay://" + peer_id)      -> RelayAddress(peer_id)
        parse_address("tcp://1.2.3.4:0")         -> ValueError (port 1-65535)
        parse_address("tcp://1.2.3.4")           -> ValueError (no port)
        parse_address("udp://1.2.3.4:5")         -> ValueError (unknown scheme)
        parse_address("relay://not-an-id")       -> ValueError
        parse_address("tcp://bad host:5")        -> ValueError
    """
    raise NotImplementedError


@dataclass(frozen=True)
class PeerRecord:
    """Where a peer can be reached, signed by that peer."""

    peer_id: str
    public_key: bytes
    addresses: tuple[str, ...]
    seq: int
    issued_at: float
    #: Small, signed facts about the peer (e.g. ``{"public": True, "relay": True}``).
    #: Values must be JSON scalars; at most 16 keys.
    attrs: dict = field(default_factory=dict)
    signature: bytes = b""

    def signed_bytes(self) -> bytes:
        """Canonical bytes covered by the signature: compact, sorted-key JSON of every
        field except ``signature``, with ``public_key`` as base64 and ``addresses`` as a list."""
        raise NotImplementedError

    def verify(self, *, now: float | None = None) -> bool:
        """True when the signature checks, ``peer_id`` matches ``public_key``, every
        address parses, there are at most :data:`MAX_ADDRESSES`, ``seq >= 0``, and
        ``issued_at`` is not more than :data:`MAX_CLOCK_SKEW_S` in the future."""
        raise NotImplementedError

    def parsed_addresses(self) -> list[Address]:
        """The addresses that parse (all of them, for a verified record)."""
        raise NotImplementedError

    def to_dict(self) -> dict:
        """JSON-safe dict (bytes as base64)."""
        raise NotImplementedError

    @classmethod
    def from_dict(cls, raw: object) -> PeerRecord:
        """Inverse of :meth:`to_dict`. ``ValueError`` on wrong shape or types; does not verify."""
        raise NotImplementedError

    def newer_than(self, other: PeerRecord | None) -> bool:
        """True when this record should replace ``other`` (same peer, higher ``seq``)."""
        raise NotImplementedError


def make_record(
    identity: Identity,
    addresses: list[str],
    *,
    seq: int,
    attrs: dict | None = None,
    now: float | None = None,
) -> PeerRecord:
    """A record for ``identity``, signed. Addresses are validated (``ValueError``)."""
    raise NotImplementedError


__all__ = [
    "MAX_ADDRESSES",
    "Address",
    "PeerRecord",
    "RelayAddress",
    "TcpAddress",
    "make_record",
    "parse_address",
]
