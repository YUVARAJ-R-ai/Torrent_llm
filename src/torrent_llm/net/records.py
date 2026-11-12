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
import binascii
import dataclasses
import ipaddress
import json
import math
import re
import time
from dataclasses import dataclass, field

from torrent_llm.net.identity import (
    PUBLIC_KEY_BYTES,
    SIGNATURE_BYTES,
    Identity,
    is_peer_id,
    peer_id_of,
    verify_signature,
)

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
        try:
            ip = ipaddress.ip_address(self.host)
        except ValueError:
            return False
        return ip.is_private or ip.is_loopback or ip.is_link_local


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
    if not isinstance(text, str):
        raise ValueError(f"address must be str, got {type(text).__name__}")

    if text.startswith("relay://"):
        relay = text[len("relay://") :]
        if not is_peer_id(relay):
            raise ValueError(f"not a peer id: {relay!r}")
        return RelayAddress(relay=relay)

    if not text.startswith("tcp://"):
        raise ValueError(f"unknown address scheme in {text!r}")

    rest = text[len("tcp://") :]
    if rest.startswith("["):
        closing = rest.find("]")
        if closing == -1:
            raise ValueError(f"unclosed IPv6 bracket in {text!r}")
        host = rest[1:closing]
        after = rest[closing + 1 :]
        if not after.startswith(":"):
            raise ValueError(f"no port in {text!r}")
        port_str = after[1:]
        try:
            ipaddress.IPv6Address(host)
        except ValueError as exc:
            raise ValueError(f"invalid IPv6 host: {host!r}") from exc
    else:
        if ":" not in rest:
            raise ValueError(f"no port in {text!r}")
        parts = rest.split(":")
        if len(parts) != 2:
            raise ValueError(f"invalid address or unbracketed IPv6: {rest!r}")
        host, port_str = parts
        if not host:
            raise ValueError(f"empty host in {text!r}")
        if host.isascii() and host.replace(".", "").isdigit():
            try:
                ipaddress.IPv4Address(host)
            except ValueError as exc:
                raise ValueError(f"invalid IPv4 host: {host!r}") from exc
        else:
            if not host.isascii() or not _HOSTNAME.fullmatch(host):
                raise ValueError(f"invalid host: {host!r}")

    if not (port_str.isascii() and port_str.isdigit()):
        raise ValueError(f"port must be 1-65535, got {port_str!r}")
    if len(port_str) > 5 or (len(port_str) > 1 and port_str[0] == "0"):
        raise ValueError(f"port must be 1-65535, got {port_str!r}")
    port = int(port_str)
    if not (1 <= port <= 65535):
        raise ValueError(f"port must be 1-65535, got {port}")

    return TcpAddress(host=host, port=port)


def _verify_attrs(attrs: object) -> bool:
    """True when attrs is a dict of at most 16 short str keys with small JSON scalar values."""
    if not isinstance(attrs, dict) or len(attrs) > 16:
        return False
    for k, v in attrs.items():
        if not isinstance(k, str) or len(k) > 32:
            return False
        if v is None:
            continue
        if isinstance(v, bool):
            continue
        if isinstance(v, (int, float)):
            if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                return False
            continue
        if isinstance(v, str):
            if len(v) > 256:
                return False
            continue
        return False
    return True


@dataclass(frozen=True)
class PeerRecord:
    """Where a peer can be reached, signed by that peer."""

    peer_id: str
    public_key: bytes
    addresses: tuple[str, ...]
    seq: int
    #: Whole unix seconds. An integer on purpose: a float would serialise as
    #: "1700000000.0" here and "1700000000" after some other JSON round trip, and the
    #: signature covers the exact text.
    issued_at: int
    #: Small, signed facts about the peer (e.g. ``{"public": True, "relay": True}``).
    #: Values must be JSON scalars; at most 16 keys.
    attrs: dict = field(default_factory=dict)
    signature: bytes = b""

    def signed_bytes(self) -> bytes:
        """Canonical bytes covered by the signature: compact, sorted-key JSON of every
        field except ``signature``, with ``public_key`` as base64 and ``addresses`` as a list."""
        data = {
            "peer_id": self.peer_id,
            "public_key": base64.b64encode(self.public_key).decode("ascii"),
            "addresses": list(self.addresses),
            "seq": self.seq,
            "issued_at": self.issued_at,
            "attrs": self.attrs,
        }
        return json.dumps(
            data,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode()

    def verify(self, *, now: float | None = None) -> bool:
        """True when the signature checks, ``peer_id`` matches ``public_key``, every
        address parses, there are at most :data:`MAX_ADDRESSES`, ``seq >= 0``, and
        ``issued_at`` is not more than :data:`MAX_CLOCK_SKEW_S` in the future."""
        try:
            if not _verify_attrs(self.attrs):
                return False
            if type(self.seq) is not int or self.seq < 0:
                return False
            if type(self.issued_at) is not int or not 0 <= self.issued_at < 2**63:
                return False
            current = time.time() if now is None else now
            if not math.isfinite(current):
                return False
            if self.issued_at > current + MAX_CLOCK_SKEW_S:
                return False
            if not isinstance(self.addresses, (tuple, list)):
                return False
            if len(self.addresses) > MAX_ADDRESSES:
                return False
            for addr in self.addresses:
                if not isinstance(addr, str):
                    return False
                parse_address(addr)
            if not isinstance(self.public_key, (bytes, bytearray)):
                return False
            if len(self.public_key) != PUBLIC_KEY_BYTES:
                return False
            if not isinstance(self.peer_id, str) or self.peer_id != peer_id_of(
                bytes(self.public_key)
            ):
                return False
            if not isinstance(self.signature, (bytes, bytearray)):
                return False
            if len(self.signature) != SIGNATURE_BYTES:
                return False
            return verify_signature(
                bytes(self.public_key),
                bytes(self.signature),
                self.signed_bytes(),
            )
        except (ValueError, TypeError, OverflowError):
            return False

    def parsed_addresses(self) -> list[Address]:
        """The addresses that parse (all of them, for a verified record)."""
        if not isinstance(self.addresses, (tuple, list)):
            return []
        result: list[Address] = []
        for addr in self.addresses:
            if not isinstance(addr, str):
                continue
            try:
                result.append(parse_address(addr))
            except ValueError:
                continue
        return result

    def to_dict(self) -> dict:
        """JSON-safe dict (bytes as base64)."""
        return {
            "peer_id": self.peer_id,
            "public_key": base64.b64encode(self.public_key).decode("ascii"),
            "addresses": list(self.addresses),
            "seq": self.seq,
            "issued_at": self.issued_at,
            "attrs": dict(self.attrs),
            "signature": base64.b64encode(self.signature).decode("ascii"),
        }

    @classmethod
    def from_dict(cls, raw: object) -> PeerRecord:
        """Inverse of :meth:`to_dict`. ``ValueError`` on wrong shape or types; does not verify."""
        if not isinstance(raw, dict):
            raise ValueError(f"expected dict, got {type(raw).__name__}")
        required = {
            "peer_id",
            "public_key",
            "addresses",
            "seq",
            "issued_at",
            "attrs",
            "signature",
        }
        missing = required - raw.keys()
        if missing:
            raise ValueError(f"missing keys in peer record dict: {sorted(missing)}")

        peer_id = raw["peer_id"]
        if not isinstance(peer_id, str):
            raise ValueError(f"peer_id must be str, got {type(peer_id).__name__}")

        pub_b64 = raw["public_key"]
        if not isinstance(pub_b64, str):
            raise ValueError(f"public_key must be str, got {type(pub_b64).__name__}")
        try:
            public_key = base64.b64decode(pub_b64, validate=True)
        except binascii.Error as exc:
            raise ValueError(f"invalid base64 in public_key: {exc}") from exc

        addresses_raw = raw["addresses"]
        if not isinstance(addresses_raw, list):
            raise ValueError(f"addresses must be list, got {type(addresses_raw).__name__}")
        for item in addresses_raw:
            if not isinstance(item, str):
                raise ValueError(f"addresses element must be str, got {type(item).__name__}")

        seq = raw["seq"]
        if type(seq) is not int:
            raise ValueError(f"seq must be int, got {type(seq).__name__}")

        issued_at = raw["issued_at"]
        if type(issued_at) is not int or not 0 <= issued_at < 2**63:
            raise ValueError("issued_at must be whole unix seconds (an integer)")

        attrs = raw["attrs"]
        if not isinstance(attrs, dict):
            raise ValueError(f"attrs must be dict, got {type(attrs).__name__}")

        sig_b64 = raw["signature"]
        if not isinstance(sig_b64, str):
            raise ValueError(f"signature must be str, got {type(sig_b64).__name__}")
        try:
            signature = base64.b64decode(sig_b64, validate=True)
        except binascii.Error as exc:
            raise ValueError(f"invalid base64 in signature: {exc}") from exc

        return cls(
            peer_id=peer_id,
            public_key=public_key,
            addresses=tuple(addresses_raw),
            seq=seq,
            issued_at=issued_at,
            attrs=dict(attrs),
            signature=signature,
        )

    def newer_than(self, other: PeerRecord | None) -> bool:
        """True when this record should replace ``other`` (same peer, higher ``seq``)."""
        if other is None:
            return True
        if not isinstance(other, PeerRecord):
            return False
        if self.peer_id != other.peer_id:
            return False
        return self.seq > other.seq


def make_record(
    identity: Identity,
    addresses: list[str],
    *,
    seq: int,
    attrs: dict | None = None,
    now: float | None = None,
) -> PeerRecord:
    """A record for ``identity``, signed. Addresses are validated (``ValueError``)."""
    if type(seq) is not int or seq < 0:
        raise ValueError(f"seq must be non-negative int, got {seq!r}")

    deduped: list[str] = []
    seen: set[str] = set()
    for addr in addresses:
        parse_address(addr)
        if addr not in seen:
            seen.add(addr)
            deduped.append(addr)

    if len(deduped) > MAX_ADDRESSES:
        raise ValueError(f"too many addresses: {len(deduped)} > {MAX_ADDRESSES}")

    record_attrs = dict(attrs) if attrs is not None else {}
    if not _verify_attrs(record_attrs):
        raise ValueError(
            "invalid attrs: must be dict with <=16 short string keys and small JSON scalar values"
        )

    clock = time.time() if now is None else now
    if not math.isfinite(clock):
        raise ValueError("now must be a finite time")
    issued = int(clock)
    unsigned = PeerRecord(
        peer_id=identity.peer_id,
        public_key=identity.public_key,
        addresses=tuple(deduped),
        seq=seq,
        issued_at=issued,
        attrs=record_attrs,
    )
    sig = identity.sign(unsigned.signed_bytes())
    return dataclasses.replace(unsigned, signature=sig)


__all__ = [
    "MAX_ADDRESSES",
    "Address",
    "PeerRecord",
    "RelayAddress",
    "TcpAddress",
    "make_record",
    "parse_address",
]
