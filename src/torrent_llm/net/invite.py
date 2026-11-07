"""Invite links: everything a device needs to join a private swarm, in one string.

    tllm://join/<base64url(JSON), no padding>

The JSON (``v`` = 1)::

    {"v": 1,
     "swarm": "<swarm id: 1-64 chars of [a-z0-9-]>",
     "name": "<human name of the swarm, <= 80 chars>",
     "psk": "<base64url of the 32-byte swarm secret>",
     "pool": "<sha256 hex of the pool config the swarm runs, 64 hex chars>",
     "boot": ["<address>", ...],          # 1-8 addresses (see records.py)
     "peers": ["<peer id>", ...],          # 0-8 peer ids the boot addresses belong to
     "exp": <unix seconds or 0 for never>}

Anyone holding the link can join, so it is a secret: it must never be logged.
``Invite.__repr__`` hides the secret for that reason. A link is checked fully on
parse (``ValueError`` with a human-readable reason) so a typo or a truncated paste
says what is wrong instead of failing later on a network timeout.
"""

from __future__ import annotations

import base64
import binascii
import json
import re
import secrets
import time
from dataclasses import dataclass

from torrent_llm.net.identity import is_peer_id
from torrent_llm.net.records import parse_address

SCHEME = "tllm://join/"
#: Longest link accepted, to keep a pasted blob from being parsed at all.
MAX_LINK_CHARS = 4096
MAX_BOOTSTRAP = 8
MAX_PEERS = 8
MAX_NAME_CHARS = 80
_SWARM_ID = re.compile(r"^[a-z0-9-]{1,64}$")
_SHA256_HEX = re.compile(r"^[0-9a-f]{64}$")
_B64URL = re.compile(r"^[A-Za-z0-9_-]*$")
_ID_ALPHABET = "abcdefghijklmnopqrstuvwxyz0123456789"


def _b64url_decode(text: str) -> bytes:
    """Strict base64url without padding: any other character is an error."""
    if not isinstance(text, str) or not _B64URL.fullmatch(text) or len(text) % 4 == 1:
        raise ValueError("damaged: not base64url")
    try:
        return base64.urlsafe_b64decode(text + "=" * (-len(text) % 4))
    except (binascii.Error, ValueError) as exc:
        raise ValueError("damaged: not base64url") from exc


def _b64url_encode(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).decode("ascii").rstrip("=")


def _check_fields(
    swarm_id: object,
    name: object,
    psk: object,
    pool_hash: object,
    bootstrap: object,
    peers: object,
    expires_at: object,
) -> None:
    """The one set of rules every invite obeys, whichever way it was made."""
    if not isinstance(swarm_id, str) or not _SWARM_ID.fullmatch(swarm_id):
        raise ValueError("damaged: bad swarm id")
    if not isinstance(name, str) or not 1 <= len(name) <= MAX_NAME_CHARS:
        raise ValueError(f"damaged: the swarm name must be 1-{MAX_NAME_CHARS} characters")
    if not isinstance(psk, bytes) or len(psk) != 32:
        # Never put the secret, or its length, in a message that may be logged.
        raise ValueError("damaged: bad swarm secret")
    if not isinstance(pool_hash, str) or not _SHA256_HEX.fullmatch(pool_hash):
        raise ValueError("damaged: bad pool hash")
    if not isinstance(bootstrap, tuple) or not 1 <= len(bootstrap) <= MAX_BOOTSTRAP:
        raise ValueError(f"no reachable address: an invite needs 1-{MAX_BOOTSTRAP} addresses")
    for address in bootstrap:
        if not isinstance(address, str):
            raise ValueError("no reachable address: an address is not text")
        try:
            parse_address(address)
        except ValueError as exc:
            raise ValueError(f"no reachable address: {exc}") from exc
    if not isinstance(peers, tuple) or len(peers) > MAX_PEERS:
        raise ValueError("damaged: too many peer ids")
    if any(not is_peer_id(p) for p in peers):
        raise ValueError("damaged: bad peer id")
    if type(expires_at) is not int or expires_at < 0:
        raise ValueError("damaged: bad expiry")


@dataclass(frozen=True)
class Invite:
    swarm_id: str
    name: str
    psk: bytes
    pool_hash: str
    bootstrap: tuple[str, ...]
    peers: tuple[str, ...] = ()
    expires_at: int = 0

    def __repr__(self) -> str:  # never print the secret
        return (
            f"Invite(swarm_id={self.swarm_id!r}, name={self.name!r}, psk=<hidden>, "
            f"bootstrap={list(self.bootstrap)!r})"
        )

    def validate(self) -> None:
        """``ValueError`` (with the phrases of :meth:`from_link`) unless every field is valid."""
        _check_fields(
            self.swarm_id,
            self.name,
            self.psk,
            self.pool_hash,
            self.bootstrap,
            self.peers,
            self.expires_at,
        )

    def to_link(self) -> str:
        """Encode as a ``tllm://join/...`` link (validates first)."""
        self.validate()
        payload = {
            "v": 1,
            "swarm": self.swarm_id,
            "name": self.name,
            "psk": _b64url_encode(self.psk),
            "pool": self.pool_hash,
            "boot": list(self.bootstrap),
            "peers": list(self.peers),
            "exp": self.expires_at,
        }
        blob = json.dumps(payload, separators=(",", ":"), ensure_ascii=True).encode("ascii")
        link = SCHEME + _b64url_encode(blob)
        if len(link) > MAX_LINK_CHARS:
            raise ValueError("too long: fewer bootstrap addresses would fit")
        return link

    @classmethod
    def from_link(cls, link: str, *, now: float | None = None) -> Invite:
        """Parse and validate a link. Surrounding whitespace and quotes are ignored.

        ``ValueError`` messages (tests match on these phrases):

        ===========================================  =========================
        not starting with ``tllm://join/``           "not an invite link"
        over :data:`MAX_LINK_CHARS`                   "too long"
        not base64url / not JSON / not an object      "damaged"
        ``v`` != 1                                    "newer version"
        bad swarm id / psk length / pool hash         "damaged"
        no bootstrap address, or one that does not
        parse, or more than 8                         "no reachable address"
        a peer id that is not one                     "damaged"
        ``exp`` in the past (and not 0)               "expired"
        ===========================================  =========================
        """
        if not isinstance(link, str):
            raise ValueError("not an invite link")
        text = link.strip()
        if len(text) >= 2 and text[0] == text[-1] and text[0] in "'\"":
            text = text[1:-1].strip()
        if len(text) > MAX_LINK_CHARS:
            raise ValueError("too long to be an invite link")
        if not text.startswith(SCHEME):
            raise ValueError("not an invite link (it should start with tllm://join/)")
        raw = _b64url_decode(text[len(SCHEME) :])
        try:
            data = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, ValueError, RecursionError) as exc:
            raise ValueError("damaged: not JSON") from exc
        if not isinstance(data, dict):
            raise ValueError("damaged: not an object")
        version = data.get("v")
        if type(version) is not int or version != 1:
            raise ValueError(
                "made by a newer version of Torrent-LLM, or damaged: update this device "
                "and ask for a new link"
            )
        boot, peers = data.get("boot"), data.get("peers", [])
        if not isinstance(boot, list):
            raise ValueError("no reachable address in the invite")
        if not isinstance(peers, list):
            raise ValueError("damaged: bad peer list")
        swarm_id = data.get("swarm")
        psk_text = data.get("psk")
        invite = cls(
            swarm_id=swarm_id,  # type: ignore[arg-type]
            name=data.get("name", swarm_id),  # type: ignore[arg-type]
            psk=_b64url_decode(psk_text) if isinstance(psk_text, str) else b"",
            pool_hash=data.get("pool"),  # type: ignore[arg-type]
            bootstrap=tuple(boot),
            peers=tuple(peers),
            expires_at=data.get("exp", 0),  # type: ignore[arg-type]
        )
        invite.validate()
        if invite.expired(now):
            raise ValueError("expired: ask for a new invite link")
        return invite

    def expired(self, now: float | None = None) -> bool:
        """True when the invite has an expiry and it has passed."""
        current = time.time() if now is None else now
        return self.expires_at != 0 and self.expires_at <= current


def new_swarm(
    name: str,
    pool_hash: str,
    bootstrap: list[str],
    *,
    peers: list[str] | None = None,
    ttl_s: int = 0,
    now: float | None = None,
) -> Invite:
    """A brand-new swarm: random 32-byte secret and a random 12-char id ``[a-z0-9]``.

    ``ttl_s`` 0 means the invite never expires. ``name`` is stripped and must be
    1-80 characters; bootstrap rules are those of :meth:`Invite.from_link`.
    """
    if type(ttl_s) is not int or ttl_s < 0:
        raise ValueError("damaged: ttl_s must be a non-negative integer")
    current = time.time() if now is None else now
    invite = Invite(
        swarm_id="".join(secrets.choice(_ID_ALPHABET) for _ in range(12)),
        name=name.strip() if isinstance(name, str) else name,
        psk=secrets.token_bytes(32),
        pool_hash=pool_hash,
        bootstrap=tuple(bootstrap),
        peers=tuple(peers or ()),
        expires_at=int(current) + ttl_s if ttl_s else 0,
    )
    invite.validate()
    return invite


__all__ = ["MAX_LINK_CHARS", "SCHEME", "Invite", "new_swarm"]
