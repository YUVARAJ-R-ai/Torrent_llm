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
import json
import os
import re
import time
from dataclasses import dataclass

from torrent_llm.net.identity import is_peer_id
from torrent_llm.net.records import parse_address

SCHEME = "tllm://join/"
#: Longest link accepted, to keep a pasted blob from being parsed at all.
MAX_LINK_CHARS = 4096
_SWARM_ID = re.compile(r"^[a-z0-9-]{1,64}$")
_SHA256_HEX = re.compile(r"^[0-9a-f]{64}$")


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

    def to_link(self) -> str:
        """Encode as a ``tllm://join/...`` link (validates first)."""
        raise NotImplementedError

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
        raise NotImplementedError

    def expired(self, now: float | None = None) -> bool:
        raise NotImplementedError


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
    raise NotImplementedError


__all__ = ["MAX_LINK_CHARS", "SCHEME", "Invite", "new_swarm"]
