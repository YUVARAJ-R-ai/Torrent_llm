"""Is this device reachable from outside? Ask peers to try connecting back.

A device cannot tell from the inside whether a router forwards its port. So it asks
peers that already know it to open a *fresh* connection to the addresses it thinks
might work, and to report which succeeded. A confirmed address goes into the
device's record; with none, the device is ``private`` and relies on relays.

``autonat.dialback``: ``{"addrs": ["tcp://ip:port", ...]} -> {"reachable": [...]}``.
The helper dials back **only addresses whose IP equals the caller's own observed
IP** (otherwise a stranger could make it connect anywhere: an amplification and
port-scanning vector), at most 4 of them, each with a short timeout, using a new
connection that runs the full handshake with ``expect_peer`` = the caller and is
closed straight after (never adopted into the node).

Candidate addresses: each IP peers report observing (``node.observed_addresses``)
combined with this node's listen port, plus any address given explicitly (a UPnP
mapping). Private-range candidates are skipped: they prove nothing about the internet.
"""

from __future__ import annotations

from torrent_llm.net.node import Node

DIALBACK_TIMEOUT_S = 3.0
MAX_DIALBACK_ADDRS = 4

PUBLIC, PRIVATE, UNKNOWN = "public", "private", "unknown"


class AutoNat:
    """Reachability for one node.

    ``attach()`` registers ``autonat.dialback`` (when ``serve``).
    ``check(helpers, extra=())`` asks each helper peer (up to 3) to dial back the
    candidates; returns the confirmed addresses, updates ``status`` (``public`` if any
    address was confirmed by at least one helper, ``private`` if every helper answered
    and none succeeded, ``unknown`` if no helper answered) and, when public, sets them via
    ``node.set_advertised`` (keeping any extra advertised addresses) and the record attr
    ``"public": True``.
    """

    def __init__(self, node: Node, *, serve: bool = True) -> None:
        raise NotImplementedError

    def attach(self) -> None:
        raise NotImplementedError

    @property
    def status(self) -> str:
        raise NotImplementedError

    @property
    def reachable(self) -> list[str]:
        raise NotImplementedError

    def candidates(self, extra: tuple[str, ...] = ()) -> list[str]:
        raise NotImplementedError

    async def acheck(self, helpers: list[str], extra: tuple[str, ...] = ()) -> list[str]:
        raise NotImplementedError

    def check(self, helpers: list[str], extra: tuple[str, ...] = ()) -> list[str]:
        raise NotImplementedError
