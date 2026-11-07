"""The control panel's swarm routes: what the desktop app (and the dashboard) use.

Mounted next to the pool routes in ``api.py`` when the device runs in a swarm. The
panel listens on 127.0.0.1 by default, and every route that reveals a secret (the
invite link) or changes something requires the admin token in the ``x-admin-token``
header (the desktop app reads it from the data directory).

====================  ======  =====  =========================================================
route                 method  admin  answer
====================  ======  =====  =========================================================
``/node``             GET     no     :class:`NodeView` as JSON
``/swarm/invite``     GET     yes    ``{"link": str, "commands": {linux_macos, windows,
                                     installed}}``; 404 ``{"detail": ...}`` in the public swarm
``/peers``            GET     no     ``[ConnectionView]`` sorted by label then peer id
``/trust``            GET     no     ``[TrustView]`` for every peer known to the trust store
``/trust/ban``        POST    yes    body ``{"peer_id": str, "reason": str (1-200 chars)}``
                                     -> ``{"banned": true}``; also disconnects the peer
``/trust/unban``      POST    yes    body ``{"peer_id": str}`` -> ``{"banned": false}``
``/generate/stream``  POST    no     body as ``/generate``; ``text/event-stream`` of
                                     ``event: token`` (``{"text": str, "index": int}``),
                                     then one ``event: done`` (the same JSON ``/generate``
                                     returns) or one ``event: error`` (``{"detail": str}``)
====================  ======  =====  =========================================================

Peer ids in request bodies must pass ``is_peer_id`` (422 otherwise). A wrong or missing
admin token is 401. Nothing here ever returns the swarm secret except ``/swarm/invite``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from fastapi import FastAPI

ADMIN_HEADER = "x-admin-token"


@dataclass(frozen=True)
class NodeView:
    peer_id: str
    label: str
    swarm_id: str
    swarm_name: str
    public: bool
    addresses: list[str]
    #: "public" | "private" | "unknown"
    reachability: str
    #: UPnP-mapped address, or "".
    mapped: str
    #: Relays holding a reservation for this device.
    relays: list[str]
    serving_relay: bool
    connections: int
    version: str


@dataclass(frozen=True)
class ConnectionView:
    peer_id: str
    label: str
    remote: str
    inbound: bool
    via_relay: str | None
    rtt_ms: float | None
    bytes_sent: int
    bytes_received: int
    #: From the pool's link estimator, bytes per second, or None before any sample.
    bandwidth_bps: float | None


@dataclass(frozen=True)
class TrustView:
    peer_id: str
    label: str
    checks: int
    failures: int
    score: float
    banned: bool
    reason: str
    served_ms: float
    used_ms: float


def add_swarm_routes(
    app: FastAPI,
    *,
    daemon: Any,
    device: Any,
    trust: Any,
    admin_token: str,
    tokenizer: Any,
    stop_token_ids: set[int] | None = None,
    system_prompt: str | None = None,
    link_estimator: Any | None = None,
) -> None:
    """Register the routes above on ``app``.

    ``daemon`` is a ``PeerDaemon`` (``.name``, ``.table``, ``.runner``, ``.config``,
    ``.record_asked``, ``.label_of``); ``device`` a ``SwarmDevice`` (``.node``, ``.spec``,
    ``.services.reachability``, ``.invite(config)``); ``trust`` a ``TrustStore``;
    ``link_estimator`` a ``LinkEstimator`` or ``None``.
    """
    raise NotImplementedError
