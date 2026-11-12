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

import dataclasses
import hmac
import json
import logging
import queue
import threading
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import torch
from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, field_validator

import torrent_llm
from torrent_llm.chat import NoChatTemplate, render_chat
from torrent_llm.net.identity import is_peer_id
from torrent_llm.pool.api import GenerateIn, _per_token_text
from torrent_llm.pool.router import NoReplica
from torrent_llm.pool.runner import PoolUnavailable
from torrent_llm.pool.swarm_cli import join_commands
from torrent_llm.sampling import Sampling

logger = logging.getLogger(__name__)

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


class BanIn(BaseModel):
    """Request body to ban a peer."""

    peer_id: str
    reason: str

    @field_validator("reason")
    @classmethod
    def _validate_reason(cls, value: str) -> str:
        """Strip whitespace and enforce length."""
        stripped = value.strip()
        if not (1 <= len(stripped) <= 200):
            raise ValueError("reason must be 1-200 characters")
        return stripped


class UnbanIn(BaseModel):
    """Request body to unban a peer."""

    peer_id: str


# Mirrors api.py's /generate body so a later refactor can extract a shared helper.
def _build_done_payload(
    prompt: str,
    daemon_name: str,
    tokenizer: Any,
    input_ids: torch.Tensor,
    result: Any,
) -> dict[str, Any]:
    """Build the JSON payload for a finished generation mirroring api.py /generate."""
    generated = result.ids[0, input_ids.shape[1] :].tolist()
    tokens = _per_token_text(tokenizer, generated)
    passes = [dataclasses.asdict(p) for p in result.passes]
    hops = [h for p in result.passes for h in p.hops]
    return {
        "prompt": prompt,
        "completion": "".join(tokens),
        "tokens": tokens,
        "requester": daemon_name,
        "passes": passes,
        "events": [dataclasses.asdict(e) for e in result.events],
        "totals": {
            "tokens": len(tokens),
            "failovers": result.failovers,
            "sent_bytes": sum(h.sent_bytes for h in hops),
            "uncompressed_bytes": sum(h.uncompressed_bytes for h in hops),
            "compute_ms": round(sum(h.compute_ms for h in hops), 2),
            "transport_ms": round(sum(h.transport_ms for h in hops), 2),
        },
    }


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
    if not admin_token:
        raise ValueError("add_swarm_routes needs a non-empty admin token")

    stop_ids = stop_token_ids or set()

    def require_admin(
        x_admin_token: str | None = Header(default=None, alias=ADMIN_HEADER),
    ) -> None:
        """Verify the admin token header matches the configured admin token."""
        if x_admin_token is None or not hmac.compare_digest(
            x_admin_token.encode("utf-8"), admin_token.encode("utf-8")
        ):
            raise HTTPException(status_code=401, detail="this needs the admin token")

    @app.get("/node")
    def node() -> dict[str, Any]:
        """Return the local node view and swarm metadata."""
        reachability = device.services.reachability
        view = NodeView(
            peer_id=device.node.peer_id,
            label=daemon.label_of(daemon.name),
            swarm_id=device.spec.swarm_id,
            swarm_name=device.spec.name,
            public=device.spec.public,
            addresses=list(device.node.addresses()),
            reachability=reachability.status,
            mapped=reachability.mapped,
            relays=list(reachability.relays),
            serving_relay=reachability.serving_relay,
            connections=len(device.node.connections()),
            version=getattr(torrent_llm, "__version__", "0.1.0"),
        )
        return dataclasses.asdict(view)

    @app.get("/swarm/invite", dependencies=[Depends(require_admin)])
    def invite() -> dict[str, Any]:
        """Return an invite link and join commands for this swarm."""
        inv = device.invite(daemon.config)
        if inv is None:
            raise HTTPException(
                status_code=404,
                detail="the public swarm has no invite: anyone can join with torrent-peer public",
            )
        link = inv.to_link()
        return {"link": link, "commands": join_commands(link)}

    @app.get("/peers")
    def peers() -> list[dict[str, Any]]:
        """List connected peers sorted by label and peer id."""
        views: list[ConnectionView] = []
        for conn in device.node.connections():
            views.append(
                ConnectionView(
                    peer_id=conn.peer_id,
                    label=daemon.label_of(conn.peer_id),
                    remote=conn.remote,
                    inbound=conn.inbound,
                    via_relay=conn.via_relay,
                    rtt_ms=conn.rtt_ms,
                    bytes_sent=conn.bytes_sent,
                    bytes_received=conn.bytes_received,
                    bandwidth_bps=(
                        link_estimator.bandwidth(conn.peer_id)
                        if link_estimator is not None
                        else None
                    ),
                )
            )
        views.sort(key=lambda c: (c.label, c.peer_id))
        return [dataclasses.asdict(c) for c in views]

    @app.get("/trust")
    def trust_route() -> list[dict[str, Any]]:
        """List trust and reputation records for known peers."""
        views = [
            TrustView(
                peer_id=rep.peer_id,
                label=daemon.label_of(rep.peer_id),
                checks=rep.checks,
                failures=rep.failures,
                score=rep.score,
                banned=rep.banned,
                reason=rep.reason,
                served_ms=rep.served_ms,
                used_ms=rep.used_ms,
            )
            for rep in trust.all()
        ]
        return [dataclasses.asdict(t) for t in views]

    @app.post("/trust/ban", dependencies=[Depends(require_admin)])
    def ban(body: BanIn) -> dict[str, bool]:
        """Ban a peer and disconnect it."""
        if not is_peer_id(body.peer_id):
            raise HTTPException(status_code=422, detail="not a peer id")
        trust.ban(body.peer_id, body.reason)
        try:
            device.node.disconnect(body.peer_id)
        except Exception as exc:  # noqa: BLE001
            logger.debug("disconnect ignored: %s", exc)
        return {"banned": True}

    @app.post("/trust/unban", dependencies=[Depends(require_admin)])
    def unban(body: UnbanIn) -> dict[str, bool]:
        """Unban a peer."""
        if not is_peer_id(body.peer_id):
            raise HTTPException(status_code=422, detail="not a peer id")
        trust.unban(body.peer_id)
        return {"banned": False}

    @app.post("/generate/stream")
    def generate_stream(body: GenerateIn) -> StreamingResponse:
        """Stream generated tokens as Server-Sent Events."""
        text = body.prompt
        if body.chat:
            try:
                chosen = system_prompt if body.system_prompt is None else body.system_prompt
                text = render_chat(tokenizer, body.prompt, chosen)
            except NoChatTemplate as exc:
                raise HTTPException(status_code=422, detail=str(exc)) from exc
        ids = tokenizer.encode(text)
        if not ids:
            raise HTTPException(status_code=422, detail="prompt encoded to zero tokens")
        input_ids = torch.tensor([ids], dtype=torch.long)

        q: queue.Queue[tuple[str, dict[str, Any]] | None] = queue.Queue()

        def worker() -> None:
            """Run generation in a worker thread and push events to the queue."""
            try:
                accumulated_ids: list[int] = []
                previous = ""
                index = 0

                def on_token(token_id: int) -> None:
                    """Incrementally push generated token text."""
                    nonlocal previous, index
                    accumulated_ids.append(token_id)
                    full_text = tokenizer.decode(accumulated_ids, skip_special_tokens=True)
                    token_text = full_text[len(previous) :]
                    previous = full_text
                    q.put(("token", {"text": token_text, "index": index}))
                    index += 1

                result = daemon.runner.generate(
                    input_ids,
                    max_new_tokens=body.max_new_tokens,
                    stop_token_ids=stop_ids,
                    sampling=Sampling(
                        temperature=body.temperature, top_k=body.top_k, seed=body.seed
                    ),
                    on_token=on_token,
                )

                generated = result.ids[0, input_ids.shape[1] :].tolist()
                daemon.record_asked(len(generated))
                done_payload = _build_done_payload(
                    body.prompt, daemon.name, tokenizer, input_ids, result
                )
                q.put(("done", done_payload))
            except (PoolUnavailable, NoReplica) as exc:
                q.put(("error", {"detail": str(exc)}))
            except Exception:
                logger.exception("generation failed in stream")
                q.put(("error", {"detail": "the answer failed"}))
            finally:
                q.put(None)

        threading.Thread(target=worker, daemon=True).start()

        def event_stream() -> Iterator[str]:
            """Yield formatted Server-Sent Events from the queue."""
            while True:
                item = q.get()
                if item is None:
                    break
                name, payload = item
                yield f"event: {name}\ndata: {json.dumps(payload)}\n\n"

        return StreamingResponse(
            event_stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache"},
        )
