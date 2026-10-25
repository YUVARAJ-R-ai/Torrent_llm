"""The HTTP face of one device.

Any device can be talked to: they all serve the same endpoints, so a dashboard
or a script can use whichever it can reach, and switch to another if that one
goes away.

========================  =====================================================
``GET  /pool``            the pool as this device sees it
``GET  /events``          what has changed lately (joins, departures, handovers)
``POST /generate``        ask the pool a question; this device routes it
``POST /gossip``          other devices swap tables here (needs the pool token)
``POST /admin/owner``     the owner wants the device, or is done with it
``POST /admin/leave``     leave the pool, saying goodbye first
========================  =====================================================

Asking questions is open, like the single-chain API: it is for people using the
pool. Anything that changes the pool itself needs the pool's shared token when
one is set.
"""

from __future__ import annotations

import dataclasses
import threading
import time
from typing import Any, Protocol

import torch
from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

from torrent_llm.chat import DEFAULT_SYSTEM_PROMPT, NoChatTemplate, render_chat
from torrent_llm.pool.daemon import PeerDaemon
from torrent_llm.pool.gossip import TOKEN_HEADER
from torrent_llm.pool.members import Member
from torrent_llm.pool.placement import coverage
from torrent_llm.pool.router import NoReplica
from torrent_llm.pool.runner import PoolUnavailable
from torrent_llm.sampling import Sampling


class Tokenizer(Protocol):
    def encode(self, text: str) -> list[int]: ...

    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str: ...


class GenerateIn(BaseModel):
    prompt: str
    max_new_tokens: int = Field(default=40, ge=1, le=256)
    chat: bool = Field(default=True, description="Ask a question, not continue raw text")
    temperature: float = Field(default=0.0, ge=0.0)
    top_k: int = Field(default=0, ge=0)
    seed: int = 0


class OwnerIn(BaseModel):
    busy: bool


class GossipIn(BaseModel):
    sender: str
    members: list[dict[str, Any]]


def create_peer_app(
    daemon: PeerDaemon,
    tokenizer: Tokenizer,
    *,
    stop_token_ids: set[int] | None = None,
    system_prompt: str | None = DEFAULT_SYSTEM_PROMPT,
    on_leave=None,
) -> FastAPI:
    app = FastAPI(
        title=f"Torrent-LLM pool device: {daemon.name}",
        description=__doc__,
        version="0.1.0",
    )
    stop_ids = stop_token_ids or set()
    config = daemon.config

    def require_token(x_pool_token: str | None = Header(default=None, alias=TOKEN_HEADER)) -> None:
        if daemon.token and x_pool_token != daemon.token:
            raise HTTPException(status_code=401, detail="this pool needs its shared token")

    # --- reading ---------------------------------------------------------------

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "device": daemon.name}

    @app.get("/pool")
    def pool() -> dict[str, Any]:
        snapshot = daemon.table.snapshot()
        usable = [m for m, liveness in snapshot if liveness == "alive"]
        ready = coverage(config, usable)
        rtt = daemon.runner.rtt_ms
        members = []
        for member, liveness in snapshot:
            row = member.to_dict()
            row["liveness"] = liveness
            row["is_self"] = member.name == daemon.name
            # Measured from this device, where a route would start.
            hosted_rtt = [
                rtt[f"{member.host}:{h.port}"]
                for h in member.hosting
                if f"{member.host}:{h.port}" in rtt
            ]
            row["rtt_ms"] = round(sum(hosted_rtt) / len(hosted_rtt), 1) if hosted_rtt else None
            members.append(row)
        if min(ready, default=0) >= config.replicas:
            health = "healthy"
        elif min(ready, default=0) >= 1:
            health = "degraded"
        else:
            health = "down"
        return {
            "self": daemon.name,
            "pool": {
                "model_id": config.model_id,
                "num_layers": config.num_layers,
                "replicas": config.replicas,
                "shard_mb": config.shard_mb,
                "stages": [
                    {"stage": s.index, "start": s.start, "end": s.end} for s in config.specs()
                ],
            },
            "members": members,
            "ready_copies": ready,
            "health": health,
            "now": time.time(),
        }

    @app.get("/events")
    def events(since: float = 0.0) -> list[dict]:
        return daemon.recent_events(since)

    # --- the pool talking to itself ---------------------------------------------

    @app.post("/gossip", dependencies=[Depends(require_token)])
    def gossip(body: GossipIn) -> dict[str, Any]:
        mine = daemon.receive(body.sender, [Member.from_dict(m) for m in body.members])
        return {"members": mine}

    # --- asking a question --------------------------------------------------------

    @app.post("/generate")
    def generate(body: GenerateIn) -> dict[str, Any]:
        text = body.prompt
        if body.chat:
            try:
                text = render_chat(tokenizer, body.prompt, system_prompt)
            except NoChatTemplate as exc:
                raise HTTPException(status_code=422, detail=str(exc)) from exc
        ids = tokenizer.encode(text)
        if not ids:
            raise HTTPException(status_code=422, detail="prompt encoded to zero tokens")
        input_ids = torch.tensor([ids], dtype=torch.long)
        try:
            result = daemon.runner.generate(
                input_ids,
                max_new_tokens=body.max_new_tokens,
                stop_token_ids=stop_ids,
                sampling=Sampling(temperature=body.temperature, top_k=body.top_k, seed=body.seed),
            )
        except (PoolUnavailable, NoReplica) as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

        generated = result.ids[0, input_ids.shape[1] :].tolist()
        tokens = _per_token_text(tokenizer, generated)
        passes = [dataclasses.asdict(p) for p in result.passes]
        hops = [h for p in result.passes for h in p.hops]
        return {
            "prompt": body.prompt,
            "completion": "".join(tokens),
            "tokens": tokens,
            "requester": daemon.name,
            "passes": passes,
            "events": [dataclasses.asdict(e) for e in result.events],
            "totals": {
                "tokens": len(tokens),
                "failovers": result.failovers,
                "sent_bytes": sum(h.sent_bytes for h in hops),
                "compute_ms": round(sum(h.compute_ms for h in hops), 2),
                "transport_ms": round(sum(h.transport_ms for h in hops), 2),
            },
        }

    # --- the owner's controls --------------------------------------------------------

    @app.post("/admin/owner", dependencies=[Depends(require_token)])
    def owner(body: OwnerIn) -> dict[str, str]:
        return {"state": daemon.set_owner_busy(body.busy)}

    @app.post("/admin/leave", dependencies=[Depends(require_token)])
    def leave() -> dict[str, str]:
        # Reply first, leave a moment later: leaving stops this very server.
        if on_leave is not None:
            threading.Timer(0.2, on_leave).start()
        return {"state": "leaving"}

    return app


def _per_token_text(tokenizer: Tokenizer, ids: list[int]) -> list[str]:
    """Each token's text, as the growth of the decoded prefix.

    Decoding a token alone mangles a character that spans two tokens, so the
    text of token *n* is whatever it added to the decoded prefix.
    """
    out: list[str] = []
    previous = ""
    for end in range(1, len(ids) + 1):
        text = tokenizer.decode(ids[:end], skip_special_tokens=True)
        out.append(text[len(previous) :])
        previous = text
    return out
