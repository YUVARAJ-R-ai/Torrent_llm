"""Shard calls over the overlay, in the shape the rest of the pool already speaks.

The pool's runner and chain were written against gRPC: a failed hop raises a
``grpc.RpcError`` whose status says whether to try another copy (``UNAVAILABLE``),
to back off (``RESOURCE_EXHAUSTED``) or to re-prefill (``FAILED_PRECONDITION``).
This module keeps that contract while the bytes travel over the encrypted overlay
(``torrent_llm.net``): requests are the same protobuf ``ForwardRequest``, sent as the
body of a ``shard.forward`` call, and overlay errors are translated to the same
status codes. So failover, busy handling and cache recovery work unchanged.

Every hop gets a deadline sized to what it carries (``hop_timeout``), instead of
the old fixed 300 s. A peer that goes silent is caught sooner still, by the
connection's liveness check (``rpc.py``).
"""

from __future__ import annotations

import logging

import grpc

from torrent_llm.codec.profiles import UnknownProfile
from torrent_llm.net.node import Node
from torrent_llm.net.rpc import CallContext, RpcError
from torrent_llm.pool.hosting import Busy, NotHosted, ShardHost
from torrent_llm.transport import activation_pb2 as pb
from torrent_llm.transport.server import CacheSessionLost

logger = logging.getLogger(__name__)

METHOD = "shard.forward"

#: Overlay error code -> the gRPC status the runner reacts to.
_STATUS = {
    "unavailable": grpc.StatusCode.UNAVAILABLE,
    "deadline": grpc.StatusCode.DEADLINE_EXCEEDED,
    "busy": grpc.StatusCode.RESOURCE_EXHAUSTED,
    "cache_lost": grpc.StatusCode.FAILED_PRECONDITION,
    "denied": grpc.StatusCode.PERMISSION_DENIED,
    # The stage is gone from that device (unloaded, failed): try another copy.
    "not_found": grpc.StatusCode.UNAVAILABLE,
    "bad_request": grpc.StatusCode.INVALID_ARGUMENT,
    "too_large": grpc.StatusCode.INVALID_ARGUMENT,
    "internal": grpc.StatusCode.INTERNAL,
    "cancelled": grpc.StatusCode.CANCELLED,
}

#: Floor of every hop's deadline: a handshake plus a little compute.
BASE_TIMEOUT_S = 8.0
#: Bandwidth assumed when sizing a deadline (bytes/s): a poor phone link, 1 Mbit/s.
WORST_BANDWIDTH_BPS = 125_000
#: Compute allowance per token in the request, generous for a CPU-only laptop.
PER_TOKEN_S = 0.05


class OverlayRpcError(grpc.RpcError):
    """An overlay failure dressed as a gRPC error, so ``HopError`` can read it."""

    def __init__(self, error: RpcError) -> None:
        super().__init__(str(error))
        self.error = error
        self._code = _STATUS.get(error.code, grpc.StatusCode.UNKNOWN)

    def code(self) -> grpc.StatusCode:
        return self._code

    def details(self) -> str:
        return f"{self.error.code}: {self.error.message}" if self.error.message else self.error.code


def hop_timeout(request: pb.ForwardRequest) -> float:
    """How long one hop may take: enough for its bytes on a slow link and its compute."""
    tokens = request.header.shape[1] if len(request.header.shape) >= 2 else 1
    return (
        BASE_TIMEOUT_S
        + len(request.payload) / WORST_BANDWIDTH_BPS
        + PER_TOKEN_S * max(1, int(tokens))
    )


class OverlaySender:
    """Sends one stage's requests to one peer: a ``Sender`` for ``ShardClient``."""

    def __init__(self, node: Node, peer_id: str, stage: int) -> None:
        self.node = node
        self.peer_id = peer_id
        self.stage = stage

    def __call__(self, request: pb.ForwardRequest) -> pb.ForwardReply:
        try:
            _, body = self.node.call(
                self.peer_id,
                METHOD,
                {"stage": self.stage},
                request.SerializeToString(),
                timeout=hop_timeout(request),
            )
        except RpcError as exc:
            raise OverlayRpcError(exc) from exc
        reply = pb.ForwardReply()
        try:
            reply.ParseFromString(body)
        except Exception as exc:  # noqa: BLE001 - protobuf raises DecodeError and others
            raise OverlayRpcError(RpcError("internal", f"bad reply: {exc}")) from exc
        return reply


def serve_shards(node: Node, host: ShardHost) -> None:
    """Register ``shard.forward`` on ``node``, served by ``host``'s stages."""

    def handle(ctx: CallContext, header: dict, body: bytes) -> tuple[dict, bytes]:
        stage = header.get("stage")
        if not isinstance(stage, int) or isinstance(stage, bool) or stage < 0:
            raise RpcError("bad_request", "stage must be a non-negative integer")
        request = pb.ForwardRequest()
        try:
            request.ParseFromString(body)
        except Exception as exc:  # noqa: BLE001 - untrusted bytes
            raise RpcError("bad_request", "not a forward request") from exc
        try:
            reply = host.forward(stage, request)
        except NotHosted as exc:
            raise RpcError("not_found", str(exc)) from exc
        except Busy as exc:
            raise RpcError("busy", str(exc)) from exc
        except CacheSessionLost as exc:
            raise RpcError("cache_lost", str(exc)) from exc
        except UnknownProfile as exc:
            raise RpcError("bad_request", str(exc)) from exc
        except ValueError as exc:
            # Malformed tensors from the network land here (codec validation).
            raise RpcError("bad_request", str(exc)) from exc
        return {}, reply.SerializeToString()

    node.register(METHOD, handle, blocking=True)
