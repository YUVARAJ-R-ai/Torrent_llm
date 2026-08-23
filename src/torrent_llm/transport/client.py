"""Client side of one hop.

Every number the profiler (issue #6) reports is collected here, because this is
the only place that sees both the wire payload and the wall-clock cost of moving
it. The split matters: ``compute_ns`` comes from the server, ``wall_ns`` is
measured locally, and the difference is what the network actually cost.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass

import grpc
import torch

from torrent_llm.codec import Codec
from torrent_llm.transport import activation_pb2 as pb
from torrent_llm.transport import activation_pb2_grpc as pb_grpc
from torrent_llm.transport.convert import (
    CHANNEL_OPTIONS,
    header_to_proto,
    message_from_proto,
)


@dataclass(frozen=True)
class HopResult:
    """One hop's output and everything measured about it."""

    tensor: torch.Tensor
    is_final: bool
    hop: int
    #: Bytes actually put on the wire by the sender.
    sent_bytes: int
    #: Bytes the peer sent back.
    received_bytes: int
    #: Elements in the logical tensor sent, before any compression.
    sent_numel: int
    #: Logical bytes the raw tensor would have occupied, for the compression ratio.
    sent_uncompressed_bytes: int
    #: Round trip measured on this side: serialise + network + remote compute + return.
    wall_ns: int
    #: Time the peer spent in the forward pass, as reported by the peer.
    compute_ns: int
    codec: str

    @property
    def transport_ns(self) -> int:
        """Round trip minus remote compute — the network's share of this hop.

        Includes local encode/decode, so on a fast link with an expensive codec
        this can grow rather than shrink. That is the tradeoff the ablation is
        meant to expose, not an artefact to explain away.
        """
        return max(0, self.wall_ns - self.compute_ns)

    @property
    def compression_ratio(self) -> float:
        """Uncompressed logical bytes / bytes on the wire. 1.0 for a raw codec."""
        if self.sent_bytes == 0:
            return 1.0
        return self.sent_uncompressed_bytes / self.sent_bytes


class ShardClient:
    """Talks to one shard server."""

    def __init__(self, address: str, codec: Codec, *, timeout: float = 300.0) -> None:
        self.address = address
        self.codec = codec
        self.timeout = timeout
        self.channel = grpc.insecure_channel(address, options=CHANNEL_OPTIONS)
        self.stub = pb_grpc.ShardServiceStub(self.channel)

    def info(self) -> pb.InfoReply:
        """Ask the peer what it hosts."""
        return self.stub.Info(pb.InfoRequest(), timeout=self.timeout)

    def forward(
        self,
        tensor: torch.Tensor,
        *,
        request_id: str | None = None,
        hop: int = 0,
        is_token_ids: bool = False,
        position_ids: torch.Tensor | None = None,
    ) -> HopResult:
        """Send one activation and wait for the shard's output."""
        request_id = request_id or uuid.uuid4().hex
        message = self.codec.encode(tensor, request_id=request_id, hop=hop)

        request = pb.ForwardRequest(
            header=header_to_proto(message.header),
            payload=message.payload,
            kind=(pb.PAYLOAD_KIND_TOKEN_IDS if is_token_ids else pb.PAYLOAD_KIND_ACTIVATION),
            position_ids=(position_ids.flatten().tolist() if position_ids is not None else []),
            sent_unix_ns=time.time_ns(),
        )

        started = time.perf_counter_ns()
        reply = self.stub.Forward(request, timeout=self.timeout)
        wall_ns = time.perf_counter_ns() - started

        out = self.codec.decode(message_from_proto(reply.header, reply.payload))
        return HopResult(
            tensor=out,
            is_final=reply.is_final,
            hop=hop,
            sent_bytes=message.payload_bytes,
            received_bytes=len(reply.payload),
            sent_numel=message.header.logical_numel,
            sent_uncompressed_bytes=tensor.numel() * tensor.element_size(),
            wall_ns=wall_ns,
            compute_ns=reply.compute_ns,
            codec=message.header.codec,
        )

    def close(self) -> None:
        self.channel.close()

    def __enter__(self) -> ShardClient:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()
