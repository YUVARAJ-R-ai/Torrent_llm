"""gRPC server hosting one shard.

The service is a thin shell around :class:`~torrent_llm.shard.ShardRuntime`:
decode the payload, run the layers, encode the result. It deliberately knows
nothing about *how* the payload was compressed — that is the codec's business —
so swapping in the learned compressor (issue #8) needs no change here.
"""

from __future__ import annotations

import logging
import time
from concurrent import futures

import grpc
import torch

from torrent_llm.codec import Codec, RawCodec
from torrent_llm.shard import ShardRuntime
from torrent_llm.transport import activation_pb2 as pb
from torrent_llm.transport import activation_pb2_grpc as pb_grpc
from torrent_llm.transport.convert import (
    CHANNEL_OPTIONS,
    MAX_MESSAGE_BYTES,
    header_to_proto,
    message_from_proto,
)

logger = logging.getLogger(__name__)


class ShardService(pb_grpc.ShardServiceServicer):
    """Serves one shard's slice of the forward pass."""

    def __init__(
        self,
        runtime: ShardRuntime,
        codec: Codec,
        *,
        model_id: str = "unknown",
    ) -> None:
        self.runtime = runtime
        self.codec = codec
        self.model_id = model_id
        # Logits are the answer, not an intermediate activation. Compressing
        # them would corrupt the very quality metric the benchmark reports, so
        # the final hop always goes out uncompressed regardless of the codec.
        self.output_codec: Codec = RawCodec() if runtime.spec.is_last else codec

    def Forward(self, request: pb.ForwardRequest, context) -> pb.ForwardReply:  # noqa: N802
        try:
            return self._forward(request)
        except Exception as exc:  # noqa: BLE001 - must not kill the server process
            logger.exception("forward failed on %s", self.runtime.spec)
            context.abort(grpc.StatusCode.INTERNAL, f"{self.runtime.spec}: {exc}")
            raise  # unreachable; abort raises, but keeps the type checker honest

    def _forward(self, request: pb.ForwardRequest) -> pb.ForwardReply:
        message = message_from_proto(request.header, request.payload)
        tensor = self.codec.decode(message)

        position_ids = (
            torch.tensor(list(request.position_ids), dtype=torch.long).unsqueeze(0)
            if request.position_ids
            else None
        )

        started = time.perf_counter_ns()
        if request.kind == pb.PAYLOAD_KIND_TOKEN_IDS:
            output = self.runtime.forward(
                input_ids=tensor.to(torch.long), position_ids=position_ids
            )
        else:
            output = self.runtime.forward(hidden_states=tensor, position_ids=position_ids)
        _synchronize(self.runtime.device)
        compute_ns = time.perf_counter_ns() - started

        tensor = output.tensor
        if self.runtime.spec.is_last:
            tensor = _trim_logits(tensor, request.logits_keep_last)

        reply_message = self.output_codec.encode(
            tensor,
            request_id=message.header.request_id,
            hop=message.header.hop + 1,
        )
        _check_reply_size(reply_message.payload_bytes, self.runtime.spec.is_last)
        return pb.ForwardReply(
            header=header_to_proto(reply_message.header),
            payload=reply_message.payload,
            kind=pb.PAYLOAD_KIND_ACTIVATION,
            compute_ns=compute_ns,
            is_final=self.runtime.spec.is_last,
        )

    def Info(self, request: pb.InfoRequest, context) -> pb.InfoReply:  # noqa: N802
        spec = self.runtime.spec
        return pb.InfoReply(
            shard_index=spec.index,
            layer_start=spec.start,
            layer_end=spec.end,
            num_layers=spec.num_layers,
            hidden_size=self.runtime.hidden_size,
            model_id=self.model_id,
            dtype=str(self.runtime.dtype).removeprefix("torch."),
            device=str(self.runtime.device),
            is_first=spec.is_first,
            is_last=spec.is_last,
        )


def serve(
    runtime: ShardRuntime,
    codec: Codec,
    *,
    host: str = "0.0.0.0",
    port: int = 0,
    model_id: str = "unknown",
    max_workers: int = 4,
) -> tuple[grpc.Server, int]:
    """Start a shard server.

    Args:
        port: ``0`` asks the OS for a free port, which is what the tests and the
            single-machine multi-process rig use.

    Returns:
        The started server and the port it actually bound.
    """
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=max_workers), options=CHANNEL_OPTIONS
    )
    pb_grpc.add_ShardServiceServicer_to_server(
        ShardService(runtime, codec, model_id=model_id), server
    )
    bound = server.add_insecure_port(f"{host}:{port}")
    if bound == 0:
        raise RuntimeError(f"could not bind {host}:{port}")
    server.start()
    logger.info("serving %s on %s:%d", runtime.spec, host, bound)
    return server, bound


def _trim_logits(logits: torch.Tensor, keep_last: int) -> torch.Tensor:
    """Return only the trailing ``keep_last`` positions of the logits.

    ``0`` means every position. Defaulting to *all* is deliberate: silently
    truncating what the caller asked for would corrupt a perplexity evaluation
    in a way that looks like a modelling result, not a bug.
    """
    if keep_last <= 0:
        return logits
    return logits[:, -keep_last:, :]


def _check_reply_size(nbytes: int, is_last: bool) -> None:
    """Fail with an actionable message instead of a bare RESOURCE_EXHAUSTED.

    The cap is almost always hit by a full-sequence logits reply, because logits
    are seq x vocab and vocab dwarfs hidden_size. Saying so is more useful than
    reporting two large numbers.
    """
    if nbytes <= MAX_MESSAGE_BYTES:
        return
    hint = (
        " This is the final shard returning logits for every position "
        "(batch x seq x vocab). Pass logits_keep_last=1 unless you genuinely "
        "need all of them, as generation does not."
        if is_last
        else ""
    )
    raise ValueError(
        f"reply payload is {nbytes / 1024**2:.0f} MiB, over the "
        f"{MAX_MESSAGE_BYTES / 1024**2:.0f} MiB transport cap.{hint}"
    )


def _synchronize(device: torch.device) -> None:
    """Make CUDA timings real.

    Without this, ``compute_ns`` measures how long it took to *queue* the kernels,
    which would understate GPU compute and overstate the network's share of
    end-to-end latency — the exact ratio this project is trying to measure.
    """
    if device.type == "cuda":
        torch.cuda.synchronize(device)
