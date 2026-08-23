"""Translation between the protobuf wire types and the plain dataclasses.

Isolated in its own module so nothing outside ``transport`` ever imports a
generated stub. The rest of the codebase works with :mod:`torrent_llm.wire`.
"""

from __future__ import annotations

import json

from torrent_llm.transport import activation_pb2 as pb
from torrent_llm.wire import ActivationHeader, ActivationMessage

# gRPC defaults to a 4 MiB message limit. A single prefill activation on an 8B
# model at 2k context is ~16 MiB, so the default would reject the very payload
# this project exists to measure. Set generously; the profiler, not the
# transport, is what we want reporting on size.
MAX_MESSAGE_BYTES = 512 * 1024 * 1024

CHANNEL_OPTIONS = [
    ("grpc.max_send_message_length", MAX_MESSAGE_BYTES),
    ("grpc.max_receive_message_length", MAX_MESSAGE_BYTES),
]


def header_to_proto(header: ActivationHeader) -> pb.Header:
    """Dataclass header -> protobuf header."""
    return pb.Header(
        request_id=header.request_id,
        hop=header.hop,
        codec=header.codec,
        dtype=header.dtype,
        shape=list(header.shape),
        # codec_meta is codec-private and open-ended, so it rides as JSON rather
        # than forcing a proto change every time a codec grows a field.
        codec_meta=json.dumps(header.codec_meta) if header.codec_meta else "",
    )


def header_from_proto(proto: pb.Header) -> ActivationHeader:
    """Protobuf header -> dataclass header."""
    return ActivationHeader(
        request_id=proto.request_id,
        hop=proto.hop,
        codec=proto.codec,
        dtype=proto.dtype,
        shape=tuple(proto.shape),
        codec_meta=json.loads(proto.codec_meta) if proto.codec_meta else {},
    )


def message_from_proto(header: pb.Header, payload: bytes) -> ActivationMessage:
    """Rebuild the codec-facing message from a request or reply."""
    return ActivationMessage(header=header_from_proto(header), payload=payload)
