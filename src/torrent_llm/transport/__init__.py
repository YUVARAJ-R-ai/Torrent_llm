"""gRPC transport for activation hops.

The generated protobuf stubs are not committed, so a clean checkout has to build
them once before this package can be imported at all.
"""

try:
    from torrent_llm.transport import activation_pb2 as _stub_check  # noqa: F401
except ImportError as exc:  # pragma: no cover - only hit on a clean checkout
    raise ImportError(
        "the gRPC stubs have not been generated yet. Run:\n\n"
        "    python -m torrent_llm.codegen\n\n"
        "They are built from src/torrent_llm/transport/activation.proto rather "
        "than committed, so the proto stays the single source of truth."
    ) from exc

from torrent_llm.transport.client import HopResult, ShardClient
from torrent_llm.transport.convert import MAX_MESSAGE_BYTES
from torrent_llm.transport.server import ShardService, serve

__all__ = ["HopResult", "MAX_MESSAGE_BYTES", "ShardClient", "ShardService", "serve"]
