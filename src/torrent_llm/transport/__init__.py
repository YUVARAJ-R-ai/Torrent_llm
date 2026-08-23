"""gRPC transport for activation hops.

The generated protobuf stubs are not committed. If importing this package fails
with ``No module named 'torrent_llm.transport.activation_pb2'``, generate them::

    python -m torrent_llm.transport.codegen
"""

from torrent_llm.transport.client import HopResult, ShardClient
from torrent_llm.transport.convert import MAX_MESSAGE_BYTES
from torrent_llm.transport.server import ShardService, serve

__all__ = ["HopResult", "MAX_MESSAGE_BYTES", "ShardClient", "ShardService", "serve"]
