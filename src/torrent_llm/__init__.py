"""Torrent-LLM: layer-sharded P2P LLM inference with compressed activation transfer.

Package layout::

    shard/      hosts a contiguous layer range and runs its slice of the forward pass
    codec/      pluggable activation (de)serialisation — the seam the compressor plugs into
    transport/  gRPC service, wire schema, tensor framing
    profile/    per-hop byte counts and wall-times, written as JSONL
    runner/     client that walks the shard chain end to end

The codec seam is the load-bearing design decision: transport only ever moves
opaque ``bytes``, so a learned low-rank compressor (issue #8) can replace the
passthrough codec without touching a line of networking code.
"""

__version__ = "0.1.0"
