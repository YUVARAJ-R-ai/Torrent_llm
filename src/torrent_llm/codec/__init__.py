"""Pluggable activation codecs.

Importing this package registers every built-in codec, so ``get_codec("raw")``
works without the caller knowing which module defines it.
"""

from torrent_llm.codec.base import Codec, available_codecs, get_codec, register_codec
from torrent_llm.codec.int8 import Int8Codec
from torrent_llm.codec.lowrank import LowRankCodec
from torrent_llm.codec.raw import RawCodec

__all__ = [
    "Codec",
    "Int8Codec",
    "LowRankCodec",
    "RawCodec",
    "available_codecs",
    "get_codec",
    "register_codec",
]
