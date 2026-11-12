"""Sharing model files between devices, like a torrent shares a file.

A device joining a swarm needs the model's weights. Downloading them from the internet
is slow on a phone hotspot and impossible on an offline LAN, while the devices already
in the swarm have them on disk. So devices serve their model files to each other over
the overlay, in fixed-size chunks, and a new device fetches chunks from several devices
at once.

Nobody has to be trusted for this. The swarm's pool config carries one hash,
``model_manifest``: the SHA-256 of the manifest (every file's path, size and SHA-256).
A fetching device asks any peer for the manifest, checks it against that hash, then
checks every finished file against the manifest. A device that sends wrong bytes
wastes a little time and nothing else; its chunks are fetched again from someone else.

Wire (``torrent_llm.net`` RPC, JSON header + raw body):

=================  ===========================================================  ==========
method             request header -> reply header                               body
=================  ===========================================================  ==========
``blob.manifest``  ``{"model": id}`` -> ``{"files": [[path, size, sha256]...]}``  -
``blob.get``       ``{"model": id, "path": p, "offset": o, "length": n}``        the bytes
=================  ===========================================================  ==========

Errors: ``not_found`` (model or path not served here), ``bad_request`` (malformed
header, offset/length out of range, length over ``CHUNK_BYTES``), ``busy`` (more than
``MAX_SERVING`` reads in progress on this device).
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from torrent_llm.net.node import Node
from torrent_llm.net.rpc import CallContext

MANIFEST = "blob.manifest"
GET = "blob.get"
#: Bytes per chunk: big enough that per-call overhead is small, small enough that a
#: lost peer costs little and a slow one does not hold up the end of a file for long.
CHUNK_BYTES = 1024 * 1024
#: Chunks a fetcher keeps in flight per peer.
PER_PEER_INFLIGHT = 4
#: Concurrent ``blob.get`` reads one device serves before answering ``busy``.
MAX_SERVING = 16
#: Most files a manifest may list (a model has tens; this bounds untrusted input).
MAX_FILES = 4096
#: Files that make up a model directory; everything else (READMEs, images, .git) is skipped.
MODEL_SUFFIXES = (".safetensors", ".json", ".model", ".txt", ".tiktoken", ".py", ".bin")


class BlobError(RuntimeError):
    """A model could not be fetched: no peer had it, or no peer sent a matching copy."""


@dataclass(frozen=True)
class BlobFile:
    """One file of a model. ``path`` is relative, POSIX-style, never ``..`` or absolute."""

    path: str
    size: int
    sha256: str


def manifest_of(directory: Path) -> list[BlobFile]:
    """Every model file under ``directory`` (recursively, following symlinks, which is
    how the Hugging Face cache stores snapshots), sorted by path. Only files whose name
    ends with one of ``MODEL_SUFFIXES``; hidden files and directories (a name part
    starting with ``.``) are skipped. Hashes are computed by streaming the file in
    ``CHUNK_BYTES`` pieces.

    Example: a dir with ``config.json`` (2 bytes ``{}``) and ``notes.md`` ->
    ``[BlobFile("config.json", 2, sha256(b"{}").hexdigest())]``.
    """
    raise NotImplementedError


def manifest_digest(files: Sequence[BlobFile]) -> str:
    """The hash that goes in ``PoolConfig.model_manifest``.

    SHA-256 hex of the canonical JSON ``[[path, size, sha256], ...]`` sorted by path,
    with ``separators=(",", ":")``. Order of ``files`` does not matter.
    """
    rows = sorted([f.path, f.size, f.sha256] for f in files)
    return hashlib.sha256(json.dumps(rows, separators=(",", ":")).encode()).hexdigest()


def parse_manifest(raw: object) -> list[BlobFile]:
    """Validate a manifest from the wire (``[[path, size, sha256], ...]``).

    ``ValueError`` unless: a list of at most ``MAX_FILES`` entries; each a 3-item list;
    ``path`` a non-empty str, relative, POSIX, no ``..``/``.``/empty parts, no backslash,
    no NUL, at most 512 chars, unique; ``size`` an int (not bool) >= 0; ``sha256``
    64 lowercase hex chars.
    """
    raise NotImplementedError


class BlobServer:
    """Serves the model files this device has, to other devices.

    Args:
        node: The overlay node; ``blob.manifest`` and ``blob.get`` are registered on it.
        locate: ``model id -> directory`` holding that model's files, or ``None`` when
            this device does not have it. Called on every request (cheap lookups only).

    Each model's manifest is computed once, on the first request for it, and cached.
    Only paths listed in the manifest can be read, so nothing outside the model
    directory is ever served, whatever the request says. ``close()`` unregisters both
    methods and is idempotent.
    """

    def __init__(self, node: Node, locate: Callable[[str], Path | None]) -> None:
        raise NotImplementedError

    def close(self) -> None:
        raise NotImplementedError

    def _on_manifest(self, ctx: CallContext, header: dict, body: bytes) -> tuple[dict, bytes]:
        raise NotImplementedError

    def _on_get(self, ctx: CallContext, header: dict, body: bytes) -> tuple[dict, bytes]:
        raise NotImplementedError


class BlobFetcher:
    """Fetches a model from the devices that have it.

    Args:
        node: The overlay node to fetch over.
        timeout: Per-call timeout (seconds) for ``blob.get`` and ``blob.manifest``.
    """

    def __init__(self, node: Node, *, timeout: float = 30.0) -> None:
        raise NotImplementedError

    def fetch(
        self,
        model_id: str,
        expected_digest: str,
        peers: Sequence[str],
        dest: Path,
        *,
        on_progress: Callable[[int, int], None] | None = None,
    ) -> list[BlobFile]:
        """Download ``model_id`` into ``dest`` and return its manifest.

        1. Ask peers (in order) for the manifest until one answers with a manifest whose
           ``manifest_digest`` equals ``expected_digest``; a peer whose manifest does
           not match is skipped for the rest of this fetch. None matches -> ``BlobError``.
        2. A file already in ``dest`` with the right size and hash is kept as is.
        3. Every other file is fetched in ``CHUNK_BYTES`` chunks into ``<path>.part``
           (parent directories created), spread over all usable peers, at most
           ``PER_PEER_INFLIGHT`` chunks in flight per peer (threads are fine). A chunk
           that fails (any ``RpcError``, or a body whose length is not what was asked)
           is retried on another peer; a peer that fails 3 times is dropped.
        4. When a file is complete its SHA-256 is checked. On mismatch the ``.part`` is
           deleted and the file is fetched again, this time all of it from ONE peer, one
           peer at a time, so the culprit shows: a peer whose whole copy mismatches is
           dropped. No peer left with a matching copy -> ``BlobError``.
        5. A matching file is moved into place with ``os.replace``.
        6. ``on_progress(done_bytes, total_bytes)`` after every chunk (from any thread).

        Raises ``BlobError`` when no peers are left or a file never matches. Files that
        were completed stay in ``dest`` (a later fetch resumes from them).
        """
        raise NotImplementedError
