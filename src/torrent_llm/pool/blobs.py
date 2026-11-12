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
wastes a little time and nothing else: the file is fetched again from one other device,
and comparing the two copies chunk by chunk names exactly who lied.

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

import collections
import hashlib
import json
import logging
import os
import re
import threading
import time
from collections.abc import Callable, Sequence
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from pathlib import Path

from torrent_llm.net.node import Node
from torrent_llm.net.rpc import CallContext, RpcError

logger = logging.getLogger(__name__)

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

#: Consecutive failures (not counting ``busy``) after which a peer is dropped.
_MAX_FAILURES = 3
#: Back-off after a ``busy`` answer: first delay, and the cap it doubles up to.
_BUSY_FIRST_S, _BUSY_MAX_S = 0.05, 1.0
#: Fetcher threads at most, however many peers there are.
_MAX_THREADS = 32
_HEX64 = re.compile(r"[0-9a-f]{64}")
_WINDOWS_RESERVED = re.compile(r"(con|prn|aux|nul|com[1-9]|lpt[1-9])(\..*)?", re.IGNORECASE)


class BlobError(RuntimeError):
    """A model could not be fetched: no peer had it, or no peer sent a matching copy."""


@dataclass(frozen=True)
class BlobFile:
    """One file of a model. ``path`` is relative, POSIX-style, never ``..`` or absolute."""

    path: str
    size: int
    sha256: str


def _hash_file(path: Path) -> str:
    """SHA-256 hex of a file, read in ``CHUNK_BYTES`` pieces."""
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while piece := f.read(CHUNK_BYTES):
            digest.update(piece)
    return digest.hexdigest()


def manifest_of(directory: Path) -> list[BlobFile]:
    """Every model file under ``directory`` (recursively, following symlinks, which is
    how the Hugging Face cache stores snapshots), sorted by path. Only files whose name
    ends with one of ``MODEL_SUFFIXES``; hidden files and directories (a name part
    starting with ``.``) are skipped. Hashes are computed by streaming the file in
    ``CHUNK_BYTES`` pieces.

    Example: a dir with ``config.json`` (2 bytes ``{}``) and ``notes.md`` ->
    ``[BlobFile("config.json", 2, sha256(b"{}").hexdigest())]``.
    """
    root = Path(directory)
    found: list[BlobFile] = []
    seen_dirs: set[str] = set()
    for here, dirs, files in os.walk(root, followlinks=True):
        real = os.path.realpath(here)
        if real in seen_dirs:  # a symlink loop
            dirs.clear()
            continue
        seen_dirs.add(real)
        dirs[:] = sorted(d for d in dirs if not d.startswith("."))
        for name in files:
            if name.startswith(".") or not name.endswith(MODEL_SUFFIXES):
                continue
            path = Path(here) / name
            if not path.is_file():
                continue
            rel = path.relative_to(root).as_posix()
            found.append(BlobFile(rel, path.stat().st_size, _hash_file(path)))
    return sorted(found, key=lambda f: f.path)


def manifest_digest(files: Sequence[BlobFile]) -> str:
    """The hash that goes in ``PoolConfig.model_manifest``.

    SHA-256 hex of the canonical JSON ``[[path, size, sha256], ...]`` sorted by path,
    with ``separators=(",", ":")``. Order of ``files`` does not matter.
    """
    rows = sorted([f.path, f.size, f.sha256] for f in files)
    return hashlib.sha256(json.dumps(rows, separators=(",", ":")).encode()).hexdigest()


def _path_ok(path: object) -> bool:
    """Whether ``path`` is safe to create under a directory on every OS."""
    if not isinstance(path, str) or not 0 < len(path) <= 512:
        return False
    if path.startswith("/") or any(c in path for c in "\\:\0"):
        return False
    for part in path.split("/"):
        if part in ("", ".", "..") or part.endswith((".", " ")):
            return False
        if _WINDOWS_RESERVED.fullmatch(part):
            return False
    return True


def parse_manifest(raw: object) -> list[BlobFile]:
    """Validate a manifest from the wire (``[[path, size, sha256], ...]``).

    ``ValueError`` unless: a list of at most ``MAX_FILES`` entries; each a 3-item list;
    ``path`` a non-empty str, relative, POSIX, no ``..``/``.``/empty parts, no backslash,
    no NUL, at most 512 chars, unique; ``size`` an int (not bool) >= 0; ``sha256``
    64 lowercase hex chars. Also refused, so every OS can store the files: ``:``,
    Windows device names (``CON``, ``nul.json``, ...), parts ending in ``.`` or space,
    two paths equal but for case, and a path that is also another path's directory.
    """
    if not isinstance(raw, list) or len(raw) > MAX_FILES:
        raise ValueError(f"a manifest is a list of at most {MAX_FILES} files")
    files: list[BlobFile] = []
    folded: set[str] = set()
    for item in raw:
        if not isinstance(item, list) or len(item) != 3:
            raise ValueError("each manifest entry is [path, size, sha256]")
        path, size, digest = item
        if not _path_ok(path):
            raise ValueError(f"unsafe path in manifest: {path!r}")
        if type(size) is not int or size < 0:
            raise ValueError(f"bad size for {path!r}")
        if not isinstance(digest, str) or not _HEX64.fullmatch(digest):
            raise ValueError(f"bad sha256 for {path!r}")
        if path.lower() in folded:
            raise ValueError(f"duplicate path in manifest: {path!r}")
        folded.add(path.lower())
        files.append(BlobFile(path, size, digest))
    for f in files:  # "a" as a file and "a/b" would need "a" to be a directory too
        parts = f.path.lower().split("/")
        if any("/".join(parts[:i]) in folded for i in range(1, len(parts))):
            raise ValueError(f"{f.path!r} is inside a path that is a file")
    return files


# --- serving --------------------------------------------------------------------------


@dataclass
class _Served:
    directory: Path
    files: dict[str, BlobFile]
    listed: list[list]


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
        self._node = node
        self._locate = locate
        self._lock = threading.Lock()
        self._served: dict[str, _Served] = {}
        #: Models whose manifest is being computed; others asking wait on the event.
        self._computing: dict[str, threading.Event] = {}
        self._reading = 0
        self._closed = False
        node.register(MANIFEST, self._on_manifest, blocking=True)
        node.register(GET, self._on_get, blocking=True)

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
        for method in (MANIFEST, GET):
            self._node.unregister(method)

    def _model(self, header: dict) -> _Served:
        """The served manifest for the requested model, computing it once if needed."""
        model = header.get("model")
        if not isinstance(model, str) or not 0 < len(model) <= 200:
            raise RpcError("bad_request", "model must be a short string")
        directory = self._locate(model)
        if directory is None:
            raise RpcError("not_found", "this device does not have that model")
        while True:
            with self._lock:
                served = self._served.get(model)
                if served is not None and served.directory == Path(directory):
                    return served
                waiting = self._computing.get(model)
                if waiting is None:
                    done = self._computing[model] = threading.Event()
                    break
            waiting.wait(timeout=600)
        try:
            # Hashing a model takes a while: never under the lock, so other models and
            # other reads are served meanwhile.
            files = manifest_of(Path(directory))
            served = _Served(
                Path(directory),
                {f.path: f for f in files},
                [[f.path, f.size, f.sha256] for f in files],
            )
            with self._lock:
                self._served[model] = served
            return served
        finally:
            with self._lock:
                self._computing.pop(model, None)
            done.set()

    def _on_manifest(self, ctx: CallContext, header: dict, body: bytes) -> tuple[dict, bytes]:
        return {"files": self._model(header).listed}, b""

    def _on_get(self, ctx: CallContext, header: dict, body: bytes) -> tuple[dict, bytes]:
        path, offset, length = header.get("path"), header.get("offset"), header.get("length")
        if not isinstance(path, str) or type(offset) is not int or type(length) is not int:
            raise RpcError("bad_request", "path, offset and length are required")
        if offset < 0 or not 0 < length <= CHUNK_BYTES:
            raise RpcError("bad_request", f"offset >= 0 and 0 < length <= {CHUNK_BYTES}")
        served = self._model(header)
        entry = served.files.get(path)
        if entry is None:
            raise RpcError("not_found", "not a file of this model")
        if offset + length > entry.size:
            raise RpcError("bad_request", "past the end of the file")
        with self._lock:
            if self._reading >= MAX_SERVING:
                raise RpcError("busy", "serving too many reads right now")
            self._reading += 1
        try:
            # The manifest's own path, never the request's, is joined onto the directory.
            with (served.directory / entry.path).open("rb") as f:
                f.seek(offset)
                data = f.read(length)
        except OSError as exc:
            raise RpcError("not_found", "that file is no longer readable here") from exc
        finally:
            with self._lock:
                self._reading -= 1
        if len(data) != length:
            raise RpcError("not_found", "that file changed on this device")
        return {}, data


# --- fetching ---------------------------------------------------------------------------


@dataclass
class _Peer:
    peer_id: str
    failures: int = 0
    inflight: int = 0
    ready_at: float = 0.0
    busy_delay: float = _BUSY_FIRST_S


@dataclass
class _Session:
    """Everything one ``fetch()`` call knows; nothing outlives the call."""

    model: str
    peers: dict[str, _Peer]
    total: int
    on_progress: Callable[[int, int], None] | None
    done: int = 0
    dropped: set[str] = field(default_factory=set)

    def drop(self, peer: str, why: str) -> None:
        if self.peers.pop(peer, None) is not None:
            logger.info("dropping %s from this fetch: %s", peer[:8], why)
            self.dropped.add(peer)

    def progress(self, n: int) -> None:
        self.done = min(self.total, self.done + n)
        if self.on_progress is not None:
            try:
                self.on_progress(self.done, self.total)
            except Exception:  # noqa: BLE001 - a display callback must never stop a fetch
                logger.exception("progress callback failed")


class BlobFetcher:
    """Fetches a model from the devices that have it.

    Args:
        node: The overlay node to fetch over.
        timeout: Per-call timeout (seconds) for ``blob.get`` and ``blob.manifest``.
    """

    def __init__(self, node: Node, *, timeout: float = 30.0) -> None:
        self.node = node
        self.timeout = timeout

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
           ``PER_PEER_INFLIGHT`` chunks in flight per peer. ``busy`` makes a peer back
           off; any other failure, of any kind, puts the chunk back, and a peer that
           fails 3 times in a row is dropped.
        4. When a file is complete its SHA-256 is checked. On mismatch the file is
           fetched again, all of it from ONE other peer, keeping the first copy. If the
           new copy matches it is used, and every chunk where the two copies differ
           names its original supplier, who is dropped. A peer whose whole copy
           mismatches is dropped too. No peer left with a matching copy -> ``BlobError``.
        5. A matching file is moved into place with ``os.replace``.
        6. ``on_progress(done_bytes, total_bytes)`` after every chunk, never decreasing.

        Raises ``BlobError`` when no peers are left, a file never matches, or nothing
        arrives for ``4 * timeout`` seconds. Files that were completed stay in ``dest``
        (a later fetch resumes from them).
        """
        files, holders = self._manifest(model_id, expected_digest, peers)
        dest = Path(dest)
        todo = []
        for f in files:
            target = dest / f.path
            if (
                target.is_file()
                and target.stat().st_size == f.size
                and _hash_file(target) == f.sha256
            ):
                continue
            todo.append(f)
        session = _Session(
            model=model_id,
            peers={p: _Peer(p) for p in holders},
            total=sum(f.size for f in todo),
            on_progress=on_progress,
        )
        threads = min(_MAX_THREADS, PER_PEER_INFLIGHT * max(1, len(holders)))
        pool = ThreadPoolExecutor(max_workers=threads, thread_name_prefix="blob-fetch")
        try:
            for f in todo:
                self._fetch_file(session, pool, f, dest)
        finally:
            # In-flight calls end within their own timeout; nothing new is started.
            pool.shutdown(wait=False, cancel_futures=True)
        return files

    # --- steps ----------------------------------------------------------------------------

    def _manifest(
        self, model_id: str, expected: str, peers: Sequence[str]
    ) -> tuple[list[BlobFile], list[str]]:
        """The matching manifest, and the peers (in order) that offered exactly it."""
        files: list[BlobFile] | None = None
        holders: list[str] = []
        for peer in dict.fromkeys(peers):
            try:
                header, _ = self.node.call(
                    peer, MANIFEST, {"model": model_id}, timeout=self.timeout
                )
                offered = parse_manifest(header.get("files"))
            except Exception as exc:  # noqa: BLE001 - any failure: this peer cannot help
                logger.debug("%s offers no usable manifest: %s", peer[:8], exc)
                continue
            if manifest_digest(offered) == expected:
                files = files or offered
                holders.append(peer)
            else:
                logger.info("%s offers different files for %s; skipping it", peer[:8], model_id)
        if files is None:
            raise BlobError(f"no peer offered the expected files of {model_id}")
        return files, holders

    def _get(self, model: str, peer: str, f: BlobFile, offset: int, length: int) -> bytes:
        """One chunk from one peer; any exception is the caller's to judge."""
        _, body = self.node.call(
            peer,
            GET,
            {"model": model, "path": f.path, "offset": offset, "length": length},
            timeout=self.timeout,
        )
        if len(body) != length:
            raise RpcError("bad_request", f"asked for {length} bytes, got {len(body)}")
        return body

    def _fetch_file(self, s: _Session, pool: ThreadPoolExecutor, f: BlobFile, dest: Path) -> None:
        target = dest / f.path
        target.parent.mkdir(parents=True, exist_ok=True)
        part = target.with_name(target.name + ".part")
        with part.open("wb") as out:
            out.truncate(f.size)
        suppliers = self._fetch_chunks(s, pool, f, part)
        if _hash_file(part) != f.sha256:
            logger.warning("%s did not match its hash; fetching it again to find out why", f.path)
            self._recover(s, f, part, suppliers)
        os.replace(part, target)

    def _fetch_chunks(
        self, s: _Session, pool: ThreadPoolExecutor, f: BlobFile, part: Path
    ) -> dict[int, str]:
        """Fill ``part`` from all peers at once; returns ``offset -> supplier``.

        Only this thread changes state: worker threads just make calls, so a call that
        raises anything at all simply comes back as a failed future.
        """
        pending = collections.deque(range(0, f.size, CHUNK_BYTES))
        inflight: dict[Future, tuple[str, int]] = {}
        suppliers: dict[int, str] = {}
        last_progress = time.monotonic()
        with part.open("r+b") as out:
            while pending or inflight:
                now = time.monotonic()
                if not s.peers and not inflight:
                    raise BlobError(f"no peer left to fetch {f.path} from")
                if now - last_progress > 4 * self.timeout:
                    raise BlobError(f"nothing arrived for {f.path} in {4 * self.timeout:.0f}s")
                for peer in sorted(s.peers.values(), key=lambda p: p.inflight):
                    while pending and peer.inflight < PER_PEER_INFLIGHT and peer.ready_at <= now:
                        offset = pending.popleft()
                        length = min(CHUNK_BYTES, f.size - offset)
                        fut = pool.submit(self._get, s.model, peer.peer_id, f, offset, length)
                        inflight[fut] = (peer.peer_id, offset)
                        peer.inflight += 1
                if not inflight:
                    time.sleep(0.02)  # every peer is backing off
                    continue
                finished, _ = wait(list(inflight), timeout=0.2, return_when=FIRST_COMPLETED)
                for fut in finished:
                    peer_id, offset = inflight.pop(fut)
                    peer = s.peers.get(peer_id)
                    if peer is not None:
                        peer.inflight -= 1
                    try:
                        data = fut.result()
                    except Exception as exc:  # noqa: BLE001 - judged below, never fatal
                        pending.append(offset)
                        self._failed(s, peer, exc)
                        continue
                    out.seek(offset)
                    out.write(data)
                    suppliers[offset] = peer_id
                    last_progress = time.monotonic()
                    if peer is not None:
                        peer.failures, peer.busy_delay = 0, _BUSY_FIRST_S
                    s.progress(len(data))
        return suppliers

    @staticmethod
    def _failed(s: _Session, peer: _Peer | None, exc: BaseException) -> None:
        """Count a failed call: ``busy`` backs off, anything else counts toward a drop."""
        if peer is None:
            return
        if isinstance(exc, RpcError) and exc.code == "busy":
            peer.ready_at = time.monotonic() + peer.busy_delay
            peer.busy_delay = min(_BUSY_MAX_S, peer.busy_delay * 2)
            return
        peer.failures += 1
        logger.debug("chunk from %s failed (%d): %s", peer.peer_id[:8], peer.failures, exc)
        if peer.failures >= _MAX_FAILURES:
            s.drop(peer.peer_id, f"{_MAX_FAILURES} failures in a row ({exc})")

    def _recover(self, s: _Session, f: BlobFile, part: Path, suppliers: dict[int, str]) -> None:
        """Fetch ``f`` again from one peer at a time until a copy matches; then every chunk
        where the first copy differs names its supplier as a liar."""
        bad = part.with_name(part.name + ".bad")
        os.replace(part, bad)
        try:
            supplied = collections.Counter(suppliers.values())
            while True:
                candidates = sorted(s.peers, key=lambda p: supplied.get(p, 0))
                if not candidates:
                    raise BlobError(f"no peer left with a matching copy of {f.path}")
                source = candidates[0]
                differs = self._copy_from(s, f, source, part, bad)
                if differs is None:
                    continue  # that peer failed or was dropped; try the next
                if _hash_file(part) == f.sha256:
                    for offset in differs:
                        liar = suppliers.get(offset)
                        if liar is not None and liar != source:
                            s.drop(liar, f"sent wrong bytes for {f.path}")
                    return
                s.drop(source, f"its whole copy of {f.path} does not match")
        finally:
            bad.unlink(missing_ok=True)

    def _copy_from(
        self, s: _Session, f: BlobFile, source: str, part: Path, bad: Path
    ) -> list[int] | None:
        """All of ``f`` from ``source`` into ``part``; returns the offsets where it differs
        from ``bad``, or ``None`` if ``source`` could not supply it."""
        differs: list[int] = []
        last_progress = time.monotonic()
        with part.open("wb") as out, bad.open("rb") as old:
            for offset in range(0, f.size, CHUNK_BYTES):
                length = min(CHUNK_BYTES, f.size - offset)
                while True:
                    peer = s.peers.get(source)
                    if peer is None:
                        return None
                    if time.monotonic() - last_progress > 4 * self.timeout:
                        s.drop(source, "no progress while re-fetching")
                        return None
                    try:
                        data = self._get(s.model, source, f, offset, length)
                    except Exception as exc:  # noqa: BLE001 - judged like any chunk failure
                        self._failed(s, peer, exc)
                        time.sleep(max(0.0, peer.ready_at - time.monotonic()))
                        continue
                    break
                last_progress = time.monotonic()
                peer.failures = 0
                old.seek(offset)
                if old.read(length) != data:
                    differs.append(offset)
                out.write(data)
                s.progress(0)  # report liveness without counting bytes twice
        return differs
