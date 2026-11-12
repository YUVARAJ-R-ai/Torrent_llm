"""Getting the model onto a device: from its own disk, or from the swarm.

A device that already has the model (in the Hugging Face cache, or fetched from peers
before) uses it and offers it to others. A device that does not asks the devices it is
connected to, and fetches it from them (``blobs.py``), before it loads anything.

Which copy to trust:

- The pool config pins one (``model_manifest``): only that manifest is accepted.
- An invite swarm: its members were invited by someone you trust, so the first
  manifest offered is accepted (each file is still checked against it).
- The public swarm: strangers. A manifest is accepted only when at least
  ``PUBLIC_AGREEMENT`` different devices offer the same one; otherwise the device says
  to download the model from the internet instead.
"""

from __future__ import annotations

import logging
import re
import sys
import time
from collections import Counter
from collections.abc import Callable
from pathlib import Path

from torrent_llm.net import paths
from torrent_llm.net.node import Node
from torrent_llm.net.rpc import RpcError
from torrent_llm.pool.blobs import (
    MANIFEST,
    BlobError,
    BlobFetcher,
    BlobServer,
    manifest_digest,
    parse_manifest,
)

logger = logging.getLogger(__name__)

#: Devices that must offer the same manifest before a public-swarm device trusts it.
PUBLIC_AGREEMENT = 2
#: Devices asked for their manifest.
ASK_AT_MOST = 6
#: Marks a fetched model directory as complete (hidden, so it is never served).
COMPLETE = ".complete"


def shared_dir(model_id: str) -> Path:
    """Where a model fetched from peers lives: ``<data>/models/<org>--<name>``."""
    safe = re.sub(r"[^A-Za-z0-9._-]", "-", model_id.replace("/", "--"))[:200].lstrip(".")
    return paths.models_dir() / (safe or "model")


def local_model_dir(model_id: str) -> Path | None:
    """This device's copy of ``model_id``, or ``None`` when it has none."""
    fetched = shared_dir(model_id)
    if (fetched / COMPLETE).is_file():
        return fetched
    try:
        from huggingface_hub import snapshot_download

        found = Path(snapshot_download(model_id, local_files_only=True))
    except Exception:  # noqa: BLE001 - not cached, or not a hub id at all
        return None
    return found if (found / "config.json").is_file() else None


def serve_models(node: Node) -> BlobServer:
    """Offer every model this device has to the swarm."""
    return BlobServer(node, local_model_dir)


def ensure_model(
    node: Node,
    model_id: str,
    *,
    public: bool,
    pinned: str | None = None,
    wait_s: float = 20.0,
    on_progress: Callable[[int, int], None] | None = None,
) -> str:
    """What to load ``model_id`` from: a local directory, fetched from peers if needed.

    When the swarm cannot supply a copy this device may trust, ``model_id`` itself is
    returned, so loading falls back to downloading from the internet as before.

    Raises:
        RuntimeError: a fetch from the swarm started and failed.
    """
    found = local_model_dir(model_id)
    if found is not None:
        return str(found)
    offers = _ask_for_manifests(node, model_id, wait_s)
    try:
        expected = _choose(offers, public=public, pinned=pinned)
    except LookupError as exc:
        print(f"{exc}; downloading {model_id} from the internet instead.", flush=True)
        return model_id
    seeds = [peer for peer, digest in offers.items() if digest == expected]
    dest = shared_dir(model_id)
    print(f"Fetching {model_id} from {len(seeds)} device(s) in the swarm...", flush=True)
    try:
        BlobFetcher(node).fetch(
            model_id, expected, seeds, dest, on_progress=on_progress or _print_progress
        )
    except BlobError as exc:
        raise RuntimeError(f"could not fetch {model_id} from the swarm: {exc}") from exc
    (dest / COMPLETE).write_text(expected + "\n", encoding="utf-8")
    print("\nModel fetched and checked.", flush=True)
    return str(dest)


def _ask_for_manifests(node: Node, model_id: str, wait_s: float) -> dict[str, str]:
    """``peer -> manifest digest`` for every connected device that offers the model."""
    deadline = time.monotonic() + wait_s
    while not node.connected_peers() and time.monotonic() < deadline:
        time.sleep(0.25)
    offers: dict[str, str] = {}
    for peer in node.connected_peers()[:ASK_AT_MOST]:
        try:
            header, _ = node.call(peer, MANIFEST, {"model": model_id}, timeout=10.0)
            offers[peer] = manifest_digest(parse_manifest(header.get("files")))
        except (RpcError, ValueError) as exc:
            logger.debug("%s does not offer %s: %s", peer[:8], model_id, exc)
    return offers


def _choose(offers: dict[str, str], *, public: bool, pinned: str | None) -> str:
    """The manifest digest to fetch, by the rules in the module docstring.

    ``LookupError`` (with the reason) when no offer may be trusted.
    """
    if not offers:
        raise LookupError("no connected device offers the model")
    if pinned is not None:
        if pinned not in offers.values():
            raise LookupError("no connected device offers the exact files this pool requires")
        return pinned
    if not public:
        return next(iter(offers.values()))
    digest, votes = Counter(offers.values()).most_common(1)[0]
    if votes < PUBLIC_AGREEMENT:
        raise LookupError(
            f"only {votes} device(s) in the public swarm offer the same copy of the model "
            f"and {PUBLIC_AGREEMENT} must agree before strangers' files are trusted"
        )
    return digest


def _print_progress(done: int, total: int) -> None:
    if total:
        sys.stdout.write(f"\r  {done / 2**20:7.1f} / {total / 2**20:.1f} MiB")
        sys.stdout.flush()
