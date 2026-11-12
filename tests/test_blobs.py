"""Sharing model files between devices: manifests, serving, fetching, and liars."""

from __future__ import annotations

import hashlib
import os
import random
import threading
import time
from pathlib import Path

import pytest

from torrent_llm.net.identity import Identity
from torrent_llm.net.node import Node, NodeConfig
from torrent_llm.net.rpc import RpcError
from torrent_llm.pool import blobs
from torrent_llm.pool.blobs import (
    GET,
    MANIFEST,
    BlobError,
    BlobFetcher,
    BlobFile,
    BlobServer,
    manifest_digest,
    manifest_of,
    parse_manifest,
)

CHUNK = 64 * 1024


@pytest.fixture(autouse=True)
def small_chunks(monkeypatch):
    monkeypatch.setattr(blobs, "CHUNK_BYTES", CHUNK)


def _node() -> Node:
    node = Node(NodeConfig(identity=Identity.generate(), listen_host="127.0.0.1"))
    node.start()
    return node


def _model_dir(root: Path) -> Path:
    """A small model: a config, two weight files (one over many chunks), a nested file."""
    rng = random.Random(7)
    root.mkdir(parents=True)
    (root / "config.json").write_bytes(b'{"layers": 2}')
    (root / "model-1.safetensors").write_bytes(rng.randbytes(CHUNK * 9 + 123))
    (root / "model-2.safetensors").write_bytes(rng.randbytes(CHUNK * 3))
    (root / "sub").mkdir()
    (root / "sub" / "tokenizer.json").write_bytes(rng.randbytes(5000))
    (root / "empty.txt").write_bytes(b"")
    return root


class Swarm:
    """One fetching node and some seeds, each serving ``model``."""

    def __init__(self, tmp_path: Path, seeds: int) -> None:
        self.source = _model_dir(tmp_path / "source")
        self.files = manifest_of(self.source)
        self.digest = manifest_digest(self.files)
        self.me = _node()
        self.seeds = [_node() for _ in range(seeds)]
        self.servers = [BlobServer(n, self.locate) for n in self.seeds]
        for seed in self.seeds:
            self.me.add_record(seed.record())
        self.dest = tmp_path / "dest"

    def locate(self, model: str) -> Path | None:
        return self.source if model == "m" else None

    def ids(self) -> list[str]:
        return [n.peer_id for n in self.seeds]

    def same_as_source(self) -> bool:
        return all(
            (self.dest / f.path).read_bytes() == (self.source / f.path).read_bytes()
            for f in self.files
        )

    def close(self) -> None:
        for s in self.servers:
            s.close()
        for n in [self.me, *self.seeds]:
            n.close()


@pytest.fixture
def swarm(tmp_path):
    made: list[Swarm] = []

    def make(seeds: int) -> Swarm:
        made.append(Swarm(tmp_path / str(len(made)), seeds))
        return made[-1]

    yield make
    for s in made:
        s.close()


# --- manifests --------------------------------------------------------------------------


def test_manifest_lists_model_files_sorted_and_skips_the_rest(tmp_path):
    root = _model_dir(tmp_path / "m")
    (root / "README.md").write_text("hi")
    (root / ".git").mkdir()
    (root / ".git" / "config.json").write_text("{}")
    (root / ".hidden.json").write_text("{}")
    if os.name == "posix":
        (root / "linked.json").symlink_to(root / "config.json")
    files = manifest_of(root)
    paths = [f.path for f in files]
    assert paths == sorted(paths)
    assert "README.md" not in paths and ".hidden.json" not in paths
    assert not any(p.startswith(".git") for p in paths)
    assert "sub/tokenizer.json" in paths
    config = next(f for f in files if f.path == "config.json")
    assert config.sha256 == hashlib.sha256(b'{"layers": 2}').hexdigest()
    if os.name == "posix":
        assert "linked.json" in paths
    assert manifest_digest(files) == manifest_digest(list(reversed(files)))


GOOD = "a" * 64


@pytest.mark.parametrize(
    "path",
    [
        "../x",
        "/abs",
        "a//b",
        "a\\b",
        "a/./b",
        "C:x",
        "a:stream",
        "CON",
        "nul.json",
        "Com1.txt",
        "a./b",
        "trail ",
        "x\0y",
        "",
        "x" * 513,
    ],
)
def test_unsafe_paths_are_refused(path):
    with pytest.raises(ValueError):
        parse_manifest([[path, 1, GOOD]])


def test_bad_entries_are_refused():
    for raw in (
        "x",
        [["a", True, GOOD]],
        [["a", -1, GOOD]],
        [["a", 1, "A" * 64]],
        [["a", 1]],
        [["a", 1, GOOD], ["a", 2, GOOD]],
        [["A.bin", 1, GOOD], ["a.bin", 2, GOOD]],
        [["x", 1, GOOD], ["x/y", 2, GOOD]],
        [["f", 1, GOOD]] * 2,
        [[f"f{i}", 1, GOOD] for i in range(blobs.MAX_FILES + 1)],
    ):
        with pytest.raises(ValueError):
            parse_manifest(raw)
    assert parse_manifest([["a/b.json", 3, GOOD], ["c.bin", 0, GOOD]]) == [
        BlobFile("a/b.json", 3, GOOD),
        BlobFile("c.bin", 0, GOOD),
    ]


# --- serving ----------------------------------------------------------------------------


def test_the_server_reads_only_manifest_paths_within_bounds(swarm):
    s = swarm(1)
    seed = s.ids()[0]

    def get(**header):
        return s.me.call(seed, GET, {"model": "m", **header}, timeout=5)

    header, _ = s.me.call(seed, MANIFEST, {"model": "m"}, timeout=5)
    assert manifest_digest(parse_manifest(header["files"])) == s.digest
    assert get(path="config.json", offset=0, length=5)[1] == b'{"lay'
    for bad, code in (
        (dict(path="../../etc/passwd", offset=0, length=5), "not_found"),
        (dict(path="README.md", offset=0, length=5), "not_found"),
        (dict(path="config.json", offset=0, length=0), "bad_request"),
        (dict(path="config.json", offset=0, length=CHUNK + 1), "bad_request"),
        (dict(path="config.json", offset=10, length=10), "bad_request"),
        (dict(path="config.json", offset=-1, length=1), "bad_request"),
        (dict(path="config.json", offset=True, length=1), "bad_request"),
    ):
        with pytest.raises(RpcError) as caught:
            get(**bad)
        assert caught.value.code == code, bad
    with pytest.raises(RpcError) as caught:
        s.me.call(seed, MANIFEST, {"model": "other"}, timeout=5)
    assert caught.value.code == "not_found"


def test_the_server_says_busy_past_its_limit(swarm, monkeypatch):
    s = swarm(1)
    monkeypatch.setattr(blobs, "MAX_SERVING", 0)
    with pytest.raises(RpcError) as caught:
        s.me.call(
            s.ids()[0], GET, {"model": "m", "path": "config.json", "offset": 0, "length": 2},
            timeout=5,
        )  # fmt: skip
    assert caught.value.code == "busy"


# --- fetching ---------------------------------------------------------------------------


def test_a_fetch_from_three_seeds_is_exact_and_uses_all_of_them(swarm):
    s = swarm(3)
    served: dict[str, int] = {}
    real_call = s.me.call

    def counting(peer, method, *args, **kwargs):
        if method == GET:
            served[peer] = served.get(peer, 0) + 1
        return real_call(peer, method, *args, **kwargs)

    s.me.call = counting
    reports: list[tuple[int, int]] = []
    files = BlobFetcher(s.me).fetch(
        "m", s.digest, s.ids(), s.dest, on_progress=lambda d, t: reports.append((d, t))
    )
    assert files == s.files
    assert s.same_as_source()
    assert (s.dest / "empty.txt").read_bytes() == b""
    assert set(served) == set(s.ids())
    assert reports[-1][0] == reports[-1][1] == sum(f.size for f in s.files)
    assert [d for d, _ in reports] == sorted(d for d, _ in reports)
    assert not list(s.dest.rglob("*.part"))


@pytest.mark.parametrize("liar_first", [True, False])
def test_a_liar_is_named_and_dropped_whatever_the_order(swarm, liar_first):
    s = swarm(2)
    liar, honest = (s.ids()[0], s.ids()[1]) if liar_first else (s.ids()[1], s.ids()[0])
    real_call = s.me.call
    lies: list[int] = []

    def lying(peer, method, header=None, body=b"", *, timeout):
        reply_header, reply_body = real_call(peer, method, header, body, timeout=timeout)
        if peer == liar and method == GET:
            lies.append(1)
            reply_body = bytes([reply_body[0] ^ 0xFF]) + reply_body[1:]
        return reply_header, reply_body

    s.me.call = lying
    fetched_from: list[str] = []
    real_get = BlobFetcher._get

    def tracking(self, model, peer, f, offset, length):
        fetched_from.append(peer)
        return real_get(self, model, peer, f, offset, length)

    fetcher = BlobFetcher(s.me)
    fetcher._get = tracking.__get__(fetcher)
    fetcher.fetch("m", s.digest, s.ids(), s.dest)
    assert s.same_as_source()
    assert lies, "the liar must have served something for this test to mean anything"
    # Once named, the liar serves nothing more: its last chunk comes before the end.
    last_from_liar = max(i for i, p in enumerate(fetched_from) if p == liar)
    assert last_from_liar < len(fetched_from) - 1
    assert fetched_from[-1] == honest


def test_busy_answers_back_off_and_never_drop_a_lone_seed(swarm):
    s = swarm(1)
    real_call = s.me.call
    count = [0]

    def sometimes_busy(peer, method, *args, **kwargs):
        if method == GET:
            count[0] += 1
            if count[0] % 3 == 0:
                raise RpcError("busy", "full")
        return real_call(peer, method, *args, **kwargs)

    s.me.call = sometimes_busy
    BlobFetcher(s.me).fetch("m", s.digest, s.ids(), s.dest)
    assert s.same_as_source()


@pytest.mark.parametrize(
    "error",
    [TimeoutError("loop stalled"), RuntimeError("node is not running"), ConnectionResetError()],
)
def test_any_exception_from_a_call_is_survivable(swarm, error):
    s = swarm(2)
    real_call = s.me.call
    raised = [False]

    def once(peer, method, *args, **kwargs):
        if method == GET and not raised[0]:
            raised[0] = True
            raise error
        return real_call(peer, method, *args, **kwargs)

    s.me.call = once
    BlobFetcher(s.me).fetch("m", s.digest, s.ids(), s.dest)
    assert raised[0] and s.same_as_source()


def test_a_peer_that_always_fails_ends_in_blob_error_quickly(swarm):
    s = swarm(1)
    real_call = s.me.call

    def broken(peer, method, *args, **kwargs):
        if method == GET:
            raise RuntimeError("node is not running")
        return real_call(peer, method, *args, **kwargs)

    s.me.call = broken
    t0 = time.monotonic()
    with pytest.raises(BlobError, match="no peer left"):
        BlobFetcher(s.me, timeout=2).fetch("m", s.digest, s.ids(), s.dest)
    assert time.monotonic() - t0 < 5


def test_a_seed_that_never_answers_is_a_blob_error_not_a_hang(swarm):
    s = swarm(1)
    real_call = s.me.call

    def forever_busy(peer, method, *args, **kwargs):
        if method == GET:
            raise RpcError("busy", "full")
        return real_call(peer, method, *args, **kwargs)

    s.me.call = forever_busy
    t0 = time.monotonic()
    with pytest.raises(BlobError, match="nothing arrived"):
        BlobFetcher(s.me, timeout=0.5).fetch("m", s.digest, s.ids(), s.dest)
    assert time.monotonic() - t0 < 6


def test_a_failing_progress_callback_does_not_stop_the_fetch(swarm):
    s = swarm(1)

    def explode(done, total):
        raise ValueError("display broke")

    BlobFetcher(s.me).fetch("m", s.digest, s.ids(), s.dest, on_progress=explode)
    assert s.same_as_source()


def test_complete_files_are_kept_and_not_fetched_again(swarm):
    s = swarm(1)
    BlobFetcher(s.me).fetch("m", s.digest, s.ids(), s.dest)
    real_call = s.me.call
    gets = [0]

    def counting(peer, method, *args, **kwargs):
        gets[0] += method == GET
        return real_call(peer, method, *args, **kwargs)

    s.me.call = counting
    (s.dest / "config.json").write_bytes(b"tampered")
    BlobFetcher(s.me).fetch("m", s.digest, s.ids(), s.dest)
    assert gets[0] == 1  # only the one damaged (small) file
    assert s.same_as_source()


def test_wrong_manifests_and_missing_peers_are_blob_errors(swarm, tmp_path):
    s = swarm(1)
    with pytest.raises(BlobError):
        BlobFetcher(s.me).fetch("m", "0" * 64, s.ids(), s.dest)
    with pytest.raises(BlobError):
        BlobFetcher(s.me).fetch("m", s.digest, [], s.dest)
    with pytest.raises(BlobError):
        BlobFetcher(s.me).fetch("other", s.digest, s.ids(), s.dest)
    # The same fetcher still works afterwards: nothing carries over between calls.
    fetcher = BlobFetcher(s.me)
    with pytest.raises(BlobError):
        fetcher.fetch("m", "0" * 64, s.ids(), s.dest)
    fetcher.fetch("m", s.digest, s.ids(), s.dest)
    assert s.same_as_source()


def test_two_fetches_on_one_fetcher_at_once(swarm, tmp_path):
    s = swarm(2)
    fetcher = BlobFetcher(s.me)
    errors: list[BaseException] = []

    def run(dest: Path) -> None:
        try:
            fetcher.fetch("m", s.digest, s.ids(), dest)
        except BaseException as exc:  # noqa: BLE001 - reported below
            errors.append(exc)

    threads = [threading.Thread(target=run, args=(tmp_path / f"d{i}",)) for i in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(30)
    assert not errors
    for i in range(2):
        s.dest = tmp_path / f"d{i}"
        assert s.same_as_source()
