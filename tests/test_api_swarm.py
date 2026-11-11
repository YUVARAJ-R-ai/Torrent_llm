"""Tests for the swarm routes on the local control API."""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from fastapi import FastAPI
from fastapi.testclient import TestClient

from torrent_llm.net.node import ConnectionInfo
from torrent_llm.pool.api_swarm import add_swarm_routes
from torrent_llm.pool.router import NoReplica
from torrent_llm.pool.runner import PoolUnavailable
from torrent_llm.pool.trust import Reputation


class FakeTokenizer:
    """Trivial character-level tokenizer for tests."""

    def encode(self, text: str) -> list[int]:
        """Encode text into ASCII code points."""
        return [ord(c) % 256 for c in text]

    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        """Decode ASCII code points into text."""
        return "".join(chr(int(i) % 256) for i in ids)


class ChatFakeTokenizer(FakeTokenizer):
    """FakeTokenizer plus a visible chat template."""

    def apply_chat_template(self, messages: Any, **kwargs: Any) -> str:
        """Format chat turns into delimited text."""
        return "".join(f"<{m['role']}:{m['content']}>" for m in messages) + "<assistant:"


class Utf8ByteTokenizer:
    """UTF-8 byte tokenizer to test multi-byte token decoding boundaries."""

    def encode(self, text: str) -> list[int]:
        """Encode text as UTF-8 byte values."""
        return list(text.encode("utf-8"))

    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        """Decode byte values to text, ignoring partial UTF-8 sequences."""
        return bytes(ids).decode("utf-8", errors="ignore")


class FakeRunner:
    """Fake runner simulating generation and error handling."""

    def __init__(
        self,
        token_ids: list[int] | None = None,
        raise_exc: Exception | None = None,
    ) -> None:
        """Initialize fake runner."""
        self.token_ids = token_ids or [ord("A"), ord("B"), ord("C")]
        self.raise_exc = raise_exc

    def generate(
        self,
        input_ids: torch.Tensor,
        *,
        max_new_tokens: int = 16,
        stop_token_ids: Any = (),
        sampling: Any = None,
        on_token: Any = None,
    ) -> Any:
        """Simulate generation pass and trigger on_token callback."""
        if self.raise_exc is not None:
            raise self.raise_exc
        for tid in self.token_ids:
            if on_token is not None:
                on_token(tid)
        out_ids = torch.cat([input_ids, torch.tensor([self.token_ids], dtype=torch.long)], dim=1)
        return SimpleNamespace(
            ids=out_ids,
            passes=[],
            events=[],
            failovers=0,
        )


class FakeDaemon:
    """Fake PeerDaemon providing runner, labels, and config."""

    def __init__(self, name: str = "node-self", runner: Any = None) -> None:
        """Initialize fake daemon."""
        self.name = name
        self.config = SimpleNamespace(signature="test-sig")
        self.runner = runner or FakeRunner()
        self.asked_counts: list[int] = []

    def label_of(self, peer_id: str) -> str:
        """Return a human-readable label for a peer."""
        if peer_id == self.name:
            return "SelfNode"
        return f"Label-{peer_id[:4]}"

    def record_asked(self, count: int) -> None:
        """Record number of generated tokens."""
        self.asked_counts.append(count)


class FakeNode:
    """Fake network node tracking connections and disconnections."""

    def __init__(self, peer_id: str) -> None:
        """Initialize fake node."""
        self.peer_id = peer_id
        self.active_connections: list[ConnectionInfo] = []
        self.advertised_addresses: list[str] = ["tcp://127.0.0.1:4000"]
        self.disconnected: list[str] = []

    def addresses(self) -> list[str]:
        """Return advertised addresses."""
        return list(self.advertised_addresses)

    def connections(self) -> list[ConnectionInfo]:
        """Return active connection list."""
        return list(self.active_connections)

    def disconnect(self, peer_id: str) -> None:
        """Record disconnected peer."""
        self.disconnected.append(peer_id)


class FakeDevice:
    """Fake swarm device containing node, spec, and services."""

    def __init__(
        self,
        node: FakeNode,
        *,
        swarm_id: str = "test-swarm",
        name: str = "Test Swarm",
        public: bool = False,
        reachability_status: str = "public",
        mapped: str = "tcp://1.2.3.4:4000",
        relays: list[str] | None = None,
        serving_relay: bool = False,
        invite_link: str | None = "tllm://join/testlink123",
    ) -> None:
        """Initialize fake swarm device."""
        self.node = node
        self.spec = SimpleNamespace(swarm_id=swarm_id, name=name, public=public)
        self.services = SimpleNamespace(
            reachability=SimpleNamespace(
                status=reachability_status,
                mapped=mapped,
                relays=relays or ["tcp://9.9.9.9:4000"],
                serving_relay=serving_relay,
            )
        )
        self.invite_link = invite_link

    def invite(self, config: Any) -> Any:
        """Return an invite object or None."""
        if self.spec.public or self.invite_link is None:
            return None
        return SimpleNamespace(to_link=lambda: self.invite_link)


class FakeTrust:
    """Fake trust store managing reputation, bans, and unbans."""

    def __init__(self) -> None:
        """Initialize fake trust store."""
        self.records: dict[str, Reputation] = {}
        self.bans: list[tuple[str, str]] = []
        self.unbans: list[str] = []

    def all(self) -> list[Reputation]:
        """Return all reputation entries sorted by peer_id."""
        return [self.records[k] for k in sorted(self.records)]

    def ban(self, peer_id: str, reason: str) -> None:
        """Record a ban for peer_id."""
        self.bans.append((peer_id, reason))
        old = self.records.get(peer_id, Reputation(peer_id=peer_id))
        self.records[peer_id] = Reputation(
            peer_id=peer_id,
            checks=old.checks,
            failures=old.failures,
            banned=True,
            reason=reason,
            served_ms=old.served_ms,
            used_ms=old.used_ms,
        )

    def unban(self, peer_id: str) -> None:
        """Record an unban for peer_id."""
        self.unbans.append(peer_id)
        old = self.records.get(peer_id, Reputation(peer_id=peer_id))
        self.records[peer_id] = Reputation(
            peer_id=peer_id,
            checks=old.checks,
            failures=old.failures,
            banned=False,
            reason="",
            served_ms=old.served_ms,
            used_ms=old.used_ms,
        )


class FakeLinkEstimator:
    """Fake link estimator returning bandwidth per peer."""

    def __init__(self) -> None:
        """Initialize fake link estimator."""
        self.bandwidths: dict[str, float] = {}

    def bandwidth(self, peer_id: str) -> float | None:
        """Return configured bandwidth for peer_id."""
        return self.bandwidths.get(peer_id)


def parse_sse_events(text: str) -> list[tuple[str, dict[str, Any]]]:
    """Parse raw Server-Sent Events text into event and payload pairs."""
    events: list[tuple[str, dict[str, Any]]] = []
    blocks = [b.strip() for b in text.strip().split("\n\n") if b.strip()]
    for block in blocks:
        event_name = ""
        data_text = ""
        for line in block.split("\n"):
            if line.startswith("event: "):
                event_name = line[len("event: ") :].strip()
            elif line.startswith("data: "):
                data_text = line[len("data: ") :].strip()
        if event_name and data_text:
            events.append((event_name, json.loads(data_text)))
    return events


def create_test_client(
    *,
    admin_token: str = "secret-admin-token",
    device: FakeDevice | None = None,
    daemon: FakeDaemon | None = None,
    trust: FakeTrust | None = None,
    estimator: FakeLinkEstimator | None = None,
    tokenizer: Any = None,
) -> tuple[TestClient, dict[str, Any]]:
    """Build a TestClient wired to add_swarm_routes with test fakes."""
    app = FastAPI()
    self_peer = "a" * 32
    node = FakeNode(self_peer)
    dev = device or FakeDevice(node)
    dae = daemon or FakeDaemon(name=self_peer)
    tru = trust or FakeTrust()
    tok = tokenizer or FakeTokenizer()

    add_swarm_routes(
        app,
        daemon=dae,
        device=dev,
        trust=tru,
        admin_token=admin_token,
        tokenizer=tok,
        link_estimator=estimator,
    )
    client = TestClient(app)
    context = {
        "admin_token": admin_token,
        "device": dev,
        "daemon": dae,
        "trust": tru,
        "estimator": estimator,
        "tokenizer": tok,
    }
    return client, context


def test_admin_token_empty_rejected() -> None:
    """Verify add_swarm_routes raises ValueError when admin_token is empty."""
    app = FastAPI()
    node = FakeNode("a" * 32)
    with pytest.raises(ValueError, match="needs a non-empty admin token"):
        add_swarm_routes(
            app,
            daemon=FakeDaemon(name="a" * 32),
            device=FakeDevice(node),
            trust=FakeTrust(),
            admin_token="",
            tokenizer=FakeTokenizer(),
        )


def test_node_shape() -> None:
    """Verify the /node route returns the expected NodeView schema and values."""
    client, ctx = create_test_client()
    res = client.get("/node")
    assert res.status_code == 200
    body = res.json()
    assert body["peer_id"] == "a" * 32
    assert body["label"] == "SelfNode"
    assert body["swarm_id"] == "test-swarm"
    assert body["swarm_name"] == "Test Swarm"
    assert body["public"] is False
    assert body["addresses"] == ["tcp://127.0.0.1:4000"]
    assert body["reachability"] == "public"
    assert body["mapped"] == "tcp://1.2.3.4:4000"
    assert body["relays"] == ["tcp://9.9.9.9:4000"]
    assert body["serving_relay"] is False
    assert body["connections"] == 0
    assert body["version"] == "0.1.0"


def test_peers_labels_and_bandwidth() -> None:
    """Verify the /peers route returns sorted peers with correct labels and bandwidth."""
    peer_b = "b" * 32
    peer_c = "c" * 32
    estimator = FakeLinkEstimator()
    estimator.bandwidths[peer_b] = 1_500_000.0

    client, ctx = create_test_client(estimator=estimator)
    device: FakeDevice = ctx["device"]
    device.node.active_connections = [
        ConnectionInfo(
            peer_id=peer_c,
            remote="192.168.1.10:5000",
            inbound=False,
            rtt_ms=12.5,
            bytes_sent=100,
            bytes_received=200,
        ),
        ConnectionInfo(
            peer_id=peer_b,
            remote="192.168.1.11:5000",
            inbound=True,
            rtt_ms=25.0,
            bytes_sent=300,
            bytes_received=400,
            via_relay="tcp://relay.local:4000",
        ),
    ]

    res = client.get("/peers")
    assert res.status_code == 200
    peers = res.json()
    assert len(peers) == 2
    # Sorted by (label, peer_id)
    assert peers[0]["peer_id"] == peer_b
    assert peers[0]["label"] == "Label-bbbb"
    assert peers[0]["bandwidth_bps"] == 1_500_000.0
    assert peers[0]["via_relay"] == "tcp://relay.local:4000"
    assert peers[0]["inbound"] is True

    assert peers[1]["peer_id"] == peer_c
    assert peers[1]["label"] == "Label-cccc"
    assert peers[1]["bandwidth_bps"] is None
    assert peers[1]["inbound"] is False


def test_peers_without_link_estimator() -> None:
    """Verify the /peers route returns None for bandwidth when no estimator is passed."""
    client, ctx = create_test_client(estimator=None)
    device: FakeDevice = ctx["device"]
    peer = "b" * 32
    device.node.active_connections = [
        ConnectionInfo(
            peer_id=peer,
            remote="192.168.1.10:5000",
            inbound=False,
            rtt_ms=10.0,
            bytes_sent=50,
            bytes_received=60,
        )
    ]
    res = client.get("/peers")
    assert res.status_code == 200
    assert res.json()[0]["bandwidth_bps"] is None


def test_trust_view() -> None:
    """Verify the /trust route returns reputation records with calculated scores."""
    trust = FakeTrust()
    peer_b = "b" * 32
    trust.records[peer_b] = Reputation(
        peer_id=peer_b,
        checks=5,
        failures=1,
        banned=False,
        reason="",
        served_ms=120.0,
        used_ms=240.0,
    )

    client, ctx = create_test_client(trust=trust)
    res = client.get("/trust")
    assert res.status_code == 200
    rows = res.json()
    assert len(rows) == 1
    row = rows[0]
    assert row["peer_id"] == peer_b
    assert row["label"] == "Label-bbbb"
    assert row["checks"] == 5
    assert row["failures"] == 1
    assert abs(row["score"] - ((5 - 1 + 1) / (5 + 1))) < 1e-6
    assert row["banned"] is False
    assert row["reason"] == ""
    assert row["served_ms"] == 120.0
    assert row["used_ms"] == 240.0


def test_ban_unban_admin_token() -> None:
    """Verify ban and unban endpoints require a valid admin token."""
    client, ctx = create_test_client(admin_token="my-token")
    peer = "b" * 32

    # Without token
    res = client.post("/trust/ban", json={"peer_id": peer, "reason": "misbehaving"})
    assert res.status_code == 401
    assert res.json() == {"detail": "this needs the admin token"}

    res = client.post("/trust/unban", json={"peer_id": peer})
    assert res.status_code == 401
    assert res.json() == {"detail": "this needs the admin token"}

    # With wrong token
    bad_headers = {"x-admin-token": "wrong-token"}
    res = client.post(
        "/trust/ban",
        json={"peer_id": peer, "reason": "misbehaving"},
        headers=bad_headers,
    )
    assert res.status_code == 401
    assert res.json() == {"detail": "this needs the admin token"}

    res = client.post(
        "/trust/unban",
        json={"peer_id": peer},
        headers=bad_headers,
    )
    assert res.status_code == 401
    assert res.json() == {"detail": "this needs the admin token"}

    # With non-ASCII token (bytes decoded by Starlette as latin-1)
    non_ascii_headers = {"x-admin-token": b"caf\xe9"}
    res = client.post(
        "/trust/ban",
        json={"peer_id": peer, "reason": "misbehaving"},
        headers=non_ascii_headers,
    )
    assert res.status_code == 401
    assert res.json() == {"detail": "this needs the admin token"}


def test_ban_unban_validation() -> None:
    """Verify 422 errors for bad peer ids or empty reasons."""
    client, ctx = create_test_client(admin_token="my-token")
    headers = {"x-admin-token": "my-token"}
    valid_peer = "b" * 32

    # Bad peer id in ban
    res = client.post(
        "/trust/ban",
        json={"peer_id": "not-valid", "reason": "misbehaving"},
        headers=headers,
    )
    assert res.status_code == 422
    assert res.json() == {"detail": "not a peer id"}

    # Bad peer id in unban
    res = client.post(
        "/trust/unban",
        json={"peer_id": "not-valid"},
        headers=headers,
    )
    assert res.status_code == 422
    assert res.json() == {"detail": "not a peer id"}

    # Empty reason in ban
    res = client.post(
        "/trust/ban",
        json={"peer_id": valid_peer, "reason": ""},
        headers=headers,
    )
    assert res.status_code == 422

    # Whitespace only reason in ban
    res = client.post(
        "/trust/ban",
        json={"peer_id": valid_peer, "reason": "    "},
        headers=headers,
    )
    assert res.status_code == 422

    # Reason exceeding 200 characters
    res = client.post(
        "/trust/ban",
        json={"peer_id": valid_peer, "reason": "x" * 201},
        headers=headers,
    )
    assert res.status_code == 422


def test_ban_unban_success() -> None:
    """Verify successful ban calls trust.ban and disconnects peer, and unban unbans."""
    client, ctx = create_test_client(admin_token="my-token")
    headers = {"x-admin-token": "my-token"}
    trust: FakeTrust = ctx["trust"]
    device: FakeDevice = ctx["device"]
    peer = "b" * 32

    # Successful ban
    res = client.post(
        "/trust/ban",
        json={"peer_id": peer, "reason": "bad results"},
        headers=headers,
    )
    assert res.status_code == 200
    assert res.json() == {"banned": True}
    assert (peer, "bad results") in trust.bans
    assert peer in device.node.disconnected

    # Successful unban
    res = client.post(
        "/trust/unban",
        json={"peer_id": peer},
        headers=headers,
    )
    assert res.status_code == 200
    assert res.json() == {"banned": False}
    assert peer in trust.unbans


def test_ban_disconnect_failure_ignored() -> None:
    """Verify ban succeeds with 200 even if node.disconnect raises an error."""
    client, ctx = create_test_client(admin_token="my-token")
    headers = {"x-admin-token": "my-token"}
    device: FakeDevice = ctx["device"]
    peer = "b" * 32

    def broken_disconnect(pid: str) -> None:
        """Simulate disconnect raising an error."""
        raise RuntimeError("disconnect socket lost")

    device.node.disconnect = broken_disconnect  # type: ignore[method-assign]
    res = client.post(
        "/trust/ban",
        json={"peer_id": peer, "reason": "bad results"},
        headers=headers,
    )
    assert res.status_code == 200
    assert res.json() == {"banned": True}
    assert (peer, "bad results") in ctx["trust"].bans


def test_swarm_invite() -> None:
    """Verify /swarm/invite requires token, returns link and commands, and 404 in public swarm."""
    client, ctx = create_test_client(admin_token="my-token")
    device: FakeDevice = ctx["device"]

    # Without token
    res = client.get("/swarm/invite")
    assert res.status_code == 401
    assert res.json() == {"detail": "this needs the admin token"}

    # With wrong token
    res = client.get("/swarm/invite", headers={"x-admin-token": "wrong-token"})
    assert res.status_code == 401
    assert res.json() == {"detail": "this needs the admin token"}

    # Valid token in private swarm
    res = client.get("/swarm/invite", headers={"x-admin-token": "my-token"})
    assert res.status_code == 200
    body = res.json()
    assert body["link"] == "tllm://join/testlink123"
    assert "linux_macos" in body["commands"]
    assert "windows" in body["commands"]
    assert "installed" in body["commands"]
    assert "testlink123" in body["commands"]["installed"]

    # Public swarm -> 404
    device.spec.public = True
    res = client.get("/swarm/invite", headers={"x-admin-token": "my-token"})
    assert res.status_code == 404
    assert res.json() == {
        "detail": "the public swarm has no invite: anyone can join with torrent-peer public"
    }


def test_generate_stream_success() -> None:
    """Verify /generate/stream yields 3 token events and 1 done event in order."""
    token_ids = [ord("X"), ord("Y"), ord("Z")]
    runner = FakeRunner(token_ids=token_ids)
    daemon = FakeDaemon(runner=runner)
    client, ctx = create_test_client(daemon=daemon)

    res = client.post(
        "/generate/stream",
        json={"prompt": "test prompt", "chat": False, "max_new_tokens": 10},
    )
    assert res.status_code == 200
    assert "text/event-stream" in res.headers["content-type"]
    assert res.headers["cache-control"] == "no-cache"

    events = parse_sse_events(res.text)
    assert len(events) == 4

    # 3 token events in order
    assert events[0][0] == "token"
    assert events[0][1] == {"text": "X", "index": 0}

    assert events[1][0] == "token"
    assert events[1][1] == {"text": "Y", "index": 1}

    assert events[2][0] == "token"
    assert events[2][1] == {"text": "Z", "index": 2}

    # 1 done event
    assert events[3][0] == "done"
    done_payload = events[3][1]
    assert done_payload["prompt"] == "test prompt"
    assert done_payload["completion"] == "XYZ"
    assert done_payload["tokens"] == ["X", "Y", "Z"]
    assert done_payload["requester"] == daemon.name
    assert done_payload["totals"]["tokens"] == 3
    assert daemon.asked_counts == [3]


def test_generate_stream_utf8_prefix_difference() -> None:
    """Verify streaming UTF-8 bytes for 'é€' matches the prefix-difference rule."""
    text = "é€"
    tok = Utf8ByteTokenizer()
    token_ids = tok.encode(text)
    runner = FakeRunner(token_ids=token_ids)
    daemon = FakeDaemon(runner=runner)
    client, _ = create_test_client(daemon=daemon, tokenizer=tok)

    res = client.post(
        "/generate/stream",
        json={"prompt": "prompt", "chat": False, "max_new_tokens": 10},
    )
    assert res.status_code == 200
    events = parse_sse_events(res.text)
    assert len(events) == len(token_ids) + 1

    token_events = [e for e in events if e[0] == "token"]
    done_event = [e for e in events if e[0] == "done"][0]

    streamed_texts = [e[1]["text"] for e in token_events]
    assert streamed_texts == done_event[1]["tokens"]
    assert streamed_texts == ["", "é", "", "", "€"]
    assert done_event[1]["completion"] == "é€"
    assert daemon.asked_counts == [len(token_ids)]


def test_generate_stream_pool_unavailable() -> None:
    """Verify /generate/stream yields an error event on PoolUnavailable."""
    runner = FakeRunner(raise_exc=PoolUnavailable("no copies for stage 1"))
    daemon = FakeDaemon(runner=runner)
    client, ctx = create_test_client(daemon=daemon)

    res = client.post(
        "/generate/stream",
        json={"prompt": "test prompt", "chat": False},
    )
    assert res.status_code == 200
    events = parse_sse_events(res.text)
    assert len(events) == 1
    assert events[0][0] == "error"
    assert events[0][1] == {"detail": "no copies for stage 1"}


def test_generate_stream_no_replica() -> None:
    """Verify /generate/stream yields an error event on NoReplica."""
    runner = FakeRunner(raise_exc=NoReplica(stage=1))
    daemon = FakeDaemon(runner=runner)
    client, _ = create_test_client(daemon=daemon)

    res = client.post(
        "/generate/stream",
        json={"prompt": "test prompt", "chat": False},
    )
    assert res.status_code == 200
    events = parse_sse_events(res.text)
    assert len(events) == 1
    assert events[0][0] == "error"
    assert events[0][1] == {"detail": str(NoReplica(stage=1))}


def test_generate_stream_chat_rendering() -> None:
    """Verify chat=True renders prompt through chat template before encoding."""
    tok = ChatFakeTokenizer()
    runner = FakeRunner(token_ids=[ord("A"), ord("B")])
    daemon = FakeDaemon(runner=runner)
    client, _ = create_test_client(daemon=daemon, tokenizer=tok)

    res = client.post(
        "/generate/stream",
        json={"prompt": "hello", "chat": True, "system_prompt": "test-sys"},
    )
    assert res.status_code == 200
    events = parse_sse_events(res.text)
    assert len(events) == 3
    done_payload = events[2][1]
    assert done_payload["completion"] == "AB"
    assert daemon.asked_counts == [2]


def test_generate_stream_generic_exception() -> None:
    """Verify /generate/stream yields a generic error event on unexpected exceptions."""
    runner = FakeRunner(raise_exc=RuntimeError("internal crash"))
    daemon = FakeDaemon(runner=runner)
    client, ctx = create_test_client(daemon=daemon)

    res = client.post(
        "/generate/stream",
        json={"prompt": "test prompt", "chat": False},
    )
    assert res.status_code == 200
    events = parse_sse_events(res.text)
    assert len(events) == 1
    assert events[0][0] == "error"
    assert events[0][1] == {"detail": "the answer failed"}
