"""Tests for the distributed hash table (DHT) in torrent_llm.net.dht."""

from __future__ import annotations

import concurrent.futures
import socket
import time
from collections.abc import Callable

import pytest

from torrent_llm.net.dht import (
    Dht,
    ProviderStore,
    RoutingTable,
    distance,
    node_key,
    topic_key,
)
from torrent_llm.net.identity import Identity
from torrent_llm.net.node import Node, NodeConfig
from torrent_llm.net.records import PeerRecord, make_record
from torrent_llm.net.rpc import RpcError


def _fake_clock(initial: float = 1000.0) -> tuple[Callable[[], float], Callable[[float], None]]:
    """Return an injected clock and a function to advance it."""
    curr = initial

    def clock() -> float:
        return curr

    def advance(delta: float) -> None:
        nonlocal curr
        curr += delta

    return clock, advance


# --- RoutingTable unit tests --------------------------------------------------


def test_routing_table_bucket_placement():
    own = b"\x00" * 32
    rt = RoutingTable(own)

    ident = Identity.generate()
    pkey = node_key(ident.peer_id)
    expected_idx = (distance(own, pkey)).bit_length() - 1

    rec = make_record(ident, ["tcp://127.0.0.1:9001"], seq=1)
    assert rt.add(rec)
    assert rec.peer_id in rt._buckets[expected_idx]
    assert len(rt) == 1


def test_routing_table_never_adds_own():
    own_id = Identity.generate()
    own_key = node_key(own_id.peer_id)
    rt = RoutingTable(own_key)

    rec = make_record(own_id, ["tcp://127.0.0.1:9001"], seq=1)
    assert not rt.add(rec)
    assert len(rt) == 0


def test_routing_table_k_limit_keeps_old_peers():
    own = b"\x00" * 32
    target_idx = 255
    clock, advance = _fake_clock(1000.0)
    rt = RoutingTable(own, k=3, stale_s=900.0, clock=clock)

    peers: list[PeerRecord] = []
    while len(peers) < 4:
        ident = Identity.generate()
        rec = make_record(ident, ["tcp://127.0.0.1:9001"], seq=1)
        if (distance(own, node_key(rec.peer_id))).bit_length() - 1 == target_idx:
            peers.append(rec)

    for r in peers[:3]:
        assert rt.add(r)
    assert len(rt) == 3

    advance(100.0)
    assert not rt.add(peers[3])
    assert len(rt) == 3
    assert peers[3].peer_id not in rt._buckets[target_idx]
    for r in peers[:3]:
        assert r.peer_id in rt._buckets[target_idx]


def test_routing_table_stale_eviction_with_injected_clock():
    own = b"\x00" * 32
    target_idx = 255
    clock, advance = _fake_clock(1000.0)
    rt = RoutingTable(own, k=3, stale_s=900.0, clock=clock)

    peers: list[PeerRecord] = []
    while len(peers) < 4:
        ident = Identity.generate()
        rec = make_record(ident, ["tcp://127.0.0.1:9001"], seq=1)
        if (distance(own, node_key(rec.peer_id))).bit_length() - 1 == target_idx:
            peers.append(rec)

    for r in peers[:3]:
        assert rt.add(r)

    advance(901.0)

    assert rt.add(peers[3])
    assert len(rt) == 3
    assert peers[0].peer_id not in rt._buckets[target_idx]
    assert peers[1].peer_id in rt._buckets[target_idx]
    assert peers[2].peer_id in rt._buckets[target_idx]
    assert peers[3].peer_id in rt._buckets[target_idx]


def test_routing_table_newer_record_replaces():
    own = b"\x00" * 32
    clock, advance = _fake_clock(1000.0)
    rt = RoutingTable(own, clock=clock)

    ident = Identity.generate()
    rec1 = make_record(ident, ["tcp://127.0.0.1:9001"], seq=1)
    rec2 = make_record(ident, ["tcp://127.0.0.1:9002"], seq=2)
    rec_old = make_record(ident, ["tcp://127.0.0.1:9000"], seq=0)

    assert rt.add(rec1)
    closest = rt.closest(own, 1)
    assert closest[0].seq == 1
    assert closest[0].addresses == ("tcp://127.0.0.1:9001",)

    advance(10.0)
    assert rt.add(rec2)
    closest = rt.closest(own, 1)
    assert closest[0].seq == 2
    assert closest[0].addresses == ("tcp://127.0.0.1:9002",)

    advance(10.0)
    assert rt.add(rec_old)
    closest = rt.closest(own, 1)
    assert closest[0].seq == 2


def test_routing_table_closest_ordering():
    own = b"\x00" * 32
    rt = RoutingTable(own)

    records: list[PeerRecord] = []
    for _ in range(10):
        ident = Identity.generate()
        rec = make_record(ident, ["tcp://127.0.0.1:9001"], seq=1)
        rt.add(rec)
        records.append(rec)

    target = topic_key("target-key")
    closest = rt.closest(target, n=5)
    assert len(closest) == 5

    distances = [distance(target, node_key(r.peer_id)) for r in closest]
    assert distances == sorted(distances)

    excluded_pid = closest[0].peer_id
    filtered = rt.closest(target, n=5, exclude=frozenset({excluded_pid}))
    assert excluded_pid not in [r.peer_id for r in filtered]


# --- ProviderStore unit tests -------------------------------------------------


def test_provider_store_ttl_expiry_with_injected_clock():
    clock, advance = _fake_clock(1000.0)
    store = ProviderStore(clock=clock)

    rec = make_record(Identity.generate(), ["tcp://127.0.0.1:9001"], seq=1)
    key = topic_key("swarm-1")

    store.add(key, rec, ttl_s=10.0)
    assert len(store.get(key)) == 1

    advance(5.0)
    assert len(store.get(key)) == 1

    advance(5.1)
    assert len(store.get(key)) == 0

    store.add(key, rec, ttl_s=10.0)
    advance(15.0)
    removed = store.expire()
    assert removed == 1
    assert len(store.get(key)) == 0


def test_provider_store_per_key_cap():
    clock, _ = _fake_clock(1000.0)
    store = ProviderStore(max_per_key=2, clock=clock)
    key = topic_key("swarm-1")

    rec1 = make_record(Identity.generate(), ["tcp://127.0.0.1:9001"], seq=1)
    rec2 = make_record(Identity.generate(), ["tcp://127.0.0.1:9002"], seq=1)
    rec3 = make_record(Identity.generate(), ["tcp://127.0.0.1:9003"], seq=1)

    store.add(key, rec1, ttl_s=10.0)
    store.add(key, rec2, ttl_s=30.0)
    store.add(key, rec3, ttl_s=20.0)

    providers = store.get(key)
    assert len(providers) == 2
    pids = {p.peer_id for p in providers}
    assert rec1.peer_id not in pids
    assert rec2.peer_id in pids
    assert rec3.peer_id in pids


def test_provider_store_max_keys():
    clock, _ = _fake_clock(1000.0)
    store = ProviderStore(max_keys=2, clock=clock)

    k1 = topic_key("swarm-1")
    k2 = topic_key("swarm-2")
    k3 = topic_key("swarm-3")

    rec1 = make_record(Identity.generate(), ["tcp://127.0.0.1:9001"], seq=1)
    rec2 = make_record(Identity.generate(), ["tcp://127.0.0.1:9002"], seq=1)
    rec3 = make_record(Identity.generate(), ["tcp://127.0.0.1:9003"], seq=1)

    store.add(k1, rec1, ttl_s=10.0)
    store.add(k2, rec2, ttl_s=20.0)
    store.add(k3, rec3, ttl_s=30.0)

    assert store.get(k1) == []
    assert len(store.get(k2)) == 1
    assert len(store.get(k3)) == 1


def test_provider_dedupe():
    store = ProviderStore()
    key = topic_key("swarm-dedupe")
    ident = Identity.generate()
    rec1 = make_record(ident, ["tcp://127.0.0.1:9001"], seq=1)
    rec2 = make_record(ident, ["tcp://127.0.0.1:9002"], seq=2)

    store.add(key, rec1, ttl_s=100.0)
    store.add(key, rec2, ttl_s=200.0)

    providers = store.get(key)
    assert len(providers) == 1
    assert providers[0].seq == 2


# --- Network & RPC tests ------------------------------------------------------


def test_dht_bad_key_raises_bad_request():
    node = Node(NodeConfig(identity=Identity.generate(), listen_host="127.0.0.1"))
    dht = Dht(node)
    try:
        node.start()
        dht.attach()

        with pytest.raises(RpcError) as exc_info:
            node.call(node.peer_id, "dht.find_node", {"key": "bad_key"}, timeout=2.0)
        assert exc_info.value.code == "bad_request"

        with pytest.raises(RpcError) as exc_info:
            node.call(node.peer_id, "dht.get_providers", {"key": 123}, timeout=2.0)
        assert exc_info.value.code == "bad_request"
    finally:
        node.close()


def test_dht_add_provider_someone_elses_record_denied():
    a = Node(NodeConfig(identity=Identity.generate(), listen_host="127.0.0.1"))
    b = Node(NodeConfig(identity=Identity.generate(), listen_host="127.0.0.1"))
    dht_b = Dht(b)
    try:
        a.start()
        b.start()
        dht_b.attach()

        a.connect(f"tcp://127.0.0.1:{b.listen_port}")

        third_id = Identity.generate()
        third_rec = make_record(third_id, ["tcp://127.0.0.1:9999"], seq=1)

        key = topic_key("swarm-test")
        with pytest.raises(RpcError) as exc_info:
            a.call(
                b.peer_id,
                "dht.add_provider",
                {"key": key.hex(), "record": third_rec.to_dict(), "ttl": 100},
                timeout=2.0,
            )
        assert exc_info.value.code == "denied"
    finally:
        a.close()
        b.close()


def test_add_provider_bad_ttl_raises_bad_request():
    a = Node(NodeConfig(identity=Identity.generate(), listen_host="127.0.0.1"))
    b = Node(NodeConfig(identity=Identity.generate(), listen_host="127.0.0.1"))
    dht_b = Dht(b)
    try:
        a.start()
        b.start()
        dht_b.attach()
        a.connect(f"tcp://127.0.0.1:{b.listen_port}")
        key = topic_key("swarm-test")

        # Missing ttl
        with pytest.raises(RpcError) as exc_info:
            a.call(
                b.peer_id,
                "dht.add_provider",
                {"key": key.hex(), "record": a.record().to_dict()},
                timeout=2.0,
            )
        assert exc_info.value.code == "bad_request"

        # Invalid ttl (string)
        with pytest.raises(RpcError) as exc_info:
            a.call(
                b.peer_id,
                "dht.add_provider",
                {"key": key.hex(), "record": a.record().to_dict(), "ttl": "invalid"},
                timeout=2.0,
            )
        assert exc_info.value.code == "bad_request"
    finally:
        a.close()
        b.close()


def test_add_provider_tampered_record_denied():
    a = Node(NodeConfig(identity=Identity.generate(), listen_host="127.0.0.1"))
    b = Node(NodeConfig(identity=Identity.generate(), listen_host="127.0.0.1"))
    dht_b = Dht(b)
    try:
        a.start()
        b.start()
        dht_b.attach()
        a.connect(f"tcp://127.0.0.1:{b.listen_port}")
        key = topic_key("swarm-test")

        rec_dict = a.record().to_dict()
        rec_dict["seq"] = rec_dict["seq"] + 1

        with pytest.raises(RpcError) as exc_info:
            a.call(
                b.peer_id,
                "dht.add_provider",
                {"key": key.hex(), "record": rec_dict, "ttl": 100},
                timeout=2.0,
            )
        assert exc_info.value.code == "denied"
    finally:
        a.close()
        b.close()


def test_provider_quota_denied_over_limit():
    node = Node(NodeConfig(identity=Identity.generate(), listen_host="127.0.0.1"))
    dht = Dht(node)
    try:
        node.start()
        dht.attach()

        caller = Node(NodeConfig(identity=Identity.generate(), listen_host="127.0.0.1"))
        try:
            caller.start()
            caller.connect(f"tcp://127.0.0.1:{node.listen_port}")
            rec = caller.record()

            # Announce 64 distinct keys
            for i in range(64):
                k = topic_key(f"topic-{i}")
                caller.call(
                    node.peer_id,
                    "dht.add_provider",
                    {"key": k.hex(), "record": rec.to_dict(), "ttl": 100},
                    timeout=2.0,
                )

            # Re-announcing an existing key succeeds
            caller.call(
                node.peer_id,
                "dht.add_provider",
                {"key": topic_key("topic-0").hex(), "record": rec.to_dict(), "ttl": 100},
                timeout=2.0,
            )

            # 65th key is denied
            with pytest.raises(RpcError) as exc_info:
                caller.call(
                    node.peer_id,
                    "dht.add_provider",
                    {"key": topic_key("topic-64").hex(), "record": rec.to_dict(), "ttl": 100},
                    timeout=2.0,
                )
            assert exc_info.value.code == "denied"
        finally:
            caller.close()
    finally:
        node.close()


def test_oversized_records_fit_in_header_budget():
    node = Node(NodeConfig(identity=Identity.generate(), listen_host="127.0.0.1"))
    dht = Dht(node)
    try:
        node.start()
        dht.attach()

        target_key = topic_key("large-key")
        large_attrs = {f"k{i}": "x" * 250 for i in range(16)}
        for i in range(20):
            ident = Identity.generate()
            rec = make_record(
                ident,
                [f"tcp://127.0.0.1:{9100 + i}"],
                seq=1,
                attrs=large_attrs,
            )
            dht._provider_store.add(target_key, rec, ttl_s=1000.0)

        res_h, _ = node.call(
            node.peer_id,
            "dht.get_providers",
            {"key": target_key.hex()},
            timeout=2.0,
        )
        assert len(res_h["providers"]) > 0
        assert len(res_h["providers"]) < 20
    finally:
        node.close()


def test_lookup_ignores_invalid_records_in_reply():
    a = Node(NodeConfig(identity=Identity.generate(), listen_host="127.0.0.1"))
    b = Node(NodeConfig(identity=Identity.generate(), listen_host="127.0.0.1"))
    c = Node(NodeConfig(identity=Identity.generate(), listen_host="127.0.0.1"))
    dht_a = Dht(a)
    dht_c = Dht(c)
    try:
        a.start()
        b.start()
        c.start()
        dht_a.attach()
        dht_c.attach()

        rec_c = c.record()

        invalid_id = Identity.generate()
        invalid_rec = make_record(invalid_id, ["tcp://127.0.0.1:9099"], seq=1)
        invalid_dict = invalid_rec.to_dict()
        invalid_dict["seq"] = 999  # breaks signature

        async def mock_find_node(ctx, header, body):
            return {"peers": [invalid_dict, rec_c.to_dict()]}, b""

        b.register("dht.find_node", mock_find_node, blocking=False)
        a.connect(f"tcp://127.0.0.1:{b.listen_port}")

        found = dht_a.find_node(topic_key("target"))
        pids = [r.peer_id for r in found]
        assert rec_c.peer_id in pids
        assert a.record_of(rec_c.peer_id) is not None
        assert a.record_of(invalid_id.peer_id) is None
    finally:
        a.close()
        b.close()
        c.close()


def test_lookup_caps_peers_per_reply_and_candidates():
    a = Node(NodeConfig(identity=Identity.generate(), listen_host="127.0.0.1"))
    b = Node(NodeConfig(identity=Identity.generate(), listen_host="127.0.0.1"))
    dht_a = Dht(a)
    try:
        a.start()
        b.start()
        dht_a.attach()

        many_records = [
            make_record(Identity.generate(), [f"tcp://127.0.0.1:{10000 + i}"], seq=1).to_dict()
            for i in range(50)
        ]

        async def mock_find_node(ctx, header, body):
            return {"peers": many_records}, b""

        b.register("dht.find_node", mock_find_node, blocking=False)
        a.connect(f"tcp://127.0.0.1:{b.listen_port}")

        dialled: list[str] = []
        real_acall = a.acall

        async def counting_acall(peer_id, method, *args, **kwargs):
            if method == "dht.find_node":
                dialled.append(peer_id)
            return await real_acall(peer_id, method, *args, **kwargs)

        a.acall = counting_acall  # type: ignore[method-assign]
        found = dht_a.find_node(topic_key("target"))
        assert len(found) <= dht_a._k
        # 50 records in one reply must not turn into 50 dials: at most K beyond b itself.
        assert len(dialled) <= dht_a._k + 1
        # Records that were never used do not flood the address book.
        known = sum(1 for r in many_records if a.record_of(r["peer_id"]) is not None)
        assert known <= dht_a._k
    finally:
        a.close()
        b.close()


def test_dead_peer_removed_after_failed_query():
    node = Node(NodeConfig(identity=Identity.generate(), listen_host="127.0.0.1"))
    dht = Dht(node)
    try:
        node.start()
        dht.attach()

        dead_ident = Identity.generate()
        dead_rec = make_record(dead_ident, ["tcp://127.0.0.1:65432"], seq=1)
        dht.table.add(dead_rec)
        assert len(dht.table) == 1

        target = topic_key("target")
        closest = dht.find_node(target)
        assert dead_rec.peer_id not in [r.peer_id for r in closest]
        assert len(dht.table) == 0
    finally:
        node.close()


def test_twenty_node_dht_network():
    nodes = [
        Node(NodeConfig(identity=Identity.generate(), listen_host="127.0.0.1")) for _ in range(20)
    ]
    dhts = [Dht(n) for n in nodes]
    late: Node | None = None
    try:
        for n in nodes:
            n.start()
        for d in dhts:
            d.attach()

        # Each node connects to node 0 only
        addr0 = f"tcp://127.0.0.1:{nodes[0].listen_port}"
        for i in range(1, 20):
            nodes[i].connect(addr0)

        # Bootstraps in parallel
        def _bootstrap_one(dht_instance: Dht) -> int:
            return dht_instance.bootstrap()

        with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
            futs = [executor.submit(_bootstrap_one, d) for d in dhts[1:]]
            for f in futs:
                size = f.result(timeout=15.0)
                assert size > 0

        # A node announces provide(topic_key("swarm-x"))
        swarm_key = topic_key("swarm-x")
        accepted = dhts[19].provide(swarm_key)
        assert accepted > 0

        # Fresh node that connects only to node 0 and does not bootstrap
        late = Node(NodeConfig(identity=Identity.generate(), listen_host="127.0.0.1"))
        late_dht = Dht(late)
        late.start()
        late_dht.attach()
        late.connect(addr0)

        # Assert late never connected to node 19 and does not hold its record
        assert not late.is_connected(nodes[19].peer_id)
        assert late.record_of(nodes[19].peer_id) is None

        # Find providers through the DHT
        providers = late_dht.find_providers(swarm_key)
        assert any(p.peer_id == nodes[19].peer_id for p in providers)

        # Dial and ping node 19 by peer id using the record found via DHT
        res, _ = late.call(nodes[19].peer_id, "node.ping", timeout=5.0)
        assert "t" in res

    finally:
        to_close = list(nodes)
        if late is not None:
            to_close.append(late)
        with concurrent.futures.ThreadPoolExecutor(max_workers=25) as executor:
            list(executor.map(lambda n: n.close(), to_close))


def test_a_refused_or_rate_limited_peer_keeps_its_place_but_a_dead_one_does_not():
    from torrent_llm.net.dht import _peer_seems_gone

    assert _peer_seems_gone(RpcError("unavailable", "could not reach abcd1234: refused"))
    assert _peer_seems_gone(RpcError("deadline", "timed out"))
    assert not _peer_seems_gone(RpcError("unavailable", "too many connections"))
    assert not _peer_seems_gone(RpcError("denied", "rate limited"))
    assert not _peer_seems_gone(RpcError("too_large", "reply too large"))


def test_a_dead_peer_with_an_address_is_dialled_and_removed():
    node = Node(NodeConfig(identity=Identity.generate(), listen_host="127.0.0.1"))
    dht = Dht(node)
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.bind(("127.0.0.1", 0))
    port = srv.getsockname()[1]
    srv.close()  # nothing listens there now: connection refused
    try:
        node.start()
        dht.attach()
        dead = make_record(Identity.generate(), [f"tcp://127.0.0.1:{port}"], seq=1)
        node.add_record(dead)
        dht.table.add(dead)
        assert dead.peer_id not in [r.peer_id for r in dht.find_node(topic_key("x"))]
        assert len(dht.table) == 0
    finally:
        node.close()


def test_expired_keys_do_not_count_against_a_providers_quota():
    """A provider at its key quota may announce again once its old keys have expired."""
    from torrent_llm.net import dht as dht_mod

    a = Node(NodeConfig(identity=Identity.generate(), listen_host="127.0.0.1"))
    b = Node(NodeConfig(identity=Identity.generate(), listen_host="127.0.0.1"))
    dht_b = Dht(b)
    try:
        a.start()
        b.start()
        dht_b.attach()
        a.connect(f"tcp://127.0.0.1:{b.listen_port}")
        record = a.record().to_dict()

        def announce(name: str) -> None:
            header = {"key": topic_key(name).hex(), "record": record, "ttl": 1}
            a.call(b.peer_id, "dht.add_provider", header, timeout=5)

        for i in range(dht_mod._MAX_KEYS_PER_PROVIDER):
            announce(f"k{i}")
        with pytest.raises(RpcError) as caught:
            announce("one-too-many")
        assert caught.value.code == "denied"
        time.sleep(1.2)  # every earlier key has expired
        announce("one-too-many")
    finally:
        a.close()
        b.close()
