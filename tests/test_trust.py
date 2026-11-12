"""Tests for trust, reputation, ledger, outputs_agree, and verifier."""

from __future__ import annotations

import json
import threading
from pathlib import Path

import pytest
import torch

from torrent_llm.pool.trust import TrustStore, Verifier, outputs_agree


def test_ban_after_two_failures_and_not_after_one() -> None:
    """Verify that a peer is banned only after two failures, not after one."""
    store = TrustStore()

    rep1 = store.record_check("peer1", ok=False)
    assert not rep1.banned
    assert rep1.failures == 1
    assert rep1.checks == 1
    assert not store.is_banned("peer1")
    assert rep1.reason == ""

    rep2 = store.record_check("peer1", ok=False)
    assert rep2.banned
    assert rep2.failures == 2
    assert rep2.checks == 2
    assert rep2.reason == "returned wrong results"
    assert rep2.banned_at > 0.0
    assert store.is_banned("peer1")

    rep3 = store.record_check("peer1", ok=False)
    assert rep3.banned
    assert rep3.failures == 3
    assert rep3.checks == 3
    assert rep3.reason == "returned wrong results"
    assert rep3.banned_at == rep2.banned_at

    rep_ok = store.record_check("peer2", ok=True)
    assert not rep_ok.banned
    assert rep_ok.failures == 0
    assert rep_ok.checks == 1


def test_manual_ban_and_unban() -> None:
    """Verify that peers can be manually banned and unbanned."""
    store = TrustStore()
    assert not store.is_banned("peer1")

    store.ban("peer1", "manual kick")
    assert store.is_banned("peer1")
    rep = store.get("peer1")
    assert rep.banned
    assert rep.reason == "manual kick"
    assert rep.banned_at > 0.0

    store.record_check("peer1", ok=False)
    assert store.get("peer1").failures == 1

    store.unban("peer1")
    assert not store.is_banned("peer1")
    rep_unbanned = store.get("peer1")
    assert not rep_unbanned.banned
    assert rep_unbanned.reason == ""
    assert rep_unbanned.banned_at == 0.0
    assert rep_unbanned.failures == 0
    assert rep_unbanned.checks == 1

    store.unban("non_existent_peer")
    assert not store.is_banned("non_existent_peer")


def test_gatekeeper_strings() -> None:
    """Verify gatekeeper return values for banned and unbanned peers."""
    store = TrustStore()
    assert store.gatekeeper("peer1") is None

    store.record_check("peer1", ok=True)
    assert store.gatekeeper("peer1") is None

    store.ban("peer1", "bad signature")
    assert store.gatekeeper("peer1") == "banned: bad signature"

    store.unban("peer1")
    assert store.gatekeeper("peer1") is None

    store.record_check("peer2", ok=False)
    store.record_check("peer2", ok=False)
    assert store.gatekeeper("peer2") == "banned: returned wrong results"


def test_priority_ordering() -> None:
    """Verify that peers with higher give-and-take ratio rank higher and banned rank 0."""
    store = TrustStore()

    store.record_used("giver", 5000.0)
    store.record_served("taker", 5000.0)
    store.record_used("balanced", 100.0)
    store.record_served("balanced", 100.0)

    p_giver = store.priority("giver")
    p_balanced = store.priority("balanced")
    p_taker = store.priority("taker")

    assert p_giver > p_balanced > p_taker

    store.ban("giver", "reason")
    assert store.priority("giver") == 0.0

    store.record_check("taker", ok=False)
    store.record_check("taker", ok=False)
    assert store.priority("taker") == 0.0


def test_score_for_new_and_checked_peers() -> None:
    """Verify reputation score calculation for new and verified peers."""
    store = TrustStore()

    rep_new = store.get("fresh")
    assert rep_new.score == 1.0

    rep1 = store.record_check("p1", ok=True)
    assert rep1.score == 1.0

    rep2 = store.record_check("p2", ok=False)
    assert rep2.score == 0.5

    store.record_check("p3", ok=True)
    store.record_check("p3", ok=False)
    rep3 = store.record_check("p3", ok=True)
    assert rep3.score == 0.75


def test_ledger_rejects_negative_nan_inf() -> None:
    """Verify that compute ledger rejects negative, NaN, infinite, and non-numeric values."""
    store = TrustStore()

    for invalid in (-1.0, -0.001, float("nan"), float("inf"), float("-inf"), True, False):
        with pytest.raises(ValueError):
            store.record_served("p1", invalid)  # type: ignore[arg-type]
        with pytest.raises(ValueError):
            store.record_used("p1", invalid)  # type: ignore[arg-type]

    store.record_served("p1", 0.0)
    store.record_used("p1", 0.0)
    assert store.get("p1").served_ms == 0.0
    assert store.get("p1").used_ms == 0.0


def test_persistence_round_trip(tmp_path: Path) -> None:
    """Verify that trust store data persists across instances on disk."""
    path = tmp_path / "trust.json"
    store1 = TrustStore(path)

    store1.record_check("peer-a", ok=True)
    store1.record_check("peer-b", ok=False)
    store1.record_check("peer-b", ok=False)
    store1.record_served("peer-a", 150.0)
    store1.record_used("peer-a", 300.0)
    store1.ban("peer-c", "custom ban")
    store1.flush()

    assert path.exists()

    store2 = TrustStore(path)
    all_reps = store2.all()
    assert [r.peer_id for r in all_reps] == ["peer-a", "peer-b", "peer-c"]

    rep_a = store2.get("peer-a")
    assert rep_a.checks == 1
    assert rep_a.failures == 0
    assert not rep_a.banned
    assert rep_a.served_ms == 150.0
    assert rep_a.used_ms == 300.0

    rep_b = store2.get("peer-b")
    assert rep_b.checks == 2
    assert rep_b.failures == 2
    assert rep_b.banned
    assert rep_b.reason == "returned wrong results"

    rep_c = store2.get("peer-c")
    assert rep_c.banned
    assert rep_c.reason == "custom ban"


def test_save_throttling_with_injected_clock(tmp_path: Path) -> None:
    """Verify that changes are only saved once save_interval_s has elapsed."""
    path = tmp_path / "trust.json"
    current_time = 1000.0

    def mock_clock() -> float:
        """Return the mocked timestamp."""
        return current_time

    store = TrustStore(path, save_interval_s=2.0, clock=mock_clock)
    assert not path.exists()

    current_time = 1001.0
    store.record_served("p1", 50.0)
    assert not path.exists()

    current_time = 1001.9
    store.record_used("p1", 20.0)
    assert not path.exists()

    current_time = 1002.5
    store.record_check("p1", ok=True)
    assert path.exists()

    current_time = 1003.0
    store.record_served("p1", 10.0)
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["peers"]["p1"]["served_ms"] == 50.0

    store.flush()
    data2 = json.loads(path.read_text(encoding="utf-8"))
    assert data2["peers"]["p1"]["served_ms"] == 60.0


def test_corrupt_file_renamed_and_store_starts_empty(tmp_path: Path) -> None:
    """Verify that corrupt trust files are renamed to .corrupt and store starts fresh."""
    path = tmp_path / "trust.json"
    corrupt_path = tmp_path / "trust.json.corrupt"

    path.write_text("{invalid json", encoding="utf-8")
    store1 = TrustStore(path)
    assert not path.exists()
    assert corrupt_path.exists()
    assert corrupt_path.read_text(encoding="utf-8") == "{invalid json"
    assert store1.all() == []

    path.write_text(json.dumps({"peers": {}, "version": 99}), encoding="utf-8")
    store2 = TrustStore(path)
    assert not path.exists()
    assert corrupt_path.exists()
    assert store2.all() == []

    path.write_text(
        json.dumps({"peers": {"p1": {"checks": "bad"}}, "version": 1}),
        encoding="utf-8",
    )
    store3 = TrustStore(path)
    assert not path.exists()
    assert corrupt_path.exists()
    assert store3.all() == []

    long_key = "x" * 65
    valid_key = "valid_peer"
    valid_entry = {
        "banned": False,
        "banned_at": 0.0,
        "checks": 1,
        "failures": 0,
        "last_seen": 10.0,
        "reason": "",
        "served_ms": 0.0,
        "used_ms": 0.0,
    }
    path.write_text(
        json.dumps({"peers": {long_key: valid_entry, valid_key: valid_entry}, "version": 1}),
        encoding="utf-8",
    )
    store4 = TrustStore(path)
    assert path.exists()
    assert [r.peer_id for r in store4.all()] == [valid_key]


def test_max_peers_eviction_keeps_bans() -> None:
    """Verify that eviction drops oldest unbanned peers but preserves all bans."""
    current_time = 0.0

    def mock_clock() -> float:
        """Return the mocked timestamp."""
        return current_time

    store = TrustStore(max_peers=3, clock=mock_clock)

    current_time = 1.0
    store.ban("banned-1", "bad")
    current_time = 2.0
    store.record_check("p2", ok=True)
    current_time = 3.0
    store.record_check("p3", ok=True)

    assert len(store.all()) == 3

    current_time = 4.0
    store.record_check("p4", ok=True)

    assert len(store.all()) == 3
    assert store.is_banned("banned-1")
    assert store.get("p2").checks == 0
    assert store.get("p3").checks == 1
    assert store.get("p4").checks == 1

    current_time = 5.0
    store.ban("banned-2", "bad 2")
    current_time = 6.0
    store.ban("banned-3", "bad 3")

    assert len(store.all()) == 3
    assert store.is_banned("banned-1")
    assert store.is_banned("banned-2")
    assert store.is_banned("banned-3")


def test_outputs_agree() -> None:
    """Verify outputs_agree across integer, float, bool, empty, and invalid cases."""
    assert outputs_agree(torch.tensor([1, 2, 3]), torch.tensor([1, 2, 3])) is True
    assert outputs_agree(torch.tensor([1, 2, 3]), torch.tensor([1, 2, 4])) is False

    torch.manual_seed(42)
    f1 = torch.randn(10, 10)
    f_close = f1 + 1e-4 * torch.randn(10, 10)
    assert outputs_agree(f1, f_close) is True

    f_far = f1 + 0.1 * torch.randn(10, 10)
    assert outputs_agree(f1, f_far) is False

    f_nan = f1.clone()
    f_nan[0, 0] = float("nan")
    assert outputs_agree(f1, f_nan) is False
    assert outputs_agree(f_nan, f1) is False

    f_inf = f1.clone()
    f_inf[0, 0] = float("inf")
    assert outputs_agree(f1, f_inf) is False

    assert outputs_agree(torch.tensor([1, 2]), torch.tensor([1, 2, 3])) is False
    assert outputs_agree(torch.zeros(2, 3), torch.zeros(3, 2)) is False

    assert outputs_agree(torch.tensor([1, 2]), torch.tensor([1.0, 2.0])) is False
    assert outputs_agree(torch.tensor([1.0, 2.0]), torch.tensor([1, 2])) is False

    assert outputs_agree(torch.tensor([True, False]), torch.tensor([True, False])) is True
    assert outputs_agree(torch.tensor([True, False]), torch.tensor([True, True])) is False
    assert outputs_agree(torch.tensor([True, False]), torch.tensor([1, 0])) is True

    assert outputs_agree(torch.zeros(0), torch.zeros(0)) is True
    assert outputs_agree(torch.zeros(0, 5), torch.zeros(0, 5)) is True
    assert outputs_agree(torch.zeros(0, 5), torch.zeros(0, 6)) is False
    assert (
        outputs_agree(torch.zeros(0, dtype=torch.int32), torch.zeros(0, dtype=torch.float32))
        is False
    )

    assert outputs_agree(torch.zeros(4, 4), torch.zeros(4, 4)) is True


def test_verifier_with_scripted_rng() -> None:
    """Verify that Verifier accurately checks requests and picks passes."""
    seq = [0.1, 0.7, 0.49, 0.5]

    def scripted_rng() -> float:
        """Return the next value in the test sequence."""
        return seq.pop(0)

    v = Verifier(rate=0.5, rng=scripted_rng)
    assert v.check_this_request() is True
    assert v.check_this_request() is False
    assert v.check_this_request() is True
    assert v.check_this_request() is False

    v0 = Verifier(rate=0.0, rng=lambda: 0.0)
    assert v0.check_this_request() is False

    v1 = Verifier(rate=1.0, rng=lambda: 1.0)
    assert v1.check_this_request() is True

    for invalid_rate in (-0.1, 1.1, float("nan"), True, False):
        with pytest.raises(ValueError):
            Verifier(rate=invalid_rate)  # type: ignore[arg-type]

    v_pick = Verifier(rate=0.5, rng=lambda: 0.25)
    assert v_pick.pick_pass(10) == 2

    v_pick_high = Verifier(rate=0.5, rng=lambda: 1.0)
    assert v_pick_high.pick_pass(10) == 9

    v_pick_zero = Verifier(rate=0.5, rng=lambda: 0.0)
    assert v_pick_zero.pick_pass(10) == 0
    assert v_pick_zero.pick_pass(1) == 0
    assert v_pick_zero.pick_pass(0) == 0


def test_record_check_concurrency() -> None:
    """Verify that concurrent calls to record_check remain consistent."""
    store = TrustStore()
    threads: list[threading.Thread] = []

    def worker() -> None:
        """Perform 200 record_check calls."""
        for _ in range(200):
            store.record_check("peer1", ok=True)

    for _ in range(8):
        t = threading.Thread(target=worker)
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    rep = store.get("peer1")
    assert rep.checks == 1600
    assert rep.failures == 0
    assert not rep.banned


# --- manager regressions ------------------------------------------------------


def test_complex_tensors_never_agree():
    import torch as _torch

    from torrent_llm.pool.trust import outputs_agree as _agree

    assert not _agree(_torch.tensor([1 + 1j]), _torch.tensor([1 - 1j]))
    assert not _agree(_torch.tensor([1 + 1j]), _torch.tensor([1 + 1j]))


def test_a_new_record_survives_a_store_full_of_bans(tmp_path):
    from torrent_llm.pool.trust import TrustStore as _Store

    store = _Store(None, max_peers=2)
    store.ban("a" * 32, "x")
    store.ban("b" * 32, "y")
    rep = store.record_check("c" * 32, True)
    assert rep.checks == 1
    assert store.get("c" * 32).checks == 1


def test_a_failing_disk_is_retried_once_per_interval_not_on_every_record(tmp_path, caplog):
    from torrent_llm.pool.trust import TrustStore as _Store

    now = [0.0]
    store = _Store(tmp_path / "trust.json", save_interval_s=10.0, clock=lambda: now[0])
    store._save_to_disk = lambda: (_ for _ in ()).throw(OSError("disk full"))  # type: ignore[method-assign]
    now[0] = 11.0
    for _ in range(5):
        store.record_check("d" * 32, True)
    assert caplog.text.count("failed to save trust store") == 1
