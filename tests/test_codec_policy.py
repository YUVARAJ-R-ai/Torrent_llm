"""Tests for LinkEstimator and CodecChooser codec policy."""

from __future__ import annotations

import concurrent.futures

import pytest

from torrent_llm.codec.profiles import CodecProfile, CodecSet
from torrent_llm.pool.codec_policy import (
    MIN_SAMPLE_BYTES,
    SMOOTHING,
    CodecChooser,
    LinkEstimator,
)


def _make_codecs() -> CodecSet:
    return CodecSet.from_config(
        {
            "raw32": {"name": "raw"},
            "bf16": {"name": "raw", "wire_dtype": "bfloat16"},
            "int8": {"name": "int8"},
            "pca": {"name": "lowrank", "rank": 256, "wire_dtype": "float16"},
        }
    )


def test_estimator_first_sample_and_rtt_subtraction():
    est = LinkEstimator()
    assert est.bandwidth("peerA") is None
    assert est.rtt("peerA") is None

    # First sample with RTT
    # transport_ms = 20.0, rtt_ms = 4.0 -> effective_ms = 16.0 ms
    # sent_bytes = 32_000 >= MIN_SAMPLE_BYTES
    # bw = 32_000 / 16.0 * 1000 = 2_000_000 B/s
    est.observe("peerA", sent_bytes=32_000, transport_ms=20.0, rtt_ms=4.0)
    assert est.rtt("peerA") == pytest.approx(4.0)
    assert est.bandwidth("peerA") == pytest.approx(2_000_000.0)


def test_estimator_smoothing():
    est = LinkEstimator()
    # Sample 1
    est.observe("peerA", sent_bytes=32_000, transport_ms=20.0, rtt_ms=10.0)
    # rtt = 10.0, effective_ms = 10.0, bw = 3_200_000.0
    assert est.rtt("peerA") == pytest.approx(10.0)
    assert est.bandwidth("peerA") == pytest.approx(3_200_000.0)

    # Sample 2
    # new rtt_ms = 20.0 -> rtt = (1 - 0.3) * 10.0 + 0.3 * 20.0 = 13.0 ms
    # effective_ms = max(23.0 - 13.0, 1.0) = 10.0 ms
    # new sample bw = 64_000 / 10.0 * 1000 = 6_400_000.0
    # smoothed bw = (1 - 0.3) * 3_200_000 + 0.3 * 6_400_000 = 2_240_000 + 1_920_000 = 4_160_000.0
    est.observe("peerA", sent_bytes=64_000, transport_ms=23.0, rtt_ms=20.0)
    assert est.rtt("peerA") == pytest.approx(13.0)
    assert est.bandwidth("peerA") == pytest.approx(4_160_000.0)


def test_estimator_transport_below_rtt_clamps_to_1ms():
    est = LinkEstimator()
    est.observe("peerA", sent_bytes=1000, transport_ms=10.0, rtt_ms=50.0)
    assert est.rtt("peerA") == pytest.approx(50.0)

    # transport_ms (20.0) < rtt (50.0) -> max(20.0 - 50.0, 1.0) = 1.0 ms
    est.observe("peerA", sent_bytes=32_000, transport_ms=20.0)
    expected_bw = (32_000 / 1.0) * 1000.0
    assert est.bandwidth("peerA") == pytest.approx(expected_bw)


def test_estimator_small_payloads_ignored_for_bandwidth_but_rtt_updated():
    est = LinkEstimator()
    # Payload below MIN_SAMPLE_BYTES
    est.observe("peerA", sent_bytes=MIN_SAMPLE_BYTES - 1, transport_ms=5.0, rtt_ms=8.0)
    assert est.rtt("peerA") == pytest.approx(8.0)
    assert est.bandwidth("peerA") is None

    # Another small payload with new RTT
    est.observe("peerA", sent_bytes=1024, transport_ms=6.0, rtt_ms=18.0)
    expected_rtt = (1.0 - SMOOTHING) * 8.0 + SMOOTHING * 18.0
    assert est.rtt("peerA") == pytest.approx(expected_rtt)
    assert est.bandwidth("peerA") is None

    # Large payload with no rtt_ms
    est.observe("peerA", sent_bytes=MIN_SAMPLE_BYTES, transport_ms=20.0)
    assert est.rtt("peerA") == pytest.approx(expected_rtt)
    assert est.bandwidth("peerA") is not None


def test_estimator_bad_inputs_ignored():
    est = LinkEstimator()

    # Negative sent_bytes with valid RTT -> bw ignored, RTT updated
    est.observe("peerA", sent_bytes=-500, transport_ms=10.0, rtt_ms=6.0)
    assert est.rtt("peerA") == pytest.approx(6.0)
    assert est.bandwidth("peerA") is None

    # NaN / inf / negative transport_ms -> bw ignored
    est.observe("peerA", sent_bytes=32_000, transport_ms=float("nan"))
    est.observe("peerA", sent_bytes=32_000, transport_ms=float("inf"))
    est.observe("peerA", sent_bytes=32_000, transport_ms=-1.0)
    assert est.bandwidth("peerA") is None

    # NaN / inf / negative rtt_ms -> rtt ignored
    est.observe("peerA", sent_bytes=MIN_SAMPLE_BYTES - 1, transport_ms=10.0, rtt_ms=float("nan"))
    est.observe("peerA", sent_bytes=MIN_SAMPLE_BYTES - 1, transport_ms=10.0, rtt_ms=float("inf"))
    est.observe("peerA", sent_bytes=MIN_SAMPLE_BYTES - 1, transport_ms=10.0, rtt_ms=-5.0)
    assert est.rtt("peerA") == pytest.approx(6.0)

    # Boolean inputs ignored
    est.observe("peerA", sent_bytes=True, transport_ms=10.0)  # type: ignore[arg-type]
    est.observe("peerA", sent_bytes=32_000, transport_ms=True)  # type: ignore[arg-type]
    assert est.bandwidth("peerA") is None

    # Completely invalid peer observation does not add peer to snapshot
    est.observe(
        "peerGhost", sent_bytes=float("nan"), transport_ms=float("nan"), rtt_ms=float("nan")
    )
    assert "peerGhost" not in est.snapshot()


def test_estimator_lru_cap():
    est = LinkEstimator(max_peers=3)
    est.observe("p1", sent_bytes=32_000, transport_ms=10.0)
    est.observe("p2", sent_bytes=32_000, transport_ms=10.0)
    est.observe("p3", sent_bytes=32_000, transport_ms=10.0)
    assert set(est.snapshot().keys()) == {"p1", "p2", "p3"}

    # Adding p4 evicts p1 (least recently updated)
    est.observe("p4", sent_bytes=32_000, transport_ms=10.0)
    assert set(est.snapshot().keys()) == {"p2", "p3", "p4"}

    # Update p2, making it most recently updated
    est.observe("p2", sent_bytes=32_000, transport_ms=10.0)
    # Adding p5 should evict p3
    est.observe("p5", sent_bytes=32_000, transport_ms=10.0)
    assert set(est.snapshot().keys()) == {"p2", "p4", "p5"}


def test_estimator_forget_and_snapshot():
    est = LinkEstimator()
    est.observe("p1", sent_bytes=32_000, transport_ms=10.0, rtt_ms=2.0)
    snap = est.snapshot()
    assert "p1" in snap
    assert snap["p1"]["rtt"] == pytest.approx(2.0)
    assert snap["p1"]["bandwidth"] is not None

    est.forget("p1")
    assert est.bandwidth("p1") is None
    assert est.rtt("p1") is None
    assert "p1" not in est.snapshot()


def test_estimator_concurrent_observe():
    est = LinkEstimator(max_peers=100)

    def worker(worker_id: int):
        for i in range(50):
            peer = f"peer_{worker_id % 5}"
            est.observe(peer, sent_bytes=32_000 + i * 100, transport_ms=10.0, rtt_ms=2.0)
            _ = est.bandwidth(peer)
            _ = est.rtt(peer)
            _ = est.snapshot()

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(worker, i) for i in range(16)]
        for f in concurrent.futures.as_completed(futures):
            f.result()

    snap = est.snapshot()
    assert len(snap) <= 100
    for m in snap.values():
        assert m["bandwidth"] is not None
        assert m["rtt"] is not None


def test_codec_chooser_default_ratios():
    codecs = _make_codecs()
    est = LinkEstimator()
    chooser = CodecChooser(
        {"mode": "fixed", "fixed": "raw32", "ladder": ["raw32", "bf16", "int8", "pca"]},
        codecs,
        est,
    )

    # raw32 -> DEFAULT_RATIOS["raw"] = 1.0
    assert chooser.ratio("raw32") == pytest.approx(1.0)
    # bf16 -> raw with bfloat16 wire -> 1.0 * 0.5 = 0.5
    assert chooser.ratio("bf16") == pytest.approx(0.5)
    # int8 -> DEFAULT_RATIOS["int8"] = 0.27
    assert chooser.ratio("int8") == pytest.approx(0.27)
    # pca -> lowrank rank 256 fp16 wire: (256 / 1024) * 0.5 = 0.125
    assert chooser.ratio("pca") == pytest.approx(0.125)


def test_codec_chooser_fixed_mode():
    codecs = _make_codecs()
    est = LinkEstimator()
    policy = {"mode": "fixed", "fixed": "bf16", "ladder": ["raw32", "bf16", "int8"]}
    chooser = CodecChooser(policy, codecs, est)

    assert chooser.choose("peer1", raw_bytes=100, seq_len=1) == "bf16"
    assert chooser.choose("peer1", raw_bytes=1_000_000, seq_len=2048) == "bf16"


def test_codec_chooser_adaptive_picks_bf16_on_fast_and_compressed_on_slow():
    codecs = _make_codecs()
    est = LinkEstimator()
    policy = {
        "mode": "adaptive",
        "ladder": ["bf16", "int8", "pca"],
        "target_ms": 50.0,
        "prefill_only": False,
    }
    chooser = CodecChooser(policy, codecs, est)

    raw_bytes = 600 * 1024  # 600 KB prefill = 614_400 B

    # Fast link: 1e9 B/s
    est.observe("fast_peer", sent_bytes=1_000_000, transport_ms=1.0)
    # bf16 estimated_ms = (614_400 * 0.5 / 1e9) * 1000 = 0.3072 ms <= 50.0 ms
    assert chooser.choose("fast_peer", raw_bytes=raw_bytes, seq_len=10) == "bf16"

    # Slow link: 1e5 B/s
    est.observe("slow_peer", sent_bytes=100_000, transport_ms=1000.0)
    # At 1e5 B/s:
    # bf16: 614_400 * 0.5 / 1e5 * 1000 = 3072 ms > 50
    # int8: 614_400 * 0.27 / 1e5 * 1000 = 1658.88 ms > 50
    # pca: 614_400 * 0.125 / 1e5 * 1000 = 768 ms > 50
    # None fits target_ms -> fallback to ladder[-1] = "pca" (most compressed)
    assert chooser.choose("slow_peer", raw_bytes=raw_bytes, seq_len=10) == "pca"


def test_codec_chooser_prefill_only():
    codecs = _make_codecs()
    est = LinkEstimator()
    # On slow link, without prefill_only: picks compressed profile
    policy_no_prefill = {
        "mode": "adaptive",
        "ladder": ["bf16", "int8", "pca"],
        "target_ms": 50.0,
        "prefill_only": False,
    }
    chooser_no_prefill = CodecChooser(policy_no_prefill, codecs, est)
    est.observe("slow_peer", sent_bytes=100_000, transport_ms=1000.0)
    assert chooser_no_prefill.choose("slow_peer", raw_bytes=600 * 1024, seq_len=1) == "pca"

    # With prefill_only=True: seq_len <= 1 always picks ladder[0]
    policy_prefill = {
        "mode": "adaptive",
        "ladder": ["bf16", "int8", "pca"],
        "target_ms": 50.0,
        "prefill_only": True,
    }
    chooser_prefill = CodecChooser(policy_prefill, codecs, est)
    assert chooser_prefill.choose("slow_peer", raw_bytes=600 * 1024, seq_len=1) == "bf16"
    assert chooser_prefill.choose("slow_peer", raw_bytes=600 * 1024, seq_len=0) == "bf16"
    # For seq_len > 1 with prefill_only=True on slow link, still uses adaptive compression
    assert chooser_prefill.choose("slow_peer", raw_bytes=600 * 1024, seq_len=2) == "pca"


def test_codec_chooser_unknown_bandwidth_returns_ladder_0():
    codecs = _make_codecs()
    est = LinkEstimator()
    policy = {
        "mode": "adaptive",
        "ladder": ["bf16", "int8", "pca"],
        "target_ms": 50.0,
    }
    chooser = CodecChooser(policy, codecs, est)
    assert chooser.choose("unknown_peer", raw_bytes=600 * 1024, seq_len=10) == "bf16"


def test_codec_chooser_learn_updates_ratio_and_choice():
    codecs = _make_codecs()
    est = LinkEstimator()
    policy = {
        "mode": "adaptive",
        "ladder": ["bf16", "int8"],
        "target_ms": 50.0,
    }
    chooser = CodecChooser(policy, codecs, est)
    est.observe("peer_mid", sent_bytes=350_000, transport_ms=1000.0)
    # Before learning: 100 KB * 0.27 / 350 KB/s = 77 ms > 50 ms, so the most compressed
    # profile is the fallback and the 0.27 guess does not fit either.
    assert chooser.choose("peer_mid", raw_bytes=100_000, seq_len=10) == "int8"
    assert 100_000 * chooser.ratio("int8") / 350_000 * 1000 > 50.0

    # Initial default int8 ratio is 0.27
    assert chooser.ratio("int8") == pytest.approx(0.27)

    # First learn sample sets ratio directly
    chooser.learn("int8", sent_bytes=2000, uncompressed_bytes=10_000)
    assert chooser.ratio("int8") == pytest.approx(0.2)

    # Second learn sample smoothed with SMOOTHING (0.3)
    chooser.learn("int8", sent_bytes=1000, uncompressed_bytes=10_000)
    expected = (1.0 - SMOOTHING) * 0.2 + SMOOTHING * 0.1
    assert chooser.ratio("int8") == pytest.approx(expected)

    # Set up a peer bandwidth where the learned ratio alters the choice
    # raw_bytes = 100_000, bw = 350_000 B/s
    # Before learn: ratio(int8) was 0.27 -> 100_000 * 0.27 / 350_000 * 1000 = 77.14 ms > 50 ms
    # With learned ratio 0.17 -> 100_000 * 0.17 / 350_000 * 1000 = 48.57 ms <= 50 ms -> fits!
    assert 100_000 * chooser.ratio("int8") / 350_000 * 1000 <= 50.0
    assert chooser.choose("peer_mid", raw_bytes=100_000, seq_len=10) == "int8"


def test_learning_moves_the_choice_off_the_fallback():
    """With a ladder of three, a learned ratio makes a middle profile fit where the
    default guess sent everything to the most compressed one."""
    codecs = _make_codecs()
    est = LinkEstimator()
    chooser = CodecChooser(
        {"mode": "adaptive", "ladder": ["bf16", "int8", "pca"], "target_ms": 50.0}, codecs, est
    )
    est.observe("p", sent_bytes=350_000, transport_ms=1000.0)
    assert chooser.choose("p", raw_bytes=100_000, seq_len=10) == "pca"
    chooser.learn("int8", sent_bytes=1000, uncompressed_bytes=10_000)
    assert chooser.choose("p", raw_bytes=100_000, seq_len=10) == "int8"


@pytest.mark.parametrize("target", [float("nan"), -1.0, 0, "50", True])
def test_bad_target_ms_is_refused(target):
    with pytest.raises(ValueError, match="target_ms"):
        CodecChooser(
            {"mode": "adaptive", "ladder": ["bf16"], "target_ms": target},
            _make_codecs(),
            LinkEstimator(),
        )


def test_codec_chooser_learn_ignores_bad_and_non_finite_inputs():
    codecs = _make_codecs()
    est = LinkEstimator()
    chooser = CodecChooser({"mode": "fixed", "fixed": "int8", "ladder": ["int8"]}, codecs, est)
    initial_ratio = chooser.ratio("int8")
    assert initial_ratio == pytest.approx(0.27)

    # uncompressed_bytes <= 0
    chooser.learn("int8", sent_bytes=100, uncompressed_bytes=0)
    assert chooser.ratio("int8") == initial_ratio
    chooser.learn("int8", sent_bytes=100, uncompressed_bytes=-10)
    assert chooser.ratio("int8") == initial_ratio

    # negative sent_bytes
    chooser.learn("int8", sent_bytes=-50, uncompressed_bytes=100)
    assert chooser.ratio("int8") == initial_ratio

    # non-finite inputs: NaN and inf
    chooser.learn("int8", float("nan"), 100)
    assert chooser.ratio("int8") == initial_ratio
    chooser.learn("int8", 10, float("inf"))
    assert chooser.ratio("int8") == initial_ratio
    chooser.learn("int8", float("-inf"), 100)
    assert chooser.ratio("int8") == initial_ratio
    chooser.learn("int8", 10, float("nan"))
    assert chooser.ratio("int8") == initial_ratio

    # Booleans ignored
    chooser.learn("int8", True, 100)  # type: ignore[arg-type]
    assert chooser.ratio("int8") == initial_ratio
    chooser.learn("int8", 100, False)  # type: ignore[arg-type]
    assert chooser.ratio("int8") == initial_ratio


def test_codec_chooser_choose_raw_bytes_validation():
    codecs = _make_codecs()
    est = LinkEstimator()
    policy = {
        "mode": "adaptive",
        "ladder": ["bf16", "int8"],
        "target_ms": 50.0,
    }
    chooser = CodecChooser(policy, codecs, est)
    est.observe("fast_peer", sent_bytes=1_000_000, transport_ms=1.0)

    # Non-finite or negative raw_bytes explicitly returns ladder[0]
    assert chooser.choose("fast_peer", raw_bytes=-100, seq_len=10) == "bf16"
    assert chooser.choose("fast_peer", raw_bytes=float("nan"), seq_len=10) == "bf16"  # type: ignore[arg-type]
    assert chooser.choose("fast_peer", raw_bytes=float("inf"), seq_len=10) == "bf16"  # type: ignore[arg-type]
    assert chooser.choose("fast_peer", raw_bytes=True, seq_len=10) == "bf16"  # type: ignore[arg-type]


def test_codec_chooser_validations():
    codecs = _make_codecs()
    est = LinkEstimator()

    # Empty ladder
    with pytest.raises(ValueError, match="ladder must be non-empty"):
        CodecChooser({"mode": "fixed", "ladder": []}, codecs, est)

    # Unknown profile in ladder
    with pytest.raises(ValueError, match="unknown profile"):
        CodecChooser({"mode": "fixed", "ladder": ["nonexistent"]}, codecs, est)

    # Unknown profile in fixed
    with pytest.raises(ValueError, match="unknown profile"):
        CodecChooser({"mode": "fixed", "fixed": "nonexistent", "ladder": ["bf16"]}, codecs, est)

    # Unknown mode (e.g., typo 'adaptiv')
    with pytest.raises(ValueError, match="fixed or adaptive"):
        CodecChooser({"mode": "adaptiv", "ladder": ["bf16"]}, codecs, est)

    # Lowrank profile with invalid rank in ladder -> ValueError naming profile
    bad_codecs = CodecSet.from_config(
        {
            "bad_pca": {"name": "lowrank", "rank": 0},
            "str_pca": {"name": "lowrank", "rank": "invalid"},
            "none_pca": {"name": "lowrank", "rank": None},
        }
    )
    with pytest.raises(ValueError, match="bad_pca.*invalid rank"):
        CodecChooser({"mode": "adaptive", "ladder": ["bad_pca"]}, bad_codecs, est)

    with pytest.raises(ValueError, match="str_pca.*invalid rank"):
        CodecChooser({"mode": "adaptive", "ladder": ["str_pca"]}, bad_codecs, est)

    with pytest.raises(ValueError, match="none_pca.*invalid rank"):
        CodecChooser({"mode": "adaptive", "ladder": ["none_pca"]}, bad_codecs, est)


def test_codec_chooser_ratio_lowrank_fallback():
    # If ratio() is called on a lowrank profile whose rank is not a positive int
    bad_profile = CodecProfile("odd_pca", "lowrank", {"rank": -10, "wire_dtype": "float16"})
    codecs = CodecSet([bad_profile, CodecProfile("bf16", "raw")])
    est = LinkEstimator()
    chooser = CodecChooser({"mode": "fixed", "fixed": "bf16", "ladder": ["bf16"]}, codecs, est)

    # A broken profile is an error that names it, never a silent guess (a rank of 0 or
    # less would make it look free and always "fit").
    with pytest.raises(ValueError, match="odd_pca"):
        chooser.ratio("odd_pca")
