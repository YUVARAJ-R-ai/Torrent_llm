"""Profiler correctness.

These numbers are the project's evidence, so the arithmetic behind them is
tested as carefully as the model code.
"""

import pytest

from torrent_llm.profile import (
    HopBudget,
    HopProfiler,
    HopRecord,
    RunMetadata,
    activation_bytes,
    bandwidth_verdict,
    decode_budget,
    format_table,
    load_records,
    prefill_budget,
    summarize_by_hop,
    transfer_ms,
)


def make_record(**overrides) -> HopRecord:
    defaults = dict(
        request_id="r",
        hop=0,
        codec="raw",
        address="127.0.0.1:1",
        batch=1,
        seq_len=128,
        hidden_size=1024,
        dtype="bfloat16",
        sent_bytes=128 * 1024 * 2,
        received_bytes=128 * 1024 * 2,
        uncompressed_bytes=128 * 1024 * 2,
        wall_ns=10_000_000,
        compute_ns=4_000_000,
    )
    return HopRecord(**{**defaults, **overrides})


# --- record arithmetic ---


def test_transport_is_wall_minus_remote_compute():
    record = make_record(wall_ns=10_000_000, compute_ns=4_000_000)

    assert record.transport_ns == 6_000_000
    assert record.transport_share == pytest.approx(0.6)


def test_transport_never_goes_negative_on_clock_noise():
    # The peer's timer can report slightly more than our round trip on a fast
    # loopback hop. That is measurement noise, not a negative network cost.
    record = make_record(wall_ns=1_000, compute_ns=5_000)

    assert record.transport_ns == 0
    assert record.transport_share == 0.0


def test_compression_ratio_is_uncompressed_over_wire():
    record = make_record(sent_bytes=1024, uncompressed_bytes=8192)

    assert record.compression_ratio == 8.0


def test_raw_hop_reports_ratio_one():
    assert make_record().compression_ratio == 1.0


def test_bytes_per_token_normalises_across_context_lengths():
    short = make_record(seq_len=128, sent_bytes=128 * 1024 * 2)
    long = make_record(seq_len=512, sent_bytes=512 * 1024 * 2)

    assert short.bytes_per_token == long.bytes_per_token == 2048


def test_effective_mbps_matches_payload_over_transport_time():
    # 1 MiB in 10 ms is ~839 Mbps.
    record = make_record(sent_bytes=1024 * 1024, wall_ns=10_000_000, compute_ns=0)

    assert record.effective_mbps == pytest.approx(838.86, rel=1e-3)


# --- persistence ---


def test_records_round_trip_through_jsonl(tmp_path):
    path = tmp_path / "run.jsonl"
    metadata = RunMetadata(
        run_id="abc", model_id="tiny", dtype="float32", codec="raw", num_shards=2
    )
    with HopProfiler(path=path, metadata=metadata) as profiler:
        profiler.records.append(make_record(hop=0))
        # write directly so the file mirrors what record() would emit
        profiler._handle.write(make_record(hop=0).to_json() + "\n")
        profiler._handle.write(make_record(hop=1).to_json() + "\n")

    loaded_meta, loaded = load_records(path)

    assert loaded_meta.run_id == "abc"
    assert loaded_meta.model_id == "tiny"
    assert [r.hop for r in loaded] == [0, 1]
    assert loaded[0].sent_bytes == make_record().sent_bytes


def test_derived_fields_are_written_for_downstream_tools(tmp_path):
    import json

    path = tmp_path / "run.jsonl"
    with HopProfiler(path=path) as profiler:
        profiler._handle.write(make_record().to_json() + "\n")

    row = json.loads(path.read_text().strip())
    # A dataframe should not have to recompute these to group by them.
    assert row["transport_ns"] == 6_000_000
    assert row["compression_ratio"] == 1.0
    assert "transport_share" in row


def test_profiler_without_a_path_still_collects_in_memory():
    profiler = HopProfiler()

    assert profiler.records == []
    assert profiler.path is None


# --- summarising ---


def test_summary_groups_by_hop():
    records = [make_record(hop=0), make_record(hop=0), make_record(hop=1)]

    summaries = summarize_by_hop(records)

    assert [s.hop for s in summaries] == [0, 1]
    assert [s.samples for s in summaries] == [2, 1]


def test_summary_uses_medians_so_one_outlier_does_not_dominate():
    records = [
        make_record(wall_ns=10_000_000),
        make_record(wall_ns=10_000_000),
        make_record(wall_ns=900_000_000),  # a scheduling hiccup
    ]

    (summary,) = summarize_by_hop(records)

    assert summary.median_wall_ms == pytest.approx(10.0)


def test_verdict_calls_out_a_compute_bound_regime():
    records = [make_record(wall_ns=10_000_000, compute_ns=9_500_000)]

    verdict = bandwidth_verdict(summarize_by_hop(records))

    assert verdict.startswith("compute-bound")
    assert "cannot improve end-to-end latency" in verdict


def test_verdict_calls_out_a_bandwidth_bound_regime():
    records = [make_record(wall_ns=10_000_000, compute_ns=500_000)]

    verdict = bandwidth_verdict(summarize_by_hop(records))

    assert verdict.startswith("bandwidth-bound")


def test_verdict_refuses_to_average_away_a_mixed_regime():
    records = [
        make_record(hop=0, wall_ns=10_000_000, compute_ns=9_500_000),
        make_record(hop=1, wall_ns=10_000_000, compute_ns=500_000),
    ]

    assert bandwidth_verdict(summarize_by_hop(records)).startswith("mixed")


def test_table_renders_a_row_per_hop():
    table = format_table(summarize_by_hop([make_record(hop=0), make_record(hop=1)]))

    assert len(table.splitlines()) == 4  # header, rule, two rows
    assert "transport_share" in table


def test_empty_summary_is_stated_not_crashed():
    assert format_table([]) == "(no hops recorded)"
    assert bandwidth_verdict([]) == "no data"


# --- projection ---


def test_activation_bytes_is_the_obvious_product():
    assert activation_bytes(4096, 2048, dtype="bfloat16") == 4096 * 2048 * 2


def test_projection_reproduces_the_seventy_b_figure():
    # 70B, hidden 8192, 2k context, bf16 -> 32 MiB per hop.
    assert activation_bytes(8192, 2048, dtype="bfloat16") == 32 * 1024 * 1024


def test_unknown_dtype_is_rejected_rather_than_guessed():
    with pytest.raises(ValueError, match="unknown dtype"):
        activation_bytes(4096, 128, dtype="int4")


def test_transfer_time_matches_the_link_rate():
    # 100 Mbps moves 100 Mbit in 1000 ms; 12.5 MB is 100 Mbit.
    assert transfer_ms(12_500_000, 100) == pytest.approx(1000.0)


def test_prefill_on_a_consumer_link_is_bandwidth_bound():
    budget = prefill_budget(hidden_size=8192, seq_len=2048, link_mbps=100, rtt_ms=30, compute_ms=50)

    assert budget.is_bandwidth_bound
    assert budget.wire_share > 0.9


def test_cached_decode_on_the_same_link_is_not():
    # The regime distinction the whole profiler exists to make: one position of
    # hidden state is 16 KiB, which crosses a 100 Mbps link in ~1.3 ms against a
    # 30 ms round trip.
    budget = decode_budget(hidden_size=8192, link_mbps=100, rtt_ms=30, compute_ms=50)

    assert not budget.is_bandwidth_bound
    assert budget.wire_share < 0.05


def test_compression_helps_where_the_hop_is_bandwidth_bound():
    base = prefill_budget(hidden_size=8192, seq_len=2048, link_mbps=100, rtt_ms=30, compute_ms=50)

    compressed = base.with_compression(8.0, codec_ms=5.0)

    assert compressed.speedup_from(base) > 3.0


def test_compression_hurts_where_the_hop_is_not():
    base = decode_budget(hidden_size=8192, link_mbps=100, rtt_ms=30, compute_ms=50)

    compressed = base.with_compression(8.0, codec_ms=5.0)

    # Saves ~1 ms of wire, pays 5 ms of codec: a net loss.
    assert compressed.speedup_from(base) < 1.0


def test_codec_cost_is_charged_not_ignored():
    base = HopBudget(payload_bytes=1_000_000, link_mbps=1000, rtt_ms=1, compute_ms=10)

    free = base.with_compression(4.0)
    costed = base.with_compression(4.0, codec_ms=3.0)

    assert costed.total_ms == pytest.approx(free.total_ms + 3.0)


def test_a_ratio_below_one_is_still_arithmetic_not_an_error():
    # A codec can inflate a payload (headers, padding). The budget should say so
    # rather than reject it.
    base = HopBudget(payload_bytes=1000, link_mbps=100, rtt_ms=1, compute_ms=1)

    assert base.with_compression(0.5).payload_bytes == 2000


def test_a_nonpositive_ratio_is_rejected():
    base = HopBudget(payload_bytes=1000, link_mbps=100, rtt_ms=1, compute_ms=1)

    with pytest.raises(ValueError, match="must be positive"):
        base.with_compression(0.0)
