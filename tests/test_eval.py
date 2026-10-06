"""Tests for the compression-quality evaluation harness (Issue #10).

All tests run 100% offline using the random tiny model fixture from conftest.py,
executing in seconds without downloading checkpoints or external datasets.
"""

from __future__ import annotations

import pytest
import torch
from scripts.eval_quality import build_arg_parser, compute_projected_speedup, parse_ranks

from torrent_llm.codec import get_codec
from torrent_llm.eval import (
    GSM8KEvaluator,
    LayerSensitivityAnalyzer,
    MMLUEvaluator,
    RateDistortionRow,
    SimpleCharTokenizer,
    analyze_layer_sensitivity,
    compute_perplexity,
    evaluate_downstream,
    evaluate_perplexity_across_ranks,
    extract_gsm8k_answer,
    extract_mmlu_choice,
    format_rate_distortion_table,
    generate_synthetic_tokens,
    get_synthetic_corpus,
)
from torrent_llm.runner import ChainRunner, TopologyConfig
from torrent_llm.shard import ShardRuntime
from torrent_llm.transport import serve


@pytest.fixture
def served_eval_chains(model_factory, num_layers):
    """Start two 2-shard chains: one raw (uncompressed) and one lowrank (rank=32)."""
    # 1. Raw chain
    base_raw = TopologyConfig.from_dict(
        {
            "model_id": "tiny",
            "num_layers": num_layers,
            "nodes": [{"address": "127.0.0.1:0"}, {"address": "127.0.0.1:0"}],
        }
    )
    raw_servers, raw_addrs = [], []
    for spec in base_raw.shard_plan():
        runtime = ShardRuntime(model_factory(), spec)
        server, port = serve(runtime, get_codec("raw"), host="127.0.0.1", model_id="tiny")
        raw_servers.append(server)
        raw_addrs.append(f"127.0.0.1:{port}")

    raw_config = TopologyConfig.from_dict(
        {
            "model_id": "tiny",
            "num_layers": num_layers,
            "nodes": [{"address": a} for a in raw_addrs],
        }
    )

    # 2. LowRank chain (rank=32)
    lr_codec_args = {"rank": 32, "seed": 42}
    lr_servers, lr_addrs = [], []
    for spec in base_raw.shard_plan():
        runtime = ShardRuntime(model_factory(), spec)
        server, port = serve(
            runtime, get_codec("lowrank", **lr_codec_args), host="127.0.0.1", model_id="tiny"
        )
        lr_servers.append(server)
        lr_addrs.append(f"127.0.0.1:{port}")

    lr_config = TopologyConfig.from_dict(
        {
            "model_id": "tiny",
            "num_layers": num_layers,
            "nodes": [{"address": a} for a in lr_addrs],
            "codec": {"name": "lowrank", **lr_codec_args},
        }
    )

    yield raw_config, lr_config

    for s in raw_servers + lr_servers:
        s.stop(grace=None)


def test_synthetic_corpus_and_token_generation():
    corpus = get_synthetic_corpus()
    assert len(corpus) > 0
    assert all(isinstance(text, str) and len(text) > 20 for text in corpus)

    tokens = generate_synthetic_tokens(vocab_size=256, seq_len=32, num_sequences=3, seed=123)
    assert len(tokens) == 3
    for t in tokens:
        assert t.shape == (1, 32)
        assert t.dtype == torch.long
        assert (t >= 0).all() and (t < 256).all()


def test_simple_char_tokenizer():
    tok = SimpleCharTokenizer(vocab_size=256)
    text = "Hello Torrent-LLM!"
    batch = tok(text, return_tensors="pt")
    assert hasattr(batch, "input_ids")
    assert batch.input_ids.shape == (1, len(text))
    decoded = tok.decode(batch.input_ids)
    assert decoded == text


def test_perplexity_calculation_uncompressed(served_eval_chains):
    raw_config, _ = served_eval_chains
    tokens = generate_synthetic_tokens(vocab_size=256, seq_len=48, num_sequences=2)

    with ChainRunner(raw_config) as runner:
        res = compute_perplexity(runner, tokens, max_chunk_size=32)

    assert res.codec == "raw"
    assert res.rank is None
    assert res.total_tokens == (48 - 1) * 2
    assert res.num_chunks > 0
    assert res.loss > 0.0
    assert res.perplexity >= 1.0
    assert res.wall_time_s >= 0.0


def test_perplexity_sliding_window_vs_chunked(served_eval_chains):
    raw_config, _ = served_eval_chains
    tokens = generate_synthetic_tokens(vocab_size=256, seq_len=60, num_sequences=1)

    with ChainRunner(raw_config) as runner:
        # Non-overlapping chunked
        chunked_res = compute_perplexity(runner, tokens, max_chunk_size=30, stride=None)
        # Sliding-window with stride=10
        sliding_res = compute_perplexity(runner, tokens, max_chunk_size=30, stride=10)

    assert chunked_res.loss > 0.0
    assert sliding_res.loss > 0.0
    assert sliding_res.num_chunks > chunked_res.num_chunks
    assert sliding_res.total_tokens == chunked_res.total_tokens


def test_perplexity_comparison_across_ranks(served_eval_chains):
    raw_config, lr_config = served_eval_chains
    tokens = generate_synthetic_tokens(vocab_size=256, seq_len=32, num_sequences=1)

    with ChainRunner(raw_config) as raw_runner, ChainRunner(lr_config) as lr_runner:
        runners = {"raw": raw_runner, 32: lr_runner}
        results = evaluate_perplexity_across_ranks(runners, tokens, max_chunk_size=32)

    assert "raw" in results
    assert 32 in results
    assert results["raw"].codec == "raw"
    assert results[32].codec == "lowrank"
    assert results[32].rank == 32
    assert results["raw"].loss > 0.0
    assert results[32].loss > 0.0


def test_layer_sensitivity_analyzer_end_to_end(served_eval_chains):
    raw_config, lr_config = served_eval_chains
    input_ids = torch.randint(0, 256, (1, 24), dtype=torch.long)

    with ChainRunner(raw_config) as raw_runner, ChainRunner(lr_config) as lr_runner:
        analyzer = LayerSensitivityAnalyzer(seed=42)
        report = analyzer.compare_chain_runs(raw_runner, lr_runner, input_ids)

    assert report.model_id == "tiny"
    assert report.rank == 32
    assert report.baseline_loss > 0.0
    assert report.compressed_loss > 0.0
    assert 0.0 <= report.top1_agreement <= 1.0
    assert report.kl_divergence >= 0.0
    assert len(report.hops) == 2

    # Intermediate hop 0 hidden state comparison
    hop0 = report.hops[0]
    assert hop0.hop == 0
    assert -1.0 <= hop0.cosine_similarity <= 1.0
    assert hop0.l2_relative_error >= 0.0
    assert hop0.mse >= 0.0

    report_dict = report.to_dict()
    assert "hops" in report_dict
    assert len(report_dict["hops"]) == 2


def test_layer_sensitivity_isolated_hop(served_eval_chains):
    raw_config, _ = served_eval_chains
    input_ids = torch.randint(0, 256, (1, 20), dtype=torch.long)

    with ChainRunner(raw_config) as raw_runner:
        report = analyze_layer_sensitivity(
            raw_runner, comp_runner=None, input_ids=input_ids, rank=32
        )

    assert report.model_id == "tiny"
    assert report.rank == 32
    assert len(report.hops) == 2
    # Hop 0 was perturbed with lowrank projection
    hop0 = report.hops[0]
    assert hop0.immediate_cosine_similarity is not None
    assert hop0.isolated_loss is not None
    assert hop0.isolated_loss_delta is not None


def test_downstream_mmlu_log_likelihood(served_eval_chains):
    raw_config, _ = served_eval_chains
    tok = SimpleCharTokenizer(vocab_size=256)

    with ChainRunner(raw_config) as runner:
        evaluator = MMLUEvaluator(eval_mode="log_likelihood")
        result = evaluator.evaluate(runner, tokenizer=tok, max_samples=3)

    assert result.benchmark_name == "mmlu-sample"
    assert result.num_samples == 3
    assert 0 <= result.correct_samples <= 3
    assert 0.0 <= result.accuracy <= 1.0
    assert result.eval_mode == "log_likelihood"
    assert len(result.items) == 3
    for it in result.items:
        assert it.prediction in ("A", "B", "C", "D")
        assert it.score in (0.0, 1.0)


def test_downstream_mmlu_greedy_generation(served_eval_chains):
    raw_config, _ = served_eval_chains
    tok = SimpleCharTokenizer(vocab_size=256)

    with ChainRunner(raw_config) as runner:
        evaluator = MMLUEvaluator(eval_mode="greedy_generation")
        result = evaluator.evaluate(runner, tokenizer=tok, max_samples=2)

    assert result.benchmark_name == "mmlu-sample"
    assert result.num_samples == 2
    assert result.eval_mode == "greedy_generation"


def test_downstream_gsm8k_evaluator(served_eval_chains):
    raw_config, _ = served_eval_chains
    tok = SimpleCharTokenizer(vocab_size=256)

    with ChainRunner(raw_config) as runner:
        evaluator = GSM8KEvaluator(max_new_tokens=4)
        result = evaluator.evaluate(runner, tokenizer=tok, max_samples=2)

    assert result.benchmark_name == "gsm8k-sample"
    assert result.num_samples == 2
    assert result.eval_mode == "greedy_generation"
    assert len(result.items) == 2


def test_evaluate_downstream_all(served_eval_chains):
    raw_config, _ = served_eval_chains
    tok = SimpleCharTokenizer(vocab_size=256)

    with ChainRunner(raw_config) as runner:
        res_dict = evaluate_downstream(runner, benchmark="all", tokenizer=tok, max_samples=2)

    assert "gsm8k-sample" in res_dict
    assert "mmlu-sample" in res_dict


def test_regex_extractors():
    # GSM8K
    assert extract_gsm8k_answer("So the total is 12 - 5 = 7. #### 7") == "7"
    assert extract_gsm8k_answer("The final answer is 42.") == "42"
    assert extract_gsm8k_answer("Cost = -15") == "-15"
    assert extract_gsm8k_answer("She made $1,250 in profit.") == "1250"
    assert extract_gsm8k_answer("No numbers here") == ""

    # MMLU
    assert extract_mmlu_choice("Answer: B") == "B"
    assert extract_mmlu_choice("  C) is correct") == "C"
    assert extract_mmlu_choice("A") == "A"
    assert extract_mmlu_choice("Option D seems right") == "D"
    assert extract_mmlu_choice("unknown") == ""


def test_rate_distortion_table_formatting():
    rows = [
        RateDistortionRow(
            rank_label="raw",
            compression_ratio=1.0,
            perplexity=14.23,
            downstream_accuracy=0.70,
            gsm8k_acc=0.60,
            mmlu_acc=0.80,
            speedup=1.00,
        ),
        RateDistortionRow(
            rank_label="128",
            compression_ratio=8.0,
            perplexity=15.10,
            downstream_accuracy=0.65,
            gsm8k_acc=0.50,
            mmlu_acc=0.80,
            speedup=2.15,
        ),
    ]

    table_str = format_rate_distortion_table(
        rows,
        model_id="tiny",
        num_shards=2,
        link_mbps=100.0,
        rtt_ms=20.0,
    )
    assert "Rate-Distortion Evaluation Summary" in table_str
    assert "tiny" in table_str
    assert "raw" in table_str
    assert "128" in table_str
    assert "14.23" in table_str
    assert "15.10" in table_str
    assert "2.15x" in table_str


def test_cli_argument_parser():
    parser = build_arg_parser()
    args = parser.parse_args(
        ["--model", "tiny", "--ranks", "raw,64,32", "--benchmark", "perplexity"]
    )
    assert args.model == "tiny"
    assert args.ranks == "raw,64,32"
    assert args.benchmark == "perplexity"

    ranks = parse_ranks(args.ranks)
    assert ranks == ["raw", 64, 32]

    # Invalid rank
    with pytest.raises(ValueError):
        parse_ranks("raw,invalid,32")


def test_compute_projected_speedup():
    ratio_raw, speedup_raw = compute_projected_speedup("Qwen/Qwen3-0.6B", "raw", 100.0, 20.0)
    assert ratio_raw == 1.0
    assert speedup_raw == 1.0

    ratio_128, speedup_128 = compute_projected_speedup("Qwen/Qwen3-0.6B", 128, 100.0, 20.0)
    assert ratio_128 == 8.0  # 1024 / 128
    assert speedup_128 > 1.0
