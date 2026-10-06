"""Quality and compression evaluation harness for Torrent-LLM (Issue #10).

Provides tools to characterize the Rate-Distortion frontier across
varying activation compression ranks:
- Memory-safe chunked & sliding-window cross-entropy loss and perplexity.
- Per-layer-depth sensitivity and hidden-state distortion analysis.
- Zero-dependency downstream reasoning evaluation (GSM8K, MMLU).
- Rate-Distortion table reporting and JSONL metrics export.
"""

from torrent_llm.eval.downstream import (
    GSM8K_SAMPLE_DATASET,
    MMLU_SAMPLE_DATASET,
    BenchmarkItemResult,
    DownstreamResult,
    GSM8KEvaluator,
    MMLUEvaluator,
    SimpleCharTokenizer,
    evaluate_downstream,
    extract_gsm8k_answer,
    extract_mmlu_choice,
)
from torrent_llm.eval.perplexity import (
    SYNTHETIC_EVAL_CORPUS,
    PerplexityResult,
    compute_perplexity,
    evaluate_perplexity_across_ranks,
    generate_synthetic_tokens,
    get_synthetic_corpus,
)
from torrent_llm.eval.sensitivity import (
    HopSensitivity,
    LayerSensitivityAnalyzer,
    SensitivityReport,
    analyze_layer_sensitivity,
)
from torrent_llm.eval.table import (
    RateDistortionRow,
    format_rate_distortion_table,
)

__all__ = [
    "GSM8K_SAMPLE_DATASET",
    "MMLU_SAMPLE_DATASET",
    "SYNTHETIC_EVAL_CORPUS",
    "BenchmarkItemResult",
    "DownstreamResult",
    "GSM8KEvaluator",
    "HopSensitivity",
    "LayerSensitivityAnalyzer",
    "MMLUEvaluator",
    "PerplexityResult",
    "RateDistortionRow",
    "SensitivityReport",
    "SimpleCharTokenizer",
    "analyze_layer_sensitivity",
    "compute_perplexity",
    "evaluate_downstream",
    "evaluate_perplexity_across_ranks",
    "extract_gsm8k_answer",
    "extract_mmlu_choice",
    "format_rate_distortion_table",
    "generate_synthetic_tokens",
    "get_synthetic_corpus",
]
