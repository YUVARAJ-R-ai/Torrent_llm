"""Zero-dependency downstream reasoning benchmark evaluator (Issue #10).

Evaluates greedy generation accuracy and log-likelihood scoring for
GSM8K-style multi-step arithmetic reasoning and MMLU-style multiple-choice
reasoning benchmarks through ChainRunner across arbitrary compression ranks.
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from torrent_llm.runner import ChainRunner

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BenchmarkItemResult:
    """Outcome for a single evaluation sample."""

    item_id: str
    prompt: str
    target: str
    prediction: str
    is_correct: bool
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "item_id": self.item_id,
            "target": self.target,
            "prediction": self.prediction,
            "is_correct": self.is_correct,
            "score": self.score,
            "metadata": self.metadata,
        }


@dataclass(frozen=True)
class DownstreamResult:
    """Aggregated outcome of a downstream benchmark evaluation."""

    benchmark_name: str
    num_samples: int
    correct_samples: int
    accuracy: float
    eval_mode: str
    codec: str
    rank: int | None = None
    items: list[BenchmarkItemResult] = field(default_factory=list)
    wall_time_s: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "benchmark_name": self.benchmark_name,
            "num_samples": self.num_samples,
            "correct_samples": self.correct_samples,
            "accuracy": round(self.accuracy, 4),
            "eval_mode": self.eval_mode,
            "codec": self.codec,
            "rank": self.rank,
            "wall_time_s": round(self.wall_time_s, 4),
            "items": [item.to_dict() for item in self.items],
        }


class SimpleCharTokenizer:
    """Minimal zero-dependency ASCII tokenizer for offline tests with tiny models."""

    def __init__(self, vocab_size: int = 256) -> None:
        self.vocab_size = vocab_size

    def __call__(self, text: str, return_tensors: str | None = None) -> Any:
        return self.encode(text, return_tensors=return_tensors)

    def encode(self, text: str, return_tensors: str | None = None) -> Any:
        token_ids = [min(ord(c), self.vocab_size - 1) for c in text]
        if return_tensors == "pt":
            class _Batch:
                def __init__(self, ids: list[int]) -> None:
                    self.input_ids = torch.tensor([ids], dtype=torch.long)

            return _Batch(token_ids)
        return token_ids

    def decode(self, token_ids: list[int] | torch.Tensor, skip_special_tokens: bool = True) -> str:
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.view(-1).tolist()
        return "".join(chr(t) if 32 <= t <= 126 or t == 10 else " " for t in token_ids)


GSM8K_FEW_SHOT_HEADER = (
    "Question: There are 15 trees in the grove. Grove workers will plant trees in the "
    "grove today. After they are done, there will be 21 trees. How many trees did "
    "the grove workers plant today?\n"
    "Answer: There are 15 trees originally. Then there were 21 trees after some more "
    "were planted. So there must have been 21 - 15 = 6. #### 6\n\n"
    "Question: If there are 3 cars in the parking lot and 2 more cars arrive, "
    "how many cars are in the parking lot?\n"
    "Answer: There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5. #### 5\n\n"
)

GSM8K_SAMPLE_DATASET: list[dict[str, Any]] = [
    {
        "id": "gsm8k-1",
        "question": (
            "Janet’s ducks lay 16 eggs per day. She eats three for breakfast every morning "
            "and bakes muffins with four. She sells the remainder at the market for $2 per egg. "
            "How much does she make daily?"
        ),
        "gold": "18",
        "solution": "16 - 3 - 4 = 9 eggs remaining. 9 * 2 = 18. #### 18",
    },
    {
        "id": "gsm8k-2",
        "question": (
            "A robe takes 2 bolts of blue fiber and half that much white fiber. "
            "How many bolts in total does it take?"
        ),
        "gold": "3",
        "solution": "White fiber = 2 / 2 = 1 bolt. Total = 2 + 1 = 3 bolts. #### 3",
    },
    {
        "id": "gsm8k-3",
        "question": (
            "Josh decides to try flipping a house. He buys a house for $80,000 and spends "
            "$50,000 on renovations. He then sells it for $150,000. How much profit does he make?"
        ),
        "gold": "20000",
        "solution": (
            "Total cost = 80000 + 50000 = 130000. Profit = 150000 - 130000 = 20000. #### 20000"
        ),
    },
    {
        "id": "gsm8k-4",
        "question": (
            "James runs 3 miles a day for 4 days a week. How many miles does he run in 2 weeks?"
        ),
        "gold": "24",
        "solution": "Miles per week = 3 * 4 = 12 miles. In 2 weeks = 12 * 2 = 24 miles. #### 24",
    },
    {
        "id": "gsm8k-5",
        "question": (
            "Every day, Wendi feeds each of her chickens 3 cups of feed. If she has 20 chickens, "
            "how many cups of feed does she need in 7 days?"
        ),
        "gold": "420",
        "solution": "Daily feed = 20 * 3 = 60 cups. In 7 days = 60 * 7 = 420 cups. #### 420",
    },
]

MMLU_SAMPLE_DATASET: list[dict[str, Any]] = [
    {
        "id": "mmlu-cs-1",
        "subject": "computer_science",
        "question": (
            "What is the worst-case time complexity of searching an element in a "
            "balanced binary search tree with n nodes?"
        ),
        "choices": ["A) O(1)", "B) O(log n)", "C) O(n)", "D) O(n log n)"],
        "gold": "B",
    },
    {
        "id": "mmlu-math-1",
        "subject": "elementary_math",
        "question": "If 3x + 5 = 20, what is the value of x?",
        "choices": ["A) 3", "B) 4", "C) 5", "D) 6"],
        "gold": "C",
    },
    {
        "id": "mmlu-physics-1",
        "subject": "physics",
        "question": "Which of the following physical quantities is a vector quantity?",
        "choices": ["A) Speed", "B) Temperature", "C) Velocity", "D) Mass"],
        "gold": "C",
    },
    {
        "id": "mmlu-general-1",
        "subject": "general_knowledge",
        "question": "What is the capital city of Australia?",
        "choices": ["A) Sydney", "B) Melbourne", "C) Brisbane", "D) Canberra"],
        "gold": "D",
    },
    {
        "id": "mmlu-logic-1",
        "subject": "formal_logic",
        "question": "If P implies Q, and Q is false, what must be true about P?",
        "choices": [
            "A) P is true",
            "B) P is false",
            "C) P is indeterminate",
            "D) P is equivalent to Q",
        ],
        "gold": "B",
    },
]


def extract_gsm8k_answer(text: str) -> str:
    """Extract final numerical answer from model output."""
    # Look for #### <number>
    match = re.search(r"####\s*(-?[\d,]+(?:\.\d+)?)", text)
    if match:
        return match.group(1).replace(",", "").strip()

    # Look for explicit statement "answer is X"
    match = re.search(r"(?:answer is|=)\s*(-?[\d,]+(?:\.\d+)?)", text, re.IGNORECASE)
    if match:
        return match.group(1).replace(",", "").strip()

    # Fallback to the last number in the text
    numbers = re.findall(r"-?[\d,]+(?:\.\d+)?", text)
    if numbers:
        return numbers[-1].replace(",", "").strip()

    return ""


def extract_mmlu_choice(text: str) -> str:
    """Extract choice letter A, B, C, or D from model text output."""
    cleaned = text.strip()
    match = re.search(r"\b([A-D])\b", cleaned)
    if match:
        return match.group(1).upper()
    if cleaned and cleaned[0].upper() in ("A", "B", "C", "D"):
        return cleaned[0].upper()
    return ""


class GSM8KEvaluator:
    """Evaluates multi-step mathematical reasoning using greedy generation."""

    def __init__(
        self,
        samples: list[dict[str, Any]] | None = None,
        max_new_tokens: int = 48,
    ) -> None:
        self.samples = samples or GSM8K_SAMPLE_DATASET
        self.max_new_tokens = max_new_tokens

    def format_prompt(self, question: str) -> str:
        return f"{GSM8K_FEW_SHOT_HEADER}Question: {question}\nAnswer:"

    def evaluate(
        self,
        runner: ChainRunner,
        tokenizer: Any = None,
        max_samples: int | None = None,
    ) -> DownstreamResult:
        tokenizer = tokenizer or SimpleCharTokenizer()
        samples_to_eval = self.samples[:max_samples] if max_samples is not None else self.samples
        items: list[BenchmarkItemResult] = []
        correct = 0
        start_time = time.perf_counter()

        for sample in samples_to_eval:
            prompt_str = self.format_prompt(sample["question"])
            encoded = tokenizer(prompt_str, return_tensors="pt")
            input_ids = encoded.input_ids if hasattr(encoded, "input_ids") else encoded

            generated_ids, _ = runner.generate(
                input_ids,
                max_new_tokens=self.max_new_tokens,
                use_cache=True,
            )

            # Only decode the generated continuation tokens
            new_tokens = generated_ids[0, input_ids.shape[1] :]
            pred_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
            extracted = extract_gsm8k_answer(pred_text)
            gold = str(sample["gold"]).strip()

            is_correct = (extracted == gold) or (
                extracted != "" and abs(float(extracted or 0) - float(gold or 0)) < 1e-4
                if extracted.replace(".", "", 1).isdigit() and gold.replace(".", "", 1).isdigit()
                else False
            )

            if is_correct:
                correct += 1

            items.append(
                BenchmarkItemResult(
                    item_id=sample["id"],
                    prompt=prompt_str,
                    target=gold,
                    prediction=extracted or pred_text.strip(),
                    is_correct=is_correct,
                    score=1.0 if is_correct else 0.0,
                    metadata={"generated_text": pred_text.strip()},
                )
            )

        wall_time = time.perf_counter() - start_time
        accuracy = (correct / len(samples_to_eval)) if samples_to_eval else 0.0

        codec_name = runner.config.codec
        rank = runner.config.codec_args.get("rank") if runner.config.codec == "lowrank" else None

        return DownstreamResult(
            benchmark_name="gsm8k-sample",
            num_samples=len(samples_to_eval),
            correct_samples=correct,
            accuracy=accuracy,
            eval_mode="greedy_generation",
            codec=codec_name,
            rank=rank,
            items=items,
            wall_time_s=wall_time,
        )


class MMLUEvaluator:
    """Evaluates multiple-choice reasoning via log-likelihood scoring or greedy generation."""

    def __init__(
        self,
        samples: list[dict[str, Any]] | None = None,
        eval_mode: str = "log_likelihood",
    ) -> None:
        self.samples = samples or MMLU_SAMPLE_DATASET
        self.eval_mode = eval_mode

    def format_prompt(self, sample: dict[str, Any]) -> str:
        choices_str = "\n".join(sample["choices"])
        return f"Question: {sample['question']}\n{choices_str}\nAnswer:"

    def evaluate(
        self,
        runner: ChainRunner,
        tokenizer: Any = None,
        max_samples: int | None = None,
    ) -> DownstreamResult:
        tokenizer = tokenizer or SimpleCharTokenizer()
        samples_to_eval = self.samples[:max_samples] if max_samples is not None else self.samples
        items: list[BenchmarkItemResult] = []
        correct = 0
        start_time = time.perf_counter()

        for sample in samples_to_eval:
            prompt_str = self.format_prompt(sample)
            encoded = tokenizer(prompt_str, return_tensors="pt")
            input_ids = encoded.input_ids if hasattr(encoded, "input_ids") else encoded
            gold = sample["gold"].strip().upper()

            if self.eval_mode == "log_likelihood":
                result = runner.forward(input_ids, logits_keep_last=1)
                last_logits = result.logits[0, -1]  # (vocab_size,)

                choice_letters = ["A", "B", "C", "D"]
                scores = {}
                for letter in choice_letters:
                    # Score both " X" and "X" to be robust to leading whitespace tokenization
                    cand_tokens: list[int] = []
                    if hasattr(tokenizer, "encode"):
                        t1 = tokenizer.encode(f" {letter}")
                        if t1:
                            t1_id = t1[-1] if isinstance(t1, list) else t1.view(-1)[-1].item()
                            cand_tokens.append(t1_id)
                        t2 = tokenizer.encode(letter)
                        if t2:
                            t2_id = t2[-1] if isinstance(t2, list) else t2.view(-1)[-1].item()
                            cand_tokens.append(t2_id)
                    else:
                        cand_tokens.append(min(ord(letter), last_logits.shape[-1] - 1))

                    cand_logits = [
                        last_logits[t].item()
                        for t in cand_tokens
                        if 0 <= t < last_logits.shape[-1]
                    ]
                    scores[letter] = max(cand_logits) if cand_logits else float("-inf")

                best_choice = max(scores, key=lambda k: scores[k])
                is_correct = (best_choice == gold)
                if is_correct:
                    correct += 1

                items.append(
                    BenchmarkItemResult(
                        item_id=sample["id"],
                        prompt=prompt_str,
                        target=gold,
                        prediction=best_choice,
                        is_correct=is_correct,
                        score=1.0 if is_correct else 0.0,
                        metadata={"choice_scores": scores, "subject": sample.get("subject")},
                    )
                )
            else:
                generated_ids, _ = runner.generate(input_ids, max_new_tokens=4, use_cache=True)
                new_tokens = generated_ids[0, input_ids.shape[1] :]
                pred_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
                extracted = extract_mmlu_choice(pred_text)
                is_correct = (extracted == gold)
                if is_correct:
                    correct += 1

                items.append(
                    BenchmarkItemResult(
                        item_id=sample["id"],
                        prompt=prompt_str,
                        target=gold,
                        prediction=extracted or pred_text.strip(),
                        is_correct=is_correct,
                        score=1.0 if is_correct else 0.0,
                        metadata={
                            "generated_text": pred_text.strip(),
                            "subject": sample.get("subject"),
                        },
                    )
                )

        wall_time = time.perf_counter() - start_time
        accuracy = (correct / len(samples_to_eval)) if samples_to_eval else 0.0

        codec_name = runner.config.codec
        rank = runner.config.codec_args.get("rank") if runner.config.codec == "lowrank" else None

        return DownstreamResult(
            benchmark_name="mmlu-sample",
            num_samples=len(samples_to_eval),
            correct_samples=correct,
            accuracy=accuracy,
            eval_mode=self.eval_mode,
            codec=codec_name,
            rank=rank,
            items=items,
            wall_time_s=wall_time,
        )


def evaluate_downstream(
    runner: ChainRunner,
    benchmark: str = "all",
    *,
    tokenizer: Any = None,
    eval_mode: str = "log_likelihood",
    max_samples: int | None = None,
    dataset_file: str | Path | None = None,
) -> dict[str, DownstreamResult]:
    """Evaluate one or more downstream benchmarks on a ChainRunner instance."""
    results: dict[str, DownstreamResult] = {}
    custom_samples = None
    if dataset_file is not None:
        p = Path(dataset_file)
        if p.exists():
            if p.suffix == ".jsonl":
                custom_samples = [
                    json.loads(line)
                    for line in p.read_text().splitlines()
                    if line.strip()
                ]
            elif p.suffix == ".json":
                custom_samples = json.loads(p.read_text())

    bench = benchmark.lower()
    if bench in ("gsm8k", "gsm8k-sample", "all"):
        evaluator_gsm = GSM8KEvaluator(samples=custom_samples if bench != "all" else None)
        results["gsm8k-sample"] = evaluator_gsm.evaluate(
            runner, tokenizer=tokenizer, max_samples=max_samples
        )

    if bench in ("mmlu", "mmlu-sample", "all"):
        evaluator_mmlu = MMLUEvaluator(
            samples=custom_samples if bench != "all" else None,
            eval_mode=eval_mode,
        )
        results["mmlu-sample"] = evaluator_mmlu.evaluate(
            runner, tokenizer=tokenizer, max_samples=max_samples
        )

    return results
