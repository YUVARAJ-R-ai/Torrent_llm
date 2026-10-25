#!/usr/bin/env python
"""Fit a low-rank activation codec to a real model and measure what it costs.

    python scripts/calibrate_codec.py                       # Qwen3-0.6B, boundary at layer 14

This is the experiment behind the project's central claim: can the activations that
cross a shard boundary be made much smaller without hurting the answers?

It loads the model once, in this one process, and never starts a server. The hand-over
between two shards is exactly "hidden states leave layer k-1, enter layer k", so the
codec is spliced in at that point (see ``torrent_llm.eval.boundary``) and the quality
and the wire bytes are measured there.

What it does:

1. Collects the hidden states crossing the boundary on a mix of prose (public-domain
   books) and chat-formatted questions, the kind of text a pool actually serves.
2. Fits a projection at several ranks, with and without leaving the first position
   (the attention sink) out of the fit. Leaving it out turned out to hurt: the sink's
   huge values need capturing, so the saved weights are fitted on every position.
3. On text it did **not** fit on, measures perplexity, and on questions it did not fit
   on, the share of answers that are still right. Compares each setting to sending the
   hidden states uncompressed, and to the random projection the project began with.
4. Saves the weights for the settings worth keeping, and every number to a JSON file.

It needs about 3 GB of GPU memory and a few minutes. It refuses to start if the machine
has under 8 GB of RAM free.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
import urllib.request
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from torrent_llm.chat import DEFAULT_SYSTEM_PROMPT
from torrent_llm.codec.calibrate import CovarianceAccumulator, Projection
from torrent_llm.codec.lowrank import LowRankCodec
from torrent_llm.eval.boundary import WireTap, boundary_hook, capture_boundary

CACHE = Path.home() / ".cache" / "torrent-llm" / "corpus"
FIT_BOOKS = {11: "Alice in Wonderland", 1342: "Pride and Prejudice", 1661: "Sherlock Holmes"}
HELD_OUT_BOOKS = {84: "Frankenstein"}

# Questions the fit sees, and different ones it does not. Each is (question, answer).
FIT_QUESTIONS = (
    [
        (f"What is the capital of {c}?", f"The capital of {c} is {k}.")
        for c, k in [
            ("Spain", "Madrid"),
            ("Germany", "Berlin"),
            ("Canada", "Ottawa"),
            ("Egypt", "Cairo"),
            ("Australia", "Canberra"),
            ("Brazil", "Brasilia"),
            ("Russia", "Moscow"),
            ("Argentina", "Buenos Aires"),
            ("Norway", "Oslo"),
            ("Greece", "Athens"),
            ("Portugal", "Lisbon"),
            ("Poland", "Warsaw"),
            ("Kenya", "Nairobi"),
            ("Peru", "Lima"),
            ("Sweden", "Stockholm"),
            ("Thailand", "Bangkok"),
            ("Turkey", "Ankara"),
            ("Ireland", "Dublin"),
            ("Cuba", "Havana"),
            ("Chile", "Santiago"),
        ]
    ]
    + [
        (f"Who wrote {b}?", f"{b} was written by {a}.")
        for b, a in [
            ("Hamlet", "William Shakespeare"),
            ("Pride and Prejudice", "Jane Austen"),
            ("Moby Dick", "Herman Melville"),
            ("Dracula", "Bram Stoker"),
            ("Don Quixote", "Miguel de Cervantes"),
            ("The Odyssey", "Homer"),
            ("Great Expectations", "Charles Dickens"),
            ("The Trial", "Franz Kafka"),
        ]
    ]
    + [
        (f"What is {a} times {b}?", f"{a} times {b} is {a * b}.")
        for a, b in [
            (3, 4),
            (6, 7),
            (8, 9),
            (12, 5),
            (7, 7),
            (9, 6),
            (11, 4),
            (13, 3),
            (15, 4),
            (6, 8),
        ]
    ]
)
# The expected substring is what must appear in the answer for it to count as right.
EVAL_QUESTIONS = (
    [
        (f"What is the capital of {c}?", k)
        for c, k in [
            ("France", "Paris"),
            ("Italy", "Rome"),
            ("Japan", "Tokyo"),
            ("India", "New Delhi"),
            ("China", "Beijing"),
            ("Mexico", "Mexico City"),
            ("Austria", "Vienna"),
            ("Finland", "Helsinki"),
            ("Hungary", "Budapest"),
            ("Netherlands", "Amsterdam"),
            ("Belgium", "Brussels"),
            ("Denmark", "Copenhagen"),
        ]
    ]
    + [
        (f"Who wrote {b}?", a)
        for b, a in [
            ("Frankenstein", "Shelley"),
            ("Macbeth", "Shakespeare"),
            ("Emma", "Austen"),
            ("Ulysses", "Joyce"),
            ("War and Peace", "Tolstoy"),
            ("The Raven", "Poe"),
        ]
    ]
    + [
        (f"What is {a} times {b}?", str(a * b))
        for a, b in [(4, 5), (6, 6), (7, 8), (9, 9), (12, 3), (5, 11)]
    ]
)


def free_ram_gb() -> float:
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable"):
            return int(line.split()[1]) / 1024**2
    return 0.0


def fetch_book(book_id: int) -> str:
    """A public-domain book from Project Gutenberg, cached on disk, without its boilerplate."""
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"pg{book_id}.txt"
    if not path.exists():
        url = f"https://www.gutenberg.org/cache/epub/{book_id}/pg{book_id}.txt"
        path.write_bytes(urllib.request.urlopen(url, timeout=60).read())  # noqa: S310
    text = path.read_text(encoding="utf-8", errors="ignore")
    start = re.search(r"\*\*\* START OF .*?\*\*\*", text)
    end = re.search(r"\*\*\* END OF .*?\*\*\*", text)
    return text[start.end() if start else 0 : end.start() if end else len(text)]


def token_chunks(
    tokenizer, text: str, count: int, length: int, offset: int = 0
) -> list[torch.Tensor]:
    """``count`` spans of ``length`` tokens, spread evenly through ``text``."""
    ids = tokenizer(text, return_tensors="pt").input_ids[0]
    usable = ids.numel() - length - offset
    # A short text cannot supply as many spans as asked for; take what it has
    # rather than slicing past the end, which gives empty spans.
    count = max(1, min(count, usable // length))
    step = max(length, usable // count)
    return [ids[offset + i * step : offset + i * step + length].unsqueeze(0) for i in range(count)]


def chat_ids(tokenizer, question: str, answer: str | None = None) -> torch.Tensor:
    messages = [
        {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    if answer is not None:
        messages.append({"role": "assistant", "content": answer})
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=answer is None,
        enable_thinking=False,
    )
    return tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids


@torch.inference_mode()
def perplexity(model, chunks: list[torch.Tensor], device: str) -> float:
    total, count = 0.0, 0
    for chunk in chunks:
        ids = chunk.to(device)
        logits = model(input_ids=ids).logits
        loss = torch.nn.functional.cross_entropy(
            logits[0, :-1].float(), ids[0, 1:], reduction="sum"
        )
        total += loss.item()
        count += ids.shape[1] - 1
    return math.exp(total / count)


@torch.inference_mode()
def qa_accuracy(model, tokenizer, device: str, max_new: int = 14) -> float:
    right = 0
    for question, expected in EVAL_QUESTIONS:
        ids = chat_ids(tokenizer, question).to(device)
        out = model.generate(ids, max_new_tokens=max_new, do_sample=False)
        answer = tokenizer.decode(out[0, ids.shape[1] :], skip_special_tokens=True)
        right += expected.lower() in answer.lower()
    return right / len(EVAL_QUESTIONS)


def make_codec(
    kind: str, rank: int, projections: dict[str, Projection], fp16: bool
) -> LowRankCodec:
    wire = "float16" if fp16 else None
    if kind == "random":
        return LowRankCodec(rank=rank, seed=0, wire_dtype=wire)
    p = projections[kind]
    return LowRankCodec(
        rank=rank,
        down_proj=p.down_proj,
        up_proj=p.up_proj,
        mean=p.mean,
        wire_dtype=wire,
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--boundary", type=int, default=14, help="first layer of the second shard")
    parser.add_argument("--ranks", default="512,256,128,64,32")
    parser.add_argument("--keep", default="256,128", help="ranks whose weights are saved")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--chunk-tokens", type=int, default=256)
    parser.add_argument(
        "--fit-chunks", type=int, default=150, help="prose chunks per book to fit on"
    )
    parser.add_argument("--out-dir", default="weights")
    parser.add_argument("--runs-dir", default="runs")
    args = parser.parse_args()

    if free_ram_gb() < 8:
        print(f"only {free_ram_gb():.1f} GB of RAM free; need 8. Close something and retry.")
        return 2
    torch.set_num_threads(4)
    ranks = [int(r) for r in args.ranks.split(",")]
    keep = {int(r) for r in args.keep.split(",")}
    started = time.time()

    print(f"loading {args.model} on {args.device}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = (
        AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32).to(args.device).eval()
    )
    hidden = model.config.hidden_size

    print("fetching text", flush=True)
    fit_prose = [
        c
        for book in FIT_BOOKS
        for c in token_chunks(
            tokenizer, fetch_book(book), args.fit_chunks, args.chunk_tokens, offset=2000
        )
    ]
    held_out = [
        c
        for book in HELD_OUT_BOOKS
        for c in token_chunks(tokenizer, fetch_book(book), 40, args.chunk_tokens, offset=2000)
    ]
    fit_chat = [chat_ids(tokenizer, q, a) for q, a in FIT_QUESTIONS]

    print(
        f"collecting activations at layer {args.boundary}: "
        f"{len(fit_prose)} prose chunks, {len(fit_chat)} chat sequences",
        flush=True,
    )
    everything = CovarianceAccumulator(hidden, args.device)
    without_sink = CovarianceAccumulator(hidden, args.device)
    for ids in fit_prose + fit_chat * 6:  # chat is short and is what is served: count it more
        states = capture_boundary(model, args.boundary, ids.to(args.device))
        everything.add(states)
        without_sink.add(states, skip_first_position=True)
    print(f"  {everything.count:,} positions", flush=True)

    curve = without_sink.variance_curve()
    print(
        "variance explained by the first k directions (sink left out): "
        + ", ".join(f"k={k}: {curve[k - 1]:.3f}" for k in (16, 32, 64, 128, 256, 512)),
        flush=True,
    )

    print("measuring the uncompressed baseline", flush=True)
    base_ppl = perplexity(model, held_out, args.device)
    base_qa = qa_accuracy(model, tokenizer, args.device)
    print(f"  perplexity {base_ppl:.2f}, answers right {base_qa:.0%}", flush=True)

    rows: list[dict] = []
    header = (
        f"{'setting':22} {'rank':>5} {'smaller':>8} {'error':>7} {'perplexity':>11} {'answers':>8}"
    )
    print("\n" + header + "\n" + "-" * len(header), flush=True)
    settings = []
    for rank in ranks:
        settings += [("random", rank, False), ("pca", rank, False), ("pca-nosink", rank, False)]
        if rank in keep:
            settings.append(("pca", rank, True))
    projections_by_rank = {
        rank: {"pca": everything.fit(rank), "pca-nosink": without_sink.fit(rank)} for rank in ranks
    }
    for kind, rank, fp16 in settings:
        codec = make_codec(kind, rank, projections_by_rank[rank], fp16)
        tap = WireTap(codec)
        with boundary_hook(model, args.boundary, tap):
            ppl = perplexity(model, held_out, args.device)
        error, ratio = tap.relative_error, tap.ratio
        tap_qa = WireTap(make_codec(kind, rank, projections_by_rank[rank], fp16))
        with boundary_hook(model, args.boundary, tap_qa):
            qa = qa_accuracy(model, tokenizer, args.device)
        label = kind + ("+fp16" if fp16 else "")
        print(f"{label:22} {rank:5d} {ratio:7.1f}x {error:7.3f} {ppl:11.2f} {qa:8.0%}", flush=True)
        rows.append(
            {
                "setting": label,
                "rank": rank,
                "fp16": fp16,
                "compression": round(ratio, 2),
                "relative_error": round(error, 4),
                "perplexity": round(ppl, 3),
                "answers_right": round(qa, 3),
                "explained_variance": (
                    round(projections_by_rank[rank][kind].explained_variance, 4)
                    if kind != "random"
                    else None
                ),
            }
        )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(exist_ok=True)
    stem = args.model.split("/")[-1].lower()
    for rank in sorted(keep & set(ranks)):
        p = projections_by_rank[rank]["pca"]
        codec = make_codec("pca", rank, projections_by_rank[rank], False)
        path = out_dir / f"{stem}-l{args.boundary}-r{rank}.safetensors"
        codec.save_weights(path, hidden)
        print(
            f"saved {path} ({path.stat().st_size / 1024:.0f} KiB, "
            f"explains {p.explained_variance:.1%})"
        )

    runs = Path(args.runs_dir)
    runs.mkdir(exist_ok=True)
    result = {
        "model": args.model,
        "boundary": args.boundary,
        "hidden_size": hidden,
        "positions_fitted": everything.count,
        "fit_text": list(FIT_BOOKS.values()) + ["chat questions"],
        "held_out_text": list(HELD_OUT_BOOKS.values()),
        "eval_questions": len(EVAL_QUESTIONS),
        "baseline": {"perplexity": round(base_ppl, 3), "answers_right": round(base_qa, 3)},
        "rows": rows,
        "seconds": round(time.time() - started),
    }
    (runs / f"calibration-{stem}-l{args.boundary}.json").write_text(json.dumps(result, indent=2))
    print(f"\ndone in {time.time() - started:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
