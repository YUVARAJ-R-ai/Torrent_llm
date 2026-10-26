#!/usr/bin/env python
"""Train the low-rank codec to preserve what the model says, not just what it carries.

    python scripts/train_codec.py --ranks 128,64

PCA (``scripts/calibrate_codec.py``) keeps the directions in which activations vary
most. That is a proxy: the next layers do not care how much a direction varies, they
care what happens to the *output* if it is removed. This trains the projection
directly on that: the model is frozen, the hand-over at the shard boundary goes through
``(h - mean) @ down @ up + mean``, and ``down`` and ``up`` are adjusted to keep the
final logits close to the uncompressed model's (KL divergence). It starts from the PCA
solution, so it can only be better than where PCA left off if it learns anything.

The wire format is unchanged: the same ``LowRankCodec``, the same weights file layout.

Needs about 5 GB of GPU memory. Refuses to start with less free.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
import calibrate_codec as cc  # noqa: E402

from torrent_llm.codec.calibrate import CovarianceAccumulator  # noqa: E402
from torrent_llm.codec.lowrank import LowRankCodec  # noqa: E402
from torrent_llm.eval.boundary import WireTap, boundary_hook, capture_boundary  # noqa: E402


def make_transform(down, up, mean, fp16: bool):
    """The differentiable stand-in for what the codec does to a hidden state."""

    def transform(hidden: torch.Tensor) -> torch.Tensor:
        latent = (hidden - mean) @ down
        if fp16:
            # Straight-through: the forward pass sees the 16-bit rounding the wire will
            # apply, the backward pass ignores it.
            latent = latent + (latent.half().float() - latent).detach()
        return latent @ up + mean

    return transform


def kl_to_teacher(student_logits: torch.Tensor, teacher_logits: torch.Tensor) -> torch.Tensor:
    teacher = torch.log_softmax(teacher_logits.float(), dim=-1)
    student = torch.log_softmax(student_logits.float(), dim=-1)
    return (teacher.exp() * (teacher - student)).sum(-1).mean()


def evaluate(model, tokenizer, device, held_out, boundary, codec) -> dict:
    tap = WireTap(codec)
    with boundary_hook(model, boundary, tap):
        ppl = cc.perplexity(model, held_out, device)
    qa_tap = WireTap(codec)
    with boundary_hook(model, boundary, qa_tap):
        qa = cc.qa_accuracy(model, tokenizer, device)
    return {
        "perplexity": round(ppl, 3),
        "answers_right": round(qa, 3),
        "relative_error": round(tap.relative_error, 4),
        "compression": round(tap.ratio, 2),
    }


def train_rank(model, tokenizer, device, rank, args, fit_chunks, chat_ids, held_out, accumulator):
    pca = accumulator.fit(rank)
    mean = pca.mean.to(device)
    down = torch.nn.Parameter(pca.down_proj.clone().to(device))
    up = torch.nn.Parameter(pca.up_proj.clone().to(device))
    transform = make_transform(down, up, mean, args.fp16)
    optimiser = torch.optim.AdamW([down, up], lr=args.lr, weight_decay=0.0)
    rng = random.Random(0)

    def codec_for(d, u):
        return LowRankCodec(
            rank=rank,
            down_proj=d.detach().cpu(),
            up_proj=u.detach().cpu(),
            mean=mean.cpu(),
            wire_dtype="float16" if args.fp16 else None,
        )

    print(f"\nrank {rank}: starting from PCA", flush=True)
    before = evaluate(model, tokenizer, device, held_out, args.boundary, codec_for(down, up))
    print(
        f"  PCA      perplexity {before['perplexity']:8.2f}  answers {before['answers_right']:.0%}",
        flush=True,
    )

    started = time.time()
    best = (math.inf, down.detach().clone(), up.detach().clone())
    for step in range(1, args.steps + 1):
        for group in optimiser.param_groups:  # cosine decay to a tenth
            group["lr"] = args.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * step / args.steps)))
        optimiser.zero_grad()
        total = 0.0
        batch = rng.sample(fit_chunks, args.batch)
        ids = torch.cat(batch).to(device)
        keep = min(64, ids.shape[1])
        with torch.no_grad():
            teacher = model(input_ids=ids, logits_to_keep=keep).logits
        with boundary_hook(model, args.boundary, transform):
            student = model(input_ids=ids, logits_to_keep=keep).logits
        loss = kl_to_teacher(student, teacher)
        loss.backward()
        total += loss.item()
        # Chat sequences are short and are what is served, so they count for more.
        for chat in rng.sample(chat_ids, args.chats):
            c = chat.to(device)
            with torch.no_grad():
                teacher = model(input_ids=c).logits
            with boundary_hook(model, args.boundary, transform):
                student = model(input_ids=c).logits
            chat_loss = args.chat_weight * kl_to_teacher(student, teacher) / args.chats
            chat_loss.backward()
            total += chat_loss.item()
        torch.nn.utils.clip_grad_norm_([down, up], 1.0)
        optimiser.step()
        if total < best[0] * 1.0:
            best = (total, down.detach().clone(), up.detach().clone())
        if step % 50 == 0 or step == 1:
            print(f"  step {step:4d}  kl {total:.4f}  ({time.time() - started:.0f}s)", flush=True)

    down.data, up.data = best[1], best[2]
    after = evaluate(model, tokenizer, device, held_out, args.boundary, codec_for(down, up))
    print(
        f"  learned  perplexity {after['perplexity']:8.2f}  answers {after['answers_right']:.0%}",
        flush=True,
    )
    return codec_for(down, up), {"rank": rank, "pca": before, "learned": after, "steps": args.steps}


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--boundary", type=int, default=14)
    parser.add_argument("--ranks", default="128,64")
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--batch", type=int, default=4, help="prose sequences per step")
    parser.add_argument("--chats", type=int, default=6, help="chat sequences per step")
    parser.add_argument("--chat-weight", type=float, default=2.0)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seq-tokens", type=int, default=192)
    parser.add_argument(
        "--no-fp16", dest="fp16", action="store_false", help="train for a 32-bit wire"
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out-dir", default="weights")
    parser.add_argument("--results", default="docs/results/learned-qwen3-0.6b-l14.json")
    args = parser.parse_args()

    if cc.free_ram_gb() < 8:
        print("under 8 GB of RAM free; close something and retry.")
        return 2
    if args.device == "cuda":
        free = torch.cuda.mem_get_info()[0] / 1024**3
        if free < 5:
            print(f"only {free:.1f} GB of GPU memory free; need about 5. Stop other GPU programs.")
            return 2
    torch.set_num_threads(4)
    torch.manual_seed(0)

    print(f"loading {args.model}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = (
        AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32).to(args.device).eval()
    )
    model.requires_grad_(False)

    print("preparing text", flush=True)
    fit_chunks = [
        c
        for book in cc.FIT_BOOKS
        for c in cc.token_chunks(tokenizer, cc.fetch_book(book), 150, args.seq_tokens, offset=2000)
    ]
    held_out = [
        c
        for book in cc.HELD_OUT_BOOKS
        for c in cc.token_chunks(tokenizer, cc.fetch_book(book), 40, 256, offset=2000)
    ]
    chat_ids = [cc.chat_ids(tokenizer, q, a) for q, a in cc.FIT_QUESTIONS]

    hidden = model.config.hidden_size
    accumulator = CovarianceAccumulator(hidden, args.device)
    for ids in fit_chunks + chat_ids * 6:
        accumulator.add(capture_boundary(model, args.boundary, ids.to(args.device)))

    base_ppl = cc.perplexity(model, held_out, args.device)
    base_qa = cc.qa_accuracy(model, tokenizer, args.device)
    print(f"uncompressed: perplexity {base_ppl:.2f}, answers right {base_qa:.0%}", flush=True)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(exist_ok=True)
    stem = args.model.split("/")[-1].lower()
    results = {
        "model": args.model,
        "boundary": args.boundary,
        "fp16_wire": args.fp16,
        "baseline": {"perplexity": round(base_ppl, 3), "answers_right": round(base_qa, 3)},
        "settings": {k: getattr(args, k) for k in ("steps", "batch", "chats", "chat_weight", "lr")},
        "ranks": [],
    }
    for rank in [int(r) for r in args.ranks.split(",")]:
        codec, row = train_rank(
            model, tokenizer, args.device, rank, args, fit_chunks, chat_ids, held_out, accumulator
        )
        path = out_dir / f"{stem}-l{args.boundary}-r{rank}-learned.safetensors"
        codec.save_weights(path, hidden)
        row["weights"] = str(path)
        results["ranks"].append(row)
        print(f"  saved {path} ({path.stat().st_size / 1024:.0f} KiB)", flush=True)
        Path(args.results).parent.mkdir(parents=True, exist_ok=True)
        Path(args.results).write_text(json.dumps(results, indent=2))
    print("\ndone")
    return 0


if __name__ == "__main__":
    sys.exit(main())
