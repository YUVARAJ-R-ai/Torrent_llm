# Compression: what it costs and what it buys

The question this project exists to answer: can the hidden states that cross a shard
boundary be made much smaller without hurting the answers?

**Short answer, measured on one model:** a little, at a real cost. 2x is nearly free.
4x to 8x loses about 12 points of answers right. 16x and beyond breaks the model. The
*uncalibrated* projection the project started with never worked at any size. On a slow
link, 8x fewer bytes made an answer about 10% faster, not 8x faster.

An earlier version of this page reported 88% right at 4x. That was measured on 30
questions and was too optimistic. The numbers below use 95 questions.

## Result

Qwen3-0.6B, cut in two at layer 14 (the boundary of a two-stage pool). Compression is
applied to the hidden states as they cross it, in the model, and the real payload bytes
are counted. Perplexity is on a book the fit never saw. "Answers right" is on 95
questions (capitals, authors, arithmetic) that are not the fit questions.

| Setting | Smaller by | Error | Perplexity | Answers right |
|---|---|---|---|---|
| **Uncompressed** | 1x | 0 | **40.4** | **88%** |
| Random projection, rank 512 | 2x | 0.70 | 22,327 | 0% |
| Random projection, rank 128 | 8x | 0.94 | 1,001,247 | 0% |
| Calibrated (PCA), rank 512 | 2x | 0.021 | 40.7 | 90% |
| Calibrated, rank 256 | 4x | 0.036 | 42.2 | 77% |
| **Calibrated, rank 256, fp16 on the wire** | **8x** | 0.036 | 42.2 | 77% |
| Calibrated, rank 128 | 8x | 0.049 | 50.1 | 50% |
| Calibrated, rank 128, fp16 on the wire | 16x | 0.049 | 50.1 | 50% |
| Calibrated, rank 64 | 16x | 0.060 | 84.7 | 6% |
| Calibrated, rank 32 | 32x | 0.068 | 195 | 0% |

"Error" is the relative error of the hidden states themselves. Full data:
[results/calibration-qwen3-0.6b-l14.json](results/calibration-qwen3-0.6b-l14.json).
Reproduce with `python scripts/calibrate_codec.py` (a few minutes, 3 GB of GPU).

## Learned projection

PCA keeps the directions that vary most. `scripts/train_codec.py` instead trains the
projection to keep the model's final output close to the uncompressed one (KL
divergence), starting from PCA. 1000 steps per rank, fp16 wire.

| Rank | Smaller by | PCA: perplexity / right | Learned: perplexity / right |
|---|---|---|---|
| 256 | 8x | 42.3 / 78% | 42.1 / 78% |
| 128 | 16x | 50.4 / 52% | 46.9 / 58% |
| 64 | 32x | 87.5 / 12% | 62.8 / 34% |

Training helps most where PCA is worst, and does nothing at rank 256. It does not move
the cliff by much: 16x at 58% and 32x at 34% are still well short of 88%. The 30-question
run looked better than this. Full data:
[results/learned-qwen3-0.6b-l14.json](results/learned-qwen3-0.6b-l14.json).

## Does it make answers faster?

`scripts/bench_compression.py`: two real devices on this machine, each holding half the
model, the link between them throttled to 15 Mbps and 100 ms (a phone hotspot). Same
prompt, 8 new tokens, median of 4, CPU on both.

| Pool | Time | Network | Compute | Bytes sent | Smaller |
|---|---|---|---|---|---|
| Raw | 2158 ms | 1155 ms | 993 ms | 549 KiB | 1.0x |
| Compressed (rank 256, fp16) | 1953 ms | 921 ms | 981 ms | 70 KiB | 7.9x |

The traffic is 7.9x smaller and the answer is **10% faster** (205 ms). Most of the
network time is the 100 ms latency of each hop, which compression cannot touch. Encoding
cost is small (compute is unchanged). On a fast LAN there is nothing to win. Data:
[results/bench-15mbps-100ms.json](results/bench-15mbps-100ms.json).

## What it shows

- **A random projection destroys the model.** At every size, including 2x. It keeps the
  energy of a vector but not the directions the next layers read.
- **Choosing the directions from data fixes it, partly.** 2x is nearly free. 4x loses
  11 points.
- **Half-precision on the wire is free here.** It halves the bytes again with no
  measurable change.
- **There is a cliff.** Below rank 128 quality collapses. The activations at this layer
  are not low-dimensional: 128 directions explain only 72% of the variation.
- **Leaving the attention-sink position out of the fit hurt.** Those huge values still
  have to be represented.

## What it does not show

- **One model, one boundary.** Another layer or model needs its own calibration. A
  weights file is tied to the boundary it was fitted on.
- **Small tests.** 95 questions is about 1 point each. Perplexity is on one book. The
  learned runs use one seed and one setting; the gains at rank 128 and 64 are a few
  questions' worth of evidence, not a proof.
- **Latency dominates a slow link here.** Fewer bytes only help when bytes are the cost.
  See [bandwidth-regimes.md](bandwidth-regimes.md).
- **Quality is not free.** 8x costs about 11 points of answers right to buy a 10%
  speed-up on a hotspot. On a LAN it is a loss. That is why the default pool is raw.

## Using it

| File | Smaller by | Use |
|---|---|---|
| `qwen3-0.6b-l14-r256.safetensors` | 4x (8x with fp16) | the compressed pool file |
| `qwen3-0.6b-l14-r128.safetensors` | 8x (16x with fp16) | slow links, accepting large loss |
| `qwen3-0.6b-l14-r128-learned.safetensors` | 16x with fp16 | learned version of the above |

See the `codec` block of `configs/pool-compressed.yaml`.
