# Compression: what it costs and what it buys

The question this project exists to answer: can the hidden states that cross a shard
boundary be made much smaller without hurting the answers?

**Short answer, measured on one model:** yes at **4x to 8x**, no at 16x and beyond, and
the *uncalibrated* projection the project started with never worked at any size.

## Result

Qwen3-0.6B, cut in two at layer 14 (the boundary of a two-stage pool). Compression is
applied to the hidden states as they cross it, in the model, and the real payload
bytes are counted. Perplexity is on a book the fit never saw; "answers right" is on
30 questions the fit never saw (capitals, authors, arithmetic).

| Setting | Smaller by | Error | Perplexity | Answers right |
|---|---|---|---|---|
| **Uncompressed** | 1x | 0 | **40.4** | **96%** |
| Random projection, rank 512 | 2x | 0.70 | 22,327 | 0% |
| Random projection, rank 128 | 8x | 0.94 | 1,001,247 | 0% |
| Calibrated, rank 512 | 2x | 0.021 | 40.7 | 92% |
| Calibrated, rank 256 | 4x | 0.036 | 42.2 | 88% |
| **Calibrated, rank 256, fp16 on the wire** | **8x** | 0.036 | 42.2 | 88% |
| Calibrated, rank 128 | 8x | 0.049 | 50.1 | 79% |
| Calibrated, rank 128, fp16 on the wire | 16x | 0.049 | 50.1 | 79% |
| Calibrated, rank 64 | 16x | 0.060 | 84.7 | 17% |
| Calibrated, rank 32 | 32x | 0.068 | 195 | 0% |

"Error" is the relative error of the hidden states themselves. Full data:
[results/calibration-qwen3-0.6b-l14.json](results/calibration-qwen3-0.6b-l14.json).
Reproduce with `python scripts/calibrate_codec.py` (about three minutes, 3 GB of GPU).

## What it shows

- **A random projection destroys the model.** At every size, including a 2x one. It
  keeps the *energy* of a vector but not the directions the next layers read. This is
  why the compressor first merged into the project gave wrong answers.
- **Choosing the directions from data fixes it.** Principal components of the real
  hidden states, with the average removed first, keep 88% of the answers at 4x and
  79% at 8x.
- **Half-precision on the wire is free here.** Sending the latent as 16-bit floats
  halves the bytes again with *no* measurable change. So 8x is available at the
  rank-256 quality.
- **There is a cliff.** Below rank 128 quality collapses. The activations at this
  layer are not low-dimensional: the first 128 directions explain only 72% of the
  variation (85% at 256, 95% at 512), so there is not a great deal of slack to remove.
- **Leaving the attention-sink position out of the fit hurt.** I expected it to help,
  since that position has enormous values. It does the opposite (rank 128: 67% right
  without it, 79% with it), because those values have to be represented too.

## What it does not show

- **One model, one boundary, one precision.** Another layer or model needs its own
  calibration (the script does it). The weights file is tied to the boundary it was
  fitted on.
- **Small tests.** 30 questions is 3.3 points each, so "88% vs 96%" is two or three
  questions. Perplexity is on one book. Treat the ranking as solid and the exact
  percentages as rough.
- **Compression helps most on prefill and agent hand-offs.** For a cached decode step
  one position is already only a few KiB, and there was never much wire time to save
  there. See [bandwidth-regimes.md](bandwidth-regimes.md).
- **Quality is not free.** 8x costs about 8 points of answer accuracy here. Whether that
  is worth it depends on the link: on a fast LAN, probably not; on a phone hotspot,
  probably yes.

## Using it

Weights for the two useful operating points are in `weights/`:

| File | Smaller by | Use |
|---|---|---|
| `qwen3-0.6b-l14-r256.safetensors` | 4x (8x with fp16) | the default compromise |
| `qwen3-0.6b-l14-r128.safetensors` | 8x (16x with fp16) | slow links, accepting more loss |

See the `codec` block of `configs/pool.yaml`.
