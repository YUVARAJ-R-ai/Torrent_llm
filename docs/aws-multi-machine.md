# The pool on real machines: AWS, 7 November 2026

Until now every pool measurement came from one computer (the lab) or one GPU laptop plus
one CPU laptop. This run put the pool on separate cloud machines, with real network cards
between them and a real link between two countries, and answered four questions:

1. Does a pool of four separate machines work, share load and survive losing one?
2. When does compression make answers faster, on a link slowed down by the kernel rather
   than by the application?
3. What happens on a real long-distance link?
4. Does the pool, with compression switched on, give the same answer quality that
   `docs/compression.md` measured offline?

**Short answers.** (1) Yes. After a crash it recovers in under a second with exactly the same
text. A device that goes silent instead of crashing stalls the answer for **5 minutes**.
(2) Only on slow links: 8x compression makes answers 23% faster at 2 Mbit/s and
makes no measurable difference at 100 Mbit/s or on a LAN. (3) Mumbai to Singapore (62 ms):
4 to 6% faster. (4) Yes, exactly: 88% / 77% / 58% right, the same as the offline numbers.

## Setup

| | |
|---|---|
| Machines | 5 x `m7i-flex.large` (2 vCPU, 8 GB, CPU only), Ubuntu 24.04 |
| Where | 4 in Mumbai (`ap-south-1a`, same subnet), 1 in Singapore (`ap-southeast-1a`) |
| Code | `main` at `9499ab0`, installed with `scripts/setup.sh` from a fresh clone |
| Model | Qwen3-0.6B, float32, 2 stages cut at layer 14 |
| Pool files | `configs/pool.yaml` (raw), `configs/pool-compressed.yaml` (PCA rank 256, fp16: 7.9x), and the same file with `weights/qwen3-0.6b-l14-r128-learned.safetensors`, rank 128 (15.5x) |
| Devices | `torrent-peer --device cpu --memory-mb 1600 --no-discover --join <first> --token <secret>`. 1600 MB allows one stage per device, so every answer crosses the network. AWS has no broadcast, so devices joined by address. |
| Slow links | Linux `tc netem` (`delay` + `rate`) on both machines, applied only to traffic to the other device. Each direction carries half the round-trip time. |
| Speed test | Same long prompt as `scripts/bench_compression.py` (about 140 tokens), 16 new tokens, raw text, median of 5 after a warm-up |

Raw data and the scripts that produced it: [results/aws-2026-11-07/](results/aws-2026-11-07/).

## 1. Four machines in one pool

The pool reached **Healthy** (two ready copies of each stage) **7 s** after the last device
started. `scripts/pool_check.py` passed: all four devices gave the same answer ("The capital
of France is Paris."), each through a different route, and five simultaneous questions finished
in 3.7 s.

**Load** (`scripts/pool_load.py`, 24 questions, each sent to a different device in turn):

| At once | Questions/s | Tokens/s | Median answer | Slowest | Forward passes per device (A, C on stage 1; B, D on stage 2) |
|---|---|---|---|---|---|
| 1 | 0.52 | 3.4 | 1.93 s | 2.91 s | 70 / 110 / 110 / 70 |
| 4 | 1.33 | 8.6 | 2.72 s | 4.87 s | 90 / 90 / 101 / 79 |
| 8 | 1.65 | 10.7 | 4.05 s | 7.49 s | 92 / 88 / 91 / 89 |

Running eight at once gives 3.1x the throughput of one at a time, and the work spreads evenly
across the four machines. Each answer gets slower, which is what sharing should do.
The absolute numbers are low because these are 2-vCPU CPU machines: one machine holding
the whole model takes 4.6 s for 16 tokens.

**Losing a device mid-answer** (60-token answer through device A, device killed 4 s in):

| What happened to the device | Switched route after | Answer total (normal: 12.4 s) | Same 60 tokens as an undisturbed run? |
|---|---|---|---|
| Process killed (`kill -9`), like a crash | **0.86 s** (UNAVAILABLE at 4.05 s, recovered at 4.91 s) | 13.5 s | **Yes** |
| Network silently dropped (firewall DROP), like a lid closing or Wi-Fi going away | **300 s** (DEADLINE_EXCEEDED at 304 s) | 313 s | Yes |

A crash is handled well. A silent device is not. The in-flight call waits out the gRPC
deadline, which is 300 s (`transport/client.py:102`, used unchanged by the pool), so the
person asking waits five minutes. Real laptops go silent far more often than they crash:
lids close, Wi-Fi drops, people walk away.

**Coming back:** the crashed device restarted and the pool was Healthy again 6 s later. When
the silenced device's firewall rule was removed, it was alive again in about 1 s.

## 2. Compression vs link speed (two machines, shaped link)

Time for one answer: about 140-token prompt plus 16 new tokens, median of 5, seconds.
Each run of 5 varied by 3% or less, and every repeat produced the same text.

| Link | Raw | 8x (PCA r256) | 16x (learned r128) | 8x saves | 16x saves |
|---|---|---|---|---|---|
| Same subnet (~0.1 ms) | 4.71 | 4.68 | 4.67 | 1% | 1% |
| 100 Mbit/s, 10 ms | 4.95 | 4.87 | 4.86 | 2% | 2% |
| 20 Mbit/s, 40 ms | 5.65 | 5.47 | 5.43 | 3% | 4% |
| 10 Mbit/s, 100 ms | 7.08 | 6.64 | 6.55 | 6% | 8% |
| 5 Mbit/s, 100 ms | 7.52 | 6.64 | 6.59 | 12% | 12% |
| 2 Mbit/s, 100 ms | 8.93 | 6.83 | 6.70 | **23%** | **25%** |

Network part of each answer (the rest is compute, 4.6 s in every row):

| Link | Raw | 8x | 16x |
|---|---|---|---|
| Same subnet | 0.06 | 0.07 | 0.06 |
| 10 Mbit/s, 100 ms | 2.45 | 2.00 | 1.90 |
| 2 Mbit/s, 100 ms | 4.30 | 2.21 | 2.06 |

Bytes sent per answer: raw 581 KiB, 8x 74 KiB, 16x 37 KiB. For reference, one device
holding both stages (no network) takes 4.58 s. On the same subnet the pool costs 3% more.

What it shows:

- **Encoding costs nothing measurable.** Compute time is the same with and without
  compression (4.59 to 4.64 s).
- **Latency sets a floor that compression cannot touch.** 16 passes at 100 ms is about 1.6 s.
  Compressed answers sit just above it (1.9 to 2.2 s) at every bandwidth from 10 down to 2
  Mbit/s. Raw answers grow as the bandwidth falls.
- **Almost all the bytes are the prompt.** Each new token sends one hidden-state vector
  (4 KiB raw), so about 90% of the 581 KiB is the prompt's prefill. Compression mostly
  speeds up the first token of a long prompt. Once the prompt is through, generation is
  latency-bound.
- **16x is barely faster than 8x** (at most 2 percentage points) and loses 19 more points of
  answer quality (section 4). On these links 16x is not worth it.

This confirms the earlier single-machine result in `docs/compression.md` (10% at 15 Mbit/s,
100 ms) and extends it: compression pays off below about 10 Mbit/s and costs nothing to
run.

## 3. A real long-distance link: Mumbai to Singapore

Measured link: 62 ms TCP round trip. One connection reaches about 209 Mbit/s once warm, but a
cold 2 MB transfer reached only 30 Mbit/s (TCP slow start).

| Pool | Answer | Network | Bytes | vs raw |
|---|---|---|---|---|
| Raw | 6.10 s | 1.54 s | 581 KiB | |
| 8x | 5.85 s | 1.30 s | 74 KiB | -4% |
| 16x | 5.75 s | 1.22 s | 37 KiB | -6% |

The network time is mostly the 16 round trips (16 x 62 ms = 1.0 s). Raw adds about 0.5 s on
top, more than 581 KiB should need on a 200 Mbit/s link, probably because each prefill starts
cold (slow start). Compression recovers part of that. A real cross-country link is
latency-bound, so compression helps a little.

## 4. Answer quality through the real pool

The 95 held-out questions from `scripts/calibrate_codec.py`, asked through a two-machine
pool, greedy, 14 new tokens:

| Pool | Right | Offline (`docs/compression.md`) |
|---|---|---|
| Raw | 84 / 95 = **88.4%** | 88% |
| 8x (PCA r256, fp16) | 73 / 95 = **76.8%** | 77% |
| 16x (learned r128, fp16) | 55 / 95 = **57.9%** | 58% |

The pool path reproduces the offline evaluation exactly. Gossip, routing, the wire format
and the codec at each end add no error of their own.

## Problems found

1. **A silent device stalls an answer for 5 minutes** (section 1). The pool's hop calls use
   the client's 300 s default deadline. A hop on this model takes well under a second, so a
   deadline of a few seconds (scaled with the number of tokens being sent), or cancelling the
   call when the device is marked down, would turn 300 s into about 1 s. This matters more for
   real laptops than anything else here.
2. **`scripts/setup.sh` is not executable in git** (mode `100644`, while `join.sh` is
   `100755`). `join.sh` runs it directly on the first run, so a fresh clone on Linux or macOS
   fails with `scripts/setup.sh: Permission denied`, which is the very first thing a new user
   does. Fix: `git update-index --chmod=+x scripts/setup.sh`.
3. **A device reaches its own stage through its advertised address.** On a network that does
   not let a machine reach itself through its public address (here AWS; also many home
   routers with Tailscale off), the device cannot use the stage it holds, and every question
   fails with `503: gave up after 6 route changes`, which does not say why. Calling its own
   stage on loopback would avoid the problem. Showing the last underlying error would make it
   diagnosable.

## Not tested

- Windows and macOS. All five machines were Linux.
- GPUs. The free-plan instance types are CPU only.
- More than two stages, so relay vs requester routing (#36, #37) was not measured.
- Discovery by broadcast. AWS networks do not carry broadcast, so devices joined by address.
- The owner-watch feature, and whether devices without the token are refused (the token was
  set, but no refusal was attempted).
- The multi-agent benchmark (#12). `scripts/run_multiagent.py` runs on one machine and has no
  pool mode yet.

## Cost

Five `m7i-flex.large` instances for about 75 minutes, plus a few MB of cross-region traffic:
roughly $1 (estimate, not read from the bill). Every instance was started with a 5-hour
self-terminate timer as a safety net, and all of it (instances, disks, security groups,
key pair) was deleted afterwards.
