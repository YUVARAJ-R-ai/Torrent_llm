# Which hops compression can actually help

_Written alongside the issue #6 profiler, 2026-08-23._

The project's framing documents justify activation compression with a single
number: *"activation tensors between layers are easily 50–200 MB per forward
pass."* That number is right, but it describes **one of two regimes**, and the
other one behaves in the opposite direction. Publishing a compression result
without saying which regime it came from is the most likely way for this project
to put a wrong number in a paper.

## The arithmetic

An activation crossing a hop is exactly `batch × seq_len × hidden_size ×
dtype_bytes`. Nothing about that is approximate, so both regimes can be computed
rather than argued about:

| Model | Regime | Payload per hop | Wire time @ 100 Mbps |
|---|---|---|---|
| Llama-3-8B (hidden 4096) | prefill, 2k context | 16.0 MiB | 1342 ms |
| Llama-3-8B | cached decode, 1 token | 8 KiB | 0.66 ms |
| Llama-3-70B (hidden 8192) | prefill, 2k context | 32.0 MiB | 2684 ms |
| Llama-3-70B | prefill, 8k context | 128.0 MiB | 10737 ms |
| Llama-3-70B | cached decode, 1 token | 16 KiB | 1.31 ms |

Prefill ships the whole sequence. A cached decode step ships **one position** —
three orders of magnitude less.

## What that does to the compression argument

Charging a hop 30 ms of round trip, 50 ms of shard compute, and 5 ms for a
compressor's encode + decode:

| Regime | Wire share of hop | Bandwidth-bound? | Speedup from 8× compression |
|---|---|---|---|
| 8B prefill @ 2k | 94.4% | yes | **5.63×** |
| 8B cached decode | 0.8% | no | **0.95×** _(a loss)_ |
| 70B prefill @ 2k | 97.1% | yes | **6.57×** |
| 70B cached decode | 1.6% | no | **0.95×** _(a loss)_ |

On a cached decode hop a compressor spends two projections of compute to save
about a millisecond of wire time. It makes the system slower. This is not a
tuning problem — no compression ratio fixes it, because the time was never being
spent on the wire.

## What this means for the plan

**It does not weaken the thesis.** It sharpens it into a claim that survives a
reviewer:

1. **Scope the headline claim to prefill and to inter-agent messages.** Both ship
   full sequences of hidden states, both are bandwidth-bound at realistic context
   lengths, and the multi-agent setting is *especially* prefill-heavy — every
   Planner→Critic→Solver handoff is a fresh sequence, not a single token. The
   RecursiveMAS-style latent message is the best case for this method, not an
   afterthought bolted onto a layer-sharding result.

2. **Report the decode regime honestly rather than hiding it.** "Compression
   helps where the hop is bandwidth-bound, and we characterise exactly where that
   boundary is" is a stronger contribution than an unqualified speedup, and it is
   the part a reviewer would otherwise find first.

3. **Always report net latency, not bytes saved.** `HopBudget.with_compression`
   charges the codec's own cost precisely so a "win" that only exists when the
   compressor is treated as free cannot be reported by accident.

4. **The profiler must record the regime.** `HopRecord.phase` is `prefill` or
   `decode`, and `bandwidth_verdict()` refuses to collapse a mixed run into one
   figure.

## Caveat on the current code

`ChainRunner.generate` has no KV cache, so every decode step re-sends the whole
prefix. Its per-step payloads are therefore repeated *prefills*, and the decode
row of the table above is **not** what the current code would measure. Prefill
measurements are unaffected and valid today. A KV cache is a prerequisite for any
decode-regime measurement, and until it exists the decode column stays analytic.

## Measured, not just computed

First real run: Qwen3-0.6B, fp16, two shards on one RTX 4060 Laptop (8 GB), gRPC
over loopback, 5 repeats per context length.
(`runs/profile-qwen3-0.6b-cuda.jsonl`)

| seq_len | activation on hop 1 | shard compute | transport | transport share |
|---:|---:|---:|---:|---:|
| 128 | 256 KiB | 76.8 ms | 6.2 ms | 8% |
| 512 | 1.0 MiB | 85.8 ms | 9.3 ms | 10% |
| 1024 | 2.0 MiB | 99.0 ms | 18.2 ms | 15% |
| 2048 | 4.0 MiB | 210.9 ms | 45.1 ms | 20% |

Two things fall out of this that were not obvious beforehand.

**Loopback is not free.** With no network at all, transport is already 20% of the
hop at 2k context. That is serialisation and gRPC framing, not bandwidth. It also
sets a ceiling: the effective throughput this transport achieves tops out around
**940 Mbps** on loopback, so on a gigabit LAN the implementation, not the link,
would be the binding constraint. Worth knowing before anyone reports a LAN number
as a network measurement.

**With real GPU compute, prefill is firmly bandwidth-bound on a consumer link.**
Projecting the measured 260 ms of shard compute onto 100 Mbps at 30 ms RTT:

| Model | Payload/hop @ 2k | Wire time | Wire share |
|---|---:|---:|---:|
| Qwen3-0.6B | 4.0 MiB | 335 ms | 54% |
| Qwen3-1.7B | 8.0 MiB | 671 ms | 70% |
| Qwen3-8B / Llama-3-8B | 16.0 MiB | 1342 ms | 82% |
| Llama-3-70B | 32.0 MiB | 2684 ms | **90%** |

Every one of those is bandwidth-bound. The project's central premise holds for
prefill, and now with a measured compute term rather than an assumed one.

_Caveat: profiling on CPU instead produces ~15 s of prefill compute, which makes
every hop look compute-bound. That is an artefact of the device, not a result.
`scripts/profile_chain.py` warns when it happens and takes `--project-compute-ms`
to override._

## A payload nobody had counted

The first real profiling run failed outright: `RESOURCE_EXHAUSTED, 622329932 vs
536870912`. Not the activation — the **reply**. The final shard returns logits,
which are `batch × seq × vocab_size`, and vocab dwarfs hidden_size on a modern
tokenizer: 151936 against 1024 on Qwen3-0.6B. A full-sequence logits reply at 1k
context is 594 MiB, roughly **148× the hidden-state activation on the same hop**.

The chain's largest single payload was never an activation at all. Generation
reads one position, so `logits_keep_last` now lets the caller ask for just that;
the default stays "all" because silently truncating logits would corrupt a
perplexity evaluation in a way that looks like a modelling result.

Worth carrying into the design: any future component that returns something
vocab-shaped over the network needs the same treatment.

## How to reproduce

```bash
python scripts/profile_chain.py --model Qwen/Qwen3-0.6B --seq-lens 128,512,1024
```

The projection table at the end of that run is generated by
`torrent_llm.profile.estimate`, and every figure on this page is asserted in
`tests/test_profile.py`.
