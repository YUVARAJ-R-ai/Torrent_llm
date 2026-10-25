# Torrent-LLM

> **BitTorrent for LLMs, where the packets are compressed thought.**

A research project on **distributed recursive multi-agent inference**: a true peer-to-peer volunteer-compute network for LLM inference, where inter-node activation tensors are compressed with learned MLA-style low-rank projections to make inference practical across bandwidth-constrained, heterogeneous consumer hardware — with multi-agent task execution built natively on top of the layer-sharded network.

This is a **2-person academic research project**, not a product. The deliverable is a working prototype plus a benchmark/ablation study suitable for a paper.

---

## The Research Question

> Can MLA-style low-rank latent compression, applied to **inter-agent and inter-node** communication in a distributed P2P setting, reduce end-to-end latency and token cost compared to both token-passing multi-agent systems and naive latent-passing multi-agent systems?

### The gap nobody has addressed

| Source project | What they study | What they miss |
|---|---|---|
| **Petals** | Distributed layer-sharded inference | Multi-agent workflows; learned activation compression |
| **RecursiveMAS** | Latent-space agent communication | Distributed / networked deployment |
| **DeepSeek MLA** | Low-rank KV-cache compression for one model | Cross-agent / cross-node latent transfer |

The insight: *RecursiveMAS's inter-agent latent communication + Petals' layer sharding + MLA's compression math = a primitive nobody has built.*

---

## Architecture

```
P2P Node Discovery (DHT — no central tracker)
        │
Layer Registry / Gossip   "Node A hosts layers 0–16 ×3 replicas"
        │
Activation Compression    low-rank projection (MLA-inspired) per hop,
        │                 ratio adapts to measured bandwidth
Multi-Agent Orchestration RecursiveMAS roles (Planner/Critic/Solver)
        │                 running across sharded layers
Credit / Incentive Layer  compute contributed ↔ inference consumed
```

The core tension: transformer inference is **inherently sequential** (layer N's output is layer N+1's input), so you can't parallelize the forward pass the way BitTorrent parallelizes downloads. What you *can* do is redundant per-layer coverage (fault tolerance), batch parallelism (different requests through different paths), and — the central contribution — **compress the activations that cross the network** instead of shipping full tensors (≈50–200 MB per forward pass on a 70B model).

See [docs/research.md](docs/research.md) for the full competitive landscape, feature list, task breakdown, tech recommendations, and risk log. The original framing is in [brainstorm.md](brainstorm.md).

---

## The Pool

```bash
scripts/pool.sh          # then open http://localhost:3000
```

Devices with different hardware join, lend what their owners are willing to lend,
and serve questions together. Each part of the model lives on several devices, so
when one disappears, even mid-answer, the request carries on through another and the
pool heals itself; an owner can take a device back at any time; and a busy pool
spreads questions across every device that has room. There is no head node: every
device runs the same program, and the dashboard works through whichever one answers.

The lab runs five simulated devices on one machine, each its own process with its own
memory, network link and (for one) GPU. See [docs/pool.md](docs/pool.md) for what is
real and what is simulated, how to run it on real laptops, and what it does not do
yet, and [ADR 002](docs/adr/002-leaderless-pool.md) for the design.

## The Two-Device Demo

```bash
scripts/demo.sh          # then open http://localhost:3000/two-devices
```

This starts two shard processes that each hold half of Qwen3-0.6B, the HTTP
API, and the dashboard. Type a prompt and the page shows the request moving
through the chain: token ids into device A, the hidden state across the link to
device B, and scores back out, one token at a time, with the bytes and timings
of every hop.

Both "devices" run on one machine. Every hop is shaped to a simulated home link
(30 ms, 100 Mbps, set in `configs/demo-2device.yaml`) so the network cost is
visible. The bytes on the wire are real; only the timing is simulated. To use two
real machines (any mix of Windows, Linux and macOS), see
[docs/multi-machine.md](docs/multi-machine.md):

```bash
python scripts/cluster.py worker --config configs/cluster-2laptop.yaml --index 1   # laptop B
python scripts/cluster.py head   --config configs/cluster-2laptop.yaml --local 0   # laptop A
```

The page replays the measured run slowed down so each transfer can be followed.
The dense testing view is still there at `/details`.

---

## Project Status

The layer-sharded pipeline and the device pool are built and measured. The learned compressor and the multi-agent layer exist in the repository but are not yet part of the working system: see the table.

### What runs today

```
torrent-shard --config configs/local-2shard.yaml --index 0   # node A: layers 0-13
torrent-shard --config configs/local-2shard.yaml --index 1   # node B: layers 14-27
torrent-run   --config configs/local-2shard.yaml --prompt "The capital of France is"
```

| Component | State |
|---|---|
| **Device pool** | built — devices find each other by gossip, hold several copies of each model stage, survive a device vanishing mid-answer, hand over when an owner reclaims a GPU, and spread questions across everything with room. See [docs/pool.md](docs/pool.md) |
| Pool access control (#29) | built — a shared token guards the HTTP and gRPC ports; it is access control, not encryption |
| Final shard picks the token (#33) | built — the last hop returns 8 bytes instead of 594 KiB of scores |
| Layer-sharded model hosting (#4) | built — sharded logits are **bit-exact** against the monolithic model |
| gRPC activation transport (#5) | built — 512 MiB message cap, per-hop compute/transport split reported |
| Per-hop bandwidth profiler (#6) | built — JSONL records, hop summaries, projection to unrun model sizes |
| Codec seam | built — `RawCodec` is the uncompressed control; the learned compressor drops in behind the same interface |
| KV cache (#24) | built — cached decode sends one position per step, not a repeated prefill |
| Heterogeneous routing weights (#17) | built — layers split proportionally to relative node capability |
| HTTP layer (#26) | built — `torrent-api`, drives and instruments the chain over HTTP |
| Dashboard (#27) | built — browser view of a run; a testing tool, not the final interface |
| Low-rank compressor (#8) | **partly built, not working** — a random orthonormal projection is merged, and it shrinks the wire 8× but breaks the output ("during" for "Paris"). It has not been calibrated or trained, so it is not in the pool's path and no compression result should be quoted from it |
| Multi-agent roles (#11) | **simulated** — the Planner, Solver and Critic roles are merged, but the latent state passed between them is random numbers and the default run uses scripted text. It is not a result |
| Peer-to-peer relay (#37), credits (#19) | not started — every hop still passes through the device that took the question |

See [docs/setup.md](docs/setup.md) to install and run, and
[docs/adr/001-minimal-harness.md](docs/adr/001-minimal-harness.md) for why this is
a purpose-built harness rather than a Petals fork.

### Two findings that reshape the claim

[docs/bandwidth-regimes.md](docs/bandwidth-regimes.md) — the "50–200 MB per forward
pass" figure describes **prefill**. A *cached decode* hop ships one position: 16 KiB
on a 70B model, ~1.3 ms on a 100 Mbps link against a 30 ms round trip. Compression
buys 6.6× on a 70B prefill hop and **loses 5%** on a cached decode hop, because
there was never any wire time to save. The claim to defend is therefore scoped to
prefill and to inter-agent latent messages — which is the multi-agent case anyway,
since every Planner→Critic→Solver handoff is a fresh sequence rather than a token.
Now measured rather than projected, since #24 landed.

**The chain's largest payload was never an activation.** The final shard returns
logits, which are `batch × seq × vocab_size` — and vocab dwarfs hidden_size on a
modern tokenizer (151936 vs 1024 on Qwen3-0.6B). A full-sequence logits reply at
1k context is 594 MiB, about **148× the hidden-state activation on the same hop**,
and it blew the transport cap on the first real profiling run. Anything returning
a vocab-shaped tensor over the network needs the same treatment.

Work is tracked on the **[GitHub Project board](https://github.com/users/YUVARAJ-R-ai/projects/5)** across 4 one-week sprints that map to the research phases:

| Sprint | Focus |
|--------|-------|
| 1 — Foundations | MHA + MLA math, 2-machine test rig |
| 2 — Reproduce Baselines | Layer-sharded pipeline, bandwidth profiler, RecursiveMAS baseline |
| 3 — Prototype | Low-rank activation compressor, quality harness, agent roles |
| 4 — Research Contribution | Headline benchmark, ablations, writeup |

**Headline experiment:** token-passing MAS vs naive latent-passing MAS vs **compressed** latent-passing MAS — measured on latency, bandwidth, and task quality.

---

## Tech Stack

| Layer | Choice |
|-------|--------|
| Language / framework | Python 3.12 + PyTorch 2.x + transformers 5.x |
| Base model | Qwen3-0.6B/1.7B (dev) → Qwen3-8B (confirmatory) → 70B projected analytically |
| P2P / DHT | [Hivemind](https://github.com/learning-at-home/hivemind) *(post-MVP, issue #15)* |
| Sharding | Purpose-built harness in `src/torrent_llm/shard` — [not a Petals fork](docs/adr/001-minimal-harness.md) |
| Compression | Custom MLA-style low-rank adapters *(the contribution)* — plugs into the existing `Codec` interface |
| Multi-agent | RecursiveMAS RecursiveLink |
| Eval / tracking | lm-eval-harness + Weights & Biases |

---

## Team & Workflow

| Member | Area |
|--------|------|
| [@YUVARAJ-R-ai](https://github.com/YUVARAJ-R-ai) | ML / Research — MLA compressor, eval & ablations, agent roles, benchmark |
| [@Haise-727](https://github.com/Haise-727) | Systems / Networking — sharded pipeline, bandwidth profiler, DHT, fault tolerance |

Pick up an assigned issue from the board's **Ready** column and run `/start-task` to branch, implement, and open a PR.

---

## Key References

- [Petals (arXiv 2209.01188)](https://arxiv.org/abs/2209.01188) — distributed inference baseline
- [Parallax (arXiv 2509.26182)](https://arxiv.org/abs/2509.26182) — heterogeneous-GPU scheduling
- [DeepSeek-V3 Technical Report (arXiv 2412.19437)](https://arxiv.org/pdf/2412.19437) — MLA in production
- [LatentMAS (arXiv 2511.20639)](https://arxiv.org/abs/2511.20639) & [Interlat (arXiv 2511.09149)](https://arxiv.org/abs/2511.09149) — latent agent communication
- [Information-Preserving Compression for Latent MAS (arXiv 2604.13349)](https://arxiv.org/html/2604.13349) — closest prior work
- [MLA explained — Sebastian Raschka](https://sebastianraschka.com/llms-from-scratch/ch04/05_mla/)

---

_License: TBD · See [docs/research.md](docs/research.md) for the detailed plan._
