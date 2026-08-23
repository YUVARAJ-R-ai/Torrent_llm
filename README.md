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

## Project Status

The layer-sharded pipeline is built and measured; the compressor is next.

### What runs today

```
torrent-shard --config configs/local-2shard.yaml --index 0   # node A: layers 0-13
torrent-shard --config configs/local-2shard.yaml --index 1   # node B: layers 14-27
torrent-run   --config configs/local-2shard.yaml --prompt "The capital of France is"
```

| Component | State |
|---|---|
| Layer-sharded model hosting (#4) | built — sharded logits are **bit-exact** against the monolithic model |
| gRPC activation transport (#5) | built — 512 MiB message cap, per-hop compute/transport split reported |
| Per-hop bandwidth profiler (#6) | built — JSONL records, hop summaries, projection to unrun model sizes |
| Codec seam | built — `RawCodec` is the uncompressed control; the learned compressor drops in behind the same interface |
| Learned low-rank compressor (#8) | not started — the contribution |
| KV cache | not started — required before any decode-regime measurement |

See [docs/setup.md](docs/setup.md) to install and run, and
[docs/adr/001-minimal-harness.md](docs/adr/001-minimal-harness.md) for why this is
a purpose-built harness rather than a Petals fork.

### One finding that reshapes the claim

[docs/bandwidth-regimes.md](docs/bandwidth-regimes.md) — the "50–200 MB per forward
pass" figure describes **prefill**. A *cached decode* hop ships one position: 16 KiB
on a 70B model, ~1.3 ms on a 100 Mbps link against a 30 ms round trip. Compression
buys 6.6× on a 70B prefill hop and **loses 5%** on a cached decode hop, because
there was never any wire time to save. The claim to defend is therefore scoped to
prefill and to inter-agent latent messages — which is the multi-agent case anyway,
since every Planner→Critic→Solver handoff is a fresh sequence rather than a token.

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
