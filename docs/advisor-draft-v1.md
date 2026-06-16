# Torrent-LLM — Initial Research Draft
_Prepared for faculty advisor review · June 2026_
_Authors: Yuvaraj R. · Haise_

---

## 1. Problem Statement

Running large language models (LLMs — AI text generation systems like GPT or Llama) requires significant compute. Most approaches either rent expensive centralised GPU (Graphics Processing Unit) servers or are limited to a single powerful machine. An alternative is to distribute the model across multiple consumer machines over the internet — each machine handles a slice of the model, and they pass intermediate results (called "activations") between each other to complete a single inference (generation) request.

Petals (Borzunov et al., 2023) demonstrated this is possible: split a model's layers across volunteer machines and run inference collaboratively. However, two significant problems remain unsolved:

1. **Bandwidth cost.** When a 70-billion-parameter model runs across the internet, the activation tensors (the intermediate numerical outputs) passed between machines can reach 50–200 MB per request. On typical consumer internet (≈100 Mbps), this makes the network the bottleneck — not the compute.

2. **No native multi-agent support.** Recent work (RecursiveMAS, Yang et al., 2026) shows that letting AI agents communicate through compressed internal states rather than plain text reduces communication cost by up to 83% and speeds up collaborative reasoning. But this system assumes all agents are co-located on low-latency hardware — it was never designed for distributed deployment over the internet.

These two problems have been studied in isolation. No existing work addresses them together.

---

## 2. Research Question

> Can learned low-rank compression — inspired by DeepSeek's Multi-head Latent Attention (MLA) technique — applied to the activations passed between nodes in a distributed peer-to-peer (P2P) network reduce end-to-end latency and bandwidth usage compared to (a) passing plain text tokens between agents, and (b) passing uncompressed internal states between agents?

The core hypothesis is that compressing what travels over the wire — rather than shipping full activation tensors — makes distributed multi-agent inference practical on consumer-grade internet hardware.

---

## 3. Proposed Approach

We combine three existing lines of work into a novel system:

| Source | What it contributes |
|--------|---------------------|
| **Petals** | Layer-sharded P2P inference — each node hosts a contiguous slice of model layers |
| **DeepSeek MLA** | Low-rank compression math — compresses large tensors into a small latent (compact representation) before transmission, then reconstructs on the receiving end |
| **RecursiveMAS** | Multi-agent roles (Planner, Critic, Solver) that communicate through internal states instead of text |

The novel contribution is applying MLA-style compression specifically at the network boundary — between nodes in a P2P mesh — so that agents running on different physical machines can communicate efficiently.

### System Architecture (simplified)

```
[Node A — layers 0–16]  →  compress activation  →  send over internet
                                                         ↓
                                               [Node B — layers 16–32]
                                               decompress → continue inference
```

On top of this layer-sharding pipeline, multi-agent roles (Planner/Critic/Solver) are distributed across nodes. The agents exchange compressed internal states rather than text, reducing communication volume while preserving task quality.

The compression ratio is **learned and tunable** — a small adapter network (down-projection matrix) reduces the activation to a compact form before transmission; a corresponding up-projection reconstructs it on arrival. The rank (size) of this bottleneck controls the quality/bandwidth tradeoff and can be adapted based on measured network speed between nodes.

---

## 4. Competitive Landscape

| System | What it does well | What it misses |
|--------|-------------------|----------------|
| **Petals** (2023) | Mature distributed inference; fault-tolerant; used in practice | No compression of activations; no multi-agent layer |
| **Parallax** (2025) | 3.6× throughput improvement over Petals on mixed hardware | Still single-model serving; no learned compression; no agents |
| **LatentMAS / Interlat** (2025–26) | Agents communicate in compressed latent space; −83% token cost | Assumes co-located agents; not designed for internet-scale P2P |
| **exo** | Easy local clustering across consumer machines | LAN-only; not internet-scale; no compression research angle |
| **arXiv 2604.13349** (closest prior work) | Compression for latent multi-agent communication; −79–89% cost | Not distributed; no layer-sharded inference across the network |

**Our differentiation:** the specific combination of layer-sharded P2P infrastructure + learned compression at the network hop + multi-agent roles running natively across that network. No existing work does all three.

---

## 5. Planned Experiments

The headline experiment compares three conditions on a multi-step reasoning task:

1. **Token-passing** — agents send plain text to each other (standard approach)
2. **Naive latent-passing** — agents send raw uncompressed internal states over the network
3. **Compressed latent-passing** — agents send MLA-compressed internal states *(our method)*

Metrics: end-to-end latency, bytes transferred per request, and task accuracy (perplexity + one reasoning benchmark such as GSM8K).

A secondary ablation study will vary the compression rank against network bandwidth and task quality to characterise the tradeoff frontier.

**Development model:** Llama-3-8B for iteration; scale to 70B for the headline bandwidth story.

---

## 6. Key References

- Borzunov et al. (2023). *Petals: Collaborative Inference and Fine-tuning of Large Models.* arXiv:2209.01188
- Yang et al. (2026). *RecursiveMAS: Recursive Multi-Agent Systems via Latent Collaboration.* arXiv:2604.25917
- DeepSeek-AI (2024). *DeepSeek-V3 Technical Report.* arXiv:2412.19437 _(MLA compression reference)_
- Wan et al. (2025). *Parallax: Heterogeneous GPU Scheduling for Distributed LLM Inference.* arXiv:2509.26182
- Chen et al. (2025). *LatentMAS — Latent Collaboration in Multi-Agent Systems.* arXiv:2511.20639
- Xu et al. (2025). *Interlat — Agents Communicate Entirely in Latent Space.* arXiv:2511.09149
- Anonymous (2026). *Information-Preserving Compression for Latent MAS.* arXiv:2604.13349 _(closest prior work)_

---

## 7. Current Status & What We Are Seeking Feedback On

The project is at the end of the planning phase. The research question, competitive landscape, and feature roadmap are documented. Implementation has not started.

**We would appreciate guidance on:**

1. **Scope** — Is the core MVP (2-node pipeline + compression + 3-way benchmark) sufficient for a publishable contribution, or does the P2P/fault-tolerance layer need to be in-scope too?
2. **Differentiation** — Given arXiv 2604.13349 (above), is the distributed/networked framing a strong enough differentiator, or should we narrow the claim?
3. **Benchmark choice** — Is GSM8K an appropriate headline task for demonstrating reasoning quality under compression, or do you recommend an alternative?
4. **Timeline** — We are targeting a 4-sprint (~4 week) implementation cycle. Does this seem realistic for a 2-person team?

---

_Internal review status: [ ] Yuvaraj · [ ] Haise_
_Sent to advisor: —_
