# Research Brainstorm: Distributed Recursive Multi-Agent Inference

> A P2P volunteer compute network for distributed LLM inference, where inter-node activation tensors are compressed using learned low-rank projections, enabling practical bandwidth-constrained inference across heterogeneous consumer hardware — with multi-agent task execution built natively on top of the layer-sharded network.

---

## The Three Source Projects

### Project 2 — Distributed LLM Network (Petals-style)

**What Petals does:**
- Server-client architecture where each server hosts a subset of model layers (transformer blocks)
- You load a small part of the model, then team up with others serving remaining parts
- Pipeline parallelism over the internet — your laptop runs embeddings, someone's 3090 runs layers 12–24, etc.
- Uses DHT (distributed hash table) for node discovery
- Inference path is sequential, coordinated through a tracker-like health monitor

**Core gap:**
- Designed for a single model, single inference stream — no notion of multi-agent workflows running across the network
- Still somewhat centralized — a node going offline mid-inference kills the request
- More like a CDN with layer sharding than true BitTorrent
- Public swarm is tiny because there's no incentive to contribute compute

---

### Project 4 — DeepSeek v4 Training & Inference Innovations

**Key innovations to understand (in learning order):**
1. Standard MHA attention math (Q, K, V projections) — prerequisite
2. **MLA (Multi-head Latent Attention)**: instead of caching full K/V, caches a compressed latent `c` and reconstructs K/V from it via up-projection matrices
   - Math: `c = W_c * h`, `K = W_k * c`, `V = W_v * c` where `W_c` is a low-rank down-projection
   - Massively reduces KV cache size — critical for long contexts
3. **MoE (Mixture of Experts) routing**: top-k gating, load balancing loss
4. **Multi-token prediction**: single forward pass predicts next N tokens via parallel heads
5. **FP8 mixed-precision training**

**Why it matters for this research:**
- MLA's low-rank compression of hidden states is the theoretical foundation for compressing inter-agent/inter-node activation transfers
- MoE routing logic is analogous to dynamic node selection in the P2P network

---

### Project 5 — RecursiveMAS (Latent-Space Multi-Agent Collaboration)

**What it does:**
- Casts the entire multi-agent system as a unified recursive computation
- Heterogeneous agents connected through lightweight **RecursiveLink** modules
- Agents iteratively exchange, refine, and evolve their **latent states** across recursion rounds
- Instead of passing text tokens between agents, they pass compressed hidden states

**Collaboration styles supported:**
- Sequential (Planner → Critic → Solver)
- Mixture (domain-specialist agents + summarizer)
- Distillation (Expert ↔ Learner recursive interaction)
- Deliberation (Reflector ↔ Tool-Caller for tool-integrated reasoning)

**Core gap:**
- Assumes agents are co-located or at least low-latency adjacent
- No consideration of distributed/networked deployment
- Naively shipping latent vectors across the internet is worse than tokens without proper compression

---

## The Convergence — Novel Research Angle

### The Gap Nobody Has Addressed

| Project | What they study | What they miss |
|---|---|---|
| Petals | Distributed inference | Multi-agent workflows |
| RecursiveMAS | Latent-space agent communication | Distributed/networked deployment |
| DeepSeek MLA | KV cache compression for single model | Cross-agent / cross-node latent transfer |

**The insight:** RecursiveMAS's inter-agent latent communication + Petals' layer sharding + MLA's compression math = a new primitive nobody has built.

### Research Question

> Can MLA-style low-rank latent compression applied to inter-agent communication in a distributed P2P setting reduce end-to-end latency and token cost compared to both token-passing MAS and naive latent-passing MAS?

---

## Architecture: True Torrent-Style (Not Client-Server)

### Why Petals isn't really BitTorrent

Petals uses DHT for discovery but inference is still sequential and centralized-adjacent. True P2P means:
- No central health monitor or tracker
- Redundant layer coverage across multiple nodes
- Dynamic routing — a request can take different paths through the network
- If a node drops mid-inference, activation reroutes to another node hosting the same layers
- Every node is simultaneously a consumer and a contributor

### The Core Tension

Transformer inference is **inherently sequential** — layer N's output is layer N+1's input. Unlike BitTorrent where chunks are independent, activations have a strict dependency chain. You can't parallelize the forward pass the way torrents parallelize downloads.

**What you CAN do:**
- **Redundancy per layer range**: 3 nodes host layers 0–16, 3 nodes host layers 16–32, etc. Fault tolerance, not speed.
- **Batch parallelism**: Different *requests* routed through different node paths simultaneously. Your query and mine go through different physical nodes but the same logical layer ranges.
- **Speculative execution**: Send activation to 2 nodes hosting the same layer, use whichever responds first. Wastes compute, kills tail latency.

### The Bandwidth Problem

For a 70B model, activation tensors between layers are roughly `hidden_size × sequence_length × float16` — easily **50–200MB per forward pass** crossing the network. On consumer internet (100Mbps asymmetric), bandwidth is the bottleneck, not compute.

**This is where MLA compression becomes the core contribution:**
- Instead of shipping full activation tensors between peers, compress to a low-rank latent
- Ship the compressed latent, reconstruct on the receiving node
- RecursiveLink modules (from RecursiveMAS) do this between agents — you'd do it between *network hops*
- The compression ratio is learnable and can be network-aware (adapt to measured bandwidth between nodes)

### Incentive Layer (What Makes It Self-Sustaining)

Petals' public swarm is tiny because there's no reason to contribute. A credit system:
- Contribute compute (host layers, process activations) → earn inference credits
- Consume inference → spend credits
- Creates a self-sustaining volunteer network
- Similar to how BitTorrent's tit-for-tat protocol incentivizes seeding

---

## System Components

```
┌─────────────────────────────────────────────────────────┐
│                    P2P Node Discovery                    │
│              (DHT — no central tracker)                  │
└─────────────────────────────────────────────────────────┘
                           │
┌─────────────────────────────────────────────────────────┐
│                  Layer Registry / Gossip                 │
│   "Node A hosts layers 0-16 of Llama3-70B (3 replicas)" │
└─────────────────────────────────────────────────────────┘
                           │
┌─────────────────────────────────────────────────────────┐
│              Activation Compression Layer                │
│   Low-rank projection (MLA-inspired) before each hop    │
│   Compression ratio adapts to measured node bandwidth   │
└─────────────────────────────────────────────────────────┘
                           │
┌─────────────────────────────────────────────────────────┐
│           Multi-Agent Orchestration Layer                │
│   RecursiveMAS-style agent roles (Planner/Critic/Solver) │
│   running across different layer shards on the network  │
└─────────────────────────────────────────────────────────┘
                           │
┌─────────────────────────────────────────────────────────┐
│                   Credit/Incentive Layer                 │
│        Compute contributed ↔ Inference consumed         │
└─────────────────────────────────────────────────────────┘
```

---

## Open Research Questions

1. **Compression fidelity vs. bandwidth tradeoff**: What rank is sufficient to preserve task performance across different layer ranges? Does it vary by layer depth?

2. **Routing under churn**: How do you handle mid-inference node failures in a P2P mesh without restarting the full forward pass?

3. **Heterogeneous hardware**: Nodes will have vastly different compute/memory (3060 vs A100). How does routing account for this without creating bottlenecks?

4. **Agent role placement**: Should specialized agent roles (planner, critic) be pinned to specific layer ranges, or dynamically placed? What's the interaction between layer depth and agent specialization?

5. **Training the compression adapters**: Can RecursiveLink-style adapters be trained in a federated/distributed way, or do you need a centralized training phase first?

6. **Credit system game theory**: How do you prevent sybil attacks (fake nodes claiming credit) and free-riding?

---

## Prior Work to Read

| Paper / Project | Why |
|---|---|
| [Petals (Borzunov et al., 2023)](https://arxiv.org/abs/2209.01188) | Baseline distributed inference architecture |
| [RecursiveMAS (Yang et al., 2026)](https://arxiv.org/abs/2604.25917) | Latent-space multi-agent communication |
| [DeepSeek-V2](https://arxiv.org/abs/2405.04434) | MLA math — read this before V3/V4, clearer derivation |
| [Hivemind](https://github.com/learning-at-home/hivemind) | P2P deep learning library (used by Petals under the hood) |
| [TextGrad](https://github.com/zou-group/textgrad) | Text-based optimization for compound agentic systems |
| [ARPO](https://github.com/RUC-NLPIR/ARPO) | Agentic tool-use, referenced by RecursiveMAS |

---

## Learning Path to Execute This

### Phase 1 — Foundations (Do first)
- [ ] Andrej Karpathy "Neural Networks: Zero to Hero" — attention math, backprop
- [ ] Implement vanilla multi-head attention from scratch
- [ ] Read DeepSeek-V2 paper, derive MLA math by hand

### Phase 2 — Reproduce Baselines
- [ ] Run RecursiveMAS sequential_light locally, read `modeling.py` (RecursiveLink implementation)
- [ ] Set up a local Petals swarm across 2 machines, profile activation tensor sizes
- [ ] Measure actual bandwidth cost per layer hop

### Phase 3 — Prototype
- [ ] Implement low-rank activation compression between Petals-style layer hops
- [ ] Measure compression ratio vs. task quality tradeoff (perplexity or downstream benchmark)
- [ ] Integrate RecursiveMAS agent roles across sharded layers

### Phase 4 — Research Contribution
- [ ] Formal benchmark: token-passing MAS vs. naive latent MAS vs. compressed latent MAS
- [ ] Ablation: compression rank × network bandwidth × task performance
- [ ] Write up and submit

---

## One-Line Pitch

> **"BitTorrent for LLMs, where the packets are compressed thought."**
