# Project Research: Torrent-LLM — Distributed Recursive Multi-Agent Inference
_Generated: 2026-06-13_

## Problem & Goal
Distributed P2P LLM inference today (Petals, Parallax, exo) is bottlenecked by the bandwidth cost of shipping full activation tensors between peers on consumer internet, and treats inference as a single-model, single-stream pipeline with no native multi-agent capability. **Goal:** prove that MLA-style learned low-rank compression applied to *inter-node and inter-agent* activation transfer in a true torrent-style P2P mesh reduces end-to-end latency and bandwidth versus both token-passing and naive latent-passing multi-agent systems — and run RecursiveMAS-style agent roles natively across the sharded network.

This is a **2-person academic research project**, not a product. The deliverable is a working prototype + a benchmark/ablation study suitable for a paper.

## Target Users
- ML systems researchers studying decentralized / volunteer-compute inference
- Multi-agent systems researchers exploring latent-space agent communication
- Open-source contributors with idle consumer GPUs (3060–3090 class)
- Labs without budget for centralized A100/H100 clusters

## Competitive Landscape
| Product | Strengths | Weaknesses | Key Takeaway |
|---------|-----------|------------|--------------|
| **Petals** (bigscience) | Mature DHT pipeline, fault-tolerant, dynamic blockwise 8-bit quant halves bandwidth | Single model/stream, no multi-agent, greedy homogeneous-ish scheduling, tiny public swarm (no incentive) | Our baseline. We reuse Hivemind/DHT, extend with learned compression + agents |
| **Parallax** (Gradient, 2025) | Two-phase scheduler for heterogeneous GPUs, 3.6× throughput vs Petals | Still single-model serving; no learned activation compression; no agent layer | Borrow scheduling ideas; differentiate on compression + MAS |
| **exo** (exo-explore) | Easy LAN clustering, Apple-silicon friendly | LAN-focused, not internet-scale P2P, no compression research angle | Good for local 2-machine test rig |
| **LatentMAS / Interlat** (2025–26) | Training-free latent agent communication, −83% tokens, 4× faster | Assumes co-located/low-latency agents; not distributed over internet | Direct MAS baseline to beat; shows latent comms works |
| **comfyui-mesh** | NVENC compresses activations live on the wire across 2 GPUs | Codec compression (not learned/low-rank), diffusion not LLM, 2-GPU only | Validates "compress activations on the wire" is real & needed |
| **DeAI compute nets** (Render, io.net, Akash) | Real tokenomics, sybil-resistant staking | GPU rental marketplaces, not layer-sharded inference | Incentive design reference only |

## Feature List

### Core (MVP)
- [ ] Reproduce a 2-node Petals-style layer-sharded pipeline locally — without this there is nothing to compress or measure
- [ ] Activation-size + bandwidth profiler per layer hop — the central measurement the whole thesis rests on
- [ ] Learned low-rank activation compressor/decompressor (MLA-inspired down/up projection) at each network hop — the core contribution
- [ ] Compression-rank ↔ task-quality evaluation harness (perplexity + 1 downstream benchmark) — proves the tradeoff
- [ ] RecursiveMAS-style agent roles (Planner/Critic/Solver) running across sharded layers — the "multi-agent on the network" claim
- [ ] Three-way benchmark: token-passing MAS vs naive latent MAS vs compressed latent MAS — the headline result

### Important (v1.1)
- [ ] DHT-based peer discovery + layer registry/gossip (replace static config) — needed for >2 nodes & churn claims
- [ ] Redundant layer coverage + reroute on mid-inference node failure — supports the "true torrent / fault tolerance" narrative
- [ ] Network-aware adaptive compression rank (adapt to measured bandwidth between peers) — the "learnable, network-aware" angle
- [ ] Heterogeneous-hardware-aware routing (3060 vs A100) — realistic swarm condition
- [ ] Ablation runner: rank × bandwidth × task performance grid — paper-quality results

### Nice-to-have (Backlog)
- [ ] Credit/incentive accounting layer (compute contributed ↔ inference consumed)
- [ ] Sybil-resistance / proof-of-inference sketch
- [ ] Speculative execution (send to 2 nodes, take fastest)
- [ ] Federated/distributed training of compression adapters
- [ ] Web dashboard for swarm visualization

### Don't Build
- Full blockchain/tokenomics — out of scope for a research prototype; cite DeAI prior work instead
- Production-grade scheduler — Parallax exists; not the contribution
- Custom model training from scratch — use an existing open model (Llama-3 / Qwen)
- Custom DHT — reuse Hivemind

## Task Breakdown

### Foundations
- [ ] Implement vanilla multi-head attention from scratch (S) — shared learning artifact
- [ ] Derive & implement MLA down/up-projection math, validate reconstruction error (M) — feeds the compressor
- [ ] Stand up 2-machine test rig (LAN) with a chosen open model (M)

### 2-Node Sharded Pipeline (Core)
- [ ] Load model, split into layer ranges, serve a range per node (M)
- [ ] Wire sequential activation passing between nodes over sockets/gRPC (M) ← depends on: test rig
- [ ] Activation/bandwidth profiler: log tensor bytes + wall-time per hop (M) ← depends on: pipeline

### Activation Compression (Core)
- [ ] Build low-rank compressor module (down-proj → ship → up-proj) (L) ← depends on: MLA math
- [ ] Insert compressor at each hop; measure bandwidth reduction (M) ← depends on: pipeline + compressor
- [ ] Quality harness: perplexity + downstream benchmark vs compression rank (L) ← depends on: compressor
- [ ] Train/calibrate compression adapters (L) ← depends on: compressor

### Multi-Agent Layer (Core)
- [ ] Integrate RecursiveMAS RecursiveLink modules across sharded layers (L)
- [ ] Implement Planner/Critic/Solver roles over the network (L) ← depends on: pipeline + agents
- [ ] Headline benchmark harness: token vs naive-latent vs compressed-latent MAS (L) ← depends on: compression + agents

### P2P & Robustness (Important)
- [ ] DHT discovery + layer registry/gossip via Hivemind (L)
- [ ] Redundant layer coverage + reroute on node drop (XL) ← depends on: DHT
- [ ] Network-aware adaptive rank selection (L) ← depends on: compressor + profiler
- [ ] Heterogeneous routing weights (M) ← depends on: DHT

### Paper / Results (Important)
- [ ] Ablation grid runner: rank × bandwidth × task (L) ← depends on: quality harness
- [ ] Write up methodology, results, related work (XL)

## Tech Recommendations
| Layer | Recommendation | Reason |
|-------|---------------|--------|
| Language | Python 3.11 + PyTorch 2.x | Ecosystem standard; Petals/RecursiveMAS are PyTorch |
| Base model | Llama-3-8B (dev) → 70B (final runs) | 8B fits one consumer GPU per shard for fast iteration; 70B for the headline bandwidth story |
| P2P / DHT | Hivemind | Battle-tested, what Petals uses; don't reinvent |
| Sharding baseline | Fork Petals primitives | Reuse layer-server abstractions, focus effort on compression |
| Transport | gRPC + (optional) compressed tensor payloads | Matches Petals; easy to instrument |
| Compression | Custom MLA-style low-rank adapters | This *is* the contribution |
| MAS framework | RecursiveMAS RecursiveLink | Provides agent-role + latent-exchange scaffolding |
| Experiment tracking | Weights & Biases or MLflow | Ablation grid needs reproducible logging |
| Eval | lm-eval-harness + perplexity scripts | Standard, comparable numbers |

## Risks & Open Decisions
### Risks
- **Scooped on the core idea** — "Information-Preserving Compression for Latent Multi-Agent Collaboration" (arXiv 2604.13349) is close. Mitigation: differentiate explicitly on the *distributed/networked + layer-sharded* setting, not just MAS; cite and benchmark against it.
- **Bandwidth gains eaten by compute** — compress/decompress cost may offset bandwidth savings on fast links. Mitigation: profiler-first; report net latency, only claim wins where bandwidth-bound.
- **Compression hurts task quality unacceptably** — low rank may degrade reasoning. Mitigation: per-layer-depth rank study; treat rank as tunable, report the frontier not a point.
- **2-person scope creep** — incentive layer + full P2P churn is a lot. Mitigation: Core MVP first; P2P/credits are stretch goals, clearly labeled backlog.
- **70B on consumer GPUs is operationally hard** — Mitigation: prove the method at 8B, project 70B bandwidth numbers analytically + one confirmatory run.

### Open Decisions
- [ ] Base model: Llama-3 vs Qwen-3? (license + RecursiveMAS compatibility)
- [ ] Fork Petals vs build a minimal sharded harness from scratch? (control vs speed)
- [ ] Are compression adapters trained centrally first, or jointly/federated? (start central)
- [ ] Which downstream benchmark is the headline metric? (GSM8K / MMLU / a reasoning set)
- [ ] Scope cut line for the paper: is the incentive layer in or explicitly future work? (recommend: future work)

## GitHub References
- [bigscience-workshop/petals](https://github.com/bigscience-workshop/petals) — baseline architecture; reuse layer-server + DHT primitives
- [learning-at-home/hivemind](https://github.com/learning-at-home/hivemind) — P2P/DHT backbone under Petals
- [exo-explore/exo](https://github.com/exo-explore/exo) — easy LAN clustering for the 2-machine dev rig
- [shootthesound/comfyui-mesh](https://github.com/shootthesound/comfyui-mesh) — proves "compress activations on the wire" is a live problem (NVENC, 2-GPU)
- [rasbt/LLMs-from-scratch (ch04/05_mla)](https://github.com/rasbt/LLMs-from-scratch/blob/main/ch04/05_mla/README.md) — clean MLA reference implementation to base the compressor on
- [zou-group/textgrad](https://github.com/zou-group/textgrad) — text-based optimization for compound agent systems (related work)
- [RUC-NLPIR/ARPO](https://github.com/RUC-NLPIR/ARPO) — agentic tool-use, referenced by RecursiveMAS

## Web / Paper References
- [Petals (arXiv 2209.01188)](https://arxiv.org/abs/2209.01188) — distributed inference baseline
- [Distributed Inference & Fine-tuning over the Internet (arXiv 2312.08361)](https://arxiv.org/pdf/2312.08361) — fault tolerance + load balancing
- [Parallax (arXiv 2509.26182)](https://arxiv.org/abs/2509.26182) — heterogeneous-GPU scheduling, 3.6× over Petals
- [DeepSeek-V3 Technical Report (arXiv 2412.19437)](https://arxiv.org/pdf/2412.19437) — MLA in production (70KB/token vs 192–328KB)
- [LatentMAS — Latent Collaboration in Multi-Agent Systems (arXiv 2511.20639)](https://arxiv.org/abs/2511.20639) — latent agent comms, −83% tokens
- [Interlat — Agents Communicate Entirely in Latent Space (arXiv 2511.09149)](https://arxiv.org/abs/2511.09149) — last-hidden-state comms + compression
- [Information-Preserving Compression for Latent MAS (arXiv 2604.13349)](https://arxiv.org/html/2604.13349) — closest prior work; −79.8–89.4% comms cost
- [MLA explained — Sebastian Raschka](https://sebastianraschka.com/llms-from-scratch/ch04/05_mla/) — derivation walkthrough
