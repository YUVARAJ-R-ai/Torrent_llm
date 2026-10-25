# Cluster Control Plane — Implementation Plan

> **Superseded in part (2026-10-23).** This plan has one head laptop driving
> worker agents. The direction since agreed is a **peer-to-peer** network:
> identical peers, a shared membership table, any peer can accept a prompt, and
> hidden states relayed peer to peer. The work is tracked as #33 (final shard
> picks the token), #37 (relay forwarding), #34 (peer daemon and discovery),
> #35 (dashboard served by any peer) and #36 (throughput benchmark). The
> connectivity, partitioning, deploy-progress and security sections below
> still apply to that design.

**Goal.** Run one model split across two (or more) laptops, managed entirely
from **one browser tab on one laptop**. Nobody types commands on the second
laptop beyond starting it once. The two laptops connect over the same Wi-Fi or
a hotspot ("direct"), or over different networks ("remote"), and it works on
any mix of Windows, Linux and macOS.

**Builds on.** The work already in this branch: `torrent-shard --bind`,
`device: auto`, `hostname`/`os` in `Info`, per-hop logs, `scripts/cluster.py`,
and the setup scripts. Today those need a hand-copied YAML on every laptop and a
terminal on each one. This plan replaces both with a control plane.

---

## 1. The idea in one picture

```
            Laptop A  (the "head": where you click)              Laptop B (and C, D…)
 ┌──────────────────────────────────────────────┐        ┌───────────────────────────────┐
 │ Browser ── Dashboard (Next.js)               │        │                               │
 │              │ HTTP                          │        │                               │
 │ torrent-api + /cluster  (the coordinator) ───┼─HTTP──▶│ torrent-agent  :50050         │
 │   - node registry, discovery, planner        │ control│   - reports hardware, models  │
 │   - deploy / stop / status                   │        │   - downloads weights         │
 │   - chain runner (unchanged)                 │        │   - starts/stops its shard    │
 │              │                               │        │              │ spawns         │
 │ torrent-agent :50050 ── spawns shard 0       │        │ torrent-shard :50051          │
 │ torrent-shard :50051  ◀──────────────────────┼─gRPC──▶│ (layers 14–27 + LM head)      │
 └──────────────────────────────────────────────┘  data  └───────────────────────────────┘
```

Two planes, kept separate on purpose:

- **Control plane (new, HTTP/JSON):** "who are you, what hardware do you have,
  download this model, load layers 14–27, stop, what's your status". It is small,
  slow and human-paced.
- **Data plane (exists, gRPC):** hidden states and logits. It is unchanged, so
  every measurement and test in the repo still holds.

The **agent** is the only thing that ever runs on laptop B. It loads no model
until told to, so it can stay running all day and be reconfigured from the
browser as many times as needed.

---

## 2. What the user does (the target experience)

1. **Once per laptop:** `scripts/setup.sh` / `setup.ps1` (exists), then
   `torrent-agent` (or double-click `start-agent.bat` / `start-agent.sh`). The
   agent prints a **6-digit pairing code** and its addresses.
2. **On laptop A:** `python scripts/cluster.py ui`. This starts A's own agent,
   the API and the dashboard, and opens the browser at **Cluster**.
3. **Cluster page → Nodes:** laptops on the same network appear by themselves
   (direct). For a remote laptop, type its Tailscale name or IP. Enter its
   pairing code. Each node card shows hostname, OS, CPU/GPU, RAM/VRAM, and
   whether the model is already downloaded.
4. **Cluster page → Partition:** pick the model. A bar of 28 layers shows a
   suggested split, sized to each laptop's memory and speed. Drag the boundary
   to change it, or click "even split".
5. **Deploy.** A progress list per node: *downloading 43%* → *loading layers
   14–27* → *serving* → *chain verified*. Failures say what to fix ("laptop B's
   firewall is blocking 50051 — here is the command").
6. **Chat page** (the current landing page) works exactly as today, now with
   real hostnames and a *Real network · 2 machines* badge.
7. **Stop / Re-partition** from the same page, without touching laptop B.

---

## 3. Connectivity: direct and remote Wi-Fi

| Situation | How nodes find each other | How they connect |
|---|---|---|
| Same Wi-Fi / router / phone hotspot | **Automatic:** UDP broadcast beacon on port 50050 | LAN IPs |
| No router at all | One laptop turns on its OS hotspot (Windows *Mobile hotspot*, `nmcli dev wifi hotspot`, macOS Internet Sharing); the other joins it | hotspot IPs, auto discovery as above |
| Different networks (home ↔ campus, remote friend) | **Manual:** type the Tailscale MagicDNS name or `100.x` IP | Tailscale (WireGuard, NAT traversal, relay fallback) |
| Campus Wi-Fi with client isolation | Broadcasts are blocked → use Tailscale or a hotspot | — |

Decisions and reasons:

- **Discovery uses a plain UDP broadcast with the standard library**, not mDNS.
  The `zeroconf` package would be more robust but adds a dependency, and mDNS is
  unreliable on Windows without Bonjour. Each agent broadcasts
  `{"torrent_agent":1, "id", "hostname", "port"}` every 2 s. The head listens and
  also sends a probe on demand ("Scan" button).
- **Remote uses Tailscale rather than our own relay.** A custom NAT traversal or
  relay server is a project of its own. Tailscale is free, cross-platform, and
  already installed on laptop A. The agent detects Tailscale (`tailscale ip -4`)
  and reports those addresses so the UI can show "reachable via Tailscale".
- **Address choice is automatic.** An agent reports *all* its addresses (LAN,
  hotspot, Tailscale). The head tries them in order (same subnet as itself first,
  then Tailscale) and uses the first that answers. The chosen address is what
  goes into the shard topology, which is why nobody has to type IPs into YAML.

---

## 4. Components

### 4.1 `torrent-agent` (new) — `src/torrent_llm/agent_daemon/`

A FastAPI process on port **50050**. It uses the `api` extra the repo already
has, so no new dependencies.

| Endpoint | Purpose |
|---|---|
| `GET /node` | id, hostname, OS, arch, CPU count, RAM, GPU name/VRAM, resolved `device: auto`, addresses, agent version, cached models |
| `POST /pair` | `{code}` → returns a bearer token; every other call needs it |
| `GET /models` | models in the local HF cache, with sizes |
| `POST /models/download` | `{model_id}`, async; progress through `/status` |
| `POST /shard` | `{topology, index}` → writes the topology to a temp file and spawns `torrent-shard --bind 0.0.0.0 --config … --index …` |
| `DELETE /shard` | stops the shard process |
| `GET /status` | state machine: `idle → downloading(p) → loading → serving → error(msg)`, plus the last 50 log lines |
| `GET /logs` (SSE) | live shard log, including the per-hop lines, so the browser can show laptop B's traffic |

It reuses the following from `scripts/cluster.py` (moved into the package as
`torrent_llm/netutil.py`): `local_addresses()`, `firewall_hint()`, and the
process-group start/stop logic.

The agent passes the topology as **JSON over the wire** instead of a copied
file. That removes the "the two laptops have different configs" failure
entirely.

### 4.2 Coordinator — `src/torrent_llm/api/cluster.py` (new router)

Mounted into `torrent-api`. `create_app` changes from taking a fixed
`TopologyConfig` to holding a **mutable current deployment**. `/generate` and the
other endpoints return `409 "no chain deployed"` until one exists. Starting
`torrent-api --config x.yaml` still works and simply pre-loads that deployment,
so the existing demo and tests do not change.

| Endpoint | Purpose |
|---|---|
| `GET /cluster/nodes` | discovered and added nodes with their `/node` info and reachability |
| `POST /cluster/nodes` | add by host/IP/Tailscale name + pairing code |
| `POST /cluster/scan` | trigger UDP discovery |
| `POST /cluster/plan` | `{model_id, node_ids, mode: auto/even/manual, boundaries?}` → proposed split plus a memory estimate per node, with warnings |
| `POST /cluster/deploy` | runs the deployment (below); progress through `GET /cluster/deployment` |
| `DELETE /cluster/deploy` | stops every shard |
| `GET /cluster/events` (SSE) | node and deployment state changes for the UI |

**Deploy sequence:**
1. Pick an address per node (§3).
2. Build the `TopologyConfig` in memory (model, dtype, nodes, boundaries).
3. Ask every node to download the model if it isn't cached, in parallel.
4. Start shards **last-to-first** so the head shard comes up last.
5. Wait for each `/status == serving`.
6. Run the existing `check_nodes` logic: TCP probe + gRPC `Info`, comparing
   `model_id` and the layer range.
7. Warm up with a 2-token generate.
8. Swap the deployment in atomically.

Any failure stops what was started and reports the step and node.

**Planner (`auto` mode).** It weights layers by usable memory: VRAM if the
device is CUDA/MPS, else ~60% of free RAM. It uses the existing
`plan_weighted()` and rejects a split where a node's estimated share
(`layers × bytes_per_layer + embed/head`) exceeds its memory. `bytes_per_layer`
comes from the model config, without downloading weights.

### 4.3 Dashboard — `dashboard/app/cluster/page.tsx` (new) + nav

- **Nodes panel:** node cards (hostname, OS icon, device, RAM/VRAM, model
  cached ✓, connection type LAN/Tailscale, latency), "Scan", "Add by address",
  and a pairing-code dialog.
- **Partition panel:** model picker, a 28-segment layer bar colored per node
  with draggable boundaries, an estimated memory per node, and Even/Auto
  buttons.
- **Deploy panel:** per-node step list driven by `/cluster/events`, a Stop
  button, and inline error and fix hints.
- **Live traffic panel:** each node's per-hop log lines (SSE through the head),
  so laptop B's traffic is visible without looking at laptop B.
- The existing landing page becomes **Chat**. It is disabled with a "Deploy a
  chain first" link when nothing is deployed.

### 4.4 Launcher

`scripts/cluster.py ui` starts the local agent, then `torrent-api` with no
config, then the dashboard, and opens the browser. `worker` becomes a thin alias
for "start the agent". `head` and `check` stay for scripted runs and benchmarks.

---

## 5. Model weights on each laptop

- **Default:** each agent downloads from the Hugging Face Hub itself, with the
  progress shown in the UI. This needs internet on each laptop at deploy time,
  or a pre-download from setup.
- **Offline venue (phase 3):** the head serves the files from its own HF cache
  over the agent connection (`GET /cluster/weights/{model}/{file}`). The
  receiving agent writes them into its local cache layout. For Qwen3-0.6B that
  is ~1.5 GB over the LAN, about 2–3 minutes on decent Wi-Fi.
- **Later (phase 3):** load only the shard's own tensors from safetensors
  instead of loading the full model and pruning it, which halves peak RAM per
  node. `load_shard`'s contract already allows this.

---

## 6. Security (resolves part of #29)

- **Pairing code:** 6 random digits printed by the agent at start. The head
  exchanges it once for a random bearer token kept in memory. Without the
  token, the agent answers only `GET /node` (for discovery) and refuses
  everything else.
- The gRPC data plane stays unauthenticated (unchanged). The shard is only
  started on demand and stopped on Stop, which shrinks the exposure window.
  The docs keep saying "trusted network or tailnet only".
- No secrets are stored on disk.

---

## 7. Phases

### Phase 1 — Centralized control, manual add (the minimum demo)
1. Move the shared helpers into `torrent_llm/netutil.py`; `cluster.py` imports
   them.
2. `torrent-agent`: `/node`, `/pair`, `/shard` (POST/DELETE), `/status`, the
   process supervisor, and the pairing code. Add a `torrent-agent` entry point
   in `pyproject.toml`.
3. Coordinator router: node registry (add by address), `/plan` (even + manual),
   `/deploy` with the sequence in §4.2, `/deployment`, Stop. Make the API
   deployment mutable.
4. Dashboard Cluster page: add node, node cards, layer bar with draggable
   boundary, deploy progress (polling is fine in phase 1), Stop. Chat gated on a
   deployment.
5. `cluster.py ui`.
6. Tests (§8). Update `docs/multi-machine.md`.

*Outcome:* start the agent on B once; everything else happens in the browser on
A. Works on the same Wi-Fi, a hotspot, or Tailscale, by typing B's address.

### Phase 2 — Discovery and smarter setup
1. UDP beacon and listener, plus the Scan button. Nodes appear on their own on
   the same network.
2. Multi-address reachability with automatic choice (LAN vs Tailscale), and
   latency shown per node.
3. Hardware reporting (RAM/VRAM/GPU) and the `auto` planner with memory checks.
4. Model download via the agent, with progress.
5. SSE for events and the live traffic panel.

### Phase 3 — Robustness and offline
1. Weight transfer from the head's cache.
2. Lazy per-shard tensor loading.
3. Agent auto-reconnect; redeploy one node without restarting all of them.
4. More than two nodes in the UI (the backend already supports N).
5. Hooks for #15/#16 (replicas, rerouting) without UI changes now.

---

## 8. Testing

All offline, matching the existing suite's style (tiny random Llama, no
downloads):

- **Agent:** pairing (refuses without a token, accepts with it); `/shard`
  spawns a shard on an OS-assigned port and `/status` reaches `serving`;
  `DELETE` frees the port. Use the tiny model via a `model_factory` hook in
  place of `load_shard`.
- **Coordinator:** deploy against two in-process agents → `/generate` output is
  bit-exact against the monolithic model (the same anchor as
  `test_transport.py`). A failed node rolls back the others. A mismatched layer
  range is refused. `/generate` returns 409 before any deployment.
- **Planner:** `auto` respects memory and never gives a node zero layers;
  manual boundaries are validated with `validate_plan`.
- **Netutil:** address choice prefers a shared subnet, then Tailscale.
- **Dashboard:** `tsc --noEmit` and eslint. A manual click-through on two real
  laptops (one Windows, one Linux) as the acceptance test.

**Acceptance (phase 1):** from a fresh boot, laptop B runs only
`torrent-agent`. From laptop A's browser alone: add B, split 28 layers, deploy,
ask a question, see B's hop lines in the UI, stop, re-split 10/18, deploy again,
ask again. This works both on a phone hotspot and over Tailscale on two
different networks.

---

## 9. Non-goals (for now)

- No public-internet swarm or untrusted peers (that is #15/#29 territory).
- No custom NAT traversal or relay server; Tailscale covers remote.
- No Wi-Fi Direct (P2P) mode. It is not reliably scriptable across
  Windows/Linux/macOS; an OS hotspot gives the same "no router" result.
- No change to the data-plane wire format or the research measurements.

## 10. Risks

| Risk | Mitigation |
|---|---|
| Windows Firewall blocks 50050/50051 | agent checks at start and shows the exact `New-NetFirewallRule` command; the UI repeats it when a probe times out |
| Broadcast blocked (campus Wi-Fi, some hotspots) | manual add always available; Tailscale |
| Laptop sleeps mid-demo | agent requests "no sleep" while a shard is serving (Windows `SetThreadExecutionState`, Linux `systemd-inhibit`, macOS `caffeinate`) — phase 2 |
| Different package versions on A and B | `/node` reports the agent version; the head refuses a mismatch with a clear message |
| Weights missing and no internet | pre-download in setup (exists); weight transfer from the head (phase 3) |

## 11. Estimate

| Phase | Rough effort |
|---|---|
| 1 | 1.5–2 days (agent ~0.5, coordinator ~0.5, dashboard page ~0.5–1) |
| 2 | 1.5 days |
| 3 | 2–3 days |

If the review is before phase 1 lands, `scripts/cluster.py worker/head`
(already built and tested) is the fallback. It is the same rig, driven from two
terminals instead of one browser.
