# The pool

Many devices, lending what their owners are willing to lend, serving questions
together, and staying up when one of them goes away.

```bash
scripts/pool.sh          # then open http://localhost:3000/pool
```

That starts five simulated devices (each its own process, with its own memory
budget and network link) and the dashboard. See [What is simulated](#what-is-simulated)
for what that means.

## The idea

The model is cut into **stages**, contiguous ranges of layers. A question passes
through every stage once per token. Each stage should live on **several devices**,
so a question can go through any one of them.

```
                    ┌─ Gaming laptop ─┐      ┌─ Gaming laptop ─┐
  question ──────▶  │   (stage 1)     │ ───▶ │   (stage 2)     │ ──────▶ answer
                    ├─ MacBook Air ───┤      ├─ Phone hotspot ─┤
                    │   (stage 1)     │      │   (stage 2)     │
                    └─────────────────┘      └─────────────────┘
                     any one of these         any one of these
```

Three things follow, and they are the point of the project:

| You want | What the pool does |
|---|---|
| **No downtime** | If a device disappears, even mid-answer, the request moves to another copy and carries on. The pool then asks a spare device to take the lost share. |
| **Idle devices used** | A device with room offers it. A busy pool spreads questions across every copy, weighted by how fast each device is. |
| **Owners stay in charge** | An owner can take a device back at any time. It hands its work over first, then frees its memory. |

## Using it

Open **Pool** in the dashboard.

- **Stage columns.** Each card is one device holding one copy of that stage. A
  device with room for two stages appears in both columns.
- **Standing by.** Devices holding nothing: spare, away, or offline.
- **Ask.** The answer streams in while the animation shows which device ran each
  stage, how many bytes crossed, and how long the network and the compute took.
- **Ask, then crash a device.** Asks a long question, then kills a device partway
  through. Watch the failed attempt, the device grey out, the request continue
  through another route, and the pool heal itself when the replay ends.
- **Hover a device** for its controls: *the owner needs it* (drain), *switch off*,
  *crash it*, *switch on*.

The page talks to whichever device answers and moves to another if that one dies.

## What is simulated

`scripts/pool.sh` runs the **lab**: `torrent-lab` starts one real `torrent-peer`
process per simulated device. Each has its own port, memory budget, network link and
(where there is one) use of the GPU, and they talk to each other over real sockets.

| Real | Simulated |
|---|---|
| Every process, port, socket and byte on the wire | *Where* the devices are: they share one machine |
| The model, the split, and every token | The brand names ("MacBook Air") |
| CUDA vs CPU: the "Gaming laptop" really runs on the GPU and the rest on the CPU, so they differ in speed | The network: each device's `link` adds that latency and bandwidth to calls to it |
| Crashes: *crash it* is `SIGKILL`, with no goodbye | The owner: "the owner needs it" is a button, not detected activity |

Moving to real laptops changes nothing in the pool. On each one:

```bash
torrent-peer --pool pool.yaml --name my-laptop --memory-mb 3200 --join <any-device>:9000
```

Give every device the same `pool.yaml` model settings, and `--join` the address of
anyone already in the pool. It learns the rest from them.

## Running a pool without the lab

```bash
# First device
torrent-peer --pool configs/pool-lab.yaml --name alpha --device cuda --memory-mb 3200 \
    --host 192.168.1.10 --bind 0.0.0.0 --port 9000

# Every other device needs one address, not a list
torrent-peer --pool configs/pool-lab.yaml --name beta --device cpu --memory-mb 1600 \
    --host 192.168.1.11 --bind 0.0.0.0 --port 9000 --join 192.168.1.10:9000
```

`--host` is the address other devices reach this one on. `--memory-mb` is what the
owner is willing to lend; it decides how many stages the device may hold
(`memory_mb // shard_mb`).

Ask any device a question:

```bash
curl -X POST http://192.168.1.11:9000/generate -H 'Content-Type: application/json' \
    -d '{"prompt": "What is the capital of France?"}'
```

### Protecting it

Set a shared secret on every device and the pool refuses everything without it:

```bash
torrent-peer ... --token "$(openssl rand -hex 16)"      # or TORRENT_POOL_TOKEN
```

With a token, devices only gossip with each other, only the holder of the token can
change a device's state, and the shard ports refuse calls that do not carry it
(issue #29). Asking questions stays open, like the single-chain API.

This is access control, not encryption. Over an untrusted network, put the pool
inside a VPN such as Tailscale.

## How it works

See [ADR 002](adr/002-leaderless-pool.md) for the reasoning. In short:

- **Membership** (`pool/members.py`, `gossip.py`): every device keeps its own table
  and swaps it with two random others every half second. Each device is the only
  writer of its own row. Liveness is each reader's judgement of whether a heartbeat
  has advanced lately.
- **Placement** (`pool/placement.py`): a pure function of the table. Every device
  runs it and acts only on its own answer. Sticky, makes before breaking, fills the
  thinnest stage first.
- **Routing** (`pool/router.py`): one copy of each stage, chosen by queued work,
  device speed and distance.
- **Resilient running** (`pool/runner.py`): if a device stops answering, switch to
  another copy, re-send the text so far to rebuild its cache, and carry on.
- **The device** (`pool/daemon.py`, `api.py`): the beat, the HTTP API, and the owner's
  controls.

### Taking a device back

`POST /admin/owner {"busy": true}` starts a drain. The device stops taking new
requests. It lets go of each stage only when a replacement is ready, unless none
could come (nobody has room), in which case it lets go at once. After at most
10 seconds it lets go regardless: the owner asked for it back.

### Measured

On the lab, with the real model: killing the GPU device 1.8 s into a 60-token answer
moved the request to two CPU devices in 0.8 s, delivered all 60 tokens, and the pool
then rebuilt two copies of each stage on its own.

`scripts/pool_load.py` asks many questions at once through different devices:

| | questions/s | tokens/s | median answer |
|---|---|---|---|
| one at a time | 2.8 | 18 | 0.29 s |
| six at once | 6.9 | 45 | 0.76 s |

Throughput goes up and each answer gets slower, which is what sharing should do.
When the pool is quiet, everything goes through the fastest device; slower ones take
work as the fast one's queue grows.

## What it does not do

- **Peer-to-peer relay.** Every hop passes through the device that took the question
  (issue #37). Relay would shorten the path; routing and failover would carry over.
- **Verify results.** A member that lies about its hardware or returns wrong tensors
  is not detected.
- **Detect a busy owner by itself.** "The owner needs the device" is a button today. A
  real deployment would watch GPU use or foreground activity and press it.
- **Compress activations.** The low-rank compressor is not in the pool's path.
- **Cross the open internet.** Discovery is "know one address". The DHT (issue #15)
  would replace it.
- **Download weights for you.** Each device must be able to load the model.

## Where things are

| | |
|---|---|
| `src/torrent_llm/pool/` | the pool: `members`, `placement`, `router`, `runner`, `hosting`, `daemon`, `api`, `peer`, `lab` |
| `configs/pool-lab.yaml` | the lab's six pretend devices |
| `scripts/pool.sh` | start the lab and dashboard |
| `scripts/pool_load.py` | many questions at once |
| `dashboard/app/pool/` | the Pool page |
| `tests/test_pool_core.py`, `test_pool.py`, `test_lab.py` | unit tests, whole-pool tests over real sockets |
