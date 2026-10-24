# One Model Across Several Laptops

How to host a single LLM split across separate computers, on any mix of Windows,
Linux and macOS, and what is actually happening while it runs. This covers the
two-machine rig of issue #3. Discovery stays static (a shared topology file)
until #15.

---

## What happens to a prompt

Qwen3-0.6B has 28 decoder layers and a hidden size of 1024. The topology file
cuts it in two:

| | Laptop A (head) | Laptop B (worker) |
|---|---|---|
| Holds | embedding + layers 0–13 | layers 14–27 + final norm + LM head |
| Runs | shard 0, the HTTP API, the dashboard | shard 1 |

One generated token takes one pass through the chain:

```
browser ──HTTP──▶ torrent-api                                  (laptop A)
                    │ tokenizer: "What is the capital…" -> token ids [1×N] int64
                    ▼ gRPC hop 0
                  shard 0: embed -> layers 0–13 -> hidden state [1×N×1024] float32
                    │
                    ▼ gRPC hop 1  ═══════ crosses the network ═══════▶  (laptop B)
                  shard 1: layers 14–27 -> norm -> LM head -> logits [1×1×151936]
                    │
                    ◀══════════════ scores come back ══════════════
                  torrent-api picks the highest-scoring token, appends it, repeats
```

**Prefill vs decode.** The first step sends the whole prompt: a hidden state of
`N × 1024 × 4` bytes (35 prompt tokens is 140 KiB). Every step after that sends
**one** position, 4 KiB, because each shard keeps a KV cache for its own layers
(#24). The prompt is processed once, and each new token costs a 4 KiB hop.

**The largest payload is the answer, not the activation.** The last shard
returns scores over the whole vocabulary: 151936 × 4 bytes = 593.5 KiB per
token, about 148× the hidden state. See the README's findings.
Compressing the hidden state (#8) is the research contribution; trimming the
logits reply is a separate, easier fix.

**Who talks to whom.** The API process is the coordinator: it calls each shard
in turn and carries the hidden state between them (hub and spoke, not A → B
directly). With the coordinator on laptop A, hop 0 stays inside laptop A and
only hop 1 crosses the network, which is the cut the demo shows.

**Why OS does not matter.** The wire format is protobuf over gRPC, with the
tensor as raw little-endian bytes and its dtype and shape in a header
(`src/torrent_llm/transport/activation.proto`). Nothing on the wire depends on
the OS, the Python build, or whether the other end runs on a GPU. A Windows CPU
laptop and a Linux CUDA laptop interoperate. The sharded chain is bit-exact
against the unsplit model on one machine (`tests/test_shard_runtime.py`).
Across different hardware the last bits of floating point can differ, as they
would for the unsplit model on that hardware.

---

## 1. Pick a network

The laptops need to reach each other on TCP port 50051. In order of preference:

1. **Tailscale.** Install it on every laptop and sign in to the same tailnet.
   Each gets a stable `100.x.y.z` address (`tailscale ip -4`) that works on any
   network, including campus Wi-Fi and two different networks. Check with
   `tailscale ping <other-laptop>`.
2. **A phone hotspot.** Both laptops join it; use their hotspot addresses
   (`ipconfig` on Windows, `ip -4 addr` on Linux, `ipconfig getifaddr en0` on macOS).
3. **A router or a LAN cable** you control.

Avoid campus/eduroam Wi-Fi without Tailscale. Those networks usually block
laptops from talking to each other (client isolation), and nothing on the
laptops can fix that.

## 2. Set up each laptop (the night before)

Needs `git` and [uv](https://docs.astral.sh/uv/). The head laptop also needs
Node.js for the dashboard.

```bash
git clone https://github.com/YUVARAJ-R-ai/Torrent_llm && cd Torrent_llm
scripts/setup.sh                                         # Linux / macOS
powershell -ExecutionPolicy Bypass -File scripts\setup.ps1   # Windows
```

This creates the venv, installs the right PyTorch (CUDA when an NVIDIA driver
is present, CPU otherwise), generates the gRPC stubs, **downloads the model**,
and runs the offline test suite. It ends with the device the laptop will use.

On the day, stop the Hub from being contacted at all:

```bash
export HF_HUB_OFFLINE=1          # bash/zsh
$env:HF_HUB_OFFLINE = 1          # PowerShell
```

## 3. Write the topology (once, then copy to both)

Edit `configs/cluster-2laptop.yaml` and put each laptop's address in:

```yaml
nodes:
  - address: 100.101.183.19:50051   # laptop A
    device: auto
  - address: 100.88.12.7:50051      # laptop B
    device: auto
```

**Both laptops must have the identical file.** It decides which layers each
one loads, and `check` refuses a node serving a different split.
`device: auto` means CUDA, then Apple silicon, then CPU, so one file works on
mixed hardware. Give a stronger laptop more layers with `weights: [1.0, 2.0]`.

## 4. Run

Laptop B first. It prints its own addresses, warns if the config's address for
it is not one of them, prints the firewall command for its OS, then serves:

```bash
python scripts/cluster.py worker --config configs/cluster-2laptop.yaml --index 1
```

Then laptop A. It starts shard 0, waits until B answers, starts the API and
dashboard, warms the model up, and prints the chain:

```bash
python scripts/cluster.py head --config configs/cluster-2laptop.yaml --local 0
```

```
  shard 0  100.101.183.19:50051   up     layers [0:14)  on nixos-y (Linux, cpu)
  shard 1  100.88.12.7:50051      up     layers [14:28)  on DESKTOP-4F2 (Windows, cuda:0)

ready: open http://localhost:3000
```

On Windows, `py` works where `python` is not on PATH. The script finds the venv
by itself, so nothing needs activating. Ctrl+C on the head stops everything it
started. Logs are in the system temp folder under `torrent-demo/`.

At any point, from either laptop:

```bash
python scripts/cluster.py check --config configs/cluster-2laptop.yaml
```

### Without the launcher

The launcher only sequences these commands:

```bash
# laptop B
torrent-shard --config configs/cluster-2laptop.yaml --index 1 --bind 0.0.0.0
# laptop A, three terminals
torrent-shard --config configs/cluster-2laptop.yaml --index 0 --bind 0.0.0.0
torrent-api   --config configs/cluster-2laptop.yaml --cors-origin http://localhost:3000
cd dashboard && npm ci && npm run dev
```

## 5. What to show

1. **Both terminals side by side.** Laptop B logs every hop it serves, with the
   sender's address:
   ```
   hop 1 from 100.101.183.19: hidden state [1x35x1024] float32, 140.0 KiB in -> shard1[14:28] in 182.4 ms -> logits 593.5 KiB out
   hop 1 from 100.101.183.19: hidden state [1x1x1024] float32, 4.0 KiB in -> shard1[14:28] in 41.2 ms -> logits 593.5 KiB out
   ```
   The first line is the prefill; every line after is one generated token.
2. **The dashboard.** The badge reads *Real network · 2 machines*, and each
   device card shows its hostname, OS and device. Ask a question and the
   animation replays the measured run: token ids into A, the hidden state across
   to B, the scores back.
3. **Detailed metrics** (`/details`) gives per-hop bytes and the compute and
   transport split for every step.
4. **The KV cache's effect**, in one number:
   ```bash
   curl -s -X POST http://127.0.0.1:8000/compare-cache -H 'Content-Type: application/json' \
     -d '{"prompt":"The capital of France is","max_new_tokens":8}'
   ```

Do not unplug a laptop mid-demo expecting recovery. Rerouting around a dead node
is #16. Today the request fails with an error naming the shard.

---

## Troubleshooting

| Symptom (`check` output) | Cause | Fix |
|---|---|---|
| `nothing listening` | worker not running, or wrong port | start `cluster.py worker` on that laptop |
| `no answer` | wrong IP, different networks, client isolation, or firewall | `tailscale ping`; open TCP 50051 (the worker prints the command) |
| `serving layers [0:14) but this config plans …` | the laptops have different topology files | copy one file to both, restart the worker |
| `serving X, but the config says Y` | different `model_id` | same as above |
| Windows: works on one network, not another | hotspot classed as *Public*, rule only allows *Private* | allow both, or `New-NetFirewallRule … -Profile Any` |
| Shard exits with CUDA out of memory | GPU too small for its share | `--device cpu` on that worker, or give it fewer layers with `weights` |
| Hangs on a campus/proxy laptop | gRPC routing to an HTTP proxy | handled: peers are always dialled directly |

## More than two machines

Add nodes to the list. The layers split evenly, or by `weights`. Run
`cluster.py worker --index N` on each extra machine. Every hop between two
non-head shards crosses the network twice, through the coordinator.

## Security

No authentication or encryption. Anyone who can reach port 50051 can run the
model and read its hidden states (#29). Use it on a tailnet or a network you
control, not on a public one.
