# Setup & Running

## Install

The virtualenv lives outside the repo so a stray `pip install` never ends up
tracked:

```bash
uv venv ~/Coding/venvs/torrent-llm --python 3.12
source ~/Coding/venvs/torrent-llm/bin/activate.fish   # or activate for bash
uv pip install -e ".[dev]"
```

Then generate the gRPC stubs. They are **not** committed — the `.proto` is the
single source of truth, and a checked-in `_pb2.py` is something people edit by
accident:

```bash
python -m torrent_llm.codegen
```

Verify:

```bash
pytest -q          # 86 tests, no network, no model downloads
ruff check .
```

If an import fails with `No module named 'torrent_llm.transport.activation_pb2'`,
you skipped the codegen step.

## Run a two-shard chain on one machine

Three terminals. The two shard processes each hold half the model's layers:

```bash
torrent-shard --config configs/local-2shard.yaml --index 0
torrent-shard --config configs/local-2shard.yaml --index 1
torrent-run   --config configs/local-2shard.yaml --prompt "The capital of France is"
```

`torrent-run --describe` prints what each node reports hosting without running
inference — the fastest way to confirm the chain is wired correctly.

## Run across two machines (issue #3)

Copy `configs/lan-2machine.yaml`, replace the addresses with the two machines'
LAN IPs, and make sure the port is open. Nothing else changes: the shard
processes and the runner read the same file either way.

Confirm the layer count matches the model before starting — a mismatch is caught
at startup, but it is faster to check first:

```bash
python -c "from torrent_llm.shard import num_layers_of; print(num_layers_of('Qwen/Qwen3-1.7B'))"
```

## Profile the hops (issue #6)

```bash
python scripts/profile_chain.py --model Qwen/Qwen3-0.6B --seq-lens 128,512,1024
```

Prints a per-hop table, states whether the hops were bandwidth-bound or
compute-bound, and projects the payload onto a real link at larger model sizes.
Records are appended to `runs/*.jsonl` as they happen, so a run that dies
halfway still leaves usable data.

Add `--config configs/lan-2machine.yaml` to profile the real rig instead of
loopback. **Loopback numbers are not network numbers** — they establish the
compute side of the budget, and the projection carries it onto a real link.

## Model access

Qwen3 checkpoints are ungated and download without credentials. The Llama-3
repos on the Hub are gated: `meta-llama/Meta-Llama-3-8B` returns a 401 until
access is granted to your HF account. Request access, then `huggingface-cli login`,
before pointing a config at Llama.

## Notes

- `torch.frombuffer` aliases its input, so the raw codec copies into a
  `bytearray` before decoding. Do not "optimise" that copy away.
- gRPC's default 4 MiB message cap is raised to 512 MiB in
  `transport/convert.py`. One prefill activation on an 8B model at 2k context is
  already ~16 MiB.
