# ADR 001 — Build a minimal sharded harness instead of forking Petals

_Status: accepted · 2026-08-23_

## Context

`docs/research.md` left this open: *"Fork Petals vs build a minimal sharded harness
from scratch? (control vs speed)"*. The assumption behind "fork" was that Petals gives
us a mature layer-server abstraction for free and we only write the compression layer.

That assumption does not survive contact with the package metadata:

| Fact | Source |
|---|---|
| Petals' last release is `2.2.0.post1`, uploaded **2023-11-20** | PyPI |
| It pins `transformers>=4.32,<4.35` | Petals package metadata |
| It pins `hivemind==1.1.10.post2` | Petals package metadata |
| Llama-3 architecture support landed in transformers ~4.40 | transformers changelog |

The pins are the problem. **Petals as shipped cannot load Llama-3 at all** — the model
class does not exist in the transformers version it requires. Forking means first
porting Petals forward across ~9 months of transformers API drift, before writing any
of our own code, on a codebase we would then have to instrument invasively anyway.

## Decision

Build a minimal layer-sharded harness in this repository.

Scope of "minimal": load an HF causal LM, slice its decoder layers into contiguous
ranges, serve a range per process, and pass hidden states between processes over gRPC.
This is a few hundred lines because we are not reimplementing Petals — we are
implementing only the part our measurements run through.

## Consequences

**Good**
- Every byte that crosses the wire is ours to instrument. Issue #6 (per-hop profiler)
  is the measurement the entire thesis rests on; it needs hooks Petals does not expose.
- No dependency archaeology before the first line of real work.
- We stay on current transformers, so any recent open model is available to us.

**Bad**
- We give up Petals' fault tolerance and DHT integration, so issues #15 (DHT discovery)
  and #16 (reroute on node drop) become genuine work rather than inherited behaviour.
  Both are already classified as post-MVP.
- "We extended the standard baseline" is no longer available as a framing. We reproduce
  Petals' *architecture* as a baseline condition instead, which is the honest comparison
  for the paper regardless.

**Neutral**
- Hivemind is unaffected by this decision. It is independently maintained (1.1.12,
  released 2026-01-03) and remains the intended DHT backbone for issue #15.
