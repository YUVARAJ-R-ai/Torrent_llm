# Chain dashboard

Testing and visualization for the layer-sharded inference chain (issue #27).

**This is deliberately a throwaway tool.** A web app is very likely not the right
final interface for this project, and this directory is not a commitment to one.
It exists so the chain's behaviour is legible while the parts that matter — the
compressor (#8, #9) and the headline benchmark (#12) — get built.

## Running it

Three processes. The dashboard talks only to the HTTP API; it has no torch, no
model, and no gRPC dependency of its own.

```bash
# 1. the shard nodes
torrent-shard --config configs/local-2shard.yaml --index 0
torrent-shard --config configs/local-2shard.yaml --index 1

# 2. the HTTP layer, with CORS opened for the dashboard's origin
torrent-api --config configs/local-2shard.yaml --cors-origin http://localhost:3000

# 3. this
cd dashboard && npm install && npm run dev
```

Then open <http://localhost:3000>.

Point it at a different API with `NEXT_PUBLIC_API_BASE`:

```bash
NEXT_PUBLIC_API_BASE=http://192.168.1.10:8000 npm run dev
```

## What it shows

- **Topology** — which node hosts which layers, asked of the nodes themselves
  rather than read off the config, so a node disagreeing with the config is
  visible rather than papered over.
- **KV cache on vs off** — the same prompt both ways in one request, with the
  bandwidth ratio and a check that both paths produced identical text. That check
  is not decoration: the two differ only in how much they recompute, so a
  mismatch is a real bug.
- **Bytes on the wire, by hop** — faceted into prefill and decode, each to its
  own scale.
- **Where each hop's time went** — compute vs transport, stacked. The transport
  share is what decides whether compressing a hop could shorten it at all.
- **Every hop** — the full table, which is also the accessible view of the charts.

## Two conventions worth keeping if this gets extended

**Prefill and decode never share an axis.** Their payloads differ by roughly
three orders of magnitude, so plotting them together flattens one to nothing —
and averaging the two regimes is precisely the mistake `docs/bandwidth-regimes.md`
exists to prevent. Facet, or state the ratio as a number.

**Series colors come from the validated palette in `app/globals.css`**, assigned
by role and never cycled. The two slots in use were checked in both light and
dark mode for colorblind separation and contrast against their surfaces. A third
series takes slot 3 (aqua `#1baf7a`), not an invented hue.

## Verifying changes

`npm run build` typechecks but proves nothing about rendering. Real bugs here —
SVG text distortion, layout overflow — were only visible in a browser. Check at
mobile, tablet, laptop and wide widths, in both color schemes, and watch the
console.
