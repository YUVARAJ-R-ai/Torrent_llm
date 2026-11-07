# Torrent-LLM: a worldwide, secure swarm — design

_Status: approved in conversation 2026-11-07 · branch `feat/worldwide-swarm`_

## Intent

Torrent-LLM should be **BitTorrent for AI inference**: any device, anywhere, joins a swarm
with one command, lends what it can spare, and the swarm runs a model together. Two
moments it must serve soon: an **advisor review** (research results that hold up) and a
**live demo on friends' laptops over a phone hotspot** (it must not break).

The main path does not change: a leaderless pool of devices that each hold stages of a
model, with hidden states compressed on the wire. Everything here is built *under* that
pool (a secure network layer that reaches the whole internet) and *beside* it (trust,
compression v2, a desktop app, research results). Nothing replaces the pool's gossip,
placement, routing or failover logic.

## Goals

1. **Demo-proof.** No stall when a device vanishes (today: 300 s), works on a phone
   hotspot, works when a device cannot reach its own public address, one command on
   Windows, macOS and Linux.
2. **Two ways to join, both easy.**
   - *Invite swarm*: the creator gets a link; anyone with the link joins from anywhere;
     nobody without it can even complete a handshake.
   - *Public swarm*: anyone can join; defended against fake identities and cheaters.
3. **Worldwide.** Devices behind home routers, CGNAT and phone networks reach each other:
   serverless discovery (DHT), reachability detection, UPnP port mapping, and relaying
   through reachable peers when no direct path exists.
4. **Secure.**
   - *Guaranteed*: every link is encrypted and authenticated; peers exchange tensors and
     JSON only, never code or pickles; every input from the network is size-checked and
     validated; floods are rate-limited; a private swarm refuses outsiders.
   - *Made hard*: wrong answers (spot-checks against another copy, reputation, bans);
     sybil identities (proof of work per identity, local reputation).
   - *Stated, not hidden*: in an open swarm a peer computing a middle stage sees hidden
     states. **Privacy mode** keeps the first and last stage on the asking device, so
     peers see neither the prompt tokens nor the answer tokens.
5. **More efficient.** bf16 on the wire by default (2x, free), an int8 codec, compress
   only the prompt pass, choose compression per link from measured bandwidth (#18),
   peer-to-peer relay routing (#37) so hidden states cross each link once.
6. **A desktop app** (Electron, Windows/macOS/Linux) that starts and manages the local
   device, creates and joins swarms, shows peers, stages, traffic, trust, and chats.
7. **Research results** for the advisor: #2, #7, #12, #13, #36, write-up #14.
8. **Test-ready backend**: every new module unit-tested; multi-node integration tests;
   a CI matrix on all three operating systems.

## Non-goals

- #32 (training a new LLM) — weeks of GPU; a separate project.
- Sending #20 to the advisor — the owner sends it; we prepare the material.
- UDP/QUIC hole punching. Relay covers what direct and UPnP cannot; recorded as future work.
- A global token/currency. Incentives are local (tit-for-tat), like BitTorrent.

## Anti-goals (failure even if it "works")

- A central server that the swarm cannot run without.
- Anything a peer sends being executed, unpickled or used as a path.
- A demo that needs Tailscale, port forwarding or editing files.
- Claims in docs or UI that the code does not back (e.g. "immune to all attacks").

## Architecture

```
desktop/ (Electron)  ──HTTP+SSE, admin token──▶  torrent-peer (local control API, 127.0.0.1)
                                                   │
                         pool/ (unchanged ideas): membership, placement, routing, failover,
                               + trust (spot-check, reputation, bans, ledger), privacy mode,
                               + codec policy (adaptive), + relay routing
                                                   │
                         net/ (new overlay): identity · secure channel · RPC · node
                               · relay circuits · autonat · upnp · DHT · LAN discovery · invite
                                                   │
                                        TCP (direct, mapped, or via relay)
```

### `net/` — the overlay

| Module | Role |
|---|---|
| `identity.py` | Ed25519 identity; `PeerId` = base32 of sha256(pubkey)[:20]; proof of work; key file in the per-OS data dir |
| `channel.py` | Authenticated key exchange (X25519 + Ed25519 signatures over the transcript, optional swarm PSK) → ChaCha20-Poly1305 framed stream. **Claude-written** |
| `stream.py` | The byte-stream abstraction the channel runs on: TCP, or a relay circuit |
| `rpc.py` | Multiplexed request/response over a secure stream: ids, deadlines, cancel, errors, keepalive pings, liveness, inbound concurrency cap |
| `node.py` | Owns identity, listeners, connections (one per peer), address book, dialing (direct → relay), handler registry, background asyncio loop with a sync facade, self-call short-circuit |
| `records.py` | Signed `PeerRecord` (peer id, pubkey, addresses, seq) and address strings |
| `relay.py` | Circuit relay: reachable peers relay end-to-end-encrypted circuits for unreachable ones, with limits |
| `autonat.py` | Observed-address reporting and dial-back reachability checks |
| `upnp.py` | Best-effort UPnP IGD port mapping (stdlib only) |
| `dht.py` | Kademlia (k=16, α=3) over the overlay: find_node, provider records with TTL |
| `lan.py` | Signed LAN broadcast announcements (replaces `pool/discovery.py` payload) |
| `invite.py` | `tllm://join/<base64url>` invite: swarm id, PSK, bootstrap addresses, pool-config hash |
| `ratelimit.py` | Token buckets per peer and method |
| `paths.py` | Per-OS data/config dirs (no new dependency) |

**Channel.** Wire: `u32 length ‖ bytes`. Handshake (initiator I, responder R):

1. I→R: `"TLLM/1"` ‖ `e_I` (X25519 public, 32 B).
2. R→I: `e_R` ‖ AEAD_k1(`pk_R` ‖ `pow_R` ‖ `sig_R`), where
   `k1 = HKDF(X25519(e_R,e_I) ‖ psk, salt = H(msg1 ‖ e_R), info="tllm k1")` and
   `sig_R = Ed25519(sk_R, "tllm-R" ‖ H(msg1 ‖ e_R))`.
3. I→R: AEAD_k2(`pk_I` ‖ `pow_I` ‖ `sig_I`), `sig_I` over `"tllm-I" ‖ H(msg1 ‖ msg2)`.
4. Session keys `k_I→R, k_R→I = HKDF(shared ‖ psk, salt = H(msg1‖msg2‖msg3))`.
   Per-direction 96-bit counters as nonces; each frame's AAD is its length.

`psk` is 32 zero bytes in the public swarm and the swarm secret in an invite swarm, so a
wrong or missing secret fails at step 2 with nothing learned. Ephemeral keys give forward
secrecy; the initiator may pin the expected `PeerId` (MITM protection when dialing a known
peer). In the public swarm each side checks the other's proof of work.

**RPC.** Frame plaintext: `u8 type ‖ u32 id ‖ u32 hdr_len ‖ hdr(JSON) ‖ body`.
Types: REQ (hdr has method `m`, deadline `d` ms), RES, ERR (hdr `code`, `msg`), CANCEL,
PING, PONG. Error codes: `unavailable deadline busy not_found bad_request denied internal
cache_lost cancelled too_large`. Each side pings when idle for 2 s and declares the
connection dead after 6 s without receiving **any bytes** (progress inside a large frame
counts). A dead connection fails every pending call at once with `unavailable` — this is
the fix for the 300 s stall.

**Node.** Sync facade: `call(peer_id, method, header, body, timeout)`; `register(method,
handler)`; `connect(addr)`; `add_record(record)`; `peers()`; `close()`. A call to its own
peer id runs the handler in-process (fixes the hairpin bug).

**Reachability, in order:** LAN address (hotspot/Wi-Fi) → public address confirmed by
dial-back → UPnP-mapped address → relay address `relay:<relay-peer-id>`. Dial tries direct
addresses concurrently, then relays.

**Discovery:** invite bootstrap addresses, LAN broadcast, the DHT (swarm topic =
sha256("tllm-swarm" ‖ swarm_id), provider records refreshed every 10 min, TTL 30 min),
and the gossiped member table. The public swarm needs at least one bootstrap address; any
reachable peer can be one (`torrent-peer bootstrap` runs a lightweight relay+DHT node).

### `pool/` changes

- **Identity = `Member.name` holds the peer id**, `label` is the human name. Every place
  that keys on `name` keeps working. Rows carry `pubkey`, `addrs`, and `sig`; a merge
  drops rows whose signature, or pubkey→peer-id, does not check.
- **Gossip over the overlay**, digest-based (`{peer_id: stamp}` then only newer rows), so
  it scales past a few hundred devices; stale `down` rows expire.
- **Stages are served as RPC `shard.forward`** (protobuf `ForwardRequest` bytes as the body)
  instead of a gRPC port per stage. `ShardClient` gains a pluggable sender; the gRPC sender
  stays for the single-chain research harness (`torrent-shard`, `torrent-run`).
- **Deadlines per hop**: `5 s + bytes / 1 Mbit/s + 50 ms × tokens`, plus connection liveness.
- **Relay routing (#37)**: `shard.relay` carries the route; each stage forwards to the next
  as a nested call; the last stage picks the token, which returns down the chain. Errors
  name the failing peer. `routing: requester | relay` in the pool file.
- **Trust**: spot-check a random pass (rate configurable; default 5% public, 0% invite) by
  recomputing it statelessly on another copy (or locally); tie-break with a third; a peer
  caught twice is banned locally (persisted). Reputation is local. Routing prefers peers
  with a clean record.
- **Incentives (#19)**: each device keeps a local ledger of compute given to and taken from
  each peer; when at capacity it admits the peers with the best give/take ratio first.
- **Privacy mode**: the asking device holds stage 0 and the last stage itself (needs a
  pool file with 3+ stages); routes pin those stages to self.

### Compression v2

- **Codec profiles** in the pool file (`codecs: {id: {name, args}}`); the message header
  carries the profile id; receivers decode by profile and refuse unknown ids.
- Profiles: `raw32`, `bf16` (raw, bfloat16 wire), `int8` (new: per-token absmax scale,
  top-k outlier channels kept in fp16), `pca256`, `learned128`.
- **Policy**: `fixed: <id>` or `adaptive: [ladder best→smallest]` with `prefill_only`.
  Adaptive picks, per hop, the first profile whose estimated transfer time fits the budget,
  from a per-peer bandwidth estimate. Replies use the request's profile.
- Default pool: `bf16`, adaptive ladder `[bf16, int8, pca256]`, prefill-only.

### Local control API

`torrent-peer` serves HTTP on 127.0.0.1 with an admin token (file in the data dir;
the desktop app reads it). `--ui-lan` additionally serves the read-only web dashboard on
the LAN. New endpoints: `/node`, `/swarm/invite`, `/peers`, `/trust`, `/trust/bans`,
`/settings`, `/generate/stream` (SSE tokens). Existing `/pool`, `/events`, `/generate`
remain.

### Desktop app (`desktop/`)

Electron + React + Vite + TypeScript, styled like the existing dashboard. The main process
installs/locates the backend (uv-managed environment in the app data dir), starts and
stops `torrent-peer`, holds the admin token, and proxies API calls; the renderer never
sees the token. Screens: Home, Swarm (create invite + QR, join, public), Peers, Model,
Chat (streaming), Activity, Trust, Settings, Diagnostics. Packaged with electron-builder
for Windows (NSIS), macOS (dmg), Linux (AppImage).

### One command

```bash
curl -LsSf https://raw.githubusercontent.com/YUVARAJ-R-ai/Torrent_llm/main/scripts/install.sh | sh -s -- 'tllm://join/…'
```
```powershell
powershell -ExecutionPolicy Bypass -c "& ([scriptblock]::Create((irm https://raw.githubusercontent.com/YUVARAJ-R-ai/Torrent_llm/main/scripts/install.ps1))) 'tllm://join/…'"
```
Each installs `uv` if missing, installs the package with Python 3.12, then runs
`torrent-peer join <invite>`. `torrent-peer create` makes an invite swarm; `torrent-peer
public` joins the public one.

### Extra ideas adopted (from brainstorming)

- **Peers share the model files** (the torrent part, literally): content-addressed chunks
  verified against sha256 hashes in the pool config, so a hotspot downloads the model once.
- **Token streaming** to the app and dashboard.
- **QR code** for invites in the app.

Deferred: speculative decoding to cut round trips; UDP hole punching; KV reuse across
failover.

## Decisions made by Claude (not by the owner)

- Own overlay in pure Python over TCP instead of Hivemind/libp2p: Hivemind needs a Go
  daemon and does not support Windows natively, which conflicts with platform independence.
- One new backend dependency: `cryptography` (wheels for all platforms).
- Relay instead of hole punching for unreachable peers.
- Electron (Node is present; Tauri would need a Rust toolchain on every builder).
- `Member.name` becomes the peer id to keep every existing keyed code path intact.
- Local-only reputation and incentives (no shared accusations, which invite framing).

## Verification

- `pytest -q` (unit + multi-node integration on localhost, including a relay-only node, a
  silent node, a cheating node, a wrong-PSK node, a 20-node DHT).
- `ruff check`, desktop `npm run typecheck && npm test && npm run build`.
- A real run: several processes on this machine with simulated NAT, then a multi-machine
  run if available.
