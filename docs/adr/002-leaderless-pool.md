# ADR 002 — A leaderless pool: gossip for membership, a pure function for placement

_Status: accepted · 2026-10-25_

## Context

The goal moved from "run one model across two machines" to **a pool**: devices with
different hardware join, lend what their owners are willing to lend, and together
serve everyone's questions, with no downtime when a device leaves and with idle
devices put to use.

That needs three things the two-machine rig did not have: a way for devices to
find each other, a way to decide which device holds which part of the model, and a
way for a request to survive a device vanishing under it.

The obvious design is a coordinator: one machine knows everyone, assigns layers and
routes requests. It is also the design that fails in exactly the way the pool is
meant not to. If the coordinator's laptop closes its lid, everyone is offline.

## Decision

**No coordinator. Every device runs the same program** (`torrent-peer`).

1. **Membership by gossip.** Each device keeps its own table of the pool and swaps it
   with two random others every half second. A device is the only writer of its own
   row and stamps each change with a heartbeat that only increases, so merging is
   "keep the higher stamp", which is order-independent. A restart begins a new
   incarnation, so its counter starting over is not mistaken for old news.
2. **Liveness is each reader's own judgement**, from whether a heartbeat has advanced
   lately (alive, then suspected, then down). It is not gossiped, because a device
   that is quiet for one reader may be fine for another. A request that fails against
   a device marks it down immediately without waiting for the timer.
3. **Placement is a pure function of the table** (`pool/placement.py`). Every device
   runs it over its own view and acts only on its own answer ("should *I* hold stage
   1?"). It is sticky (nobody drops a stage because someone joined), makes before
   breaking (a draining device stops counting as cover, so a replacement comes up
   first), and always fills the thinnest stage next. Views that agree give answers
   that agree; views that briefly disagree can at worst load one copy too many.
4. **Requests choose one copy of each stage** (`pool/router.py`) by queued work scaled
   by device speed plus distance, ties broken at random. When a device is lost
   mid-answer the runner re-sends the text so far as a prompt to the new route, which
   rebuilds the attention cache there, and carries on (`pool/runner.py`).
5. **A shared token guards the pool** (issue #29): the HTTP control plane for
   anything that changes the pool, and the gRPC shard ports for every call.

The lab (`torrent-lab`) is deliberately outside the pool. It only starts and kills
processes to stand in for laptops, so a real deployment is the same code minus it.

## Consequences

**Good**
- Any device can take a question and route it. The dashboard talks to whichever
  answers and moves to another when one dies, so losing even the device you are
  talking to does not blank the screen.
- No election, no consensus protocol, no special case for "the leader died".
- The whole decision logic (membership, placement, routing) is pure and unit tested
  without a network. Integration tests run real devices over real sockets, including
  killing one mid-answer and checking the tokens are still exactly the monolithic
  model's.

**Costs we accept**
- **Eventual, not instant, agreement.** For a second or two after a change, devices
  may disagree about who holds what. The worst outcome is one extra copy loaded.
- **Recovery costs a pause.** The new route has none of the request's cache, so the
  text so far is processed again. That is one extra prefill, not a restart, but it is
  not free on a long answer.
- **The pool is requester-routed.** Every hop passes through the device that took the
  question. Peer-to-peer relay (issue #37) would remove that, and the routing and
  failover here would carry over unchanged.
- **Trusted membership.** The token keeps strangers out; it does not defend against a
  member who lies about its hardware or returns wrong results. Verifying results is a
  research problem of its own and out of scope.
- **Not encrypted.** The token and the activations cross the network in the clear.
  Over an untrusted link the pool belongs inside a VPN such as Tailscale.

## Alternatives considered

- **A coordinator process.** Simplest, and wrong for the stated goal (above).
- **Raft or another consensus protocol for the plan.** Correct, and far more than a
  group of friends' laptops need. The sticky pure function gets the same stability
  from the data it already shares.
- **Hivemind's DHT (issue #15).** It would give discovery across the open internet. For
  a trusted group of up to a few dozen devices, gossip is smaller, has no dependency,
  and is easy to test. It can replace the gossip layer later; nothing above it would
  change.
