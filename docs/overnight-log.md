# Overnight log, 2026-10-26

What actually happened, newest last. Each checkpoint is a git tag.

- **Start.** Branch `overnight/2026-10-26` from `main` at the merge of PR #40.
  16 GB free, 20 cores, network reachable.
- **Checkpoint 1: compression is real (`checkpoint-1`).** Calibrated a low-rank projection
  from real Qwen3-0.6B activations. Held-out results: uncompressed perplexity 40.4 and
  96% of questions right; calibrated 4x gives 42.2 and 88%; 8x with fp16 on the wire is the
  same; 16x and beyond collapses; the old random projection gives 0% at every size.
  Leaving the attention-sink position out of the fit made things worse (my hypothesis was
  wrong). Weights for rank 256 and 128 are committed. See `docs/compression.md`.
- **The real-device feature (asked for at the start of the night), done.** `torrent-peer`
  now finds its own address, lends a share of free memory, finds other devices by UDP
  broadcast, uses fixed ports, serves the dashboard itself (static build, 1.1 MB, no Node
  needed) and has `--check` that says what is wrong and how to fix it. `scripts/join.sh`
  and `join.ps1` set up, check and start. Guide: `docs/real-devices.md`.
  - **Verified for real:** two separate `torrent-peer` processes with default options found
    each other with no address, used their advertised ports, served the page on the LAN
    address, and answered a question asked through the second device.
  - **Not verified:** Windows and macOS (join.ps1 was written, never run), three or more
    machines, a real router, Tailscale.
  - **Bugs caught on the way:** the owner monitor would have counted a Windows device's own
    process as its owner and backed off from itself (fixed, tested); my earlier name-clash
    edit had broken the feed's warm-up guard (fixed, tested).
- **Tests at this point:** 409 passing. Lint clean (Python and dashboard).
- **Checkpoint 3: compression is in the pool's path, verified for real.**
  `configs/pool-compressed.yaml` sets the calibrated rank-256 fp16 codec; every device
  builds it, devices with different pool settings refuse each other with a message, the
  API and dashboard show the bytes saved. On two real devices (one GPU, one CPU): four of
  four answers right, hidden-state traffic exactly 8x smaller (168 KiB to 21 KiB per
  answer), routes crossing between the CPU and GPU device.
  - **A real bug only the real model could show:** RawCodec cast integer token ids to
    fp16, which cannot hold ids above 2048, so the first shard hit a device-side assert.
    The tiny test model (vocab 256) hid it. Fixed at the source with regression tests.
- **Checkpoint 4: the real-device flow is hardened, and the pool defends itself.** 448 tests
  pass (the suite now takes two minutes). Lint clean, Python and dashboard.
  - **Found and fixed by checking the Windows path by eye:** Ctrl+C could not stop a device on
    Windows (a wait with no timeout is uninterruptible there); `join.ps1`'s own probes would
    have been fatal under PowerShell 5.1's stderr handling. Neither could be run here.
  - **Found by the chaos test:** with a seeded storm of crashes, polite leaves, newcomers and
    owners reclaiming GPUs while three clients asked continuously, no request failed and no
    answer was wrong, over several seeds and about 200 answers each. It also exposed that a
    device that had found nobody blamed the pool; it now says it is not connected and points
    at `--join` and `--check`.
  - **New:** overload protection (a full device refuses new questions, never ones in progress,
    and requesters route around it), contribution accounting (compute given and tokens asked,
    shown on the page), the VPN-address warning in `--check`, `scripts/pool_check.py`
    (asks every device the same question and checks they agree), the OS shown on cards when
    a pool mixes systems, the pool token accepted from the page's address.
  - **A real flaw found by thinking about the learned codec:** the requester decoded each
    hidden state and encoded it again for the next shard, applying the codec twice. Harmless
    for a projection, wrong for a learned codec. Hops now forward the compressed reply
    untouched; tests over real gRPC servers fail without the change.
  - **My own numbers were optimistic.** On a 95-question held-out set (up from 30) the
    baseline is 88% and PCA at 4x scores 77%, not the 88% the first 30 easier questions gave.
- **Checkpoint 5: compression measured honestly.**
  - Learned projection (1000 steps, 95-question eval): rank 256 no gain (78% vs 78%), rank 128
    58% vs 52%, rank 64 34% vs 12%. Training helps only where PCA is weakest and does not
    remove the cliff.
  - Throttled link (15 Mbps, 100 ms, two real CPU devices): 7.9x fewer bytes, answer 10%
    faster (2158 ms to 1953 ms). Latency, not bytes, dominates on a hotspot. The default pool
    stays raw. `docs/compression.md` rewritten with these numbers.
