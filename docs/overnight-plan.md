# Overnight plan, 2026-10-26

Work done unattended, in order, on the branch `overnight/2026-10-26`. Read
`docs/overnight-log.md` for what actually happened; this file is what was intended.

## Rules

- **Gentle on the machine.** Heavy runs (anything loading the real model) go one at
  a time and are stopped before the next starts. Free memory is checked first and a
  heavy step is skipped if under 8 GB is free. Nothing heavy runs alongside a
  browser. Tool calls stay short; long jobs run detached and are polled.
- **Light verification by default.** Tiny-model tests and the mock pool. The real
  model only where it is the point.
- **Nothing is pushed or merged.** Everything stays on this branch.
- **Checkpoint after every phase:** tests pass, lint passes, committed in small
  commits, tagged `checkpoint-N`, and a line added to the log.
- **Fail twice, move on.** If a phase fails twice, reset to the last checkpoint,
  note why in the log, and go to the next phase.
- **Honest results.** Anything measured is recorded as measured, including bad news.

## Phases

| # | Goal | Why it matters | Cost |
|---|---|---|---|
| 1 | **Make the compressor real.** Calibrate a low-rank projection (PCA) from real hidden states, sweep ranks, and measure output quality and bytes. | The central research claim. Today the compressor breaks the output. | one real-model process, about 3 GB |
| 2 | **Put compression in the pool's path.** Configurable codec for pool devices, shown in the dashboard as bytes saved. | Makes compression part of the working system, not a side script. | tiny tests |
| 3 | **Chaos testing.** Randomly kill, add, drain and return devices while requests run; assert every answer is exact. | The "no downtime" claim should survive more than the cases I thought of. | tiny tests |
| 4 | **Protect the pool from many users.** Per-device limits with back-pressure, so a flood cannot melt one device. | "Serving everyone" needs fairness, not just spreading. | tiny tests |
| 5 | **Automatic discovery on a local network.** UDP broadcast so a device needs no address. | Real laptops should just find each other. | tiny tests |
| 6 | **Contribution accounting.** Per-device compute given and used, shown on the page (issue #19). | The incentive half of the vision. | tiny tests + mock |
| 7 | **Dashboard: results and device detail.** A compression results view and per-device statistics. | Gives a mentor numbers to look at. | mock |
| 8 | **Final check and writeup.** Full tests, one short three-device real run, results doc, README. | Leaves it ready to merge. | one short real run |

## Out of scope tonight

Peer-to-peer relay (#37) is the teammate's. Two real machines are not available. A
real multi-agent run needs design decisions that should not be made unattended.
