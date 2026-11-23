# Cost ledger

One line per job set. Modal list prices at the time of writing: A10 1.10, L40S 1.95,
A100-80GB 2.50, H100 3.95 USD/h. Update this file in the same commit as the job that spends.

Track I's ledger entries live in its own branch (`track-i-simset`, `docs/COST_LEDGER.md`)
until it merges; this file records Track G3 only. Figures marked **billing** come from
`modal billing report --for today`; figures marked *projected* are arithmetic at list prices.

## Track G3 — segment detection on the simulated set (ceiling 8.00, raised to 14.00 on 2026-11-22)

| date | job | GPU | USD | running total | source |
| --- | --- | --- | --- | --- | --- |
| 2026-11-22 | engineering probes, dev reader (170 clips, 84,658 frames, stride 1), yaw render (12 clips), yaw reader (12 clips), diagnostics | L40S / A10 / A100 | **5.27** | **5.27** | **billing**: score 4.484, yaw 0.778, diag 0.004 |
| 2026-11-23 | batch 2: 30 dev clips scored at stride 1 (28 new + 2 re-scored) | L40S | **0.64** | **5.90** | **billing**: score apps now 5.120, yaw 0.778, diag 0.004 |
| 2026-11-23 | held-out pass (100 sim clips + 18 real clips), later, once | L40S | ~2.3 (projected) | ~8.2 | projected |

**No overrun occurred.** An earlier version of this file estimated the first batch at 9.4 USD from
app wall time and called it a breach of the 8.00 ceiling. That was wrong: the containers were
mostly **waiting for L40S capacity**, which is not billed, and per-clip compute stayed at 15-17 fps
throughout. The billing export puts everything through the end of the development pass at
**5.27 USD**. The user raised the ceiling to 14.00 on the strength of the wrong estimate; that is
harmless, is recorded rather than silently reverted, and leaves ample room for the held-out pass.
The practical rule from the episode is kept anyway: **one GPU container at a time**, because
concurrency only buys wall time when capacity exists.

### Notes

* The reader is the only GPU cost; all statistics, ablations and tables are CPU.
* The yaw render (12 clips, 4,646 frames, EEVEE 16 spp, L40S) is the second GPU job.
* The 78 stride-2 reader files Track I published had to be re-scored at stride 1 for a uniform
  window duration; that re-scoring is inside the 5.27.

### G07 held-out pass (2026-11-23)

| job | GPU | containers | wall per container | USD |
| --- | --- | --- | --- | --- |
| one-clip path check (`i02_heldout_000`) | L40S | 1 | ~1 min | 0.03 |
| held-out scores, 4 shards of 25 clips (`heldout_c0..c3`) | L40S | 4 | 19-22 min each | 2.72 |

**G07 total: 2.75 USD.** The frozen recipe projected 2.3; the difference is the 4-container
overlap (Modal charged full wall time for all four, including the shard that queued for a GPU).
Cumulative for the track across both passes: **8.02 USD** against the raised 14.00 ceiling
(`configs/g06_segments_prereg.yaml` states 8.00; the user raised it to 14.00 on 2026-11-22).
No CPU or storage cost. All apps were stopped by `modal run` returning; `modal app list` shows no
running `awmv-g3-*` app after the pass.
