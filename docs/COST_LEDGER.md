# Cost ledger

One line per job set. GPU prices are Modal list prices at the time of the job (A10G 1.10,
L40S 1.95, A100-80GB 2.50, H100 3.95 USD/h) unless a dashboard figure says otherwise.

Spend from earlier tracks is recorded in their own reports (F01-F03 on `track-f-pixel-path`,
E04, …); this file starts at Track F2 and does not restate numbers it did not measure.

| date | track | job | GPU | hours | USD | note |
| --- | --- | --- | --- | --- | --- | --- |
| 2026-11-21 | F2 | F04/F05 parts 1-2: spline fits, lean-error scoring | none (CPU) | 0 | 0.00 | stored poses only |
| 2026-11-21 | F2 | F06 part 3 **estimate** (posted before the job set): two image builds, MapAnything 10x150 frames, Pi3X 10x150 frames, probes | A100-80GB | ~1.0 | <= 8.00 | ceiling 30 USD |
| 2026-11-21 | F2 | F06 part 3 actual: MapAnything 10 sequences (501 s), Pi3X 10 sequences (378 s), one probe each | A100-80GB | ~0.25 | ~0.70 | from job wall times; image builds are CPU-billed |

Ceilings: part 3 has a US$30 ceiling. Actual spend is filled in after each job set; the
estimate above is posted before the run, not after.
