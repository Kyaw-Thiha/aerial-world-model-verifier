# Cost ledger

One line per job set. GPU prices are Modal list prices at the time of the job (A10G 1.10,
L40S 1.95, A100-80GB 2.50, H100 3.95 USD/h) unless a dashboard figure says otherwise.

Spend from the pre-Track-F tracks is recorded in their own reports (for example Track E's
~US$6.2 cumulative in `reports/E04_v3_r2_coherence.md`); this file starts at Track F and does
not restate numbers it did not measure.

| date | track | job | GPU | hours | USD | note |
| --- | --- | --- | --- | --- | --- | --- |
| 2026-11-16 | F | F01 stage 0: five pixel-path statistics on stored DPVO poses | none (CPU) | 0 | 0.00 | reads stored artefacts only; Modal volume `awmv-c-data` read-only, no app started |
| 2026-11-16 | F | F02 stage 1 actual: ViPE on 10 sequences (2.5 min each) + image builds | A100-80GB | ~0.45 | ~1.15 | from job wall times (1547 s + 172 s); builds are CPU-billed |
| 2026-11-16 | F | F02 stage 1 **estimate** (posted before the job set): two image builds, 10 sequences x ViPE, 10 x MASt3R-SLAM, one smoke run each | A100-80GB | ~2.0 | <= 6.00 | 2.50 USD/h; ceiling 40 USD; under the 20 USD pre-approval line |

Ceilings: Track F stage 1 has a US$40 ceiling. Actual spend is filled in after each job
set from the Modal dashboard; the estimate above is posted before the run, not after.
