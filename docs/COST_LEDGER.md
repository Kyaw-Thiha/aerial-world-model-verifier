# GPU cost ledger

One line per job set. GPU prices are Modal list prices at the time of the job (A10/A10G 1.10,
L40S 1.95, A100-80GB 2.50, H100 3.95 USD/h) unless a dashboard figure says otherwise; CPU jobs
are recorded at $0.05/core-hour where the track did so. Rows are carried **verbatim** from each
track's branch ledger, which is the source of truth; no spend was added, removed or recomputed
during the round-2 consolidation.

The tables have different column sets because each track kept its own; "estimate" rows were
posted before the job set, "actual" rows are measured. Totals per track are the sum of that
track's USD column as recorded in its own report.

## Track F — pixel-path verdict, stage 0 (2026-11-16)

Spend from the pre-Track-F tracks is recorded in their own reports (for example Track E's
~US$6.2 cumulative in `reports/E04_v3_r2_coherence.md`); this file starts at Track F and does
not restate numbers it did not measure.

| date | track | job | GPU | hours | USD | note |
| --- | --- | --- | --- | --- | --- | --- |
| 2026-11-16 | F | F01 stage 0: five pixel-path statistics on stored DPVO poses | none (CPU) | 0 | 0.00 | reads stored artefacts only; Modal volume `awmv-c-data` read-only, no app started |
| 2026-11-16 | F | F03 stage 2 actual: ViPE on 82 level + 45 injected ANWM clips (plus an 8-clip retry) | A100-80GB | ~1.4 | ~3.50 | 36-40 s per 16-frame clip; one local wrapper timed out at 1 h, the remote function kept writing and the remaining 8 were finished in a second call |
| 2026-11-16 | F | F02 stage 1 actual: ViPE on 10 sequences (2.5 min each) + image builds | A100-80GB | ~0.45 | ~1.15 | from job wall times (1547 s + 172 s); builds are CPU-billed |
| 2026-11-16 | F | F02 stage 1 actual: MASt3R-SLAM on 10 sequences (789 s) | A100-80GB | ~0.22 | ~0.55 | plus one 45 s smoke and two build retries (CPU) |
| 2026-11-16 | F | F02 stage 1 **estimate** (posted before the job set): two image builds, 10 sequences x ViPE, 10 x MASt3R-SLAM, one smoke run each | A100-80GB | ~2.0 | <= 6.00 | 2.50 USD/h; ceiling 40 USD; under the 20 USD pre-approval line |

Ceilings: Track F stage 1 has a US$40 ceiling. Actual spend is filled in after each job
set from the Modal dashboard; the estimate above is posted before the run, not after.

## Track G — verifier statistics (2026-11-16)

| date | track | job | GPU | hours | USD | notes |
| --- | --- | --- | --- | --- | --- | --- |
| 2026-11-16 | G | `awmv-g-tilt-reader` smoke runs (3×: gradient flow, step cost, fixed pass) | A10G | 0.1 | 0.11 | before and after the batch-size change |
| 2026-11-16 | G | TII folds, **run 1** (8 folds, batch 2) | A10G | 2.0 | 2.20 | **wasted**: `NameError` on the volume handle killed every fold at the write step, after training |
| 2026-11-16 | G | TII folds, run 2 (8 folds; train 740-780 s each, then evaluation of 270-1284 frames each) | A10G | 6.7 | 7.37 | the reported TII result |
| 2026-11-16 | G | UZH folds, run 2 (3 folds) | A10G | 0.5 | 0.55 | reported in the UZH block of G03 |
| | | **Track G total** | | **9.3** | **10.23** | pre-registered estimate 5 USD; ceiling 50 USD |

Notes
* The pre-registration estimated 5 USD. The run cost about twice that, for two reasons: the
  LM-in-the-loop step cost is ~4 s per batch of two at 320 px (the smoke run measured 1.25 s at
  batch 2 before the label fix, and the evaluation pass over 1,267-1,284 frames per fold is not
  cheap either), and run 1 had to be repeated after a code bug. Both are recorded rather than
  absorbed, per the spend rule. Nothing approached the 20 USD pre-post line or the 50 USD
  ceiling.
* G01 and G02 are CPU-only and cost nothing.

## Track F2 — pixel-path round 2 (2026-11-21)

Spend from earlier tracks is recorded in their own reports (F01-F03 on `track-f-pixel-path`,
E04, …); this file starts at Track F2 and does not restate numbers it did not measure.

| date | track | job | GPU | hours | USD | note |
| --- | --- | --- | --- | --- | --- | --- |
| 2026-11-21 | F2 | F04/F05 parts 1-2: spline fits, lean-error scoring | none (CPU) | 0 | 0.00 | stored poses only |
| 2026-11-21 | F2 | F06 part 3 **estimate** (posted before the job set): two image builds, MapAnything 10x150 frames, Pi3X 10x150 frames, probes | A100-80GB | ~1.0 | <= 8.00 | ceiling 30 USD |
| 2026-11-21 | F2 | F06 part 3 actual: MapAnything 10 sequences (501 s), Pi3X 10 sequences (378 s), one probe each | A100-80GB | ~0.25 | ~0.70 | from job wall times; image builds are CPU-billed |

Ceilings: part 3 has a US$30 ceiling. Actual spend is filled in after each job set; the
estimate above is posted before the run, not after.

## Track G2 — segment detection, round 2 (2026-11-21)

| date | track | job | GPU | hours | USD | notes |
| --- | --- | --- | --- | --- | --- | --- |
| 2026-11-21 | G2 | none | — | 0 | 0.00 | G04, G05 and the spike pilot are CPU-only and read stored artifacts |

Notes
* Track G2's ceiling is 10 USD (handoff 18). Nothing was spent: no Modal app was started, no
  volume created, and no R2 object written.
* Round 1 of this track (branches `track-g-segments` and `track-g-segments2`) spent 10.23 USD on
  the tilt-reader fine-tune; its ledger entry lives on that branch, not here, because this
  branch is off `main`.
