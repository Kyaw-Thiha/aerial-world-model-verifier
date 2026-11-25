# GPU cost ledger

One line per job set. GPU prices are Modal list prices at the time of the job (A10/A10G 1.10,
L40S 1.95, A100-80GB 2.50, H100 3.95 USD/h) unless a dashboard figure says otherwise; CPU jobs
are recorded at $0.05/core-hour where the track did so. Rows are carried **verbatim** from each
track's branch ledger, which is the source of truth; no spend was added, removed or recomputed
during the round-2 consolidation.

The tables have different column sets because each track kept its own; "estimate" rows were
posted before the job set, "actual" rows are measured. No total is recomputed here: each
table's USD column is the record, and its rows say whether they are estimates or actuals.

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

## Track H — generator pilot (2026-11-16)

| date | job | GPU | hours | USD | status | notes |
| --- | --- | --- | --- | --- | --- | --- |
| 2026-11-16 | H01 ANWM seed generation (80 clips, seeds 1-4, 20 designed paths) | A10 | 4.9 (218.8 s/clip measured, 80 clips) | 5.35 | done | `awmv-h-anwm-b` (specs[40:], 8872 s) and `awmv-h-anwm-c` (specs[:40], 8776 s) |
| 2026-11-16 | H01 LingBot download (33 GB NF4 weights) | CPU 16 cores | ~0.2 | ~0.16 | done | `awmv-h-lingbot` download function |
| 2026-11-16 | H01 LingBot probe (3 clips) + model load | A100-80GB | 0.9 (3 x 652 s + 0.33 load) | 2.25 | done | level, +20 ramp, -20 ramp; 21.1 GB peak |
| 2026-11-16 | H01 LingBot strong probes (2 clips: fast20, 60) | A100-80GB | 0.6 (2 x 648 s + 0.22 load) | 1.45 | done | diagnostic only; both show no usable roll response |
| 2026-11-16 | H01 LingBot grid (60 clips at 20 steps) | A100-80GB | 0 | 0 | **not run** | conditional on a roll response; the probe failed, so the spend was avoided |
| 2026-11-16 | H01 Wan 2.2 TI2V-5B: 5 clips | A10 | 0.2 (5 x 135 s + load) | 0.21 | done | 49 frames, 40 steps, CPU offload, 15.2 GB |
| 2026-11-16 | H01 depicted-tilt extraction (ANWM seeds, LingBot, Wan) | A10/A10G | ~0.3 | 0.33 | done | 224 px GeoCalib; ANWM 80 clips, LingBot 5, Wan 5, plus partial passes |

**Track H pilot total (estimate): about US$13 of the US$75 ceiling.** ANWM seed generation
about US$5.4 (80 clips plus the early discarded runs), LingBot-World about US$6.5 (five probe
clips plus four model loads, including the stopped/aborted attempts), Wan 2.2 about US$0.6,
reader extractions about US$0.2. No scale-up job was run, so the conditional 60-clip LingBot
grid (US$14) was not spent.

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

## Track H2 — generator coverage with path control (2026-11-21)

| date | job | GPU | hours | USD | status | notes |
| --- | --- | --- | --- | --- | --- | --- |
| 2026-11-21 | H02 LingBot full-precision weights download (85 GB: fast 74 + T5/VAE) | CPU 16 | ~0.2 | ~0.16 | done | `awmv-h2-lingbot-full`, volume `awmv-h2-weights` |
| 2026-11-21 | H02 LingBot full-precision roll probe (5 clips, 49->45 frames) | A100-80GB | 0.15 (5 x ~60 s + 2 min load) | 0.38 | done | 55.2 GB peak, bf16, 51.7-62.9 s/clip |
| 2026-11-21 | H02 SolarWM 5B weights download (~61 GB) + image | CPU 16 | ~0.4 | ~0.32 | done | `awmv-h2-solarwm`, volume `awmv-h2-weights`; token account accepted 2026-11-21 |
| 2026-11-21 | H02 SolarWM probes (5 clips) + 3 failed setup calls | A100-80GB | 0.3 | 0.75 | done | 264 s for 5 clips including load; failures were publication/decode bugs, not model time |
| 2026-11-21 | H02 SolarWM grid (75 clips, 81 frames, 4 steps) | A100-80GB | 0.58 (2084 s incl. load/decode) | 1.45 | done | all 75 clips, 81 frames each |
| 2026-11-21 | H02 SolarWM seed batch (30 clips) | A100-80GB | 0.22 (782 s incl. load/decode) | 0.54 | done | 10 paths x 3 seeds |
| 2026-11-21 | H02 SolarWM seed reader (30 clips, every 2nd frame) | A100-80GB | 0.18 (649 s) | 0.45 | done | 1,230 frames |
| 2026-11-21 | H02 SolarWM grid reader (75 clips, every 2nd frame) | A100-80GB | 0.29 (1046 s) | 0.73 | done | 3,075 frames, GeoCalib |

**Track H2 pilot total (estimate): about US$5 of the US$90 ceiling.** SolarWM download CPU and
grid/seed generation/readers about US$4.0, LingBot download and full-precision probes about
US$0.6, R2/uploads negligible. No 14B or larger SolarWM route was run.

## Track J — reader error anatomy and bias-robust verdicts (2026-11-21)

| date | track | job | GPU | hours | USD | notes |
| --- | --- | --- | --- | --- | --- | --- |
| 2026-11-21 | J | none (Part 0, J01 + J02) | — | 0 | 0.00 | anatomy and noise budget are CPU-only and read stored reader outputs, mocap and poses |

## Track J

* Ceiling for the track: 30 USD total (handoff 19), with a cost estimate required before any job
  set expected to exceed 20 USD.
* Spent so far: **0.00 USD**. No Modal app was started, no volume created, no R2 object written.
* The only GPU branch the rules could select is 1c (learned reader confidence). Rule 3 fires on
  its literal reading; the estimate and the decision are recorded in `reports/J01` and
  `docs/decisions.md` [J] before any job.

## Other tracks

Round 1 (F/G/H) and round 2 (G2, I) kept their ledgers on their own branches
(`track-g-segments2`, `track-g2-segments`, `track-i-simset`), because those branches are not
merged into `main`. This file starts on `main` with the J entry.

## Track N — attitude from video rotation (2026-11-22)

Ceiling for the track: **30 USD** (handoff 26). Estimate posted before any job; the Part 0 job
set is expected to be about **1.5 USD** (rental), so no 20 USD pre-approval is needed.

| date | track | job | GPU | hours | USD | notes |
| --- | --- | --- | --- | --- | --- | --- |
| 2026-11-22 | N | `awmv-n-rotation` Part 0 (estimate, before the run) | A10 | 0.75 | 1.50 est. | ~53k frames (11 UZH sequences x 450; 80 sim clips x 300-900) at a conservative 20 frames/s, plus container start and a cached image build; A10 at 1.10 USD/h |
| 2026-11-22 | N | `awmv-n-rotation` Part 0 (actual) | A10/A10G | 0.231 | **0.25** | 94 runs, 42,782 frames, 830.4 s of measured DPVO wall time at 1.10 USD/h; one failed test run before the grayscale fix adds ~0.05; image builds are CPU |

Actual Part 0 spend: **about 0.30 USD**, one fifth of the estimate. The Part 1 translation
addendum ran locally on CPU at zero spend.

The Part 1 translation-direction addendum is pre-registered in
`configs/n02_initial_gravity_prereg.yaml` with an estimate of **0.00 USD**: SIFT +
`findEssentialMat` run on local CPU on already-local frames (a Modal CPU container is used only
if local runtime exceeds 3 h; it did not).
| 2026-11-22 | N | `awmv-n-rotation` Part 4 (generated clips) | A10/A10G | 0.12 | **0.13** | 234 H04 clips x 81 frames at 864x480 plus 24 focal-sensitivity runs; ~430 s of measured DPVO wall time at 1.10 USD/h |
| 2026-11-22 | N | `awmv-n-rotation` Part 2 (UZH edited clips) | A10/A10G | 0.03 | **0.03** | 63 clips x 60 frames at 320x240; ~80 s of measured DPVO wall time at 1.10 USD/h |

## Track N, round 2 (2026-11-23)

| date | track | job | GPU | hours | USD | notes |
| --- | --- | --- | --- | --- | --- | --- |
| 2026-11-23 | N | N05/N06 estimate (before the runs) | A10/A10G | 6.0 | 7.0 est. | reader 200 clips at stride 2 (4.9 h) + DPVO 200 clips (0.6 h) + N06 320x180/UZH full-res (0.5 h); A10 1.10 USD/h |
| 2026-11-23 | N | N05 reader, 200 clips stride 2 | A10G x4 | 2.57 | **2.83** | 2,312 s wall over 4 shards; 49,795 estimates; USD 1.10/h |
| 2026-11-23 | N | N05/N06 DPVO (two full draws, half-res, full-res UZH) | A10/A10G | 1.10 | **1.21** | ~330 clip-runs; sum of per-clip wall time |

## Track N2 — verdict tiers on the propagated lean (2026-11-23)

Ceiling for this track: **25 USD** (handoff 28). Track N had spent 4.45 USD before this branch
(0.41 round 1 + 4.04 round 2, above). No new GPU job, volume or R2 write was created: every N2
number is a CPU reanalysis of the stored N05 reader scores, the stored DPVO poses and the stored
H04 reader/path data.

| date | track | job | GPU | hours | USD | notes |
| --- | --- | --- | --- | --- | --- | --- |
| 2026-11-23 | N2 | N07 (F0/sign/C2/coherence, 5 arms × 2 draws × 200 clips) | — | 0 (CPU ~20 s) | **0.00** | local CPU |
| 2026-11-23 | N2 | N08 (two-draw aggregation, 80 clips) | — | 0 (CPU ~2 s) | **0.00** | local CPU |
| 2026-11-23 | N2 | N09 (Part C, 78 H04 clips) | — | 0 (CPU ~2 s) | **0.00** | local CPU |

**N2 spend: 0.00 USD.** Track N total after N2: **4.45 USD** of its 30 USD ceiling; the 25 USD
ceiling of this handoff is untouched (24.94 USD unspent).

## Track N3 — path-anchored start (2026-11-24)

Ceiling for this track: **15 USD** (handoff 29). Nothing was estimated above 5 USD because nothing
needed a GPU: every N3 number is a CPU reanalysis of the stored Track N truth, reader scores and
DPVO poses, the UZH mocap truth and the TII mocap CSVs. No Modal app, volume or R2 write was
created, and no long job was run detached.

| date | track | job | GPU | hours | USD | notes |
| --- | --- | --- | --- | --- | --- | --- |
| 2026-11-24 | N3 | N3A anchor availability (200 clips + 14 UZH + 8 TII) | — | 0 (CPU ~20 s) | **0.00** | local CPU |
| 2026-11-24 | N3 | N3B anchored tiers (10 arms × 2 draws × 200 clips) | — | 0 (CPU ~12 s) | **0.00** | local CPU |
| 2026-11-24 | N3 | N3C drift flag (2 draws, 200 clips) | — | 0 (CPU ~17 s) | **0.00** | local CPU |
| 2026-11-24 | N3 | N3D fallback coverage | — | 0 (CPU ~2 s) | **0.00** | local CPU |

**N3 spend: 0.00 USD.** No billing check was needed (`modal billing report` covers only Modal
spend, and no Modal resource was created or used). Track N total across N, N2 and N3: **4.45 USD**.

## Track P — rate consistency and thrust envelope (2026-11-25)

Ceiling for this track: **3 USD** (handoff 30). Nothing was estimated above 5 USD because nothing
needed a GPU: P0, P1 and P2 are CPU reanalyses of stored truth paths, stored DPVO rotations and
the sim manifest. No Modal app, volume or R2 write was created and no detached job was run.

| date | track | job | GPU | hours | USD | notes |
| --- | --- | --- | --- | --- | --- | --- |
| 2026-11-25 | P | P0 error budget (220 units, 8 noise draws) | — | 0 (CPU ~160 s) | **0.00** | local CPU |
| 2026-11-25 | P | P1 Check B (454 units incl. 234 generated) | — | 0 (CPU ~50 s) | **0.00** | local CPU |
| 2026-11-25 | P | P2 envelope (20 real flights, 222 paths) | — | 0 (CPU ~11 s) | **0.00** | local CPU |

**P spend: 0.00 USD** of the 3 USD ceiling. No billing check was needed.
