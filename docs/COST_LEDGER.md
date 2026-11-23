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

## Track I — I01, the simulator spike (2026-11-16)

Track ceiling: **USD 40** (handoff 15); ask before exceeding.

| job | GPU | hours | USD | notes |
| --- | --- | --- | --- | --- |
| `awmv-i-flightmare` probe, 1st attempt | CPU image build | 0.02 | 0.00 | `libzmqpp-dev` absent on 22.04 |
| `awmv-i-flightmare` probe, 2nd attempt | CPU container | 0.4 | 0.04 | zmqpp + flightlib build; stopped during the Unity download |
| `awmv-i-isaac` probe, image build + start | L40S | 0.05 | 0.10 | app reached "Startup Complete" in 24 s |
| `awmv-i-isaac` probe, API iterations | L40S | 0.6 | 1.17 | three driver-side failures, one run that did not return |
| `awmv-i-blender` throughput and EEVEE clip (7 runs) | L40S | 0.6 | 1.17 | 1 GB image; 91-frame EEVEE clip at 0.145 s/frame |
| `awmv-i-geocalib` spike scoring, 3 clips | A10 | 0.13 | 0.14 | 91 frames per clip |
| **subtotal** | | | **2.62** | |

## Track I — I02, the labelled set (2026-11-21)

| job | GPU | hours | USD | notes |
| --- | --- | --- | --- | --- |
| `awmv-i-blender` scene-family previews (3 runs) | L40S | 0.1 | 0.20 | five families, one static frame each |
| `awmv-i-dataset` smoke tests (4 failed calls) + 2 clips | L40S | 0.2 | 0.39 | image/tooling debugging |
| `awmv-i-dataset` first 12-clip batch | L40S | 0.4 | 0.78 | 24 min wall; 0.14-0.36 s/frame |
| `awmv-i-dataset` family x mount grid (11 clips, one cancelled call + resume) | L40S | 0.35 | 0.68 | forest clips 2.3x the cost per frame |
| `awmv-i-score` first dev pass, 6 clips | A10 | 0.25 | 0.28 | 3,058 frames at 2.5-7.7 fps |
| `awmv-i-score` grid pass, 11 clips at stride 2 | A10 | 0.2 | 0.22 | 2,700 frames in 468 s |
| **subtotal** | | | **2.55** | |

## Track I — I02, the full-set render (2026-11-21, in progress)

| job | GPU | hours | USD | notes |
| --- | --- | --- | --- | --- |
| `awmv-i-dataset` render batch 1 (150 clips, detached) | L40S | 4.2 | 8.19 | 0.15-0.36 s/frame, ~2 min per clip with encode and truth |
| `awmv-i-dataset` render batch 2 (145 clips, detached, in progress) | L40S | ~4.0 | ~7.8 | |
| `awmv-i-score` dev feasible grid, 80 clips at stride 2 | A10 | 1.1 | 1.21 | 19,081 frames |
| **subtotal so far** | | | **~7.2** booked, ~7.8 committed | |

## Booked and committed

| item | GPU | hours | USD |
| --- | --- | --- | --- |
| booked so far | | | **12.4** |
| render batch 2 (in progress) | L40S | ~4 | ~7.8 |
| remaining clips after batch 2 (about 0) | L40S | ~0 | ~0 |
| dev-pass scoring of the defect clips, stride 2 | A10 | ~4.5 | ~5.0 |
| **planned total** | | | **~25.2** |

Remaining headroom against the ceiling: about USD 15, which is the allowance for reruns.

The per-frame cost is what sets the plan and it is scene-dependent: open scenes render at
0.14-0.21 s/frame and the dense forest at 0.35 s/frame on the same L40S. The 149,450-frame
total at a 0.23 s/frame average is the ~10 h above. Actual per-clip wall time averages 2 min
(encode, truth write and volume commit included), which is what the estimates use.

## Track H3/H4 — reader validity and SolarWM continuous campaign (2026-11-22)

| date | job | GPU | hours | USD | status | notes |
| --- | --- | --- | --- | --- | --- | --- |
| 2026-11-22 | H03 reader image build + rotation sign probe (12 reads, GeoCalib + Perspective Fields) | A10 | 0.09 | 0.10 | actual | image build dominates; 28 s of inference |
| 2026-11-22 | H03 readers on the 150-frame sample (GeoCalib pinhole + native fisheye, Perspective Fields) | A10 | 0.10 | 0.11 | actual | 75 s of inference, 150 frames |
| 2026-11-22 | H03 Perspective Fields recompute on full clips (LingBot, SolarWM grid/seeds/probes, ANWM, sample) | A10 | 0.70 | 0.77 | actual | about 9,000 frames, ~20 s per 81-frame clip |
| 2026-11-22 | H04 SolarWM convention probes (2 clips x 81 frames) | A100-80GB | 0.09 | 0.23 | actual | 330 s including model load |
| 2026-11-22 | H04 convention-probe reader | A10 | 0.03 | 0.03 | actual | 162 frames, 59 s |
| 2026-11-22 | H04 campaign, batch 1 (234 clips) | A100-80GB | 1.60 | 4.00 | wasted | **stopped before writing any output**: the paths were exactly rank-deficient for the frozen statistic (amendment 2) |
| 2026-11-22 | H04 campaign, batch 2 (234 clips, amendment-2 paths), several failed attempts | A100-80GB | 2.30 | 5.75 | actual | four 59-clip calls died after ~15 min (sub-batch of 40 too large for the release's infer step; no clip written), plus 3 diagnostic calls (3, 12, 9 clips) that succeeded |
| 2026-11-22 | H04 campaign, batch 3 (234 clips, sub-batch 12) | A100-80GB | 2.25 | 5.63 | actual | four parallel 59-clip calls, ~1930-2440 s each, all 234 clips written |
| 2026-11-22 | H04 reader (234 clips x 81 frames = 18,954 frames) | A10 | 2.10 | 2.31 | actual | four shards, 1613-1928 s each, 18,954 rows |
| 2026-11-22 | H04 yaw control (22 clips warped on CPU, then read) | CPU 8 + A10 | 0.16 + 0.23 | 0.08 + 0.25 | actual | warp 575 s, reader 825 s |
| 2026-11-22 | H04 publish (468 objects to R2 with sha256 read-back) | CPU 4 | 0.4 | 0.02 | actual | `track-h3/clips/solarwm-main/` + manifest |
| 2026-11-22 | H04 yaw-control warp (22 clips, CPU) | CPU 8 | 0.10 | 0.04 | actual | pure numpy remap, per-clip commits |
| 2026-11-22 | H04 yaw-control reader (22 clips) | A10 | 0.23 | 0.25 | actual | 1,782 frames |
| 2026-11-22 | H04 14B download (183.4 GB, 81 files) | CPU 16 | 0.22 | 0.18 | actual | 799 s; volume `awmv-h3-14b-weights` |
| 2026-11-22 | H04 14B config-validation attempts (4 failures) | A100-80GB:2 | ~0.05 | 0.13 | actual | each died in 11-20 s on a config contract, not on the model |
| 2026-11-22 | H04 14B timing test (3 clips, 50 steps) | A100-80GB:2 | 0.67 | 1.67 | actual | loaded in ~14.5 min, then failed in the generation pass (dtype contract); no clip |
| 2026-11-22 | H04 14B single-clip probe, 4 GPUs, sp=2 | A100-80GB:4 | 1.75 | 4.38 | actual | 26.6 min, failed on the release's distributed contract; no clip |

**Track H3/H4 total (actual): about US$30 of the US$60 ceiling** (the 14B attempt adds the
183 GB download, the config-contract failures, the two-rank timing attempt and the probe), of
which about US$8 is
wasted work: the aborted first campaign (rank-deficient paths) and the failed 40-clip
sub-batches.  The pre-campaign estimate posted in the H04 pre-registration was
US$6.0 for the campaign and reader; the actual is higher by the aborted batch and by the
Perspective Fields recompute that the failed pre-registered rule triggered.  The 14B release
was downloaded and driven through its own CLI but produced no clip; LTX-22B and H3-33B were
never downloaded or run.

| date | job | GPU | hours | USD | status | notes |
| --- | --- | --- | --- | --- | --- | --- |
| 2026-11-22 | H03 reader image build + rotation sign probe (12 reads, GeoCalib + Perspective Fields) | A10 | 0.09 | 0.10 | actual | image build dominates; 28 s of inference |
| 2026-11-22 | H03 readers on the 150-frame sample (GeoCalib pinhole + native fisheye, Perspective Fields) | A10 | 0.10 | 0.11 | actual | 75 s of inference, 150 frames |
| 2026-11-22 | H03 Perspective Fields recompute on full clips (LingBot, SolarWM grid/seeds/probes, ANWM, sample) | A10 | 0.70 | 0.77 | actual | about 9,000 frames, ~20 s per 81-frame clip |
| 2026-11-22 | H04 SolarWM convention probes (2 clips x 81 frames) | A100-80GB | 0.09 | 0.23 | actual | 330 s including model load |
| 2026-11-22 | H04 convention-probe reader | A10 | 0.03 | 0.03 | actual | 162 frames, 59 s |
| 2026-11-22 | H04 campaign, batch 1 (234 clips) | A100-80GB | 1.60 | 4.00 | wasted | **stopped before writing any output**: the paths were exactly rank-deficient for the frozen statistic (amendment 2) |
| 2026-11-22 | H04 campaign, batch 2 (234 clips, amendment-2 paths), several failed attempts | A100-80GB | 2.30 | 5.75 | actual | four 59-clip calls died after ~15 min (sub-batch of 40 too large for the release's infer step; no clip written), plus 3 diagnostic calls (3, 12, 9 clips) that succeeded |
| 2026-11-22 | H04 campaign, batch 3 (234 clips, sub-batch 12) | A100-80GB | 2.25 | 5.63 | actual | four parallel 59-clip calls, ~1930-2440 s each, all 234 clips written |
| 2026-11-22 | H04 reader (234 clips x 81 frames = 18,954 frames) | A10 | 2.10 | 2.31 | actual | four shards, 1613-1928 s each, 18,954 rows |
| 2026-11-22 | H04 yaw control (22 clips warped on CPU, then read) | CPU 8 + A10 | 0.16 + 0.23 | 0.08 + 0.25 | actual | warp 575 s, reader 825 s |
| 2026-11-22 | H04 publish (468 objects to R2 with sha256 read-back) | CPU 4 | 0.4 | 0.02 | actual | `track-h3/clips/solarwm-main/` + manifest |
| 2026-11-22 | H04 yaw-control warp (22 clips, CPU) | CPU 8 | 0.10 | 0.04 | actual | pure numpy remap, per-clip commits |
| 2026-11-22 | H04 yaw-control reader (22 clips) | A10 | 0.23 | 0.25 | actual | 1,782 frames |
| 2026-11-22 | H04 14B download (183.4 GB, 81 files) | CPU 16 | 0.22 | 0.18 | actual | 799 s; volume `awmv-h3-14b-weights` |
| 2026-11-22 | H04 14B config-validation attempts (4 failures) | A100-80GB:2 | ~0.05 | 0.13 | actual | each died in 11-20 s on a config contract, not on the model |
| 2026-11-22 | H04 14B timing test (3 clips, 50 steps) | A100-80GB:2 | 0.67 | 1.67 | actual | loaded in ~14.5 min, then failed in the generation pass (dtype contract); no clip |
| 2026-11-22 | H04 14B single-clip probe, 4 GPUs, sp=2 | A100-80GB:4 | 1.75 | 4.38 | actual | 26.6 min, failed on the release's distributed contract; no clip |

**Track H3/H4 total (actual): about US$30 of the US$60 ceiling** (the 14B attempt adds the
183 GB download, the config-contract failures, the two-rank timing attempt and the probe), of
which about US$8 is
wasted work: the aborted first campaign (rank-deficient paths) and the failed 40-clip
sub-batches.  The pre-campaign estimate posted in the H04 pre-registration was
US$6.0 for the campaign and reader; the actual is higher by the aborted batch and by the
Perspective Fields recompute that the failed pre-registered rule triggered.  The 14B release
was downloaded and driven through its own CLI but produced no clip; LTX-22B and H3-33B were
never downloaded or run.

## Track L — L01, scene-stratified reader error

Earlier tracks keep their ledgers on their own branches; this file starts at Track L because
`main` (the base of this branch) did not carry one.

| date | job | GPU | hours | USD | notes |
| --- | --- | --- | --- | --- | --- |
| 2026-11-22 | `awmv-l-reader` feasible-dev re-score with gate and confidence fields (80 clips, stride 1) | A10 | 3.7 (actual, 4 shards x ~55 min) | 4.1 | 37,401 frames; two attempts, the first stopped when per-clip volume commits serialised; the resumed pass skips stored clips |

Estimated before running: **USD 3.0** (2.7 GPU-h); the actual wall clock is longer because each
clip decodes and commits separately, and the ledger is updated with the billed hours when the
pass finishes. Below the USD 20 pre-post line. The estimate
first written here (1.2 h at stride 2) was revised to stride 1 before the run, on the grounds
that the anatomy split wants the full 30 Hz; recorded rather than overwritten.

## Track L — L02, reader fine-tune

| date | job | GPU | hours | USD | notes |
| --- | --- | --- | --- | --- | --- |
| 2026-11-22 | smoke: 32 frames, batch 1, mixed route | A10 | 0.01 | 0.01 | 0.67 s/frame measured |
| 2026-11-22 | first finals attempt (mixed arm aborted on a frame-shape bug) | A10G | 1.3 | 1.4 | real and sim finals completed; mixed rerun after the fix |
| 2026-11-22 | first sim fold attempt (stopped: volume-read JPEGs dominated) | A10G | 1.0 | 1.1 | measured about 2 s/frame once the LM was in the loop on forest frames; no fold record |
| 2026-11-22 | arms: 18 folds (a 6, b 3, c 9) | A10G | 6.0 (actual, ~20 min each) | 6.6 | 600 frames x 2 epochs, plus held-out eval; two mixed folds rerun |
| 2026-11-22 | 3 final models + held-out evaluation | A10G | 1.1 (actual) | 1.2 | checkpoints written for the edit pass |
| 2026-11-22 | failed attempts: frame-shape stack, detached-LM abort | A10G | 1.5 (actual) | 1.7 | recorded as bugs, fixed, rerun |
| 2026-11-22 | downstream edit pass: 4 readers x 8 flights x ~5 clips x 5 conditions | A10G | 14.4 (actual, 4 shards x ~0.9 h per reader) | 15.8 | 24.5k frames per reader at 4-5 fps; frozen, real, sim, mixed |

## Running total

| item | USD |
| --- | --- |
| L01 scoring (all 80 clips, actual) | 4.10 |
| L02 training, finals, failed attempts and edit passes (actual, above) | 25.30 |
| **booked total** | **29.40** |
| remaining against the 50.00 ceiling | 20.60 |

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
