# Report 04 — Gate 3: the verdict test (Phase 3a)

Track A. Written 2026-11-09. Main run `results/phase3a_verdict/20261109T090957Z_bc3a398`
(commit `bc3a398`), pre-registration `configs/verdict_preregistration.yaml` at sha256
`b8fd70130f17…` (commits `d6b2003`, amendment `addd9ec`).

## Plain-language summary

The test: take a real drone video, make a doctored copy with the banking removed (and
other copies with it halved, exaggerated, or mirrored), and ask whether the checker scores
the doctored ones as worse than the real one.

**What works.** The checker ranks a real clip *below* its tilt-removed twin 92 % of the
time, and below its mirrored-roll twin 93 % of the time. In plain terms, when the banking
is clearly wrong, the check sees it. On the five flights where the attitude measurement
behaves, a 90 %-acceptance operating point rejects **89 % of full tilt removals and 96 %
of mirrored rolls**.

**What does not work.** It cannot certify the 95 % acceptance target the plan asks for, and
it cannot reliably catch a *half* tilt change (only 45 % at that operating point). The
reason is not the idea; it is a handful of real clips. **Five of 83 real clips score
49-80°** — the attitude measurement is simply wrong on them, four of them on one flight.
Any 95th-percentile threshold therefore has to sit above 44°, where only the biggest
violations live. With those five clips out of the threshold fit, the numbers become the
"what works" numbers above.

**Also measured, because the plan asked:** the attitude estimator reads *exactly level*
ANWM frames as level (median 0.57°, nothing above 10°), so a level-camera fake is far
outside what the measurement can invent; and the abstention rate (clips too boring to
test) is 17 % overall — 57 % on the autonomous flights, 0 % on the piloted ones.

## The question and the setup

**Question.** Does the verifier separate a real banking flight from the same frames with
the tilt removed, scaled or mirrored, while still accepting real flights?

**Mode.** *Commanded-path mode*: the path is known (mocap for TII, commands for ANWM) and
only the depicted attitude comes from pixels, because report 03 shows the path cannot be
recovered. Heading comes from the path because gravity cannot observe it; this is a stated
restriction, not a fitted nuisance.

**Protocol, fixed before any doctored clip was scored.** 4 s non-overlapping clips, every
10th frame (~12 Hz), 100 clips, split by flight. Implied axis from mocap position
(Savitzky-Golay 0.37 s, `k_d` = 0.295 fixed). Depicted axis from GeoCalib on the raw
fisheye; body attitude reconstructed by inverting the ZYX decomposition (gravity gives
roll and pitch, the path gives yaw). Residual = angle between depicted and implied thrust
axes; score = median over testable frames. Two bounded nuisances per clip: mount uptilt
±7°, lag ±150 ms. Abstention: < 12 frames with implied tilt ≥ 10°, testable fraction
< 0.20, or < 40 % usable depicted frames. Threshold: conformal 95th percentile of held-out
real scores, per speed bin, fitted on the fit flights only. Doctored families: tilt × 0,
× 0.5, × 1.5, mirrored roll, plus two pixel controls (identity, yaw rotation) and two
analysis-only negatives (delay, retime).

Doctoring is a pure camera rotation about the camera centre in the raw fisheye
(equidistant `r = f·θ`, `f` from the declared 175° field), because the released
calibration cannot express the periphery a 66° tilt removal needs.

## Results

### The estimator's false-alarm floor on ANWM's level frames (Task 1)

186 released ANWM frames from 12 test trajectories; roll = pitch = 0 exactly (verified
independently over 33 trajectories / 11,949 frames). Run `remote/a_geocalib_p3.py --job
floor`, output `results/anwm_scoring/anwm_floor_v1/floor_{512,224}.jsonl`, summary
`results/anwm_scoring/<run>/noise_floor.json`:

| size | median | p90 | p99 | max | > 5° | > 10° |
| --- | --- | --- | --- | --- | --- | --- |
| 512×512 (f = 256) | **0.57°** | 1.62° | 4.20° | 4.49° | 0.5 % | 0.0 % |
| 224×224 (f = 112) | **0.57°** | 1.52° | 4.20° | 4.49° | 0.0 % | 0.0 % |

### Abstention (fixed before any negative was scored)

| group | clips abstained |
| --- | --- |
| overall | **17.0 %** (17 of 100) |
| autonomous | **56.7 %** |
| piloted | 0 % |
| trackRATM | 20.7 % |
| 01a / 04a / 07a / 13a | 60 % / 57 % / 57 % / 55 % |
| 01p / 07p / 13p / 16p | 0 % / 0 % / 0 % / 0 % |

The autonomous ellipse and lemniscate flights spend most of their time near level, which is
the same fact Phase 0 measured (31-40 % of frames carry a ≥ 10° implied tilt).

### Thresholds

One speed group survived merging (the pre-registered rule: a bin with fewer than 25 fit
clips merges into the neighbour below). Fit set = the five non-trackRATM flights, 60 clips:

| group | bins merged | n | threshold |
| --- | --- | --- | --- |
| 0-2+2-5 | 0-2, 2-5, 5-10, 10-20, 20-99 | 60 | **5.76°** |
| pooled (same set) | — | 60 | 5.76° |

Unmerged counts: 0-2 n=4, 2-5 n=37, 5-10 n=18, 10-20 n=1, 20+ n=0.

### Real clips, per flight (the applicability boundary)

| flight | testable | median score | p90 | max | abstained |
| --- | --- | --- | --- | --- | --- |
| 01p-ellipse | 26 | 2.09° | 3.02° | 3.64° | 0 % |
| 07p-lemniscate | 26 | 2.60° | 3.46° | 3.60° | 0 % |
| 04a-ellipse | 3 | 2.85° | 5.18° | 5.76° | 57 % |
| 07a-lemniscate | 3 | 5.33° | 40.08° | **48.77°** | 57 % |
| 01a-ellipse | 2 | 7.07° | 9.10° | 9.61° | 60 % |
| 13p-trackRATM | 11 | 7.05° | 8.64° | 8.95° | 0 % |
| 16p-trackRATM | 7 | 8.42° | 9.19° | 9.37° | 0 % |
| **13a-trackRATM** | 5 | **68.11°** | 78.59° | **79.65°** | 55 % |

Five clips above 40°: four on 13a, one on 07a. Those five clips, and nothing else, set the
real-scores' 95th percentile (2.91° median, 8.91° p90, **44.85° p95**).

### Negatives at the pre-registered operating point

| family | n | rejection rate | AUROC | real median | negative median |
| --- | --- | --- | --- | --- | --- |
| tilt removal (×0) | 83 | 0.373 | **0.918** | 2.91° | 18.90° |
| tilt ×0.5 | 83 | 0.373 | 0.803 | 2.91° | 8.80° |
| tilt ×1.5 | 83 | 0.349 | 0.743 | 2.91° | 5.06° |
| mirrored roll | 83 | 0.386 | **0.926** | 2.91° | 26.29° |
| yaw-rotation control | 83 | 0.349 | 0.617 | 2.91° | 3.81° |

Pooled AUROC over the three tilt families: 0.821. Per-family flight-level bootstrap
intervals are stored in `metrics.json` (`families.*.auroc_ci95`).

### Gate 3 against the pre-registered criteria

| criterion | target | measured | verdict |
| --- | --- | --- | --- |
| held-out acceptance | ≥ 0.95 | 0.711 (primary); **0.967 on the five non-trackRATM flights**, 0.043 on trackRATM | **fail** |
| AUROC, tilt removal | > 0.9 | **0.918** | pass |
| AUROC, tilt scaling | > 0.9 | 0.803 (×0.5), 0.743 (×1.5) | **fail** |
| minimum detectable violation at 80 % rejection | a curve | none of 27.2 / 13.6 / 13.6° reaches 80 % | **fail** |
| Spearman, score vs injected tilt change | > 0.8 | **0.82** | pass |
| control accepted at the real rate (±5 pp) | — | identity is exact (it *is* the real condition); yaw control rejected 35 % vs 10 % | **fail** |

### The three threshold policies (all reported; the primary is pre-registered)

| policy (fit set) | threshold | acceptance | tilt ×0 | ×0.5 | ×1.5 | mirror | control |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `non_trackratm` (primary) | 5.76° | 0.711 | 0.37 | 0.37 | 0.35 | 0.39 | 0.35 |
| `all_flights` | 67.16° | 0.952 | 0.06 | 0.06 | 0.06 | 0.11 | 0.06 |
| `all_but_13a` (sensitivity) | 9.37° | 0.904 | **0.89** | 0.45 | 0.33 | **0.96** | 0.31 |

`all_flights` meets the acceptance target only because the five bad real clips become the
threshold, which makes the test vacuous. `all_but_13a` is the policy the data support, and
it is reported as a sensitivity analysis — not swapped in for the pre-registered primary.

### Minimum detectable violation and the analysis-only negatives

| negative | injected change | median score | rejection at the operating point |
| --- | --- | --- | --- |
| tilt ×0 | 27.2° | 18.90° | 0.373 |
| tilt ×0.5 | 13.6° | 8.80° | 0.373 |
| tilt ×1.5 | 13.6° | 5.06° | 0.349 |
| delay 100 ms | — | 5.49° | inside the ±150 ms nuisance bound |
| delay 200 ms | — | 7.53° | outside the bound |
| delay 300 ms | — | 9.71° | outside the bound |
| retime 0.5× | — | 18.13° | analysis only |
| retime 2× | — | 19.20° | analysis only |

Smallest violation rejected 80 % of the time: **none** (`-`), i.e. the frontier is above
27°.

### Nuisance ablations

`results/phase3a_verdict/20261109T091536Z_ad3952a_ablations/ablations.json` (commit
`ad3952a`). Score medians in degrees:

| condition | both | no uptilt | no lag | neither | absorbed |
| --- | --- | --- | --- | --- | --- |
| real (identity) | 2.9 | 3.4 | 3.9 | 4.4 | 1.5 |
| tilt ×0 | 18.9 | 21.8 | 20.4 | 23.0 | 4.1 |
| tilt ×0.5 | 8.8 | 11.8 | 11.7 | 12.9 | 4.1 |
| tilt ×1.5 | 5.1 | 11.1 | 5.2 | 11.3 | **6.2** |
| mirrored roll | 26.3 | 26.5 | 28.2 | 28.3 | 2.0 |
| yaw control | 3.8 | 5.4 | 4.7 | 6.2 | 2.4 |

The nuisances are needed on real clips (4.4 → 2.9), do not absorb the large violations
(4.1° of 27.2°), and absorb **6.2° of the 13.6° half-tilt** — 46 % of the smallest
violation, which is the expected cost of a ±7° mount bound. Verdict-level AUROC moves only
0.821 → 0.835 with both nuisances off, so the ranking is robust even though the per-clip
score is not.

### Two checks that the pipeline is not fooling itself

**The identity edit changes nothing.** Same frames, same estimator, with the resampling
warp applied as the identity: per-flight median disagreement with the Phase 1 run (which
never saw a warp) is 0.03-0.16° (p90 0.09-0.43°) over all eight flights.

**The estimator moves when the content moves.** A rotation about world up cannot change the
tilt, so the motion of the depicted tilt under the yaw control is pure artifact. Median
absolute tilt change per flight, all frames and (in brackets) frames rotated by more than
10°: 01a 1.20 (4.35), 01p 1.89 (1.97), 04a 0.74 (4.75), 07a 0.87 (5.56), 07p 1.96 (2.03),
13a 1.19 (18.02), 13p 6.17 (7.38), 16p 7.36 (7.60); correlation with rotation angle 0.22-
0.72. That is the floor under any detectable violation, and it is why the control is
flagged above the real rate.

### Nuisance fits, as a diagnostic

| condition | fitted uptilt offset (median) | at the ±7° bound | fitted lag (median) | at the ±150 ms bound |
| --- | --- | --- | --- | --- |
| real | −1.0° | 13 % | −50 ms | 10 % |
| tilt ×0 | +7.0° | 87 % | −150 ms | 70 % |
| tilt ×0.5 | +7.0° | 61 % | −125 ms | 35 % |
| tilt ×1.5 | −7.0° | 90 % | −25 ms | 11 % |
| mirrored roll | −3.0° | 16 % | −150 ms | 80 % |
| yaw control | −3.0° | 16 % | −50 ms | 8 % |

On real clips the fit sits near zero; on the negatives it walks to the bound. That is the
absorption visible from the other side, and the −50 ms lag on real clips is a real
exposure/latency offset, not a fitted convenience.

## What this supports

- A working **ranking**: 0.918 (tilt removal) and 0.926 (mirrored roll) AUROC, with a
  90 %-acceptance operating point rejecting 89 % / 96 %. Sufficient to use the check as a
  filter or a best-of-N selector on well-behaved footage.
- A quantitative **false-alarm floor for ANWM** (0.57° median on level frames), so a
  level-camera violation is unambiguously outside the measurement noise.
- A measured **artifact floor** (~5° tilt change under content motion) and a measured
  **absorption budget** (4-6°) — the two numbers any future threshold must respect.
- A reusable **protocol** with the pre-registration committed before scoring, a code guard
  that refuses to score edits against a dirty spec, and every threshold rule fixed in
  advance.

## What this does not show

- **Not** that a 95 %-acceptance, 80 %-rejection detector exists for this dataset. It does
  not, at 83 real clips with five bad ones.
- **Not** that mirrored roll is detectable in general: it is detectable here because the
  path supplies the heading. A verifier without a heading reference can only compare
  scalar tilt and would be blind to it.
- **Not** anything about ANWM's behaviour. Track B's clips were downloaded and the scoring
  job was started, but at the user's instruction the run was stopped before results
  existed; the section below records exactly where it stands.
- Not a test of the VLM-judge or learned-classifier baselines (excluded by the handoff).

## Track B clip set: started, not finished

- Clips were fetched from R2 (`track-b/clips/`, 1,882 objects, 254 MB) and matched against
  their index: **82 clips** with 16 frames each, regimes A/B/C.
- Specs were built (`data/processed/anwm_specs/clips_v1`, 82 clips, dt ∈ {0.25, 0.667,
  1.0} s, `k_d` = 0.295), and the required tilt at 1 s was cross-checked against Track B's
  own `required_tilt_1s_kd_deg` column.
- The 224×224 depicted-tilt job was launched on Modal (`--job anwm --tag clips_v1`) and
  then stopped on instruction. Any rows it wrote before stopping are at
  `awmv-a-data:/geocalib/anwm_clips_v1/depicted.jsonl`; the scorer
  (`experiments/phase3a_anwm_scoring.py --specs … --depicted …`) has not been run on them.
- Nothing in this section is a result. No threshold was touched.

## Caveats and open items

- **The fix the data point at**, not applied here because it needs a new pre-registration:
  a depicted-consistency abstention — a clip whose estimated tilt contradicts its own
  temporal median for most frames is not testable. The five tail clips are exactly that
  kind of clip, and the `all_but_13a` row shows the operating point it would buy.
- **Threshold estimation** is the other half: 60 real clips cannot pin a 95th percentile
  with a heavy tail. More clips (shorter clips, more flights), a trimmed quantile, or a
  tail model would each help; none is applied.
- The yaw control is capped at 60° while the tilt edits reach 70° on two flights, so on
  those flights the control is slightly *easier* than the treatment.
- **Inferred:** that the view-dependence of the estimator (not resampling) drives the
  control's false flags. The correlation with rotation angle supports it; no ablation
  isolates resampling from content motion.
- Task 4 (DPVO on a well-calibrated dataset) is parked at the user's instruction; report
  05 records the two attempts and the blocker.
