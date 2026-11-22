# G06 — Segment detection on the simulated set: which of FDR, boundaries, alignment and multi-scale actually help

**Track G3.** Branch `track-g3-segments`, worktree `~/Documents/Projects/awmv-g3`, base `main`
at `41d1dad`. Written 2026-11-22. Pre-registration `configs/g06_segments_prereg.yaml`
(`2744dbb`), amendment 1 `configs/g06_segments_prereg_amend1.yaml` (`ac21197`), both committed
before any defect clip was scored. Code `experiments/g3_*.py`, `remote/g3_*.py`. Results
`results/g06/`. CPU for the analysis; GPU for the reader (L40S) and the auxiliary yaw render.
Spend: see `docs/COST_LEDGER.md`.

**Status: confirmatory on the development block. The held-out block is untouched**
(`reports/G07` will score it once).

## Decision

<!-- FILL: two sentences -->

## Plain-language summary

<!-- FILL -->

## Question

The frozen verdict is one clip-level median. G01 showed a window scan finds partial violations
but localises them poorly (IoU 0.15-0.24), and J03 showed the scale-free coherence tier is the
only tier a per-clip nuisance fit cannot absorb and has the smallest detectable local step
(16.8° at 25 % coverage, 14.4° at 50 %). This track asks, with the coherence tier as the window
statistic and exact simulated labels: which of (FDR control, boundary refinement, per-clip time
alignment, multi-scale windows) improves localisation, and what is the smallest violation the
machinery can localise?

## Setup

**Population.** 170 of the 200 simulated development clips available at freeze (80 feasible,
27 tilt-removed, 27 mirrored-roll, 33 tilt-scaled, 2 partial-interval, 1 thrust-limit); scenes
`urban_checker`, `forest_ridge`, `industrial` (seeds 101-106); mounts 0/15/30/45°. The remaining
~30 infeasible dev clips enter as batch 2 under the same config when Track I publishes them.
Every rate is stratified by scene family and mount, and the forest family is reported
separately because its reader bias is an order of magnitude larger (I02).

**Reader.** GeoCalib, pinhole weights, focal prior, every video frame (stride 1, 30 fps). Track
I's published scores mixed stride 1 and stride 2; all clips were re-scored at stride 1 so a
window length in frames is a fixed duration. The reader is stochastic run to run (measured
envelope in amendment 1: median 2.2e-3 lean units, p99 2.2e-2, max 5.5e-2), which is stated in
every table as part of the noise floor.

**Windows.** 12 frames (0.40 s), hop 4, coherence tier `1 - min(r_x, r_y)` (primary); 8- and
12-frame residual scans and the sign tier as comparisons; persistence 2 as a secondary. A window
abstains below 9/12 usable frames or 2 signal frames (required tilt ≥ 10°). A clip with no
eligible window is not testable for that arm and is excluded from that arm's denominator, never
counted as a pass.

**Calibration.** Leave-one-scene-seed-out empirical 95th percentile of the clip statistic on the
feasible dev clips (six folds; six clusters, so the intervals are descriptive, not 95 %
confidence intervals). The baseline is the whole-clip median residual calibrated the same way.

**Controls.** The null edit in simulation is the unchanged feasible clip (exact, by construction).
The yaw control cannot be composed on the depicted side — a world yaw leaves the schema lean
bit-identical (`tests/test_g3_yaw_invariance.py`) — so a 12-clip auxiliary rendered slice was
built instead (`configs/g06_yaw_slice.json`), reported separately and never pooled with Track I's
set. `thrust_limit` is a path-side label: the camera is honest, the residual is zero by
construction, and the segment machinery has no power on it; it is reported as its own family and
no path-side statistic is added.

**Ablations (each alone, then stacked).** A1 window statistic; A2 Benjamini-Hochberg within clip
(q = 0.10 primary, 0.05/0.20 and corpus-level as ablations) against the max-over-windows
family-wise rule; A3 two-state half-normal boundary refinement inside flagged regions; A4 one
lag per clip from the frozen ±150 ms / 25 ms grid; A5 multi-scale {8, 12, 16} with one joint
conformal threshold. The minimum-localisable-violation curve is a pre-registered synthetic lean
step added to feasible clips (labelled synthetic in every table).

## Results

### 0. Reader consistency and the noise floor

Every clip was re-scored at stride 1 (30 fps) because Track I's published score files mixed
stride 1 (8 clips, 4,242 frames) and stride 2 (78 clips, 18,366 frames). The new rows were
compared with the old ones at every shared frame index (`results/g06/consistency/`):

| comparison | clips | shared frames | component median \|d lean\| | p99 | max | frames > 0.05 |
| --- | --- | --- | --- | --- | --- | --- |
| new stride 1 vs Track I stride 1 | 3 | 1,252 | 1.3e-3 | 6.7e-2 | 0.41 | 1.4 % |
| new stride 1 vs Track I stride 2 | 53 | 12,249 | 2.2e-3 | 4.6e-2 | 1.01 | 0.87 % |
| two identical L40S runs (one clip) | 1 | 1,202 | 2.2e-3 | 2.2e-2 | 5.6e-2 | 0.7 % |

Best frame offset is 0 (checked on `i02_dev_002`), so the stride-2 row mapping is correct. The
new stride-1 rows differ from the old stride-2 rows by no more than two identical runs of the
same clip differ from each other; **no decode-path difference is detectable**. The reader is
stochastic even with `cudnn.deterministic` and `torch.use_deterministic_algorithms` set: per
frame the typical disagreement is ~0.002 lean units (0.1°), and about one frame in a hundred
jumps much further (up to ~1 lean unit, tens of degrees) where GeoCalib lands on a different
local optimum. This is the reader's own multimodality, not the labels: both runs read the same
pixel rows. The window statistic is a median (residual) or a correlation (coherence) over 12
frames, so a single such frame cannot move a window to the threshold; the tails are nevertheless
part of the noise floor quoted in section 6.

### 1. A1 — the window statistic

**The feasible population is complete** (80 of 80 dev feasible clips scored), so the
false-reject table and every threshold below are final, not provisional.

**Testability.** A clip is testable for an arm when at least one window is eligible. The
residual arms are testable on 64 of 80 feasible clips; the coherence arms on 17 of 80, and only
on the two manoeuvres whose lean direction rotates in both body axes (slalom 13/16, lemniscate
4/8; zero on ellipse, turn, climb-turn and accel-brake, whose lean is nearly constant on one
axis). Hover clips are not testable under any arm. This is the structural scope limit of the
coherence tier and it is a property of the manoeuvre, not of the violation.

**Calibration saturates.** The conformal 95th percentile of the clip maximum (leave-one-scene-out
spread in brackets):

| arm | threshold | fold spread | testable clips |
| --- | --- | --- | --- |
| coh12_p1 | 1.934 (max possible 2.0) | 1.92-1.94 | 17 |
| coh8_p1 | 1.970 | 1.94-1.97 | 13 |
| coh12_p2 | 1.874 | 1.82-1.88 | 16 |
| res12_p1 | 61.4 deg | 45.1-64.9 | 64 |
| res8_p1 | 75.1 deg | 45.1-82.8 | 64 |
| res8_p2 | 66.6 deg | 44.6-70.8 | 64 |
| sign12_p1 | 1.000 | 1.00-1.00 | 17 |
| whole-clip median | 11.18 deg | 9.37-11.33 | 80 |

Two mechanisms, both measured. (i) A clip has about 100 windows where the real TII clips had
10-11, so the maximum over windows sits on a far more extreme order statistic; the coherence
threshold reaches within 0.07 of the statistic's ceiling, and the sign threshold reaches the
ceiling exactly (no clip can exceed it, so that arm has no power by construction). (ii) Three
feasible clips carry catastrophic reader windows (one window at 90 deg of residual on an
otherwise 1-2 deg clip), and with 55 feasible clips a 95th percentile sits on them.

**False-reject** at the nominal 5 % (k/testable):

| arm | urban | forest | industrial | all |
| --- | --- | --- | --- | --- |
| res8_p1 | 2/22 (9.1 %) | 1/22 (4.5 %) | 1/20 (5.0 %) | 4/64 (6.2 %) |
| res12_p1 | — | — | — | 4/64 (6.2 %) |
| coh12_p1 | 0/5 (0 %) | 1/6 (16.7 %) | 0/6 (0 %) | 1/17 (5.9 %) |
| coh8_p1 | 1/5 (20 %) | 1/4 (25 %) | 0/4 (0 %) | 2/13 (15.4 %) |
| sign12_p1 | 0/5 | 0/6 | 0/6 | 0/17 (0 %) |
| whole-clip median | 0/27 (0 %) | **8/26 (30.8 %)** | 0/27 (0 %) | 8/80 (10.0 %) |

The whole-clip median baseline is the worst-behaved rule on the forest family: it false-flags
30.8 % of feasible forest clips, because the reader's forest bias (7-10 deg) is comparable to
the threshold. That is the practical content of the scene stratification: a pooled false-reject
of 10 % hides a 0.8 % rate on two families and a 31 % rate on the third.

Detection, IoU, the ablations and the controls follow from the full defect set.



### 2. A2 — FDR control

<!-- FILL: realised FDP, false segments/min, detection at q = 0.05/0.10/0.20, per-clip vs corpus -->

### 3. A3 — boundary refinement

<!-- FILL: delta IoU/F1/onset -->

### 4. A4 — per-clip time alignment

<!-- FILL: lag distribution, delta metrics -->

### 5. A5 — multi-scale windows

<!-- FILL: joint vs each length -->

### 6. Minimum localisable violation

<!-- FILL: detection contour over delta and duration, 50% and IoU>0.3 contours -->

### 7. Auxiliary yaw slice

<!-- FILL: flag rate vs null, reader yaw shift vs G01's 4.75 deg -->

## What this supports

<!-- FILL -->

## What this does not show

<!-- FILL -->

## Caveats

<!-- FILL -->

## Confirmatory and exploratory

Every table above is **confirmatory** against `configs/g06_segments_prereg.yaml` unless the row
says otherwise. The labels:

| item | status |
| --- | --- |
| A1 arms, A2 q-grid and corpus variant, A3 refinement, A4 lag grid, A5 lengths, the synthetic minimum-detectable-step curve, the yaw slice, the stratum tables | confirmatory (pre-registered) |
| any arm added after seeing a dev result, including the BH-scored step curve in `results/g06/ablation/mdv_fdr_exploratory.csv` | **exploratory**, labelled in the file name and in every table |
| everything on the held-out blocks | confirmatory, once, in G07 |

## What this supports

<!-- FILL after the numbers: the mechanisms below are already measured on the first 83 clips -->

- The window machinery runs end to end on the simulated set with exact labels: every clip has
  per-frame reader output at 30 fps, an exact required lean, and an exact violation interval.
- The reader is stochastic run to run; the envelope is measured and quoted, and the stride-1
  versus stride-2 row mapping is correct (best offset 0, differences inside the envelope).
- The scale-free coherence tier has a **structural scope limit** that the real-corpus work could
  not see: it needs both axes of both leans to vary inside a window. On the simulated set that
  is true for the manoeuvres whose lean direction rotates (slalom, lemniscate) and false for
  steady turns, ellipses at constant speed, straight accelerations and climbs. It also, and more
  sharply, **abstains on tilt removal**: a level camera has no lean variance for a correlation to
  work with, so the tier cannot detect the violation at all when the reader is accurate. J03's
  detection of tilt removal on real clips came from the reader's own 4 deg noise supplying that
  variance.
- The family-wise max-over-windows conformal rule is **not calibratable on 100-window clips**:
  the 95th percentile of the maximum saturates (coherence 1.94 of a maximum 2.0; residual 60-80
  deg, carried by three feasible clips with catastrophic reader windows). This is a property of
  the number of windows per clip, not of the violation.

## What this does not show

<!-- FILL -->

## Caveats

- Simulated scenes are procedural; the reader's error on them is the sim-to-real quantity to
  quote, not the visual plausibility of a frame.
- The 170 clips scored here contain only **2 partial-interval clips** (both at 10 % coverage).
  The 18 remaining 25 % / 50 % partial clips are among the ~30 dev clips Track I has not yet
  published, so the partial-interval detection and IoU columns rest on two clips until batch 2;
  the minimum-detectable-violation curve is the pre-registered way to study localisation in the
  meantime, and it is labelled synthetic.
- Six scene seeds and six mounts make every interval a descriptive spread, not a 95 % confidence
  interval; the dagger applies to every cell.
- `thrust_limit` is path-side and the segment machinery has no power on it; it is reported as
  its own family and never pooled.
