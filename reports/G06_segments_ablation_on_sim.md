# G06 — Segment detection on the simulated set: which of FDR, boundaries, alignment and multi-scale actually help

**Track G3.** Branch `track-g3-segments`, worktree `~/Documents/Projects/awmv-g3`, base `main`
at `41d1dad`. Written 2026-11-22. Pre-registration `configs/g06_segments_prereg.yaml`
(`2744dbb`), amendment 1 `configs/g06_segments_prereg_amend1.yaml` (`ac21197`), both committed
before any defect clip was scored. Code `experiments/g3_*.py`, `remote/g3_*.py`. Results
`results/g06/`. CPU for the analysis; GPU for the reader (L40S) and the auxiliary yaw render.
Spend: see `docs/COST_LEDGER.md`.

**Status: development pass closed on all 170 clips available at freeze (80 feasible, 90 defect).
The held-out block is untouched** (`reports/G07` will score it once). Spend 5.27 USD from the
billing export; see `docs/COST_LEDGER.md`.

## Decision

**The smallest violation this design localises on the simulated set is a whole-clip one:** the
12-frame coherence scan catches 5 of 5 testable mirrored rolls and 3 of 5 testable tilt removals
at a 5.9 % false-reject, but it is testable on only 17 of 80 feasible clips (the manoeuvres whose
lean direction rotates) and nothing below whole-clip coverage is localised, because the
family-wise max-over-windows threshold saturates (coherence 1.934 of a possible 2.0; residual
61-75 deg) and the pre-registered synthetic step curve stays at the false-reject rate for every
step up to 24 deg. **What helps is false-discovery-rate control:** per-clip Benjamini-Hochberg on
the 8-frame residual scan detects 37 % of the defect clips at a 4.7 % false-reject, against 8 %
for the family-wise rule at 6.2 %; boundary refinement adds 0.09 IoU on the eight clips where a
region was flagged, while per-clip time alignment (-0.002 IoU) and the joint multi-scale maximum
(detects nothing) do not help.

## Plain-language summary

1. **The window machinery runs and the labels are exact, but the population is not what the
   protocol assumed.** Of the 170 development clips available at freeze, only two have partial
   intervals (both 10 %); the 18 remaining partial clips are among the ~30 dev clips Track I has
   not yet published. The confirmatory partial-interval criteria (S1, S2) therefore cannot be
   evaluated on this batch, and the report says so rather than substituting a different test.
2. **The coherence tier is precise but narrow.** Where its window is defined (the lean direction
   rotates on both body axes: slalom and lemniscate, 17 of 80 feasible clips) it detects every
   testable mirrored roll and most testable removals, with a 5.9 % false-reject. Where the lean is
   constant on one axis (steady turns, constant-speed ellipses, straight accelerations, climbs) it
   abstains by construction. It also loses power on tilt removal, because a level camera supplies
   almost no lean variance for a correlation to work with.
3. **The family-wise maximum is the wrong calibration on long clips.** A simulated clip has about
   100 windows where the real TII clips had 10-11, so the 95th percentile of the maximum sits at
   the ceiling of the statistic. The result is a threshold no realistic violation can cross: the
   residual scan detects 8 % of removals and 20 % of mirrors at 6.2 % false-reject, *worse* than
   the whole-clip median baseline (74 % and 48 %) which is designed for whole-clip edits.
4. **FDR control is the one change that works.** Benjamini-Hochberg within a clip, on the same
   8-frame residual windows, gives 37 % detection at a 4.7 % clip-level false-reject (q = 0.05)
   and 47 % at 10.9 % (q = 0.10), against 8 % for the family-wise rule. It restores the power the
   multiplicity of 100 windows had taken away.
5. **Localisation below whole-clip coverage is not achieved.** The pre-registered synthetic step
   curve is flat at the false-reject rate for the coherence arm at every step and duration tested;
   the residual arm only begins to respond at 20-24 deg lasting at least 0.8 s, and even there
   reaches 11-16 %. The exploratory BH-scored curve is also flat: a short step produces evidence
   only in the two windows that straddle its boundaries, and the multiplicity penalty removes it.
6. **The baseline's failure is the forest.** The whole-clip median false-flags 8 of 26 feasible
   forest clips (30.8 %) against 0 of 54 in the two open families; a pooled 10 % false-reject
   hides that. Any pooled claim on this set is an average over two very different readers.




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
| new stride 1 vs Track I (all 86 published scores) | 86 | 22,608 | 9.4e-4 | 2.4e-2 | 1.01 | 0.27 % |
| ... of which Track I's stride-2 files | 78 | 18,366 | — | — | 1.01 | — |
| ... of which Track I's stride-1 files | 8 | 4,242 | — | — | 0.23 | — |
| two identical L40S runs (one clip) | 1 | 1,202 | 2.2e-3 | 2.2e-2 | 5.6e-2 | 0.7 % |

Best frame offset is 0 (checked on `i02_dev_002`), so the stride-2 row mapping is correct. The
new stride-1 rows differ from the old stride-2 rows by no more than two identical runs of the
same clip differ from each other; **no decode-path difference is detectable**. The reader is
stochastic even with `cudnn.deterministic` and `torch.use_deterministic_algorithms` set: per
frame the typical disagreement is ~0.002 lean units (0.1°), and about one frame in four hundred
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

**Detection on whole-clip defects** (nominal threshold; k/testable, with the testable
denominator below the full family size in brackets):

| arm | tilt removed | mirrored roll | tilt scaled | false-reject |
| --- | --- | --- | --- | --- |
| coh12_p1 | 3/5 (60 %) | 5/5 (100 %) | 0/7 (0 %) | 1/17 (5.9 %) |
| coh8_p1 | 2/5 (40 %) | 5/5 (100 %) | 0/6 (0 %) | 2/13 (15.4 %) |
| res12_p1 | 2/25 (8 %) | 7/20 (35 %) | 4/25 (16 %) | 4/64 (6.2 %) |
| res8_p1 | 2/25 (8 %) | 4/20 (20 %) | 5/25 (20 %) | 4/64 (6.2 %) |
| res8_p2 | 2/25 (8 %) | 6/20 (30 %) | 4/25 (16 %) | 4/64 (6.2 %) |
| sign12_p1 | 0/5 | 0/5 | 0/7 | 0/17 |
| whole-clip median | 20/27 (74 %) | 13/27 (48 %) | 14/33 (42 %) | 8/80 (10 %) |

The family sizes are 27 removal, 27 mirror, 33 scale, so the arm denominators (5, 20, 25) show
the abstention: the arms are scored on the clips where a window exists, never on the full family.
The whole-clip median is testable on every clip, which is why its counts are larger; at the
*family-wise* threshold it is also the better detector of whole-clip edits, which is the regime
it was built for. The window scan's advantage has to come from partial coverage, and this batch
does not contain enough partial clips to measure it (section 5).

**Localisation.** For the clips the coherence arm detects, the mirrored-roll segment has a median
IoU of 0.95 (12-frame, 4/5 detected) and 0.69 (8-frame), because the edit covers the whole clip
and the flagged mask covers it too. Tilt-removal detections have IoU 0.06: the arm fires on a few
windows rather than the whole clip, so the segment is a fragment. The residual arm's detected
segments have IoU 0.20-0.68 depending on family.



### 2. A2 — FDR control

Benjamini-Hochberg within each clip, on the same windows and the same empirical null (the fit
clips' window distribution, leave-one-scene-out):

| arm | rule | q | detection (k/90 defect clips) | feasible clip false-reject |
| --- | --- | --- | --- | --- |
| residual 8 | per clip | 0.05 | 33 (36.7 %) | 3/64 (4.7 %) |
| residual 8 | per clip | 0.10 | 42 (46.7 %) | 7/64 (10.9 %) |
| residual 8 | per clip | 0.20 | 53 (58.9 %) | 14/64 (21.9 %) |
| residual 8 | corpus | 0.05 | 10 (11.1 %) | 1/64 (1.6 %) |
| residual 8 | corpus | 0.10 | 37 (41.1 %) | 12/64 (18.8 %) |
| coherence 12 | per clip | 0.05 | 7 (7.8 %) | 0/17 (0 %) |
| coherence 12 | per clip | 0.10 | 9 (10.0 %) | 0/17 (0 %) |

This is the ablation that changes the conclusion. The family-wise rule's residual arm detects 8 %
of removals and 20 % of mirrors at a 6.2 % false-reject; the same windows under per-clip BH detect
37 % of all defect clips at a **lower** false-reject (4.7 %), and 47 % at 10.9 %. The corpus-level
variant is worse at q = 0.05 and over-flags at q = 0.10, so the per-clip multiplicity correction
is the useful one. The coherence arm gains little (it is still limited by its abstention), and
its BH rule keeps a 0/17 false-reject. The realised false-discovery proportion is ~0.001 in every
cell, but that number is uninformative on this batch: 88 of the 90 defect clips are whole-clip, so
a flagged window is almost always inside the truth. The informative control is the clip-level
false-reject on the 80 feasible clips, which is in the last column.

### 3. A3 — boundary refinement

Eight clips had a flagged region to refine (the coherence arm's detections). Inside each region the
two-state half-normal fit moved the median IoU from **0.686 to 0.824** (mean Δ +0.088) and the
median F1 from **0.814 to 0.902**, meeting the pre-registered S4 bar (Δ ≥ 0.05). The eight clips
are all whole-clip edits, so the gain is the trim of the flagged region's false-positive edges
rather than the discovery of an interior onset; with two 10 % partial clips in the batch, the
interior-onset case is not exercised.

### 4. A4 — per-clip time alignment

The chosen lag is zero on most clips: median 0 ms, IQR 0-25 ms (industrial IQR 0 ms, urban 25 ms, forest 0 ms). Applying it moves the mean IoU from 0.0491 to 0.0471 and leaves the false-segment
rate unchanged (1.0 per 20.8 real minutes). **Alignment does not help**; the pre-registered S5
criterion is not met. The simulated videos are rendered on the same clock as the truth, so a
positive result would have been surprising; the measurement confirms the machinery is not
introducing a lag of its own.

### 5. A5 — multi-scale windows

Lengths {8, 12, 16} standardised per length and maximised jointly: mean IoU 0.000, i.e. the joint
rule detects essentially nothing, because the standardised maximum's conformal threshold inherits
the saturation of each component. Each length alone: 8 frames IoU 0.033, 12 frames 0.049, 16
frames 0.053. **The joint rule does not help** and fails S6; 16 frames alone is marginally the best
single length, which is consistent with the correlation needing a longer window, but the
difference from 12 is inside the noise of 90 clips.

### 6. Minimum localisable violation

The pre-registered synthetic step (a lean offset of Δ over a centred interval of duration D,
added to the reader's output on a feasible clip, labelled synthetic):

| Δ (deg) | coherence 12, 0.4-3.2 s | residual 8, 0.4 s | residual 8, 1.6 s | residual 8, 3.2 s |
| --- | --- | --- | --- | --- |
| 2-14 | 1 % (the FRR) | 5 % | 5 % | 5 % |
| 16 | 1 % | 7 % | 9 % | 11 % |
| 20 | 1 % | 11 % | 12 % | 12 % |
| 24 | 1 % | 11 % | 14 % | 16 % |

The coherence arm is **flat at its false-reject rate at every step and duration**: a constant
offset inside an interval does not change a correlation computed inside that interval, and only
the boundary windows carry evidence, which the family-wise maximum over 100 windows does not
reward. The residual arm responds only above ~20 deg and only for durations of 0.8 s or more, and
even then at 11-16 %. The exploratory BH-scored curve (same injection, BH instead of the maximum)
is **also flat at zero** at every cell: with a 0.4-3.2 s step, at most a few windows straddle the
boundaries, and a within-clip multiplicity correction over ~100 windows removes them.

The batch's two 10 % partial clips are detected by no arm, including the whole-clip median
baseline (0 of the 1-2 testable clips per arm), which is consistent with a 10 % interval being
shorter than any window the design can calibrate.

**No violation below whole-clip coverage is localised by this design.** The pre-registered
criteria S1 and S2 (partial-interval detection and IoU) cannot be evaluated on this batch for the
separate reason that the batch contains two partial clips; that is stated as a population
limitation, not as a failure.


### 7. Auxiliary yaw slice

A world yaw leaves the schema lean bit-identical, so the control can only be tested by rendering
(`configs/g06_yaw_slice.json`, 12 feasible clips, 3 families x 4 mounts, camera yawed by the
clip's own required tilt, alpha median per clip 10-50 deg). The slice is separate from Track I's
300 and is never pooled with it.

| quantity | value |
| --- | --- |
| clips rendered / scored | 12 / 12 |
| testable under the primary arm (null, yaw) | 2, 2 |
| flagged (null, yaw) | 0, 0 |
| reader lean shift, median of clip medians | 1.4 deg |
| per-clip median shift, range | 0.3-10.3 deg |
| per-clip p90 shift, range | 0.6-39.5 deg |
| max shift | 56.5 deg |
| real-clip reference (G01) | 4.75 deg median |

The control is **negative**: on the two clips where the coherence window is defined, neither the
null nor the yawed render is flagged, so the yaw edit does not create a false detection. The
reader, however, is not indifferent to the viewpoint: its median lean shift is 1.4 deg across
clips, but reaches 10 deg on a forest clip and its per-clip 90th percentile reaches 39 deg at 45
deg mounts, far above the 4.75 deg measured on real footage. Those tails sit on frames where
GeoCalib picks a different local optimum — the same multimodality as section 0 — and they are the
reason the yaw control cannot be quoted as "yaw is harmless": with only 2 of 12 clips testable
under the primary arm, the slice has little power to detect a yaw-induced false flag, and the
honest reading is that the control neither supports nor refutes a yaw effect at this sample size.

## What this supports

- **FDR control is the ablation that matters.** Per-clip Benjamini-Hochberg on the 8-frame
  residual scan detects 36.7 % of the 90 whole-clip defect clips at a 4.7 % clip-level
  false-reject (q = 0.05), and 46.7 % at 10.9 % (q = 0.10). The family-wise max-over-windows rule
  on the same windows detects 8 % of removals and 20 % of mirrors at 6.2 %. The corpus-level BH
  variant is worse at q = 0.05 (11.1 %) and over-flags at q = 0.10 (12/64 feasible), so the
  per-clip correction is the useful form.
- **The family-wise maximum is not calibratable on long clips.** A simulated clip has ~100
  windows; the 95th percentile of the maximum sits at 1.934 of the coherence ceiling 2.0, at
  1.000 for the sign tier (which therefore has no power), and at 61-75 deg for the residual.
  This is a property of the window count, not of the violation, and it is why the pre-registered
  step curve is flat (section 6).
- **Boundary refinement works where a region exists.** The two-state half-normal fit raises the
  median IoU from 0.686 to 0.824 and F1 from 0.814 to 0.902 on the eight clips with a flagged
  region, meeting S4.
- **Per-clip time alignment and the joint multi-scale maximum do not help.** The chosen lag is
  zero on most clips (median 0 ms, IQR 0-25 ms), applying it changes mean IoU by -0.002, and the
  joint standardised maximum detects essentially nothing (mean IoU 0.000) because it inherits the
  saturation of each component. S5 and S6 fail.
- **The scale-free coherence tier is precise but structurally narrow.** On the clips where its
  window is defined it detects 5/5 testable mirrored rolls and 3/5 testable removals at a 5.9 %
  false-reject with IoU 0.95 for mirror; but it is testable on 17 of 80 feasible clips (slalom
  13/16, lemniscate 4/8, zero on ellipse, turn, climb-turn and accel-brake), and a level camera
  removes most of its removal power. Its removal detection on real clips (J03) depended on the
  reader's ~4 deg noise, which the simulated reader does not supply.
- **The whole-clip median baseline's failure is scene-specific.** It false-flags 8 of 26 feasible
  forest clips (30.8 %) against 0 of 54 in the open families, because the forest reader bias is
  comparable to its threshold. Any pooled rate on this set averages two different readers.

## What this does not show

- **Nothing about partial-interval detection.** The 170-clip batch contains two partial clips
  (both 10 %), so the pre-registered S1/S2 criteria (25 % and 50 % coverage) are **not evaluated**
  rather than failed. The 18 remaining partial clips are in Track I's unrendered ~30 and will be
  scored as batch 2 under the same thresholds.
- **Nothing about the real-clip transfer of these knobs.** The ablations run on simulated dev
  clips; the real held-out block (`flight-13p-trackRATM`, `flight-07a-lemniscate`) is reserved for
  G07 and has not been scored here beyond the frozen round-1 population.
- **Nothing about held-out scenes.** G07 scores them once.
- **No path-side claim.** `thrust_limit` is a path label; the camera is honest and the residual is
  zero by construction. The segment machinery has no power on it, and no path-side statistic is
  added in this track.
- **The exact minimum localisable violation is not established.** The pre-registered curve is flat
  for the coherence arm and rises only above ~20 deg for the residual arm; a different window
  length or a change-point model might localise less, but that is not tested here.

## Caveats

- Simulated scenes are procedural; the reader's error on them is the sim-to-real quantity to
  quote, not the visual plausibility of a frame.
- Six scene seeds and six mounts make every interval a descriptive spread, not a 95 % confidence
  interval; the dagger applies to every cell.
- The 170 clips scored here contain only two partial-interval clips (both 10 %); the
  partial-interval ablation is re-run when Track I publishes the rest, under the same thresholds.
- `thrust_limit` is path-side and the segment machinery has no power on it; it is reported as its
  own family and never pooled.
- The reader is stochastic run to run; the envelope (section 0) enters every detection number as
  a noise floor, and no detector is claimed to resolve a violation smaller than it.

## Confirmatory and exploratory

Every table above is **confirmatory** against `configs/g06_segments_prereg.yaml` unless the row
says otherwise. The labels:

| item | status |
| --- | --- |
| A1 arms, A2 q-grid and corpus variant, A3 refinement, A4 lag grid, A5 lengths, the synthetic minimum-detectable-step curve, the yaw slice, the stratum tables | confirmatory (pre-registered) |
| any arm added after seeing a dev result, including the BH-scored step curve in `results/g06/ablation/mdv_fdr_exploratory.csv` | **exploratory**, labelled in the file name and in every table |
| everything on the held-out blocks | confirmatory, once, in G07 |
