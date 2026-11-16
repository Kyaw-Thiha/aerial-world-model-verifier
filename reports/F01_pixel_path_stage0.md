# Report F01 — Pixel-path verdict, stage 0: statistics that avoid the second derivative

Track F. Written 2026-11-16. Pre-registration `configs/f01_pixel_path_prereg.yaml`
(`6b88d5a`) and amendment 1 `configs/f01_pixel_path_prereg_amend1.yaml` (`17d21c6`), both
committed before any real clip was scored. Code and run: `3aa8a13`; results in
`results/f01_stage0/` (`summary.json`, `rows.csv`, `analysis.json`, `edit_path_check.json`).

**Decision: fail.** With estimated attitude and the stored DPVO path the pre-registered
primary statistic (V2, shooting) flags 1/8 real clips (0.125, inside the 15 % budget) but
only 6/8 tilt removals (0.750) and 4/8 mirrored rolls (0.500), short of the 90 % bar, and no
secondary variant or scale-free tier does better. The same statistics reach 8/8 removal and
7–8/8 mirror on the **truth** path, so the remaining obstacle is the path, not the statistic.

## Plain-language summary

The earlier pixel-path pilot failed after differentiating the recovered path twice. This stage
tried five ways of not doing that, on the same stored UZH paths, with the attitude supplied
first from truth and then from the pixels so path error and lean error could be separated.

Avoiding the second derivative helps, and the ordering is clean: the statistic that never
differentiates the path (integrating the depicted lean instead) detects 6 of 8 tilt removals
and 4 of 8 mirrored rolls, where the old pilot separated nothing; one derivative already
collapses removal detection to 1 of 8; two derivatives inside the acceleration band collapse
it further. But 0.75 and 0.50 are below the pre-registered 0.90 bar, so the verdict for this
stage is **fail**.

The reason is visible in the oracle ladder: put the truth path in place of the DPVO path and
the same code detects 8/8 removals and 7–8/8 mirrors. The statistic is not the problem; the
recovered path still is. A threshold-free view says the same thing more sharply — the shooting
statistic separates real from removed clips with rank AUC 0.95 and from mirrored with 0.89 —
but with four sequences a conformal 95th percentile is nearly the maximum of six calibration
clips, so a statistic that separates well does not clear a 0.90 detection bar.

## The question

Given a path from pixels, can the verifier say a flight is infeasible without taking a second
derivative of that path and without a metric scale? The variants, the gate and the analysis
plan were fixed in advance; nothing here was retuned after the numbers were seen.

## Setup

### Inputs

| input | what | where |
| --- | --- | --- |
| pixel path | Track C's DPVO camera-to-world poses, prepared UZH pinhole frames, 150 frames, 15.07 Hz, Kalibr intrinsics supplied | Modal volume `awmv-c-data`, `poses/dpvo/*.npz`; local copy `data/pathrec/poses/dpvo/` (sha256 in `summary.json`) |
| depicted lean | Track E v3 fixed-crop GeoCalib through the published Kalibr mount, schema v2 at the 0.37 s window | `data/clips_v2_w370ms/<seq>/*.npz` |
| truth | Leica MS60, pos + quat (body→world) at the frame times, 500 Hz table for testability | `data/uzh/<seq>/truth.npz` |
| window rule | E05 `0.5/f95` per sequence, floor 8 samples at the truth rate | `results/e_window_rule.json` |
| depth prior | DA3METRIC-LARGE median depth per frame | `results/pathrec_calib/depth.json` |

The DPVO path's Sim(3) quality on the ten sequences (evaluation only, never an input to a
statistic): median direction error 1.5–20.9 deg (C07), ATE 0.10–3.32 m, path length
40.7–318.9 DPVO units, fitted scale 0.17–0.99 m per unit (`summary.json`, `path_quality`).

### Population

Nine forward sequences × three 60-frame clips = 27 real clips, each with seven conditions.
The stored DPVO paths cover prepared frames 0–149, while clips start at frames 0/60/120
(indoor) or 60/120/180 (outdoor), so a clip is scoreable only if it ends by frame 150:

| group | clips | treatment |
| --- | --- | --- |
| full 60-frame path, usable | 14 (indoor 3/5/6/9/10 ×2, outdoor 1/3 ×1 and 5 ×2) | population |
| truncated to 30 frames | 8 (clip #02 of five indoor and three outdoor) | exploratory only |
| no overlap (outdoor_1#03, outdoor_3#03) | 2 | not scoreable |
| `indoor_forward_7` | 2 full + 1 truncated | excluded: its v3 export is broken (E04) |

**Primary population: 8 real clips in 4 indoor-forward sequences** — `indoor_forward_3`,
`_5`, `_9`, `_10`, clips #00 and #01 — those Track E finds R2/coherence-testable, the frozen
confidence gate keeps, and the stored path covers in full. This is E's testable set minus
`indoor_forward_5#02`, whose path is truncated. No path was extrapolated and DPVO was not
re-run for this stage (CPU only, zero spend).

### What was fitted and what was fixed

Every statistic works in the plane perpendicular to an up direction supplied by the attitude
source — the truth quaternion, or the depicted lean inverted through the published mount.
A constant yaw offset ψ and a positive path scale are nuisance parameters, fitted on the
clip's **real** condition and held fixed for its edits (amendment 1). That pairing matters:
with a free scale the tilt-removed condition is explained by `s → 0` (the model claims no
acceleration, so the residual is minimised by shrinking the ruler), and V1/V2/V4 become blind
to removal by construction. The degeneracy was found on synthetic trajectories before any real
clip was scored and is asserted in `tests/test_pathverdict.py`.

The five statistics, all scale-free except V5:

* **V2 shooting (primary).** Integrate `a_h = g·Rz(ψ+yaw)·l_dep − k_d v_h` twice from a free
  initial state and least-squares the observed path **undifferentiated**. Statistic
  `ρ = sqrt(SSE/(s²·SST))`, the unexplained fraction of the path.
* **V1 preintegration.** One Savitzky–Golay derivative of the path, integrate the same ODE
  once, fit scale and initial velocity; statistic `ρ_v` on velocity. Per-axis correlation and
  sign agreement are reported beside it.
* **V3 band-limited coherence.** Complex coherence between the depicted lean and the path's
  horizontal acceleration inside `[0.05, f95]` Hz, statistic `1 − |ρ_C|`. Invariant to a
  constant rotation, so yaw-free.
* **V4 velocity-drag.** Near-steady flight, `l ≈ (k_d/g)v`, one derivative. Testable only when
  `median |a_h|/(k_d |v_h|) < 0.5`.
* **V5 physics scale.** The scale V2 recovers, checked for identifiability (relative standard
  error, positive sign) and against the depth prior's plausibility band.

### Thresholds

Leave-one-sequence-out conformal 95th percentile on **real** clips, per variant and arm; a
held-out clip is flagged when its statistic exceeds the threshold built from the other three
sequences. In-sample thresholds are printed for transparency but are not the headline. Every
rate below carries its `k/n`; with four clusters the sequence-clustered intervals are
descriptive, not 95 % confidence intervals.

## Results

### 1. The gate arm

Estimated attitude, stored pixel path, primary population (8 clips, 4 sequences):

| variant | real | null | yaw | removal | mirror | x0.5 | x1.5 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| V1 preintegration | 0.125 (1/8) | 0.125 (1/8) | 0.250 (2/8) | 0.125 (1/8) | 0.625 (5/8) | 0.250 (2/8) | 0.750 (6/8) |
| **V2 shooting (primary)** | **0.125 (1/8)** | 0.125 (1/8) | 0.250 (2/8) | **0.750 (6/8)** | **0.500 (4/8)** | 0.375 (3/8) | 0.750 (6/8) |
| V3 band coherence | 0.250 (2/8) | 0.250 (2/8) | 0.125 (1/8) | 0.125 (1/8) | 0.000 (0/8) | 0.250 (2/8) | 0.125 (1/8) |
| V4 velocity-drag | untestable | untestable | untestable | untestable | untestable | untestable | untestable |
| V5 physics scale | 0.000 (0/8) | 0.000 (0/8) | 0.000 (0/8) | 0.250 (2/8) | 0.000 (0/8) | 0.000 (0/8) | 0.000 (0/8) |

Sequence-clustered intervals over the 4 primary sequences (descriptive):

| variant | real | removal | mirror |
| --- | --- | --- | --- |
| V1 | 0.00–0.375 | 0.00–0.375 | 0.25–1.00 |
| V2 | 0.00–0.375 | 0.50–1.00 | 0.00–1.00 |
| V3 | 0.00–0.50 | 0.00–0.375 | 0.00–0.00 |

The false-reject budget is met (1/8 = 12.5 % ≤ 15 %) and detection is not (6/8 and 4/8 against
90 %). **The gate fails.** The single real false reject is `indoor_forward_9#01` (ρ 0.467
against τ 0.261); V1's is `indoor_forward_3#01` (1.391 against 0.897); V3's two are
`indoor_forward_5#01` and `indoor_forward_9#01`.

Thresholds actually used (LOSO, 95th percentile of the six training real clips):

| variant | indoor_3 | indoor_5 | indoor_9 | indoor_10 |
| --- | --- | --- | --- | --- |
| V1 | 0.897 | 1.310 | 1.142 | 1.310 |
| V2 | 0.405 | 0.419 | 0.261 | 0.419 |
| V3 | 0.771 | 0.652 | 0.707 | 0.771 |
| V5 | 0.0 | 0.0 | 0.0 | 0.0 |

Abstention on the primary real clips: none for V1/V2/V3/V5; V4 abstains on 8/8.

### 2. The oracle ladder — the statistic versus the path

Same population, same fitting code, with the truth position in place of the pixel path
and/or the truth attitude in place of the depicted lean:

| path | attitude | real | removal | mirror |
| --- | --- | --- | --- | --- |
| pixel | estimated | 0.125 (1/8) | 0.750 (6/8) | 0.500 (4/8) |
| truth | estimated | 0.125 (1/8) | **1.000 (8/8)** | **0.875 (7/8)** |
| pixel | truth | 0.125 (1/8) | 0.125 (1/8) | 0.125 (1/8) |
| truth | truth | 0.125 (1/8) | 0.125 (1/8) | 0.125 (1/8) |

(The shooting statistic. Preintegration behaves the same way: pixel path 0.125/0.625, truth
path 1.000/1.000; with the truth attitude every condition is identical to real, because that
arm never reads the pixels.)

Two readings. First, the statistic is sound: on a good path it detects removal perfectly and
the mirror nearly perfectly, at the same false-reject rate. Second, path quality is the
binding term, and the oracle rows say nothing about the depicted side — they are a ceiling,
not a candidate verdict.

Threshold-free separation (rank AUC, real against each edit, primary clips) makes the failure
mode explicit:

| variant, path | removal | mirror | x1.5 | null | yaw |
| --- | --- | --- | --- | --- | --- |
| V2, pixel path | 0.953 | 0.891 | 0.938 | 0.500 | 0.641 |
| V2, truth path | 1.000 | 0.984 | 1.000 | 0.547 | 0.578 |
| V1, pixel path | 0.750 | 0.891 | 0.906 | 0.516 | 0.609 |
| V3, pixel path | 0.672 | 0.609 | 0.578 | 0.500 | 0.562 |

The shooting statistic separates real from removed clips better than 0.95 in rank terms while
the pre-registered rule converts that into 0.75 detection. The gap is the conformal threshold:
with four clusters, a 95th percentile over six calibration clips is all but their maximum
(V2's 0.405–0.419 against a median real statistic of 0.123), so the rule buys a very low false-reject rate
and pays for it in detection. Lowering the level would trade one for the other; that was not
pre-registered and is not done here.

### 3. Which derivative costs what

Ordering the statistics by how many times they differentiate the observed path, on the same
eight clips and with the same threshold rule:

| derivatives of the path | variant | removal | mirror |
| --- |  --- | --- | --- |
| 0 | V2 shooting | 0.750 (6/8) | 0.500 (4/8) |
| 1 | V1 preintegration | 0.125 (1/8) | 0.625 (5/8) |
| 2, band-limited | V3 coherence | 0.125 (1/8) | 0.000 (0/8) |

Removal detection falls monotonically with the derivative count, which is the mechanism this
stage was built to test; the mirror rate does not order as cleanly (one derivative keeps 5/8),
so the claim is about removal, not about both edits. The old double-differentiation pilot
(C09) separated nothing at all; V2 does separate, and this table is the size of the
improvement.

### 4. Controls and the scale edits

Controls behave as controls: the null edit reproduces the real statistic to within 0.001
on every clip (depicted-lean agreement is 0.03–0.15 deg in E04's validity gate), and the yaw
control moves it on two clips (0.250 for V2) because the yawed re-render changes the depicted
lean slightly while ψ is held fixed. That is the artefact floor this design can see and must be
read beside every detection rate.

The positive scale edits are **not** at chance under the amended design. The parent
pre-registration expected `x0.5`/`x1.5` to be absorbed by the fitted scale for V1/V2/V4 and by
the correlation for V3; amendment 1 fixes the scale on the real condition, so a lean scaled by
1.5 against an unchanged path is genuinely inconsistent and is detected 6/8 by V2. This is a
logical consequence of the paired ruler, not a tuned result, and it enters no gate. V3, which
is scale-free, keeps the expected chance behaviour (2/8 and 1/8).

### 5. V4 and V5

**V4 is not testable on this population.** The pre-registered drag-dominated condition
(`median |a_h|/(k_d|v_h|) < 0.5`, one-differentiation pipeline) holds on none of the 22 real
clips that have a path: the observed ratio is 0.73–9.35 (median 3.55). UZH-FPV is not
near-steady flight at a 4 s clip and 15 Hz, so a velocity-only drag test has no population
here. The result is an abstention, not a failure of the statistic, and the rate columns are
empty rather than zero.

**V5's ruler is identifiable but biased.** On the 8 primary real clips the shooting fit's
scale is positive with relative standard error 0.010–0.095 (median 0.028), the depth-consistency ratio is
1.1×10⁻³ to 1.0×10⁻¹ (inside the pre-registered `(1e-4, 0.5)` band), and no real clip is
flagged. The fitted scale is 0.44–0.84 of the Sim(3) truth scale (median 0.59), i.e. the
`g·lean` ruler systematically under-scales the path by about 1.7×. The data here cannot
attribute that between estimator attenuation (E's calibration slope on truth attitude is
0.86) and the coupling between scale and the drag term; it is reported as a bias, not fitted
away. V5 flags 2/8 removals through unidentifiability and no scale edit, because the
plausibility band is deliberately loose.

### 6. The exploratory population

Every real clip outside the primary set, scored with the pooled 95th percentile of the primary
real statistics as its threshold (labelled: this threshold is not held out):

* 14 further real clips have a pixel path (8 truncated, 6 full); 4 of them are testable
  under the same rules — `indoor_forward_5#02`, `indoor_forward_10#02`,
  `outdoor_forward_1#02`, `outdoor_forward_3#01`. The outdoor rigs otherwise abstain on the
  frozen confidence gate, `indoor_forward_6` fails the field rule, and `indoor_forward_7` is
  excluded. The pooled threshold is 0.400 for V2.
* V2: real 0/4, removal 0/4, mirror 1/4. V1: real 0/4, removal 0/4, mirror 0/4. V3: 0/4
  everywhere.
* With four testable clips these rates are anecdotal. Their value is that they do not
  contradict the primary result: nothing separates once the threshold is the conservative
  pooled one.

### 7. Scale, path quality, and the edit-invariance check

| sequence | Sim(3) scale (m/unit) | ATE (m) | path length (units) |
| --- | --- | --- | --- |
| indoor_forward_3 | 0.679 | 0.15 | 70.3 |
| indoor_forward_5 | 0.227 | 0.14 | 105.6 |
| indoor_forward_6 | 0.259 | 1.04 | 212.6 |
| indoor_forward_9 | 0.263 | 0.10 | 121.5 |
| indoor_forward_10 | 0.237 | 0.16 | 134.0 |
| outdoor_forward_1 | 0.172 | 0.82 | 208.4 |
| outdoor_forward_3 | 0.220 | 0.75 | 186.4 |
| outdoor_forward_5 | 0.256 | 3.32 | 318.9 |

The same path is used for the real clip and all six edits. The v3 edits are pixel-space
rotations; a rotation about the optical centre leaves the camera's translation in its own
frame unchanged. The one sequence where DPVO was run on the edited frames
(`indoor_forward_5`, C09's pilot) reproduces the real path to a rigid ATE of 4.98 units on a
105.6-unit path (4.7 %) with a 0.81° median step-direction difference for tilt removal, and
4.93 units / 0.82° for the yaw control (`edit_path_check.json`). One sequence only; the
general re-render sensitivity is unmeasured and needs DPVO on the edited frames.

## What this supports

* **Avoiding the second derivative works, up to a point.** The zero-derivative statistic
  separates removal (AUC 0.95) and mirror (0.89) where the old pilot separated nothing. This
  is the first pixel-path verdict statistic in the project with measurable power.
* **The path is still the binding error.** On the truth path the same code reaches 8/8 and
  7/8 with the same false-reject level; no statistic-side improvement can substitute for a
  better translation estimate.
* **One derivative is enough to lose removal detection** on this footage at this rate
  (1/8), which sharpens C09's diagnosis: it is not only the second derivative that hurts.
* **A free scale must not be the ruler for an edit test.** The tilt-removal degeneracy is
  structural, not numerical, and any future pixel-path statistic that fits a scale must fit
  it on a reference condition and hold it.
* **The conformal threshold is now the second binding constraint.** With four clusters it
  converts a 0.95-AUC statistic into 0.75 detection. Any future gate at this scale needs
  either more sequences or a pre-registered lower conformal level.
* **A better path by the bake-off's own metric does not fix it.** Stage 1 (`F02`) ran ViPE,
  which passes Track C's bake-off 10/10 without the Kalibr intrinsics and improves the median
  direction error (4.0 vs 4.2 deg), yet the same stage-0 statistics on its path give a worse
  false-reject rate (2/8) and the same detection (6/8 removal, 4/8 mirror). The verdict needs
  path *shape* at the frequencies the depicted lean excites, which the bake-off's median
  direction error does not rank.

## What this does not show

* That a different pixel path fails the same way: only DPVO's stored 150-frame paths were
  used. Stage 1 (MegaSaM, ViPE, a SLAM system) is where that question lives.
* Whether the edits would be detected if the path were re-estimated on the edited frames:
  the estimator's re-render sensitivity is measured on one sequence only.
* Anything about generated video: no ANWM clip was scored here.
* That V4 is useless: it is untestable on this flight population, not wrong.
* A calibrated V5: the depth band is a consistency check, and the 1.7× ruler bias is
  unexplained.

## Caveats

* **Eight clips, four sequences.** Every rate is n=8 with 1/8 granularity; the clustered
  intervals span half the unit line and are descriptive, not confidence intervals. One clip
  (`indoor_forward_9#01`) is both a false reject and a detected edit under every variant, so
  the unconditional detection rate is slightly flattering: conditional on the real clip
  passing, V2 detects 5/7 removals and 3/7 mirrors, V1 1/7 and 4/7.
* **Truncation and exclusions.** The primary population is 8 of the 25 real clips that have
  a pixel path: 8 further clips are truncated to 30 frames and 6 more are full but outside
  the four primary sequences; `indoor_forward_7` (3 clips) is excluded as a broken export;
  all outdoor clips abstain on the frozen confidence gate except `outdoor_forward_3#01`.
* **Path reuse across conditions**, mitigated only by the one-sequence check above.
* **Scale edits** are detectable here because amendment 1 fixed the ruler; the parent
  pre-registration's "at chance" expectation is superseded by that amendment and the
  observation is not used for the gate.
* **Threshold granularity** is part of the result: the gate's detection bar is missed by a
  statistic whose rank separation is high, because the LOSO conformal level is coarse with
  four clusters.
* **Truth-path rows** use the truth sampled at the same 15.07 Hz clip grid, not the 500 Hz
  table, so they isolate the path source at the clip rate rather than reproducing E's
  dense-truth ceiling exactly.
* **Yaw** comes from the pixel path's rotation in the estimated-attitude arm; GeoCalib cannot
  supply it. A rotation error there would enter both sides of the comparison.

## Reproduce

```
# 1. pull Track C's DPVO poses (once; read-only from the Modal volume)
pixi run python scripts/pull_volume.py awmv-c-data poses/dpvo data/pathrec/poses/dpvo
# 2. statistics (CPU, no spend) and the descriptive tables
python experiments/f01_stage0.py --out results/f01_stage0
python experiments/f01_analysis.py
python experiments/f01_tables.py
python experiments/f01_edit_path_check.py
```

Provenance for every number: `results/f01_stage0/summary.json` (code commit `3aa8a13`,
per-sequence pose sha256, population rules, thresholds, flags), `rows.csv` (one row per
sequence × clip × condition × arm × path × variant), `analysis.json` (AUCs and the per-clip
excitation table), `edit_path_check.json`. Cost: zero GPU spend; CPU only.
