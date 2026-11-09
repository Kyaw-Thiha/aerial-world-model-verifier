# Track A: the verdict test (Phase 3a, handoff 04)

Status: pre-registration committed before any edited clip was scored
(`configs/verdict_preregistration.yaml`, commit `d6b2003`, amendment 1 `addd9ec`).
Every number below is from a `results/` run; the run id and git hash are in each
`metrics.json`.

## 0. What changed in the plan, and what this test now claims

DPVO's translation is unusable (path 41 m against mocap's 479 m), so the earlier
pixel-only implied-tilt route cannot carry a verdict. The test therefore runs in
**commanded-path mode**: the path is known (mocap for TII, the commanded action
sequence for ANWM) and only the *depicted* attitude comes from pixels.

The claim being tested: **does the verdict separate a real banking flight from the same
frames with the tilt removed, scaled or mirrored?** Two things it deliberately does
not claim: that heading is observable from gravity (it is taken from the path), and
that DPVO's translation can be replaced (it is not used here).

## 1. Oracle ladder (Task 0.1)

The rows that were missing from the Phase-1/2 summary, with the definition of each (a
rung is either an accuracy number against mocap, or a self-consistency number, and the
two are never mixed):

| # | rung | number | what it is |
| --- | --- | --- | --- |
| 1 | mocap position + mocap attitude | **1.24 deg** median residual | physics floor (Phase 0) |
| 2 | VO position (oracle scale) + mocap attitude | **28.97 deg** RMSE | accuracy, position estimated |
| 3 | mocap position + estimated attitude | **8.73 deg** tilt RMSE | accuracy, attitude estimated |
| 4 | VO position (oracle scale) + estimated attitude | **2.75-8.30 deg** residual | **self-consistency**, not accuracy (shared VO frame) |
| 5 | fully estimated, depth scale | not run | no MoGe-2 pass |
| 6 | fully estimated, fitted scale | **29.59 deg** RMSE | accuracy |
| 7 | commanded-path mode (this test) | see section 5 | mocap position + pixel attitude, heading from the path |

Reading: replacing the position source costs ~28 deg and replacing the attitude source
costs ~7 deg. The position is the binding error, which is why this test gives it away
and spends its budget on the depicted side.

## 2. Noise floor (Task 1): GeoCalib on ANWM's level frames

ANWM's released frames are exactly level (roll = pitch = 0.0 in every frame of 33
sampled test trajectories), so the estimated tilt on them is pure false alarm. 186
frames from 12 test trajectories, pinhole route, the focal implied by the declared 90
deg field:

| size | median | p90 | p99 | max | > 5 deg | > 10 deg |
| --- | --- | --- | --- | --- | --- | --- |
| 512x512 (f = 256) | **0.57 deg** | 1.62 | 4.20 | 4.49 | 0.5% | 0.0% |
| 224x224 (f = 112) | **0.57 deg** | 1.52 | 4.20 | 4.49 | 0.0% | 0.0% |

The estimator reads essentially zero, at both resolutions, with a sub-degree median and
a 1.6 deg p90. So the false-alarm floor for an ANWM verdict is small compared with the
8-35 deg tilt a level-camera violation implies: **if ANWM's clips depict the tilt its
paths require, this estimator will see it.**

Run: `results/anwm_scoring/<run>/noise_floor.json`.

## 3. Lean direction re-scored with windowed azimuth alignment (Task 0.2)

| window | 0.5 s | 1 s | 2 s | 3 s | 4 s | 5 s |
| --- | --- | --- | --- | --- | --- | --- |
| median flight lean error | **4.4 deg** | 10.9 | 19.3 | 84.6 | 114.6 | 89.7 |

One azimuth per clip (the Phase 1b number) is 85.5 deg. See `docs/track_a_report.md`
section 6b for the full table and the per-flight figure. The error is fast, not a slow
drift: the 3-5 s windows the plan expected are worse than one per clip, while sub-second
anchoring brings it inside the Gate 1 target. Commanded-path mode supplies that
anchoring for free from the path's yaw, which is why the verdict fits no azimuth.

## 4. Protocol, as pre-registered

- 4 s non-overlapping clips, stride 10 (~12.04 Hz), split by flight, leave-one-flight-out.
- implied axis: mocap position, native rate, Savitzky-Golay 0.37 s / poly 3, `k_d` = 0.295
  **fixed** (never fitted).
- depicted axis: GeoCalib `simple_divisional` on the raw fisheye; body attitude from the
  measured gravity (gravity gives roll and pitch, the path gives yaw).
- residual: `r(t) = angle(z_dep(t), z_imp(t + tau))`, score = median over testable frames;
  heading-free scalar tilt reported alongside.
- nuisances, one fit per clip: mount uptilt +/- 7 deg, lag +/- 150 ms on a 25 ms grid.
- abstention: < 12 frames with implied tilt >= 10 deg, testable fraction < 0.20, or fewer
  than 40% usable depicted frames.
- thresholds: conformal 95th percentile of held-out real scores, per speed bin, fitted on
  the fit flights; amendment 1 sets the fit set to the five non-trackRATM flights and
  reports the all-flights variant as well.
- edits (pure camera rotation about the camera centre, raw fisheye, equidistant model
  `r = f theta` with f from the declared 175 deg field): tilt x 0, 0.5, 1.5, mirrored
  roll, identity null edit, and a yaw-rotation control matched in warp magnitude.

## 5. Gate 3: the numbers

Rendered from `results/phase3a_verdict/<run>/metrics.json` by
`experiments/phase3a_gate3_tables.py --inject docs/track_a_verdict.md`, so every number
here is regenerated rather than typed.

<!-- GATE3:BEGIN -->
run `20261109T090957Z_bc3a398` · git `bc3a398` · prereg `b8fd70130f17` (bc3a398) · 600 clip-condition records

### Real clips

- held-out acceptance (non_trackratm threshold policy): **0.711** over 83 clips
- pooled acceptance: 0.711
- acceptance, non-trackRATM: 0.967 (n=60); trackRATM: 0.043 (n=23)
- abstention: overall 0.170, trackRATM 0.207

### Thresholds (conformal 95th percentile)

| speed bin group | bins merged | n | threshold (deg) |
| --- | --- | --- | --- |
| 0-2+2-5 | 0-2, 2-5, 5-10, 10-20, 20-99 | 60 | 5.76 |
| pooled | - | 60 | 5.76 |

Unmerged per-bin counts: 0-2 n=4, 2-5 n=37, 5-10 n=18, 10-20 n=1, 20-99 n=0

### Negatives

| condition | family | n | rejection | AUROC | real score | negative score |
| --- | --- | --- | --- | --- | --- | --- |
| tilt_k0 | tilt_removal | 83 | 0.373 | 0.918 | 2.91 | 18.90 |
| tilt_k05 | tilt_scaling | 83 | 0.373 | 0.803 | 2.91 | 8.80 |
| tilt_k15 | tilt_scaling | 83 | 0.349 | 0.743 | 2.91 | 5.06 |
| mirror_roll | mirrored_roll | 83 | 0.386 | 0.926 | 2.91 | 26.29 |
| yaw_control | control | 83 | 0.349 | 0.617 | 2.91 | 3.81 |

Pooled AUROC over the three tilt families: **0.821**

### Minimum detectable violation

| edit | k | injected tilt change (deg) | rejection |
| --- | --- | --- | --- |
| tilt_k0 | 0.0 | 27.2 | 0.373 |
| tilt_k05 | 0.5 | 13.6 | 0.373 |
| tilt_k15 | 1.5 | 13.6 | 0.349 |

Smallest violation rejected 80% of the time: **- deg**

### Delay negatives (analysis, no pixel edit)

| delay (ms) | n | score median | note |
| --- | --- | --- | --- |
| 100 | 83 | 5.49 | inside the +/-150 ms nuisance bound |
| 200 | 83 | 7.53 | outside the bound |
| 300 | 83 | 9.71 | outside the bound |

### Retime negatives (analysis)

| rate | n | score median |
| --- | --- | --- |
| 0.5 | 81 | 18.13 |
| 2.0 | 81 | 19.20 |

### Real clip score by flight (the applicability boundary)

| flight | n testable | median | p90 | max | abstain |
| --- | --- | --- | --- | --- | --- |
| flight-01a-ellipse | 2 | 7.07 | 9.10 | 9.61 | 0.600 |
| flight-01p-ellipse | 26 | 2.09 | 3.02 | 3.64 | 0.000 |
| flight-04a-ellipse | 3 | 2.85 | 5.18 | 5.76 | 0.571 |
| flight-07a-lemniscate | 3 | 5.33 | 40.08 | 48.77 | 0.571 |
| flight-07p-lemniscate | 26 | 2.60 | 3.46 | 3.60 | 0.000 |
| flight-13a-trackRATM | 5 | 68.11 | 78.59 | 79.65 | 0.545 |
| flight-13p-trackRATM | 11 | 7.05 | 8.64 | 8.95 | 0.000 |
| flight-16p-trackRATM | 7 | 8.42 | 9.19 | 9.37 | 0.000 |

### Diagnostics

| speed bin | n | real median | real p90 |
| --- | --- | --- | --- |
| 0-2 | 4 | 1.97 | 3.29 |
| 10-20 | 18 | 8.15 | 67.45 |
| 2-5 | 38 | 2.23 | 3.09 |
| 5-10 | 23 | 3.44 | 9.34 |

| condition | n | score median | uptilt offset median | at bound | lag median (ms) | at bound | abstain |
| --- | --- | --- | --- | --- | --- | --- | --- |
| identity | 83 | 2.91 | -1.00 | 0.133 | -50.0 | 0.096 | 0.170 |
| tilt_k0 | 83 | 18.90 | 7.00 | 0.867 | -150.0 | 0.699 | 0.170 |
| tilt_k05 | 83 | 8.80 | 7.00 | 0.614 | -125.0 | 0.349 | 0.170 |
| tilt_k15 | 83 | 5.06 | -7.00 | 0.904 | -25.0 | 0.108 | 0.170 |
| mirror_roll | 83 | 26.29 | -3.00 | 0.157 | -150.0 | 0.795 | 0.170 |
| yaw_control | 83 | 3.81 | -3.00 | 0.157 | -50.0 | 0.084 | 0.170 |

### Score by flight and condition (median over testable clips, deg)

| flight | testable/total | identity | tilt_k0 | tilt_k05 | tilt_k15 | mirror_roll | yaw_control |
| --- | --- | --- | --- | --- | --- | --- | --- |
| flight-01a-ellipse | 2/5 | 7.07 | 35.46 | 20.57 | 15.92 | 18.65 | 11.42 |
| flight-01p-ellipse | 26/26 | 2.09 | 11.61 | 4.54 | 4.22 | 17.03 | 2.59 |
| flight-04a-ellipse | 3/7 | 2.85 | 23.09 | 18.80 | 17.02 | 12.12 | 9.77 |
| flight-07a-lemniscate | 3/7 | 5.33 | 30.29 | 16.82 | 8.99 | 44.17 | 5.81 |
| flight-07p-lemniscate | 26/26 | 2.60 | 13.91 | 6.40 | 3.92 | 22.25 | 2.77 |
| flight-13a-trackRATM | 5/11 | 68.11 | 55.30 | 31.55 | 56.75 | 66.36 | 46.94 |
| flight-13p-trackRATM | 11/11 | 7.05 | 48.97 | 31.16 | 15.91 | 53.23 | 15.80 |
| flight-16p-trackRATM | 7/7 | 8.42 | 56.14 | 33.52 | 22.71 | 61.87 | 14.31 |

### Gate 3

- real_acceptance_at_least_0.95: **False**
- auroc_above_0.9_tilt_families: **False**
- control_accepted: **False**
<!-- GATE3:END -->

### Gate 3 against the pre-registered criteria

| criterion | target | measured | verdict |
| --- | --- | --- | --- |
| real-clip acceptance, held out | >= 0.95 | 0.711 (primary policy); **0.967 on the five non-trackRATM flights**, 0.043 on trackRATM | **fail** |
| AUROC, tilt removal | > 0.9 | **0.918** (95% CI 0.76-0.98) | pass |
| AUROC, tilt scaling | > 0.9 | 0.803 (k=0.5), 0.743 (k=1.5) | **fail** |
| minimum detectable violation (80% rejection) | a curve | none of 27.2 / 13.6 / 13.6 deg reaches 80% under the primary policy | **fail** |
| Spearman, score vs injected tilt change | > 0.8 | **0.82** | pass |
| null edit / control accepted at the real rate | within 5 pp | identity is exact (it *is* the real condition); the yaw control is rejected 35% vs 10% | **fail** |

**The score separates; the threshold is what fails.** The diagnosis, in the order the
numbers point at:

1. **A five-clip tail, not discrimination, is binding.** The real clips are tight —
   median 2.9 deg, p90 8.9 — and then there are **5 of 83 clips above 40 deg** (49-80),
   four of them on `flight-13a-trackRATM` and one on `flight-07a-lemniscate`. Those five
   push the 95th percentile from about 9 to 44.9, and this is threshold-free: the 20th
   percentile of *any* negative family (11.2 deg at best) does not clear the real p95,
   so **an 80%-rejection operating point does not exist for any family**, whatever the
   threshold rule. Tilt removal still has AUROC 0.918 with a median of 18.9 against 2.9;
   only 37% of its clips clear the per-fold thresholds because the folds inherit the
   tail.
2. **One broken flight moves the threshold by an order of magnitude.** The three
   threshold policies span acceptance 0.711 / 0.952 / 0.904 with thresholds 5.76 /
   67.16 / 9.37 deg:

   | policy (fit set) | threshold | acceptance | tilt removal | half tilt | x1.5 | mirror | yaw control |
   | --- | --- | --- | --- | --- | --- | --- | --- |
   | `non_trackratm` (primary) | 5.76 | 0.711 | 0.37 | 0.37 | 0.35 | 0.39 | 0.35 |
   | `all_flights` | 67.16 | 0.952 | 0.06 | 0.06 | 0.06 | 0.11 | 0.06 |
   | `all_but_13a` (sensitivity) | 9.37 | 0.904 | **0.89** | 0.45 | 0.33 | **0.96** | 0.31 |

   The `all_flights` policy meets the 95% acceptance target only because the same five
   clips become the threshold, which makes the test vacuous. The policy the data support is `all_but_13a`: 90% acceptance
   and 89% rejection of full tilt removal, 96% of mirrored roll, 45% of half tilt.
   It is reported as a sensitivity analysis, not swapped in for the pre-registered
   primary.
3. **The control is flagged above the real rate, and the mechanism is measured.** A
   rotation about world up cannot change the tilt, yet the yaw control's median score is
   3.8 deg against 2.9 for the real clips and 35% of controls are rejected. Per frame,
   the same rotation moves the estimated tilt by 4.75 deg (median) on the frames it
   rotates more than 10 deg, correlated 0.50 with the rotation angle (section 8). This
   is the estimator's view dependence, and it is the floor under the detectable
   violation: nothing below ~5 deg can be separated from it.
4. **Half-tilt is where it stops.** The scaling edits remove 13.6 deg of tilt, and the
   nuisances absorb 4-6 deg of that (section 9) while the artifact floor is another
   ~5 deg. 0.45 rejection for half tilt is the honest consequence; the 80%-rejection
   frontier sits at the full-removal magnitude, not below it.
5. **Delays are absorbed inside the bound, as designed.** The pre-registration bounds
   the lag at +/-150 ms, and a 100 ms delay is absorbed (median score 5.5 deg against
   2.9 real). 200 and 300 ms leave the bound and score 7.5 and 9.7 deg — detectable in
   principle, but below the levels the unstable thresholds can resolve. Retiming (0.5x,
   2x) scores 18.1 and 19.2 deg: heavier than the delays, light against a full tilt
   removal, which is what a 0.5-s window over a smooth trajectory would predict.

**Where the fix is.** The tail is the target: 5 clips, and every one of them is a clip
where more than half the frames carry a wrong estimate. A depicted-consistency
abstention (a clip whose estimated tilt disagrees with its own temporal median for most
frames is not testable) would remove them without touching a negative, and the
`all_but_13a` row shows what that buys: threshold 9.4, 90% acceptance, 89% rejection of
full tilt removal. That rule was not in this pre-registration, so it is the first item of
the next handoff rather than something applied here.

**What this means for the claim.** On the five flights where the depicted estimator
works, the verifier ranks a real clip below the same clip with its tilt removed 92% of
the time and can reject 89% of such edits at a 90%-acceptance operating point. It cannot
yet certify the 95% acceptance the RQ2 criterion asks for, and it cannot detect a
half-tilt or a mirrored-roll-scale violation at 80%. The binding limits are the
threshold's sample size (60 real clips from 5 flights) and the estimator's
view-dependence, not the residual's formulation.

## 8. Threats to validity, measured where possible

These are the mechanisms that could make the verdict look better than it is. Each one
has a measurement attached; none is left as a caveat in prose.

1. **The estimator is view-dependent, so the edits themselves move the reading.** A
   rotation about world up cannot change the tilt, so every degree the depicted tilt
   moves under the yaw control is an artifact. On 04a the per-frame tilt change is
   median 0.74 deg over all frames but **4.75 deg on the frames whose content was
   rotated by more than 10 deg**, and it correlates 0.50 with the rotation magnitude
   (`evaluation.artifact_floor`). This sets a floor of roughly 5 deg under the minimum
   detectable violation: a violation smaller than that cannot be separated from the
   pipeline's own response to content motion.
2. **Nuisances absorb part of a violation, by construction.** The mount-uptilt term is
   bounded at +/-7 deg, so dropping it must remove up to 7 deg of any tilt violation. The
   ablation measures it: on the same clips, tilt x 0.5 scores 26.1 deg with no nuisance,
   19.3 with the lag only, 18.8 with both. The cost is real and bounded; the abstention
   and the reported MDV are what remain.
3. **Excitation and estimator reliability are anti-correlated on the gate-dense track.**
   The clips that survive abstention are the ones with the most acceleration, which on
   trackRATM is exactly when the view tips away from the informative structure. That is
   why that flight's real clips score 68 deg and why it is reported separately.
4. **Heading is given.** The residual uses the path's yaw. A verifier on real footage
   without a heading reference would have to drop the lean term and keep the scalar
   tilt, which cannot see a mirrored roll. Stated as the limit of the claim, not a
   fitted parameter.
5. **The equidistant model is an approximation.** The identity warp is exact and the
   yaw control carries the same model error as the treatments, so a model-induced bias
   common to both is measured by the control rather than assumed absent.
6. **Small n.** Five threshold-fitting flights, and the per-speed-bin thresholds merge to
   one group at this sample size (reported with the merge chain). Intervals on AUROC and
   rejection rates resample whole **flights**, not clips, because clips inside a flight
   share scene, light and estimator state.

## 9. Nuisance ablations

Same per-frame estimates, four fits, from `experiments/phase3a_ablations.py`
(`results/phase3a_verdict/<run>_ablations/ablations.json`). Score medians in degrees:

| condition | both nuisances | no uptilt | no lag | neither | absorbed by the nuisances |
| --- | --- | --- | --- | --- | --- |
| identity (real) | 2.9 | 4.0 | 4.1 | 4.4 | 1.5 |
| tilt x 0 | 18.9 | 20.7 | 21.6 | 23.0 | 4.1 |
| tilt x 0.5 | 8.6 | 10.3 | 11.0 | 12.6 | 4.0 |
| tilt x 1.5 | 5.0 | 6.4 | 7.3 | 11.3 | **6.2** |
| mirrored roll | 26.4 | 26.7 | 28.4 | 28.5 | 2.1 |
| yaw control | 3.8 | 5.0 | 5.4 | 6.2 | 2.4 |

Reading:

- The nuisances are **needed**: they take the real-clip median from 4.4 to 2.9 deg, and
  the uptilt term is the one that does it (the lag term alone leaves 4.1).
- They do **not** absorb the large violations: 4.1 deg of a 27.2 deg tilt removal, and
  2.1 of the mirrored roll.
- They **do** absorb most of the smallest one: 6.2 of the 13.6 deg change at k = 1.5,
  i.e. 46%. That is the expected cost of a +/-7 deg mount bound and it is why the
  half-tilt frontier stops where it does.
- At the verdict level the ablations barely move: acceptance 0.711 with both nuisances
  against 0.723 with neither, AUROC 0.822 against 0.833. The ranking is robust to the
  nuisance choice even though the per-clip score is not.
