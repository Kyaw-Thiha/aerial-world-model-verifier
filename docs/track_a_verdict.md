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
(pending)
<!-- GATE3:END -->

How to read it, beyond the tables:

- **The applicability boundary is per flight, not per dataset.** One flight
  (`flight-13a-trackRATM`) has a real-clip median near 68 deg: the estimator is wrong
  on that scene, and a tilt removal *lowers* its score because a systematic tilt partly
  cancels the random error. That is not a nuisance to fit; it is a flight where the
  verdict does not apply, and it is the reason the score is reported per flight before
  any pooled number.
- **A control flagged above the real rate is the honest limit of the method, not a
  bug to hide.** The yaw control rotates the frame by as much as the tilt edit without
  changing the tilt, so everything it scores above the real clips is the estimator's
  own response to content motion (section 8, item 1).
- **The nuisance fit is visible in the rows it moves.** On real clips the mount offset
  lands near zero with a modest fraction at the +/-7 deg bound; on the negatives it
  walks to the bound, which is the absorption the ablation table quantifies.

## 6. Task 3: Track B's generated clips

Status: the scoring path is built and exercised end to end on a fixture; Track B had
not delivered clips into `track-b/clips/` when this was written (their R2 prefix held
weights only). The script writes `not_available.json` rather than a placeholder number.

**Hypothesis, stated before the data.** ANWM's released frames are level by
construction, so a 4-DoF generator should depict tilt at its false-alarm floor
(0.57 deg median, section 2) while the commanded path requires a tilt that grows with
speed and turn rate. If that holds, the depicted-versus-required dose-response is flat
and every arguably testable clip will be flagged; if the generated frames do depict
tilt, the two curves track.

Reported per regime (A native, B real-flight replay, C factorial sweep), per dt in
{0.25, 0.667, 1.0} s, always as **curves over dt**: the release states no dt, so no
result here is assigned to one.

For each clip: depicted tilt per frame (Task 1 setup, 224x224, f = 112, pinhole), a
coherence statistic (median frame-to-frame movement of the estimated up direction — a
collapsed rollout jumps, and is marked not testable rather than failed), the implied
tilt from the commanded path at each dt (Savitzky-Golay 11 frames, `k_d` = 0.295), the
residual against the pre-registered thresholds, the Regime C dose-response against the
control set, and the spread of the clip score across seeds for the repeated paths.

Interface: `experiments/phase3a_anwm_specs.py` reads Track B's manifests and writes one
spec per clip; `remote/a_geocalib_p3.py --job anwm` runs the estimator on the 224x224
frames; `experiments/phase3a_anwm_scoring.py` scores the result. The required tilt I
compute at 1 s is cross-checked against the `required_tilt_1s_kd_deg` column of their
index rather than assumed equal.

<!-- ANWM:BEGIN -->
(pending)
<!-- ANWM:END -->

## 7. Task 4: why DPVO's translation failed

Not run. The workstation has 4.1 GB free and no calibration-bearing sequence that small;
UZH-FPV's own release is tens of GB and the HF derivative at 128x128 does not carry a
trustworthy calibration, which is the one property the test needs. The plan stands: one
sequence of UZH-FPV (or a Blackbird replay) with its shipped intrinsics, DPVO's
oracle-scale translation correlation against truth. If it correlates there, the TII
calibration is the suspect (fx/fy = 0.749 is a pinhole fit of a 175 deg fisheye and its
polynomial is only invertible to 56 deg); if it does not, the method is.

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
