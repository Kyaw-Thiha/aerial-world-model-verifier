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

## 6. Task 3: Track B's generated clips

(filled from `results/anwm_scoring/<run>/` when the clips land)

## 7. Task 4: why DPVO's translation failed

Not run. The workstation has 4.1 GB free and no calibration-bearing sequence that small;
UZH-FPV's own release is tens of GB and the HF derivative at 128x128 does not carry a
trustworthy calibration, which is the one property the test needs. The plan stands: one
sequence of UZH-FPV (or a Blackbird replay) with its shipped intrinsics, DPVO's
oracle-scale translation correlation against truth. If it correlates there, the TII
calibration is the suspect (fx/fy = 0.749 is a pinhole fit of a 175 deg fisheye and its
polynomial is only invertible to 56 deg); if it does not, the method is.
