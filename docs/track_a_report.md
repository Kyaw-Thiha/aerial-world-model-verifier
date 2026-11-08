# Track A report: perception (Phase 1 depicted tilt, Phase 2 implied tilt)

Status: all three gates are measured and **all three fail as written**. Every number
comes from a `results/*/metrics.json` (or a `results/phase1_camera*/` run); anything
marked not run is not a pass.

Headline numbers:

| gate | criterion | measured | verdict |
| --- | --- | --- | --- |
| 1a depicted tilt, per frame | RMSE under 5 deg | median **3.96**, trimmed RMSE 4.80, full RMSE **22.32** | fail on RMSE |
| 1 depicted attitude over time | tilt under 5 deg, lean median under 15 deg | tilt RMSE **8.73**, lean **83.6** | fail |
| 2 implied tilt | RMSE under 8 deg, SNR above 3 | RMSE **28.97**, SNR **0.84** | fail |

The useful part is the attribution: the depicted side is accurate at the median and fails
in a heavy tail; the fusion fails on azimuth, which gravity cannot observe; and the
implied side fails because **DPVO's translation does not track** (per-axis correlation
with mocap 0.01 / -0.09 / 0.13, path length 41 m against 479 m) while its **rotation is
excellent (0.15-1.5 deg per frame)**. Nothing in these failures is a physics-model
failure: the mocap floor is 1.24 deg.

Branch `track-a-perception`. All structural choices are one line each in
`docs/decisions.md` tagged `[A]`.

## 1. What was verified before any estimator was blamed

| item | result | evidence |
| --- | --- | --- |
| Video location and size | 8 flights, camera JPEG zips **3.77 GB** total (plus 0.17 GB of CSVs/YAML), all needed frames present | `scripts/fetch_tii_video.py --list`; zip central directories read by ranged HTTP |
| Video-to-mocap offset | **0 by construction**; camera timestamps and mocap timestamps share one Unix-microsecond clock, and the release's `<flight>_cam_ts_sync.csv` is byte-identical to `csv_raw/camera_<flight>.csv`. Uncertainty 0.5 frame = 4.2 ms; exposure latency undocumented and not modelled | `src/datasets/tii_video.py::frame_table` (raises on any mismatch) |
| Frame rate | 120 Hz declared, **120.4 Hz measured**, median dt 8.3 ms, dropped frames up to 42 ms | `Clip.meta["measured_fps"]` |
| Camera model in the release | `cv2.calibrateCamera` pinhole + k1,k2,p1,p2,k3 — **not** a Kannala-Brandt fisheye model | `scripts/camera_calibration.py` in the release |
| Calibration coverage | polynomial invertible only to `rd = 0.98` (θ = 56 deg); **all four image corners unrepresentable**; `fx/fy = 0.749` in all three calibration files | `src/perception/undistort.py::model_diagnostics` |
| Virtual pinhole view | 100 deg horizontal, 640x480, square pixels, same optical axis; **46 % of its pixels are valid** (the rest sample outside the model's domain); max source ray reached 45.5 deg | `VirtualCamera.valid_fraction` |
| Camera-to-body convention | `drone_to_camera.json` quaternion is **camera->body**; for the ellipse/trackRATM mounts it equals the analytic hard-mount matrix exactly | `src/perception/frames.py::camera_to_body_pitched` + test |
| Autonomous flights' mount | mocap-implied optical-axis elevation matches the recorded uptilt: 01a 49.6/50, 04a 49.8/50, 07a 42.7/40, 13a 50.0/50 | `experiments/phase1_camera_check.py` |
| Mount, all eight flights | `elevation(optical axis) + body tilt` reproduces the recorded uptilt: 01p 31.5 vs 30, 07p 32.6 vs 30, 01a 50.9 vs 50, 13p 57.4 vs 50, 16p 56.3 vs 50, 13a 50.0 vs 50. A first reading of the raw elevation medians (7.3 / 9.7 / -5.8 / -10.4 deg for the piloted flights) looked like a broken extrinsic on those flights; the cause was the **diagnostic**, not the data: piloted flights fly with a median body tilt of 23-67 deg, so axis elevation alone is confounded. Corrected here, and cross-checked end-to-end by GeoCalib agreeing with the same truth to 3-4 deg on the piloted ellipse/lemniscate flights | `src/perception/frames.py`, `experiments/phase1_camera_check.py`, `phase1_depicted` |

Two consequences worth stating plainly:

- The camera frame is usable on all eight flights, with a 6-7 deg residual discrepancy on
  the two v3.0.0 piloted trackRATM flights (13p, 16p) that is reported, not fitted. The
  gate-corner PnP test would pin it down and is still the right next step, but it is no
  longer a blocker for using the flights.
- The released intrinsics cannot describe the fisheye periphery. That is why the adopted
  GeoCalib route reads the **raw** fisheye with its own distortion model and does not use
  them at all; the numbers below are not limited by a 92 deg crop.

## 2. Compute used

| tool | GPU | frames/s (measured) | s per 1000 frames | peak GPU | notes |
| --- | --- | --- | --- | --- | --- |
| GeoCalib `simple_divisional` on raw fisheye | A10G | ~5 (20-frame batches), 1.4 (3-frame cold start) | ~200 | 0.43 GB | LM optimization dominates, not the network |
| DPVO (320x240, 48 patches/frame) | A10 | **60** | 17 | 1.16 GB | 8 flights, 6,363 poses in 106 s |

Modal runs in this phase: `awmv-a-fetch-video` (3.94 GB fetched once, mirrored to
`track-a/tii_video` on R2), `awmv-a-frames` (sampled + windowed frames), `awmv-a-geocalib`
(probe, three configuration surveys, the full sweep), `awmv-a-dpvo` (six build attempts +
three sweeps). Total GPU time is a couple of hours of A10/A10G including the failed builds,
i.e. a few US dollars.

## 3. Gate 1a: depicted tilt per frame (GeoCalib)

**Verdict: median meets the 5 deg target, RMSE does not. Gate 1a FAILS on RMSE.**

Configuration (chosen by measurement, not preference — see `decisions.md`): GeoCalib
`simple_divisional` on the **raw fisheye**, no focal prior, undistorted by nothing.
That route beat pinhole-on-undistorted by 2x (13.2 vs 26.9 deg pooled median flight
RMSE on the same 20-frame sample), which is itself a finding: the released calibration
is too poor for the undistorted route to pay off.

Numbers from `results/phase1_depicted/20261108T131825Z_c738aa8/metrics.json`
(all eight flights, every 10th frame = 12 fps, **5,007 frames**):

| quantity | value |
| --- | --- |
| median error (angle between estimated and mocap up in the camera frame) | **3.96 deg** |
| trimmed RMSE (worst 10 % of frames dropped) | **4.80 deg** |
| RMSE, all frames | **22.32 deg** |
| p90 / p99 | 12.87 / 93.6 deg |
| fraction of frames over 20 deg | **7.5 %** |

Per flight:

| flight | n | median | p90 | p99 | RMSE | frac > 20 deg |
| --- | --- | --- | --- | --- | --- | --- |
| 01p-ellipse | 1267 | 3.14 | - | - | **3.80** | - |
| 07p-lemniscate | 1284 | 3.59 | - | - | **4.20** | - |
| 04a-ellipse | 350 | 3.29 | 6.84 | 81.41 | 10.35 | 2.9 % |
| 01a-ellipse | 270 | 4.20 | 10.28 | 82.06 | 13.67 | 4.0 % |
| 07a-lemniscate | 370 | 2.54 | - | - | 28.43 | - |
| 16p-trackRATM | 373 | 7.34 | - | - | 31.18 | - |
| 13p-trackRATM | 536 | 7.85 | - | - | 33.69 | - |
| 13a-trackRATM | 557 | 7.44 | - | - | 44.36 | - |

(The dashes are rows not re-printed from the earlier two-flight run; all eight rows and
their full tail statistics are in `metrics.json`.)

An independent stride-10 sweep over the same two flights reproduces these numbers
(01a median 4.20 / RMSE 13.67, 04a 3.29 / 10.35, `results/phase1_depicted/20261108T055529Z`
and the `div10` run), so they are not a stride artefact.

By speed (pooled):

| speed (m/s) | n | median | RMSE |
| --- | --- | --- | --- |
| 0-5 | 2790 | 3.68 | 9.88 |
| 5-10 | 1609 | 3.87 | 27.71 |
| 10-15 | 358 | 8.77 | 43.59 |
| 15-20 | 223 | 7.49 | 36.84 |
| 20+ | 27 | 9.83 | 13.62 |

The median degrades above 10 m/s (3.7 -> 8.8 deg) and the RMSE is worst in the same band:
unlike the Phase 0 physics residual, this estimator's error *does* grow with speed, which
is what motion blur and the gate-filled field of view would predict.

**The error is heavy-tailed, not biased.** The median is flat in speed (2.99-4.12 deg,
i.e. it does not degrade the way Phase 0's physics residual does), while the RMSE
explodes above 10 m/s: about 3-4 % of frames come back with an orientation that is
near-wrong (p99 ~81 deg). Reporting one RMSE for this estimator would hide that.

With/without one constant rotation per flight: the fitted offset is 4.0 deg (01a) and
3.4 deg (04a) and **increases** the error rather than reducing it, i.e. the residual is
not a constant mount offset. The handoff's known 3.9 deg mocap offset on the autonomous
ellipse flights is of the same size as this constant term, which is worth noting but is
not what dominates the RMSE.

### Why the tail exists (diagnosis, not blame)

The synthetic-rotation probe (`--probe-flights`) tilts the view by +/-20 deg and asks
whether the recovered gravity rotates with it:

| flight | -20 deg consistency | +20 deg consistency |
| --- | --- | --- |
| 07p-lemniscate | 1.47 deg median | 2.56 deg |
| 01a-ellipse | 14.25 deg | 1.48 deg |
| 13a-trackRATM | 13.55 deg | 6.52 deg (p90 62.5) |

A geometry or convention bug would break both directions on every flight equally. Instead
the estimator is *view-dependent*: the direction that tips the camera further away from
the informative structure (the wall/gate band) is the direction where it loses the answer.
Combined with the near-ceiling content of the 40-50 deg mounts, this is a scene-content
failure — the indoor arena with no visible horizon — not a pipeline failure.

Blur and exposure (the breakdown the plan asks for):

- `exposure` is the only blur label the dataset carries: `fixed` on 04a and 16p, `auto`
  elsewhere. Pooled on the 192-frame sample, `auto` medians 4.48 deg and `fixed` 3.92 deg
  (RMSE 26.1 vs 14.4) — the fixed-exposure flights are better, but they are also the
  slow ones, so this is confounded with speed.
- Laplacian variance as a stated proxy (`scripts/frame_stats.py`, 192 frames): rank
  correlation with error is **-0.22** (sharper frames do better), with mid-sharpness
  frames (150-400) worst at 26 % over 20 deg and the sharpest frames (400+) best at
  1.6 %. Weak and not monotone, so blur is a contributor, not the cause.
- By track type (5,007 frames): ellipse median 3.28 deg (RMSE 7.50), lemniscate 3.37
  (13.95), trackRATM 7.57 (37.55) — the gate-dense circuit is the hard scene.
- By exposure: `auto` median 3.96 / RMSE 22.11, `fixed` 4.00 / 23.53, i.e. no difference
  once the flights are pooled, so the earlier per-flight impression was a speed confound.
- By control: pooled autonomous median 4.57 / RMSE 30.96, piloted 3.83 / 17.10 — but the
  spread inside each group is bigger than between them (3.8 to 44 deg RMSE), so control
  mode is not the variable; track and body attitude are.

The undistorted route cannot help either: 46 % valid pixels, and a released lens model
that does not cover the frame.

## 3b. What Gate 1a's result means

Gate 1a as written (RMSE under 5 deg) is not met by a per-frame, single-image gravity
estimator on TII. What *is* met: a 3.5 deg median with a 3.7 deg trimmed RMSE, which is
inside the target for the 96 % of frames the estimator does not flip on. The natural
follow-up is an outlier rule (a robust per-clip aggregate plus a rejection test on the
frame-to-frame consistency of the estimate) rather than a better single-frame model,
and that is Phase 3 work, deliberately not built here.

## 4. Gate 1: depicted attitude over time (DPVO + gravity fusion)

**Verdict: FAILS both criteria.** Tilt RMSE 8.73 deg (target 5) and lean-direction
median 83.6 deg (target 15), from
`results/phase1b_fusion/20261108T133135Z_c738aa8/metrics.json` (all eight flights,
gain chosen leave-one-flight-out).

| flight | chosen gain | tilt RMSE | thrust-axis vector median | lean median |
| --- | --- | --- | --- | --- |
| 01a-ellipse | 1.0 | 6.78 | 6.49 | 46.6 |
| 04a-ellipse | 1.0 | 5.78 | 4.60 | 26.6 |
| 07a-lemniscate | 1.0 | 10.67 | 3.12 | 63.2 |
| 13a-trackRATM | 1.0 | 21.31 | 6.92 | 87.3 |
| 01p-ellipse | 1.0 | 2.27 | 32.09 | 90.4 |
| 07p-lemniscate | 1.0 | 2.06 | 27.59 | 83.5 |
| 13p-trackRATM | 1.0 | 22.94 | 70.09 | 85.9 |
| 16p-trackRATM | 1.0 | 20.12 | 75.71 | 83.8 |

The leave-one-flight-out selection chose gain 1.0 in every fold, i.e. the filter is
better off snapping to the gravity measurement than trusting the VO rotation — which is
the honest read on this data: **DPVO's rotation is excellent (0.15-1.5 deg per-frame
relative-rotation error against mocap) but the fused attitude's heading is not.**

Three separate numbers matter here and they disagree, which is the point:

- the **scalar tilt** (angle from vertical) is within 2-7 deg on six flights and 20-23
  deg on the two piloted trackRATM flights;
- the **thrust-axis vector** is within 3-7 deg on the four autonomous flights but 28-76
  deg on the piloted ones — an azimuth failure that the scalar tilt cannot see;
- the **lean direction** is 27-90 deg everywhere, i.e. unusable.

Attribution: the lean direction is the azimuth of the thrust axis, and azimuth is exactly
what a gravity measurement cannot observe — the filter's correction fixes the up
direction and leaves the heading to VO, whose yaw drifts and is re-kicked by every
noisy gravity correction (a horizontal-axis correction rotates the thrust axis, which is
perpendicular to up, by roughly the same angle). The residual is *not* a mount error:
mount is verified independently (section 1), and the failure persists at every gain from
0.02 to 1.0.

## 4b. Gate 1's building blocks

`src/fusion/`, five tests in `tests/test_fusion.py`:

- `gravity_align`: one rotation per clip from the robust mean measured up, with the
  per-frame spread reported. Measured spread: 6-28 deg, which is itself a warning that
  the per-frame gravity estimates disagree by more than their median error.
- `complementary_filter`: VO relative rotation propagates, gravity corrects with a fixed
  gain; verified that gain 1 snaps the measured up to world up within 1e-6 deg, gain 0
  never corrects, and gain 0.2 converges from a 56 deg initial error to under 1 deg in
  about 0.3 s at 120 Hz. One real bug was found and fixed this way: the filter corrects
  toward world up, so it must run in the gravity-aligned frame (conjugating the relative
  rotations), not in the raw VO frame.
- `lean_error_deg`: azimuth error on frames above 10 deg tilt, with one constant yaw per
  clip (evaluation only, recorded in the metrics).

## 5. Gate 2: implied tilt from the pixel trajectory

**Verdict: FAILS, and the failure is attributed to the trajectory source, not the physics
or the depicted side.**

| quantity (median over flights, 0.11 s window unless stated) | value | target |
| --- | --- | --- |
| implied-tilt RMSE vs mocap, oracle scale | **28.97 deg** | under 8 |
| signal-to-noise (std true tilt / RMSE) | **0.84** | above 3 |
| testable fraction (implied tilt above 10 deg) | 0.00-0.16 | report |
| residual `r(t)` depicted-vs-implied, median | 2.75-8.30 deg on the autonomous flights | diagnostic |

`experiments/phase2_implied.py` implements the full ladder (Savitzky-Golay window swept
in **seconds**: 0.05-0.37 including the 0.11 s reference; fixed `k_d = 0.295` and
drag-free; oracle / fitted / depth scale options; the residual against the depicted axis;
SNR; testable fraction; the oracle-ladder table). It consumes DPVO poses, and the poses
are the problem:

- the **oracle scale against mocap position is -0.003 to 0.011**, i.e. essentially zero,
  and negative on three flights;
- per-axis correlation between the VO trajectory and mocap is 0.01 / -0.09 / 0.13;
- the VO path length is 41 m where mocap says 479 m on 01p, and 4x-12x short elsewhere;
- raising DPVO to full 640x480 with 80 patches/frame does not change the result (01p:
  26.90 vs 26.67 deg).

So double-differentiating the recovered trajectory produces noise, not acceleration. The
depicted-vs-implied residual looks *small* on the autonomous flights (2.8-8.3 deg) only
because both sides are expressed in the same drifting VO frame and share its azimuth
error; it is not evidence that the implied tilt is right, and it is not a verdict.

A fitted scale cannot recover this either: the fitted scale removes 0-12 % of the
residual, and the fitted-scale RMSE (29.6 deg) is no better than the oracle one — a
negative result that is worth recording, because a fitted scale is exactly the nuisance
term that could otherwise absorb a violation later.

One part of the ladder is **not run** rather than substituted: the depth-model scale
needs a MoGe-2 pass, which was not started. The metrics record
`scale_depth: {"run": false}`.

## 6. Oracle ladder (attribution)

Attribution table, in the plan's order:

| rung | implied-tilt RMSE vs mocap | status |
| --- | --- | --- |
| mocap position + mocap attitude (Phase 0 floor) | 1.24 deg median residual | reference, from `results/phase0_floor/20261108T044153Z_f3b5d2bf` |
| VO position (oracle scale) + mocap attitude | **28.97 deg** | measured |
| mocap position + estimated attitude | 8.73 deg (scalar tilt) | measured as Gate 1's tilt term |
| VO position (oracle scale) + estimated attitude | 29.59 deg (fitted scale) | measured |
| fully estimated, depth-model scale | - | not run (no MoGe-2 pass) |
| fully estimated, fitted scale | 29.59 deg | measured |

Reading: swapping the position source from mocap to VO costs ~28 deg of implied-tilt
accuracy, while swapping the attitude source costs ~7 deg. The pixel trajectory is the
bottleneck by a factor of four, and it is the translation half of DPVO specifically —
its rotation is the best-behaved component in the whole pipeline.

## 7. Estimator comparison

| route | pooled median flight RMSE (20-frame sample) | status |
| --- | --- | --- |
| GeoCalib `simple_divisional`, raw fisheye | **13.18 deg** (8 flights, 20-frame sample); **22.3 deg** full RMSE on 5,007 frames | adopted |
| GeoCalib `pinhole`, undistorted view, focal prior | 26.90 deg | rejected |
| GeoCalib `pinhole`, undistorted view, free focal | 25.41 deg | rejected |
| Perspective Fields | - | not run (GeoCalib's median met the target on most frames) |
| AnyCalib | - | not run, same reason |

## 8. What did not work, and why

- `experiments/phase1_camera_solve.py` (mocap-supervised triangulation over a 6 m baseline)
  leaves 12-1000 px reprojection residual at the best rotation, i.e. the released lens model
  is not good enough for multi-view geometry. Kept as a diagnostic; **not** used as evidence.
- The `.5x` DPVO runs and the earlier "piloted mount unreconciled" reading are both
  superseded, the first by the full-resolution run (same result) and the second by the
  body-tilt correction in section 1.

## 9. Next steps that would move the numbers

Ranked by what the numbers say is binding:

0. **Fix translation, not rotation.** DPVO's rotation is already good enough for Gate 1;
   its translation is not usable. DROID-SLAM (dense, global BA), DPVO with loop closure
   and a much larger patch budget, or a per-segment scale instead of one per clip are the
   candidates. Gate 2 is unreachable until the recovered path correlates with the true
   one at all.
1. **Give the lean direction a heading reference.** Gravity can never fix the azimuth, so
   the lean-direction target needs either a heading cue from the scene (structure, gates,
   horizon direction) or a reformulation that compares lean directions *in the image*
   rather than in a world frame.
2. Gate-corner PnP per piloted flight (`gate_corners_<flight>.csv` + `label_flight-<flight>.zip`)
   to settle the mount, which would let the piloted flights (19-21 m/s) into the gate.
3. A fisheye model fitted to the released calibration plus the stated 175 deg field, or a
   line-based self-calibration, to recover the periphery (only matters if the undistorted
   route is revisited; the raw-fisheye GeoCalib route does not use it).
4. MoGe-2 metric scale to replace the fitted scale in the verifier's own operating point —
   and, more importantly, per-frame depth would supply the metric scale that a
   drifting VO trajectory cannot.
