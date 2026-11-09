# Report 02 — Phase 1: the depicted attitude (Gates 1a and 1)

Track A. Written 2026-11-09. Run directories and commits are named in every table header.

## Plain-language summary

Two separate questions were asked of the pixels, and they have different answers.

**From one frame at a time:** a frozen model (GeoCalib) can read the camera's tilt to
within about **4 degrees** on a typical frame — good enough to be useful. But on roughly
**7 % of frames** it is badly wrong (tens of degrees off), so the average error (22°) is
three times worse than the typical error. A single number would have hidden this.

**From a sequence of frames:** visual odometry gives an excellent *rotation* estimate
(0.15-1.5° per frame against motion capture), and combining it with the per-frame tilt
gives a good tilt reading on six of the eight flights. But the *direction* the drone is
leaning in (its heading) comes out essentially random — about **84° off** — because
gravity cannot tell you which way is north. That is a structural limit, not a bug: the
checker later takes the heading from the known flight path instead.

## The question and the setup

**Gate 1a.** Does per-frame depicted tilt agree with motion capture to RMS ≤ 5°?
**Gate 1.** Does the attitude over a whole clip agree to ≤ 5° tilt with lean direction
within 15°?

**Setup.** 8 TII flights, every 10th camera frame (5,007 frames, 12.04 Hz measured). The
camera mount is a hard mount at the recorded uptilt (30/40/50°, verified independently in
Phase 1's camera check). GeoCalib is run on the **raw fisheye** with its own strong
distortion model, because the released TII intrinsics cannot describe the periphery (their
polynomial inverts only to 56°, `fx/fy` = 0.749). Fusion: DPVO relative rotations predict,
GeoCalib gravity corrects, gain chosen leave-one-flight-out.

## Results

### Why this GeoCalib route (survey JSONL under `data/raw/geocalib/`, re-derived here)

Pooled median flight RMSE on a 20-frame-per-flight sample, using the same truth function
as the main run (`experiments/phase1_depicted.py`, commit `c738aa8`):

| route | pooled median flight RMSE | frames | status |
| --- | --- | --- | --- |
| GeoCalib `simple_divisional`, raw fisheye (no focal prior) | **13.18°** | 192 | adopted |
| GeoCalib pinhole on the undistorted view, focal prior | 26.90° | 48 | rejected |
| GeoCalib pinhole on the undistorted view, free focal | 25.41° | 192 | rejected |

The undistorted view is also only 46 % valid pixels, so the raw-fisheye route is not a
convenience choice.

### Gate 1a: per-frame depicted tilt

**Verdict: fails the RMSE criterion, meets it at the median.**
`results/phase1_depicted/20261108T131825Z_c738aa8`, commit `c738aa8`, 5,007 frames:

| quantity | value |
| --- | --- |
| median error | **3.96°** |
| trimmed RMSE (worst 10 % dropped) | **4.80°** |
| RMSE, all frames | **22.32°** |
| p90 / p99 | 12.87° / 93.61° |
| fraction over 10° / over 20° | 12.7 % / 7.5 % |

First run versus revised (both kept deliberately):

| run | frames | median | RMSE | trimmed RMSE |
| --- | --- | --- | --- | --- |
| `20261108T055529Z_3697b73` (first, 2 flights) | 1,033 | 3.52° | 12.53° | 3.66° |
| `20261108T060140Z_3697b73` (8 flights, 20-frame sample) | 192 | 4.37° | 23.71° | 5.43° |
| `20261108T131825Z_c738aa8` (final, all 8 flights) | 5,007 | 3.96° | 22.32° | 4.80° |

The revision did not change the verdict; it changed the *picture*: the first run's
two flights were the easy ones.

Per flight (same final run):

| flight | n | median | p90 | RMSE | fraction > 20° |
| --- | --- | --- | --- | --- | --- |
| 01p-ellipse | 1267 | 3.14 | 5.37 | **3.80** | 0.0 |
| 07p-lemniscate | 1284 | 3.59 | 6.03 | **4.20** | 0.0 |
| 04a-ellipse | 350 | 3.29 | 6.57 | 10.35 | 1.7 % |
| 01a-ellipse | 270 | 4.20 | 10.71 | 13.67 | 4.1 % |
| 07a-lemniscate | 370 | 2.54 | 71.64 | 28.43 | 10.5 % |
| 16p-trackRATM | 373 | 7.34 | 80.14 | 31.18 | 15.5 % |
| 13p-trackRATM | 536 | 7.85 | 85.40 | 33.69 | 18.3 % |
| 13a-trackRATM | 557 | 7.44 | 88.32 | 44.36 | 29.4 % |

By speed (pooled, `pooled_all_frames.by_speed`):

| speed | 0-5 | 5-10 | 10-15 | 15-20 | 20+ |
| --- | --- | --- | --- | --- | --- |
| n | 2790 | 1609 | 358 | 223 | 27 |
| median | 3.68 | 3.87 | **8.77** | 7.49 | 9.83 |
| RMSE | 9.88 | 27.71 | 43.59 | 36.84 | 13.62 |

By track / control / exposure (same run): ellipse 3.28 median (RMSE 7.50), lemniscate 3.37
(13.95), trackRATM 7.57 (37.55); autonomous 4.57 (30.96), piloted 3.83 (17.10); auto
exposure 3.96 (22.11), fixed 4.00 (23.53). The spread *inside* each group is larger than
between groups, so the track and the body attitude matter more than the control mode.

**Where the tail comes from.** A synthetic-rotation probe tilted the view by ±20° and asked
whether the recovered gravity rotated with it (probe rows quoted from the Phase 1 report,
`docs/track_a_report.md` §3, run `remote/a_geocalib.py --probe-flights`, output
`data/raw/geocalib/probe/probe.jsonl`). A convention bug would break both directions
on every flight equally; what was measured is view-dependence:

| flight | −20° consistency | +20° consistency |
| --- | --- | --- |
| 07p-lemniscate | 1.47° | 2.56° |
| 01a-ellipse | 14.25° | 1.48° |
| 13a-trackRATM | 13.55° | 6.52° (p90 62.5) |

Directionally asymmetric and scene-dependent: the estimator loses the answer when the
view tips away from the arena's informative structure. This is the same property that
later limits the verdict test's artifact floor.

### Gate 1: attitude over time (DPVO rotation + gravity)

**Verdict: fails both criteria.** `results/phase1b_fusion/20261108T133135Z_c738aa8`,
commit `c738aa8`. Tilt RMSE **8.73°** (target < 5), lean median **83.6°** (target < 15).
Leave-one-flight-out chose gain 1.0 in every fold, i.e. the filter is better off snapping
to the gravity measurement than trusting VO.

| flight | gain | tilt RMSE | tilt median abs | thrust-axis vector median | lean median |
| --- | --- | --- | --- | --- | --- |
| 01a-ellipse | 1.0 | 6.78 | 3.23 | 6.49 | 46.6 |
| 04a-ellipse | 1.0 | 5.78 | 2.00 | 4.60 | 26.6 |
| 07a-lemniscate | 1.0 | 10.67 | 2.19 | 3.12 | 63.2 |
| 13a-trackRATM | 1.0 | 21.31 | 5.01 | 6.92 | 87.3 |
| 01p-ellipse | 1.0 | 2.27 | 1.90 | 32.09 | 90.4 |
| 07p-lemniscate | 1.0 | 2.06 | 1.51 | 27.59 | 83.5 |
| 13p-trackRATM | 1.0 | 22.94 | 4.48 | 70.09 | 85.9 |
| 16p-trackRATM | 1.0 | 20.12 | 3.84 | 75.71 | 83.8 |

Three numbers disagree and that is the finding: scalar tilt is 2-21°, the thrust-axis
*vector* is 3-7° on the autonomous flights but 28-76° on the piloted ones, and the lean
direction is unusable everywhere. DPVO's own rotation is excellent — per-frame error
0.15-1.5° (`vo_relative_rotation_check`), 8.9° and 11.0° only on 13p/16p.

### Lean direction re-scored with windowed azimuth alignment (handoff 04, Task 0.2)

Gate 1 used one azimuth offset per clip. Re-scoring the same saved per-frame axes with one
fitted azimuth per sliding window (`results/phase3a_verdict/20261109T074134Z_492033f_lean`,
commit `492033f`):

| window | 0.5 s | 1 s | 2 s | 3 s | 4 s | 5 s |
| --- | --- | --- | --- | --- | --- | --- |
| median flight lean error | **4.4°** | 10.9° | 19.3° | 84.6° | 114.6° | 89.7° |

One azimuth per clip is 85.5°. The azimuth error is **fast** (sub-second), not the slow
drift the plan assumed, so 3-5 s windows are worse than one per clip; ~1 s anchoring brings
the lean error inside the Gate 1 target.

## What this supports

- A usable per-frame tilt estimator (median ~4°) with a known, quantified failure mode.
- The choice of GeoCalib on the raw fisheye, by measurement over three alternatives.
- The specific claim used later by the verdict test: *tilt magnitude is observable,
  heading is not*, so the verifier must take heading from the path.

## What this does not show

- That the tail can be fixed. Outlier rejection was deliberately left to Phase 3.
- That the fusion's tilt (8.7°) would survive on other scenes; it fails on the two
  hardest trackRATM flights and the gain was forced to 1.0, i.e. the filter adds nothing
  over the raw measurement here.
- Any claim about DPVO's translation (that is report 03).

## Caveats and open items

- **Measured, not inferred:** all errors above are against motion capture. The one
  inferred item is the cause of the tail (scene content), which is supported by the ±20°
  probe but not isolated from motion blur; a Laplacian-variance proxy correlates −0.22
  with error, i.e. weak.
- The piloted trackRATM flights carry a 6-7° mount discrepancy that is reported and not
  fitted; a gate-corner PnP would settle it.
- `phase1_camera_solve` (triangulation self-calibration) leaves 12-1000 px reprojection
  residual and is kept as a negative diagnostic, not as evidence.
