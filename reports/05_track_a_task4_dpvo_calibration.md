# Report 05 — Task 4: DPVO on a well-calibrated dataset (parked)

Track A. Written 2026-11-09. Parked at the user's instruction. This report records the two
attempts, what was built, and why no number exists.

## Plain-language summary

We wanted to know whether our "tilt from the motion" route failed because of the TII
camera calibration or because of the method. The clean test is to run the same pipeline on
a dataset whose camera calibration is known to be good. We downloaded one such sequence
(UZH-FPV, 417 MB, 2,091 frames). **It turned out to ship no camera calibration at all**,
and the calibration bundle is not at any of the documented download paths. Without a
calibration the experiment would compare "our wrong numbers" against "their right
numbers", so it was not run. Two attempts were the time-box; the work is parked.

## The question and the setup

**Question.** Does DPVO's recovered translation correlate with ground truth on footage
whose intrinsics are trustworthy? If yes, the TII calibration (a pinhole fit of a 175°
fisheye, `fx/fy` = 0.749, polynomial invertible only to 56°) is the likely cause of the
Phase 2 failure; if no, the method is.

**Intended setup.** One UZH-FPV sequence, undistorted from its Kannala-Brandt model to a
pinhole view with `cv2.fisheye`, then the same DPVO build and settings as Phase 2, then
oracle scale (least squares, no rotation) against the laser-tracker ground truth, per-axis
correlation, path-length ratio and implied-tilt RMSE with `k_d` = 0.295.

## What happened

| attempt | what was tried | outcome |
| --- | --- | --- |
| 1 | Disk budget: full UZH-FPV release (tens of GB) vs 3.9 GB free; 128 px HuggingFace derivative as an alternative | full release does not fit; the derivative carries no trustworthy calibration, which is the one property the test exists to isolate |
| 2 | Downloaded `indoor_forward_3_davis_with_gt.zip` (417,718,727 bytes, 2,091 frames) from `download.ifi.uzh.ch/rpg/web/datasets/uzh-fpv/` | the zip has `images.txt`, `imu.txt`, `events.txt`, `groundtruth.txt`, `img/*.png` and no calibration **inside it**; my probes for `calib.zip`, `uzh_fpv/calib.zip`, `calib/<sequence>.yaml` and `*_snapdragon*` returned 404 — **but the release does ship a Kalibr camchain elsewhere** (found by Track C), so this was a wrong-path probe, not a missing calibration |

Two facts found while probing, so the next attempt is short:

- `groundtruth.txt` is `id timestamp tx ty tz qx qy qz qw` — **nine columns, id first**
  (not the TUM eight-column form).
- `images.txt` is `id timestamp image_name`.

## What was built (committed, unit-tested, unused)

| file | what it does |
| --- | --- |
| `scripts/uzh_prepare.py` | parses a `*_with_gt.zip`, undistorts KB4 → pinhole with `cv2.fisheye`, writes `frames/*.png` and a `meta.json` with intrinsics, timestamps and the ground-truth track |
| `remote/a_dpvo.py --uzh` | same DPVO image and settings as Phase 2, frames read from the Modal Volume |
| `experiments/phase3a_uzh_dpvo.py` | oracle scale, per-axis correlation, path ratio, implied-tilt RMSE; unit tests cover scale recovery, correlation behaviour, and that a constant-velocity track gives the analytic drag tilt `atan(k_d·v/g)` |

## What this supports

- The Phase 2 attribution is **suspended, not settled**: "DPVO's translation fails, its
  rotation does not" is measured on TII; "the TII calibration is the cause" remains a
  hypothesis.
- **Corrected 2026-11-10:** the "UZH-FPV ships no calibration" claim above is withdrawn — the
  release ships a Kalibr camchain (Track C found it). The task is parked on effort, not on
  missing data.
- Nothing else. No number in this report is a measurement of DPVO on good calibration.

## What this does not show

- Whether DPVO's translation would work with a trustworthy calibration.
- Whether the released TII calibration is the cause of anything. It is *suspect*: the
  polynomial inverts only to 56°, all four image corners are unrepresentable, the
  distortion model is a pinhole fit of a 175° fisheye, and a mocap-supervised
  triangulation check leaves 12-1000 px reprojection residual.

## Caveats and open items

- **Inferred, not measured:** the sequencing "calibration first, then method". It is the
  cheaper hypothesis to test, not a demonstrated cause.
- Next step, in order: get the calibration from the dataset authors' `calib` folder (the
  zip we found has none), or use a sequence that ships `calib.yaml` (the snapdragon zips
  are documented to; their download names were not found at the paths probed). If neither
  works in one more attempt, drop the pixel-path route for general video models — it is
  not needed for the ANWM verdict, which runs in commanded-path mode.
- The sequence zip was deleted after inspection; re-downloading takes ~2.5 minutes.
