# N04 — Video rotation on generated clips (H04 SolarWM, H01 ANWM)

**Decision.** Rotation extraction works on the H04 SolarWM clips: all 234 clips returned a pose
for every one of their 81 frames, and the propagated tilt follows the commanded attitude better
than the per-frame reader at every horizon (arm A 3.54/4.40/6.69° at 1/2/5 s against a reader of
9.07°; level arm 0.36/0.77/1.04° against 2.21°; mirrored arm 4.00/4.70/6.16° against 8.14°).
The H01 ANWM frames do not: only 20 of 80 clips give a usable two-frame rotation fit.

Pre-registration `configs/n04_generated_rotation_prereg.yaml` (committed before any DPVO run on
generated video). Code `experiments/n04_generated_rotation.py`; GPU `remote/n_rotation.py` task
`gen`, app `awmv-n-rotation`, volume `awmv-n-data` (`gen/videos`, `poses/gen`). Results
`results/n04/`, run summaries `data/n04/poses`.

## Plain-language summary

1. **Tracking never failed.** 234 of 234 clips, 81 of 81 frames each, no frozen estimate under
   the pre-registered check (median per-step rotation at least 10 % of the commanded attitude's).
   This is the opposite of the earlier ViPE experience on ANWM and a direct answer to Part 4's
   question: generated SolarWM video carries enough stable image motion for a geometric rotation
   estimator.
2. **The propagated tilt follows the command.** With an oracle start (so the number is pure
   relative-rotation fidelity), the error against the attitude the generator was *commanded* to
   render is 3.54° at 1 s and 6.69° at 5 s for the physically consistent arm — about 2.4° below
   the reader on the same clips at 5 s, and well below it at 1 s.
3. **The level arm is the cleanest sanity check** (0.36° at 1 s, 1.04° at 5 s): a nearly static
   commanded attitude is tracked almost exactly, so the 6.7° on the flatness arm is real
   relative-rotation drift under fast commanded motion, not a convention error.
4. **The gap to real imagery is large.** UZH real outdoor drifts 0.43° after 10 s; generated
   video drifts 6.7° after 5 s. Generated video is good enough to *support* the rotation route,
   but not as good as real footage, and the 5 s limit is the clip length (81 frames at 16 fps),
   not a measured floor.
5. **The focal assumption is not fatal.** Re-running 12 clips at 70°, 90° and 110° horizontal
   field of view moves the 5 s propagated error from 1.48° to 1.80° to 2.36° (same clips), so the
   conclusion survives the unknown intrinsics; the absolute numbers carry that ±0.5° uncertainty.
6. **ANWM is different.** On the 80 H01 ANWM clip directories, only 20 (25 %) give ≥ 20 inliers
   on a two-frame essential-matrix fit (median 24.5 inliers, Sampson 2.8 px). No attitude horizon
   is claimed there because the action rate is undocumented; the usable fraction is the finding.

## Setup

**Population.** The 234 H04 clips that are local (78 per arm, draws g0-g2), listed with sha256 in
`results/n04/clips.csv`; the 80 ANWM clip directories in the M02 mirror for the secondary check.
Nothing was re-downloaded from R2 and no held-out simulated clip was read.

**Truth attitude.** Track H4's `results/h04/paths/paths.npz` (roll, pitch, yaw per path, 81
frames), turned into the camera-to-world the generator was conditioned on:
`R_cw = body_rotation(yaw, pitch, roll) @ R_bc` with `R_bc = camera_to_body_pitched(0)`, exactly the
construction in `h04_solarwm_inputs.py`; arm B zeroes pitch and roll, arm C negates roll.

**Extractor.** DPVO at the standing settings on the native 864x480 frames, 81 per clip at 16 fps.
Intrinsics are unknown for generated video; 90° horizontal (f = 432) is assumed, with the focal
sensitivity below. Statistic: oracle start, propagate, angle to the commanded up at 1/2/5 s,
median over starts per clip, then median across clips.

**Reader reference.** The frozen reader's per-frame up on the same clips
(`track-h3 results/h04/depicted_main/depicted.jsonl`), error against the same commanded attitude.

## Results

| arm | clips | tracked | 1 s | 2 s | 5 s | reader (same clips) |
| --- | --- | --- | --- | --- | --- | --- |
| A flatness | 78 | 78 | **3.54** | **4.40** | **6.69** | 9.07 |
| B level | 78 | 78 | **0.36** | **0.77** | **1.04** | 2.21 |
| C mirrored roll | 78 | 78 | **4.00** | **4.70** | **6.16** | 8.14 |

All cells are the median of per-clip medians of the angle between propagated and commanded up.
No clip failed (crash, partial poses or frozen estimate). Per-clip values are in
`results/n04/summary.json`; per-horizon rows in `results/n04/horizons.csv`.

**Focal sensitivity** (12 clips, 4 per arm, same clips at three assumed fields of view):

| assumed horizontal FOV | 70° | 90° | 110° |
| --- | --- | --- | --- |
| median 5 s propagated error | 1.48° | 1.80° | 2.36° |

**ANWM secondary check.** 80 clip directories, frames 0 and 2, SIFT + essential matrix with the
same settings: 20/80 (25 %) reach 20 inliers; median 24.5 inliers and 2.8 px Sampson among those.
The ANWM frames are 16 images per action at an undocumented rate, so no horizon in seconds is
reported. LingBot ramps were not measured: their frames are not in the local mirrors used here.

## What this supports

* Generated SolarWM video **can** support attitude from video rotation over its 5 s length, with
  the propagated tilt closer to the commanded attitude than the per-frame reader. This is the
  piece that the earlier ViPE-on-ANWM experience suggested would fail.
* The rotation route's weakness on generated video is **drift under fast commanded motion**
  (6.7° in 5 s against 0.4° in 10 s on real UZH), not tracking failure.

## What this does not show

* **No verdict.** Whether the propagated lean detects a tilt violation on generated clips is a
  verdict experiment, not run here; the addendum and Part 2 were not run on generated video.
* **No absolute attitude.** The oracle start means the numbers measure relative rotation only;
  a start read from generated pixels would add the reader's own error (about 9° on arm A).
* **No multi-view consistency claim.** Good rotation tracking and consistent geometry are
  different properties; N02's 0.23 px two-view residual does not certify the latter either.
* **One generator, one draw family.** SolarWM-5B only; ANWM's 25 % usable fraction is a separate,
  weaker result.

## Caveats

* Intrinsics unknown: the 90° assumption is stated and the sensitivity arm bounds its effect
  (1.5-2.4° at 5 s), but the absolute numbers are not calibration-grade.
* The horizon stops at 5 s because the clips are 81 frames at 16 fps; longer behaviour is not
  measured.
* The commanded attitude is the generator's *input*, not evidence that the generator rendered it;
  the reader comparison (which reads the rendered pixels) is the cross-check, and the level arm
  (near-zero command, 1.04° error) is the cleanest evidence that the pipeline is sound.
* ANWM's usable fraction uses only frames 0 and 2 of 16; a different pair could differ.

## Provenance

* Pre-registration `configs/n04_generated_rotation_prereg.yaml`; decisions `[N]`.
* Clip hashes `results/n04/clips.csv`; poses `awmv-n-data/poses/gen/fov{70,90,110}` (234 + 12 + 12
  runs, 81 frames each, about 400 s of A10 = USD 0.12 at 1.10 USD/h).
* Inputs read-only from track-h3 (`paths/paths.npz`, `depicted_main/depicted.jsonl`) and track-m
  (`data/m02_inputs/h04`); no R2 write; every app stopped.
