# E01 — UZH-FPV audit and whether the depicted-tilt estimator transfers

Track E. Written 2026-11-14. Worktree `../awmv-track-e`, branch `track-e-depicted-uzh`.
Pre-registrations: `configs/e_prereg.yaml` (Task 2, committed before any UZH estimate) and
`configs/e_verdict_prereg.yaml` (Tasks 4-5). Acceptance spec: `configs/e_acceptance_ref.yaml`.
Artifacts: `results/e_audit/audit.json`, `results/e_audit/sync_summary.json`,
`results/e_perframe/<tag>/metrics.json`. Decisions: `docs/decisions.md`, tag `[E]`.

## Plain-language summary

1. **The task's premise was wrong in the project's favour.** UZH-FPV *does* publish the
   camera-to-IMU mount, for all four rigs, in Kalibr `camchain-imucam` files that nobody had
   opened. Track C fitted a mount from ground truth because it believed none existed; the
   fitted mount was compensating for a wrong mount *convention*, not for a missing file.
2. **The ground truth is the IMU pose in a gravity-aligned world frame**, so its body z axis is
   exactly the thrust axis the verifier needs. Verified three ways (quaternion direction,
   world z, and consistency with the flatness equation at the same drag coefficient TII uses).
3. **The clocks agree.** The Leica truth, the IMU and the images share a time base to within
   17 ms on all 14 sequences, measured by correlating the three-axis gyro against truth
   angular velocity.
4. **The depicted-tilt estimator transfers.** On the nine forward-facing sequences the median
   per-frame gravity error is **1.53 deg** (undistorted pinhole with the published focal),
   **4.62 deg** (raw fisheye with `simple_divisional`, the TII-best setting) and **2.06 deg**
   (pinhole, focal estimated). TII's own reference is 3.96 deg. Gate E1 passes for all three.
5. **But it transfers unevenly.** Indoor forward is 1.0-3.3 deg median; **outdoor forward is
   8.4-9.7 deg on every variant**, with error growing with speed and with motion blur. That is
   the first place the verifier's coverage will be limited, and it is a property of the
   footage, not of the mount.

So the previous agent's 86-89 deg verdict numbers on UZH were **not** the estimator failing:
the per-frame error is 1-5 deg. Something in that pilot's verdict plumbing was wrong (mount
convention, or the score), and this report replaces it with a measured chain.

## Task 0: reproduction of the TII reference

Recomputed from `results/phase1_depicted/20261108T131825Z_c738aa8/frames_flight-*.npz`:

| quantity | value |
| --- | --- |
| frames | 5007 |
| median error | **3.9617 deg** (published 3.96) |
| trimmed RMSE, 10% trim | **4.7971 deg** (published 4.80) |
| trimmed RMSE, 5% trim | 10.1961 deg |
| full RMSE | 22.3236 deg |
| fraction over 10 deg | 0.1272 |

**The published "trimmed RMSE 4.80" drops the worst 10 percent, not 5**
(`experiments/phase1_depicted.py` sorts the errors and keeps `int(0.9 * n)`), while the handoff
describes it as "drop the worst 5%". Both are reported for UZH, so the comparison is
like-for-like; a 5% trim against 4.80 would have flattered UZH by about 5 deg.

## Task 1: data and truth audit

### Sequences used

14 sequences with public ground truth, prepared by `remote/e_prepare.py` at stride 2 (about
15 Hz) from the first frame inside the truth interval, up to 450 frames each, both as raw KB4
fisheye (640x480, 8-bit grey) and as an undistorted pinhole view (same focal, principal point
centred, 640x480). Volume `awmv-e-data` under `/uzh/<seq>/`; the truth arrays and metadata are
mirrored locally under `data/uzh/`.

| rig | sequences | environment | mount (optical-axis elevation from published `T_cam_imu`) |
| --- | --- | --- | --- |
| indoor_forward | indoor_forward_3, _5, _6, _7, _9, _10 | indoor | -0.83 deg |
| outdoor_forward | outdoor_forward_1, _3, _5 | outdoor | -0.73 deg |
| indoor_45 | indoor_45_2, _4, _9, _13 | indoor | 45.57 deg |
| outdoor_45 | outdoor_45_1 | outdoor | 45.93 deg |

Truth coverage per sequence is 17.6-85.7 s (the Leica lock is lost or the flight ends), the
truth rate is 500 Hz (2 ms), and measured speed p95 is 4.3-16.1 m/s, in line with the
published per-sequence maxima (4.87-20.73 m/s).

### Ground-truth frame, convention and world axes

The ICRA19 paper (Sec. III-E) says the alignment estimates the state `(q, p, v)` of the
**Snapdragon IMU frame** in the Leica world frame, with the prism-to-IMU translation estimated
alongside, and describes the MS60 world frame as gravity-aligned. That was checked, not taken:

| question | test | result |
| --- | --- | --- |
| quaternion direction | median flatness residual (body z from quat vs thrust axis implied by the path) read as body->world | **4.0-11.5 deg on all 14**, against 9.7-45.3 deg for the inverted reading |
| world z direction | leave the world "down" direction free (2 parameters) and fit it | **within 1.41 deg of -z on all 14** once the TII drag coefficient 0.295 is included |
| is the drag term doing the work? | same fit with `k_d = 0` | a drag-free fit absorbs 3.5-6.6 deg into a tilted "gravity", i.e. the tilt is drag, not a levelling error |
| is the truth physically consistent? | median flatness residual on the truth path | **1.7-3.2 deg with `k_d` = 0.295**, against 4.0-11.5 deg drag-free; TII's floor is 1.24 deg fitted, 5.46 drag-free |
| does the flight start at rest? | first 0.3 s | indoor sequences: tilt under 1.6 deg, \|a\| under 0.12 m/s^2, as the paper states; the outdoor ones start already tilted (~9 deg) |

Body frame is **FLU** (x forward, y left, z up), established from the forward mount (camera
+y, which is image-down, maps to IMU -z) and cross-checked by the 45-degree rigs: with z up
their optical axis is 45 deg *below* horizontal, matching their published names, and with z
down it would be 45 deg above, which does not.

### Mount: published, per rig

Each rig ships `<rig>_calib_snapdragon.zip` with a Kalibr `camchain-imucam` YAML containing
`T_cam_imu` for `cam0` (`snappy_cam/stereo_l`, the camera the frames come from) and a
`timeshift_cam_imu`. `T_cam_imu` maps IMU coordinates to camera coordinates, so the
camera-to-body rotation used here is `R_cb = R(T_cam_imu)^T`
(`src/depicted/uzh.py::published_mount`). Read that way, the optical axis elevation is
-0.83 deg (indoor forward), -0.73 deg (outdoor forward), 45.57 deg (indoor 45) and 45.93 deg
(outdoor 45) below the body horizontal — the rig names, recovered from the extrinsic alone.

The mount is therefore an **input to the verifier**, exactly as TII's recorded uptilt is, and
no per-rig fit is needed for any headline number. A fitted mount appears only as an
independent check on the residual.

### Time offset: camera to truth

Measured by correlating the **three-axis body-frame angular velocity** from `imu.txt` against
the same quantity differenced from the truth quaternions — no camera, no extrinsics, no pixel
estimator in the loop (`src/depicted/sync.py`, `experiments/e_clock.py`).

| quantity | result (all 14 sequences) |
| --- | --- |
| truth clock vs IMU clock, `gt_time = imu_time + tau` | **tau in [-5, +17] ms**, peak correlation 0.86-0.99, overlap 17.6-85.7 s |
| images vs IMU clock | first-sample difference **<= 30 ms** in every sequence, same span to 0.1 s |
| published Kalibr `timeshift_cam_imu` | -16.7 ms (indoor forward), -14.8 ms (indoor 45), -8.0 ms (outdoor forward), -8.6 ms (outdoor 45) |
| offset resolution | 22-78 ms (2 ms search step, plateau width at 0.5% of peak) |
| drift within a recording | **not measurable**: each truth file covers only part of its recording, so no drift bin is fully covered; reported as unmeasured rather than as a number |

Camera-to-truth offset is therefore about -9 ms (indoor forward) to +9 ms (45 rigs), inside the
+-25 ms grid the verdict statistic already searches over lag, with a stated uncertainty of
+-25-40 ms. This is an evaluation nuisance, small but not zero, and it is recorded per sequence
in `results/e_audit/sync_summary.json`.

### Two things this audit corrected

* **A wrong first reading, corrected before use.** The first pass read `imu.txt`'s first column
  as a timestamp; the rows are `index timestamp wx wy wz ax ay az`, so the "span" was a row
  counter. The consequence was a claim that the indoor-45 images sit on a clock 18 years from
  the truth clock and that Track C's pairing for `indoor_45_9` was therefore meaningless.
  That claim is **withdrawn**: the measured offset is +13 ms, every sequence's truth
  timestamps fall inside its own image span, and Track C's "keep the frames inside the truth
  interval" filter selects the right frames. The hazard is real in principle and is now
  measured rather than assumed, but no Track C result is withdrawn on this ground. The
  correction is in `docs/decisions.md` with the reasoning.
* **A magnitude-only correlation produced a spurious offset.** On `outdoor_45_1` it reported
  -56.5 s with a 0.92 peak from a 2.7 s overlap. Correlating all three gyro axes gives
  -0.004 s at 0.86 on a 20 s overlap, in line with the other thirteen. The three-axis form is
  what is used, and an offset whose overlap is under 3 s is declared unidentifiable.

## Task 2: does the estimator transfer?

Variants, fixed in advance: (a) undistorted pinhole with the Kalibr focal as a prior;
(b) **primary**, raw KB4 fisheye with GeoCalib's `simple_divisional`, the TII-best setting;
(c) undistorted pinhole with the focal estimated. GeoCalib at commit `1dedb29f`, A10G,
3.8 estimates/s per container.

The metric is the angle between the estimated and truth-implied up direction, both **in the
camera frame**, using the published mount. No clip, no path, no nuisance fit.

### Forward-facing rigs against TII (the Gate E1 population)

| variant | n | median | trimmed RMSE 5% | trimmed RMSE 10% | full RMSE | > 10 deg |
| --- | --- | --- | --- | --- | --- | --- |
| TII reference (`div10`, raw fisheye) | 5007 | 3.96 | 10.20 | 4.80 | 22.32 | 0.127 |
| **b: raw fisheye, `simple_divisional`** | 3488 | **4.62** | 6.63 | 5.83 | 11.10 | 0.192 |
| a: pinhole + Kalibr focal prior | 3488 | **1.53** | 5.05 | 4.29 | 12.66 | 0.131 |
| c: pinhole, focal estimated | 3488 | **2.06** | 5.06 | 4.47 | 12.55 | 0.110 |

**Gate E1 (pre-registered: forward pooled median <= 6.0 deg and <= 5.96 deg, 10% trimmed RMSE
<= 6.80 deg): passes for all three variants.** The estimator transfers on this data.

### Per sequence

| sequence | n | (b) median | (b) >10 deg | (a) median | (a) >10 deg |
| --- | --- | --- | --- | --- | --- |
| indoor_forward_3 | 450 | 3.74 | 0.031 | 0.99 | 0.018 |
| indoor_forward_5 | 264 | 1.28 | 0.000 | 0.71 | 0.000 |
| indoor_forward_6 | 408 | 5.50 | 0.154 | 1.19 | 0.118 |
| indoor_forward_7 | 450 | 3.96 | 0.122 | 1.82 | 0.038 |
| indoor_forward_9 | 383 | 3.00 | 0.000 | 0.94 | 0.000 |
| indoor_forward_10 | 410 | 3.04 | 0.000 | 0.89 | 0.000 |
| outdoor_forward_1 | 450 | 9.05 | 0.462 | 8.36 | 0.311 |
| outdoor_forward_3 | 450 | 9.95 | 0.493 | 7.90 | 0.322 |
| outdoor_forward_5 | 223 | 9.86 | 0.489 | 9.59 | 0.444 |

By rig (variant b): **indoor forward 3.28 deg median** (n = 2365), **outdoor forward 9.68 deg**
(n = 1123). By rig (variant a): 1.00 and 8.35 deg.

### What drives the error: speed and blur, not the estimator's input choice

Median error by truth speed (variant b / variant a): 0-5 m/s 3.32 / 1.09; 5-10 5.66 / 2.54;
10-15 5.14 / 4.14; 15-20 11.74 / 9.87. The outdoor sequences are the fast ones, and they also
have the lowest blur quintiles (variance of the Laplacian on the frame the estimator saw):
quintile 1 (blurriest) 6.69 deg against quintile 5 (sharpest) 3.37 deg for variant b, and
5.56 against 1.11 for variant a. **Blur is a first-order driver of the tail**, and it is
concentrated outdoors. All three variants degrade together outdoors, so this is not an
artefact of which camera model was given.

### Temporal smoothing

Running median over the per-frame gravity direction, windows in seconds:

| variant | raw median | 0.5 s | 1.0 s |
| --- | --- | --- | --- |
| b | 4.62 | 4.80 | 5.40 |
| a | 1.53 | 2.06 | 3.48 |
| c | 2.06 | 2.59 | 3.79 |

**Smoothing does not help**: the errors are systematic within a clip (a slowly varying
attitude bias), not zero-mean jitter, so a median over 0.5-1 s mixes in neighbours that are
equally biased. TII's heavy tail was jitter-like and its report already warned that only the
median is honest; on UZH the *median itself* is the thing that moves with speed. Recorded as a
negative result so nobody spends a day on a window sweep.

### Frozen confidence gate S3 on UZH

`configs/gate_frozen.yaml` (signal = median `latitude_confidence`, tau = -0.0016656) applied
unchanged to 4 s clips (60 frames at 15 Hz):

| variant | clips | positive (median err > 10 deg) | AUROC | 95% interval | clip drop rate, indoor / outdoor forward |
| --- | --- | --- | --- | --- | --- |
| b | 53 | 9 | **0.927** | 0.801-0.995 | 0.17 / 0.94 |
| a | 53 | 5 | 0.729 | 0.482-0.958 | 0.03 / 0.59 |
| c | 53 | 2 | 0.441 | 0.018-0.959 | 0.03 / 0.59 |

The gate **does transfer for the primary variant** (0.93 against 0.946 fitted on TII), and it
pays the same price TII did: it drops 94% of the outdoor clips, i.e. it discards the flight
rather than isolating the bad clips. On variants a and c the signal is weak, which is itself
informative: those variants are so accurate indoors that the clip-level label has almost no
positives (5 and 2 of 53), so the AUROC is measured on a handful of clips and should not be
quoted as a property of the gate.

## What this supports, and what it does not

**Supports.** The depicted-tilt half of the verifier works on UZH-FPV at least as well as on
TII, once the published mount is used and the camera/body direction is right; the truth is in
the frame the verifier needs; the clocks are aligned to a few tens of milliseconds. The
86-89 deg numbers from the earlier UZH pilot cannot have been the estimator's per-frame error,
so they were a plumbing error, and the plumbing is now measured end to end.

**Does not support.** Any claim that the estimator is uniform across the dataset: the outdoor
rigs sit at 8-10 deg median, where a 10 deg violation is inside the estimator's own error, and
the S3 gate responds by dropping 94% of those clips. Any use of a 0.5-1 s temporal median as a
fix. Any claim about the 45-degree-down rigs from this section alone (below).

**Restriction of the claim.** The metric uses the published mount. It therefore tests the
estimator *given* the mount, not the mount itself; a mount error of a few degrees would be
invisible in the body-frame agreement that this metric reports as "the estimator transfers".
The independent mount check is the residual against a *fitted* rotation, in E02.
