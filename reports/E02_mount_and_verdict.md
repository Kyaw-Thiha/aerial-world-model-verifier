# E02 — Mount check, verdict clips and the frozen-threshold transfer test

Track E. Written 2026-11-14. Pre-registration: `configs/e_verdict_prereg.yaml` (committed before
any edited clip was scored). Depends on `reports/E01` (data audit and per-frame validation).
Artifacts: `results/e_mount/mount.json`, `results/e_clips/index.json`,
`results/e_verdict/<tag>/metrics.json`.

## Task 3: is the published mount the right reading?

The mount is published, so this is a **check**, not a calibration. The test is designed so that a
misread mount cannot pass: fit one constant rotation per rig on the pre-registered fit
sequences, then evaluate it on the held-out sequences of that rig. If `R_cb` were transposed or
inverted, no constant rotation could repair it and the held-out residual would stay large.

| variant | rig | fitted rotation | residual, fit set (published -> fitted) | residual, held-out (published -> fitted) |
| --- | --- | --- | --- | --- |
| a | indoor_forward | **0.74 deg** | 0.85 -> 0.66 | **1.08 -> 0.86** |
| a | outdoor_forward | 28.07 deg | 8.36 -> 6.24 | 8.35 -> 8.49 |
| b | indoor_forward | **3.40 deg** | 2.21 -> 1.55 | **3.70 -> 2.48** |
| b | outdoor_forward | 35.13 deg | 9.05 -> 6.21 | 9.91 -> 10.56 |
| c | indoor_forward | **1.11 deg** | 1.20 -> 0.70 | **1.64 -> 0.97** |
| c | outdoor_forward | 25.62 deg | 8.62 -> 6.34 | 8.34 -> 7.33 |

Indoor forward: the best constant rotation differs from the published one by **0.7-3.4 deg**,
and fitting it on other sequences buys 0.2-1.2 deg on held-out data. That is the size of
GeoCalib's own view-dependent bias (Track A measured 0.36-1.47 deg on TII with the same probe),
so **the published mount is confirmed** and the residual is the estimator.

Outdoor forward: a large rotation (26-35 deg) fits the fit sequence better and **does not
generalise** — the held-out residual is unchanged or worse. A constant rotation cannot explain
the outdoor error, so it is not a mount problem: it is a scene-dependent estimator bias (open
field, sloping horizon, trees) that also correlates with speed and blur (E01). This is the
answer to the question that stalled the earlier attempt: **fitting a mount on UZH cannot fix a
large verdict error, and the fact that fitting "only" brought 86 deg to 56 deg says the
problem was never the mount.**

### Depicted roll and pitch against truth

Roll and pitch are recovered from the depicted up-vector in the body frame and compared with
truth's own roll and pitch (degrees, roll wrapped to (-180, 180], pitch signed). Per sequence,
with the published mount:

| variant | sequence | roll error | pitch error | body-axis error |
| --- | --- | --- | --- | --- |
| b | indoor_forward_5 | 0.28 | 1.22 | 1.28 |
| b | indoor_forward_10 | 0.54 | 2.99 | 3.04 |
| b | indoor_forward_3 | 0.36 | 3.66 | 3.74 |
| b | outdoor_forward_3 | 5.11 | 6.02 | 9.95 |
| a | indoor_forward_5 | 0.20 | 0.64 | 0.71 |
| a | outdoor_forward_3 | 6.45 | 4.38 | 7.90 |

(medians over each sequence; all 14 sequences in `results/e_mount/mount.json`)

**The primary variant's indoor error is almost entirely a pitch bias**: 0.28-0.54 deg of roll
against 1.2-3.7 deg of pitch, on every indoor sequence. The pinhole variants split the error
more evenly (0.2-0.5 roll, 0.6-1.7 pitch). A bias of this shape is what a slightly wrong
distortion model on a forward-looking fisheye would produce, and it matters for the verdict:
a pitch-only bias is not averaged away by the median, and pitch is exactly the axis a forward
camera's tilt violation shows up in.

Speed dependence: **within** the indoor rig the per-frame residual does not track speed
(Spearman -0.27 to +0.30, no consistent sign across sequences), so E01's speed-bin trend
(1.1 deg at 0-5 m/s against 9.9 at 15-20) is a **rig** effect — the fast frames are all
outdoor — and not a within-rig speed effect. Corrected here rather than left as an implied
claim; the honest statement is "outdoor forward is worse and outdoors is fast", not "speed
causes the error".

## Task 4: verdict clips

### Edit geometry, validated before any clip was scored

Edits are exact pure rotations about the camera centre in the dataset's own KB4 geometry
(`src/depicted/edits.py`), applied to the raw fisheye that variant b reads. The KB4 model is
checked against OpenCV's own `fisheye.projectPoints`/`undistortPoints` on random rays (agreement
better than 5e-3 px), the identity warp is pixel-exact, and a rotation followed by its inverse
returns the original sampling coordinates to 1e-6 (`tests/test_depicted_edits.py`).

The estimator-side check (`remote/e_export.py --validate`) applies a known rotation, runs
GeoCalib again, and compares with the prediction `M^-1 u_before`:

| sequence | 0 deg (null edit) | 5 deg | 10 deg | 20 deg |
| --- | --- | --- | --- | --- |
| indoor_forward_5 (84 estimates) | **0.06 deg** | **0.76** | **1.39** | **1.31** |

Pre-registered bound: 2.0 deg median at every angle. The indoor sequence passes at every angle
with a factor of 1.4 to spare, and the null edit moves the estimate by 0.06 deg, so the warp
itself contributes essentially nothing.

**The same probe fails on `outdoor_forward_5` (the 20.7 m/s clip): 25.9 deg on the null edit
and 23.6-24.1 deg at 5/10/20 deg.** A re-encode of the *same* frame that changes the estimate by
26 deg is not an edit artefact, it is the estimator being unstable on that footage — the same
sequence where 49% of frames are already more than 10 deg off (E01). Its clips are exported and
scored, and they are the noisiest rows in the tables; this is recorded rather than hidden by
dropping the sequence.

### Border handling

Each frame gets **one mask shared by all its conditions, including the null edit**: the
intersection of the conditions' valid regions for that frame, with black fill outside. The
measured valid fraction of an edited frame is 0.77-0.80 on the clean indoor sequence, so the
50%-valid usability rule rarely fires.

An earlier version took the intersection across *all frames of the clip* as well as across
conditions. A clip that manoeuvres through 45 deg then keeps 40% of the frame, every edited
frame fails the usability rule, and every edit condition abstains — which is exactly what the
first export produced and why it was discarded and rebuilt. Recorded because the failure is
silent: the clips existed, looked well formed, and had no depicted estimate in them.

### Pipeline consistency

The exported `real` clip's depicted gravity agrees with the independent Task 2 per-frame run on
the same frames to **0.046 deg median**, so the export pipeline does not perturb the estimate.
(For comparison, Track A's identity edit reproduced its Phase 3 run to 0.13-0.16 deg.)

## Task 5: the frozen-threshold transfer test

Filled in from `results/e_verdict/<tag>/metrics.json`.

### Graded output (T5) coverage

The statistic produces, per clip and condition: required tilt (median), depicted tilt
(median), the residual score, the fitted lag, the depicted confidence, the abstention reason
and the testable flag. It does **not** produce a gain with an interval or a drag estimate —
those belong to Track D's gain statistic, which is not published yet (see below). No number in
this report is a gain.

### Track D's statistic (5b)

`track-d-verdict-stat` is at the same commit as `track-a2-gate` and has no
`configs/gain_frozen.yaml` and no `scripts/score_clip.py`, so 5b is **not run**. The exported
clips are the input for it: they are in the shared schema, validated, and uploaded to R2.

### The black-fill border invalidated the first clip family (measured, then fixed)

The first complete export used black fill outside a per-frame common mask, the same border
handling Track A used on TII. The **null-edit control caught it immediately**, which is what
that control exists for:

| condition (42 clips, 37 testable) | rejected at 14.73 deg | median score |
| --- | --- | --- |
| real | 0.14 | 5.65 deg |
| **null edit (identity warp, same mask)** | **0.62** | **18.70 deg** |
| yaw control (same mask, no tilt change) | 0.59 | 18.17 deg |
| tilt removal | 0.89 | 28.04 deg |
| mirrored roll | 0.89 | 27.02 deg |
| tilt x0.5 | 0.62 | 20.35 deg |
| tilt x1.5 | 0.62 | 17.51 deg |

A null edit that changes the score by 13 deg is measuring the border, not the content: the mask
is identical for every condition of a frame, so the border bias is a fixed offset that both
controls inherit. Subtracting it gives a coherent picture (removal +9.3 deg, mirror +8.3, yaw
control -0.5), but a statistic whose controls reject at 0.59-0.62 is invalid under the
pre-registered invalidation clause ("null-edit or yaw control rejected above the real rate plus
5 points: the pixel-edit family is declared invalid").

Fix: **crop** every condition of a frame to the largest centred rectangle that lies fully
inside the common mask, instead of black-filling the rest. Every pixel the estimator sees is
then a real pixel, all conditions share the crop, and the crop is applied to the null edit
identically. GeoCalib receives no intrinsics for variant b, so a centred crop costs nothing but
resolution. This is the version all Task 5 numbers below come from.
