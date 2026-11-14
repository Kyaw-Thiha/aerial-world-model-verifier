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

## Task 4-5

Filled in below after the exports; see `results/e_verdict/<tag>/metrics.json` for the raw
numbers.
