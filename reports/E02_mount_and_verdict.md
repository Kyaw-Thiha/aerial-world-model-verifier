# E02 — Mount check, verdict clips and the frozen-threshold transfer test

Track E. Written 2026-11-14. Pre-registration: `configs/e_verdict_prereg.yaml` (committed before
any edited clip was scored). Depends on `reports/E01` (data audit and per-frame validation).
Artifacts: `results/e_mount/mount.json`, `results/e_clips/index.json`,
`results/e_verdict/<tag>/metrics.json`.

## Plain-language summary

1. **The published mount is right for the forward-facing rigs** (fitted rotation 0.7-3.4 deg,
   held-out residuals improve under the fit) and **wrong or view-biased by 14 deg for the
   45-degree-down rigs**. A constant rotation fitted on the outdoor rig does not generalise,
   which rules out "the mount explains everything" — the same conclusion that killed the
   earlier attempt's fitted mount, now with numbers.
2. **The clip pipeline works**: schema-validated 4 s clips, exact rotation edits, the null edit
   reproduces the real clip, and the exported real clips agree with the independent per-frame run
   to 0.046 deg. Getting there required finding three defects in this task's own code (a
   clip-wide border mask, black-fill borders, and a broken fisheye inversion for 7 of the 14
   cameras); each is documented with the number that exposed it.
3. **T2 fails on false-reject**: 0.333 against a 0.15 budget, with detection at 0.881 for both
   removal and mirrored roll against a 0.80 budget. **The failure is localised**: on the
   forward-facing rigs the same frozen threshold gives 0.06-0.11 false reject and 0.78-0.89
   detection; the 45-degree-down rigs are rejected wholesale (0.75 and 1.00) because the
   estimator's own error there is as large as the threshold.
4. **The cause is a scale mismatch, not a broken estimator.** Real UZH clips score a median of
   7.72 deg in this mode where TII's score 2.47, so a threshold frozen at 14.73 deg sits at two
   UZH standard deviations. Refitting the threshold on UZH (sensitivity) makes the test blind
   (detection 0.07-0.12), and enabling A2's mount-offset nuisance only takes false reject to
   0.19. A threshold frozen on TII is simply too tight for this dataset, and the honest reading
   is that **the verifier transfers on forward-facing UZH footage and does not transfer as a
   whole**.

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

**The same probe failed on `outdoor_forward_5` with 25.9 deg on the null edit**, and the first
reading of that was "the estimator is unstable on fast footage". It was not: it was a bug in
this task's own fisheye inversion. The Kannala-Brandt polynomial of *six* of the fourteen
cameras is **non-monotone inside the frame** (θ_d turns over at 94 deg for the indoor-45 rig and
at 102 deg for the outdoor-forward rig), and a lookup built over the whole range then folds
radii back on themselves. The result is a radial smear, which the estimator reads as an
arbitrary gravity. The tell was visual: the null edit of an indoor-45 clip was a radial blur, not
the same picture.

Fix (`src/depicted/edits.py`): build the inversion only on the increasing branch of θ_d and mark
rays beyond it as outside the model. After the fix the identity warp round-trips to 0.000 px on
all fourteen cameras. Seven sequences (4 indoor-45, 3 outdoor-forward) were re-exported; the
seven others had monotone polynomials and are unchanged. Cost of the mistake: one discarded clip
family (~US$1.5 of GPU) and the re-export (~US$0.8).

This also means the claim "the estimator degrades outdoors" in E01 stands on its own evidence
(the per-frame error, which never involved a warp) and is **not** the explanation for the control
failure here.

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

Statistic: Track A2's median axis residual, drag-free path side, one lag nuisance per clip
(-150 to +150 ms, 25 ms steps), mount fixed at the published rotation, no uptilt offset.
Threshold **14.73 deg**, A2's pooled drag-free lag-only operating point, **not refit on UZH**.
42 clips (3 per sequence after the excitation rule), 294 files, schema-validated, all testable
(abstention 0 because the excitation rule is applied at export time and the crop leaves every
frame usable).

### Per condition, unmatched

| condition | n | rejected | rate | Wilson 95% | median score |
| --- | --- | --- | --- | --- | --- |
| real | 42 | 14 | **0.333** | 0.21-0.48 | 7.72 |
| null edit | 42 | 14 | 0.333 | 0.21-0.48 | 8.34 |
| yaw control | 42 | 15 | 0.357 | 0.23-0.51 | 9.16 |
| tilt removal | 42 | 37 | **0.881** | 0.75-0.95 | 23.88 |
| mirrored roll | 42 | 37 | **0.881** | 0.75-0.95 | 26.77 |
| tilt x0.5 | 42 | 20 | 0.476 | 0.33-0.62 | 14.63 |
| tilt x1.5 | 42 | 19 | 0.452 | 0.31-0.60 | 14.26 |

### Paired (population: the 28 clips whose real condition was accepted)

| condition | n | detected | rate | Wilson 95% |
| --- | --- | --- | --- | --- |
| tilt removal | 28 | 23 | **0.821** | 0.64-0.92 |
| mirrored roll | 28 | 23 | **0.821** | 0.64-0.92 |
| tilt x0.5 | 28 | 7 | 0.250 | 0.13-0.43 |
| tilt x1.5 | 28 | 7 | 0.250 | 0.13-0.43 |
| yaw control | 28 | 3 | 0.107 | 0.04-0.27 |
| null edit | 28 | 1 | 0.036 | 0.01-0.18 |

The two controls behave: the null edit rejects at exactly the real rate (14 of 42, median 8.34
against 7.72), and the yaw control is within 2 points of it. Whatever remains above the real
rate is resampling, not tilt, and it is reported rather than netted out.

### Where the false rejects are

| rig | real clips | false reject | removal | mirror | real-score median (all 14 sequences) |
| --- | --- | --- | --- | --- | --- |
| indoor_forward | 18 | **0.06** | 0.78 | 0.78 | 4.32 |
| outdoor_forward | 9 | **0.11** | 0.89 | 0.89 | 5.43 |
| indoor_45 | 12 | **0.75** | 1.00 | 1.00 | 17.81 |
| outdoor_45 | 3 | **1.00** | 1.00 | 1.00 | 43.96 |

The forward rigs read like a working verifier: 6-11% false reject against a 15% budget, 78-89%
detection of removal and mirror. The 45-down rigs are rejected wholesale, which makes their
100% detection meaningless — every condition is above the threshold there, including the null
edit (median 18.05 and 43.02).

### The verdict against T2

**T2 fails: false reject 0.333 (budget 0.15); detection of removal 0.881 and of mirrored roll
0.881 (budget 0.80).** Detection passes, false reject does not, and the failure is concentrated
on the 45-degree-down rigs and on the noisiest outdoor sequence. On the forward-facing rigs the
same frozen threshold gives 0.06-0.11 false reject and would pass.

The reason is measurable rather than mysterious: on TII the real clips score a median of
2.47 deg in this mode; on UZH they score 7.72 (4.32 indoors forward), so a 14.73 deg cut sits at
about two of UZH's own standard deviations instead of four of TII's. The threshold was frozen on
footage whose depicted-tilt error is half of UZH's.

### Sensitivities (labelled, never the headline)

* **Mount-pitch offset nuisance enabled**, A2's grid of -7 to +7 deg, 1 deg steps:
  false reject 0.19, removal 0.69, mirror 0.86. Per rig, the fitted offset is 0 deg on both
  forward rigs (false reject 0.00 on each) and pins at +7 deg on both 45 rigs (false reject 0.42
  and 1.00). That is the 14 deg discrepancy of the mount section showing up again from an
  independent direction: the forward mounts are right and the 45-degree ones are off by more
  than the grid allows.
* **Frozen S3 gate applied** (real clips gated, decision inherited by their edits): 10 of 42 real
  clips dropped; on the surviving 32, removal 0.875 and mirror 0.875. The gate removes the
  outdoor and 45-rig clips, i.e. it does its job, at a 24% coverage cost.
* **Thresholds recalibrated on UZH** by leave-one-sequence-out (the sensitivity the handoff asks
  for): detection falls to 0.071 (removal) and 0.119 (mirror). A UZH-fitted 95th percentile of
  real scores is above 20 deg, so it hides the violations. Refitting does not rescue the test;
  it makes it blind.
* **Abstention**: 0 for every condition, against TII's 41%. T4 (abstention at most 10 points above
  the current statistic's rate) is trivially satisfied, and that is itself a warning: with no
  gate and every clip testable, the statistic's failures are rejections rather than abstentions.

### Graded output (T5) coverage

Per clip and condition the run emits required tilt (median), depicted tilt (median), the residual
score, the fitted lag, a per-frame valid fraction, the depicted confidence, and the testable flag
with its reasons — `results/e_verdict/<tag>/metrics.json`, field `rows`. It does **not** emit a
gain with an interval or a drag estimate: those belong to Track D's gain statistic, which is not
published (5b not run). No number in this report is a gain.

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

### T6: can a scorer reproduce this from the shared schema alone?

The schema carries `pos` (60 samples at 15 Hz per 4 s clip) and `yaw`, not the 500 Hz truth
table, so a scorer that only sees `verdict_clip.npz` must differentiate `pos` twice to recover
the required thrust axis. Doing exactly that (`--path-source schema`) changes the picture:

| condition | path side from the 500 Hz truth | path side from the clip's own `pos` |
| --- | --- | --- |
| real | 1.54 deg | **7.40 deg** |
| null edit | 1.54 | 7.43 |
| yaw control | 1.19 | 6.68 |
| tilt removal | 12.33 | 14.78 |
| mirrored roll | 10.63 | 12.61 |
| tilt x0.5 | 6.22 | 11.13 |

Two things follow. **The ranking survives** — the controls sit at the floor in both readings,
and the edits sit above it — so the schema is sufficient in principle. But the **floor rises by
about 6 deg**, because a second derivative of 15 Hz positions with a 0.46 s window carries much
more noise than the 500 Hz truth does with a 0.10 s window. Against a 14.73 deg threshold that
is the difference between a usable test and a marginal one. Recommendation to whoever owns the
shared schema (Track D): either carry a denser `pos` (or the required axis itself) in the npz,
or fix the path-side window in the schema. Track E reports every headline number on the dense
truth and quotes this as the schema-only sensitivity.

## Why the earlier UZH pilot read 86-89 deg

The handoff left two candidates: the estimator does not transfer to UZH footage, or something
in the truth, frames or mount handling is wrong. The measurements above settle it.

* The estimator **does** transfer: 1.5-4.6 deg median per frame on the forward rigs (E01), and
  the exported real clips score 1.5-5.6 deg against the frozen 14.73 deg threshold.
* The mount **is** published, per rig, and the published reading is confirmed indoors to
  0.7-3.4 deg.
* Therefore a verdict residual of 86 deg cannot come from the depicted estimator.

**The one transposition this pipeline is one line away from produces exactly that number.**
While building this task the camera->body rotation was applied in the wrong direction
(`R_cb @ up_body` instead of `R_cb^T @ up_body`) in one function. The body-frame agreement was
unaffected (1.3-4.0 deg) and the camera-frame error jumped to **92-118 deg** — the same
magnitude as the pilot's 86-89 deg, on the same dataset, with everything else correct. It was
caught within one run because the same quantity was computed in the body frame as a cross-check
(`tests/test_depicted_uzh.py` now asserts the direction). A pilot that (a) had no published
mount, (b) fitted one from truth, and (c) reported a ~56 deg residual after the fit is
consistent with a mount *convention* error that a fit cannot remove, which is what the E02 mount
table shows independently: a fitted rotation on the outdoor rig does not generalise.

This is offered as the most likely explanation with the evidence that makes it plausible, not as
a proven diagnosis of another agent's code — that code was not re-run here.

## Cost, reproduction, and what is left

**Cost (measured from container wall time).** Task 2 GeoCalib on 14 sequences x 3 variants:
about 71 GPU-minutes = **US$1.3**. Clip export, three times: black-fill family (discarded)
US$1.5, cropped family US$1.4, cropped re-export of 7 sequences US$0.8; validation and test
runs US$0.3. **Total about US$5.3**, i.e. slightly over the "about US$5" approval line, because
two clip families were thrown away after the controls exposed them. Everything else (truth
audit, clock alignment, mount checks, verdict scoring) ran on CPU locally.

**Reproduction.**

```
pixi run modal run remote/e_sync.py                 # clocks (or experiments/e_clock.py locally)
pixi run modal run remote/e_prepare.py              # frames + truth -> awmv-e-data
pixi run modal run remote/e_geocalib.py --parallel  # Task 2, three variants
python experiments/e_uzh_audit.py                   # Task 1 audit
python experiments/e_perframe.py --tag v1_all       # Task 2 tables
python experiments/e_mount.py --perframe results/e_perframe/v1_all   # Task 3
pixi run modal run remote/e_export.py --variant b --max-clips 3 --parallel   # Task 4
python scripts/export_verdict_clips.py --clips data/clips --r2       # validate + publish
python experiments/e_verdict.py --clips data/clips --tag primary     # Task 5
```

**Open items, in the order they would matter.**

1. The 14 deg 45-rig discrepancy is the single largest unexplained number in this study. It
   costs the whole 45-down rig in the verdict test. Resolving it needs an estimator-independent
   check that the published 45-rig calibration is right — a Kalibr re-calibration from the
   released calibration bags, or a horizon check on a sequence where the camera sees the
   building's verticals. Neither was attempted here.
2. A threshold that matches UZH's noise floor. T2's false-reject budget cannot be met with a
   TII-frozen threshold given a 7.7 deg median real-clip score; the honest options are to
   pre-register a per-dataset threshold with its own coverage accounting, or to report the
   smallest violation detectable at the achieved false-reject rate instead of a fixed budget.
   Both are Track D's call, not Track E's.
3. Track D's gain statistic (5b) is unrun: `track-d-verdict-stat` publishes neither
   `configs/gain_frozen.yaml` nor `scripts/score_clip.py`. The clips, the schema and the index
   are in R2 (`track-e/clips/`, 589 objects) ready for it.
4. The shared schema needs a denser `pos` (or the required axis) for T6 to be satisfiable
   without a 6 deg penalty; see the T6 section.
