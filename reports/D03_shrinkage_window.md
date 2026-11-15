# Report D03 — Shrinkage diagnosis, per-type margins, flight-level FRR, window sensitivity

Written 2026-11-14. Companion to `reports/D02_gain_test_v2.md`. Runs:
`results/d_shrinkage/d01/`, `results/d_calibrate/d01/`,
`results/d_eval/d02/margins.csv`, `results/d_flights/d01/`,
`results/d_window/d01/` (once the window rerun completes). The frozen configs were **not**
changed; the validation status is amendment 2 in `configs/gain_prereg.yaml`, and the
window proposal is `configs/window_amendment_proposal.yaml` for user approval.

## 1. What kind of shrinkage is it?

Per frame, depicted lean against truth-attitude lean, after the gain fit's lag, on the 57
testable real clips (2735 frames). OLS slope, R² and the residual (noise) share:

| type | frames | slope x | slope y | scalar R² | noise share |
| --- | --- | --- | --- | --- | --- |
| piloted | 2401 | **1.000** | **0.792** | **0.988** | **1.2 %** |
| autonomous | 191 | 0.777 | 0.494 | 0.748 | 25.2 % |
| track | 143 | 0.312 | 0.651 | 0.732 | 26.8 % |

By truth-tilt bin, **piloted** (the only type with enough frames below 45°):

| tilt bin | n | slope x | slope y | R² |
| --- | --- | --- | --- | --- |
| 0-5° | 34 | 1.005 | 0.703 | 0.956 |
| 5-10° | 31 | 0.926 | 0.657 | 0.962 |
| 10-20° | 476 | 1.047 | 0.763 | 0.984 |
| 20-30° | 1254 | 0.951 | 0.764 | 0.995 |
| 30-45° | 565 | 0.905 | 0.794 | 0.988 |
| 45-90° | 41 | 1.075 | 1.009 | 0.897 |

**Diagnosis: linear, axis-specific shrinkage, not saturation and not (on piloted flights)
noise.** The pitch/forward axis is unbiased (0.91-1.08 across the populated bins, no
trend); the roll/lateral axis is compressed by 0.66-0.79 across the populated bins (0.76-
0.79 over the 10-45° bulk) with no monotone decline with tilt.
The scalar R² is 0.988 and only 1.2 % of the variance is residual noise. Autonomous and
track are a different regime: their testable clips sit at high tilt, their slope is lower
and 25-27 % of their variance is noise, so compression and noise cannot be separated with
4 and 3 clips. Track is not validated; the 0.31 pitch slope is not evidence of a physical
compression.

Plot: `results/d_shrinkage/d01/shrinkage_scatter.png` and
`slope_by_tilt_bin.png`.

### Does a monotone calibration curve help?

Because the piloted compression is monotone (linear on the roll axis), two calibration
curves were fitted on the fit flights only and applied leave-one-flight-out: a **single**
isotonic (PAVA) curve on the pooled |lean| magnitudes of both axes, and **per-axis**
curves. The gain statistic was then refitted on the calibrated depicted lean with the same
nuisance settings.

| method / rule | FRR | removal | mirror | x0.5 | x1.5 | yaw control |
| --- | --- | --- | --- | --- | --- | --- |
| uncalibrated R2 | 5.3 % | 100 % | 100 % | **100 %** | **96.5 %** | 28.1 % |
| single curve R2 | 5.7 % | 100 % | 98.5 % | **59.0 %** | **60.1 %** | 12.7 % |
| per-axis curve R2 | 6.4 % | 100 % | 100 % | 95.4 % | 78.1 % | 17.3 % |
| uncalibrated R1 | 10.5 % | 100 % | 98.2 % | 100 % | 98.2 % | 19.3 % |
| single curve R1 | 6.1 % | 98.2 % | 98.5 % | 85.5 % | 97.8 % | 15.6 % |
| per-axis curve R1 | 6.8 % | 100 % | 97.8 % | 60.3 % | 94.7 % | 19.1 % |

At the **matched 5.3 % false-reject** the single R2 curve detects x0.5 40.6 % and x1.5
45.0 % against 100 / 96.5 % uncalibrated; the per-axis R2 curve is about 91 / 75 %. The
single R1 curve at its 6.1 % point is 85.5 / 97.8 %; matched to 10.5 % it is about
91 / 98 % against 100 / 98.2 %. **No calibration curve improves detection at matched
false-reject.** It reduces the yaw-control artifact (28.1 → 12.7 % for the single curve)
but that is a cost, not a gain: the calibration is mostly undoing the roll-axis linear
compression, which the R2 axis band already absorbs as a reparameterization, and the
isotonic curve adds distortion on the edited distributions. The uncalibrated statistic
stays frozen. This is also the direct empirical confirmation of the linear-shrinkage
diagnosis: a monotone recalibration cannot create separation that the axis-specific band
did not already have.

## 2. Per-flight-type real distributions and margins (frozen R2)

The rates below apply the **pooled frozen band**, which is what a deployment uses. The
D02 per-type table used per-fold leave-one-flight-out bands and gives piloted **2.0 %**,
autonomous 25 %, track **33.3 %**; both are reported because the track margins are within
0.4 of the band and one clip flips the rate. `margin_R2 = T − tau` with `T = max(|gx−c_x|/s_x, |gy−c_y|/s_y)`,
`c_x=0.8137, s_x=0.2654, c_y=0.7181, s_y=0.0735, tau=2.1922`; `margin_corr = (1−r_min) −
0.6197`. Positive margin = flagged.

| type | n | G1 median (IQR) | gx median (IQR) | gy median (IQR) | r_min median (IQR) | R2 FRR | coherence FRR | margin_R2 median | margin_corr median |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| piloted | 50 | 0.770 (0.739-0.823) | 0.838 (0.774-0.971) | 0.725 (0.673-0.754) | 0.896 (0.795-0.934) | 4.0 % | 6.0 % | −1.36 | −0.52 |
| autonomous | 4 | 0.645 (0.627-0.659) | 0.642 (0.609-0.664) | 0.753 (0.645-0.966) | 0.629 (0.564-0.642) | **25.0 %** | 0.0 % | −0.59 | −0.25 |
| track | 3 | 0.541 (0.525-0.543) | **0.242** (0.238-0.325) | 0.627 (0.605-0.652) | 0.642 (0.581-0.659) | 0.0 % | 0.0 % | −0.04 | −0.26 |

The pooled-band rate is 4.0 % on the 50 piloted clips (LOFO: 2.0 %). Track is **not
validated**: its gx is
0.24 against a band centre of 0.81, but the band's pitch scale (0.265) lets it sit just
inside (margins −0.007 to −0.35). That is fragility, not validation.

### The autonomous clips, one by one

The decomposition's **3-clip** autonomous set is 01a#02, 01a#03, 04a#03. The frozen d02
testable set has a fourth, 04a#04, because it moved in when the dense required lean
changed the confidence interval. All four:

| clip | G1 | gx | gy | T | margin_R2 | r_min | margin_corr | R2 flag |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 01a#02 | 0.652 | 0.675 | 0.654 | 0.871 | −1.322 | 0.634 | −0.253 | no |
| **01a#03** | 0.639 | 0.661 | **1.312** | **8.080** | **+5.888** | 0.383 | **−0.003** | **yes** |
| 04a#03 | 0.591 | 0.562 | 0.851 | 1.809 | −0.384 | 0.624 | −0.244 | no |
| 04a#04 | 0.678 | 0.624 | 0.616 | 1.388 | −0.804 | 0.666 | −0.286 | no |

On the 3-clip set the R2 false-reject is **1/3 = 33 %**; on the frozen 4-clip set it is
**1/4 = 25 %**. The flagged clip is a roll-axis outlier (gy = 1.31, z = 8.1); the other
three sit inside the band. Coherence does **not** flag any of them, and 01a#03 is 0.003
from the correlation threshold — the two statistics disagree on the same clip. That is
the definition of "no calibration evidence", not "infeasible".

### Decision: not validated, not abstain

**Autonomous and track flights are marked `validation_status: not_validated` by rule, not
abstained.** They have enough excitation and confidence to produce a graded result; what
they lack is calibration evidence (4 and 3 testable clips, 25 % and 33 % false-reject,
100 % and 67 % yaw-control flag rates). Abstaining would hide a measurable result;
`not_validated` keeps the numbers but sets `usable_as_evidence: false`, so a flag on an
autonomous clip cannot be presented as a violation and a pass cannot be presented as
consistency. The rule is amendment 2 in `configs/gain_prereg.yaml` (no refit; thresholds,
bands, abstention and coherence thresholds unchanged). `scripts/score_clip.py` now emits
`validation_status` and `usable_as_evidence`; `flight_type` is written into the clip
metadata by `experiments/d_build_clips.py`. Track E must set `flight_type` in its
`meta.json` for the rule to apply.

## 3. How many independent flights are behind 5.3 %?

**3 false rejects in 57 clips from 6 independent flights** (01a, 04a, 01p, 07p, 13p, 16p).
Three of the six flights contribute a false reject; the 57 clip count overstates the
independent evidence because 50 clips come from two piloted flights.

| flight | testable clips | false rejects | rate |
| --- | --- | --- | --- |
| 01a-ellipse | 2 | 1 | 50.0 % |
| 04a-ellipse | 2 | 0 | 0 % |
| 01p-ellipse | 26 | 1 | 3.8 % |
| 07p-lemniscate | 24 | 1 | 4.2 % |
| 13p-trackRATM | 2 | 0 | 0 % |
| 16p-trackRATM | 1 | 0 | 0 % |

Flight-cluster bootstrap (resample whole flights, 20,000 draws): pooled rate **5.26 %**,
95 % interval **[2.86 %, 20.0 %]**. The unweighted mean of the six per-flight rates is
**9.7 %**, 95 % interval **[0.6 %, 26.3 %]**. The honest statement is therefore
"3/57 clips, 3/6 flights; flight-level 95 % interval roughly 3-20 %", not "5.3 % with a
clip-level Wilson interval of 1.8-14.4 %". `results/d_flights/d01/interval.json`.

## 4. Required-lean window sensitivity (proposal only)

Track E found the UZH verdict depends on the required-lean window. On TII dense mocap the
window was rerun at 0.10, 0.185, 0.37 and 0.74 s, with the gain statistic and nuisance
settings unchanged, on the **fixed frozen d02 population** (57 clips) so only the window
changes. Bands are re-fitted leave-one-flight-out; abstention is reported separately from
the window's own refit.

| window | samples | window abstain | R2 band gx / gy / tau | FRR | removal | mirror | x0.5 | x1.5 | yaw control |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.10 s | 27 | 41.0 % | 0.739 / 0.703 / 2.652 | 5.3 % | 100 % | 100 % | **64.9 %** | **64.9 %** | **12.3 %** |
| 0.185 s | 51 | 41.0 % | 0.818 / 0.716 / 2.596 | 5.3 % | 100 % | 100 % | 100 % | 96.5 % | 21.1 % |
| 0.37 s (frozen) | 102 | 43.0 % | 0.814 / 0.718 / 2.192 | 5.3 % | 100 % | 100 % | 100 % | 96.5 % | 28.1 % |
| 0.74 s | 203 | 43.0 % | 0.703 / 0.724 / 2.312 | **12.3 %** | 100 % | 100 % | 100 % | 93.0 % | 26.3 % |

**Band shift.** The frozen 0.37 s band applied unchanged to the other windows gives
false-reject 15.8 % (0.10 s), 8.8 % (0.185 s), 5.3 % (0.37 s) and 17.5 % (0.74 s). A
window change requires its own conformal band; the bands do not transfer across windows.

**Coherence is window-insensitive.** At the frozen tau 0.6197 / 0.2921 the correlation
tier's false-reject is 1.8 / 1.8 / 5.3 / 8.8 % for 0.10 / 0.185 / 0.37 / 0.74 s, with
removal 98.2-100 %, mirror 100 %, x0.5 17.5-19.3 %, x1.5 22.8-24.6 % and yaw 8.8-12.3 %.
The shape tier does not need the window changed for its own sake.

**Principle: manoeuvre bandwidth, not sample rate.** The window is a physical low-pass on
the commanded acceleration and must preserve the manoeuvre bandwidth; the sample rate only
decides whether the window is resolvable. A sample-rate rule would pick 0.10 s on dense
TII but ~0.6 s on 10 Hz UZH, which is the wrong direction. The proposed rule is: choose
the window from the manoeuvre bandwidth, require at least 8 samples, and mark the clip
not testable (rather than lengthening the window) if the floor is not met.

**Rules that pick a window without the test dataset.** The **truth-gain rule** (on fit
flights, the window whose truth-attitude gain is closest to 1) picks **0.37 s in all 8
folds**. The fixed **physical 0.10 s** rule and a **sample-rate floor** pick 0.10 s on TII.
None uses the test edits. The 0.185 s recommendation is a policy compromise informed by
the TII window sensitivity, so it is written as a proposal, not applied.

**Proposal.** `configs/window_amendment_proposal.yaml`: adopt **0.185 s** with an 8-sample
floor, re-fit the R2 band (`gx` 0.8176 / 0.2586, `gy` 0.7159 / 0.0726, tau 2.5956) and
re-freeze. It keeps detection, cuts the yaw artifact from 28.1 % to 21.1 % and adds two
testable clips. 0.37 s remains frozen until approved.

## 5. Bundle restore from R2

The bundle was downloaded from R2, cloned into a scratch directory, and verified:

```
sync_r2 --get track-d/track-d-verdict-stat.bundle --dest /tmp/awmv-restore-final/
cd /tmp/awmv-restore-final && git clone track-d-verdict-stat.bundle restored
cd restored && git rev-parse HEAD          # 038aed7
git rev-parse --abbrev-ref HEAD            # track-d-verdict-stat
git fsck --full                            # clean
pytest tests/test_verdict.py -q            # 41 passed
```

The R2 object's sha256 matched the local upload (`8cae4087…5f6e`). The bundle is created
with `HEAD track-d-verdict-stat`, so a **plain `git clone`** checks out the branch; a
bundle created without `HEAD` needs `git clone -b track-d-verdict-stat`. The final bundle
is re-created after this report commit; the upload log records its sha256, and the chat
reports the final HEAD. Push is still blocked by credentials, so this bundle is the
recovery path.
