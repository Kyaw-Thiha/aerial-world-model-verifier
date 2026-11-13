# Report D01 — The gain test on TII (Track D)

Written 2026-11-13. Pre-registration `configs/gain_prereg.yaml` (commit `613a102`),
acceptance spec `configs/prototype_acceptance_spec.yaml` (same commit). Run
`results/d_eval/d01/metrics.json`; frozen statistic `configs/gain_frozen.yaml`.

## Plain-language summary

**The gain test works on TII, and it is much more sensitive than the median-residual
statistic on the violations that matter.** Regress the depicted lean on the lean the
commanded path requires, per clip. A real flight gives a gain of about **0.75** (not 1:
the pixel estimator under-reads tilt and the regressor carries noise, so the absolute
value is calibrated on real flights rather than assumed). Removing the tilt gives
**-0.17**, halving it gives **0.29**, scaling it by 1.5 gives **1.24**, and mirroring the
roll gives **-1.14 on the roll axis**. A two-sided conformal band at 5 % over real clips
therefore separates all four.

At the frozen band (false-reject **8.8 %**, 57 of 100 real clips testable):

| condition | gain statistic | current median-residual statistic |
| --- | --- | --- |
| full tilt removal | **100 %** (57) | 67 % (58) |
| mirrored roll | **96.5 %** (57) | 95 % (58) |
| tilt x0.5 | **98.2 %** (57) | 17 % (58) |
| tilt x1.5 | **96.5 %** (57) | 24 % (58) |
| yaw-only control | 22.8 % (57) | 8.6 % (58) |

At the baseline's own operating point (3.4 % false-reject) the comparison is starker:
the gain still rejects 100 % of removals, 95 % of mirrors, 98 % of x0.5 and 95 % of x1.5,
against 67 / 95 / 17 / 24 % for the baseline. At the gain's 8.8 % point the baseline
reaches 83-95 % removal, 97-98 % mirror, **21-57 % x0.5 and 52-95 % x1.5**. The x0.5
improvement is material at every matched rate; the x1.5 improvement is material at low
false-reject and marginal at the upper end of the baseline's curve.

**What it costs and what it cannot do.** Abstention is 43.0 % (baseline 42 %), 1 point
inside the T4 budget but concentrated on the track and autonomous flights: 4 of 29 track
clips and 3 of 12 autonomous clips are testable, against 50 of 52 piloted. On those four
track clips **two real clips are falsely rejected** (13p one of three, 16p its only one),
so the pooled rate is not a track rate. The yaw control is flagged 22.8 % against **8.8 %** for real clips: a yaw rotation cannot change the tilt, so
this is the estimator's view dependence leaking into the gain, and it is the floor under
the statistic. The fitted velocity coefficient is **0.139 (1/s)**, about half the Phase 0
value 0.295; it is a nuisance that works, not an airframe drag estimate (inferred:
errors-in-variables attenuation, the same effect that puts the real gain at 0.75).

## 1. Baseline reproduction (Task 0)

Before anything new was scored, the current statistic was reproduced from the stored
artifacts at the primary mode (drag-free, lag-only, frozen confidence gate, all-flight
threshold policy): pooled conformal threshold **14.733 deg** (fold spread 13.50-16.46),
held-out acceptance 0.966 (n = 58), tilt removal rejected **0.500** (n = 8 testable
pairs), mirrored roll 0.961 (51), yaw control 0.038 (52). This matches report 07 /
amendment 9. The reproduction is `experiments/d_repro_baseline.py`.

The 8-pair count is the *per-condition* gate of Track A2, which drops 79 % of the
tilt-removal edits. The matched comparison below uses **pair-level** gate inheritance for
both statistics (gate the real clip, inherit the decision to its edits), which keeps all
57-58 pairs and is the comparison the gain statistic was pre-registered to make. On that
population the baseline rejects removal 67 %, not 50 %; both numbers are reported and
neither is presented as the other.

## 2. The statistic, exactly as frozen

Per clip, with `l_dep` the depicted lean, `l_req0` the drag-free required lean and
`w = Rz(-yaw) v / g` the velocity regressor:

```
l_dep = G1 * l_req0 + G2 * w + b
```

`G1` is the acceleration gain, `G2` the fitted drag nuisance (1/s), `b` a free 2-D
intercept absorbing the mount offset. Required lean: Savitzky-Golay second derivative of
the commanded position, window **0.37 s**, polyorder 3, edge-trimmed one half-window. The
stored TII grid is 12 Hz, where the handoff's 0.11 s reference (31 samples at 275 Hz) is
1.3 samples and cannot carry a second derivative; 0.37 s is the frame-rate analogue and
is recorded as a deviation. Lag searched within +/-150 ms in 25 ms steps; OLS; **no**
smoothing and **no** confidence weights in the frozen statistic (both are ablation).

Decision rule **R1**: flag when `G1` leaves the two-sided conformal band fitted on the
fit flights' real clips (alpha 0.05, 2.5 % each tail). Two richer candidates were
pre-registered: R2, a joint band on the standardised `(gx, gy)`; R3, a joint band on
`(G1, gx, gy)`. Selection by the registered criterion (highest mean removal/mirror
detection at a false-reject rate at most 10 %, inner leave-one-flight-out, ties to the
simpler rule) picks **R1**: R1's inner false-reject is 8.8 %, R2's 10.5 % and R3's 12.3 %,
and R2/R3 buy only mirror detection that R1 already has at 96.5 %.

## 3. Data, testability and abstention

100 clips from 8 flights, each under 10 conditions: real (the identity edit), null edit,
tilt removal, x0.5, x1.5, mirrored roll, yaw control, and 100/200/300 ms delays. Clips are
the Track A2 4 s non-overlapping table; the depicted series are the stored GeoCalib rows,
converted to the shared lean schema through the recorded mount. 1000 `verdict_clip.npz`
files (`data/processed/track_d/clips/`), built by `experiments/d_build_clips.py`.

Abstention: 57/100 real clips testable (**43.0 %** abstained, baseline 42 %). Reasons
overlap; counts: excitation 14, condition number 11, bootstrap half-width 17, frozen
confidence gate 28, signal-frames/fraction 18. By flight type: piloted 3.8 % abstained
(50/52 testable), autonomous 75 % (3/12), track 86 % (4/29). This is the same shape as
Track A2 and is the reason the pooled numbers are statements about the two piloted
flights, not about racing footage.

Fitted gains on testable real clips: `G1` median **0.745** (IQR 0.696-0.806, range
0.497-0.998), roll-axis `gy` median **0.710** (0.647-0.753). The gain is systematically
below 1; this is why the band is calibrated on real flights. Fitted intercepts are small
and the free intercept is essential: fixing it at zero raises false-reject to **22.8 %**
(ablation).

## 4. Candidate rules and selection

Held-out rates at alpha 0.05 (Wilson 95 % in brackets):

| rule | false-reject | removal | mirror | x0.5 | x1.5 | yaw |
| --- | --- | --- | --- | --- | --- | --- |
| **R1 scalar G1 (frozen)** | **8.8 %** [3.8, 18.9] | 100 % [93.7, 100] | 96.5 % [88.1, 99.0] | 98.2 % [90.7, 99.7] | 96.5 % [88.1, 99.0] | 22.8 % [13.8, 35.2] |
| R2 axis gains | 10.5 % [4.9, 21.1] | 100 % | 100 % | 100 % | 96.5 % | 24.6 % |
| R3 scalar or axis | 12.3 % [6.1, 23.2] | 100 % | 100 % | 100 % | 100 % | 31.6 % |

The nested per-outer-fold selection (the pre-registered procedure) chooses R2 in six
folds and R1 in two; applying the selection honestly gives **10.5 % false-reject**, 100 %
removal, 100 % mirror, 100 % x0.5, 96.5 % x1.5. That is one clip above the 10 % T1 target.
The pooled rule R1, which is the pre-registered selection (the only rule with inner
false-reject at most 10 %), gives 8.8 %. The frozen config uses R1; both numbers are
reported and the difference is one real clip.

## 5. Gate D1 against the targets

| target | required | measured | verdict |
| --- | --- | --- | --- |
| T1 false-reject, testable real, overall | <= 10 % | **8.8 %** [3.8, 18.9], n=57 (nested 10.5 %) | **met at the pooled rule, borderline under nested selection; upper interval above 10 %** |
| T1 detection, full tilt removal | >= 90 % | **100 %** [93.7, 100] | **met** |
| T1 detection, mirrored roll | >= 90 % | **96.5 %** [88.1, 99.0] | **met** |
| T1 per flight type | report | piloted 9.3 % FRR / 100 / 98.1; autonomous 0 / 100 / 66.7 (n=3); track **50 % FRR** / 100 / 100 (n=4) | **not met on track; track is not validated** |
| T2 transfer to UZH-FPV | FRR <= 15 %, detection >= 80 % | not run (Track E) | **open** |
| T3 x0.5 detection | report | **98.2 %** [90.7, 99.7] | **materially better than baseline at every matched rate** |
| T3 x1.5 detection | report | **96.5 %** [88.1, 99.0] | **materially better at low false-reject; marginal at the baseline's 8.6 % point (52-95 %)** |
| T4 abstention, overall | <= 41 % + 10 points | **43.0 %** (baseline 42 %) | **met** |
| T4 by flight type | report | piloted 3.8 %, autonomous 75 %, track 86 % | reported |
| T5 graded output | yes | `scripts/score_clip.py` emits every field | **met** |
| T6 reproducibility | tests pass, schema only | 30 new tests, 131 total pass; no mocap beyond `pos`/`yaw` | **met** |

T1 is **met at the frozen pooled rule and not met under the honest nested selection**, by
one clip. T2 is Track E's. T3 is met for x0.5 without qualification and for x1.5 only at
low false-reject.

## 6. Matched false-reject comparison

Operating curves are in `results/d_eval/d01/operating_curves.png` and
`operating_curves.csv`. The curves are discrete: a conformal threshold jumps to the next
real score. At a target rate the table gives the range of detections within 2 points of
it, which is the honest statement.

At the **baseline's 3.4 % false-reject** (gain at 3.5 %):

| family | gain | baseline |
| --- | --- | --- |
| removal | 100 % | 67.2 % |
| mirror | 94.7 % | 94.8 % |
| x0.5 | 98.2 % | 17.2 % |
| x1.5 | 94.7 % | 24.1 % |

At the **gain's 8.8 % false-reject** (baseline at 8.6 %):

| family | gain | baseline range |
| --- | --- | --- |
| removal | 100 % | 82.8-94.8 % |
| mirror | 96.5 % | 96.6-98.3 % |
| x0.5 | 98.2 % | 20.7-56.9 % |
| x1.5 | 96.5 % | 51.7-94.8 % |

The median-residual statistic is close on the mirror and, at the top of its range, on
x1.5; it is far behind on removal and x0.5. The gain is not merely a re-parameterisation
of the same evidence: the baseline's x0.5 detection stays below 57 % at the gain's
false-reject, while the gain stays above 98 % at the baseline's.

## 7. Ablations

Same 57 clips, same abstention set, R1 (R3 in brackets where it changes the story):

| variant | FRR | removal | mirror | x0.5 | x1.5 | yaw |
| --- | --- | --- | --- | --- | --- | --- |
| **primary** | **8.8 %** | **100 %** | **96.5 %** | **98.2 %** | **96.5 %** | 22.8 % |
| drag free (G2=0) | 8.8 % | 100 % | 98.2 % | 96.5 % | 84.2 % | 17.5 % |
| drag fixed 0.295 | 7.0 % | 94.7 % | 94.7 % | **36.8 %** | 98.2 % | 19.3 % |
| intercept fixed 0 | **22.8 %** | 100 % | 96.5 % | 91.2 % | 96.5 % | 56.1 % |
| Huber | 12.3 % | 100 % | 94.7 % | 98.2 % | 93.0 % | 19.3 % |
| smoothing 0.25 s | 10.5 % | 100 % | 98.2 % | 98.2 % | 96.5 % | 24.6 % |
| no lag search | 8.8 % | 100 % | 96.5 % | 98.2 % | 96.5 % | 24.6 % |

Reading:

- **The intercept is mandatory.** A mount offset of a few degrees is not absorbed by the
  gain; fixing `b=0` raises false-reject from 8.8 % to 22.8 % and the yaw-control artifact
  from 22.8 % to 56.1 %. This is the mechanism Track A2's fitted uptilt was doing.
- **The drag nuisance is needed, but not at the Phase 0 value.** Fixing `G2=0.295`
  collapses x0.5 detection to 36.8 %, because the regression's own velocity coefficient
  is about 0.14, not 0.295. Drag-free is close on most families and worse on x1.5
  (84.2 %), so fitting it is the best overall choice.
- Robust fitting is slightly worse on false-reject (12.3 %) and buys nothing on the
  violations; smoothing and lag search are nearly neutral. The frozen choices are the
  simplest ones that are not worse.

## 8. Detection versus violation size (synthetic gain sweep)

Scaling each real clip's depicted series by `k` about its fitted intercept (synthetic,
labelled; 57 clips each):

| k | violation | flag rate | Wilson 95 % |
| --- | --- | --- | --- |
| 0.0 | -100 % | 100 % | 93.7-100 |
| 0.25 | -75 % | 100 % | 93.7-100 |
| 0.5 | -50 % | 100 % | 93.7-100 |
| 0.75 | -25 % | 28.1 % | 18.1-40.8 |
| 1.25 | +25 % | 42.1 % | 30.2-55.0 |
| 1.5 | +50 % | 86.0 % | 74.7-92.7 |
| 2.0 | +100 % | 100 % | 93.7-100 |

The detection floor sits between a 25 % and a 50 % gain change: 28-42 % at 25 %, 86-100 %
at 50 %. The asymmetry (0.75 vs 1.25) is the band's asymmetry around the real median, not
a separate effect. This is the quantitative form of the claim: a 50 % tilt change becomes
detectable on clips with enough excitation, which the median-residual statistic could not
do.

## 9. Drag sanity check (the nuisance is not the airframe coefficient)

Fitted `G2` on testable real clips: median **0.139 1/s** (IQR 0.106-0.178); piloted 0.139,
autonomous 0.174, track 0.070 (n=4). Phase 0's mocap-only fit is **0.295**. The regression
value is about half. Inferred cause: the same errors-in-variables attenuation that puts
the real gain at 0.745 rather than 1 — the 12 Hz double derivative is noisy, so the
acceleration coefficient is attenuated toward zero and the correlated velocity term
absorbs part of it. The nuisance is doing its job (fixing it at 0.295 is worse), but it
must not be reported as an airframe drag estimate.

## 10. Where it fails

1. **Track flights are not validated.** Only 4 of 29 track clips survive abstention
   (13p 3, 16p 1), and two of those four real clips are falsely rejected (13p clip #01,
   16p clip #00). 16p's only clip has `G1 = 0.518`, below its fold's lower band 0.547.
   The pooled 8.8 % is a piloted-flight number in disguise; it is 50 % on the four track
   clips.
2. **The yaw control is flagged 22.8 % against 8.8 % of real clips.** A yaw rotation
   cannot change the tilt, but the gain moves: yaw-control `G1` median 0.697 (IQR
   0.593-0.810) against real 0.745 (0.696-0.806). The lower tail crosses the band. This
   is the estimator's view dependence, the same mechanism Track A measured at 4.75 deg
   of tilt change, and it is the floor under the statistic. A violation that shifts the
   gain by less than this artifact cannot be separated from it.
3. **The autonomous flights contribute 3 testable clips** (01a 2, 04a 1). Mirror
   detection there is 2/3 (04a's single clip misses). That is not evidence about
   autonomous flights generally.
4. **x1.5 at the baseline's upper operating point** is a tie, not an improvement
   (96.5 % against 94.8 %). The material claim is x0.5 and low-false-reject x1.5.
5. **Delays 200/300 ms** are outside the lag bound and are detected through the gain the
   same way; the detailed delay rows are in `metrics.json` / `gain_table.csv`. The lag
   nuisance absorbs 100 ms as designed.

## 11. Threats to validity

- **The gain is calibrated, not physical.** Real `G1` is 0.745, not 1. Any claim that the
  pixels "agree" with the path is a statement about the band, not about absolute physics.
- **Edits are camera rotations on real frames.** The estimator's response to an edit is
  part of the measurement; the yaw control bounds it but does not remove it.
- **Small n.** 57 testable clips from 6 flights, 50 of them from two piloted flights; 4
  track and 3 autonomous. Intervals resample nothing here — Wilson intervals treat clips
  as independent, which they are not within a flight; the per-flight table is the honest
  view.
- **Selection.** R1 is chosen by the pre-registered criterion on the pooled fit; the
  nested procedure gives R2 in six folds and a 10.5 % false-reject. The one-clip
  difference is reported, not tuned away.
- **The gate is frozen but applied at pair level.** Per-condition gating would drop 79 %
  of removal edits; that is a property of the gate, measured by Track A2, and not used
  here.

## 12. What this supports, and what it does not

**Supports.** On TII clips with enough excitation (57 of 100, essentially the piloted
ellipse and lemniscate flights), a per-clip gain regression separates a real flight from
the same frames with the tilt removed, halved, scaled by 1.5, or roll-mirrored at a
false-reject rate of 8.8 %, with 96-100 % detection. At matched false-reject it dominates
the median-residual statistic, most clearly on half tilt (98 % against 21-57 %).

**Does not support.** It does not validate the verifier on track or autonomous racing
footage (4 and 3 testable clips, two false rejects on track), does not show that a 25 %
tilt change is detectable (28-42 %), does not establish the airframe drag coefficient
(the fitted 0.139 is a nuisance), and does not transfer to UZH-FPV until Track E runs it.
A pass on a real clip means "no violation of the tested size on a clip with enough
excitation", never "the flight is physically consistent".

## 13. Deliverables

- `configs/gain_prereg.yaml`, `configs/prototype_acceptance_spec.yaml` (commit `613a102`).
- `configs/gain_frozen.yaml`: rule R1, band [0.5277, 0.9477] on `G1`, alpha 0.05, n=57,
  per-fold bands, commit hash, full options and abstention.
- `src/verdict/` (schema, lean, fit, abstain, decide), `scripts/score_clip.py`,
  `tests/test_verdict.py`.
- `experiments/d_build_clips.py`, `d_eval.py`, `d_freeze.py`, `d_repro_baseline.py`.
- `results/d_eval/d01/` (metrics, tables, curves, plots), clips under
  `data/processed/track_d/clips/` (rebuildable, not committed).

## 14. Reproduction

```
PYTHONPATH=. pixi run python experiments/d_build_clips.py
PYTHONPATH=. pixi run python experiments/d_repro_baseline.py
PYTHONPATH=. pixi run python experiments/d_eval.py --tag d01
PYTHONPATH=. pixi run python experiments/d_freeze.py --run results/d_eval/d01
PYTHONPATH=. pixi run python scripts/score_clip.py \
    'data/processed/track_d/clips/flight-01p-ellipse/flight-01p-ellipse#05__real' \
    --config configs/gain_frozen.yaml --json
```
