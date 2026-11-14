# Report D02 — Gain decomposition, schema v2, coherence tier, OLS vs Deming

Written 2026-11-14. Follows `reports/D01_gain_test.md` (v1, schema v1, rule R1). Run
`results/d_eval/d02/metrics.json`; decomposition `results/d_decomposition/d01/`;
coherence `results/d_coherence/d01/`; Deming `results/d_deming/d01/`. Frozen statistic:
`configs/gain_frozen.yaml` (v2, rule **R2_axis_gains**). Frozen coherence thresholds:
`configs/coherence_frozen.yaml`.

Every detection rate below has the **yaw-control rate** beside it at the same operating
point, and **track flights are marked not validated** wherever they appear. That is a
reporting rule, not a caveat added at the end.

## Plain summary

1. **The 0.745 gain is not a formula error.** Truth attitude regressed on the required
   path gives **0.991** [0.935, 1.010] and a velocity coefficient of **0.239 1/s**
   (Phase 0: 0.295). The loss is **estimator shrinkage**: estimated lean on truth
   attitude gives **0.828** (piloted 0.831, autonomous 0.587, track 0.558).
2. **Dense required lean helps, but modestly.** Moving the required side to 274.73 Hz
   (`lean_req`, schema v2) raises the verdict gain from **0.745 to 0.764** and changes the
   selected rule from R1 to R2. The v2 axis band halves false-reject (5.3 % vs 8.8 %)
   while keeping removal/mirror/x0.5 at 100 % and x1.5 at 96.5 %. The gain does not reach
   1 because the pixels, not the differentiation, carry the error.
3. **The scale-free coherence tier is a strong complement, not a replacement.** AUROC for
   removal **0.999**, mirror **1.000**, but x0.5 **0.706**, x1.5 **0.622**, yaw control
   **0.600**. Sign agreement is 0.998/0.999 for removal/mirror and ~0.5 for scaling, as a
   scale-free shape statistic should be. It catches shape/sign violations; the gain
   catches magnitude.
4. **Deming does not fix the gain.** With the error-variance ratio measured from the
   coarse-vs-dense and estimator-vs-truth differences, Deming is 0.730 (v1) and 0.762
   (v2) against OLS 0.753 and 0.770: the dominant error is on the depicted side, so the
   correction is neutral-to-negative and worsens x0.5 detection. OLS stays frozen.
5. **The yaw-control artifact is the binding limitation.** Real false-reject is 5.3 %,
   but the yaw control is flagged **28.1 %** under the frozen R2 rule (19.3 % under R1).
   A yaw rotation cannot change the tilt; this is the estimator's view dependence. Track
   flights remain unvalidated (3 testable clips, 1 false reject under R2).

## 1. Decomposition of the real-flight gain (task 1)

Same 57 testable real clips, dense mocap. (a) truth attitude on required lean with a
fitted drag nuisance and free intercept; (b) estimated lean on truth-attitude lean, free
intercept, no drag; (c) the verdict gain, v1 (12 Hz required lean) and v2 (dense
`lean_req`). Medians [flight-cluster bootstrap 95 % interval], disjoint flight types.

| quantity | all (57) | piloted (50) | autonomous (3) | track (4) |
| --- | --- | --- | --- | --- |
| (a) truth gain on required | **0.991** [0.935, 1.010] | 0.997 [0.969, 1.012] | 0.935 [0.930, 0.952] | 0.939 [0.906, 0.948] |
| (a) fitted drag, 1/s | **0.239** [0.114, 0.245], IQR 0.221-0.268 | 0.244 | 0.086 | 0.147 |
| (a) gain, drag fixed 0.295 | 1.048 [0.957, 1.071] | 1.053 | 1.010 | 0.949 |
| (a) gain, drag-free | 0.755 [0.688, 0.916] | 0.714 | 0.901 | 0.926 |
| (b) estimator shrinkage | **0.828** [0.582, 0.832] | 0.831 [0.827, 0.834] | 0.587 | 0.558 |
| (c) verdict gain, v1 | **0.745** [0.573, 0.762] | 0.753 | 0.627 | 0.548 |
| (c) verdict gain, v2 dense | **0.764** [0.568, 0.786] | 0.770 | 0.645 | 0.541 |

Correlation of the verdict gain with `(a) × (b)`: 0.814 (v1), 0.860 (v2). It is
approximately, not exactly, multiplicative because of the intercept, the drag nuisance and
the nonlinear normalization.

Reading:

- **The formula is validated.** Truth attitude on the path is 0.99 on the piloted
  flights, and the drag nuisance is 0.24 against Phase 0's 0.295. Fixing drag at 0.295
  raises the gain to 1.05; removing it drops the gain to 0.76. The velocity coefficient is
  doing real work, but the exact relation is not linear in `v/g`, so 0.24 rather than 0.295
  is the honest fitted value, not a new drag estimate.
- **The pixels, not the path, carry the loss.** Estimator shrinkage is 0.83 on piloted
  flights and 0.56-0.59 on autonomous/track. The dense required side moves the verdict
  gain by only +0.019.
- **Per-type gains differ widely.** The autonomous and track clips that survive
  abstention are shrunk and low-gain; that is why the band is calibrated on real clips and
  why pooled rates are not type rates.

## 2. Schema v2 and the rerun (task 2)

`docs/verdict_clip_schema_v2.md` adds `lean_req` (drag-free required lean) and `vel_req`
(velocity regressor), both computed from **dense** truth with a **0.37 s / poly-3**
Savitzky-Golay window, sampled at the frame times, NaN outside the native interior. `pos`
and `yaw` stay for provenance and fallback. The amendment is `configs/gain_prereg.yaml`
amendment 1, committed before v2 scoring.

### Bands

| rule | v1 (12 Hz required) | v2 (dense required) |
| --- | --- | --- |
| R1 scalar `G1` | [0.5277, 0.9477] | [0.5430, 0.9364] |
| R2 axis joint | `gx` center 0.7279, scale 0.2566; `gy` 0.7102, 0.0924; tau 2.1647 | `gx` **0.8137**, 0.2654; `gy` **0.7181**, **0.0735**; tau **2.1922** |

The dense required lean raises the pitch-axis centre by 0.086 and tightens the roll-axis
spread from 0.092 to 0.074. That is why R2 wins selection in v2.

### False-reject and detection (full testable sets, n=57)

| version / rule | FRR | removal | mirror | x0.5 | x1.5 | yaw control | track |
| --- | --- | --- | --- | --- | --- | --- | --- |
| v1, R1 (D01 frozen) | 8.8 % | 100 % | 96.5 % | 98.2 % | 96.5 % | 22.8 % | **not validated** |
| v1, R2 | 10.5 % | 100 % | 100 % | 100 % | 96.5 % | 24.6 % | **not validated** |
| v2, R1 | 10.5 % | 100 % | 98.2 % | 100 % | 98.2 % | 19.3 % | **not validated** |
| **v2, R2 (frozen)** | **5.3 %** | **100 %** | **100 %** | **100 %** | **96.5 %** | **28.1 %** | **not validated** |

Nested per-fold selection: v1 10.5 %, v2 **5.3 %**. Pooled selection: v1 R1, v2 R2.

The testable composition is not identical between versions: 04a clip #04 moves in and
13p clip #00 moves out (same total 57). On the **56-clip intersection**, v1 R1 is
8.9 % FRR / 100 / 96.4 / 98.2 / 98.2 / 23.2 % yaw; v2 R2 is **5.4 %** / 100 / 100 / 100 /
96.4 / **26.8 % yaw**. The direction survives the composition change.

Per disjoint flight type for the frozen v2 R2 rule:

| type | n | FRR | removal | mirror | x0.5 | x1.5 | yaw control |
| --- | --- | --- | --- | --- | --- | --- | --- |
| piloted | 50 | 2.0 % | 100 % | 100 % | 100 % | 98.0 % | 20.0 % |
| autonomous | 4 | 25.0 % | 100 % | 100 % | 100 % | 75.0 % | 100 % |
| track | 3 | 33.3 % | 100 % | 100 % | 100 % | 100 % | 66.7 % |

**Track is not validated.** Three testable clips, one false reject, one x1.5 miss, and a
66.7 % yaw-control rate on three clips. The pooled 5.3 % is a piloted-flight number.

### What did not change

- **Baseline**: rerun gives the same numbers — threshold **14.733°** (folds 13.50-16.46),
  acceptance 0.966 (58), per-condition removal 0.500 (8), mirror 0.961, yaw 0.038. The
  baseline's required side was already dense mocap, so v2 does not touch it. In the
  pair-level comparison it is FRR 3.4 %, removal 67.2 %, mirror 94.8 %, x0.5 17.2 %,
  x1.5 24.1 %, **yaw 8.6 %**, abstention 42 %.
- **Abstention**: 43.0 % (57/100), unchanged in total.
- **Fitted drag** under the verdict: median **0.158 1/s** (was 0.139), still about half of
  Phase 0's 0.295; the decomposition shows why (estimator shrinkage on the velocity term).
- **Synthetic gain sweep** under the frozen rule improves at ±25 %: k=0.75 **71.9 %**
  (v1 R1: 28.1 %), k=1.25 **59.6 %** (42.1 %), k=1.5 **94.7 %** (86.0 %), k=0.5 98.2 %
  (100 %). Part of the gain is the dense input, part is the R2 rule; both are v2.

## 3. Scale-free coherence tier (task 3)

Pre-registered in `configs/coherence_prereg.yaml` before any edited clip was scored.
Per-axis Pearson correlation `r_i` between depicted and required lean, and sign agreement
`s_i`, over the same usable frames and with the gain fit's lag. Violation scores
`C_r = 1 − min(r_x, r_y)` and `C_s = 1 − min(s_x, s_y)`; higher is suspicious.
Thresholds are the 95th percentile of real clips, leave-one-flight-out.

### AUROC and operating point (n=57 each)

| family | AUROC (corr) | corr det | AUROC (sign) | sign det | yaw det (corr) |
| --- | --- | --- | --- | --- | --- |
| tilt removal | **0.999** | 100 % | **0.998** | 100 % | 10.5 % |
| mirror roll | **1.000** | 100 % | **0.999** | 100 % | 10.5 % |
| x0.5 | 0.706 | 17.5 % | 0.568 | 7.0 % | 10.5 % |
| x1.5 | 0.622 | 24.6 % | 0.464 | 8.8 % | 10.5 % |
| yaw control | 0.600 | 10.5 % | 0.522 | 7.0 % | 10.5 % |

Real false-reject at the fold thresholds is **8.8 %** (correlation) and **7.0 %** (sign).
The frozen pooled thresholds are **tau_corr = 0.6197**, **tau_sign = 0.2921**; per-fold
corr 0.548-0.622, sign 0.242-0.371.

**x0.5/x1.5 are not at 0.5, but close enough to be a negative result.** A positive scale
is invariant for a pure product, but the mount intercept and estimator noise make the
correlation fall when the tilt is scaled down (0.706 for x0.5) or rise when scaled up
(0.622 for x1.5). Sign agreement is at chance (0.568/0.464). The tier is **not** a
magnitude detector; use it with the gain, not instead of it.

### Real / null distributions by flight type

Median (IQR) of `r_min`; `null_edit` is the same pixels as real by construction.

| type | real r_min | null r_min | yaw-control r_min | real s_min | yaw s_min |
| --- | --- | --- | --- | --- | --- |
| piloted (50) | 0.896 (0.795-0.934) | identical | 0.859 (0.767-0.907) | 0.948 | 0.938 |
| autonomous (4) | 0.629 (0.564-0.642) | identical | **0.133 (−0.165-0.351)** | 0.685 | 0.611 |
| track (3) | 0.642 (0.581-0.659) | identical | 0.716 (0.623-0.726) | 0.886 | 0.830 |

The autonomous yaw-control collapse (four clips) is the view-dependence artifact in its
clearest form: a rotation that cannot change tilt destroys the correlation. **Track is not
validated (3 testable clips).**

### Transfer to UZH

One threshold is frozen on TII and written to `configs/coherence_frozen.yaml`
(`tau_corr` 0.6197, `tau_sign` 0.2921) for Track E to apply **unchanged** to schema-v2 UZH
clips. The per-fold spread on TII (corr 0.548-0.622) is the only proxy available now;
the actual UZH transfer is future work and must not recalibrate the threshold.

## 4. OLS versus total least squares (Deming) (task 4)

`lambda = sigma_y^2 / sigma_x^2` measured per clip from the coarse-vs-dense difference
(`sigma_x`) and the depicted-vs-truth difference (`sigma_y`); the drag and intercept are
residualised out with OLS, then the Deming slope is fitted on the residualised data.
The same 57 clips and conformal bands.

| method | band | FRR | removal | mirror | x0.5 | x1.5 | yaw control |
| --- | --- | --- | --- | --- | --- | --- | --- |
| v1 OLS | 0.528-0.985 | 7.0 % | 100 % | 94.7 % | 98.2 % | 93.0 % | 19.3 % |
| v1 Deming | 0.486-0.907 | 7.0 % | 100 % | 96.5 % | 73.7 % | 93.0 % | 24.6 % |
| v2 OLS | 0.543-0.936 | 10.5 % | 100 % | 98.2 % | 100 % | 98.2 % | 19.3 % |
| v2 Deming | 0.519-0.872 | 7.0 % | 100 % | 98.2 % | 84.2 % | 96.5 % | 31.6 % |

Real-clip gain medians: v1 OLS 0.753 piloted / 0.627 autonomous / 0.518 track; v1 Deming
0.730 / 0.633 / 0.524; v2 OLS 0.770 / 0.645 / 0.541; v2 Deming 0.762 / 0.623 / 0.534.

**Deming is not the remedy.** The dominant error is on the depicted (y) side, so `lambda`
is large and Deming approaches OLS; where it moves, it lowers the gain slightly and costs
x0.5 detection (98.2 → 73.7 % v1; 100 → 84.2 % v2) while raising the yaw-control artifact.
OLS remains the frozen estimator. This is consistent with the decomposition: the gain is
short because the estimator shrinks, not because the path regressor is noisy.

## 5. Frozen deliverables for Track E

- **Gain**: `configs/gain_frozen.yaml`, rule **R2_axis_gains**, v2 dense input. Joint band
  `gx` center 0.8137 / scale 0.2654, `gy` center 0.7181 / scale 0.0735, tau 2.1922,
  alpha 0.05, n=57; per-fold bands included. Commit `1df4084`.
- **Coherence**: `configs/coherence_frozen.yaml`, tau_corr 0.6197, tau_sign 0.2921,
  per-fold thresholds.
- **Schema**: `docs/verdict_clip_schema_v2.md`; re-export UZH clips with `lean_req`,
  `vel_req`, optional `tilt_req_deg`, keeping `pos`/`yaw`. Score with the frozen configs
  unchanged.
- **Scorer**: `scripts/score_clip.py --config configs/gain_frozen.yaml`.
- **Read from git** (no push yet): `git show track-d-verdict-stat:configs/gain_frozen.yaml`.

## 6. What this does not show

- It does not validate track flights: 3 testable clips, one false reject, one x1.5 miss.
- It does not validate autonomous flights: 4 testable clips, 25 % FRR, 100 % yaw-control
  flag rate.
- It does not make the gain physical: real `G1` is 0.758 (v2), and the estimator shrinks
  the depicted tilt by 0.83. A pass is relative to the calibrated real band.
- It does not show that Deming or TLS is needed; it shows the opposite on this data.
- It does not test UZH transfer; that is Track E's run with the frozen thresholds.
- The coherence tier is not a magnitude detector: x0.5/x1.5 AUROC is 0.71/0.62, and the
  sign statistic is at chance for scaling.

## 7. Reproduction

```
PYTHONPATH=. pixi run python experiments/d_build_clips.py
PYTHONPATH=. pixi run python experiments/d_decompose.py
PYTHONPATH=. pixi run python experiments/d_eval.py --tag d02
PYTHONPATH=. pixi run python experiments/d_coherence.py --run results/d_eval/d02
PYTHONPATH=. pixi run python experiments/d_deming.py
PYTHONPATH=. pixi run python experiments/d_freeze.py --run results/d_eval/d02
PYTHONPATH=. pixi run python experiments/d_repro_baseline.py
PYTHONPATH=. pixi run pytest -q
```
