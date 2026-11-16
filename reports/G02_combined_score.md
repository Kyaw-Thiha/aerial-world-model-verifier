# G02 — One decision from three tiers: max of standardised statistics, calibrated conformally

Track G. Written 2026-11-16. Pre-registration `configs/g02_combined_prereg.yaml`. Code:
`experiments/g02_combined_score.py` and config `configs/g02_combined_prereg.yaml`, both at
`7074f9d`. Results: `results/g02_combined/{dev,heldout_devfit}/`, committed at `2ba24a3`. CPU only; **zero GPU spend**.

## Plain-language summary

1. **The combination works as a robustness device and does not buy sensitivity.** Taking the
   maximum of three standardised tiers (clip median residual, coherence, frozen axis-gain band)
   and calibrating that maximum leave-one-flight-out at 5 % gives **100 % detection of tilt
   removal and of a mirrored roll** on the dev block, with the yaw control at 16.4 % and the null
   edit at 7.3 %. But the best *single* tier is already at 99.9-100 % on those two families, so
   there is nothing left for a combination to add: the pre-registered "+0.05 AUC over the best
   single tier" is unreachable, and on the graded minimum-detectable-violation sweep the
   combination (50 % detection at a 60 % lean change) is **worse than the gain tier alone**
   (50 % detection at a 30 % change).
2. **Where the combination helps is scope**: it is the only rule that is simultaneously strong
   on shape-changing edits (removal, mirror: coherence and residual do that) and on pure magnitude
   edits (x0.5/x1.5: the gain tier does that, the others are at chance). Each single tier has a
   family where it is at chance; the max-of-tiers has none.
3. **The cost is the yaw control.** The combined score flags the yaw control at 16.4 % against a
   7.3 % false-reject, and that is inherited from the gain tier (29.1 % on the same clips). The
   yaw rotation is a known view-dependence of the reader (4.75 deg median shift, report 04), and
   the gain tier is the tier most exposed to it.
4. **Implementation verified against the frozen tables**: the recomputed per-clip axis gains match
   Track D's frozen `results/d_eval/d02/gain_table.csv` exactly (max |Δgx| = max |Δgy| = 0) and
   the coherence scores match `results/d_coherence/d01` to 0.025, so the tiers in this report are
   the published ones, not a look-alike.

## Question

Do the three existing verdict tiers, combined into a single calibrated decision, beat the best
single tier at the same whole-clip false-reject — and what is the smallest violation detected
half the time?

## Setup

**Tiers** (frozen definitions, never refit here):

| tier | statistic | source |
| --- | --- | --- |
| residual | clip median of the per-frame depicted-vs-required thrust-axis angle (deg) | schema v2 clips, frozen construction `src.depicted.stat_a2` |
| coherence | `1 - min(r_x, r_y)`, lag from the clip's own gain fit | `src.verdict.coherence`, frozen taus in `configs/coherence_frozen.yaml` (not used here, recalibrated as part of the max) |
| sign | `1 - min(s_x, s_y)` | same, secondary combination only |
| gain | frozen JointBand statistic `max(|gx-c_x|/s_x, |gy-c_y|/s_y)` with per-fold centres and scales | `configs/gain_frozen.yaml` |

**Combination.** Standardise each tier on the fit flights of its fold (`median`, robust scale
`(q97.5 - q2.5)/3.92`), take the **maximum** of the standardised tiers, calibrate the
leave-one-flight-out conformal 95th percentile of that maximum. No weights, nothing fitted that
could be tuned on the held-out block. Secondaries, declared in advance:
`G_max2_no_resid` (does the residual tier add anything?), `G_max3_sign` (sign instead of
coherence as the shape tier), `G_fisher` (−2 Σ log p over the three tiers' conformal p-values).

**Population.** 100 TII real clips; 82 on the dev block, of which **55 are testable** under the
frozen gain abstention (24 frames, usable fraction 0.50, 12 signal frames, excitation 2 deg,
condition number 1000, bootstrap CI half-width 0.35); the held-out block is the two flights
frozen in G01. Abstained clips are reported as abstained, never as passes. Testability is a
property of the path and of the depicted side's coverage, so scaling the depicted lean (the MDV
sweep) cannot change it.

**Minimum-detectable-violation curve.** The real clip's depicted lean vector is scaled,
`l_dep → s·l_dep`, for `s ∈ {1.0, 0.95, 0.9, 0.85, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0}`, so the
violation size is `1 - s`. This is a synthetic manipulation of the depicted side — no pixels are
re-estimated — and it measures the *statistic*, not the reader. The pixel families anchor it
(s = 0.5, 1.5 and 0 are the measured x0.5, x1.5 and removal edits).

## Results

### Detection at the 5 % conformal threshold (dev block, 55 testable clips)

| method | real (FRR) | tilt removal | mirror roll | x0.5 | x1.5 | yaw control | null edit |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **G_max3** (primary) | 7.3 % (4/55) | **100 % (55/55)** | **100 % (55/55)** | **87.3 % (48/55)** | 83.6 % (46/55) | 16.4 % (9/55) | 7.3 % (4/55) |
| G_max2_no_resid | 7.3 % (4/55) | 100 % (55/55) | 100 % (55/55) | 92.7 % (51/55) | 85.5 % (47/55) | 14.5 % (8/55) | 7.3 % (4/55) |
| G_max3_sign | 3.6 % (2/55) | 100 % (55/55) | 100 % (55/55) | 70.9 % (39/55) | 74.5 % (41/55) | 12.7 % (7/55) | 3.6 % (2/55) |
| G_fisher | 5.5 % (3/55) | 90.9 % (50/55) | 98.2 % (54/55) | 16.4 % (9/55) | 23.6 % (13/55) | 7.3 % (4/55) | 5.5 % (3/55) |
| residual alone | 5.5 % (3/55) | 60.0 % (33/55) | 92.7 % (51/55) | 18.2 % (10/55) | 60.0 % (33/55) | 5.5 % (3/55) | 5.5 % (3/55) |
| coherence alone | 9.1 % (5/55) | 100 % (55/55) | 100 % (55/55) | 18.2 % (10/55) | 21.8 % (12/55) | 12.7 % (7/55) | 9.1 % (5/55) |
| gain alone | 5.5 % (3/55) | 100 % (55/55) | 100 % (55/55) | 98.2 % (54/55) | 98.2 % (54/55) | **29.1 % (16/55)** | 5.5 % (3/55) |

Read down the columns: removal and mirror are saturated for coherence, gain and the two
combinations that include them; x0.5/x1.5 are the gain tier's; the residual tier is the only one
that is mediocre everywhere. The yaw control spans 5.5 % (residual) to 29.1 % (gain), so the
combination's 16.4 % is the price of including the gain tier.

### ROC, leave-one-flight-out scores

| method | tilt removal | mirror roll | x0.5 | x1.5 |
| --- | --- | --- | --- | --- |
| G_max3 | 0.983 [0.95, 1.00] | 0.999 [0.96, 1.00] | 0.966 [0.70, 1.00] | 0.965 [0.90, 0.99] |
| G_max2_no_resid | 0.983 [0.94, 1.00] | 0.999 [0.95, 1.00] | 0.968 [0.84, 1.00] | 0.965 [0.88, 1.00] |
| G_max3_sign | 0.983 [0.95, 1.00] | 0.999 [0.96, 1.00] | 0.947 [0.78, 1.00] | 0.943 [0.88, 0.98] |
| G_fisher | 0.985 [0.92, 1.00] | 0.996 [0.84, 1.00] | 0.916 [0.86, 0.95] | 0.936 [0.88, 0.97] |
| residual | 0.915 [0.81, 1.00] | 0.963 [0.79, 1.00] | 0.553 [0.44, 0.89] | 0.920 [0.80, 0.97] |
| coherence | 0.999 [1.00, 1.00] | 1.000 [1.00, 1.00] | 0.716 [0.69, 0.89] | 0.625 [0.60, 1.00] |
| gain | 0.999 [0.96, 1.00] | 1.000 [0.95, 1.00] | 0.980 [0.73, 1.00] | 0.981 [0.83, 1.00] |

Brackets are flight-clustered bootstrap spreads (200 replicates pooled per fold, 6 clusters):
**descriptive, not 95 % confidence intervals**.

The combination never beats the best single tier on any family. It is 0.983 vs 0.999 (gain) on
removal, 0.999 vs 1.000 (gain, coherence) on mirror, 0.966 vs 0.980 (gain) on x0.5 and 0.965 vs
0.981 (gain) on x1.5. Where it does beat a tier, it beats the *worst* one.

### Minimum detectable violation (detection at the 5 % threshold vs lean-scale change)

| method | 0.05 | 0.10 | 0.20 | 0.30 | 0.40 | 0.50 | 0.60 | 0.70 | 0.80 | 1.00 | mdv50 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **G_max3** | 0.07 | 0.11 | 0.15 | 0.15 | 0.22 | 0.47 | 0.60 | 1.00 | 1.00 | 1.00 | **0.60** |
| G_max2_no_resid | 0.07 | 0.11 | 0.15 | 0.15 | 0.22 | 0.47 | 0.60 | 1.00 | 1.00 | 1.00 | 0.60 |
| G_max3_sign | 0.07 | 0.13 | 0.20 | 0.24 | 0.31 | 0.55 | 0.78 | 1.00 | 1.00 | 1.00 | 0.70 |
| G_fisher | 0.07 | 0.07 | 0.07 | 0.09 | 0.20 | 0.42 | 0.67 | 0.87 | 0.95 | 1.00 | 1.00 |
| residual | 0.05 | 0.05 | 0.07 | 0.11 | 0.13 | 0.16 | 0.22 | 0.36 | 0.55 | 0.91 | 0.80 |
| coherence | 0.09 | 0.09 | 0.09 | 0.09 | 0.09 | 0.09 | 0.09 | 0.09 | 0.09 | 1.00 | 1.00 |
| gain | 0.07 | 0.11 | 0.18 | **0.53** | 0.91 | 0.98 | 1.00 | 1.00 | 1.00 | 1.00 | **0.30** |

The gain tier detects a 30 % lean-scale change half the time; the combination needs a 60 %
change, because its scale is set by three tiers and the residual tier (which is insensitive to
pure magnitude changes: 0.16 at 50 %) drags the maximum's null distribution up. The coherence
tier is flat at its false-reject rate until the lean vanishes entirely, exactly as the frozen
config predicted ("x0.5/x1.5 are expected to sit at chance").

### Held-out block (scored once; thresholds fitted on the six dev flights)

Of the 18 held-out real clips only **2 are testable** under the frozen abstention, so the block
cannot confirm anything: every method scores 0 of 2 real clips flagged (the dev-fitted
standardisation puts both below every threshold) and the per-condition rates are 0/n or n/n with
n ≤ 2. Reported because it was pre-registered, not because it is evidence.

Two things follow, and both are design lessons rather than results: the held-out flights should
have been chosen for *testability*, not for control-mode coverage; and a two-flight held-out
block cannot test a statistic whose per-flight spread is the dominant uncertainty.

## What this supports

* A single calibrated decision over the three existing tiers, with no fitted weights, reaches
  100 % detection of tilt removal and mirrored roll and 87-84 % of x0.5/x1.5 at a 7.3 %
  whole-clip false-reject on the dev block. That is the operating point the verdict currently
  lacks, and it needs no new physics.
* The tiers are complementary in exactly the way the frozen configs claimed: coherence owns
  shape edits, the gain band owns magnitude edits, and the combination is the only rule with no
  family at chance.

## What this does not show

* **Not a sensitivity gain, and it misses its own pre-registered bar.** No family improves on the
  best single tier; the MDV is worse than the gain tier alone. On the evidence here, if only one
  tier is affordable, it should be the gain band, not the combination.
* **Not a claim that the combined score is safe on yaw.** 16.4 % on the yaw control against 7.3 %
  real is a real weakness, inherited from the gain tier; a yaw-heavy clip can be rejected for a
  reason that is not a violation.
* **The MDV curve is a statistic-level curve, not a pixel-level one.** Scaling the depicted lean
  is not the same as re-estimating pixels on a scaled edit; only the discrete x0.5/x1.5/removal
  points are anchored to measured pixel edits.
* **Not validated outside TII.** UZH has too few testable clips for this calibration (E05), and
  ANWM clips are not in scope for a clip-level gain regression at 1 Hz paths.
* **No statement about flights.** 55 testable clips over 6 flights, 6-cluster bootstrap spreads.

## Caveats

* The held-out block was chosen in G01 for coverage of control modes and tracks, not for
  testability, and it turns out to hold almost no testable clips (see the block above). Any
  held-out claim from it is empty by construction, and that is a design lesson rather than a
  result.
* The frozen coherence tier's per-fold taus are *not* reused here: the max-of-tiers calibrates
  one joint threshold, so the coherence tier enters as a statistic, not as a decision. The
  frozen decision remains `configs/coherence_frozen.yaml` for anyone who wants the published
  tier.
* Gain fits for the synthetic sweep are run with the bootstrap disabled and at the clip's own
  stored lag; the bootstrap only enters the frozen abstention, which is taken from the real clip.
  The point estimates match the frozen table exactly (checked above, Δ = 0).
* Flight-clustered spreads with 6 clusters are descriptive.

## Regenerate

```
PYTHONPATH=. python experiments/g02_combined_score.py --population dev
PYTHONPATH=. python experiments/g02_combined_score.py --population heldout --confirm-once
```

Outputs: `results/g02_combined/dev/summary.json` (all rates, AUC, MDV curves, per-fold tier
parameters), `per_clip.csv` (LOFO score per clip per condition), and the held-out directory.
Config: `configs/g02_combined_prereg.yaml`.
