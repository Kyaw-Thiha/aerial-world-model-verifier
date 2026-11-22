# Report F04 — Spline-smoothed joint fit (round 2, part 1)

Track F2. Written 2026-11-21. Pre-registration `configs/f04_round2_prereg.yaml` (commit
`63002b8`), written before any statistic in this report was computed. Code and runs: see
"Reproduce"; results in `results/f04_spline/` (one combined run over all five paths).

**Decision: fail.** With estimated attitude and any stored pixel path the pre-registered
spline statistic flags 1–2/8 real clips but detects only 0–1/8 tilt removals and 2–3/8 mirrored
rolls (AUC 0.53–0.70), far short of the 90 % bar on both paths the gate names; on the truth
path the same statistic detects 8/8 removals and 6/8 mirrors (AUC 1.00/0.97), so the
analytic-derivative form is sound but more path-sensitive than round 1's integrated statistic
(pixel-path AUC 0.95/0.89).

## Plain-language summary

The round-1 report ended with the path as the binding error and the conformal threshold as the
second constraint. Round 2's part 1 tries the idea nearest to the Kinetodynamic priors: fit a
smooth B-spline to the recovered path, take its acceleration analytically, and ask whether the
flatness prediction `a = g·lean − k_d·v` fits that acceleration after jointly choosing the
path scale, the yaw offset and the initial velocity.

On the truth path the answer is yes: 8/8 removals and 6/8 mirrored rolls are detected at the
same 1/8 false-reject rate as round 1. On every stored pixel path the answer is no: removal
detection falls to 0–1/8 and the rank AUCs are 0.53–0.70. Two mechanisms are visible and are
reported rather than tuned away:

1. **The joint fit is degenerate.** On real clips whose acceleration shape disagrees with the
   lean, the free initial velocity absorbs the forcing and the fitted scale collapses — the
   median fitted scale is 0.33–0.54 of the Sim(3) truth scale, and 4–5 of the 8 primary clips
   sit below half of it. The real statistic then reaches 2.5–6.1, the leave-one-sequence-out
   threshold follows it to 4.5–5.9, and an edited clip at 1–3 is not flagged.
2. **Comparing accelerations is less forgiving than comparing positions.** The statistic is a
   ratio of acceleration energies; a path error of the size that round 1's position-integrated
   statistic absorbed becomes a large fraction here. The truth-path floor is 0.17–0.54, the
   pixel-path real values 0.26–6.09.

A no-free-velocity variant, added after seeing this and labelled exploratory, repairs some
clips (indoor_forward_5#01 from 5.14 to 0.79) but not others and does not improve the AUCs.
The result is not a threshold artefact: the 0.90, 0.95 and 0.99 conformal levels give the same
1–2/8 detection.

## The question

Does a spline-smoothed path, with analytic derivatives and a joint fit of scale, yaw and
initial velocity against the flatness prediction, carry a pixel-path verdict that round 1's
zero-differentiation statistic did not? Secondarily, is the conformal threshold or the
statistic the binding constraint?

## Setup

### Statistic S_dyn (pre-registered)

* **Spline.** Cubic B-spline, least squares, clamped ends, uniform knots with spacing
  `h = 0.5/f95` from the E05 window rule (per sequence, not fitted). The number of intervals is
  `K = max(4, round(T/h))`, capped at `(N−1)/3`; the 60-frame clips give K ∈ [4, 12], median
  4.5. Derivatives are analytic and evaluated on the interior `[3, N−3)`, because a clamped
  spline's second derivative is unreliable at the ends.
* **Model.** `v_m' = −k_d v_m + g·Rz(ψ+yaw)·lean`, `v_m(t0) = v0`, `a_m = g·Rz(ψ+yaw)·lean −
  k_d·v_m`, with `k_d = 0.295` fixed.
* **Fit.** Per clip and per yaw offset ψ on a 1° grid, least squares in the positive scale `s`
  and the initial velocity `v0` of `s·a_spline = a_m`. Scale positivity is enforced in closed
  form by projecting out the `v0` columns.
* **Statistic.** `ρ = sqrt(SSE / Σ|s·a_spline|²)`, the fraction of observed horizontal
  acceleration energy not explained. Higher is more suspicious. Model-normalised and
  demeaned denominators are pre-declared secondaries.
* **Pairing (amendment 1 of round 1, carried forward).** `s`, `ψ` and `v0` are fitted on the
  clip's real condition and held fixed for its six edits. Without this, tilt removal is
  absorbed by `s → 0`.
* **Arms.** Estimated attitude (GeoCalib depicted lean, up through the published mount, yaw
  from the path) and oracle attitude (truth quaternion); pixel paths DPVO, ViPE, MASt3R
  (exploratory) and Pi3X (part 3); truth path as a labelled ceiling.
* **Population.** Identical to F01: 8 real clips in `indoor_forward_3/_5/_9/_10` (#00, #01)
  with a full 60-frame path, the frozen confidence gate and E's testability rules; the
  exploratory set is every other real clip with a path, with a pooled threshold.
* **Thresholds.** Leave-one-sequence-out conformal 95th percentile on real clips, with 0.90
  and 0.99 reported as a pre-registered descriptive curve.
* **Gate.** Estimated attitude + pixel path, primary population, real false-reject ≤ 15 % and
  removal and mirror detection ≥ 90 % on both DPVO and ViPE; pass = both, partial = one.

### Population actually scored

27 real clips × 7 conditions × 2 arms × 2 path kinds per extractor. 10 of the 108
(clip, arm, path) groups are skipped for DPVO, ViPE, Pi3X and MapAnything (no full path or no
positive-scale solution) and 14 for MASt3R. The primary population is 8 clips in 4 sequences; no primary clip
abstains.

## Results

### The gate arm (estimated attitude, pixel path, 8 clips / 4 sequences)

| path | real | null | yaw | removal | mirror | x0.5 | x1.5 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| DPVO | 0.125 (1/8) | 0.125 (1/8) | 0.125 (1/8) | 0.125 (1/8) | 0.250 (2/8) | 0.000 (0/8) | 0.250 (2/8) |
| ViPE | 0.250 (2/8) | 0.250 (2/8) | 0.250 (2/8) | 0.125 (1/8) | 0.250 (2/8) | 0.250 (2/8) | 0.250 (2/8) |
| MASt3R (exploratory) | 0.125 (1/8) | 0.125 (1/8) | 0.250 (2/8) | 0.125 (1/8) | 0.375 (3/8) | 0.000 (0/8) | 0.375 (3/8) |
| Pi3X (part 3) | 0.125 (1/8) | 0.125 (1/8) | 0.125 (1/8) | 0.000 (0/8) | 0.250 (2/8) | 0.000 (0/8) | 0.250 (2/8) |
| MapAnything (part 3, not fed) | 0.125 (1/8) | 0.125 (1/8) | 0.125 (1/8) | 0.125 (1/8) | 0.125 (1/8) | 0.125 (1/8) | 0.125 (1/8) |

Rank AUC, real against each condition:

| path | removal | mirror | x0.5 | x1.5 | null | yaw |
| --- | --- | --- | --- | --- | --- | --- |
| DPVO | 0.62 | 0.67 | 0.58 | 0.67 | 0.50 | 0.48 |
| ViPE | 0.53 | 0.62 | 0.44 | 0.66 | 0.52 | 0.47 |
| MASt3R | 0.58 | 0.70 | 0.55 | 0.70 | 0.50 | 0.50 |
| Pi3X | 0.58 | 0.67 | 0.39 | 0.70 | 0.52 | 0.50 |
| MapAnything | 0.58 | 0.58 | 0.45 | 0.67 | 0.52 | 0.45 |

Sequence-clustered intervals (descriptive, 4 clusters), DPVO: real 0.00–0.375, removal
0.00–0.375, mirror 0.00–0.50. MapAnything was run through the statistic for completeness but
was not fed as a candidate: it is the worst path by both path metrics (F05, F06). Thresholds used (LOSO 95th percentile): indoor_3 5.85,
indoor_5 5.18, indoor_9 4.47, indoor_10 5.85.

**The gate fails on both named paths.** No secondary path passes either, so the outcome is
fail, not partial.

### The oracle ladder

| path | attitude | real | removal | mirror | AUC removal/mirror |
| --- | --- | --- | --- | --- | --- |
| pixel (DPVO) | estimated | 0.125 (1/8) | 0.125 (1/8) | 0.250 (2/8) | 0.62 / 0.67 |
| truth | estimated | 0.125 (1/8) | **1.000 (8/8)** | **0.750 (6/8)** | **1.00 / 0.97** |
| pixel (DPVO) | oracle | 0.125 (1/8) | 0.125 (1/8) | 0.125 (1/8) | 0.50 / 0.50 |
| truth | oracle | 0.125 (1/8) | 0.125 (1/8) | 0.125 (1/8) | 0.50 / 0.50 |

Same pattern for ViPE, MASt3R and Pi3X: with the truth path the statistic detects 8/8 and 6/8
at AUC 1.00/0.97; with the oracle attitude every condition is identical to real, because that
arm never reads the pixels. The statistic is sound; the path kills it.

### Threshold versus statistic

LOSO conformal levels (DPVO, estimated/pixel):

| level | real | removal | mirror |
| --- | --- | --- | --- |
| 0.90 | 2/8 | 1/8 | 2/8 |
| 0.95 (gate) | 1/8 | 1/8 | 2/8 |
| 0.99 | 1/8 | 1/8 | 2/8 |

The same holds for ViPE, MASt3R and Pi3X: moving the level by an order of magnitude in
false-reject budget does not buy detection. Unlike round 1, the threshold is **not** the
binding constraint here — the statistic's separation on pixel paths is too low for any level to
convert into 0.90 detection.

### Pre-registered sensitivities (DPVO, estimated/pixel, q0.95)

| setting | real | removal | mirror | AUC removal/mirror |
| --- | --- | --- | --- | --- |
| headline (K from rule, k_d 0.295) | 1/8 | 1/8 | 2/8 | 0.62 / 0.67 |
| knots K−1 | 1/8 | 1/8 | 2/8 | 0.62 / 0.67 |
| knots K+1 | 1/8 | 1/8 | 2/8 | 0.61 / 0.69 |
| k_d = 0 | 1/8 | 0/8 | 1/8 | 0.64 / 0.67 |
| no free `v0` (exploratory) | 2/8 | 0/8 | 2/8 | 0.62 / 0.67 |
| ψ refitted per condition | 1/8 | 1/8 | 1/8 | 0.62 / 0.61 |

Knot density and the drag term do not explain the failure. Re-searching ψ per condition can
only lower an edited clip's statistic (the real condition is unchanged by construction, since
the joint fit already chose that ψ), and it does: mirror detection falls from 2/8 to 1/8. The
statistic is not rescued by letting each condition rotate itself into agreement.

### The degeneracy (diagnostic, not a gate)

Per primary real clip (`results/f04_spline/diagnostic.json`):

| path | median real ρ | max real ρ | fitted s / Sim(3) s (median) | clips with s < 0.5·Sim3 |
| --- | --- | --- | --- | --- |
| DPVO | 0.79 | 6.09 | 0.54 | 4/8 |
| ViPE | 1.69 | 6.13 | 0.33 | 5/8 |
| MASt3R | 0.86 | 5.15 | 0.46 | 4/8 |

The mechanism is the free initial velocity: `v0` can cancel a constant part of the forcing in
`a_m = f − k_d(H v0 + F)`, so the least squares can drive `s → 0` while keeping `a_m` bounded.
The clips where this happens are exactly the ones whose acceleration shape disagrees with the
lean (e.g. `indoor_forward_5#01`, ρ 5.14 with s 0.021 against a Sim(3) scale of 0.227). The
truth path does not trigger it (ρ 0.17–0.54), because there the shape agrees.

**Exploratory variant (declared after the result, not part of the gate): no free `v0`.** The
drag term uses the observed spline velocity, so the only free parameter is `s`. It repairs
`indoor_forward_5#01` (5.14 → 0.79) but not `indoor_forward_3#01` (1.26 → 6.00) or
`indoor_forward_9#01` (6.09 → 5.05), and it does not improve the AUCs (0.62/0.67 for DPVO,
0.62/0.67 MASt3R, 0.50/0.67 ViPE). Reported as exploratory in the summary and in every table.

### Controls and scale edits

The null edit reproduces the real statistic to three decimals (`indoor_forward_10#00`:
0.3244 vs 0.3244), as it must — the path and the depicted lean are unchanged up to the
re-encode. The yaw control moves a few clips (1–2/8), which is the artefact floor. The scale
edits sit at 0–2/8 (x0.5) and 2–3/8 (x1.5), below the removal rate of round 1's statistic;
with the collapsed ruler the fixed-scale inconsistency is masked by the same high real floor.

## What this supports

* **The Kinetodynamic-style derivative is not the missing piece for a pixel path.** On stored
  paths it is *worse* than round 1's zero-differentiation statistic by a wide margin on the
  statistic's own terms (AUC 0.62/0.67 versus 0.95/0.89), while remaining perfect on the truth
  path. The limiting factor is the same one both rounds found: the recovered path.
* **A joint fit with a free initial velocity is a trap for an edit test.** `v0` and `s` trade
  against each other; when the path's acceleration shape disagrees, the fit prefers a collapsed
  ruler and a large real statistic. Any future joint fit must either bound the scale
  independently or not let velocity and scale be free together on the reference condition.
* **The conformal threshold is not always the binding constraint.** Round 1's threshold
  converted a separating statistic into 0.75 detection; here no level converts a
  non-separating statistic into anything. The two failure modes need different fixes.
* **Comparing accelerations is a stricter path test than comparing positions.** The same
  clips that round 1's shooting statistic fitted to ρ 0.08–0.47 give 0.26–6.09 here. That is
  a useful diagnostic: a path can be good enough for a position-integrated verdict and not for
  an acceleration-level one.

## What this does not show

* That no spline-based statistic can work: this one's particular joint parameterisation is
  degenerate, and the no-`v0` exploratory variant was not tuned further.
* That MegaSaM, ViPE or a SLAM system would change the outcome: part 3's Pi3X (best lean error
  among the new extractors) fails identically.
* Anything about the simulated calibration set: part 4 is waiting on Track I (F06).

## Caveats

* **8 clips in 4 sequences.** Every rate has 1/8 granularity; the clustered intervals span
  half the unit line and are descriptive, not 95 % confidence intervals.
* **The real path is reused for every edit**, as in round 1; the one-sequence edit-invariance
  check (DPVO on indoor_forward_5 edited frames, 0.81° median step direction) is the only
  direct evidence.
* **The knot rule is fixed by E05**, not validated per sequence; K = 4 for the sequences with
  windows near 1 s makes the spline stiff, which is the rule's intent but also its risk.
* **The no-v0 variant is post-hoc** and labelled exploratory everywhere; it changes no
  decision.
* **The scale diagnostic uses the Sim(3) truth scale**, which is evaluation-only and is not
  available to a verdict.

## Reproduce

```
# statistic and gate (CPU, no spend)
python experiments/f04_spline_fit.py --path all --out results/f04_spline
python experiments/f04_diagnostic.py     # the scale-collapse table
# single-path re-runs (e.g. --path pi3x) write their own directory and are not the cited run
# unit tests
pytest -q tests/test_spline_fit.py
```

Provenance: `results/f04_spline/summary.json` (commit at run time, per-path pose sha256,
population, thresholds at three levels, sensitivities), `rows.csv` (one row per sequence ×
clip × condition × arm × path), `diagnostic.json` (scale collapse).
Cost: zero GPU.
