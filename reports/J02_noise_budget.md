# J02 — The synthetic noise budget: how small a violation the verdict can see

Track J. Written 2026-11-21. Pre-registration `configs/j02_noise_budget_prereg.yaml`, committed
at `9652848` before any cell was computed. Code `experiments/j02_noise_budget.py`, window
machinery `src/verdict/segments.py` (copied unchanged from `track-g2-segments` `4c6eacf`).
Results `results/j02_noise_budget/summary.json`. CPU only, **zero GPU spend**.

## Decision

**The floor is 6.9 deg of tilt reduction at the measured reader accuracy on a perfect path, 10.7
deg with the reader's actual measured error, and the published 14.7 deg threshold is reproduced
(14.46, [13.21, 15.51]) as soon as the real actual-vs-required mismatch is included.** Reader
accuracy dominates (MDV 4.7 / 6.7 / 9.6 deg at σ = 3 / 4.8 / 8), the per-flight bias adds about
1.5 deg, and **the 8-frame window the G-round segment detector uses costs 3.6 deg of sensitivity
against the whole-clip median** (10.5 vs 6.9), recovering to within 0.5 deg only at 24 frames.

## Plain-language summary

1. **Every construction in this report reproduces a published number before it adds one.** The
   analytic cross-check agrees with the exact frozen residual construction to 9e-13 deg; the
   replay of the measured reader error gives a per-frame median of 3.97 deg, exactly report 02's
   figure; and the handoff's own construction (mocap attitude as the perfect depiction) gives a
   leave-one-flight-out 95th percentile of **14.46 deg against the frozen verdict's 14.73**.
2. **A uniform tilt reduction of 6.9 deg is found half the time** at σ = 4.8 deg, a 2 deg bias
   and a 0.5 s correlation time, with the clip-level median residual calibrated
   leave-one-flight-out (H0 median 6.45 deg, threshold 9.04). With the reader's *measured* error
   instead of the parametric model the same construction gives **10.65 deg**, because the real
   error is much heavier-tailed than a Gaussian with the same σ.
3. **Window length is a real cost and the G-round 8-frame window is on the wrong side of it.**
   MDV against window length: 6.92 deg at the whole clip, 7.42 at 24 frames, 8.17 at 16, 9.01 at
   12 and **10.52 at 8 frames**. That is the same mechanism G01 and G04 measured from the other
   side (the max-over-windows threshold inflates as the window shortens); here it is quantified
   in degrees of violation, at a fixed 5 % false-reject.
4. **Accuracy is the lever, bias is a smaller one, correlation time is nearly none.** At zero
   bias, MDV goes 4.66 (σ=3) → 6.70 (σ=4.8) → 9.63 (σ=8). At σ=4.8 it goes 6.70 (bias 0) → 7.10
   (bias 2) → 8.23 (bias 5). Across the requested correlation-time grid the measured MDV is flat
   (6.6-6.8) — but the generator's *realised* correlation time only spans 0.30-0.73 s because
   half the variance is white (printed per cell), so this is a statement about that range.
5. **What makes a slow error expensive is its spectrum, not its τ.** Pure white error (fast share
   1.0, realised τ 0.22 s) gives MDV 6.64; pure slow (fast share 0, realised τ 1.04 s) gives
   7.77; the pre-registered intermediate shares give 6.80-7.28. A low-frequency error is not
   averaged away by a clip-level median, so it is worth about 1.1 deg.
6. **The analytic formula is not a usable guide at these parameters.** It overestimates the
   correlation-time dependence badly (3.53 → 11.17 deg across the τ grid where the measurement
   is flat) and is off by up to 2.5× in either direction on the rest of the grid. The reason is
   that the threshold is calibrated on the *same* clip statistic, so what sets MDV is the width
   of the H0 clip-statistic distribution, not σ/√n_eff.
7. **The number the reader has to hit**: to put the whole-clip floor at 5 deg with a 2 deg bias,
   σ has to come down to about 3 deg — i.e. the trimmed RMSE would have to fall from 4.8 to ~3,
   a further 40 % on top of what G03's fine-tune achieved (3.71). To put the 8-frame segment
   floor at 5 deg, σ would have to fall below 1.5 deg, which nothing in this project is close to.

## Question

The handoff's back-of-envelope model says a reader error with bias σ_b and standard deviation σ,
averaged over a window, can only resolve a violation of order
`z·√(se² + σ_b²) + z·se` with `se = σ/√n_eff` and `n_eff = window/τ_int`. Is that right on this
corpus, what does it imply for the clip-level and segment-level floors, and what accuracy would
the reader need for a given target?

## Setup

**Construction.** Every real clip's path side is kept (`lean_req`, `vel_req`, `tilt_req_deg`,
`t`, `pos`, `yaw`), and the depicted side is replaced by a perfect depiction plus a controlled
error. The violation is a **uniform reduction of the required tilt by Δ degrees along its own
direction, saturating at zero** — the small-violation limit of the tilt-removal edit — swept over
Δ ∈ {0, 2, 4, 6, 8, 10, 12, 15, 20, 30} deg.

**Three depictions, three questions.**

| arm | depicted side | what it measures |
| --- | --- | --- |
| (a) perfect path | `lean_req` + error | what reader accuracy alone implies |
| (b) measured error replay | `lean_req` + the frozen reader's actual per-frame error | the real error's heavier tail |
| (c) mocap truth | the mocap attitude lean + a synthetic error at the centre point | the handoff's construction; carries the real actual-vs-required mismatch |

**Error model.** Per flight one bias vector of magnitude `bias` in a random direction; per clip an
independent AR(1) process, split with the same orthogonal DCT-II split at 0.5 Hz J01 uses, and
each part rescaled to the requested share of `σ²`. The realised fast share and τ_int are measured
with J01's estimators and printed beside every cell; the centre point reproduces 0.500 and 0.49 s
against 0.5 requested.

**Statistics.** The frozen clip-level median residual, and the G-round window scan statistic
(maximum over windows of the window-median residual, hop = half the window, complete windows
only), both calibrated with the leave-one-flight-out 95th percentile over the fit flights' clips.
MDV is the smallest Δ whose detection rate reaches 50 %, interpolated between grid points.
120 draws for the window sweep, 60 per grid cell; 100 clips, 8 flights.

**Cross-checks before any result is read.** (i) The vectorised 2-vector residual equals the frozen
`frame_residual_deg` construction to 9e-13 deg. (ii) The replay reproduces the published per-frame
median error (3.97 deg). (iii) Arm (c) reproduces the published threshold (below).

## Results

### Window length, at the measured centre point

σ = 4.8 deg, bias 2.0 deg, τ = 0.5 s, fast share 0.5 (realised 0.45 s):

| window | H0 clip statistic, median | LOFO threshold | **MDV** |
| --- | --- | --- | --- |
| whole clip (≈48 frames) | 6.45 | 9.04 | **6.92** |
| 24 frames (2.0 s) | 7.49 | 11.08 | 7.42 |
| 16 frames (1.3 s) | 8.40 | 13.06 | 8.17 |
| 12 frames (1.0 s) | 9.21 | 14.72 | 9.01 |
| 8 frames (0.67 s) | 10.50 | 17.51 | **10.52** |

The direction is monotone and the magnitude is large: an 8-frame window needs a violation half
again as big as the whole clip's.

### Parameter sensitivity (whole-clip median, 36 cells)

| σ (deg) | bias (deg) | realised τ (s) | H0 median | threshold | MDV | analytic |
| --- | --- | --- | --- | --- | --- | --- |
| 3.0 | 0 | 0.45 | 3.89 | 5.78 | **4.66** | 3.49 |
| 3.0 | 2 | 0.47 | 4.29 | 6.48 | 5.12 | 5.47 |
| 3.0 | 5 | 0.47 | 6.32 | 8.87 | 7.06 | 10.15 |
| 4.8 | 0 | 0.41 | 6.21 | 8.67 | **6.70** | 5.58 |
| 4.8 | 2 | 0.45 | 6.43 | 9.13 | 7.10 | 7.11 |
| 4.8 | 5 | 0.35 | 7.79 | 10.78 | 8.23 | 11.48 |
| 8.0 | 0 | 0.49 | 10.23 | 13.21 | **9.63** | 9.31 |
| 8.0 | 2 | 0.30 | 10.38 | 13.30 | 9.52 | 10.35 |
| 8.0 | 5 | 0.53 | 11.14 | 14.16 | 10.05 | 14.10 |

The τ sweep is in `summary.json`; across the realised range 0.30-0.73 s the MDV varies by less
than 0.2 deg at σ ≤ 4.8 and by less than 0.5 deg in every cell.

Spectral shape, at the centre σ and bias (the 0.2/0.5/0.8 rows are the pre-registered grid; the
0.0/1.0 rows are **exploratory anchors**):

| fast share | realised τ (s) | MDV |
| --- | --- | --- |
| 0.0 (all slow) | 1.04 | 7.77 |
| 0.2 | 0.80 | 7.28 |
| 0.5 | 0.41 | 6.82 |
| 0.8 | 0.23 | 6.80 |
| 1.0 (all white) | 0.22 | 6.64 |

So the whole range of spectral shapes is worth 1.1 deg, and the analytic formula's τ term is
worth 7.6 deg over the same span. The formula is answering a different question.

### The analytic formula versus the measurement

`MDV = z√(se² + σ_b²) + z·se`, `se = σ/√n_eff`, `n_eff = window/τ_int`, z = 1.645:

* it is **too optimistic** where the error is heavy-tailed (σ = 8, bias 0: 9.31 predicted against
  9.63 measured; σ = 3, bias 5: 10.15 against 7.06 — there it is too pessimistic);
* it is **badly wrong on the correlation time** (3.53 → 11.17 across τ = 0.2 → 2.0 where the
  measurement is 6.64-6.79);
* its one structural miss is that the threshold is calibrated on the same statistic that is being
  tested. Under H0 the clip statistic has a distribution whose width (not σ/√n_eff) sets the
  distance a violation must travel, and for a two-axis Gaussian magnitude that width is set by σ
  and the bias, not by the number of independent samples.

### Anchor: the handoff's construction

| construction | H0 median | LOFO threshold | published |
| --- | --- | --- | --- |
| mocap attitude + synthetic centre-point error | 9.22 | **14.46** [13.21, 15.51] | frozen verdict **14.73** deg (drag-free, lag-only) |
| perfect path + the frozen reader's measured error | 3.97 | 7.66 | per-frame median 3.97 deg (report 02) |

The 14.73 deg threshold is not a property of the reader alone: about half of it is the natural
mismatch between the attitude a real quadrotor flies and the attitude the drag-free path model
demands. That is worth knowing before anyone reads the floor as "the reader's accuracy".

### The floor in one table

| construction | whole clip | 24 frames | 12 frames | 8 frames |
| --- | --- | --- | --- | --- |
| perfect path, synthetic centre error | 6.92 | 7.42 | 9.01 | 10.52 |
| perfect path, measured error | 10.65 | 13.25 | 23.96 | > 30 (never reaches 50 %) |

## What this supports

* A numeric target for the reader, stated in degrees of violation rather than in RMSE: **σ ≈ 3 deg
  (trimmed RMSE ≈ 3 deg) puts the whole-clip floor at 5 deg; σ ≈ 1.5 deg would be needed to put
  an 8-frame segment floor at 5 deg.** G03's fine-tune reached 3.71 deg trimmed RMSE, i.e. it
  bought roughly 1 deg of MDV against the frozen reader.
* **A quantified price for the segment detector's short window**: 3.6 deg of minimum detectable
  violation at 8 frames against the whole-clip median, at the same false-reject. Any future
  segment work should either use ≥24-frame windows or report its floor beside this table.
* **The published 14.7 deg threshold is reproduced from the measured error statistics**, which
  validates both the noise budget and the frozen calibration.
* The analytic formula in the handoff is not usable for design; the measured table is.

## What this does not show

* **Not a detection rate on real edits.** The violation is a uniform tilt reduction on a perfect
  path; the real `tilt_removal` edit also changes the depicted lean's *direction* structure, which
  the coherence and gain tiers see and the median residual does not. The G-round numbers stay the
  reference for real edits.
* **Not a bias-removal experiment.** The bias enters as an unremoved per-flight offset. What a
  per-clip calibration would recover is Stage 1a, which the rules did not select (J01 §rules).
* **Not a claim about τ beyond 0.3-0.7 s.** The generator's realised τ saturates because half the
  variance is white; the exploratory pure-slow anchor (τ 1.04 s, MDV 7.77) is the only point
  above that range, and it is labelled exploratory.
* **Not UZH, not ANWM.** TII clips only; the 1 Hz ANWM paths cannot carry a window statistic.

## Caveats

* The error model is Gaussian per component with a hard band split at 0.5 Hz. The measured error
  is heavier-tailed (7.6 % of frames above 20 deg) and that is why the replay's MDV is 4 deg worse
  than the matched-σ synthetic one; the parametric grid therefore understates the floor the real
  reader implies.
* Detection saturates at 0.80-0.81 rather than 1.0 because a fifth of the clips demand less tilt
  than the largest Δ, so the saturating reduction stops growing there. That ceiling is a property
  of the corpus's excitation, not of the statistic.
* 60 draws per grid cell with 100 clips: the MDV resolution is a couple of tenths of a degree;
  differences below 0.3 deg inside the grid should not be read.
* Eight flights, flight-clustered LOFO calibration: the threshold is a leave-one-flight-out
  quantity with 7 fit flights, not a population quantile.

## Regenerate

```
PYTHONPATH=. python experiments/j02_noise_budget.py     # ~130 s, CPU
```

Outputs: `results/j02_noise_budget/summary.json` (by window length, the 36-cell grid with
realised τ, the share sweep, the exploratory anchors, the measured-error replay and arm (c)).
