# K02 — Offline physics-model refinement: the drag term is the refinement

Track K, Part 2. Written 2026-11-22. Pre-registration `configs/k02_model_prereg.yaml`
(commit `022ebfe`), code `src/airframe/`, `experiments/k02_fit.py`,
`experiments/k02_downstream.py`, `experiments/k02_diagnostics.py`, `scripts/export_schema_v3.py`.
Results `results/k02_fit/20261122T084458Z/`, `results/k02_downstream/20261122T091115Z/`,
`results/k02_diagnostics/`.
CPU only, zero GPU, zero spend. No frozen config or published report was edited.

## Decision

**The current k_d = 0.295 model is already at the drag-model limit: the best of M1-M5
(M2) reduces the leave-one-flight-out angular RMS by 1.2 %, M0 is selected by the
pre-registered parsimony rule, the pre-registered
25 % criterion fails, and the exact k_d = 0.295 F0 threshold on the gate-kept dev clips is
8.87 deg against 14.10 deg drag-free — the published drag-free floor's "physics mismatch"
is removed by the drag term that was already in the schema, not by a new model.** The
learned residual (M5) does not generalise: at its inner-LOFO choice it is 36 % worse
than M0, and strong regularisation collapses it back to M1; the residual is a
0.5-10 Hz, jerk-correlated attitude response that no drag model in this family reproduces.

## Plain-language summary

1. **What the candidate models do.** On eight TII flights, leave-one-flight-out: the
   current M0 (k_d = 0.295) has a frame-weighted angle RMS of 3.648 deg. Per-axis linear
   drag (M1, k_forward 0.274 / k_lateral 0.232) gives 3.615 (−0.9 %), M1 plus a first-order
   attitude lag (M2, τ ≈ 12 ms) gives 3.604 (−1.2 %), quadratic drag (M3) is worse
   (3.665, +0.5 %), rotor drag (M4) is worse (3.695, +1.3 %), and the ridge residual (M5)
   fails (4.95 at its selected α, +36 %). **Nothing approaches the pre-registered 25 %.**
2. **The pre-registered selection rule therefore selects M0, the current model.** Its
   schema-v3 required side is the exact k_d = 0.295 side. The exact k_d = 0.295 F0 threshold is 8.87 deg
   (gate-kept six dev flights, 69 clips, no lag), against 14.10 deg for the drag-free
   construction and 8.90 deg for A2's frozen exact construction (K01). The candidate models
   move it by 0.00 deg, far short of 1.5.
3. **The famous 14.7 deg floor is not evidence that a better airframe model is needed.**
   It is the *drag-free* mode. The k_d = 0.295 mode, which the project already carries,
   removes the 5.2 deg of required-side mismatch (K01 §3) and lands at 8.9. The
   handoff's "remove half the mismatch → 10.7-12.1, all of it → 9-9.3" bracket is already
   attained by the existing drag term; the new candidates add nothing.
4. **M5 is an honest failure, not a hidden gain.** A ridge residual on (a, v, jerk) has a
   modest in-training signal (jerk correlates 0.5-0.8 with the residual magnitude) but the
   held-out racing flights are outside its feature range and the correction points the
   wrong way: 4.95 deg against 3.65. At α = 100 it collapses to the M1/M0 level (mean
   3.78 deg), i.e. the apparent signal is extrapolation. Reported separately and last, as
   pre-registered.
5. **Cross-domain transfer is flat.** TII-fitted M0/M1/M2 evaluated on the nine UZH forward
   sequences: 3.582 / 3.599 / 3.569 deg; UZH-fitted M0/M1/M2 on TII: 3.648 / 3.627 / 3.554.
   The spread across M0/M1/M2 is 0.03 deg in the TII→UZH direction and 0.09 deg in the
   reverse, and outdoor_forward_5 is the worst sequence for every model (5.15-5.76 deg).
   There is no vehicle-class-specific gain to claim.
6. **The residual is not drag.** Its variance is 0.5-10 Hz (TII 40-74 % in 0.5-2 Hz,
   17-52 % in 2-10 Hz, under 1 % above 10 Hz), and its magnitude correlates with jerk
   (0.48-0.77) and speed (0.2-0.8) on the racing flights. That is a dynamic attitude
   response, but neither a lag, a lead, a quadratic term nor a body-frame drag reproduces
   it; the models that add those terms do not move the RMS.
7. **Deliverable.** Schema v3 (new files, v2 untouched) stores the *exact* model required
   lean **and the required thrust axis** at the clip's frame times, with explicit semantics
   (no `(k_d − 0.295)` rescaling), and a generator (`scripts/export_schema_v3.py`). The
   axis is necessary: 4,507 TII native frames have a required tilt above 85 deg and some
   are inverted, which a lean cannot represent. K01 showed the linear schema
   reparameterisation is off by a median 2.2 deg and a p95 20.4 deg on testable frames; v3
   removes that defect whether or not any model ever beats M0.

## Question

Does an offline, per-vehicle-class model of how the airframe realises a commanded
acceleration reduce the actual-vs-required mismatch and the real-clip verdict threshold,
with every parameter fixed before any test clip is seen? The pre-registered success
criterion was a LOFO RMS reduction of at least 25 % **and** a real-clip threshold lower by
at least 1.5 deg with false-reject unchanged.

## Setup

**Data.** TII: eight flights, mocap 274.73 Hz, position and body attitude, 0.37 s / poly-3
Savitzky-Golay derivatives, first and last half-window masked (103 samples). UZH-FPV:
nine forward sequences from the 500 Hz Leica truth, truncated to the image-frame span,
0.10 s / poly-3; the five 45-degree-down sequences are a sensitivity only. Truth attitude
gives the actual thrust axis (`body z`); the path gives the model axis. No depicted lean,
camera, mount or edit enters a fit.

**Models (pre-registered, in order of complexity).** M0 current (`a + g e3 + 0.295 v`).
M1 per-axis linear drag in the heading frame (`k_forward`, `k_lateral`). M2 M1 plus a
first-order lag on the axis, one τ per vehicle class, allowed negative so the data can
reject the direction. M3 M1 plus quadratic drag. M4 body-frame rotor drag
`R diag(d_x,d_y,0) R^T v`, solved by fixed-point iteration (Faessler, Franchi &
Scaramuzza, RA-L 2018, arXiv:1712.02402 — verified). M5 a clipped, standardised ridge
residual on 11 features from (a, v, jerk), α by inner LOFO, never selected for the
downstream. All models share the frozen per-unit lag nuisance (±150 ms, 25 ms grid) and the
M0 vertical drag term.

**Protocol.** Fit by pooled squared lean residual, LOFO over flights; per-flight lag
refitted on the evaluated flight only. Cross-domain: fit on all TII, evaluate UZH, and the
reverse. Thresholds: J03's F0 (per-clip median axis residual, LOFO 95th percentile) on the
gate-kept six dev flights, with and without the lag; J02's arm (c) anchor with the schema
semantics corrected. Detection: tilt removal, mirror, x0.5, x1.5 at full and 25/50 %
coverage, median and coherence tiers, at the matched false-reject, with the yaw and null
controls. The two held-out flights (`13p`, `07a`) are never scored.

## Results

### LOFO mismatch (eight TII flights, per-flight lag nuisance)

| model | pooled RMS (frame-weighted, deg) | change vs M0 | per-flight mean (deg) | fitted parameters |
| --- | --- | --- | --- | --- |
| M0 current | 3.648 | — | 3.766 | none (k_d = 0.295) |
| M1 per-axis linear | 3.615 | −0.9 % | 3.786 | k_f 0.274 / k_l 0.232 |
| M2 M1 + lag | 3.604 | −1.2 % | 3.705 | k_f 0.290 / k_l 0.310, τ +11.9 ms |
| M3 M1 + quadratic | 3.665 | +0.5 % | 3.827 | k_q 0.0011 |
| M4 rotor drag | 3.695 | +1.3 % | 3.884 | d_x = d_y = 0.25 |
| M5 ridge residual | 4.95 at α = 0.1 | +36 % | 4.93 | α 0.1 (inner LOFO) |

Per-flight RMS for M0: 01a 3.54, 04a 2.56, 07a 2.99, 13a 4.70, 01p 2.52, 07p 1.28,
13p 6.01, 16p 6.52. The two racing flights carry the RMS; the candidate models improve
13p/16p (M2: 5.11/6.32 against 6.01/6.52) and *worsen* 13a (5.78 against 4.70).

### Cross-domain (fixed model, each target unit's lag fitted)

| model | TII → UZH forward (9) | TII → UZH 45 (5) | UZH forward → TII (8) |
| --- | --- | --- | --- |
| M0 | 3.582 | 2.653 | 3.648 |
| M1 | 3.599 | 2.469 | 3.627 |
| M2 | 3.569 | 2.617 | 3.554 |
| M4 | 3.714 | 2.341 | 3.761 |

The reverse direction for M0 is identical to the TII in-domain number because M0 has no
fitted parameter. Where a parameter is fitted, the cross-domain result is within 0.05 deg
of the in-domain result, i.e. no transferable vehicle-class signal.

### Why: the residual's spectrum and correlates (M0, per-flight lag)

| unit | RMS | median | >10° | >20° | corr speed | corr accel | corr jerk | <0.5 / 0.5-2 / 2-10 / >10 Hz |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 01a | 3.17 | 1.25 | 2.4 % | 0.0 % | 0.80 | 0.33 | 0.60 | 0.09 / 0.74 / 0.17 / 0.00 |
| 01p | 2.40 | 0.80 | 0.4 % | 0.3 % | 0.02 | 0.12 | 0.24 | 0.31 / 0.57 / 0.11 / 0.00 |
| 07p | 1.23 | 0.77 | 0.0 % | 0.0 % | 0.16 | 0.21 | 0.18 | 0.24 / 0.63 / 0.13 / 0.00 |
| 13a | 3.71 | 0.84 | 2.9 % | 0.9 % | 0.50 | 0.31 | 0.67 | 0.05 / 0.43 / 0.52 / 0.00 |
| 13p | 5.29 | 2.30 | 6.5 % | 1.1 % | 0.21 | −0.07 | 0.48 | 0.10 / 0.44 / 0.45 / 0.00 |
| 16p | 5.68 | 2.72 | 8.7 % | 0.8 % | 0.23 | −0.10 | 0.52 | 0.08 / 0.40 / 0.52 / 0.01 |

(UZH forward sequences in `results/k02_diagnostics/diagnostics.json`; they sit at
2.2-4.5 deg RMS, 32-46 % in 0.5-2 Hz and 18-52 % in 2-10 Hz.) The residual is a
low-frequency dynamic response, not vibration and not a speed-proportional drag: adding
speed-proportional, per-axis, quadratic or body-frame drag does not move it.

### Downstream thresholds (gate-kept six dev flights, 69 clips, no lag)

| required side | F0 LOFO threshold | median residual | FR |
| --- | --- | --- | --- |
| v2 exact k_d = 0.295 | 8.87 | 3.72 | 5/69 |
| v2 linear `lean_req + 0.295 vel_req` | 13.43 | 3.34 | 4/69 |
| v3 selected model (M0, the current model) | 8.87 | 3.72 | 5/69 |

The anchor (J02 arm c with the schema read correctly, 120 draws): v2 exact 6.56 deg H0 /
13.81 deg LOFO threshold; v3 6.56 deg H0 / 13.81 deg LOFO threshold.

### Detection at the matched false-reject (gate-kept dev clips)

| edit / coverage | median tier 25 / 50 / full | coherence tier 25 / 50 / full |
| --- | --- | --- |
| tilt removal | 7/69 · 43/69 · 55/69 | 39/55 · 52/55 · 54/55 |
| mirrored roll | 6/69 · 24/69 · 54/69 | 39/55 · 53/55 · 55/55 |
| x0.5 | 6/69 · 17/69 · 48/69 | 24/55 · 20/55 · 10/55 |
| x1.5 | 6/69 · 29/69 · 53/69 | 17/55 · 9/55 · 6/55 |
| yaw control | 5/69 · 6/69 · 8/69 | 9/55 · 8/55 · 15/56 |
| null edit | 5/69 · 5/69 · 5/69 | 3/55 · 3/55 · 3/55 |

False reject: median tier 5/69, coherence tier 3/55. The coherence denominators vary by
one or two clips between coverages because a window with no upright frame is not scored.
Flight-clustered percentile intervals for the full-coverage column (5,000 replicates,
`rng(1)`, six dev flights, descriptive rather than calibrated): median tier removal
0.30-0.94, mirror 0.23-0.94, x0.5 0.27-0.85, x1.5 0.24-0.92, yaw control 0.06-0.38,
null edit 0.02-0.32; coherence tier removal 0.96-1.00, mirror 1.00-1.00, x0.5 0.09-0.44,
x1.5 0.03-0.57, yaw control 0.16-0.67, null edit 0.01-0.43.

Every rate carries its denominator; the yaw control and null edit are in the same table.
The coherence tier is evaluated on the upright frames only (55-56 clips); the median tier
uses thrust axes, so an inverted required axis is scored as the large violation it is.

## What this supports

* **A negative answer to the refinement hypothesis, with a quantified reason.** The
  residual that the published drag-free floor carries is removed by the existing k_d = 0.295
  term: F0 14.10 → 8.87 (−5.2 deg) on the same population. The five candidate refinements
  reduce the RMS by at most 1.2 % and move the threshold by 0.00 deg. The pre-registered 25 % and
  1.5 deg criteria both fail.
* **The drag-family models and their cross-domain behaviour are measured**, not assumed:
  per-axis linear drag is isotropic in effect (k_f ≈ k_l), quadratic drag is worse, rotor
  drag is worse than the scalar model (+1.3 %), and a first-order lag is worth 1.2 %
  with a +12 ms time constant.
* **A schema-v3 required side that is exact and unambiguous**, with the axis stored beside
  the lean, generated from the committed fit file, and never editing v2.
* **A diagnostic of where the remaining mismatch lives**: 0.5-10 Hz, jerk-correlated,
  flight-specific. Any future work on it should target attitude dynamics/vibration or the
  reader, not another drag coefficient.

## What this does not show

* **Not that the residual is irreducible.** It is not reproduced by this model family, but
  a structural model (flexible arm, motor lag, controller) was not fitted and is not
  excluded by these data.
* **Not a detection improvement.** The v3 and v2-exact detection tables are the same by
  construction when M0 is selected; the report does not claim a gain from relabelling the
  schema.
* **Not a UZH verdict.** Only the path-side mismatch was fitted and evaluated on UZH; no
  depicted side, threshold or edit was used there.
* **Not a re-run of J03's absorption.** The per-clip calibration question is Part 1/J03's;
  this track's v3 is a *fixed* offline model, and no nuisance is fitted per clip inside it.
* **Not a claim about the 45-degree rigs.** Their 2.3-2.7 deg RMS is reported as a
  sensitivity; the rig's estimator is out of scope (Track E).

## Caveats

* Eight TII flights and nine UZH sequences: every flight/sequence-clustered interval would
  be descriptive, and the tables are point estimates with denominators, not 95 % intervals.
  The two racing flights dominate the TII RMS.
* The derivative window differs by dataset (0.37 s TII = 103 samples, 0.10 s UZH),
  following the project's own schema choices; K01 uses A2's frozen 101-sample window, which
  is why its exact k_d = 0.295 F0 is 8.90 deg here against 8.87; a common 0.20 s window was not swept (it would not change
  the conclusion: the candidate models do not move the residual at either window).
* M5's α is selected by inner LOFO on the training flights; the selection picks the
  smallest α and the held-out result is clearly worse. The α = 100 sensitivity is reported
  beside it and is labelled exploratory.
* The 4,507 high-tilt (`> 85 deg`) TII native frames are included in the angle RMS and
  excluded from lean-difference fitting; the schema-v3 lean is NaN where the axis is not
  finite. This is stated rather than silently masked.
* Amendment A (`configs/k02_amendment_a.yaml`) replaces Powell with a deterministic
  coarse-to-fine grid for M4, because the implicit solve's singularity made the line search
  unstable and it never converged inside 15 minutes. The parameterisation, bounds, 12
  fixed-point iterations, objective, lag nuisance and every downstream rule are unchanged.
  The amendment was written before any M4 LOFO, cross-domain, threshold or detection
  number existed (the in-sample check had not completed; when it did, it gave 3.659 deg,
  i.e. no better than M0).
* The pre-registration discloses that a feasibility probe ran before it; no LOFO,
  cross-domain, threshold or detection number was seen before it.

## Regenerate

```
PYTHONPATH=. python experiments/k02_fit.py          # ~22 min, CPU
PYTHONPATH=. python experiments/k02_downstream.py --fit results/k02_fit/<run>/fit.json
PYTHONPATH=. python experiments/k02_diagnostics.py
PYTHONPATH=. python scripts/export_schema_v3.py --fit results/k02_fit/<run>/fit.json
```

Schema v3 files are written to `data/processed/track_k/clips_v3/` (gitignored, 20 MB,
1000 files, regenerated in under a minute); the v2 tree is read only. The manifest with
every sha256 is `results/k02_downstream/<run>/schema_v3_manifest.csv`.
