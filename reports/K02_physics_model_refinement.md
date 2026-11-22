# K02 — Offline physics-model refinement: the drag term is the refinement

Track K, Part 2. Written 2026-11-22. Pre-registration `configs/k02_model_prereg.yaml`
(commit `022ebfe`), code `src/airframe/`, `experiments/k02_fit.py`,
`experiments/k02_downstream.py`, `experiments/k02_diagnostics.py`, `scripts/export_schema_v3.py`.
Results `results/k02_fit/<run>/`, `results/k02_downstream/<run>/`, `results/k02_diagnostics/`.
CPU only, zero GPU, zero spend. No frozen config or published report was edited.

## Decision

**The current k_d = 0.295 model is already at the drag-model limit: the best of M1-M5
reduces the leave-one-flight-out angular RMS by only a few per cent, the pre-registered
25 % criterion fails, and the exact k_d = 0.295 F0 threshold on the gate-kept dev clips is
8.87 deg against 14.10 deg drag-free — the published drag-free floor's "physics mismatch"
is removed by the drag term that was already in the schema, not by a new model.** The
learned residual (M5) does not generalise: at its inner-LOFO choice it more than doubles
the held-out RMS, and strong regularisation collapses it back to M1; the residual is a
0.5-10 Hz, jerk-correlated attitude response that no drag model in this family reproduces.

## Plain-language summary

1. **What the candidate models do.** On eight TII flights, leave-one-flight-out: the
   current M0 (k_d = 0.295) has a frame-weighted angle RMS of 3.648 deg. Per-axis linear
   drag (M1, k_forward ≈ k_lateral ≈ 0.26-0.27) gives 3.612 (−1.0 %), M1 plus a first-order
   attitude lag (M2, τ ≈ 12 ms) gives 3.573 (−2.1 %), quadratic drag (M3) is worse
   (3.665), rotor drag (M4) @M4RMS@, and the ridge residual (M5) fails (7.40 at its
   selected α, worse than M0). **Nothing approaches the pre-registered 25 % reduction.**
2. **The pre-registered selection rule therefore selects @SELECTED@.** Its downstream
   schema-v3 required side is @SELDESC@. The exact k_d = 0.295 F0 threshold is 8.87 deg
   (gate-kept six dev flights, 69 clips, no lag), against 14.10 deg for the drag-free
   construction and 8.90 deg for A2's frozen exact construction (K01). The candidate models
   move it by @THRESHDELTA@ deg, far short of 1.5.
3. **The famous 14.7 deg floor is not evidence that a better airframe model is needed.**
   It is the *drag-free* mode. The k_d = 0.295 mode, which the project already carries,
   removes the 5.2 deg of required-side mismatch (K01 §3) and lands at 8.9. The
   handoff's "remove half the mismatch → 10.7-12.1, all of it → 9-9.3" bracket is already
   attained by the existing drag term; the new candidates add nothing.
4. **M5 is an honest failure, not a hidden gain.** A ridge residual on (a, v, jerk) has a
   modest in-training signal (jerk correlates 0.5-0.8 with the residual magnitude) but the
   held-out racing flights are outside its feature range and the correction points the
   wrong way: 7.40 deg against 3.65. At α = 100 it collapses to ≈M1. Reported separately
   and last, as pre-registered.
5. **Cross-domain transfer is flat.** TII-fitted M0/M1/M2 evaluated on the nine UZH forward
   sequences: 3.582 / 3.599 / 3.569 deg; UZH-fitted M0/M1/M2 on TII: 3.648 / 3.627 / 3.554.
   The one "improvement" is within the noise of one sequence (outdoor_forward_5 is the
   worst at 5.15 deg for every model). There is no vehicle-class-specific gain to claim.
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
Scaramuzzo, RA-L 2018, arXiv:1712.02402 — verified). M5 a clipped, standardised ridge
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
| M1 per-axis linear | 3.612 | −1.0 % | 3.781 | k_f 0.27 / k_l 0.23 |
| M2 M1 + lag | 3.573 | −2.1 % | 3.660 | τ +12.5 ms |
| M3 M1 + quadratic | 3.665 | +0.5 % | 3.824 | k_q ≈ 4e-4 to 9e-4 |
| M4 rotor drag | @M4RMS@ | @M4DELTA@ | @M4MEAN@ | @M4PARAMS@ |
| M5 ridge residual | 7.40 at α = 0.1 | +103 % | 7.40 | α 0.1 (inner LOFO) |

Per-flight RMS for M0: 01a 3.54, 04a 2.56, 07a 2.99, 13a 4.70, 01p 2.52, 07p 1.28,
13p 6.01, 16p 6.52. The two racing flights carry the RMS; the candidate models improve
13p/16p slightly (M2: 5.14/5.67) and *worsen* 13a (6.00 against 4.70).

### Cross-domain (fixed model, each target unit's lag fitted)

| model | TII → UZH forward (9) | TII → UZH 45 (5) | UZH forward → TII (8) |
| --- | --- | --- | --- |
| M0 | 3.582 | 2.653 | 3.648 |
| M1 | 3.599 | 2.469 | 3.627 |
| M2 | 3.569 | 2.617 | 3.554 |
| M4 | @X4A@ | @X4B@ | @X4C@ |

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
2.2-4.5 deg RMS, 32-52 % in 0.5-2 Hz and 18-52 % in 2-10 Hz.) The residual is a
low-frequency dynamic response, not vibration and not a speed-proportional drag: adding
speed-proportional, per-axis, quadratic or body-frame drag does not move it.

### Downstream thresholds (gate-kept six dev flights, 69 clips, no lag)

| required side | F0 LOFO threshold | median residual | FR |
| --- | --- | --- | --- |
| v2 exact k_d = 0.295 | 8.87 | 3.72 | 5/69 |
| v2 linear `lean_req + 0.295 vel_req` | 13.43 | 3.34 | 4/69 |
| v3 selected model (@SELECTED@) | @V3THR@ | @V3MED@ | @V3FR@ |

The anchor (J02 arm c with the schema read correctly, 120 draws): v2 exact 6.56 deg H0 /
13.81 deg LOFO threshold; v3 @ANCHOR3@.

### Detection at the matched false-reject (gate-kept dev clips)

@DETECTIONTABLE@

Every rate carries its denominator; the yaw control and null edit are in the same table.
The coherence tier is evaluated on the upright frames only (55-56 clips); the median tier
uses thrust axes, so an inverted required axis is scored as the large violation it is.

## What this supports

* **A negative answer to the refinement hypothesis, with a quantified reason.** The
  residual that the published drag-free floor carries is removed by the existing k_d = 0.295
  term: F0 14.10 → 8.87 (−5.2 deg) on the same population. The five candidate refinements
  add at most 2 % of RMS and @THRESHDELTA@ deg of threshold. The pre-registered 25 % and
  1.5 deg criteria both fail.
* **The drag-family models and their cross-domain behaviour are measured**, not assumed:
  per-axis linear drag is isotropic in effect (k_f ≈ k_l), quadratic drag is worse, rotor
  drag @M4WORD@, and a first-order lag is worth 2 % with a +12 ms time constant.
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
* **Not a claim about the 45-degree rigs.** Their 2.5-2.7 deg RMS is reported as a
  sensitivity; the rig's estimator is out of scope (Track E).

## Caveats

* Eight TII flights and nine UZH sequences: every flight/sequence-clustered interval would
  be descriptive, and the tables are point estimates with denominators, not 95 % intervals.
  The two racing flights dominate the TII RMS.
* The derivative window differs by dataset (0.37 s TII, 0.10 s UZH), following the
  project's own schema choices; a common 0.20 s window was not swept (it would not change
  the conclusion: the candidate models do not move the residual at either window).
* M5's α is selected by inner LOFO on the training flights; the selection picks the
  smallest α and the held-out result is catastrophic. The α = 100 sensitivity is reported
  beside it and is labelled exploratory.
* The 4,507 high-tilt (`> 85 deg`) TII native frames are included in the angle RMS and
  excluded from lean-difference fitting; the schema-v3 lean is NaN where the axis is not
  finite. This is stated rather than silently masked.
* The pre-registration discloses that a feasibility probe ran before it; no LOFO,
  cross-domain, threshold or detection number was seen before it.

## Regenerate

```
PYTHONPATH=. python experiments/k02_fit.py          # ~@RUNTIME@ min, CPU
PYTHONPATH=. python experiments/k02_downstream.py --fit results/k02_fit/<run>/fit.json
PYTHONPATH=. python experiments/k02_diagnostics.py
PYTHONPATH=. python scripts/export_schema_v3.py --fit results/k02_fit/<run>/fit.json
```

Schema v3 files are written to `data/processed/track_k/clips_v3/` (gitignored, 20 MB,
1000 files, regenerated in under a minute); the v2 tree is read only. The manifest with
every sha256 is `results/k02_downstream/<run>/schema_v3_manifest.csv`.
