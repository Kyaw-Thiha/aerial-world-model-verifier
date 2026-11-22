# J03 — Per-clip calibration on the gate-kept population, and what it absorbs

Track J, Stage 1a. Written 2026-11-21. Pre-registration `configs/j03_perclip_prereg.yaml`
(committed at `d01d398`, before any fit or threshold). Code
`experiments/j03_per_clip_calibration.py`. Results `results/j03_perclip/`. CPU only, **zero GPU
spend**.

## Decision

**A per-clip fit absorbs the violations that matter: on the clip-median tier it loses 31 of 42
full-clip tilt-removal detections, 19 of 52 mirrored-roll detections and 44 of 48 x1.5 detections
at a matched 5 % false-reject, and it is break-even at best on partial edits (window tier: 14/69
against 10/69 at 25 % removal).** The fit-free coherence tier, which is invariant to an offset and
to a gain by construction, is the better relative test by a wide margin — 33/69 at 25 % coverage
against 10/69 fixed and 14/69 fitted — and it cannot be absorbed at all.

## Plain-language summary

1. **The offset is the problem and the drag does not save it.** A free 2-vector offset per clip is
   allowed because J01 measured one (+2.15 to +2.69 deg of pitch on the five non-racing flights).
   But a uniform tilt removal moves the depicted lean in exactly the direction an offset can
   follow, so the fit eats it: full-clip removal detection falls from **42/69 to 11/69** and
   x1.5 from **48/69 to 4/69**.
2. **Adding the bounded drag recovers a little and no more.** Allowing k_d on a fixed grid in
   [0, 0.40] (the handoff's allowance) lifts full-clip removal 7→11 and mirrored roll 26→33 over
   the offset-only fit, and leaves the scale families where the offset put them (4-5/69). The
   frozen fixed-calibration statistic gets 42 and 52 on the same clips.
3. **The bound is where the fit goes.** On 65 of the 72 gate-kept real clips the per-clip k_d sits
   on the upper bound 0.40 (6 at 0.0, 1 at 0.10). The drag nuisance is not measuring drag; it is a
   second magnitude absorber, which is precisely why it partially undoes the offset's damage.
4. **On the window tier the fit is roughly break-even for partial edits and still loses on full
   ones.** Partial tilt removal at 25 % coverage goes 10/69 → 14/69 and mirrored roll 20/69 →
   21/69; full-clip removal goes 34/69 → 20/69. So the relative statistic does buy a little where
   the violation is local, which is what a change-based test is for, and it is still not enough to
   justify the fit.
5. **The coherence tier beats everything, and no fit can touch it.** Per-axis correlation is
   invariant to a translation and to a scaling, so an offset or a gain cannot hide a violation from
   it. At a 3/69 false-reject it detects 33/45/54 of partial-25 / partial-50 / full tilt removal,
   40/54/55 of mirrored roll, and 16/11/6 and 15/15/11 of the two scale families. Its cost is the
   yaw control: 7-9/69 against its own 3/69, the view-dependence J01 measured (roll error vs
   heading, t ≈ 2.4) and G02 recorded as the gain tier's weakness.
6. **The minimum detectable local step tells the same story.** For a uniform tilt step injected
   over 25 % or 50 % of a clip at a 5 % false-reject, the coherence tier needs **16.8 deg (25 %) and
   14.4 deg (50 %)**; the fitted window statistic needs 25.1 and 27.2; the whole-clip median with
   fixed calibration needs more than 30 at 25 % coverage and 22.3 at 50 %; the offset-fitted clip
   median never reaches 50 % detection within the grid.
7. **Two independent cross-checks land.** The fixed-calibration clip threshold comes out at
   **14.05 deg** against the published 14.73, and the coherence threshold at **1.75** against
   G01's 1.73-1.84 on the same gate-kept population. The machinery is the published one.

## Question

If a per-clip offset (and, as the handoff allows, a bounded drag coefficient) is fitted on the clip
under test before scoring, how much detection does the fit absorb for each edit type, and is what
survives — the clip's internal deviation from its own fit — a better statistic at a matched
false-reject than the frozen fixed-calibration median residual?

## Setup

**Population.** 100 TII real clips; the gate-kept stratum (72 clips) per the user's decision, of
which the six dev flights give 69 and the two held-out flights (`13p`, `07a`) are **not scored**.
Edits and controls are the schema-v2 clip set: `tilt_removal`, `mirror_roll`, `scale_0.5`,
`scale_1.5`, and the `yaw_control` / `null_edit` controls, at full coverage and at 25 % / 50 %
coverage composed on the depicted side by the G01 rule.

**Nuisances, fixed before scoring.** A free 2-vector offset per clip, fitted by Huber IRLS (knot
3 deg); and k_d on the grid {0, 0.05, …, 0.40} chosen to minimise the same Huber loss. **No gain is
fitted in any arm** and **nothing is fitted inside a window** — the fit is clip-level, which is
what makes the window statistic relative. The required side is the schema's own
`lean_req + (k_d − 0.295)·vel_req`, never a re-derivation from `pos` (see the caveat below).

**Arms.**

| arm | nuisance | statistic |
| --- | --- | --- |
| F0 | none (k_d = 0.295) | the frozen residual; clip median and window max |
| C1 | per-clip offset | the same two tiers on the offset-removed residual |
| C2 | per-clip offset + bounded k_d | the same two tiers |
| R | none, and none possible | `1 − min(r_x, r_y)` per window, maximum over windows |

**Thresholds.** Leave-one-flight-out 95th percentile on the real gate-kept dev clips, per arm and
tier. The fit is refitted on the clip under test, including on the composed partial edits — that is
what "per-clip calibration" means and the absorption is the price of it.

## Results

### Detection at a matched false-reject (k/n of 69 gate-kept dev clips)

Clip-median tier:

| edit / coverage | F0 fixed | C1 offset | C2 offset + k_d |
| --- | --- | --- | --- |
| removal 25 / 50 / 100 % | 6 / 17 / **42** | 4 / 9 / 7 | 4 / 9 / 11 |
| mirror 25 / 50 / 100 % | 7 / 19 / **52** | 5 / 11 / 26 | 4 / 12 / 33 |
| x0.5 25 / 50 / 100 % | 4 / 8 / 14 | 4 / 4 / 5 | 4 / 4 / 5 |
| x1.5 25 / 50 / 100 % | 5 / 17 / **48** | 4 / 4 / 4 | 4 / 4 / 4 |
| yaw control (25 / 50 / 100 %) | 3 / 3 / 9 | 4 / 4 / 4 | 4 / 4 / 4 |
| null edit (25 / 50 / 100 %) | 3 / 3 / 3 | 4 / 4 / 4 | 4 / 4 / 4 |
| **false reject** | **3/69** | 4/69 | 4/69 |
| LOFO threshold (deg) | 14.05 | 15.91 | 13.84 |

Window tier (8 frames, hop 4, maximum over windows):

| edit / coverage | F0 fixed | C1 offset | C2 offset + k_d | R coherence |
| --- | --- | --- | --- | --- |
| removal 25 / 50 / 100 % | 10 / 18 / 34 | 13 / 15 / 18 | 14 / 16 / 20 | **33 / 45 / 54** |
| mirror 25 / 50 / 100 % | 20 / 32 / 49 | 20 / 27 / 41 | 21 / 30 / 45 | **40 / 54 / 55** |
| x0.5 25 / 50 / 100 % | 5 / 5 / 6 | 7 / 6 / 7 | 6 / 5 / 7 | **16 / 11 / 6** |
| x1.5 25 / 50 / 100 % | 5 / 5 / 5 | 5 / 5 / 6 | 5 / 5 / 5 | **15 / 15 / 11** |
| yaw control (25 / 50 / 100 %) | 5 / 5 / 5 | 5 / 5 / 5 | 5 / 5 / 5 | 7 / 6 / 9 |
| null edit (25 / 50 / 100 %) | 5 / 5 / 5 | 5 / 5 / 5 | 5 / 5 / 5 | 3 / 3 / 3 |
| **false reject** | 5/69 | 5/69 | 5/69 | **3/69** |
| LOFO threshold | 35.56 deg | 31.35 deg | 28.21 deg | 1.75 |

### Absorption (F0 − C2, same clips, same tier)

| edit / coverage | clip tier: clips (pp) | window tier: clips (pp) |
| --- | --- | --- |
| removal 25 % | +2 (+2.9) | **−4 (−5.8)** |
| removal 50 % | +8 (+11.6) | +2 (+2.9) |
| removal 100 % | **+31 (+44.9)** | **+14 (+20.3)** |
| mirror 25 % | +3 (+4.3) | −1 (−1.4) |
| mirror 50 % | +7 (+10.1) | +2 (+2.9) |
| mirror 100 % | **+19 (+27.5)** | +4 (+5.8) |
| x0.5 50 / 100 % | +4 (+5.8) / +9 (+13.0) | 0 / −1 |
| x1.5 50 / 100 % | +13 (+18.8) / **+44 (+63.8)** | 0 / 0 |

A positive number is detection lost to the nuisance parameters; a negative one is detection gained.
The offset-only arm (C1) is worse still on the clip tier: 7/69 removal and 26/69 mirror at full
coverage.

### Minimum detectable local step

Injected as a uniform reduction of the displayed tilt by Δ degrees along the required direction
over a centred interval, keeping the reader's error in the injected region; Δ grid 0-30 deg;
threshold leave-one-flight-out at 5 %; MDV = smallest Δ detected in half the clips, interpolated.

| statistic | 25 % coverage | 50 % coverage |
| --- | --- | --- |
| R coherence | **16.8 deg** | **14.4 deg** |
| C2 window (relative residual) | 25.1 | 27.2 |
| C1 window | 28.5 | > 30 |
| C2 clip | > 30 | 23.6 |
| F0 clip (fixed calibration) | > 30 | 22.3 |
| F0 window | > 30 | 28.0 |

### Diagnostics

* **The drag nuisance pins to its bound**: per-clip k_d on the 72 real gate-kept clips is 0.40 for
  65, 0.00 for 6 and 0.10 for 1. It is absorbing a magnitude mismatch, not estimating drag.
* **The controls behave for the fitted arms**: C1 and C2 sit at exactly their false-reject (4/69) on
  every control cell, which is what a properly calibrated statistic should do. F0's yaw control is
  9/69 on the clip tier against its 3/69, the known view-dependence.
* **The coherence tier is not free**: its yaw control is 7-9/69 against its 3/69. Any use of it
  carries the same caveat G02 recorded for the gain tier.

## What the pre-registered success criterion says

The detector clause was "C2 beats F0 on at least one family at 25 % or 50 % coverage without losing
more than its absorption on the full-clip cells". C2 does beat F0 at 25 % on the window tier
(14 vs 10 removal, 21 vs 20 mirror) — four and one clips, inside the noise of 69 — while losing 14
clips on full removal and 4 on full mirror. **Read plainly: the branch is a negative result for
per-clip calibration as a primary tier.** The measurement clause is met in full: every absorption
number, false-reject and control is reported with its denominator.

## What this supports

* **A quantitative answer to the handoff's own warning.** A per-clip offset can absorb 45-64 % of
  the detection of the edits that matter, measured rather than argued; the bounded drag recovers
  part of it and is itself pinned at its bound.
* **The fit-free coherence tier is the right relative test.** It is invariant to an offset and to a
  gain by construction, it detects twice as much at 25 % coverage as anything fitted, and its
  minimum detectable local step (14-17 deg) is half the fitted statistics' (>25 deg).
* **A concrete recommendation**: keep per-clip calibration out of the decision layer, or use it only
  as one tier beside coherence and never as the only statistic. If a nuisance is wanted, an offset
  is acceptable only for *localisation*, never for *adjudication*.

## What this does not show

* **Not that calibration is useless.** It removes a real 2-3 deg bias (J01) and it makes the
  statistic's false-reject exactly nominal on every control; what it cannot do is decide.
* **Not a gain-fit study.** No gain was fitted anywhere, by design; the x0.5/x1.5 cells show what
  the offset and drag do to magnitude edits but say nothing about a fitted gain, which would absorb
  them completely.
* **Not held-out evidence.** The two held-out flights are unscored, as pre-registered.
* **Not a localisation result.** No IoU or onset number is reported here; the question was
  absorption and relative sensitivity.

## Caveats

* **The required lean is taken from the schema, not re-derived.** Recomputing it from the stored
  `pos`/`yaw`/`t` at the 12 Hz clip rate disagrees with the stored `lean_req` by a median of 8.6 deg
  over the 100 clips (84 above 1 deg) because the second derivative is computed upstream at the
  native 274.73 Hz rate; the stored lean and `tilt_req_deg` are internally consistent to 3e-4. So
  k_d is varied linearly as `lean_req + (k_d − 0.295)·vel_req`. Recorded in `docs/decisions.md`.
* **The gate-kept population is selected**: 72 of 100 clips, and J01 measured that 88 % of the tail
  is in the 28 it drops. Rates describe the admissible corpus.
* **Six dev flights** make every flight-clustered quantity descriptive; the thresholds are
  leave-one-flight-out with five fit flights.
* **The MDV grid stops at 30 deg**, so several cells are censored (`> 30`) rather than measured;
  five cells need a larger grid to be resolved and are reported as censored.
* The Huber knot (3 deg) and the k_d grid were fixed in the pre-registration; a different knot moves
  the fit's robustness but not the absorption mechanism.

## Regenerate

```
PYTHONPATH=. python experiments/j03_per_clip_calibration.py     # ~30 s, CPU
```

Outputs: `results/j03_perclip/{table.json, absorption.json, mdv.json, diagnostics.json, rows.csv,
mdv_rows.csv}`.
