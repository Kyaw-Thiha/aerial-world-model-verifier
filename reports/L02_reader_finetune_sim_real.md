# L02 — Training the reader on simulated scenes: what transfers and what does not

**Status.** Simulated held-out scenes (`field_settlement`, `coastal_cliff`): **not run** — 0 of 40 held-out clips are on R2; the deferred pass stays open with I02.

**Decision.** Training the reader on simulated scenes does not transfer to the real held-out
flights: real-only is the best arm (7.51 deg pooled trimmed RMSE against G03's 8.29, a 9.4 % gain
that misses the pre-registered 15 % bar), while simulated-only and mixed+DR are 2.6-2.7x worse
(19.87 and 20.44), and no new reader lowers the coherence tier's false-reject (frozen 5/72,
real-only 7/72, sim-only and mixed 6/72). The simulated scene-family spread does shrink
(2.26 -> 1.05 and 0.63 deg), but by levelling the open families from 0.9-1.3 deg to 2-3.3 deg, and
the simulated-only reader collapses full-removal detection from 70/72 to 43/72.

## Question

The reader's error is scene-dependent (L01: 7-10 deg in the simulated forest, 0.5-2.8 deg in
the open families) and its real bias is a forward pitch offset that no single rig rotation
removes (J01). Does training it on hundreds of simulated scenes with exact attitude — alone, or
mixed with the real flights under domain randomisation — reduce scene-dependent bias and improve
it on held-out real flights, and where does sim-to-real fail?

## Setup

**Data.** Simulated: the 80 feasible development clips of the frozen Track I set, 3,200 frames
(40 per clip, spread over each flight), labels exact to 5.7e-12 deg from the render pose. Real:
the 5,007 TII frames and mocap labels G03 used. The two held-out real flights frozen in round 1
(`flight-13p-trackRATM`, `flight-07a-lemniscate`) are never in a training pool, a normalisation,
a schedule or a gate fit; the six development flights supply the real folds. No simulated
held-out scene is scored: `field_settlement` and `coastal_cliff` are 0/40 on R2 (I02 incomplete),
so the held-out-scene half of the pre-registered evaluation is **deferred**, and the simulated
development proxy is leave-one-scene-family-out.

**Arms.** All three use G03's architecture and loss, and the same number of gradient steps
(600 per epoch, two epochs, batch 2, lr 2e-5 cosine, LM 10 steps in the loop):

* **(a) real-only** — TII dev frames, the G03 recipe (`distorted`, `simple_divisional`), six
  leave-one-flight-out folds, final on the six dev flights;
* **(b) sim-only** — simulated dev frames (`pinhole`), three leave-one-family-out folds, final on
  all 18 scene units, evaluated on the real held-out flights through `simple_divisional` (the
  weights are unchanged; the camera model is a forward-pass setting);
* **(c) mixed + domain randomisation** — real + simulated frames, `distorted` both sides. The
  simulated side gets the pre-registered DR: photometric jitter, a camera-roll rotation (label
  rotated exactly), and a synthetic `simple_divisional` fisheye warp whose label needs no change
  because a change of camera model leaves the gravity direction invariant. Folds: three
  leave-one-family-out on the simulated side and six leave-one-flight-out on the real side.

**Budget amendment.** The pre-registration asked for 1,200 frames per epoch. The first full-budget fold
measured 0.8-1.5 s per step with the LM in the loop, which would have cost 28 A10G-hours for
21 models, so the budget was halved to 600 frames per epoch **before any fold outcome existed**
(commit `2bf2cb2`); every arm keeps the same number of steps.

**Downstream.** The scale-free coherence tier (the primary verdict statistic), on the 100 real
clips and the frozen gate-kept 72. Only the depicted side changes: each reader's gravity on each
clip's own frames rebuilds `depicted_lean`; the path side is frozen. Full tilt removal and
mirrored roll are applied on the depicted side; the yaw control and null edit are run on
regenerated pixels through Track A's own edit pipeline, validated on the condition-relative
change. The check is exact where the edits are small (identity within 0.07 deg on every flight)
and fails on the extreme ceiling-view flights at the largest edits: tilt removal +0.81 deg on 13p
and 16p, mirror -0.41 deg on 04a, yaw +0.37 deg on 16p (amendment C and the full-pass numbers in
`docs/decisions.md`). The detection rates are measured on those same pixels and are unaffected at
16-27 deg edit magnitudes; the controls carry the stated uncertainty. Thresholds
are leave-one-flight-out 95th percentiles; intervals are flight-clustered bootstraps (5,000
draws, seed 1, 8 clusters) and are descriptive.

## Results

### Held-out real flights: the real-only arm helps a little, simulated-only hurts

| reader | 13p median | 13p trimmed RMSE | 07a median | 07a trimmed RMSE | pooled trimmed RMSE |
| --- | --- | --- | --- | --- | --- |
| frozen | 7.75 | 18.40 | 2.54 | 6.54 | 14.75 |
| G03 fine-tune | 6.52 | 9.86 | 1.90 | 5.22 | 8.29 |
| arm real | 6.30 | 8.85 | 1.49 | 6.06 | 7.51 |
| arm sim | 8.64 | 25.71 | 3.98 | 6.51 | 19.87 |
| arm mixed | 7.40 | 27.05 | 1.02 | 5.64 | 20.44 |

The pre-registered S1 bar was a 15 % reduction in the pooled 90 %-trimmed RMSE against G03's
8.29 deg on the same 906 frames, i.e. 7.04 deg. **Arm (a), real-only, reaches 7.51 deg (9.4 %);
arm (b), simulated-only, reaches 19.87 deg and fails by a wide margin** (13p alone 25.71 deg).
Arm (b)'s failure is not a uniform degradation: on 07a it is 6.51 deg, essentially the frozen
reader's 6.54, while 13p -- the ceiling-view racing flight with 35 % of frames above 10 deg --
collapses to 8.64 deg median and 25.7 trimmed RMSE. The simulated fine-tune has learned features
that do not survive that flight's imagery.

Arm (a) itself is mixed per flight: 13p improves from G03's 9.86 to 8.85, 07a worsens from 5.22
to 6.06. The pooled 9.4 % is driven by 13p. Arm (a)'s development folds show the same
heterogeneity: it beats G03 on four of six dev flights (01a 1.88 vs 3.39, 01p 1.40 vs 1.88, 04a
1.60 vs 2.74, 07p 2.52 vs 2.82) and loses on the two hard ones (13a 33.70 vs 31.53, 16p 18.87 vs
9.21).

### Simulated scenes: the family spread shrinks by levelling, not by improving everything

Out-of-family predictions for arm (b): forest 4.60 deg median (frozen 8.72), urban 2.11 (0.94),
industrial 2.27 (1.29). The family bias spread falls from 2.26 to 1.05 deg (-54 %, S2 met on this
measure), and the out-of-family *error* spread falls from 7.78 to 2.49 deg (-68 %), but the
mechanism is a trade: the forest improves by half while both open families get two to three times
worse. The mixed arm reaches a spread of **0.63 deg (-72 %)** with the same mechanism (urban 1.84,
forest 2.02, industrial 1.39; out-of-family error 3.29/6.73/3.13, an error spread of 3.60 deg
against the frozen 7.78): it is the best spread of the three and one of the worst readers on the
real held-out flight. A scene-spread criterion alone can therefore be satisfied by flattening
every scene to the same mediocre level, which is why S1 and the downstream block decide the
outcome. Any claim that simulated training "reduces scene dependence" has to say that it does so by
moving every scene toward 2-4.6 deg, not by making each scene as good as it can be.

| flight | G03 | frozen | arm a real | arm c mixed |
| --- | --- | --- | --- | --- |
| flight-01a-ellipse | 3.39 | 4.27 | 1.88 | 2.72 |
| flight-01p-ellipse | 1.88 | 3.27 | 1.40 | 2.75 |
| flight-04a-ellipse | 2.74 | 3.25 | 1.60 | 2.67 |
| flight-07a-lemniscate | 5.22 | 6.54 |  |  |
| flight-07p-lemniscate | 2.82 | 3.71 | 2.52 | 4.07 |
| flight-13a-trackRATM | 31.53 | 34.66 | 33.70 | 30.99 |
| flight-13p-trackRATM | 9.86 | 18.40 |  |  |
| flight-16p-trackRATM | 9.21 | 14.14 | 18.87 | 13.73 |

Frozen reader spread: **2.26** deg (urban_checker 0.17, forest_ridge 2.42, industrial 0.44)
- arm sim: spread **1.05** deg (urban_checker 0.17, forest_ridge 1.22, industrial 0.61), reduction 54 %
- arm mixed: spread **0.63** deg (urban_checker 1.84, forest_ridge 2.02, industrial 1.39), reduction 72 %

### Downstream

| reader | real FRR | removal | mirror | yaw | null |
| --- | --- | --- | --- | --- | --- |
| frozen | 5/72 | 70/72 | 69/72 | 9/72 | 5/72 |
| mixed | 6/72 | 71/72 | 69/72 | 11/72 | 6/72 |
| real | 7/72 | 68/72 | 70/72 | 11/72 | 7/72 |
| sim | 6/72 | 43/72 | 62/72 | 10/72 | 6/72 |

S3 was "the false-reject does not rise". **It rises for every arm**: 5/72 for the frozen reader,
7/72 for real-only, 6/72 for sim-only and 6/72 for mixed+DR. A one-to-two-clip change at n=72 is
descriptive (the flight-clustered 95 % intervals overlap: frozen 0.7-27.8 %, real 0-42 %, sim
0-36.7 %, mixed 0-34.6 %), but the criterion is written as a direction and no arm meets it.

The two arms differ in what the extra false-reject buys. **Mixed+DR keeps the tier's detection**
(removal 71/72 against the frozen 70/72, mirror 69/72 against 69/72) for its one extra false-reject
and a slightly higher yaw control (11/72 against 9/72). **Sim-only does not**: its removal detection
collapses to 43/72, and its mirror to 62/72, so it is both less specific and much less sensitive.
Excluding the two racing flights leaves the ordering unchanged (frozen 5/68, real 7/68, sim 6/68,
mixed 6/68), so the result is not driven by the flights where the edit-pipeline check is weakest.

The null edit equals the real clip by construction (both are the identity-warped frame) and has the
same rate; the yaw control rises from 9/72 to 11/72 for real-only and mixed, consistent with those
readers' larger per-frame view dependence, and is reported beside every detection number.



## What this supports

* **A clear negative answer on transfer.** Simulated-only and mixed+DR readers are 2.6-2.7x the
  real-only reader's held-out trimmed RMSE (19.87 and 20.44 against 7.51), and both are worse than
  the *frozen* reader's 14.75 on the hard ceiling-view flight. Adding simulated data under this DR
  recipe actively hurt.
* **The real-only fine-tune is the useful reader.** 7.51 deg pooled on the two held-out flights
  against G03's 8.29 (-9.4 %) and the frozen 14.75; it beats G03 on four of six development folds
  and on 13p, and loses on 07a and the two hard dev flights. The 15 % S1 bar is not met.
* **Scene-spread reduction is not the same as improvement.** Both sim arms meet the S2 spread
  criterion (-54 %, -72 %) while making the open scenes two to three times worse; the only arm
  that improves real held-out error is the one that never saw a simulated frame.
* **The scene effect measured in L01 is the mechanism to fix.** The forest is bias-dominated and
  the open families are noise-dominated; a training set that equalises them has to preserve the
  open families' accuracy, which neither sim arm did.

## What it does not show

* No held-out simulated scene: `field_settlement` and `coastal_cliff` are not on R2, so the
  pre-registered held-out-scene half is deferred, not failed. The simulated criterion here is the
  out-of-family dev proxy.
* No statement that simulated training cannot help: one budget (600 steps/epoch, 2 epochs), one
  seed per fold, one DR recipe and one architecture were tested. The negative result is about
  those choices.
* No per-clip calibration, fusion or confidence work: Track J already showed those do not move
  the decision, and this track kept the reader as the only variable.
* No claim about UZH: the UZH route was not re-run for this track.

## Caveats

* **Single seed per fold.** Real-only's 9.4 % vs G03 includes seed and budget differences; the
  per-flight spread (it loses 16p by 9.7 deg of trimmed RMSE) is larger than the pooled gain.
* **G03's baseline used seven training flights per fold; these final models use six** (the two
  held-out flights are excluded from every pool, as the round-1 protocol requires). The arms are
  therefore at a data disadvantage against G03, which makes the real-only gain more notable and
  the simulated arms' failure more damning, not less.
* **The finals used seed 1, not the pre-registered 101** (entrypoint default); recorded in
  `docs/decisions.md`. No selection was made on any outcome.
* **The mixed arm's objective did not converge**: training loss rose from 0.045 to 0.054 in epoch
  2, and the arm's 13p result is the worst measured. A different DR strength or schedule might
  behave differently; that is a hypothesis, not a result.
* **Downstream intervals are descriptive**: 8 flight clusters, 5,000 bootstrap draws; not 95 %
  confidence intervals. The edit-pipeline check is exceeded on the extreme ceiling-view flights
  (tilt +0.81 deg on 13p/16p, mirror -0.41 deg on 04a, yaw +0.37 deg on 16p), so the controls and
  the 13p/16p detection cells carry that uncertainty; excluding those two flights leaves every
  rate ordering unchanged.

## Provenance

* **Config**: `configs/l02_reader_prereg.yaml` (commit `c37b12b`), amendment B
  (`configs/l02_amendment_b.yaml`), budget amendment (`2bf2cb2`).
* **Training**: `remote/l_tilt_reader.py`, app `awmv-l-tilt-reader`, A10G, 21 models x 600 frames
  x 2 epochs at the measured rate; fold records in `awmv-l-data /l02/folds`, checkpoints in
  `/l02/models`. **Downstream**: `remote/l_edit_reader.py` (`awmv-l-edit-reader`).
* **Results**: `results/l02/` (summary, per-clip and per-flight tables, downstream).
* **Spend**: `docs/COST_LEDGER.md`.
