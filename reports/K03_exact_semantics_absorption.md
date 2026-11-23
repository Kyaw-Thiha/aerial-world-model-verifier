# K03 — Do J03's absorption and J02's floor survive exact drag semantics?

Track K, Part 3. Written 2026-11-22. Pre-registration
`configs/k03_exact_semantics_prereg.yaml` (commit `3110976`), written before any K03 number.
Code `experiments/k03_exact_semantics.py`. Results
`results/k03/20261122T194910Z/` (exact) and `results/k03/20261122T195100Z/` (J03's linear
construction, pipeline validation). CPU only, zero GPU, zero spend. No frozen config and no
published report was edited.

## Decision

**Under the exact required side, J03's absorption result survives at its natural thresholds
(F0 56→C2 35 tilt removals, 55→47 mirrored rolls, 52→4 x1.5, over 69 clips) but not at a
matched false-reject: there the per-clip offset+drag loses only the full-clip x1.5 gain
(52→7, 45 clips), loses 2-3 clips on removal and mirror, and is *better* than fixed
calibration on every partial edit — so "per-clip calibration absorbs the violations that
matter" holds only for multiplicative gain edits.** The fit-free coherence tier is no longer
the better relative test (it ties removal/mirror at 54/55, loses x0.5 17 against 41, flags
23/69 yaw controls against 10, and its local-step MDV is not lower); J02's floor table
survives after K01's relabelling (exact drag-free whole-clip MDV 6.95° against 6.88°; exact
k_d = 0.295 7.29°), while its 4.90° "drag-free" entry is the k_d = −0.295 arm and its 14.46°
anchor is drag-free, not k_d = 0.295 (the exact k_d = 0.295 anchor is 6.65° / 11.40°).

## Plain-language summary

1. **The pipeline is validated against J03 exactly.** Running the same code on J03's own
   linear construction reproduces its F0, C1 and C2 rows, its false-rejects and its
   absorption table digit for digit (F0 3/69, C1 4/69, C2 4/69; F0 removal 42/69, C2 11/69;
   absorption 31/19/9/44 clips = 44.9/27.5/13.0/63.8 pp). The only change in the exact run
   is the required side.
2. **The exact side raises fixed-calibration detection.** F0 clip tier: removal 56/69,
   mirror 55/69, x0.5 37/69, x1.5 52/69 (J03's drag-free F0: 42/52/14/48). The threshold is
   8.87° instead of 14.05°.
3. **Natural-threshold absorption shrinks but remains material.** F0−C2 on the clip tier:
   removal 21 clips (30.4 pp, J03 31/44.9), mirror 8 (11.6, J03 19/27.5), x0.5 30 (43.5,
   J03 9/13.0), x1.5 48 (69.6, J03 44/63.8). The window tier: removal 26, mirror 4, x0.5 12,
   x1.5 3.
4. **Matched false-reject reverses the removal/mirror conclusion.** With each arm's
   leave-one-flight-out cut tuned to F0's false-reject (5/69 clip, 6/69 window), C2 loses
   only 3 removal clips and 2 mirror clips (4.3 and 2.9 pp; window 2 and 0), while x1.5 is
   still absorbed by 45 clips (65.2 pp; window 16). On partial edits the fitted clip
   statistic detects *more* than F0 at every coverage (removal 25/50% 16/54 against 7/30;
   mirror 12/35 against 6/17; x1.5 20/41 against 6/10). The natural-threshold absorption is
   largely an operating-point effect: C2's natural false-reject is 4/69 against F0's 5/69,
   and relaxing it by one real clip recovers most of the removal/mirror detections.
5. **Coherence is a lateral, not a superior, relative test.** At the matched point it
   detects removal 54/69 (C2 53), mirror 55 (C2 53), x0.5 17 (C2 41), x1.5 11 (C2 7), with
   the yaw control at 23/69 (C2 10) and the null edit at 6 (C2 5). Its local-step MDV is
   18.5°/14.6° (25/50 % coverage) against 16.6°/19.1° for C1 and 15.7°/16.5° for C2 on the
   window tier and 6.85° for C2 on the 50 % clip tier. It wins on *localised* 25 % edits
   (47/69 removal against C2's 16) and loses on scale edits; neither statistic dominates.
6. **False-reject calibration survives.** Natural FR: F0 5/69, C1 4/69, C2 4/69, R 3/69,
   all within the 5/69 budget. The fitted k_d under exact semantics has median 0.35 (29 at
   0.40, 25 at 0.35, 6 at 0.295, 7 at 0.00), not the hard pin at 0.40 that J03 reported
   (64/72 at 0.40 on its linear construction).
7. **J02's floor table survives after relabelling, with one short-window qualification.**
   Exact drag-free whole-clip MDV 6.95° against J02's 6.88° (their "k_d = 0.295" row, which
   was drag-free); exact k_d = 0.295 7.29°. The 8-frame floor is 8.81° (drag-free) and 9.58°
   (k_d 0.295) against J02's 10.52°, i.e. 1.7° lower; the direction and the masking are
   stated. J02's 4.90° "drag-free" entry is k_d = −0.295. The anchor is 6.65°/11.40° at the
   true k_d = 0.295 and 9.31°/14.80° drag-free, the latter matching J02's 9.22°/14.46°
   drag-free-labelled row.

## Question

Which of J03's statements about absorption, coherence and false-reject, and which of J02's
floor numbers, survive when the required side is the exact flatness axis at a physical drag
coefficient rather than `lean_req + (k_d − 0.295) vel_req` with a drag-free `lean_req`?

## Setup

* **Population.** 100 TII real clips, 8 flights; the frozen gate keeps 72, the six dev
  flights give 69; `13p` and `07a` are held out and never scored. Edits and controls are the
  schema-v2 clip set; partial coverage 25 %/50 % by J03's centred composition.
* **Exact required side.** `u = normalise(a + g e3 + k_d v)` from native mocap (274.73 Hz,
  Savitzky-Golay 101 samples, poly 3), sampled at the clip's frame times. No
  `(k_d − 0.295)` term anywhere. The k_d grid is `[0, 0.05, …, 0.40]` recomputed exactly at
  each point; schema v3 carries the same semantics for k_d = 0.295.
* **J03 arms.** F0 (no nuisance, k_d = 0.295); C1 (per-clip free 2-vector Huber offset,
  δ = sin 3° in lean units); C2 (offset plus k_d on the physical grid, refitted on the
  composed partial edit); R (fit-free coherence, 8-frame window max of `1 − min r`). Tiers:
  clip median axis angle; 8-frame window max, hop 4. Thresholds: per-clip leave-one-flight-out
  95th percentile, median over clips.
* **Operating points.** (i) Natural: each arm at its own 5 % LOFO threshold, J03's
  convention. (ii) Pooled matched: a single empirical cut at F0's false-reject count (the
  pre-registered wording). (iii) LOFO-matched: the same LOFO procedure with alpha tuned per
  arm to F0's false-reject count. All three are reported; (iii) is the like-for-like
  comparison and agrees with (ii) qualitatively.
* **J02.** Synthetic reader error (σ 4.8°, bias 2°, τ 0.5 s, fast share 0.5, rng(1)), 120
  draws for the window table and anchor, 60 per grid cell; perfect-path and mocap-attitude
  arms; LOFO 95th thresholds; MDV at 50 % detection. Exact k_d = 0.295 and exact drag-free.
  Frames with required tilt > 85° are masked in the lean-space constructions (242 frames for
  k_d 0.295, 239 drag-free); the median/window tiers use thrust axes and include them.

## Results

### 1. Validation: J03's linear construction is reproduced exactly

| statistic (69 clips) | J03 published | this code, linear | this code, exact |
| --- | --- | --- | --- |
| F0 clip FR / threshold | 3/69 / 14.05 | **3/69 / 14.05** | 5/69 / 8.87 |
| C1 clip FR | 4/69 | **4/69** | 4/69 |
| C2 clip FR | 4/69 | **4/69** | 4/69 |
| F0 removal / mirror / x0.5 / x1.5 | 42 / 52 / 14 / 48 | **42 / 52 / 14 / 48** | 56 / 55 / 37 / 52 |
| C1 removal / mirror | 7 / 26 | **7 / 26** | 28 / 45 |
| C2 removal / mirror | 11 / 33 | **11 / 33** | 35 / 47 |
| C2 absorption (clip) | +31 / +19 / +9 / +44 | **+31 / +19 / +9 / +44** | +21 / +8 / +30 / +48 |
| C2 k_d at 0.40 | 65/72 | 64/72 | 29/72 |

The one-clip difference in the k_d histogram is the Huber threshold: J03's loss applies an
extra `deg2rad` to an already dimensionless δ (making it L1-like); K03 uses δ = sin 3° in
both the offset and the loss. The F0 row (no Huber) is exact, and C1/C2 detections and the
absorption table reproduce J03 exactly, so the pipeline is validated.

### 2. J03 absorption, natural thresholds (F0 − C2, clips and pp)

| edit / coverage | clip: F0 | clip: C2 | clip absorption | window: F0 | window: C2 | window absorption |
| --- | --- | --- | --- | --- | --- | --- |
| removal 25 / 50 / 100 % | 7 / 30 / 56 | 4 / 34 / 35 | +3 / −4 / **+21** (30.4 pp) | 30 / 39 / 53 | 31 / 32 / 27 | −1 / +7 / **+26** (37.7 pp) |
| mirror 25 / 50 / 100 % | 6 / 17 / 55 | 4 / 15 / 47 | +2 / +2 / **+8** (11.6 pp) | 33 / 47 / 55 | 30 / 42 / 51 | +3 / +5 / **+4** (5.8 pp) |
| x0.5 25 / 50 / 100 % | 7 / 8 / 37 | 4 / 12 / 7 | +3 / −4 / **+30** (43.5 pp) | 10 / 14 / 23 | 7 / 7 / 11 | +3 / +7 / **+12** (17.4 pp) |
| x1.5 25 / 50 / 100 % | 6 / 10 / 52 | 4 / 4 / 4 | +2 / +6 / **+48** (69.6 pp) | 6 / 6 / 9 | 6 / 5 / 6 | 0 / +1 / **+3** (4.3 pp) |
| yaw control | 8 | 4 | +4 | 6 | 5 | +1 |
| null edit | 5 | 4 | +1 | 6 | 5 | +1 |

At J03's own convention, the exact side gives: removal absorption **21 clips** (J03 31),
mirror **8** (19), x0.5 **30** (9), x1.5 **48** (44). The direction of the change is not
uniform: removal/mirror absorb less, x0.5 more, x1.5 about the same.

### 3. The same table at a matched false-reject

| edit / full | F0 clip | C2 clip | clip absorption | F0 window | C2 window | window absorption |
| --- | --- | --- | --- | --- | --- | --- |
| removal | 56/69 | 53/69 | **+3 (4.3 pp)** | 56/69 | 54/69 | **+2 (2.9 pp)** |
| mirror | 55/69 | 53/69 | **+2 (2.9 pp)** | 56/69 | 56/69 | **0** |
| x0.5 | 37/69 | 41/69 | **−4 (−5.8 pp)** | 44/69 | 31/69 | +13 (18.8 pp) |
| x1.5 | 52/69 | 7/69 | **+45 (65.2 pp)** | 26/69 | 10/69 | +16 (23.2 pp) |
| yaw control | 8/69 | 10/69 | −2 | 10/69 | 11/69 | −1 |
| null edit | 5/69 | 5/69 | 0 | 6/69 | 6/69 | 0 |

LOFO-matched alphas: F0 clip 0.050, C1 0.116, C2 0.090; F0 window 0.108, C1 0.110,
C2 0.112, R 0.088. The pooled empirical cut (pre-registered) gives the same picture
(removal 55→52, mirror 54→53, x0.5 48→38, x1.5 53→9 on the clip tier). On partial edits at
the matched point the fitted clip statistic is better than F0 at every coverage: removal
25/50 % 16/54 against 7/30, mirror 12/35 against 6/17, x1.5 20/41 against 6/10.

So the natural-threshold removal/mirror absorption is not a property of the nuisance at a
fixed operating point; it appears when C2's own 95 % threshold is used (natural FR 4/69
against F0's 5/69). Absorbing one more real clip moves C2's cut enough to recover them. The
absorption that survives at the matched point is the full-clip multiplicative gain (x1.5,
and x0.5 on the window tier).

### 4. Coherence versus the fitted statistics

| statistic | removal | mirror | x0.5 | x1.5 | yaw | null |
| --- | --- | --- | --- | --- | --- | --- |
| F0 clip (matched) | 56/69 | 55/69 | 37/69 | 52/69 | 8/69 | 5/69 |
| C2 clip (matched) | 53/69 | 53/69 | 41/69 | 7/69 | 10/69 | 5/69 |
| C1 clip (matched) | 56/69 | 53/69 | 49/69 | 16/69 | 18/69 | 5/69 |
| R window (matched) | 54/69 | 55/69 | 17/69 | 11/69 | 23/69 | 6/69 |

Local-step MDV (25 % / 50 % coverage, degrees): R 18.5 / 14.6; C1 window 16.6 / 19.1;
C2 window 15.7 / 16.5; C2 clip not reached at 25 %, 6.85 at 50 %; F0 window 17.4 / 15.8;
F0 clip not reached at 25 %, 10.6 at 50 %. Coherence is not the most sensitive relative
test: it beats the fitted clip tier on localised 25 % removal (47/69 against C2's 16/69)
but loses on x0.5 (17 against 41), on the 50 % MDV (14.6 against 6.85), and on the yaw
control (23/69 against 10/69).

Flight-clustered descriptive intervals (5,000 replicates, rng(1), six dev flights) for the
natural full-coverage rates: F0 clip removal 0.34-0.94, mirror 0.34-0.92, x1.5 0.30-0.88;
C2 clip removal 0.17-0.61, mirror 0.17-0.82, x1.5 0.01-0.29; R window removal 0.96-1.00,
mirror 1.00-1.00, x1.5 0.03-0.57. These are descriptive, not calibrated 95 % intervals.

### 5. J02 floors under exact semantics

Perfect-path synthetic-error MDV (degrees) by window; H0 median and LOFO threshold in
brackets for the whole clip:

| window | exact k_d = 0.295 | exact drag-free | J02 "k_d = 0.295" (was drag-free) |
| --- | --- | --- | --- |
| whole | **7.29** (6.49 / 9.40) | **6.95** (6.46 / 9.10) | 6.88 (6.44 / 9.05) |
| 24 | 7.76 | 7.42 | 7.33 |
| 16 | 8.50 | 7.97 | 8.17 |
| 12 | 9.10 | 8.30 | 9.02 |
| 8 | 9.58 | 8.81 | 10.52 |

The whole-clip and 24/16-frame cells match J02's drag-free-labelled row; the 12- and
especially 8-frame cells are lower (8.81 against 10.52) because the exact floor masks 239
frames with required tilt > 85° and an 8-frame window with two masked frames is dropped.
The exact k_d = 0.295 floor is 0.3-0.8° above drag-free, not the 2° gap J02 reported; that
gap was the negative-drag arm.

Parameter grid (whole clip, MDV): both modes move from about 4.7-4.8° (σ 3, bias 0) to
10.0-10.1° (σ 8, bias 0) and 10.0-10.7° (σ 8, bias 5); k_d = 0.295 and drag-free differ by
at most 0.7° in any cell, with no consistent sign. J02's σ/bias sensitivity is reproduced
(drag-free: 4.75 → 10.00 against J02 4.66 → 9.63).

Anchor (mocap attitude + synthetic error):

| mode | H0 median | LOFO threshold | J02 row |
| --- | --- | --- | --- |
| exact k_d = 0.295 | 6.65° | **11.40°** | (not computed by J02) |
| exact drag-free | 9.31° | **14.80°** | 9.22° / 14.46° (J02's drag-free row) |
| k_d = −0.295 | — | — | 16.38° / 31.11° (J02's "drag-free") |

The exact drag-free anchor reproduces J02's 14.46° row to 0.34°; the true k_d = 0.295 anchor
is 11.40°, which is the number J02's "matches k_d = 0.295" claim should have used.

**Measured-error replay: inconclusive.** The replay's H0 median is 4.01°, matching J02's
3.97°, but its LOFO threshold is 46.6° at the whole clip (84-88° in short windows), because
the 100-clip set includes the racing flights' 30-40° reader failures as their clip medians.
J02's 7.66° used J01's frame table, which is not an input to the clip schema; the replay
cannot be compared across the two and is reported as not re-derived, not as a contradiction.

### 6. Pre-registered decision rules

| rule | outcome |
| --- | --- |
| absorption survives (≥ 7 clips of full-coverage loss for removal, mirror or x1.5, at natural **or** matched) | **met literally**: natural removal 21, mirror 8, x1.5 48. Substantively it survives only for x1.5 at the matched point. |
| coherence survives (≥ 7 clips more than both fitted arms for full removal and mirror at matched FRR, and MDV ≥ 3° lower) | **not met**: removal +1 against C2, mirror +2, and its MDV is not lower. Partial only (localised 25 % edits). |
| false-reject calibrated (≤ 5/69) | **met**: F0 5, C1 4, C2 4, R 3. |
| J02 floor survives (exact k_d 0.295 whole MDV within 1° of 6.92; exact drag-free within 1° of 4.90) | **met for k_d 0.295** (7.29); **not met literally for drag-free** (6.95 against 4.90), but met after the K01 relabelling (against 6.88, difference 0.07). |

## What survives, and what does not

**Survives.** False-reject calibration of every arm; the fact that a per-clip offset+drag
absorbs the full-clip multiplicative gain (x1.5, and x0.5 on the window tier); J02's
whole-clip floor of about 7° in a physical mode; the ordering of floors by window length;
J02's σ/bias sensitivity; the exact drag-free anchor at 14.80°, which reproduces J02's
drag-free row.

**Does not survive at a matched false-reject.** "Per-clip calibration absorbs tilt removal
and mirrored roll": at 5/69 it loses 3 and 2 clips, and it detects *more* partial edits than
fixed calibration. "The coherence tier is the better relative test": it ties removal/mirror,
loses scale edits and the yaw control, and its MDV is not lower. "The drag nuisance pins to
its bound and is not measuring drag": the exact fit has median 0.35 with a spread from 0.00
to 0.40 (29 at 0.40), not 64/72 at the bound. "The floor changes by about 2° with the
calibration mode": the physical modes differ by at most 0.7° in the whole-clip grid, and the
large J02 mode gap was the k_d = −0.295 arm.

**Changes, with the direction stated.** J03's absorption for removal/mirror is smaller on
the exact side at natural thresholds (21/8 against 31/19 clips) and collapses at the matched
point (3/2); x0.5 absorption grows (30 against 9); x1.5 is unchanged (48 against 44). The
short-window floor is 1.7° lower than J02's at 8 frames.

## What this does not show

* **Not a statement that per-clip calibration is useless.** It removes a real 2-3° bias and
  is the most sensitive detector of localised 25 % removal in this table; it is simply not
  an absorber of removal/mirror at a matched operating point.
* **Not an interval-calibrated comparison.** Six dev flights make every clustered interval
  descriptive; the point estimates are all k/n.
* **Not a re-derivation of J02's measured-error replay.** The clip schema lacks J01's frame
  table; the replay's H0 matches but its threshold does not, and the reason is stated.
* **Not a claim about UZH or ANWM.** TII real clips only.

## Caveats

* J03's Huber loss contains an extra `deg2rad` on an already dimensionless δ; K03 uses the
  correctly scaled δ = sin 3°. F0 (no Huber) reproduces J03 exactly, and C1/C2 detections
  and absorption do too, so the difference is confined to the loss's linear/quadratic
  switch and one k_d count.
* The matched operating point is sensitive to one real clip: C2's natural FR is 4/69
  against F0's 5/69, and moving its LOFO alpha from 0.05 to 0.09 changes removal detection
  from 35 to 53. That sensitivity is the finding, not an inconvenience; both the natural
  and matched tables are shown.
* The exact floor masks 239-242 frames with required tilt > 85° in the lean constructions;
  this lowers the 8-frame floor. The median/window tiers use axes and keep those frames.
* The k_d grid stops at 0.40 (J03's allowance); the exact fit selects 0.35-0.40, so a wider
  grid could move it further and is not tested.
* The full-coverage intervals are descriptive with six clusters and are not corrected for
  multiplicity.

## Regenerate

```
PYTHONPATH=. python experiments/k03_exact_semantics.py --part all          # ~105 s
PYTHONPATH=. python experiments/k03_exact_semantics.py --part A --linear   # validation, ~30 s
```

Outputs `results/k03/<run>/{k03.json,A_table.json,A_matched.json,A_matched_lofo.json,
A_absorption.json,A_mdv.json,A_intervals.json,A_rows.csv,...}`. The exact run is
`20261122T194910Z`; the linear validation is `20261122T195100Z`.
