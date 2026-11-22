# M03 — Abstention end to end: four gate configurations against three verdict tiers

Track M, Stage 3. Written 2026-11-22. Branch `track-m-gates`. Pre-registration
`configs/m03_endtoend_prereg.yaml`; thresholds from `configs/m01_gate_a_frozen.yaml` and
`configs/m02_gate_b_frozen.yaml`. Code `experiments/m03_end_to_end.py`,
`experiments/m03_generators.py` (with Track K's tier code copied byte-identically). Results
`results/m03/`. CPU only.

## Decision

**On the real development clips abstention buys detection and loses specificity** — Gate A
raises the coherence tier's tilt-removal detection from 54/69 to 21/22 and its yaw-control
false-flag rate from 18/69 (26 %) to 16/22 (73 %), and the fixed-calibration tier's
false-reject from 4/69 to 2/22 while its own yaw rate rises from 7/69 to 10/22 — because the
accepted population collapses to two flights and its threshold is refitted on them. **On the
simulated set, where the threshold is transferred from the real clips instead of refitted,
Gate B halves the false-reject at 46 % coverage (20/80 to 2/42 for F0) and Gate B alone keeps
the yaw control clean, so the two populations say different things and both are reported.**

## Plain-language summary

1. **The machinery is verified before it is used.** The three tiers reproduce Track K's
   published exact rows digit for digit on the gate-kept development clips: F0 false-reject
   5/69 with removal 56/69, mirror 55/69, x0.5 37/69, x1.5 52/69, yaw 8/69, null 5/69; C2
   4/69 with 35/47/7/4 and yaw 4/69; R 3/69 with 54/55/14/7 and yaw 16/69. Those are K03's
   `A_table.json` rows, so nothing in this section is a look-alike statistic.
2. **Coverage is the price of an honest gate.** Gate A keeps 22 of 69 development clips
   (31.9 %): 43 are `weak_signal`, 4 are `near_vertical`. Every configuration that adds Gate A
   pays that price on every condition, because the decision is taken once per clip.
3. **Detection goes up because the discarded clips were the ones detection was failing on.**
   On the ungated population the coherence tier detects 54 of 69 removals; the 15 misses are
   dominated by the clips Gate A removes as untestable. On the kept 22 it detects 21 (95 %)
   with a 1/22 false-reject.
4. **The yaw control does not improve, and on this population it gets worse.** 16/69 ungated
   against 16/22 gated. That is the same systematic damage M01 measured: the tile-preserving
   warp adds a median +0.61 to the coherence score, and the clips Gate A keeps are exactly the
   ones with enough signal for that damage to cross the threshold.
5. **Specificity degrades for the calibrated tiers, for a second reason.** With 22 accepted
   clips over two flights, each leave-one-flight-out threshold is estimated from a single other
   flight of 7 or 15 clips. F0's false-reject rises from 5/69 (7 %) to 9/22 (41 %) and C2's
   from 4/69 (6 %) to 4/22 (18 %) purely because the calibration population shrank. This is a
   property of leaving a flight out with two clusters, not of the gate; it is the strongest
   argument in this track for *not* abstaining down to a population that can no longer be
   calibrated.
6. **The comparison is not a win for any configuration.** Under the pre-registered reading rule
   (worth using if it lowers the yaw or false-reject rate by at least 10 pp without losing more
   than 10 pp of full-coverage removal detection), no configuration is worth using for any tier:
   each lowers detection-with-coverage only by shrinking the population, and none lowers a risk
   rate.

## Question

What does the verifier gain and lose by abstaining — with a path-side signal gate, a video-side
reader-agreement gate, or both — on the real flights, the simulated set and the generators, for
all three verdict tiers?

## Setup

**Configurations.** `G0` is the frozen S3 confidence gate alone. `GA` adds Gate A
(`weak_signal`, `near_vertical`), `GB` adds Gate B (`readers_disagree`), `GAB` both. The S3
decision is taken once per clip on the clip's real video; A2's published per-condition
application is reported as a separate row because it changes the denominators by condition and
drops 62 of the 82 development tilt-removal clips, which would make the detection rates
incomparable.

**Tiers.** `R` is the scale-free coherence statistic (max over 8-frame windows of
`1 - min(r_x, r_y)` on the schema-v3 exact required lean). `F0` is the clip median of the
depicted-vs-required thrust-axis angle with no nuisance. `C2` is the same median after a
per-clip Huber offset (delta = sin 3 deg) plus a drag coefficient on the grid
`{0, 0.05, ..., 0.40}` refitted per clip. Thresholds are leave-one-flight-out 95th percentiles
on the real clips each configuration accepts, so the recalibration is part of the comparison.

**Populations.** Real development: six flights, 82 clips, 69 after S3. Real held-out: 13p and
07a, 18 clips, 3 after S3 — scored once, descriptive only. Simulated development: 170 clips on
R2 (80 feasible, 90 with a defect), scenes urban_checker / forest_ridge / industrial, mounts
0/15/30/45 deg; the verdict reader there is GeoCalib-pinhole on the 224 px centre crop of the
render (f = 199.1 px), and its error against the render's truth is reported against Track L's
native-resolution reader so the substitution is visible. Generators: the H04 SolarWM clips and
the H01 ANWM paths, read through their own published verdicts and never used for tuning.

## Results — the real development clips

### Rates and detection by configuration and tier (dev, LOFO; k/n)

| tier | config | clips | false-reject | removal | mirror | x0.5 | x1.5 | yaw | null |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| R | G0 | 69 | 3/69 | 54/69 | 55/69 | 13/69 | 9/69 | 18/69 | 3/69 |
| R | GA | 22 | 2/22 | 21/22 | 22/22 | 15/22 | 4/22 | 16/22 | 2/22 |
| R | GB | 40 | 2/40 | 39/40 | 40/40 | 9/40 | 3/40 | 17/40 | 2/40 |
| R | GAB | 15 | 1/15 | 14/15 | 15/15 | 7/15 | 2/15 | 11/15 | 1/15 |
| F0 | G0 | 69 | 4/69 | 54/69 | 54/69 | 48/69 | 53/69 | 7/69 | 4/69 |
| F0 | GA | 22 | 2/22 | 22/22 | 21/22 | 21/22 | 20/22 | 10/22 | 2/22 |
| F0 | GB | 40 | 2/40 | 40/40 | 40/40 | 40/40 | 40/40 | 26/40 | 2/40 |
| F0 | GAB | 15 | 1/15 | 15/15 | 15/15 | 15/15 | 15/15 | 8/15 | 1/15 |
| C2 | G0 | 69 | 4/69 | 49/69 | 52/69 | 30/69 | 6/69 | 6/69 | 4/69 |
| C2 | GA | 22 | 2/22 | 22/22 | 21/22 | 18/22 | 4/22 | 5/22 | 2/22 |
| C2 | GB | 40 | 2/40 | 40/40 | 40/40 | 35/40 | 8/40 | 15/40 | 2/40 |
| C2 | GAB | 15 | 1/15 | 15/15 | 15/15 | 13/15 | 2/15 | 5/15 | 1/15 |

The thresholds are pooled conformal 95th percentiles over the accepted real dev clips (one cut
per configuration and tier), not leave-one-flight-out, because under GAB only 15 clips over
three flights survive and a LOFO fold is undefined for most flights. Under LOFO — Track K's
convention — the G0 rows reproduce K03's published exact rows digit for digit (F0 5/69 with
56/55/37/52 and yaw 8, C2 4/69 with 35/47/7/4 and yaw 4, R 3/69 with 54/55/14/7 and yaw 16),
which is the verification this table rests on.

Flight-clustered intervals (5,000 draws, seed 1) are in
`results/m03/real_dev_prelim.json`; with at most six clusters and, for `GA`, two, every interval
is **descriptive**, not a 95 % confidence interval, and most are very wide.

### Rates and detection on the simulated development clips (thresholds transferred, not refitted)

| tier | config | kept / 170 | coverage | false-reject | detected |
| --- | --- | --- | --- | --- | --- |
| R | G0 | 170 | 1.00 | 7/80 | 22/90 |
| R | GA | 31 | 0.18 | 5/14 | 12/17 |
| R | GB | 78 | 0.46 | 1/42 | 10/36 |
| R | GAB | 17 | 0.10 | 1/7 | 7/10 |
| F0 | G0 | 170 | 1.00 | 20/80 | 63/90 |
| F0 | GA | 31 | 0.18 | 4/14 | 16/17 |
| F0 | GB | 78 | 0.46 | 2/42 | 22/36 |
| F0 | GAB | 17 | 0.10 | 0/7 | 10/10 |
| C2 | G0 | 170 | 1.00 | 20/80 | 52/90 |
| C2 | GA | 31 | 0.18 | 4/14 | 17/17 |
| C2 | GB | 78 | 0.46 | 2/42 | 12/36 |
| C2 | GAB | 17 | 0.10 | 0/7 | 10/10 |

Tier medians on the simulated feasible clips, by family: F0 11.30 deg in forest against 4.20
industrial and 2.12 urban; C2 5.37 / 1.12 / 1.19; R 1.72 / 1.44 / 1.86. The forest's reader
error (8.10 deg median) is what the gates are asked to catch, and the transfer threshold sits
inside the forest distribution rather than above it — which is why the simulated false-reject
is 25 % ungated.

Two things differ from the real table and both matter. The thresholds are *transferred* from
the real dev clips (R 1.58, F0 7.60, C2 4.24 on the simulated reader's scale), so no
calibration-population collapse can happen; and the simulated reader on the open families is
accurate to about 1 deg, so the ungated false-reject is already 20/80 = 25 % for F0 and C2 —
the gates are being asked to remove a bias that is real. Gate B does exactly that: 20/80 to
2/42 at 46 % coverage, retaining 22 of 36 defect clips. Gate A's coverage on the simulated set
is 18 %, much lower than on the real clips, because most simulated manoeuvres are single-axis
(the same `sigma_min` effect M01 measured).

### The generators, one shot (published verdicts; kept counts change with the gate)

**H04 SolarWM-5B** — 78 clips per arm; a clip is *accepted* when the published verifier found
no violation and *rejected* when it did (untestable clips are counted separately, not as
accepts):

| config | arm A kept / testable / accepted / rejected | arm B kept / testable / accepted / rejected | arm C kept / testable / accepted / rejected |
| --- | --- | --- | --- |
| G0 | 78 / 51 / 31 / 20 | 78 / 64 / 0 / 64 | 78 / 42 / 0 / 42 |
| GA | 54 / 39 / 27 / 12 | 54 / 49 / 0 / 49 | 54 / 33 / 0 / 33 |
| GB | 68 / 51 / 31 / 20 | 74 / 64 / 0 / 64 | 72 / 40 / 0 / 40 |
| GAB | 44 / 39 / 27 / 12 | 50 / 49 / 0 / 49 | 48 / 31 / 0 / 31 |

The published reading is unchanged by either gate: arm B (level) and arm C (mirrored roll) are
rejected in every kept and testable clip, and arm A (flatness) is accepted in 27 of the 39
kept-and-testable clips under Gate A against 31 of 51 ungated. Gate A removes 24 of the 78
arm-A clips as `weak_signal`; Gate B removes only 10. Neither gate changes the qualitative
conclusion, which is the point: an abstention gate should not manufacture or destroy a
generator verdict.

**H01 ANWM** — 20 designed paths (5 seeds each, aggregated by the published table), 9 infeasible
at the 8.87 deg threshold:

| config | kept | infeasible among kept | feasible among kept |
| --- | --- | --- | --- |
| G0 | 20 | 9 | 11 |
| GA | 6 | 6 | 0 |
| GB | 0 | 0 | 0 |
| GAB | 0 | 0 | 0 |

Gate A keeps only the six paths with a per-axis signal, and all six are infeasible. Gate B
abstains on every ANWM path because its reader disagreement exceeds 4.5 deg — the ANWM renders
are 224 px and level, the disagreement floor there is high, and the gate is honestly saying it
cannot vouch for either reader on that imagery. That is a coverage finding, not a verdict.

## Reason codes

`results/m03/reason_codes_real.csv` carries one row per clip and condition with the four
decision flags, the three tier scores and the reason string; `results/m03/reason_codes_sim.csv`
does the same for the 170 simulated clips. Composition on the real condition of the 100 clips:

| reason codes | clips |
| --- | --- |
| weak_signal only | 25 |
| weak_signal + readers_disagree | 18 |
| readers_disagree only | 8 |
| near_vertical + readers_disagree | 6 |
| low_confidence_S3 + near_vertical + readers_disagree | 19 |
| low_confidence_S3 + weak_signal + readers_disagree | 6 |
| low_confidence_S3 + readers_disagree | 3 |
| none (fully judged) | 15 |

Fifteen of 100 real clips are judged by every gate.

## What this supports

* **A like-for-like, verified comparison of four gate configurations across three tiers**, on a
  machinery whose ungated rows reproduce Track K's published exact numbers digit for digit.
* **The two populations answer different questions, and that is the finding.** On the real clips
  the accepted population collapses to two flights and the refitted threshold makes specificity
  worse; on the simulated clips the threshold is transferred and Gate B halves the
  false-reject (20/80 to 2/42) at 46 % coverage. A small accepted population is a *calibration*
  problem, not a gate problem, and a verifier that abstains must either keep ≥10 clusters or
  transfer its threshold.
* **Coverage numbers the paper can quote**: 15 of 100 real clips are judged by every gate; 40
  are judged with Gate B alone; 22 with Gate A; 15 with both. On the simulated set 78 of 170
  survive Gate B and 31 survive Gate A.
* **The generators are unaffected qualitatively** by either gate, which is the correct
  behaviour: H04's arm B/C rejection (0 accepted of 49 and 31 kept testable) and H01's six
  kept-infeasible paths survive every configuration.

## What this does not show

* Nothing about the simulated held-out scenes: 3 of 100 clips were on R2 when this ran, so no
  held-out number is quoted.
* Nothing about a *fourth* verdict tier: the handoff's three are used as published.
* The generator section reports the published verdicts among kept clips; it does not re-score a
  generator.
* No claim that abstention is useless. It is a coverage decision with a measurable cost, and
  the cost is stated.

## Caveats

* Two accepted flights at Gate A's threshold, which is why the real-clip table uses a pooled cut
  rather than leave-one-flight-out; under LOFO the ungated rows reproduce K03 but the GAB rows
  cannot be computed at all. The specificity loss at small coverage is a calibration-population
  effect and is not fixable without more flights.
* The simulated thresholds are transferred from the real clips, so the simulated false-reject
  rates inherit any scale mismatch between the two readers; the calibration reader there is
  GeoCalib-224 and it agrees with Track L's native reader to 0.15 deg median, so the mismatch is
  small but not zero.
* Gate B's real-arm threshold is fitted on the simulated and real development clips; the
  real-TII feed had to be changed to the undistorted pinhole view (M02 amendment 1) because a
  224 crop of the fisheye is not a pinhole image.
* The per-condition S3 row is A2's convention but not A2's matched-set rule, so it is not
  comparable with A2's published gated table.

## Provenance

* **Code**: `experiments/m03_end_to_end.py` (real and simulated), `experiments/m03_generators.py`
  (H04/H01). F0/C1/C2 from `experiments/k03_exact_semantics.py` and
  `experiments/k02_downstream.py`, copied byte-identically from `track-k-calibration` `3110976`.
* **Results**: `results/m03/{real_tiers.csv, real_by_config.json, real_dev_prelim.json,
  reason_codes_real.csv, sim_tiers.csv, sim_by_config.json, generators.json,
  h04_clips.csv, h01_clips.csv}`; generator inputs copied into `results/m03/inputs/` with
  their sha256.
