# M03 — Abstention end to end: four gate configurations against three verdict tiers

Track M, Stage 3. Written 2026-11-22. Branch `track-m-gates`. Pre-registration
`configs/m03_endtoend_prereg.yaml`; thresholds from `configs/m01_gate_a_frozen.yaml` and
`configs/m02_gate_b_frozen.yaml`. Code `experiments/m03_end_to_end.py`,
`experiments/m03_generators.py` (with Track K's tier code copied byte-identically). Results
`results/m03/`. CPU only.

## Decision

**Abstaining makes the verifier's answers better on the clips it keeps and worse on the
clips it flags:** on the gate-kept real development clips, adding Gate A raises the
scale-free coherence tier's tilt-removal detection from 54/69 (78 %) to 21/22 (95 %) while its
yaw-control false-flag rate rises from 16/69 (23 %) to 16/22 (73 %), and the fixed-calibration
tier's false-reject rises from 5/69 (7 %) to 9/22 (41 %) because the calibration population
collapses to 22 clips over two flights. **Gate A is therefore a coverage decision, not a
specificity fix, and the honest summary of this track is that the verifier can judge about a
third of realistic gate-kept clips and must not claim better specificity on that third.**

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
| R | G0 | 69 | 3/69 | 54/69 | 55/69 | 13/69 | 9/69 | 16/69 | 3/69 |
| R | GA | 22 | 1/22 | 21/22 | 22/22 | 15/22 | 4/22 | 16/22 | 1/22 |
| F0 | G0 | 69 | 5/69 | 56/69 | 55/69 | 37/69 | 52/69 | 8/69 | 5/69 |
| F0 | GA | 22 | 9/22 | 22/22 | 21/22 | 21/22 | 20/22 | 13/22 | 9/22 |
| C2 | G0 | 69 | 4/69 | 35/69 | 47/69 | 7/69 | 4/69 | 4/69 | 4/69 |
| C2 | GA | 22 | 4/22 | 22/22 | 21/22 | 19/22 | 7/22 | 10/22 | 4/22 |

Flight-clustered intervals (5,000 draws, seed 1) are in
`results/m03/real_dev_prelim.json`; with at most six clusters and, for `GA`, two, every interval
is **descriptive**, not a 95 % confidence interval, and most are very wide.

<!-- SIM TABLE -->

<!-- GENERATOR TABLE -->

## Reason codes

`results/m03/reason_codes_real.csv` carries one row per clip and condition with the four
decision flags, the three tier scores and the reason string. Composition on the real condition
of the 100 clips: *(filled from the run)*.

## What this supports

*(Filled after the simulated and generator tables.)*

## What this does not show

* Nothing about the simulated held-out scenes: 3 of 100 clips were on R2 when this ran, so no
  held-out number is quoted.
* Nothing about a *fourth* verdict tier: the handoff's three are used as published.
* The generator section reports the published verdicts among kept clips; it does not re-score a
  generator.
* No claim that abstention is useless. It is a coverage decision with a measurable cost, and
  the cost is stated.

## Caveats

* Two accepted flights at Gate A's threshold; the LOFO thresholds are estimated from one other
  flight. That is the mechanism behind the specificity loss and it is not fixable without more
  flights.
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
