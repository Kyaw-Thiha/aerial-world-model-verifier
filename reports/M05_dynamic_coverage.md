# M05 — Coverage restricted to dynamic clips

Track M follow-up. Written 2026-11-22. Branch `track-m-gates`. Pre-registration
`configs/m05_dynamic_coverage_prereg.yaml` (committed at `c6df271`, **before** any count was
taken). Code `experiments/m05_dynamic_coverage.py`. Results `results/m05/`. CPU only,
**zero GPU spend**. No threshold is refitted: Gate A (`λ = 3.82°` + the 80° near-vertical rule),
Gate B (`τ = 4.5°`) and S3 are used exactly as frozen.

## Decision

**A verifier's coverage is much worse where it matters most: among the 85 real clips whose path
demands at least 15° of tilt, Gate A judges 23 (27 %), Gate B 40 (47 %) and both 15 (18 %), and
among the 107 dynamic simulated clips Gate B keeps 29/38 urban and 26/31 industrial but only
2/38 forest — the scene bias is precisely what it abstains on, and it abstains on 12 of the 16
dynamic simulated tilt-removal clips as well.** The dominant abstention is not the path's signal
in the open scenes but disagreement in the dense ones, which is the failure this track was
built to expose rather than to hide.

## Plain-language summary

1. **Most clips are dynamic.** 85 of 100 real clips and 107 of 170 simulated clips reach 15° of
   required tilt, so the restriction is not a small corner of the population.
2. **On the real clips the dynamic restriction hurts Gate A.** Coverage among dynamic clips is
   27 % against 32 % over the whole gate-kept population, because the dynamic real clips are
   dominated by the racing flights, whose tilt runs above 80° where a lean vector stops being
   valid: `near_vertical` is a reason on 25 of the 85 dynamic clips, against 4 of the 69
   gate-kept ones.
3. **Gate B is the better gate on dynamic clips** (40/85 = 47 %), and on the S3-kept subset it
   keeps 40 of 61 (66 %).
4. **On the simulated set the family split is the finding.** Gate B keeps 29/38 urban, 26/31
   industrial and **2/38 forest**; Gate A keeps 13/38, 12/31 and 4/38. In forest the reader
   disagreement is 5.7° against 3.6° in the open families, so the gate is doing exactly what
   M02 measured — and the consequence is that the verifier cannot judge the generated or
   rendered forest at all.
5. **It costs detection-relevant coverage.** Among the 16 dynamic simulated tilt-removal clips,
   Gate B keeps 4 and Gate A keeps 4; among 16 mirrored-roll clips, 11 and 5. A gate that
   abstains on three quarters of the clips that carry the violation is honest but expensive, and
   that trade is the whole story of Track M.

## Question

Among clips whose path actually demands a large tilt — the ones a tilt-removal or mirrored-roll
fake would give itself away on — what fraction can each gate configuration still judge, and why
are the rest abstained on?

## Setup

**Dynamic clip**: peak required tilt (max over frames of `degrees(arccos(u_z))` for
`u = unit(a + g e3 + 0.295 v)`, schema v3) at or above **15°**. Chosen before scoring: 15° is
nearly four times the reader's per-frame spread (median 3.96°, Track J) and above the
fixed-calibration detection floor (about 9°, Track K).

**Configurations**: G0 (S3 alone), GA, GB, GAB. Each gate decides once per clip on the clip's
real video.

**Populations**: the 100 real TII clips (with the S3-kept 61 dynamic subset shown separately);
the 170 simulated development clips. Stratified by flight and track for the real clips, by
scene family, mount and defect family for the simulated ones.

## Results — real clips

85 dynamic of 100.

| configuration | all dynamic | among S3-kept dynamic (61) |
| --- | --- | --- |
| G0 (S3 alone) | 61/85 = 0.718 | 61/61 = 1.000 |
| GA | 23/85 = 0.271 | 23/61 = 0.377 |
| GB | 40/85 = 0.471 | 40/61 = 0.656 |
| GAB | 15/85 = 0.176 | 15/61 = 0.246 |

Reason codes among the 85 dynamic clips (a clip can carry several): `readers_disagree` 45,
`weak_signal` 34, `near_vertical` 25, `low_confidence_S3` 24. Among the S3-kept 61:
`weak_signal` 32, `readers_disagree` 21, `near_vertical` 6.

By track (kept / dynamic):

| track | dynamic clips | G0 | GA | GB | GAB |
| --- | --- | --- | --- | --- | --- |
| ellipse | 33 | 32 | 7 | 22 | 5 |
| lemniscate | 29 | 25 | 15 | 18 | 10 |
| trackRATM | 23 | 4 | 1 | 0 | 0 |

The racing track is where the dynamic clips are and where almost nothing survives: S3 itself
keeps 4 of 23, and Gate B keeps none.

## Results — simulated clips

107 dynamic of 170.

| configuration | kept / 107 | coverage |
| --- | --- | --- |
| G0 | 107 | 1.000 |
| GA | 29 | 0.271 |
| GB | 57 | 0.533 |
| GAB | 16 | 0.150 |

By scene family (kept / dynamic, with reason counts among the dynamic clips):

| family | dynamic | G0 | GA | GB | GAB | weak_signal | readers_disagree |
| --- | --- | --- | --- | --- | --- | --- | --- |
| urban_checker | 38 | 38 | 13 | 29 | 9 | 25 | 9 |
| industrial | 31 | 31 | 12 | 26 | 7 | 19 | 5 |
| forest_ridge | 38 | 38 | 4 | **2** | 0 | 34 | **36** |

By mount: 0° 3/11 (GA), 7/11 (GB), 3/11 (GAB); 15° 8/35, 19/35, 6/35; 30° 8/29, 16/29, 4/29;
45° 10/32, 15/32, 3/32.

By defect family (dynamic clips only, kept / total): feasible 14/53 (GA), 34/53 (GB), 7/53
(GAB); mirrored roll 5/16, 11/16, 5/16; tilt removed 4/16, 4/16, 0/16; tilt scaled 6/21, 8/21,
4/21; thrust limit 0/1, 0/1, 0/1.

## What this supports

* **A single coverage statement the paper can use**: of the clips where a large tilt is actually
  required, the verifier can judge 27 % (Gate A), 47 % (Gate B) or 18 % (both) on real flights,
  and 27 / 53 / 15 % on the simulated set.
* **The reason is not the path.** In the open simulated scenes the dominant abstention is
  `weak_signal`; in forest it is `readers_disagree` (36 of 38 dynamic forest clips), and that is
  the same measurement that made Gate B's AUROC 0.88 and 0.84.
* **Coverage and detection trade against each other in the same direction**: the gates abstain
  on most of the clips that carry a violation (12 of 16 dynamic simulated removals under Gate B),
  so a coverage number alone flatters the verifier and must be read beside detection.

## What this does not show

* Whether 15° is the right dynamic threshold. It was pre-registered as one value; a sweep would
  move the numbers (a lower threshold admits more weak-signal clips, a higher one admits more
  near-vertical ones).
* Anything about the real held-out flights: 3 gate-kept clips, not stratified here.
* Any new gate design. The reasons are reported, not acted on.

## Caveats

* The simulated reason codes carry no `low_confidence_S3`: the simulated clips have no S3 signal
  fitted by this track, so S3 is not among their reasons and their G0 row is trivially 1.000.
* "Peak required tilt" is a maximum, so a single transient frame above 15° makes a clip dynamic.
  That is deliberate — a one-step violation is still a violation — but it means the population is
  broader than "clips that sustain a large tilt".
* Counts are small in the by-defect cells (16, 16, 21) and no interval is quoted.

## Provenance

* **Code**: `experiments/m05_dynamic_coverage.py`.
* **Results**: `results/m05/{dynamic_coverage.json, real_dynamic.csv, sim_dynamic.csv}`.
* **Inputs**: `results/m03/reason_codes_real.csv` and `reason_codes_sim.csv` (the frozen gate
  decisions), `results/m01/signal_by_clip.csv` (peak required tilt for the real clips), the
  simulated tier table for the peak required tilt there.
