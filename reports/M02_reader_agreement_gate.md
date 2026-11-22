# M02 — Gate B: do two independent tilt readers disagree where the verdict reader is wrong?

Track M, Stage 2. Written 2026-11-22. Branch `track-m-gates`. Pre-registration
`configs/m02_gate_b_prereg.yaml`, amendment 1 `configs/m02_gate_b_amend1.yaml` (the real-TII
feed), both committed before any Gate B statistic was computed; frozen threshold
`configs/m02_gate_b_frozen.yaml`. Code `src/verdict/gate_b.py`, `experiments/m02_gate_b.py`,
`remote/m_readers.py`. Results `results/m02/`. GPU: Modal A10, about 2.7 GPU-hours, about USD 2.95
(`docs/COST_LEDGER.md`); every app is stopped.

## Decision

**Reader disagreement is a far better predictor of the verdict reader's error than the frozen
S3 confidence gate — clip-level AUROC 0.880 against 0.226 on the real development clips and
0.835 against 0.058 on the simulated ones — and it catches the scene bias that motivated this
track: the frozen gate keeps 11 of the 26 dense-forest clips with a median error of 9.08 deg,
and Gate B abstains on 9 of those 11.** As an abstention gate it controls the yaw-control risk
at its frozen threshold (bound 0.049, 40 of 68 clips accepted, 58.8 % coverage) but not the
coherence false-reject risk (best bound 0.101 at full coverage, so it removes no false rejects).

## Plain-language summary

1. **The measurement works once the feed is right.** On the simulated renders, where the camera
   is a genuine pinhole, GeoCalib-224 reproduces Track L's native-resolution reader (median
   error 1.42 deg against 1.27 deg) and Perspective Fields is competent at 3.36 deg. On the
   real TII fisheye the 224 centre crop with a 90 deg focal prior is *not* a pinhole image: the
   second reader's lean is a median 23.75 deg from truth there against the frozen reader's
   3.71 deg, so the real arm uses the project's 100 deg undistorted pinhole view instead
   (amendment 1).
2. **The ensemble helps the tail and hurts the typical clip.** Pooled trimmed RMSE on the 80
   feasible simulated clips: frozen reader 4.08 deg, GeoCalib-224 3.93, Perspective Fields
   3.34, mean of the two 3.11 — but the medians run the other way (1.27, 1.42, 3.36, 2.25).
   In the forest the mean reader is a large win (trimmed RMSE 5.44 against the frozen reader's
   8.24, and Perspective Fields alone is 3.54); in the open scenes it is a loss (urban 1.69
   against 0.87). Averaging two readers of unequal skill cannot improve the typical clip, and
   it does improve the scene where one reader fails systematically.
3. **Disagreement orders the clips the S3 gate mis-orders.** Real clips: AUROC 0.880
   [0.567, 0.938] against S3's 0.226. Simulated clips: 0.835 [0.806, 0.865] against 0.058.
   Per mount on the simulated set: 0.46 at 0 deg, 1.00 at 15 and 30 deg, 0.76 at 45 deg; S3 is
   at or below chance in every cell.
4. **The forest check is the result the track was built for.** The simulated forest clips that
   the frozen S3 gate keeps have a median reader error of 9.08 deg against 7.72 deg for the ones
   it drops (reproducing Track L's direction), and their median reader disagreement is 5.69 deg
   against 3.57 deg in the open families. Gate B abstains on 9 of the 11 kept forest clips and
   on 22 of the 26 forest clips overall.
5. **What the gate can and cannot guarantee.** Conformal risk control reaches the 5 % yaw-control
   bound at tau = 4.5 deg (0.049, 40 of 68 accepted) but the coherence false-reject risk is
   unreachable at any threshold: the best bound is 0.101, at full coverage. Gate B therefore
   protects the *yaw control* — the failure M01 could not fix — and does nothing for
   false-rejects.

## Plain-language summary

*(Filled from `results/m02`.)*

## Question

Does disagreement between two independent tilt readers, run on the same pixels, predict the
verdict reader's error — and can it be turned into an abstention gate that catches the scene
bias the frozen S3 confidence gate keeps?

## Setup

**Readers.** `R_frozen` is the verdict reader (GeoCalib at `1dedb29`, `simple_divisional` on the
raw fisheye) and is read from the stored schema clips, never re-run. `R_pf` is Perspective
Fields (`Paramnet-360Cities-edina-centered`, commit `d54be73`, Adobe Research License,
noncommercial research use only; weights downloaded at run time and never committed or
re-hosted). `R_gc224` is GeoCalib with a pinhole model. `R_pf` and `R_gc224` form the
disagreement pair because they are different model classes reading identical pixels.

**Feeds.** Simulated clips (640x360, 90 deg horizontal, a genuine pinhole render): square centre
crop to 224 px, focal prior 199.1 px. Generated clips (224x224, 90 deg): 224 px, focal prior
112 px, H03's feed. Real TII fisheye: **the project's 100 deg virtual pinhole view**
(`src.perception.undistort`, the released OpenCV calibration, 640x480, f = 268.5 px, 46-47 % of
the output pixels valid), applied after the condition's pixel edit.

**Why the real feed changed (amendment 1, with the measurement).** The pre-registration used the
224 crop with f = 112 for every source. On the first pass (tag `v1`, 5,542 frame-reads over 40
clips x 6 conditions) that feed gave a median lean error of **24.75 deg for `R_gc224`** and
13.50 deg for `R_pf`, against **6.85 deg for `R_frozen`**, and a median reader disagreement of
21.21 deg; the roll-only disagreement is 1.89 deg at a 40 deg mount and 13.55 deg at 30 deg.
A 224 crop of a ~150 deg fisheye is not a 90 deg pinhole image, so the focal prior is wrong by a
large factor and the pitch component of any pinhole reader is invalid on it. The roll survives
(that is why H03's 1.30 deg roll check on 30 real fisheye controls passed), but a *lean*
difference needs the pitch. The `v1` rows are kept in `results/m02/crop_feed_invalidity.json`
as the measurement and are not used for any gate statistic.

**Reader sign conventions are fitted, not assumed.** A probe rotates every image by -10, 0 and
+10 deg about its centre (a pure camera roll) and reads both estimators. GeoCalib's roll moves
by -19.3 deg for a +20 deg image rotation; Perspective Fields' `pred_roll` moves by +18.7 deg, so
it is negated (H03's convention). PF's pitch sign is then chosen by minimising the median
angular disagreement with `R_gc224` on the probe frames: the four hypotheses give 20.95, 20.46,
5.77 and **4.19** deg, and the last (roll sign -1, pitch sign -1) is used. Fitted values are in
`results/m02/probe.json`; every fit is a global convention, never a per-clip adjustment.

**Statistic.** Per frame, the angle between the two leans:
`180/pi * asin(min(1, |l_pf - l_gc224|))`, each lean built through the clip's own mount. Per
clip: the median (primary) and the p90; the 8-frame window median with the maximum over windows
is reported as the tier-style variant.

**Gate.** Abstain when the clip-median disagreement exceeds `tau_B`, chosen by conformal risk
control on the real development clips (accept iff `disagreement <= tau`, accepted risk bounded
by `(k+1)/(n+1)`, two risks: coherence false-reject and yaw-control flag rate, each at most
0.05). Where no threshold reaches a target, that is the reported result.

**Populations.** Development: the six real dev flights (gate-kept by S3, 69 clips), the 170
simulated development clips on R2 (80 feasible + 90 defect; frames read every 4th at 30 Hz), the
234 H04 SolarWM clips (every 2nd frame) and 80 H01 ANWM clips. Evaluation, read once at the end:
the real held-out flights 13p and 07a, the simulated held-out clips if rendered (3 of 100 present
when this was written), and the generator clips.

## Results

### The feed (probe and the crop failure)

| probe | GeoCalib roll span | PF roll span | expected |
| --- | --- | --- | --- |
| +20 deg image rotation | -19.3 deg | +18.7 deg | 20 deg |

| feed, real TII, real condition | `R_frozen` | `R_gc224` | `R_pf` | mean | median disagreement |
| --- | --- | --- | --- | --- | --- |
| 224 crop, f = 112 (tag `v1`, 40 clips) | 3.71 deg | 23.75 deg | 13.50 deg | — | 21.21 deg |
| 100 deg pinhole view, f = 268.5 (tag `v3`, 100 clips) | 4.00 deg | 5.29 deg | 7.25 deg | 5.18 deg | 6.47 deg |

### Reader error and the ensemble (80 feasible simulated dev clips, trimmed RMSE / median, deg)

| family | frozen (L01) | GeoCalib-224 | Perspective Fields | mean of the two |
| --- | --- | --- | --- | --- |
| forest_ridge | 8.24 / 8.55 | 7.76 / 8.08 | **3.54 / 3.39** | 5.44 / 5.71 |
| industrial | **1.32 / 1.22** | 1.23 / 1.29 | 3.27 / 3.28 | 1.76 / 1.87 |
| urban_checker | **0.87 / 0.93** | 0.96 / 0.81 | 3.20 / 3.19 | 1.69 / 1.86 |
| all | 4.08 / 1.27 | 3.93 / 1.42 | 3.34 / 3.36 | **3.11** / 2.25 |

### Does disagreement predict the reader's error (AUROC for clip or frame error > 5 deg)

| population | n (positives) | disagreement | S3 confidence | combined |
| --- | --- | --- | --- | --- |
| real clips, dev | 69 (9) | **0.880** [0.567, 0.938] | 0.226 | 0.684 |
| real frames, dev | 1,971 (661) | 0.759 [0.646, 0.802] | 0.724 | **0.773** |
| simulated clips, feasible | 80 (26) | **0.835** [0.806, 0.865] | 0.058 | 0.542 |
| simulated, mount 0 / 15 / 30 / 45 | 20 each | 0.46 / 1.00 / 1.00 / 0.76 | 0.00 / 0.00 / 0.17 / 0.17 | — |

Intervals are flight- (or scene-family-) clustered bootstraps, 2,000 draws, `default_rng(1)`,
and **descriptive**: six clusters for the real clips, three families for the simulated ones.
On the simulated set the "error > 5 deg" label is almost collinear with the scene family (26 of
26 forest clips are positive, 0 of 27 in each open family), so that AUROC largely measures
"can the signal identify forest". The real-clip AUROC is the better test and it points the same
way.

### The forest clips the frozen gate keeps

| quantity | value |
| --- | --- |
| forest feasible clips | 26 |
| kept by the frozen S3 gate | 11 |
| median reader error, kept / dropped | 9.08 deg / 7.72 deg |
| median disagreement, kept forest / all forest / open families | 5.69 / 5.71 / 3.57 deg |
| Gate B abstains on kept forest | **9 of 11** |
| Gate B abstains on all forest | 22 of 26 |

### Gate B as selective prediction (real dev clips, 68 gate-kept)

| threshold | accepted | coverage | coherence false-reject | yaw control | null edit |
| --- | --- | --- | --- | --- | --- |
| 4.5 deg (frozen) | 40 | 0.588 | 2/40 | 0/40 | 0/40 |

Conformal risk control reaches the yaw-control target at 4.5 deg (bound 0.049) and cannot reach
the false-reject target: the best any threshold achieves is 0.101, at full coverage. Per the
pre-registration the threshold of the risk that reached is frozen and the other risk is
reported beside every number.

## What this supports

* **A second reader is a usable reliability signal where the frozen confidence gate is not**,
  with a large margin on both populations (0.88 against 0.23 on real clips, 0.84 against 0.06
  on simulated).
* **The scene-bias failure M01 and L01 identified is detectable**: 9 of the 11 S3-kept forest
  clips are abstained on, and the forest/open disagreement gap (5.7 against 3.6 deg) is the
  mechanism.
* **A mean reader is worth reporting as its own instrument**: it is the best pooled trimmed
  RMSE on the simulated set (3.11 deg) and the best in the forest (5.44), while being worse
  than the frozen reader in the open scenes. It is *not* proposed as a replacement reader.
* **A yaw-control guarantee**, which is the one thing Gate A could not deliver (M01): the gate
  reaches a 0.049 bound at 58.8 % coverage.

## What this does not show

* Nothing about the real held-out flights beyond three gate-kept clips.
* Two readers can share a bias; disagreement cannot see an error both make. If the forest bias
  is a property of pinhole readers on dense canopy, Gate B will miss it, and that is reported
  rather than worked around.
* The simulated renders are procedural: the forest/open contrast is a statement about *these*
  renders, not about real forests.
* The real-TII arm reads a 100 deg window of a ~150 deg fisheye through an undistortion that
  uses only 46 % of the output pixels; the outer periphery is not modelled and is not read.

## Caveats

* Three scene families and 6-7 clips per cell: every family-level interval is descriptive.
* The generator clips carry no ground-truth attitude; Gate B is evaluated on them only through
  the published verdicts of the tracks that generated them.
* Frame subsampling (every 4th simulated frame, every 2nd generator frame) changes precision,
  not the quantity: the pre-registered statistic is a clip median.

## Provenance

* **Code**: `remote/m_readers.py` (Modal A10, `awmv-m-readers`, volume `awmv-m-data`),
  `src/verdict/gate_b.py`, `experiments/m02_gate_b.py`, `experiments/m02_upload_inputs.py`.
* **Inputs**: TII camera zips and A2 edit specs (`awmv-a-data`, read-only); R2 `track-i/`,
  `track-l/l01/per_frame/`, `track-h3/`, `track-h/`.
* **Results**: `results/m02/{probe.json, crop_feed_invalidity.json, real_by_clip.csv,
  real_frames.csv, sim_by_clip.csv, auroc.json, gate_b_selection.json, gate_b_at_tau.json,
  forest.json, ensemble.json}`.
