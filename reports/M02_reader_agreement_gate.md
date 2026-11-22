# M02 — Gate B: do two independent tilt readers disagree where the verdict reader is wrong?

Track M, Stage 2. Written 2026-11-22. Branch `track-m-gates`. Pre-registration
`configs/m02_gate_b_prereg.yaml`, amendment 1 `configs/m02_gate_b_amend1.yaml` (the real-TII
feed), both committed before any Gate B statistic was computed; frozen threshold
`configs/m02_gate_b_frozen.yaml`. Code `src/verdict/gate_b.py`, `experiments/m02_gate_b.py`,
`remote/m_readers.py`. Results `results/m02/`. GPU: Modal A10, ~USD (see the ledger).

## Decision

*(Filled from `results/m02`; see the summary tables below.)*

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

| feed | `R_frozen` lean error | `R_gc224` | `R_pf` | median disagreement |
| --- | --- | --- | --- | --- |
| 224 crop, f = 112, real TII | 6.85 deg | 24.75 deg | 13.50 deg | 21.21 deg |

*(Results tables are filled from `results/m02/auroc.json`, `gate_b_selection.json`,
`gate_b_at_tau.json`, `forest.json`, `ensemble.json` and `sim_by_clip.csv`.)*

## What this supports

*(Filled from the results.)*

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
