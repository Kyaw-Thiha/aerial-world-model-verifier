# V0 — Commanded vertical motion: feasibility

**Decision.** The check stops at feasibility: the pre-registered stop rule fires on condition A,
because the EyeHeight-style height-ratio estimator's error on real UZH-FPV outdoor footage
(median |log ratio error| 0.495 over 97 windows from three sequences) is 4-8 times the effect of
the smallest labelled vertical defect it would have to catch (median 0.079 over the 28
simulated climb clips), on 28 of 28 such clips. On the same estimator the rendered clips behave
as specified (median |log ratio error| SIM_MED over SIM_N windows), so the failure is the
geometry of real low-altitude footage, not the statistic; no defect render, no real
false-reject table and no generated-clip arm is reported.

Run directory: `results/v/v00/` (summary.json, table.md, per-unit window tables,
`fig_depicted_vs_truth__outdoor_forward_1.png`). Code at the commit that adds this report;
pre-registration `configs/v00_feasibility_prereg.yaml` (`2e948fa`) plus amendments 1-3.

## Question

EyeHeight (arXiv 2611.21782) shows that a scale-free multi-view reconstruction can measure a
world model's camera height above the floor, and that six models drift it toward human eye level
on constant-height indoor paths. Aerial world models take altitude as an action. The question
here is whether the same measurement can decide that a commanded climb or descent was or was not
depicted with the commanded ratio — and first, whether it can do so accurately enough that real,
feasible drone flights are not rejected.

## Setup

Populations (frozen before any output; amendment 1-2):

* **Simulated pilot**: every development clip of the frozen Track I set whose manoeuvre is
  `climb_turn` (28 of 200 — the only manoeuvre with commanded vertical motion; peak |dz|
  9.3-48.4 m, median 21.7 m, from a 12 m start; 8 of the 28 are feasible clips and 20 carry an
  attitude defect, which does not change the camera's altitude) plus two constant-altitude
  controls (`i02_dev_000` hover, `i02_dev_001` ellipse).
* **Real pilot**: UZH-FPV's three outdoor forward sequences (29.7/29.7/14.7 s at 15.07 Hz, Leica
  truth): 3 s |dz| of 0.80/1.60/0.80 m (medians) and up to 4 m, vertical speed up to 3.8 m/s,
  altitude range 4.3-4.8 m. TII is **not** used: its camera is a fisheye whose distortion model
  the reader feeds do not carry, so a reconstruction there would measure the estimator's fisheye
  failure. The held-out simulated clips and the held-out TII flights are not read.

Estimator (frozen; amendments 1-3): Pi3X, every 4th frame (7.5 Hz: simulator every 4th 30 fps
frame, UZH every 2nd prepared 15.07 Hz frame), reconstructed in overlapping 8 s chunks of 60
frames with 4 s stride so the reconstruction density does not fall with clip length; exact
intrinsics (simulator: the render's own K; UZH: Kalibr pinhole). The ground plane is fitted per
3 s window from that chunk's points in the window ± 0.5 s, by RANSAC (2000 trials, 1 % of scene
scale, three candidates), admitting a plane when ≥ 95 % of camera centres are above it, its mean
camera height exceeds 0.01 scene scale, it is not parallel to the flight, and it holds ≥ 10 % of
the points. A window without an admissible plane is an abstention.

Statistic: e = log(h(t2)/h(t1)) − log(ratio_cmd) per window, 3 s with 0.5 s stride;
ratio_cmd = z(t2)/z(t1) on the simulator (scene ground at z = 0, exact truth) and
1 + (z(t2)−z(t1))/(s·h(t1)) on real footage, with one similarity scale s per chunk (ratio of RMS
radii of the camera-centre cloud to the Leica track). A scale-free alternative,
q = (ρ_dep − ρ_cmd)/(|ρ_cmd| + 0.05) with ρ = Δz/L, is reported beside it.

Stop rule (frozen): **STOP** if the real median |e| exceeds the clip's own |defect log change|
for ×0.5 or ×1.5 on more than half of the 28 vertical-motion clips (the defect change is
computed analytically from the truth path, so it needs no reconstruction), or if ground-plane
coverage is below 50 % on the vertical-motion clips or on the real sequences.

## Results

### Part 0a — where vertical motion exists

Simulated development set: 28/200 clips, all `climb_turn`. By scene family and camera mount
(uptilt): industrial 4/3/5/4 at 0°/15°/30°/45°, urban checker 3/2/2/2, forest ridge 0/2/1/0
(forest has too few clips to cover all mounts). Peak |dz| 9.3-48.4 m from a 12 m start; the
per-3 s commanded ratio is 1.13-1.31 (median 1.20).

Real: the three UZH outdoor sequences have continuous vertical motion (|dz| over 3 s: medians
0.80/1.60/0.80 m, maxima 4.0/3.8/3.0 m). Their per-3 s commanded ratio has median 1.18, i.e. the
estimated camera height at the window start is about 2 m — the flights are low and fast.

### Part 0b — the estimator on the rendered clips

SIM_SECTION

### Part 0c — the estimator on real outdoor footage

| sequence | windows ok/window | coverage | median \|e\| | p90 \|e\| | median \|q\| |
| --- | --- | --- | --- | --- | --- |
| outdoor_forward_1 | 45/54 | 0.83 | 0.555 | 1.361 | 0.627 |
| outdoor_forward_3 | 35/54 | 0.65 | 0.373 | 1.912 | 0.344 |
| outdoor_forward_5 | 17/24 | 0.71 | 0.495 | 1.376 | 0.504 |
| **pooled** | **97/132** | 0.73 | **0.495** | **1.539** | **0.499** |

Sequence-clustered bootstrap over three clusters: median 0.495, range [0.373, 0.555] — three
clusters is descriptive, not a 95 % interval. Coverage passes the 50 % bar (all three sequences
covered); the abstentions are windows where the plane was inadmissible or the window ends fell
outside the reconstructed span.

The error is not a scale mistake: the reconstruction's *horizontal* path length matches the
truth per chunk to 0.96-1.46×, so the camera trajectory is reconstructed approximately. The
failure is in the height above the plane. Per chunk, the correlation between the depicted height
and the truth altitude is +1.00 in the first chunk of `outdoor_forward_1` and then 0.04, 0.35,
−0.55, 0.03, −0.71, −0.92; the implied ground-normal error (the tilt that would explain how
strongly the depicted height depends on horizontal position, with the truth altitude as the
reference) has a median of 6.9° (range 1.3-52°) on that sequence and 11.1° on
`outdoor_forward_3`. At a camera height of 1-3 m and 15-45 m of horizontal travel per chunk, a
normal error of a few degrees is larger than the height itself, and the ratio statistic then
divides by a number comparable to its own error: ratio_cmd reaches 0.07 and 3.6 in the worst
windows.

### The stop rule

Real median |e| = 0.495. The per-clip strictest ×0.5/×1.5 defect bar has median 0.079 (range
0.057-0.112); the "removed" bar has median 0.179 and the "reversed" bar 0.398. 0.495 exceeds the
strictest bar on **28/28** clips, so condition A fires. Condition B does not (climb-clip coverage
SIM_COV%, real coverage 100 % of sequences). **STOP**: per the pre-registration no defect clip is
rendered and Parts 1-3 are not run.

### Exploratory (not confirmatory): the rest of UZH

EXPLORATORY_SECTION

## What it supports

1. EyeHeight's measurement transfers to *commanded* vertical motion on rendered aerial clips at
   12-60 m: on the frozen simulator the depicted height ratio follows the commanded ratio to a
   few per cent, with full coverage (numbers above).
2. The check as specified is not certifiable on real low-altitude forward-camera drone footage.
   The estimator's real error is 4-8× the smallest defect it must catch, on every clip.
3. The mechanism is geometric and quantified: a ground-plane normal error of several degrees,
   multiplied by the horizontal travel per chunk, swamps a 1-3 m camera height. This is a
   property of the viewing geometry, not of the window length, the threshold or the model.

## What it does not show

* No detection rate is claimed. The pre-registered stop fired before any vertical-motion defect
  was rendered, so the check's power on simulated defects and on generated clips is unmeasured.
  From the estimator's own sim error (SIM_MED) and the analytic defect bars (0.06-0.11 for ×0.5
  and ×1.5, 0.18 for removal, 0.40 for reversal) the removal and reversal arms would be expected
  to separate and the ×0.5/×1.5 arms to be marginal at a 3 s window; that is an expectation, not
  a measurement.
* Nothing is claimed about any generator's vertical fidelity. ANWM's conditioning paths do carry
  a 22.5 m climb in 16 of its H01 clips and SolarWM's "climb turn" is a 0.19/0.37 m excursion, but
  neither was reconstructed for a score (the inputs are staged: `data/v/gen/`).
* The failure is *not* shown to be unavoidable. Three untested alternatives change the geometry:
  a downward-looking rig (UZH's 45°-down indoor sequences are the natural first test), a
  near-field-only plane fit, or a gravity-referenced height from inertial-aligned visual odometry.

## Caveats

* The real pilot has three sequence clusters; every interval here is descriptive.
* The real statistic needs a similarity scale per chunk. It is estimated from RMS radii and the
  horizontal path check above says it is approximately right, but a scale error of 10 % would
  move ratio_cmd by 10 % of (ratio_cmd − 1) — small compared with the observed error.
* Pi3X's `conf` field is a log confidence; the floor was fixed at −1.0 by amendment 3 after a
  probe, before any height was computed. The plane-admissibility rule also changed there (the
  camera-axis "facing" test was removed because the reconstruction's gauge is arbitrary); both
  changes are recorded and neither touches the stop rule.
* One reconstruction directory was produced twice by two concurrent code versions (a killed
  `modal run` does not stop the remote function). It was deleted and the pilot re-run once from
  a single detached job; the analysis refuses units without the new confidence-field stamp.
* Cost so far: about 2.5 USD of the 15 USD ceiling (see `docs/COST_LEDGER.md`), of which about
  1.0 USD was the wasted concurrent passes.

## Reproduction

```
PYTHONPATH=. python experiments/v_stage_frames.py --uzh-all      # frames at 7.5 Hz
modal volume put awmv-v-data data/v/frames v_frames              # upload
modal run remote/v_recon.py --all-units                          # Pi3X + plane fits
python scripts/pull_volume.py awmv-v-data recon data/v/recon
PYTHONPATH=. python experiments/v00_feasibility.py               # every number here
PYTHONPATH=. python experiments/v00_figure.py outdoor_forward_1
```
