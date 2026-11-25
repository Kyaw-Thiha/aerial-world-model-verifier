# V0 — Commanded vertical motion: feasibility

**Decision.** The check stops at feasibility. The pre-registered stop rule fires on condition A:
the EyeHeight-style height-ratio estimator's error on real UZH-FPV outdoor footage (median
|log ratio error| 0.480 over 95 windows from three sequences) exceeds the effect of the smallest
labelled vertical defect on 28 of 28 simulated climb clips (median bar 0.077), so no defect was
rendered and no detection rate is claimed. The estimator itself behaves as specified on the
rendered clips (median 0.094 over 761 windows, 27/28 clips covered), so the failure is the
geometry of low-altitude flight, not the statistic — and it is quantified below.

Run directory `results/v/v00/` (`summary.json`, `table.md`, `rigs.md`, per-unit window tables,
`fig_depicted_vs_truth__outdoor_forward_1.png`). Pre-registration
`configs/v00_feasibility_prereg.yaml` (`2e948fa`) and amendments 1-4; the V1 pre-registration
(`configs/v10_vertical_prereg.yaml` + amendment 1) stays **unrun**.

## Question

EyeHeight (arXiv 2611.21782) measures whether a world model *keeps* a camera's height above the
floor on constant-height indoor trajectories, with a scale-free multi-view reconstruction. Aerial
world models take altitude as an action, so the complementary question is whether a commanded
climb or descent from `z0` to `z0+dz` is *depicted* with the commanded ratio `(z0+dz)/z0`. Before
any detection rate can mean anything, the measurement must be good enough that real, feasible
drone flights are not rejected; that is what this report measures.

## Setup

**Populations** (frozen before any output; amendments 1-2). *Simulated pilot*: every development
clip of the frozen Track I set whose manoeuvre is `climb_turn` — 28 of 200, the only manoeuvre
with commanded vertical motion — plus two constant-altitude controls (`i02_dev_000` hover,
`i02_dev_001` ellipse). Peak |dz| is 9.3-48.4 m (median 21.7 m) from a 12 m start; 8 of the 28 are
feasible clips and 20 carry an attitude defect, which does not change the camera's altitude.
*Real pilot*: UZH-FPV's three outdoor forward sequences (29.7/29.7/14.7 s; 3 s |dz| medians
0.80/1.60/0.80 m, maxima 4.0/3.8/3.0 m; vertical speed to 3.8 m/s; altitude range 4.3-4.8 m).
TII is **not** used: its 640x480 fisheye distortion model is not carried by the reader feeds, so
a reconstruction there would measure the estimator's fisheye failure. The held-out simulated
clips and the held-out TII flights (13p, 07a) are not read.

**Estimator** (frozen; amendments 1-4). Pi3X, every 4th frame at 7.5 Hz (simulator: every 4th
30 fps frame; UZH: every 2nd prepared 15.07 Hz frame), reconstructed in overlapping 8 s chunks of
60 frames with a 4 s stride, so the reconstruction density does not fall with clip length; exact
intrinsics. The ground plane is fitted per 3 s window from the window's own chunk, using points
from frames in the window ±0.5 s, by RANSAC over the chunk's most confident quartile of pixels
(2000 trials, 1 % of scene scale, three candidates). A plane is admissible when ≥95 % of camera
centres are above it, its mean camera height exceeds 0.01 scene scale, it is not parallel to the
flight (|n·v̂| ≤ 0.6), and it holds ≥10 % of the points. A window without an admissible plane is
an abstention and is counted.

**Statistic**. `e = log(h(t2)/h(t1)) − log(ratio_cmd)` over 3 s windows with a 0.5 s stride,
where `h` is the camera's signed distance to the window's plane. On the simulator
`ratio_cmd = z(t2)/z(t1)` (scene ground at z = 0, exact truth); on real footage
`ratio_cmd = 1 + (z(t2)−z(t1))/(s·h(t1))` with one similarity scale `s` per chunk (ratio of RMS
radii of the camera-centre cloud to the Leica track). A scale-free alternative,
`q = (ρ_dep − ρ_cmd)/(|ρ_cmd| + 0.05)`, is reported beside it.

**Stop rule** (frozen): STOP if the real median |e| exceeds a clip's own |defect log change| for
×0.5 or ×1.5 on more than half of the 28 clips, or if ground-plane coverage is below 50 %. The
defect change is computed analytically from the truth path, so it needs no reconstruction.

## Results

### Part 0a — where commanded vertical motion exists

Within 200 development clips: 28, all `climb_turn`. By scene family and camera mount (uptilt):
industrial 4/3/5/4 at 0°/15°/30°/45°, urban checker 3/2/2/2, forest ridge 0/2/1/0 (forest has too
few clips to cover every mount). The three UZH outdoor sequences have continuous vertical motion
throughout, which is why they are the real pilot.

### Part 0b — the estimator on the rendered clips

28 clips, 761 of 782 windows reconstructable (97.3 %), 27 of 28 clips covered (the abstaining
clip has 48 % of its windows). Pooled median |e| = **0.094**, p90 = 0.227; per-clip medians have
median 0.093 and maximum 0.214. The two constant-altitude controls read 0.00008 (hover) and 0.107
(ellipse): no systematic bias, but a 10 % false signal on a moving constant-altitude clip, which
is the estimator's own noise level.

Power at these numbers: the labelled defects change a per-window ratio of about 1.16 by
|log| 0.077 (×0.5), 0.175 (removal) and 0.388 (sign reversal) at the clip median. The estimator's
own error (0.094) is therefore of the same size as the ×0.5/×1.5 effect even on rendered clips,
twice smaller than removal, and four times smaller than reversal. In other words the check would
have been marginal for the smallest magnitude even where the estimator works.

### Part 0c — the estimator on real outdoor footage

| sequence | windows ok/all | coverage | median \|e\| | p90 \|e\| | median \|q\| |
| --- | --- | --- | --- | --- | --- |
| outdoor_forward_1 | 45/54 | 0.83 | 0.618 | 1.453 | 0.569 |
| outdoor_forward_3 | 33/54 | 0.61 | 0.502 | 1.816 | 0.377 |
| outdoor_forward_5 | 17/24 | 0.71 | 0.433 | 1.600 | 0.498 |
| pooled | 95/132 | 0.72 | **0.480** | 1.653 | 0.511 |

Sequence-clustered bootstrap over three clusters: 0.480 in [0.433, 0.618] — three clusters is
descriptive, not a 95 % interval. Coverage passes (all three sequences covered); the abstentions
are windows whose plane was inadmissible or whose ends fell outside the reconstructed span.

**Why it fails — the mechanism, measured.** The error is not a scale mistake: per chunk the
reconstruction's horizontal path length matches the truth to 0.96-1.46×, so the camera trajectory
is approximately right. What fails is the *height above the fitted plane*. The implied
ground-normal error, obtained by solving |e| = δ·L/h for δ per window, is 1.8-3.7° on the real
sequences (medians) and 6.6° on the rendered ones (median, range 1.0-28.5°) — the plane is fitted
about equally badly in both. What differs by an order of magnitude is the **leverage** L/h: 7.9-12.0
on the real flights (camera 1-2 m up, 15-45 m of travel per window) against 0.23-1.63 on the
rendered clips (12-60 m up). The same 2-7° normal error that costs ~0.5 in log ratio at 2 m costs
~0.05-0.1 at 12-60 m. The per-chunk correlation between depicted height and truth altitude is
consistent with this: −0.93…+1.00 across the three sequences, i.e. no stable relation at the
scale of a chunk (figure).

### The stop rule

Real median |e| = 0.480. Per-clip strictest ×0.5/×1.5 bar: median 0.077, range 0.057-0.112. The
real error exceeds it on **28/28** clips, so condition A fires. Condition B does not (climb
coverage 96 %, real coverage 100 %). **STOP**: per the pre-registration, no defect clip is
rendered and Parts 1-3 (simulated detection by family and mount, real false-reject table,
generated-clip arm) are not run.

### Exploratory (not confirmatory): every UZH rig

`results/v/v00/rigs.md` runs the same statistic on all 14 UZH development sequences. Median |e|
is 0.136-0.618 across rigs; the two 45°-down camera families do **not** do better than the
forward ones (indoor 45°: 0.366-0.445; indoor forward: 0.136-0.578; outdoor 45°: 0.220), because
their L/h is still 6-7 — the flights are low, not because the view is forward. `outdoor_45_1`
flies higher (8.9 m, L/h 1.6) and has the lowest error of the 45° group (0.220) with an implied
normal error of 8.0°. So no UZH rig sits in the regime where this estimator works.

## What it supports

1. EyeHeight's measurement transfers to *commanded* vertical motion on rendered aerial clips at
   12-60 m: the depicted height ratio follows the commanded ratio to a median 9 %, with 97 %
   window coverage and no bias on a hover.
2. The check as specified cannot be certified on real drone footage of this kind. The real error
   is 4-8× the smallest defect it must catch, on every one of 28 clips, in all 14 rigs tried.
3. The mechanism is geometric and quantified: the ground-normal error is 2-7° in both domains and
   the ratio error scales as δ·L/h, so the same plane error is an order of magnitude worse where
   the camera flies low.

## What it does not show

* No detection rate. The pre-registered stop fired before any vertical defect was rendered, so
  the check's power on simulated defects and on generated clips is unmeasured. From the
  measured error (0.094) and the analytic bars (0.077/0.175/0.388), removal and reversal would be
  expected to separate on rendered clips and ×0.5/×1.5 to be marginal — an expectation, not a
  measurement.
* Nothing about any generator. ANWM's conditioning paths do carry a 22.5 m climb (16 H01 clips)
  and SolarWM's "climb turn" is a 0.19/0.37 m excursion rather than a climb, but neither was
  reconstructed for a score; the inputs are staged in `data/v/gen/` if the route is reopened.
* Not shown that no vertical-motion check is possible. The untested alternatives change the
  geometry or the estimator: a near-field-only plane fit (the ground points nearest the camera),
  a gravity-referenced height from inertial-aligned visual odometry, or a plane-free formulation
  that compares *depicted* vertical velocity against a metrically scaled path. What would have to
  improve is concrete: to hold a 0.08 log-ratio tolerance at h ≈ 2 m and L ≈ 20 m, the ground
  normal must be known to about 0.5°, and the estimate is 2-7°.

## Caveats

* Three real sequence clusters; every interval here is descriptive.
* The real statistic needs a per-chunk similarity scale. The horizontal path check says it is
  approximately right (0.96-1.46), and a 10 % scale error moves `ratio_cmd` by 10 % of
  (ratio_cmd−1), small against the observed error.
* Pi3X's confidence is a log field; the floor became a per-chunk 75th percentile (amendment 4)
  after an absolute floor abstained on five clips for reasons unrelated to the scene. The pilot
  was re-run once from a single job under the amended rule; the real median moved from 0.495 to
  0.480, so the stop is not an artefact of the floor.
* One volume directory was initially written by two concurrent code versions (a killed
  `modal run` does not stop the remote function); it was deleted and re-run, and the analysis
  refuses incomplete units.
* Cost: about 5.5 USD of the 15 USD ceiling (`docs/COST_LEDGER.md`), including about 1 USD lost
  to that accident and about 2.5 USD for the amended re-run.

## Reproduction

```
PYTHONPATH=. python experiments/v_stage_frames.py --uzh-all
modal volume put awmv-v-data data/v/frames v_frames
modal run remote/v_recon.py --all-units
python scripts/pull_volume.py awmv-v-data recon data/v/recon
PYTHONPATH=. python experiments/v00_feasibility.py      # every number in this report
PYTHONPATH=. python experiments/v00_rigs.py             # the exploratory rig table
PYTHONPATH=. python experiments/v00_figure.py outdoor_forward_1
```
