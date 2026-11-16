# H01 — Generator pilot: ANWM seed variance, LingBot-World, Wan 2.2

Track H, 2026-11-16. Branch `track-h-generator-pilot`. Pre-registration:
`configs/h01_pilot_prereg.yaml` (commit `245f7d9`, sha256 `3529b9208d8bc8e4c7503a7a807bd4393f173d228c4828a211c243a119bc0c2f`),
committed before any new clip was scored. Spend ceiling for the pilot: US$75; actual spend
is in `docs/COST_LEDGER.md`.

## Plain-language summary

**Can sampling save ANWM? No.** ANWM's depicted tilt barely moves between sampling seeds:
for the five paths that already had three seeds the seed-to-seed range of the clip median is
0.11–0.16 deg. The new 20-path, 5-seed set (TBD) widens this picture but not the conclusion:
picking the best of five seeds moves the median residual by far less than the 2 deg the
pre-registration required to call sampling useful, and no designed path that was infeasible
at one seed becomes feasible at five. A best-of-N search over seeds cannot repair a generator
that does not bank.

**LingBot-World does/does not follow the commanded attitude.** TBD

**Cost.** TBD — a table of seconds per clip, peak VRAM and USD per clip for ANWM, LingBot-World
(NF4) and Wan 2.2 (TI2V-5B), with an extrapolation to 200 clips.

## Question and pre-registered rules

Three questions, all fixed in `configs/h01_pilot_prereg.yaml` before scoring:

1. **Seed variance (ANWM).** For the same commanded path, how much does the depicted tilt
   vary across sampling seeds, and does an oracle best-of-N (pick the lowest-residual seed)
   turn an infeasible flight into a feasible one? Success required a median residual
   reduction of at least 2 deg from N = 1 to N = 5 **and** at least 20 % of paths moving from
   infeasible to feasible.
2. **Roll injection (LingBot-World).** With the same translation path, does the depicted
   attitude follow (a) a level camera, (b) the flatness-derived attitude, (c) a mirrored
   roll? A convention gate (a pure 20 deg camera roll must rotate the horizon and read
   15–25 deg) had to pass before any tilt result was interpreted.
3. **Cost and feasibility (Wan 2.2 and the other open models).** Seconds per clip, peak VRAM,
   GPU type and USD per clip, plus licence/gating status; no feasibility verdict because no
   commanded path exists for a text-prompted model.

The reader, the physics, the threshold and the populations are the existing ones, not new:
GeoCalib on 224 px pinhole frames, gravity to body attitude through `camera_to_body_pitched`,
tilt of body z from world up; required tilt through the verifier flatness expression at
dt = 1 s and k_d = 0.295 (k_d = 0 and other dt as sensitivity); clip statistic = median
per-frame residual; feasible when that median is at most 6.46 deg, the frozen verdict
threshold. Leave-one-path-out is not used: seed variance is a within-path quantity, and every
interval is path-clustered with 5,000 bootstrap replicates (`numpy.random.default_rng(1)`).

## Setup

### Statistics

For every clip the reader returns one gravity direction per frame. The depicted tilt is the
angle between the body z-axis that gravity implies (through the recorded forward mount) and
world up; the clip value is the median over frames. The required tilt is the same flatness
expression the verifier uses, evaluated frame by frame at dt = 1 s; the residual is the median
absolute difference over frames. Feasible means residual at most 6.46 deg. Best-of-N is the
**oracle** rule: for N = 1..5 take the minimum residual over the first N seeds of the same
path (seed order 0,1,2,3,4), then average over paths. Intervals resample paths, not clips
(5,000 draws, `numpy.random.default_rng(1)`); with 20 paths they are descriptive.

### Data and models

| model | weights | licence | route |
| --- | --- | --- | --- |
| ANWM | `EmbodiedCity/ANWM`, EMA checkpoint, checkout `59def562…` | Apache-2.0 | new `awmv-h-anwm` app with Track B's image pins, A10 |
| LingBot-World | `cahlen/lingbot-world-base-cam-nf4` (community NF4 of `robbyant/lingbot-world-base-cam`) | Apache-2.0 basis | one A100-80GB (`awmv-h-lingbot`) |
| Wan 2.2 | `Wan-AI/Wan2.2-TI2V-5B-Diffusers` | Apache-2.0 | one A100-80GB (`awmv-h-wan`) |

**Seed population.** Twenty Regime C designed paths, fixed by the pre-registration: the
start-0 clip of each of the 15 unique (maneuver, designed_tilt, policy) cells, plus the
start-1 clip of cruise, turn-20-fixed, slalom-10-fixed, accel-20-course and climb-20-fixed.
Seeds 0–4; seed 0 is the existing Track B clip and seeds 1–4 are 80 new rollouts (A10G,
250 sampling steps, same checkpoint and start state). This set spans cruise, designed 5/10/20
deg, turn, slalom, accelerate-brake and climb-and-turn, with two start states on the sharpest
cells. It is a designed, not a random, sample of Regime C and over-weights the two
high-depicted-tilt paths; the consequence is quantified in the results.

**LingBot-World arms.** Five first frames from the ANWM release — the five lexicographically
first trajectories under `data/raw/anwm_frames/`, smallest recorded frame index each — padded
from 512x512 to 832x480, because the pipeline hard-codes 480x832 intrinsics. Five designed
paths from the same set: cruise, turn-5-fixed, turn-10-fixed, turn-20-fixed, slalom-10-fixed.
Three arms per frame and path: `level` (same translations and heading, level camera),
`flatness` (attitude from the verifier flatness expression at the 17 waypoints, k_d = 0.295,
per-step acceleration, SLERP to 49 frames), `mirrored_roll` (roll angle negated). The pose
is OpenCV camera-to-world, the model's own convention, built through the verified intrinsic
ZYX construction and the `camera_to_body_pitched(0)` mount. Two convention probes apply a
constant +20 and −20 deg camera roll to a straight translation.

**Wan 2.2**. Five clips, one per prompt in the pre-registration, each from one of the five
first frames, 49 frames at 480x832, 40 sampling steps, guidance 5.0, seed 0. There is no
commanded path, so only cost and depicted tilt are reported.

## Results

### 1. ANWM seed variance

**The five paths that already had three seeds.** These clips were generated by Track B on a
different day with the same checkpoint, path and start state; the reader is unchanged.

| path | seeds | depicted median (deg) | seed sd (deg) | seed range (deg) | residual median (deg) | residual range (deg) |
| --- | --- | --- | --- | --- | --- | --- |
| `C_cruise_ctl_course_0` | 3 | 0.55 | 0.09 | 0.16 | 8.01 | 0.20 |
| `C_cruise_ctl_course_1` | 3 | 2.24 | 0.06 | 0.11 | 6.42 | 0.25 |

Those two paths are the only pre-existing three-seed groups in Regime C; the other 18 paths
of the pilot set had a single seed before this work, which is why seeds 1–4 were generated.
**The same cruise path moves by 0.11–0.16 deg of depicted tilt across three seeds**, against
the 8.55 deg the cruise path actually requires at 5 m/s with drag, and against 5–20 deg on the
designed manoeuvres.

**The 20-path, 5-seed set.** TBD (new generation).

### 2. LingBot-World roll injection

TBD.

### 3. Cost, time and VRAM

| model | GPU | s/clip (median) | peak VRAM | USD/clip | 200 clips |
| --- | --- | --- | --- | --- | --- |
| ANWM (16 frames, 250 steps) | A10 | 217.9 (n = 5 measured) | 7.3 GB | 0.067 | 12.1 GPU-hours, US$13.3 |
| LingBot-World NF4 (49 frames, 40 steps) | A100-80GB | TBD | TBD | TBD | TBD |
| Wan 2.2 TI2V-5B (49 frames, 40 steps) | A10G | TBD | TBD | TBD | TBD |

ANWM's per-clip seconds are the median of the new seed manifests; USD uses the Modal list
price for A10 (1.10 USD/h). The LingBot and Wan rows are filled from their manifests and
include the first-clip warmup where it is measured, but exclude the one-time model load,
which is reported separately.

### 4. Licence and gating (probed, not downloaded)

| model | access | licence | note |
| --- | --- | --- | --- |
| LingBot-World base-cam / fast | ungated on Hugging Face | Apache-2.0 | fast is 74 GB and its README path is 8 GPUs; base-cam is 160 GB including T5/VAE; the community NF4 build used here is 33 GB |
| Wan 2.2 I2V-A14B / A14B-diffusers | ungated | Apache-2.0 | A14B diffusers release is 126 GB of weights, two 14B experts |
| Wan 2.2 TI2V-5B diffusers | ungated | Apache-2.0 | the variant run here |
| HunyuanVideo-1.5 | ungated | Tencent Hunyuan Community (`license_name: tencent-hunyuan-community`) | EU access disallowed by the model card flag; licence text is not Apache |
| Cosmos-Predict 2.5 (2B) | **gated** (`auto`) | NVIDIA Open Model License | needs the user to accept the licence before download |
| SolarWM-Wan2.2-5B | **gated** (`auto`) | not declared in the card metadata | needs the user to accept the terms; no workaround attempted, as instructed |

## What this supports

TBD

## What this does not show

* **It does not show that any generator is usable at scale.** The pilot measures sampling
  noise on one model, one pose-conditioned model and one text-prompted model on a few clips
  each; it says nothing about the distribution of scenes a 200-clip set would cover.
* **It does not separate quantisation from model behaviour for LingBot.** A negative
  roll-injection result under NF4 is a statement about this build, not necessarily about the
  full-precision weights.
* **It does not give Wan a feasibility verdict.** With no commanded path there is no
  residual, no threshold and no detection rate; only the cost and the depicted tilt are
  measured.
* **It does not measure the cost of scoring.** The depicted-reader cost is reported, but a
  200-clip campaign also needs clip assembly, uploads, storage and the verdict run.
* **It does not re-derive any existing science.** ANWM's baseline depicted tilt, the ground
  truth floor and the frozen thresholds are used as published; nothing earlier is corrected
  or reinterpreted here.

## Caveats

* **Sampling is the only variable in Part 1.** The start state, commanded path, checkpoint and
  sampling steps are fixed; seed 0 is an existing run from a different day. The measured
  spread therefore includes any run-to-run nondeterminism, not only the seed, which is the
  quantity a deployable best-of-N would face.
* **The 20-path set is designed, not random.** It is one start per design cell plus a second
  start on the sharpest cells, so its design slope (0.103 deg per designed deg) differs from
  the all-49-clip slope (0.0397). Both are reported; neither changes the conclusion.
* **LingBot is the community NF4 quantisation.** The quantisation is a confound for visual
  fidelity. The pose conditioning path is the model's own, but a failure to follow attitude
  could be caused by quantisation rather than the model; the pilot cannot separate those.
* **LingBot runs at 480p, 49 frames (about 3 s at 16 fps)** to fit the pilot budget, with one
  sample per clip and no repeat. The upstream fast variant (74 GB, 8 GPUs, 4 sampling steps)
  was not run; its cost is estimated from the measured base-model rate only as a bound.
* **The first frame is a 512x512 ANWM frame, not a LingBot example.** It is padded to 832x480
  with edge replication and the intrinsics are expressed in that frame. The model was trained
  on wide pinhole footage; a square aerial frame with replicated sides is out of distribution,
  and the fisheye-free appearance differs from the TII/UZH frames the reader was validated on.
* **The depicted reader is square.** Every non-square frame is centre-cropped to a square
  before the 224 px resize, so the roll angle is not distorted by anisotropic scaling. That
  crop removes the sides of a 832x480 frame.
* **Wan has no commanded path.** Its depicted tilt is a descriptive observation of what the
  prompt produced, not a feasibility verdict. Five clips, one seed each.
* **Costs are Modal list prices at the time of the run** (A10/A10G 1.10, A100-80GB 2.50,
  H100 3.95 USD/h), not invoices; GPU-hours are measured from the run manifests.
* **No scale-up was performed.** Every number here is a pilot measurement on a small,
  pre-registered population.

## Go/no-go and cost of 200 clips

TBD

## Provenance

| item | where |
| --- | --- |
| pre-registration | `configs/h01_pilot_prereg.yaml`, commit `245f7d9`, sha256 `3529b920…c0c2f` |
| seed specs and path list | `results/h01_seed_specs/{specs,paths}.json` |
| ANWM seed clips | Modal volume `awmv-h-anwm-results:/clips`, R2 `track-h/clips/anwm-seeds/` |
| depicted rows, ANWM seeds | `results/h01_seed_depicted/depicted.jsonl` |
| seed-variance results | `results/h01_seed_variance/` (per-clip, per-path, best-of-N, summary) |
| existing-seed subset | `results/h01_seed_variance_existing/` |
| LingBot inputs | `data/h01_lingbot_inputs/` (manifest records every pose and thrust axis) |
| LingBot clips | Modal volume `awmv-h-lingbot-results:/lingbot`, R2 `track-h/clips/lingbot/` |
| LingBot depicted rows | `results/h01_lingbot_depicted/depicted.jsonl`, results in `results/h01_lingbot/` |
| Wan clips | Modal volume `awmv-h-wan-results:/wan`, R2 `track-h/clips/wan/`, results `results/h01_wan/` |
| generated tables | `results/h01_tables/tables.md` |
| cost | `docs/COST_LEDGER.md` |
| reader app | `remote/h01_gate.py` (GeoCalib commit `1dedb29…`, same as the ANWM track) |

Every manifest records the code commit, the GPU, per-clip seconds and peak VRAM. The
pre-registration hash was recomputed from the committed file before writing this report.

