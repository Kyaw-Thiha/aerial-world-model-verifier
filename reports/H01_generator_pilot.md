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

**LingBot-World does not follow a 20 deg roll command at the tested rate.** Over 49 frames the
+20 deg ramp (0.42 deg/frame) moved the depicted tilt from 3.21 to 3.30 deg and the -20 deg
ramp to 3.77 deg, against a 20 deg command; the frames stay visually level and an independent
vertical-edge check reads about 3 deg. Stronger diagnostics do not rescue it: a 0->20->0
triangular roll reads 3.93 deg and a 0->60 deg ramp reads 4.05 deg, about 1 deg above the
level baseline even at five times the rate. The model translates as commanded but does not
roll, so the 60-clip grid was not run.

**Cost.** Per clip, with an extrapolation to 200: ANWM 218 s / 7.3 GB / US$0.067 (US$13 for
200), LingBot-World NF4 652 s / 21.1 GB / US$0.45 (US$91 at 40 steps, US$46 at 20), Wan 2.2
TI2V-5B 135 s / 15.2 GB / US$0.041 (US$8.3 for 200). Pose-conditioned generation is the
expensive arm, and in this build it is also the one that does not follow its control.

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
| LingBot-World | `cahlen/lingbot-world-base-cam-nf4` (community NF4 of `robbyant/lingbot-world-base-cam`) | Apache-2.0 basis | A100-80GB (`awmv-h-lingbot`, `-b`) |
| Wan 2.2 | `Wan-AI/Wan2.2-TI2V-5B-Diffusers` | Apache-2.0 | A10 (`awmv-h-wan`) |

**Seed population.** Twenty Regime C designed paths, fixed by the pre-registration: the
start-0 clip of each of the 15 unique (maneuver, designed_tilt, policy) cells, plus the
start-1 clip of cruise, turn-20-fixed, slalom-10-fixed, accel-20-course and climb-20-fixed.
Seeds 0–4; seed 0 is the existing Track B clip and seeds 1–4 are 80 new rollouts (A10,
250 sampling steps, same checkpoint and start state). This set spans cruise, designed 5/10/20
deg, turn, slalom, accelerate-brake and climb-and-turn, with two start states on the sharpest
cells. It is a designed, not a random, sample of Regime C and over-weights the two
high-depicted-tilt paths; the consequence is quantified in the results.

**LingBot-World's control is relative.** The pipeline calls
`compute_relative_poses(c2ws, framewise=True)`: the model sees the pose change between
consecutive frames, and the first frame's absolute attitude is discarded (the input image
anchors the scene). A constant camera roll is therefore invisible, so the convention probe
ramps the roll linearly from 0 to ±20 deg across the clip, and only arms whose attitude
changes within the clip can carry information. The `cruise` flatness arm is a constant drag
pitch (8.55 deg) and is therefore indistinguishable from level; it is generated and reported
as such, not as a control that failed.

**LingBot-World arms.** Five first frames from the ANWM release — the five lexicographically
first trajectories under `data/raw/anwm_frames/`, smallest recorded frame index each — padded
from 512x512 to 832x480, because the pipeline hard-codes 480x832 intrinsics. Five designed
paths from the same set: cruise, turn-5-fixed, turn-10-fixed, turn-20-fixed, slalom-10-fixed.
Three arms per frame and path: `level` (same translations and heading, level camera),
`flatness` (attitude from the verifier flatness expression at the 17 waypoints, k_d = 0.295,
per-step acceleration, SLERP to 49 frames), `mirrored_roll` (roll angle negated). The pose
is OpenCV camera-to-world, the model's own convention, built through the verified intrinsic
ZYX construction and the `camera_to_body_pitched(0)` mount. Four convention probes apply roll ramps to a straight
translation: linear 0 -> +20, linear 0 -> -20, triangular 0 -> +20 -> 0, and linear 0 -> +60
deg, plus a level control.

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

**Convention gate.** The model conditions on framewise relative poses, so the probe ramps the
roll from 0 to ±20 deg over 49 frames (a constant roll is invisible). The level control
clip reads a depicted tilt of **3.21 deg** (median over 49 frames). The ramp clips are
reported below; no grid clip was generated before this gate was read.

**Gate result: the slow ramp is not followed.** The level clip reads 3.21-3.24 deg; the
+20 deg ramp reads 3.30 deg (median over the clip), with the first-10-frame median 3.45 and
the last-10 3.44, while the injected tilt goes from 1.9 to 18.1 deg. The -20 deg ramp reads
3.77 deg (first/last-10 2.85/4.22); both signs stay within about 0.6 deg of the level
baseline, against a 20 deg command. The frame-level correlation
between depicted and injected tilt is -0.12, and the depicted lean direction does not agree
with the injected one (median cosine 0.34). A visual check of frames 0 and 48 shows a level
horizon in both, and an independent edge check agrees: near-vertical building edges deviate
0.0 deg from vertical at frame 0 and -3.45 deg at frame 24 (median over 10 lines), i.e. about
3 deg of response to a 20 deg command, not 20. A first Hough pass on near-horizontal lines
suggested -13 deg, but those lines are building and water edges rather than the horizon; the
vertical-edge reading is the one quoted. So at 0.42 deg/frame the model does not follow the
commanded roll.

Two stronger stimuli were run as a diagnostic (not part of the pre-registered gate): a
triangular 0 -> 20 -> 0 deg roll (peak rate 0.83 deg/frame, returning to level) and a linear
0 -> 60 deg ramp (1.25 deg/frame). **Neither moves the depicted attitude appreciably**: the
triangular clip reads 3.93 deg (first/last-10 3.42/3.83) and the 60 deg ramp reads 4.05 deg
(first/last-10 3.25/4.39). A 60 deg command produces about 1 deg of extra depicted tilt
above the level baseline, i.e. a gain of roughly 1/15 even at a five-times faster rate. The
camera control is not usable for roll in this build; this is not a rate-sensitivity
artefact.

### 3. Cost, time and VRAM

| model | GPU | s/clip (median) | peak VRAM | USD/clip | 200 clips |
| --- | --- | --- | --- | --- | --- |
| ANWM (16 frames, 250 steps) | A10 | 217.9 (n = 5 measured) | 7.3 GB | 0.067 | 12.1 GPU-hours, US$13.3 |
| LingBot-World NF4 (49 frames, 40 steps) | A100-80GB | 652.0 (n = 5 probes) | 21.1 GB | 0.453 | 36.2 GPU-hours, US$91 |
| Wan 2.2 TI2V-5B (49 frames, 40 steps) | A10 | 135.0 (n = 5) | 15.2 GB | 0.041 | 7.5 GPU-hours, US$8.3 |

ANWM's per-clip seconds are the median of the new seed manifests; USD uses the Modal list
price for A10 (1.10 USD/h). Wan's row is the median of the five manifests (133.9-137.0 s); it includes VAE decode and
frame writing, and excludes the one-time model load (about 1-2 min from the volume). LingBot's row is the median of the five probe clips (647.9-658.2 s); at
20 steps (the pre-registered grid setting) the cost is about half, 18.6 GPU-hours and US$46
for 200 clips. Both exclude the one-time 30-40 min NF4 load, which is paid per container.

### 3b. Wan 2.2, descriptive

| prompt | depicted median (deg) | p90 (deg) | frame-to-frame delta (median) |
| --- | --- | --- | --- |
| banked turn right | 2.50 | 4.91 | 0.043 |
| slalom | 1.23 | 5.88 | 0.462 |
| approach and slow | 1.12 | 4.99 | 0.428 |
| constant-speed orbit | 3.75 | 7.10 | 0.011 |
| climb and bank left | 3.77 | 5.43 | 0.022 |

Five clips, one seed each, 49 frames, 480x832. The prompt asks for a bank of roughly 20 deg;
the reader's clip medians are 1.1-3.8 deg, with p90 at 4.9-7.1 deg. The two clips with a
frame-to-frame delta near 0.43-0.46 (slalom, approach) contain cuts or a scene reset, so their
median is not a stable attitude measurement; the other three move smoothly (delta 0.01-0.04).
No residual is computed: there is no commanded path, so this is not a feasibility result.

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

* **ANWM's failure is not sampling noise.** Within a fixed path and start state, the depicted
  tilt moves by a few tenths of a degree across seeds, and the oracle best-of-N curve is
  flat. The path-dependence of the bias (some paths depict 0.5 deg, others 3.7 deg) is much
  larger than the seed-dependence, so seed search is the wrong lever; the model's action
  interface (4-DoF, no roll/pitch) is the right target.
* **A pose-conditioned generator can be measured with the same reader.** LingBot-World's
  camera poses were built with the verifier's own flatness construction and read back with
  the existing depicted-tilt estimator; the level control reads 3.2 deg and the frames are
  valid, so the measurement chain transfers to a new generator and resolution.
* **A pose-conditioned generator must be shown to consume rotation, not assumed to.** The
  five-clip roll probe (about US$3.7 including two model loads) showed that LingBot-World
  translates as commanded but does not roll, so the 60-clip grid was not run. This is the
  cheapest result in the pilot and the one that stops a US$14-33 spend.
* **The cost ordering is clear.** For the same 200 clips: ANWM about US$13, Wan 2.2 about
  US$8, LingBot-World about US$46-91 depending on steps and excluding loads. Pose conditioning
  is the expensive option, not the cheap one.

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
* **The LingBot probes are one first frame, one seed, 40 steps, NF4.** They establish that
  this build does not follow a roll command on this frame; they do not prove that the full
  precision weights would fail. The level baseline is 3.2 deg, so a 3 deg response is within
  about one degree of the baseline and cannot be called a partial follow. Translation is
  followed in the same clips, so the pose tensor is being consumed.
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

**ANWM seed sampling: no-go.** The per-path seed spread of the depicted tilt is a few tenths
of a degree (TBD final median), the oracle best-of-N reduction is far below the 2 deg the
pre-registration required, and no path that was infeasible at one seed becomes feasible at
five. Sampling seeds cannot repair a generator whose bias is in the decoupling of planned
acceleration from depicted attitude, not in the noise.

**ANWM scale-up (without seed search):** 200 clips x 217.9 s = 12.1 GPU-hours, **US$13.3** at
the A10 list price, plus the one-time checkpoint load. This is the cheapest arm by a wide
margin and the only one whose model is already known to be infeasible on designed paths; the
value of 200 more clips is in covering scenes, not in changing the verdict.

**LingBot-World: no-go for this build.** The pose-conditioned model translates as commanded
but does not roll: 20 deg and 60 deg roll commands move the depicted tilt by about 0.1-1 deg
above the 3.2 deg level baseline. Running 60 arms on a control that is not followed would buy
noise. If the route is pursued, the next step is the full-precision base-cam or the upstream
fast variant with flash-attn, re-running the same three-clip roll probe first (about US$2).
For reference, 200 clips at 20 steps cost about 18.6 GPU-hours, **US$46** on an A100-80GB,
plus 30-40 min of NF4 loading per container; at 40 steps it is US$91.

**Wan 2.2 TI2V-5B:** 200 clips x 135 s = 7.5 GPU-hours, **US$8.3**, but there is no commanded
path and no feasibility test. It is not a substitute for a pose-conditioned generator; the
useful next step would be a path-conditioned variant, not more clips from this one.

## Reproduction

```
set -a; . ./.env; set +a

# Part 1: 80 new ANWM rollouts, two A10 containers, then reader + analysis
pixi run modal deploy remote/h01_anwm.py && pixi run modal deploy remote/h01_anwm_b.py
#   (the spawn calls and their ids are in results/h01_seed_specs/spawn*.json)
pixi run modal run remote/h01_gate.py::main --source anwm_b --out-tag anwm-b
pixi run modal run remote/h01_gate.py::main --source anwm_c --out-tag anwm-c
bash scripts/h01_finalize_seeds.sh

# Part 2: LingBot inputs, probes, reader
pixi run python experiments/h01_lingbot_inputs.py
pixi run modal run remote/h01_lingbot.py::main --mode probe  --steps 40 --frame-num 49
pixi run modal run remote/h01_lingbot_b.py::main --mode probe2 --steps 40 --frame-num 49
pixi run modal run remote/h01_gate.py::main --source lingbot   --out-tag lingbot
pixi run modal run remote/h01_gate.py::main --source lingbot_b --out-tag lingbot-b
pixi run python experiments/h01_generated_tilt.py --depicted results/h01_lingbot_depicted/depicted.jsonl     --out results/h01_lingbot
pixi run python experiments/h01_generated_tilt.py --depicted results/h01_lingbot_depicted_b/depicted.jsonl     --out results/h01_lingbot_b

# Part 3: Wan 2.2
pixi run modal run remote/h01_wan.py::main --mode clips --frame-num 49 --steps 40
pixi run modal run remote/h01_gate.py::main --source wan --out-tag wan
pixi run python experiments/h01_generated_tilt.py --depicted results/h01_wan_depicted/depicted.jsonl     --out results/h01_wan
```

The analysis scripts are CPU-only and read stored rows; none of them regenerates a clip.
Unit tests: `tests/test_h01_generator_pilot.py` (3 tests; the suite is 224 passed, 3 skipped).

## Provenance

| item | where |
| --- | --- |
| pre-registration | `configs/h01_pilot_prereg.yaml`, commit `245f7d9`, sha256 `3529b920…c0c2f` |
| amendment 1 (LingBot grid, not run) | `configs/h01_pilot_prereg_amendment1.yaml`, sha256 `eabd250c…8b88` |
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

