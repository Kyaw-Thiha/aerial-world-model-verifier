# H04 — SolarWM-5B on continuously changing commanded attitude

Track H4, 2026-11-22. Branch `track-h3-reader-validity`, based on `main` (`41d1dad`).
Pre-registration `configs/h04_solarwm_continuous_prereg.yaml`, amendments 1 and 2; the app is
a copy of H02's (`track-h2-generator-coverage`, `97fda68`), read read-only.

**Decision.** @@DECISION@@

## Plain-language summary

@@SUMMARY@@

## Question and pre-registered rules

The verifier is built to say *no* to a flight that accelerates without tilting. This campaign
asks the other half of the question: **can it say yes?** SolarWM-5B is the one released
path-controlled model that follows a *changing* attitude (H02's ramps), so the campaign gives
it paths whose commanded attitude changes continuously over three seconds, generates the same
path three ways, and asks whether the frozen verifier accepts the physically consistent arm
and rejects the two deliberately wrong ones.

* **Arm A (flatness)**: the camera attitude is the flatness attitude of the commanded path,
  `z_B = unit(a + g e3 + k_d v)`, `k_d = 0.295`; the physically consistent generator.
* **Arm B (level)**: same translations and heading, camera held level — no attitude at all.
* **Arm C (mirrored roll)**: arm A with the roll negated — the mirror defect.

Each (path, arm) is generated three times.  The release exposes one `noise_seed`, so the three
are **validation draws, not a seed knob**; sample variance is measured and best-of-3 is an
oracle over draws.

Pre-registered success criteria:

* **S1 — the verifier can say yes**: at least 30 % of the *testable* arm-A clips are accepted
  by the coherence tier, and at least 10 clips in absolute terms.  Unassessable if fewer than
  10 arm-A clips are testable.
* **S2 — it rejects the wrong arms**: the coherence-tier flag rate is at least 80 % of the
  testable arm-B clips and at least 80 % of the testable arm-C clips.
* **S3 — sampling does not rescue it**: the oracle best-of-3 residual reduction is at least
  2 deg and at least 20 % of paths change from flagged to accepted (carried over from
  H01/H02 unchanged).

Beside every rate: the **yaw control** (arm A draw 0, each frame rotated about world up by
20 deg, which leaves the tilt unchanged) and the **same-arm null** (one draw's depicted lean in
place of the required series, no commanded path).

## Setup

**Model.** `junchaoh-cs/SolarWM-Wan2.2-5B`, `SolarWM-5B-sgf-stage2-81f` on `SolarWM-5B-base`,
Stage2 four-step SGF, 81 frames at 480x864, one A100-80GB, the same fixed aerial-scene prompt
as H02.  The code is `Junchao-cs/SolarWM` at `bef02908`; the checkpoint is Apache-2.0 and was
already accepted by the user's account under H02.

**Convention gate.** Before the campaign, two pure-roll probes (linear 0 -> +20 and 0 -> -20
across 81 frames) were generated and read with the frozen reader.  The +20 probe moves from
3.9 deg of depicted tilt in the first ten frames to 21.1 deg in the last ten, with the lean
y-component going from -0.05 to -0.35; the -20 probe moves from 1.1 deg to 16.8 deg with the
opposite lean sign (+0.003 to +0.283).  **The gate passes** and the pose convention is H02's.

**Paths.** 22 closed-form paths on a 1 kHz grid over the 5.0625 s clip, active window the
middle 3.0 s with a C2 ramp-in/out.  The design is written as a commanded bank profile
`phi(t)` and integrated, `omega = -(g/v) tan(phi)`, so the attitude is smooth at a known peak:

| family | profile over the 3 s window | peaks |
| --- | --- | --- |
| `arc` | 0.5 cycle (one bank lobe) | 5, 8, 10, 20 deg |
| `s` | 1 cycle (S-turn) | 5, 8, 10, 12, 20 |
| `slalom` | 1.5 cycles (three lobes) | 5, 8, 10, 12, 20 |
| `fig8` | 1 cycle, 0.6 : 1 lobes (asymmetric S) | 5, 8, 10, 20 |
| `climb_turn` | `arc` bank plus a vertical bell | 10, 20 |
| `accel_brake` | speed pulse, pitch only | 10, 12 |

Every heading-family path also carries a 35 % forward-speed modulation over the same window
(amendment 2): without it the frozen design matrix is exactly rank-deficient and the pitch
axis has no signal, so a *perfect* generator would abstain for conditioning and be flagged by
`r_min`.  With it the frozen condition number is 20, the excitation is 6.8-11.6 deg, and a
synthetic perfect generator gives G1 = 0.98-1.04 and coherence `r_min` 0.88-0.99.  All 22 paths
pass the design checks: achieved peak within 0.2 deg of the request, peak thrust-to-weight at
most 1.05 g, maximum first-frame-relative translation 14.0 m, maximum attitude change 4.4 deg
per frame at 16 fps.

**First frames, six scenes.** Three ANWM release frames (`anwm_city`, three views) and the
first frames of three Track I renders (`urban_checker`, `industrial`, `forest_ridge`).  Path
`i` uses frame `F(1 + i mod 6)`, fixed in the pre-registration.  The exploratory scene swap
regenerates four paths (one per family, peak 10) on the three synthetic scenes, one draw.
234 clips: 22 x 3 arms x 3 draws, plus 36 scene-swap, plus the two probes.

**Scoring.** The frozen reader (GeoCalib `1dedb29`, 224 px square pinhole, f = 112 px) on all
81 frames; the required side is the **commanded** path differentiated at 1 kHz with the
label-free window rule `0.5 / f95` (primary; the frozen 0.37 s is the secondary), so the
generator is granted its best case — this is not a path recovered from pixels.  Tiers:
coherence (primary, thresholds from `configs/coherence_frozen.yaml`), the R2 axis-gain band,
and the 6.46 deg median residual.  Abstention is the frozen rule set in its frozen order.
The R2 band was calibrated with a mocap required side and is reported as a **calibration
transfer**, never as a standalone detector; the coherence tier is scale-free and is the
primary.  Possible lag comes from the frozen gain fit's search (+-150 ms, 25 ms steps).

## Results

### 1. Arm A: what fraction does the verifier accept?

@@RESULTS_A@@

### 2. Arms B and C

@@RESULTS_BC@@

### 3. Where it accepts, how large is the depicted tilt?

@@RESULTS_TILT@@

### 4. Sample variance and best-of-3

@@RESULTS_SAMPLING@@

### 5. Scenes

@@RESULTS_SCENE@@

### 6. Controls: yaw and null

@@RESULTS_CONTROLS@@

### 7. Gated extension: 14B, LTX-22B and H3-33B

@@RESULTS_EXTENSION@@

## What this supports

@@SUPPORTS@@

## What this does not show

* **It is a commanded-path application, not the pixel-path verdict.** The required side is the
  path the generator was told to fly; no path is recovered from the pixels.  This grants the
  generator its best case and does not test the full verifier, whose pixel-path tier is parked
  (F01-F06).
* **One checkpoint, one resolution, one prompt.** 5B, 480x864, 81 frames, four sampling steps;
  the 14B, LTX-22B and H3-33B routes are estimated, not run.
* **Three draws are not seeds.** The release exposes a single `noise_seed`; the three
  generations measure sampling variance, not a deployable seed knob.
* **The scenes are six first frames.** Three are ANWM's own city release, three are Blender
  renders that are out of distribution for SolarWM; scene is a factor with a few levels, not a
  sampled distribution.
* **The coherence tier needs two-axis signal.** The pitch-only `accel_brake` paths abstain
  under the frozen rules by construction, and are reported as abstentions rather than removed.

## Caveats

* **The first path design could not be scored, and the first two serving attempts failed.**
  The amendment-1 paths were exactly rank-deficient for the frozen statistic (amendment 2);
  the first generation batch was stopped before it wrote anything; and two serving attempts
  with 40-clip and 59-clip sub-batches died inside the release's inference step after about
  fifteen minutes without writing a clip.  The campaign as reported runs in sub-batches of 12
  (the size at which the release's infer step was verified end to end), each committed to the
  volume as it completes, so a stopped container loses at most one sub-batch.  These are
  engineering facts about the serving route, not about the model, and they are in the cost
  ledger.
* **Wall-clock and spend.** @@WALLCLOCK@@
* **The R2 band is a calibration transfer**, as stated above: its centre came from real TII
  clips with a mocap required side.  It is reported with the raw axis-gain point estimates and
  never used alone to accept or reject a generated clip.
* **The coherence tier is scale-free by construction.** A generator that depicts the right
  *shape* at a third of the required magnitude passes it.  The depicted/required ratio and the
  median-residual tier are reported beside every acceptance rate for exactly that reason.

## Reproduction

```
set -a; . ./.env; set +a
PYTHONPATH=. pixi run -e gpu python experiments/h04_paths.py            # 22 verified paths
PYTHONPATH=. pixi run -e gpu python experiments/h04_solarwm_inputs.py   # 236 samples
pixi run modal deploy remote/h04_solarwm.py
pixi run modal run remote/h04_solarwm.py::main --mode read --ids "<ids>" --tag main
pixi run modal run remote/h04_solarwm.py::main --mode read --ids probe_p20,probe_m20 --tag probes
pixi run modal run remote/h04_post.py::main --mode read --tag main       # reader, all 81 frames
pixi run modal run remote/h04_post.py::main --mode yaw --tag main        # yaw control
PYTHONPATH=. pixi run -e gpu python experiments/h04_score.py
pixi run modal run remote/h04_publish.py::main --tag main                # R2 + sha256 manifest
```

## Provenance

| item | where |
| --- | --- |
| pre-registration | `configs/h04_solarwm_continuous_prereg.yaml`, commit `c629016` |
| amendment 1 (path realisation) | `configs/h04_solarwm_continuous_prereg_amend1.yaml`, commit `141ac4d` |
| amendment 2 (speed modulation) | `configs/h04_solarwm_continuous_prereg_amend2.yaml`, commit `dfd425b` |
| paths and checks | `results/h04/paths/paths_summary.json` (and `paths.npz`, gitignored) |
| inputs | `data/h04_solarwm_inputs/` (236 samples, gitignored; manifest committed copy in `results/h04/inputs_manifest.json`) |
| clips | Modal `awmv-h3-results:/solarwm/main/<clip>/`, R2 `track-h3/clips/solarwm-main/` with a sha256 manifest at `track-h3/manifests/main.json` |
| reader rows | `results/h04/depicted_main/depicted.jsonl` |
| scores | `results/h04/scored_main.json`, `scored_main_rows.json` |
| model code | `Junchao-cs/SolarWM` `bef02908` |
| cost | `docs/COST_LEDGER.md` |
