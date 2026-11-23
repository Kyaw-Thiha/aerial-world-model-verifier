# H04 — SolarWM-5B on continuously changing commanded attitude

Track H4, 2026-11-22. Branch `track-h3-reader-validity`, based on `main` (`41d1dad`).
Pre-registration `configs/h04_solarwm_continuous_prereg.yaml`, amendments 1 and 2; the app is
a copy of H02's (`track-h2-generator-coverage`, `97fda68`), read read-only.

**Decision.** On 78 arm-A clips SolarWM-5B is accepted by the frozen coherence tier 31/51 times it is testable (S1 passes), while the level and mirrored arms are flagged 100 % (64/64) and 100 % (42/42) (S2 passes); best-of-3 does not rescue it (median reduction 0.04 deg, S3 fails).  The verifier can say yes: with continuously changing commands this generator produces clips that pass the frozen tiers, which the earlier held-bank grid did not.

## Plain-language summary

1. **Arm A (the physically consistent arm)**: 78 clips, 51 testable
   (27 abstained), **31 accepted**; the coherence tier
   flags 39 % (20/51) of the testable clips. The median depicted/required tilt ratio
   on the testable clips is 0.95, and
   35/51 testable clips have a ratio in
   [0.5, 1.5]. Of the 20 rejected testable clips, **9 are weak-signal** (the required tilt
   exceeds 10° on under 30 % of frames), **0 are attributable to yaw sensitivity alone**, and
   **11 are genuine failures** — 4 roll-shape, 3 pitch-axis (the reader's least certain axis)
   and 4 sign-tier marginal (§1b).
2. **Arm B (level camera)**: 64 testable, 0 accepted, coherence
   flag 100 % (64/64). **Arm C (mirrored roll)**: 42 testable,
   0 accepted, coherence flag 100 % (42/42). S2
   passes.
3. **Sampling does not rescue the failures**: across three validation draws the median
   residual standard deviation is 0.18 deg (median range
   0.41); the oracle best-of-3 reduction is
   0.04 deg and 1 of 22
   paths change from flagged to accepted. S3 fails.
4. **The null behaves; the yaw control does not.** The same-arm null flags 6/66 draw pairs
   (9 %). The yaw control, however, flags 10 of its 11 testable clips against 39 % in arm A:
   a world-yaw rotation leaves the tilt unchanged, so this is an artefact of the reader's
   sensitivity to the warped view, not a property of the generator (see §6).


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

Of the 78 arm-A clips 51 are testable under the frozen abstention
rules; abstentions: signal_frames 6, signal_fraction 6, gain_ci_half_width 20, confidence_gate 10.
**31 are accepted** by the coherence tier, i.e. 61 % (31/51)
of the testable clips and 40 % (31/78) of all arm-A clips. The R2 band
flags 65 % (33/51) and the median-residual tier is not applied as a decision
(median 3.95 deg); the median axis gains are gx 0.91, gy 0.66. S1 passes.


### 1b. Why the 20 rejected arm-A clips were rejected

Of the 51 testable arm-A clips, 31 are accepted and **20 rejected**.  Each rejection is
assigned one explanation by the rule below, fixed before the table was read (post hoc and
labelled exploratory; it changes no acceptance decision):

1. **weak signal** — the frozen signal rules barely passed: the required tilt exceeds 10° on
   fewer than 30 % of frames (`signal_fraction < 0.30`) or the excitation is under 4°;
2. **yaw sensitivity** — not weak, and the same path's yaw-warped draw-0 clip is *not*
   flagged while this clip is, i.e. a tilt-preserving +20° view change alone clears it;
3. **genuine failure** — otherwise, sub-classified by the per-axis correlations:
   `roll_shape` (r_y ≤ 0.30), `pitch_axis` (r_y > 0.30, r_x ≤ 0.50 — the axis the reader's
   fixed 90° focal prior makes uncertain), `sign_tier` (flagged by the sign sub-tier only
   with r_min ≥ 0.6).

**Result: weak signal 9/20, yaw sensitivity 0/20,
genuine failure 11/20** (roll shape 4,
pitch axis 3, sign tier 4).

The yaw column is the measured control on the same path: the +20° warp flags every one of
these paths, so it widens the uncertainty band but does **not** explain any single rejection;
no threshold was changed on its account.

| clip | family/peak | excitation | signal frac. | r_x | r_y | flagged by | yaw counterpart | class |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `swap_arc_10_F5_A_flatness_g0` | arc/10° | 7.33° | 0.32 | 0.81 | -0.22 | corr | flagged | genuine_failure (roll_shape) |
| `main_arc_20_A_flatness_g0` | arc/20° | 9.08° | 0.52 | 0.32 | 0.79 | corrsign | flagged | genuine_failure (pitch_axis) |
| `main_arc_20_A_flatness_g1` | arc/20° | 9.08° | 0.52 | 0.32 | 0.78 | corrsign | flagged | genuine_failure (pitch_axis) |
| `main_arc_20_A_flatness_g2` | arc/20° | 9.15° | 0.52 | 0.45 | 0.66 | sign | flagged | genuine_failure (pitch_axis) |
| `main_climb_turn_10_A_flatness_g0` | climb_turn/10° | 7.37° | 0.32 | 0.83 | -0.11 | corr | flagged | genuine_failure (roll_shape) |
| `main_climb_turn_10_A_flatness_g1` | climb_turn/10° | 7.37° | 0.32 | 0.83 | -0.08 | corr | flagged | genuine_failure (roll_shape) |
| `main_climb_turn_10_A_flatness_g2` | climb_turn/10° | 7.37° | 0.32 | 0.81 | -0.04 | corr | flagged | genuine_failure (roll_shape) |
| `main_fig8_5_A_flatness_g0` | fig8/5° | 6.88° | 0.22 | 0.96 | 0.45 | sign | flagged | weak_signal |
| `main_fig8_5_A_flatness_g1` | fig8/5° | 6.88° | 0.22 | 0.96 | 0.46 | sign | flagged | weak_signal |
| `main_fig8_5_A_flatness_g2` | fig8/5° | 6.88° | 0.22 | 0.96 | 0.45 | sign | flagged | weak_signal |
| `main_fig8_8_A_flatness_g0` | fig8/8° | 6.88° | 0.25 | 0.97 | 0.91 | sign | flagged | weak_signal |
| `swap_s_10_F4_A_flatness_g0` | s/10° | 6.87° | 0.47 | 0.83 | 0.85 | sign | flagged | genuine_failure (sign_tier) |
| `main_s_5_A_flatness_g0` | s/5° | 6.88° | 0.25 | 0.94 | 0.78 | sign | flagged | weak_signal |
| `swap_slalom_10_F4_A_flatness_g0` | slalom/10° | 8.06° | 0.30 | 0.90 | 0.81 | sign | flagged | weak_signal |
| `main_slalom_12_A_flatness_g0` | slalom/12° | 8.63° | 0.35 | 0.76 | 0.75 | sign | flagged | genuine_failure (sign_tier) |
| `main_slalom_12_A_flatness_g1` | slalom/12° | 8.63° | 0.35 | 0.74 | 0.78 | sign | flagged | genuine_failure (sign_tier) |
| `main_slalom_12_A_flatness_g2` | slalom/12° | 8.63° | 0.35 | 0.76 | 0.75 | sign | flagged | genuine_failure (sign_tier) |
| `main_slalom_5_A_flatness_g0` | slalom/5° | 7.22° | 0.25 | 0.79 | 0.26 | corr | flagged | weak_signal |
| `main_slalom_5_A_flatness_g1` | slalom/5° | 7.27° | 0.25 | 0.76 | 0.16 | corrsign | flagged | weak_signal |
| `main_slalom_5_A_flatness_g2` | slalom/5° | 7.24° | 0.25 | 0.94 | -0.30 | corrsign | flagged | weak_signal |

Two mechanisms deserve a note.  The four `sign_tier` clips have r_min 0.74–0.91 — the shape
is right — but the sign sub-tier flags them because its 1e-4 dead-band admits almost every
frame, so a small lag or offset pushes the agreement to chance (0.49–0.53).  And the three
`pitch_axis` clips (`arc_20`) follow the roll axis (r_y 0.66–0.79) and fail only on the
forward axis, which is the one the reader's assumed focal length makes least certain.  Both
are reported as mechanisms, not as generator failures.

### 2. Arms B and C

Arm B: 64 testable of 78, coherence flag 100 % (64/64),
accepted 0. Arm C: 42 testable of 78, coherence
flag 100 % (42/42), accepted 0. S2 passes.
The R2 band flags 100 % (64/64) of B and 100 % (42/42) of C.


### 3. Where it accepts, how large is the depicted tilt?

On the testable arm-A clips the depicted tilt is a median 0.95 times the required tilt
(35/51 clips inside [0.5, 1.5]); the
coherence tier is scale-free, so a clip can be accepted while under-banking. The gain
tier, which is not scale-free, flags 65 % (33/51) of the same clips.


### 4. Sample variance and best-of-3

Across the three validation draws the median per-path standard deviation of the clip
median residual is 0.18 deg (median range 0.41); the oracle
best-of-3 reduces the residual by a median 0.04 deg
(path-clustered interval [0.00, 0.18]), and 1/22 paths change class. S3 fails.
The per-path spread is in `results/h04/scored_main.json`.


### 5. Scenes

* **ANWM city view 1** (F1): 36 clips, 25 testable, 4 accepted.
* **ANWM city view 2** (F2): 36 clips, 27 testable, 9 accepted.
* **ANWM city view 3** (F3): 36 clips, 36 testable, 9 accepted.
* **urban_checker** (F4, simulated): 48 clips, 27 testable, **0 accepted**.
* **industrial** (F5, simulated): 39 clips, 32 testable, 9 accepted.
* **forest_ridge** (F6, simulated): 39 clips, 10 testable, **0 accepted**.

Arm A by path family (clips / testable / accepted): arc 15/11/7, S-turn 18/12/10,
slalom 18/17/10, figure-eight 15/8/4, climb-turn 6/3/0, accelerate-brake 6/0/0.  The
pitch-only accelerate-brake paths abstain by construction under the two-axis rule.  The two
simulated scenes accept nothing, and they do not have the same explanation: `forest_ridge` is
the scene the reader is known to misread most (I02: 7-10 deg of up-vector bias), while
`urban_checker` is the scene it reads *best* (0.5-1.5 deg) — so the zero acceptance there is
a property of what the generator produced on that first frame, not of the reader.  The three
ANWM city views accept 4-9 clips each and the industrial scene 9.


### 6. Controls: yaw and null

**Yaw control** (arm A draw 0, every frame warped by a +20 deg rotation about world up):
22 clips, **11 testable, 10 of them flagged by the coherence tier (91 %), 1 accepted**. Arm A
itself is flagged 39 % of the time, so the control does **not** sit where it should.

The control is valid as a transform: read on the same frames, the warped clips give an
estimated up direction 2.3-4.0 deg from the unwarped one (roll difference 1.4-2.6 deg,
interquartile range 0.8-3.7), i.e. the warp preserves the tilt to within the reader's
sensitivity and does not roll the image. The flag rate therefore says that a 2-4 deg
perturbation of the depicted series is enough to break the coherence tier on clips whose
required signal is weak — the same clips that make up most of arm A's abstentions and a good
part of its rejections. Two consequences are reported rather than fixed: arm A's 61 %
acceptance is more likely an **underestimate** than an overestimate of what a cleaner reader
would give, and the arms B and C rejections (100 %) are far outside that perturbation and
remain robust. The per-clip pairs are in `results/h04/scored_main.json`.

**Same-arm null** (one draw's depicted lean in place of the required series): 6/66 draw pairs flagged, i.e. the statistic does
not manufacture coherence from noise alone.


### 7. Gated extension: 14B pilot, LTX-22B and H3-33B

The extension was re-scoped after inspecting the releases.  Only the 14B was attempted (a
35-clip pilot, amendment 3, which produced no clip — see §8); the other two are estimated from
their published weights and configs and were **not downloaded or run**.

| checkpoint | release (inspected) | download | steps / resolution | per clip (est. unless measured) | 35 clips | 200 clips |
| --- | --- | --- | --- | --- | --- | --- |
| 5B (campaign) | Stage2 SGF, causal | 61 GB | 4 / 480x864 | **28 s (measured)** | ~0.3 GPU-h, ~US$0.7 | ~1.5 GPU-h, ~US$3.9 |
| 14B (pilot) | `bid-stage0p5-81f` + `base-high`, I2V-A14B, bidirectional flow matching | **183 GB** | 50 / 480x832, **min 2 ranks (sp=2)** | not measurable here: the released path fails at DP 1 (dtype contract) and its trained topology is 16 ranks; **~20 min/clip at 8 GPUs** as a bound | ~US$230 (8 GPUs) | ~US$1,300 (8 GPUs) |
| LTX-22B | `ltx-22B-bid-stage0p5-153f` (LoRA r384) + LTX-2.5-22B, video-only | 73.7 GB | 30 / 512x768, 153 frames | ~20 min | ~12 GPU-h, ~US$29 | ~92 GPU-h, ~US$230 |
| H3-33B | `h3-33B-bid-stage0p5-158f` + MiniMax-H3, packed | 169 GB | 4 / 768x1344, 158 frames, **8 GPUs (SP 4)** | ~3-5 min wall on 8 GPUs | ~10 GPU-h, ~US$40-60 | ~60 GPU-h, ~US$400 |

The 14B is **not a scaling point on the 5B curve**: different stage (stage0.5 flow matching with
50 bidirectional steps vs Stage2 self-forcing with 4 causal steps), different base family
(Wan2.2 I2V-A14B vs TI2V-5B) and a different resolution (832 vs 864 wide).  The LTX-22B and
H3-33B rows are extrapolations from parameter counts, frame counts, pixel counts and steps, with
the 5B measured rate as the anchor; they are order-of-magnitude numbers, not quotes.

### 8. The 14B pilot

The pilot was attempted and **no 14B clip exists**; the denominators below are 0/20 and 0/15.
What was measured, in order:

* the release is **183.4 GB** (81 files): `SolarWM-14B-base-high` (the Wan2.2 I2V-A14B
  high-noise expert + T5) and `SolarWM-14B-bid-stage0p5-81f` (a 114.3 GB consolidated
  stage0.5 checkpoint). Download took 13 minutes on 16 CPU cores;
* the config contracts pin the topology. The A14B backend refused four configurations in turn
  (`raw_world_size must be divisible by sp_size`; `wan22_i2v_a14b requires
  sequence_parallel_size=2`; `sample_sharding must be logical_dp`; `global_batch_size must
  equal DP*micro*grad_accum`), which fixes **two ranks minimum** and makes the pilot a 2-GPU
  job at the smallest;
* at two ranks (2x A100-80GB, `torchrun --nproc-per-node=2`) the model loaded in about
  14.5 minutes and then failed **inside the generation pass** with `Input type (float) and bias
  type (c10::BFloat16) should be the same` on both ranks — a dtype contract in the release's
  stage0.5 camera/i2v conditioning, which the release exercises at its trained topology
  (world size 16, DP 8). The same failure would apply to any DP-1 run;
* a single-clip probe at four GPUs with the required `sp_size=2` (DP 2) ran 26.6 minutes and
  failed on the same distributed contract (`solar` reported the `sequence_parallel_size=2`
  requirement against a config written with `sp_size=2`), i.e. the release's config path, not
  ours, decides that field; it was stopped there rather than patched further. The two-rank
  dtype failure is the decisive measurement: it happens after a full model load, inside the
  generation pass, and no 14B frame was ever produced.

**Cost of the attempt**: download 0.22 CPU-hours (~US$0.18), the four config failures
(~US$0.13), the two-rank timing attempt (~US$1.67) and the probe. Nothing was scored, because
nothing was generated.

**What this means.** The 14B route is released as *training* code plus a stage0.5 checkpoint;
its inference path is written for the 16-rank training topology and does not run, as released,
at the small topologies this project can afford. Running it at its trained topology would cost
roughly 8 GPUs x ~20 min per clip, i.e. **about US$6.7 per clip** at A100 list prices and
US$230 for the 35-clip pilot — far outside the remaining allowance and outside the US$25
condition in the pre-registration. **The pilot is therefore reported as not runnable here, not
as a model result.** The 5B campaign stands on its own; the LTX-22B and H3-33B rows in §7
carry the same caveat (H3's released inference config is 8 GPUs with sequence parallelism 4).



## What this supports

* **The frozen tiers behave on a physical generator**: a synthetic perfect generator
  (depicted lean equal to the required lean plus 1 deg of noise) is testable and
  accepted on all 20 turning paths of the pre-generation design check (G1 0.98-1.04,
  coherence r_min 0.88-0.99; amendment 2) and in `tests/test_h04_score.py`, so the
  campaign's rejections are not a broken statistic.
* **Arm A acceptance is 61 % (31/51)** of the testable clips;
  the level and mirrored arms are flagged 100 % (64/64) and 100 % (42/42).
* **Where it accepts, the depicted tilt is 0.95 times
  the required tilt**, which separates 'the verifier can say yes' from 'the generator is
  physically consistent'.
* **The same-arm null sits where it should** (6/66 draw pairs flagged, 9 %), so the
  statistic does not manufacture coherence from noise alone.  The **yaw control does not**
  (10/11 testable flagged): a 2-4 deg perturbation of the depicted series is enough to break
  the tier on weak-signal clips, which makes arm A's acceptance an underestimate rather than
  an overestimate.


## What this does not show

* **It is a commanded-path application, not the pixel-path verdict.** The required side is the
  path the generator was told to fly; no path is recovered from the pixels.  This grants the
  generator its best case and does not test the full verifier, whose pixel-path tier is parked
  (F01-F06).
* **One checkpoint, one resolution, one prompt.** 5B, 480x864, 81 frames, four sampling steps;
  the 14B, LTX-22B and H3-33B routes are estimated, not run.
* **Three draws are not seeds.** The release exposes a single `noise_seed`; the three
  generations measure sampling variance, not a deployable seed knob.
* **The 14B pilot did not produce a clip.** Its release is a stage0.5 *training* stack whose
  inference path is written for a 16-rank topology; at the two ranks its config contracts
  allow, it fails in the generation pass, and at its trained topology it costs about US$6.7
  per clip. So nothing here says how a larger SolarWM would behave.
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
* **Wall-clock and spend.** Generation and reading ran on Modal A100-80GB and A10 containers; the measured spend and wasted attempts are in `docs/COST_LEDGER.md`.
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
| clips | Modal `awmv-h3-results:/solarwm/main/<clip>/`; R2 `track-h3/clips/solarwm-main/`, 234 clips (mp4 + release manifest), 179 MB, every sha256 re-read from R2 and matching, manifest at `track-h3/manifests/main.json` (local copy `results/h04/r2_manifest.json`) |
| reader rows | `results/h04/depicted_main/depicted.jsonl` |
| scores | `results/h04/scored_main.json`, `scored_main_rows.json` |
| model code | `Junchao-cs/SolarWM` `bef02908` |
| cost | `docs/COST_LEDGER.md` |
