# H02 — Generator coverage with path control: SolarWM, LingBot full precision, inventory

Track H2, 2026-11-21. Branch `track-h2-generator-coverage`, based on `main` (`41d1dad`),
which does not yet contain the first-round tracks. Depends on the read-only branch
`track-h-generator-pilot` (H01) for the reader, the five first frames, the designed paths and
the LingBot comparison; no H01 file was modified. Pre-registration:
`configs/h02_pilot_prereg.yaml` (commit `5c6e5c6`), committed before any H2 clip was generated
or scored. Spend ceiling US$90; actual spend in `docs/COST_LEDGER.md`.

## Plain-language summary

**Inventory.** Only one released model both takes a camera trajectory and is practical here:
SolarWM-Wan2.2-5B. Aero-World (IMU control, trained on UZH/TII) has no public code or weights;
CT-1 predicts a camera trajectory rather than video; CameraCtrl, MotionCtrl and ReCamMaster
are older, smaller-backbone camera controllers and were not run.

**SolarWM follows changing attitude, not a held one.** The five probes pass decisively: level
1.4 deg, +20 ramp ending 22.4 deg, -20 ending 18.2 deg with opposite lean, +60 ending 45.8 deg.
The designed grid is weaker: only cruise shows a flatness tilt at least half the injected
magnitude; turn-10 and turn-20 hold a 13-21 deg attitude and the reader sees 3.8-7.1 deg. The
pre-registered roll criterion fails (1/5 paths by tilt, 1/5 by mirror sign). The frozen verdict
calls 5 of 15 testable flatness clips feasible — all slalom-10 — and rejects every testable
turn-10/20 clip. That 5/15 uses H02's absolute 12-frame rule; scaling H01's 12/16 rule to 81
frames leaves only the ten turn clips testable, all infeasible. The generator follows the *derivative* of the camera path, not the attitude
the path requires. Seed sampling moves the depicted tilt much more than ANWM did (median range
2.6 deg), but best-of-3 cuts the residual by only about 1 deg and flips one of ten paths, so it
does not meet the pre-registered sampling criterion either.

**Full-precision LingBot.** The reader still shows no coherent roll response in bf16
(level 4.56 deg; +20 ramp 3.87; -20 4.91; triangular 4.29; 60 ramp 4.06), although the bf16
frames look visibly more tilted than H01's NF4 frames. The disagreement between the eye and
the pre-registered gravity reader is recorded, not resolved; quantisation is not established
as the cause of H01's failure.

## Question and pre-registered rules

The scope is **only world models that take a path or camera pose as control**, because the
verifier's question is whether a generator depicts the attitude its commanded path requires.
Three questions, fixed in the pre-registration:

1. **SolarWM.** Does the 5B camera-controlled model depict the injected attitude? A roll-ramp
   convention gate must pass first; then the same translation path is generated with a level
   camera, the flatness-derived attitude, and a mirrored roll. Depicted tilt is read with the
   existing reader and, where testable, the frozen commanded-path verdict is applied.
2. **Seed variance.** Ten designed paths x three seeds, same statistic as H01; does best-of-3
   sampling reduce the median residual by at least 2 deg and flip at least 20 % of paths?
3. **Was H01's LingBot roll blindness a quantisation artefact?** Repeat the five roll probes
   with bf16 weights and compare beside H01's NF4 numbers.

## Inventory (no spend)

Verified from primary sources on 2026-11-21: model cards, GitHub, papers. "Control" states the
exact pose parameterization and whether roll is carried. Costs marked *estimate* were not
measured in this pilot.

| model | control (parameterization, roll) | weights | licence | gating | VRAM (expected) | cost/clip |
| --- | --- | --- | --- | --- | --- | --- |
| SolarWM-Wan2.2-5B | frame-aligned **absolute OpenCV c2w** + per-frame intrinsics; model rebases to first frame and injects first-frame-relative pose through fused-PRoPE; **roll carried** | `junchaoh-cs/SolarWM-Wan2.2-5B` (base + Stage2 81f) | code Apache-2.0; checkpoint `LICENSE.txt` Apache-2.0, `NOTICE.md` says modified Wan2.2 | gated (auto); account accepted 2026-11-21 | 80 GB (A100, measured below) | measured below |
| SolarWM-Wan2.2-14B | same shared contract; roll carried | `junchaoh-cs/SolarWM-Wan2.2-14B` | code Apache-2.0; per-directory notices; gated | accepted | 80+ GB | estimate only if 5B follows roll |
| SolarWM-LTX-2.5-22B | same contract; video-only route; roll carried | `junchaoh-cs/SolarWM-LTX-22B` | per-directory notices; gated | auto | not checked | not estimated |
| SolarWM-H3-33B | same contract, H3 packed interface; roll carried | `junchaoh-cs/SolarWM-H3-33B` | per-directory notices; gated | auto | 8 Blackwell GPUs for the realtime demo | not estimated |
| Aero-World | **30 Hz IMU** (linear acceleration + angular velocity), not a camera path; roll carried as angular velocity | **none found** | paper CC BY 4.0 | no public code or weights (GitHub + HF search) | n/a (CogVideoX-1.5-5B) | not measurable |
| CameraCtrl | c2w extrinsics + intrinsics -> Plücker rays; AnimateDiff-V3 / SVD; roll carried | `hehao13/CameraCtrl` (3 GB) | weights card MIT; code Apache-2.0 | ungated | 24 GB | ~US$0.01 (16 frames; estimate) |
| MotionCtrl | camera RT trajectory (relative) and/or object trajectory; SVD/VideoCrafter; roll carried | `TencentARC/MotionCtrl` (22 GB) | Apache-2.0 | ungated | 24 GB | ~US$0.02 (estimate) |
| ReCamMaster | source video + target camera trajectory (novel-view re-rendering), **not image-only**; roll carried | `KlingTeam/ReCamMaster-Wan2.1` (2 GB) | weights Apache-2.0; code MIT | ungated | 24-40 GB | ~US$0.02 (Wan2.1-1.3B, 81 frames; estimate) |
| CT-1 | predicts a camera trajectory from image + text intent for another video model; **not a video generator** | no public HF checkpoint located | MIT | no public weights located | ~30 GB fp32 (README) | not a generator |

Excluded by the scope rule: AirScape (text-intention), Cosmos-Predict2.5 (no camera-control
checkpoint found), Wan 2.2 and HunyuanVideo (no path/pose input).

The only released model that both **takes a path/pose** and is practical for this pilot is
SolarWM-5B; Aero-World has no public artefacts, CT-1 does not generate video, and
CameraCtrl/MotionCtrl/ReCamMaster are older, smaller-backbone camera controllers that could be
used as secondary coverage but were not run in this pilot.

## Setup

**SolarWM-5B.** `SolarWM-5B-base` + `SolarWM-5B-sgf-stage2-81f` (Apache-2.0, gated and
accepted), Stage2 four-step SGF inference, 81 frames at 480x864, one A100-80GB. The model takes
absolute OpenCV c2w + per-frame intrinsics and rebases to the first frame; the arm is realized
as a bank-in from level over the first 12 of 81 frames, then the path-required attitude, and
path translations are uniformly scaled to at most 16 m to satisfy the released camera guards
(amendment 1; rotations unchanged). The five ANWM first frames are padded from 512x512 to
864x480 with edge replication, matching the H01 convention. Prompt is the same fixed aerial
scene description for every sample.

**LingBot full precision.** `robbyant/lingbot-world-fast` (74 GB, bf16, four-step; Apache-2.0,
Hugging Face/ModelScope, upstream repo no longer actively maintained) with the VAE and T5
encoder from `robbyant/lingbot-world-base-cam`, on one A100-80GB. Same five probe
poses as H01 (level, linear +/-20, triangular 0->20->0, linear 0->60), same first frame,
49 input frames (45 decoded). The upstream pipeline's multi-clip offload bug is fixed explicitly by moving the
transformer and VAE back to the GPU before each clip.

**Population and statistics.** The reader, required-tilt expression and 6.46 deg threshold are
H01's. Testability follows H02's pre-registration (`min_frames_required_ge_10_deg: 12`), which
is absolute rather than H01's 12 of 16; the mismatch and its sensitivity are reported below.
The five SolarWM probes are read on all 81 frames and the LingBot probes on 45; the grid and
seed clips are read every second frame (41) to bound the reader cost. No 95 % interval is
claimed: the clusters are five paths/first frames (ten for the seed set), below the ~10-cluster
rule, so every rate is descriptive with its denominator. The SolarWM grid is 5 frames x 5 paths x 3 arms = 75
clips; the seed set is 10 designed paths x 3 seeds (flatness arm). Intervals are
path-clustered (5,000 draws, `numpy.random.default_rng(1)`); with fewer than 10 clusters they
are descriptive. No threshold or success criterion changed after any outcome.

## Results

### 1. SolarWM-5B

**The release runs and the roll gate passes.** Five probes, 81 frames, 480x864, four sampling
steps:

| probe | injected roll | depicted median | first-10 | last-10 | lean cosine |
| --- | --- | --- | --- | --- | --- |
| level | 0 | 1.44 | 2.28 | 0.48 | 0.00 |
| +20 ramp | 0 -> +20 | 9.94 | 3.58 | 22.37 | 0.878 |
| -20 ramp | 0 -> -20 | 5.95 | 0.90 | 18.19 | 0.815 |
| triangular 0->20->0 | 0 -> 20 -> 0 | 18.84 | 4.10 | 18.23 | 0.988 |
| 0->60 ramp | 0 -> +60 | 30.75 | 4.43 | 45.76 | 0.980 |

The level baseline is 1.44 deg. The +20 ramp ends at 22.4 deg, the -20 ramp at 18.2 deg
(opposite lean), and the 60 ramp at 45.8 deg; the depicted lean direction correlates with the
injected direction (cosine 0.82-0.99) on every ramp. **The convention gate passes**: unlike
LingBot, SolarWM's depicted attitude follows a continuously changing attitude.

**The 75-clip designed grid is weaker than the ramps.** Each row is the median over five first
frames; the injected value is the path-required attitude.

| path | arm | depicted median | injected median | first-10 | last-10 | lean cosine |
| --- | --- | --- | --- | --- | --- | --- |
| cruise | level | 1.82 | 0.00 | 1.96 | 1.19 | — |
| cruise | flatness | 7.40 | 8.55 | 3.22 | 9.82 | 0.99 |
| cruise | mirrored | 6.85 | 8.55 | 3.64 | 7.06 | 0.99 |
| turn5 | level | 1.25 | 0.00 | 1.41 | 1.23 | — |
| turn5 | flatness | 4.68 | 9.83 | 6.26 | 2.20 | 0.85 |
| turn5 | mirrored | 6.51 | 9.83 | 3.95 | 5.34 | 0.96 |
| turn10 | level | 2.02 | 0.00 | 1.81 | 2.01 | — |
| turn10 | flatness | 3.82 | 12.95 | 5.22 | 4.78 | 0.14 |
| turn10 | mirrored | 8.15 | 12.95 | 5.26 | 8.92 | 0.90 |
| turn20 | level | 3.08 | 0.00 | 1.80 | 2.99 | — |
| turn20 | flatness | 7.06 | 21.19 | 4.41 | 9.14 | 0.33 |
| turn20 | mirrored | 10.14 | 21.19 | 5.07 | 15.01 | 0.20 |
| slalom10 | level | 1.11 | 0.00 | 1.76 | 1.95 | — |
| slalom10 | flatness | 3.36 | 9.03 | 2.05 | 4.49 | 0.07 |
| slalom10 | mirrored | 9.08 | 9.03 | 1.84 | 10.50 | 0.31 |

Against the pre-registered criterion the grid **fails**: only the cruise path shows a flatness
tilt at least 50 % of the injected magnitude (1 of 5 paths, required 3); the flatness lean
cosine is positive on 5 of 5 but only 0.07-0.33 on the three turning-20/10/slalom paths; and
the mirrored arm flips the horizontal lean x-sign on 1 of 5 paths (required 4). The pattern is
consistent with the model following **changes** in the pose trajectory (the probes ramp
continuously) but not **holding** a path-required attitude: the arms bank in over 12 frames,
hold for ~60, and return; the depicted tilt during the hold is only a few degrees.

**Frozen commanded-path verdict.** On the 25 flatness clips (5 paths x 5 frames): 15 testable,
**5 feasible**, 10 infeasible, 0 gate abstentions. All five slalom-10 clips are feasible (median
residual 5.8 deg); every testable turn-10 and turn-20 clip is infeasible (9.1 and 14.1 deg);
cruise and turn-5 are not testable by the 12-frame excitation rule. SolarWM therefore passes
the verdict on the slalom, whose attitude keeps changing, and fails it on the sustained
10/20-degree banks.

That split depends on how H01's excitation rule is carried to 81 frames. H02 pre-registered
an absolute minimum of 12 frames; H01 wrote 12 of 16 frames (75 %), and only 44 % of the
slalom frames carry a required tilt above 10 deg. Under the stricter 12/16 fraction (61 of
81), **only the ten turn-10/20 clips are testable and all ten are infeasible**; the five
"feasible slalom" clips become untestable. The held-bank failure is robust to the choice;
the slalom pass is not.

### 2. SolarWM seed variance

Ten designed paths x three seeds on the flatness arm (30 clips). The depicted tilt varies much
more than ANWM's: the median per-path seed standard deviation is **1.41 deg** and the median
range **2.61 deg** (worst path 9.88 deg), against 0.17/0.43 for ANWM. The oracle best-of-3
residual reduction is **1.02 deg median** (mean 0.87) and **one of ten paths** moves from
infeasible to feasible, against the pre-registered 2 deg / 20 % criterion. Sampling helps more
than it did for ANWM but still not enough to rescue designed paths generally.

| source path | seeds | depicted sd | depicted range | residual median | r1 | r3 | feasible any |
| --- | --- | --- | --- | --- | --- | --- | --- |
| C_accel_brake_10_course_0 | 3 | 1.07 | 2.06 | 2.59 | 2.48 | 1.57 | yes |
| C_accel_brake_20_course_1 | 3 | 2.85 | 5.12 | 3.29 | 3.88 | 2.57 | yes |
| C_climb_turn_10_fixed_0 | 3 | 1.35 | 2.65 | 9.38 | 9.68 | 7.90 | no |
| C_climb_turn_20_fixed_0 | 3 | 4.21 | 8.06 | 10.04 | 6.83 | 6.83 | no |
| C_cruise_ctl_course_0 | 3 | 5.13 | 9.88 | 4.45 | 4.36 | 3.22 | yes |
| C_slalom_10_fixed_0 | 3 | 1.48 | 2.56 | 6.06 | 4.20 | 4.20 | yes |
| C_slalom_10_fixed_1 | 3 | 0.09 | 0.17 | 6.05 | **6.76** | **5.56** | yes |
| C_slalom_5_course_0 | 3 | 4.48 | 8.71 | 4.40 | 3.49 | 3.49 | yes |
| C_turn_10_fixed_0 | 3 | 1.27 | 2.49 | 9.16 | 10.00 | 8.15 | no |
| C_turn_5_fixed_0 | 3 | 0.80 | 1.60 | 5.47 | 5.36 | 4.81 | yes |

The seed spread is large enough to matter for individual clips. On `C_slalom_10_fixed_1` the
best seed crosses the 6.46 deg threshold (6.76 -> 5.56), but the median reduction and the flip
rate stay below the pre-registered 2 deg / 20 % bar; the other nine paths do not change class.

Exploratory best-of-N curve (only N=3 is pre-registered): N=1 gives a 0 deg reduction and 0
flips; N=2 gives 0.34 deg median (0.56 mean) and 1 flip; N=3 gives 1.02 deg median (0.87 mean)
and 1 flip. The curve is shallow over the measured range; it does not support more seeds as a
route to the 2 deg bar.

### 3. LingBot full precision (bf16) versus H01's NF4

| probe | NF4 median (H01) | bf16 median | bf16 first-10 | bf16 last-10 | bf16 injected max |
| --- | --- | --- | --- | --- | --- |
| level | 3.21 | 4.56 | 3.04 | 5.89 | 0.0 |
| +20 ramp | 3.30 | 3.87 | 3.51 | 4.14 | 18.3 |
| -20 ramp | 3.77 | 4.91 | 2.98 | 5.17 | 18.3 |
| triangular 0->20->0 | 3.93 | 4.29 | 3.47 | 4.84 | 20.0 |
| 0->60 ramp | 4.05 | 4.06 | 3.27 | 4.73 | 55.0 |

The pre-registered reader does **not** show a coherent roll response in bf16 either: no ramp
raises the clip median by more than 0.35 deg above the 4.56 deg level control, and the largest
last-10 value (5.9 deg) is on the level clip, not tied to an injected ramp. The bf16 images look different from NF4 — the
+20 and -20 clips have a visibly oblique skyline while H01's NF4 probe frames were level —
but the independent vertical-edge check has no usable long verticals on this painterly imagery,
so the eye and the gravity reader disagree. The disagreement is recorded, not resolved; on the
pre-registered instrument, **quantisation is not established as the cause** of H01's roll
blindness. Cost: 51.7-62.9 s/clip, 55.2 GB peak VRAM, about US$0.04/clip on an A100-80GB
(US$8.8 for 200, excluding one model load).

## What this supports

* **The released path-controlled field is thin.** Of the candidates, one model has usable
  weights and camera-pose control (SolarWM-5B); Aero-World has no public artefacts, CT-1 does
  not generate video, and the older controllers are not world models. Any claim about
  "aerial world models in general" must say which single model it rests on.
* **Full precision did not overturn H01's failure in the reader.** The bf16 LingBot probes
  stay at 4-5 deg for every ramp, like NF4, so a strong quantisation effect is not established;
  the visible oblique horizons in the bf16 clips remain an unresolved discrepancy and do not
  let us rule one out either.
* **Path conditioning can work for time-varying commands.** SolarWM-5B follows a roll ramp:
  1.4 deg level baseline, 22.4 deg at the end of a +20 ramp, 18.2 deg for -20 with opposite
  lean, 45.8 deg for +60. The generator layer is not the obstacle in principle; the action
  interface and the relative-pose representation are.
* **Path-conditioned sampling is measurable end to end with the existing verifier**: the same
  reader, required-tilt expression and verdict threshold carry over to SolarWM's 480x864
  output and a different pose convention.

## What this does not show

* **SolarWM's result is one checkpoint and one resolution.** Five first frames and three
  generations per path are covered, but the 14B, LTX-22B and H3-33B routes are not run; a
  scaling estimate is only meaningful if their control path behaves like the 5B's. The
  three "seeds" are independent validation draws: the release exposes a single
  `noise_seed: 42`, the three generations differ in their pixel output, and no per-sample
  seed argument was exposed. This measures sampling variance, not a controlled seed knob.
* **The full-precision LingBot result is five probe clips, one first frame, one seed.** It does
  not establish a strong quantisation effect on this frame; it cannot rule one out on other
  scenes either.
* **The inventory is not a benchmark.** Licences, VRAM and cost are read from cards and
  measured only for the models actually run; the rest are labelled estimates.
* **SolarWM follows changing attitude, not held attitude.** The probes ramp continuously and
  pass; the designed arms bank in and hold, and the reader sees only a fraction of the held
  10-20 deg. The 75-clip grid and verdict are about this regime, not about ramps. The primary
  5/15 feasible count uses H02's absolute 12-frame rule; scaling H01's 12/16 rule to 81
  frames leaves 0/10 feasible (only the ten turn clips testable).
* **The grid reader uses every second frame** (41 of 81); the probes use all 81. This is stated
  in the methods; it changes precision, not the statistic.
* **The verdict is a point-estimate application, not H01's lower-bound scorer.** H02
  pre-registers the clip median of |required - depicted| at the designed path's point required
  tilt. H01's scored table used a lower-bound construction over path rate/drag; the 6.46 deg
  threshold is unchanged, but these numbers are not a byte-for-byte re-run of that scorer.
* **No scale-up was run.** Every cost for 200 clips is an extrapolation from a pilot.

## Caveats

* **Main does not contain the first-round tracks.** H2 depends read-only on the H01 branch for
  the reader, first frames and paths; the dependency is stated in the report and decisions.
* **SolarWM camera control is first-frame-relative**, so a constant attitude is invisible; the
  flatness arm uses a 12-frame bank-in and the probes are ramps. Path translations are scaled
  to 16 m to satisfy the released camera guards; rotations are unchanged.
* **The reader is the pre-registered instrument and it is imperfect on this imagery.** It
  reads a 3-5 deg baseline on level generator frames and did not track the visible oblique
  horizon in the bf16 LingBot +20/-20 clips; a horizontal-edge check on those frames is not a
  stable substitute (the images are painterly and the horizon is not a clean line). The
  discrepancy is recorded rather than resolved.
* **SolarWM output is 81 frames at 480x864**, 4 sampling steps; the released causal model
  supports longer rollouts, not tested here.
* **The Wan/SolarWM/Aero-World inventory is time-stamped 2026-11-21.** Gating and licences can
  change; the report records the status at that date.

## Cost of 200 clips

| model | GPU | s/clip (median) | peak VRAM | USD/clip | 200 clips |
| --- | --- | --- | --- | --- | --- |
| SolarWM-5B Stage2 (81 frames, 4 steps) | A100-80GB | 27.8 (75-clip grid including load/decode) | ~36 GB | 0.019 | 1.5 GPU-hours, US$3.9 |
| LingBot-World fast bf16 (45 frames, 4 steps) | A100-80GB | 62.9 (5 probes) | 55.2 GB | 0.044 | 3.5 GPU-hours, US$8.8 |

The SolarWM figure includes the model load and mp4 decode amortised over 75 clips and excludes
the reader; the LingBot figure is the median of five probe clips and excludes one 30-40 min
load. At 200 clips SolarWM is the cheapest path-controlled option measured in this project.

## Go/no-go

**SolarWM-5B: conditional no-go for sustained commanded paths.** It is cheap, released and
does follow continuously changing camera attitude, but on the designed paths it does not hold
the required 10-20 deg bank: the roll criterion fails (1/5 paths by tilt, 1/5 by mirrored
sign), and the frozen verdict rejects every testable turn-10 and turn-20 clip; it passes
slalom-10 only under H02's absolute 12-frame rule, and none of the ten clips testable under
the 12/16 fraction is feasible. Generating 200 more clips of the same designed manoeuvres
would buy more evidence of the same failure. A useful follow-up would use **time-varying** commanded attitudes (a
continuously rolling probe or a path whose bank changes throughout), which is exactly the
regime where the model works.

**Full-precision LingBot: not a route on the reader's evidence.** It does not follow the ramps
in the pre-registered reader either; the eye/reader discrepancy is unresolved. A 200-clip run
is cheap (US$9) but would test a model that the verifier currently cannot see responding.

**Inventory:** no other released model both takes a path and is practical (Aero-World has no
weights, CT-1 is not a generator, CameraCtrl/MotionCtrl/ReCamMaster are older controllers not
run).

## Reproduction

```
set -a; . ./.env; set +a
pixi run python experiments/h02_solarwm_inputs.py
pixi run modal deploy remote/h02_solarwm.py
pixi run modal deploy remote/h02_gate.py
pixi run python scripts/h02_run_solarwm.py --batch probes   # then grid, seeds
pixi run python scripts/h02_collect_solarwm.py --batch probes
pixi run modal run remote/h02_gate.py::main --source solarwm --clip-root solarwm/probes \
    --out-tag solarwm-probes
pixi run python experiments/h02_analyze_generated.py --depicted results/h02_solarwm_probes_depicted/depicted.jsonl \
    --poses data/h02_solarwm_inputs --out results/h02_solarwm_probes
pixi run python experiments/h02_verdict.py --depicted results/h02_solarwm_grid_depicted/depicted.jsonl \
    --out results/h02_solarwm
pixi run python experiments/h02_solarwm_seed.py --depicted results/h02_solarwm_seeds_depicted/depicted.jsonl \
    --out results/h02_solarwm

# full-precision LingBot
pixi run modal run remote/h02_lingbot_full.py::main --mode download
pixi run modal run remote/h02_lingbot_full.py::main --mode probe
pixi run modal run remote/h02_gate.py::main --source lingbot_full --out-tag lingbot-full
pixi run python experiments/h02_analyze_generated.py --depicted results/h02_lingbot_full_depicted/depicted.jsonl \
    --poses data/h02_lingbot_probe_poses --out results/h02_lingbot_full
```

## Provenance

| item | where |
| --- | --- |
| pre-registration | `configs/h02_pilot_prereg.yaml`, commit `5c6e5c6` |
| amendment 1 | `configs/h02_pilot_prereg_amendment1.yaml` (bank-in, 16 m scale) |
| inventory | `results/h02_inventory.json` (primary-source URLs per model) |
| SolarWM inputs | `data/h02_solarwm_inputs/` (110 samples: image, 81 poses, prompt, meta) |
| SolarWM clips | Modal volume `awmv-h2-results:/solarwm/{probes,grid,seeds}`, R2 `track-h2/clips/solarwm-*` |
| SolarWM results | `results/h02_solarwm/` (clips, frames, by-arm, roll criteria, verdict, cost.json), `results/h02_solarwm_*_depicted/` |
| full LingBot clips | volume `awmv-h2-results:/lingbot_full`, R2 `track-h2/clips/lingbot-full` |
| full LingBot results | `results/h02_lingbot_full/` (clips, frames, compare_h01.csv) |
| reader | `remote/h02_gate.py` (GeoCalib commit `1dedb29…`) |
| R2 manifests | `results/h02_r2/*.json` |
| cost | `docs/COST_LEDGER.md` |

Each SolarWM manifest records the model/pass/step count and the source sample id; the R2
manifests record sha256 for every uploaded object. The H2 worktree is based on `main`
(`41d1dad`); the H01 branch is a read-only dependency.
