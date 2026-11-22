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

**SolarWM.** The 5B Stage2 model runs on one A100-80GB and visibly follows the roll ramp on
the first test sample (level first frame, strongly oblique last frame; 81 frames, four steps).
Full probe/grid numbers TBD.

**Full-precision LingBot.** TBD — the reader still shows no coherent roll response in bf16
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

**LingBot full precision.** `robbyant/lingbot-world-fast` (74 GB, bf16, four-step) with the
VAE and T5 encoder from `robbyant/lingbot-world-base-cam`, on one A100-80GB. Same five probe
poses as H01 (level, linear +/-20, triangular 0->20->0, linear 0->60), same first frame,
49 frames. The upstream pipeline's multi-clip offload bug is fixed explicitly by moving the
transformer and VAE back to the GPU before each clip.

**Population and statistics.** The reader, required-tilt expression, 6.46 deg threshold and
testability rule are exactly H01's. The SolarWM grid is 5 frames x 5 paths x 3 arms = 75
clips; the seed set is 10 designed paths x 3 seeds (flatness arm). Intervals are
path-clustered (5,000 draws, `numpy.random.default_rng(1)`); with fewer than 10 clusters they
are descriptive. No threshold or success criterion changed after any outcome.

## Results

### 2. SolarWM-5B

**The released model runs and its camera control is visible.** The first test sample (`probe_p20`,
linear roll ramp 0 -> 20 deg over 81 frames) produced a level first frame and a strongly
oblique last frame; the release's output is 81 frames at 480x864, four sampling steps. Full
probe and grid statistics follow.

TBD SolarWM statistics.

### 3. LingBot full precision (bf16) versus H01's NF4

| probe | NF4 median (H01) | bf16 median | bf16 first-10 | bf16 last-10 | bf16 injected max |
| --- | --- | --- | --- | --- | --- |
| level | 3.21 | 4.56 | 3.04 | 5.89 | 0.0 |
| +20 ramp | 3.30 | 3.87 | 3.51 | 4.14 | 18.3 |
| -20 ramp | 3.77 | 4.91 | 2.98 | 5.17 | 18.3 |
| triangular 0->20->0 | 3.93 | 4.29 | 3.47 | 4.84 | 20.0 |
| 0->60 ramp | 4.05 | 4.06 | 3.27 | 4.73 | 55.0 |

The pre-registered reader does **not** show a coherent roll response in bf16 either: no ramp
raises the clip median above the level control, and the largest last-10 value (5.9 deg on the
level clip) is not tied to an injected ramp. The bf16 images look different from NF4 — the
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
* **Full precision did not reproduce H01's roll failure in the reader.** The bf16 LingBot
  probes stay at 4-5 deg for every ramp, like NF4, so quantisation is not established as the
  cause; the visible oblique horizons in the bf16 clips remain an unresolved discrepancy.
* **Path-conditioned sampling is measurable end to end with the existing verifier**: the same
  reader, required-tilt expression and verdict threshold carry over to SolarWM's 480x864
  output and a different pose convention.

## What this does not show

* **SolarWM's result is one checkpoint, one resolution, one seed.** It does not establish how
  the 14B, LTX-22B or H3-33B routes behave; those are only estimated if the 5B follows roll.
* **The full-precision LingBot result is five probe clips, one first frame, one seed.** It can
  rule out a strong quantisation effect on this frame; it cannot rule out one on other scenes.
* **The inventory is not a benchmark.** Licences, VRAM and cost are read from cards and
  measured only for the models actually run; the rest are labelled estimates.
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
  stable substitute (the images are painterly and vertical edges are missing). The
  discrepancy is recorded rather than resolved.
* **SolarWM output is 81 frames at 480x864**, 4 sampling steps; the released causal model
  supports longer rollouts, not tested here.
* **The Wan/SolarWM/Aero-World inventory is time-stamped 2026-11-21.** Gating and licences can
  change; the report records the status at that date.

## Go/no-go and cost of 200 clips

TBD.

## Reproduction

TBD.

## Provenance

TBD.
