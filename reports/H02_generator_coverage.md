# H02 — Generator coverage with path control: SolarWM, LingBot full precision, inventory

Track H2, 2026-11-21. Branch `track-h2-generator-coverage`, based on `main` (`41d1dad`),
which does not yet contain the first-round tracks. Depends on the read-only branch
`track-h-generator-pilot` (H01) for the reader, the five first frames, the designed paths and
the LingBot comparison; no H01 file was modified. Pre-registration:
`configs/h02_pilot_prereg.yaml` (commit `5c6e5c6`), committed before any H2 clip was generated
or scored. Spend ceiling US$90; actual spend in `docs/COST_LEDGER.md`.

## Plain-language summary

TBD.

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

TBD.

## Results

TBD.

## What this supports

TBD.

## What this does not show

TBD.

## Caveats

TBD.

## Go/no-go and cost of 200 clips

TBD.

## Reproduction

TBD.

## Provenance

TBD.
