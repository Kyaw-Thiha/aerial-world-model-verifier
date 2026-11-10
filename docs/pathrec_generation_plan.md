# Track C — plan for the general-model generation study (not yet run)

Status: **plan only, for approval.** Nothing here has been executed. Written 2026-11-10.
Prerequisites: Gate 1 and Gate 2 of handoff 06 (the extractor bake-off and the priors
study) must have produced a pixel-path extractor that meets the pre-registered criteria on
at least some testbed. If Gate 1 fails outright, this plan is not run.

## Question

Given a text- or image-conditioned video model, are the flights it generates physically
possible? With no input path, the path has to come from the pixels, so the answer inherits
every assumption measured in Gate 2: an fps, a scale (or a scale interval), a differentiation
window, and a drag coefficient.

## What is being measured

For every generated clip: the violation rate **with abstention**, i.e.

    violation rate = (# clips called inconsistent) / (# clips the extractor accepted)

reported per model, per prompted manoeuvre, and with `dt` and `k_d` as explicitly varied
axes (a factor-2 error in `dt` is a factor-4 error in acceleration, so `dt` is not a nuisance
parameter — it is one of the results).

## Models

| model | weights | licence | why in the study | availability |
| --- | --- | --- | --- | --- |
| Wan2.2-I2V-A14B | `Wan-AI/Wan2.2-I2V-A14B` (~126 GB) | Apache-2.0 | strongest open image-to-video model; the reference | verified: HF model card + `Wan-Video/Wan2.2` repo |
| Wan2.1-I2V-1.3B | `Wan-AI/Wan2.1-I2V-1.3B` | Apache-2.0 | scale trend: same family, 10x smaller | verified in the Wan2.1 release |
| Cosmos-Predict2.5 | `nvidia-cosmos/cosmos-predict2.5` | Apache-2.0 code, NVIDIA Open Model licence for weights | trained for "physical AI"; the model most likely to claim physical plausibility | verified: repo + Cosmos Lab page |
| CogVideoX (5B) | `zai-org/CogVideoX` | to confirm at download time | third family, different training data | repo exists; licence not yet read — **unverified** |

Excluded on cost or licence: DA3-GIANT (1.4 B, non-commercial), VGGT-1B-Commercial (gated
application), and any model whose weights need an HF token we do not have configured.

## Camera-controlled model (a harder extractor test, availability only)

**WorldForge V2** (`arXiv:2610.31640`) is the one to use: its camera trajectory is an
*input*, so the video comes with a known path and the extractor can be scored on
general-model output without needing a text prompt to imply a trajectory.
Availability verified 2026-11-10 from the project's own repo: code and weights released
2026-10 (`h3_for_worldforge/`, weights at `Chenxis27/WorldForge-V2` on HuggingFace),
repository `Westlake-AGI-Lab/WorldForge` is Apache-2.0, built on Wan2.1. Not run.

## Starting frames

1. **TII frames** (preferred): real aerial imagery, already mirrored on the Modal volume
   (`awmv-a-data/tii_video`), with a known camera rig and mocap truth for the same instant.
   One starting frame per flight, eight flights.
2. **MotionScape clips** if obtainable under a usable licence; recorded only if the download
   is available from an official page.

## Prompted manoeuvres

Each prompt is paired with a declared fps and a clip length that the model actually
supports (Wan2.2: 81 frames at 16 fps = 5 s; Cosmos: 93 frames at 24 fps).

| manoeuvre | prompt sketch | what the verifier should see |
| --- | --- | --- |
| straight cruise | "fly forward at constant speed over the rooftops" | tilt ≈ drag-only, small |
| accelerate | "accelerate hard forwards" | forward lean growing |
| brake | "brake sharply and slow down" | backward lean |
| banked turn | "bank right into a tight turn" | lateral lean with yaw rate |
| climbing turn | "climb while turning left" | combined lean |
| **control: yaw-only** | "rotate in place to look left" | no lean without translation |

The yaw-only control is scored beside every manoeuvre: a model that only rotates can be
mistaken for a model that banks, and the verifier must not confuse them.

## Seeds and repeats

Four seeds per (model, prompt, start frame). Fixed seed list recorded in the run config;
each clip gets a stable id `model_startframe_prompt_seed`.

## Expected cost

- Wan2.2-A14B I2V at 720p, 81 frames: roughly 8-12 min on an A100 (scaled from the released
  inference timings; **inferred**, to be replaced by a one-clip measurement before the run).
- 4 seeds x 6 manoeuvres x 2 models = 48 clips; plus 32 clips for the 1.3B and Cosmos models.
- Estimate: 6-8 A100-hours, **US$15-25**, plus WorldForge V2 (another 8-10 clips, ~US$5).

Because that is above the US$5 threshold, **the run waits for explicit approval** after a
one-clip timing measurement replaces the inferred numbers.

## Abstention rules (proposed, to be frozen before the run)

A generated clip is *not testable* if any of these holds, all computed without truth:

1. the extractor's cross-extractor step-direction disagreement exceeds 60 deg;
2. the recovered path's jitter ratio exceeds the 90th percentile of the fit-split clips of
   the same extractor;
3. the clip's testable fraction (implied tilt ≥ 10 deg) is below 0.2;
4. the required tilt from the recovered path exceeds 60 deg anywhere (beyond the flatness
   regime the model was derived in);
5. the depth prior disagrees with the extracted depth by more than a factor of 3 (scale is
   then unknown, so only direction could be judged — and direction alone is not a verdict).

A clip is called *inconsistent* only if no scale inside the depth-prior interval makes the
depicted and implied tilt agree within the pre-registered tolerance. Everything else is
"not testable", never "infeasible".

## Reporting

- Primary table: violation rate with abstention, per model x prompted manoeuvre, with the
  abstention rate beside it and the yaw-only control in the last row.
- Axes: `dt` at the model's declared fps and at 2x and 0.5x that; `k_d` at 0 and 0.295.
- Every number carries its fps, its scale source (depth prior or interval) and the extractor
  version. SE(3)-aligned and direction-only results are reported beside any Sim(3) number.
- Negative results are kept: if a model produces no measurable flight at all, that is a
  finding, not a failed run.

## What this would not show

- That a generated flight is physically impossible in the real world — only that it is
  inconsistent with the flatness model under a stated scale and time base.
- Anything about manoeuvres the prompts do not cover, or about flights longer than the
  generated clip.
- Anything about models released after the run's fixed model list.
