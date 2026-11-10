# Report C04 — Gate 2: what a pixel-path verdict may claim

Track C. Written 2026-11-10. Task 4 of handoff 06.

## Plain-language summary

A pixel-path verdict can only be as sharp as the scale and time base it assumes, and both are
unknown for generated video. This report measures that dependence on the UZH testbed, where
the truth is known, and states the claim the verifier is allowed to make afterwards.

The short version, to be finalised with the numbers below:

- **Scale is a single, stable number per testbed** (the Sim(3) scale), so it can be calibrated
  once on held-out sequences and carried into generated video *of the same kind*.
- **An unknown fps is not survivable**: a factor-2 error in the time base multiplies the
  recovered acceleration, and therefore the apparent gravity, by 4. The model's declared fps
  is an assumption the verdict inherits.
- **Direction survives both**, which is why the pre-registered direction criterion is the one
  that can carry a verdict on data like this.

## Setup

Poses stored by the bake-off, truth from the prepared clips, differentiation window 0.5 s,
polyorder 3, `k_d` = 0.295. Scale is estimated per clip by Sim(3) alignment against truth;
the interval is fitted on the fit split (`indoor_forward_5`, `indoor_forward_3`) and evaluated
on the held-out split (`indoor_45_9`, `indoor_forward_6`, `outdoor_forward_5`).

Two statistics:

1. **Scale ratio** per clip and the empirical 90 % interval.
2. **Apparent gravity** `g_app = 1 / slope` of the through-origin regression of a lean vector
   on the recovered horizontal acceleration. Correlation and R² are unchanged by an unknown
   scale or time base; the slope is not, and `g_app` should be about 9.81 m/s² when the lean
   in the pixels matches the path. Here the lean is the **oracle lean from the true
   trajectory** (the depicted-lean version needs GeoCalib on UZH, which was not run), so this
   isolates extractor and time-base error from pixel error.

## Results

Run: `results/pathrec_priors/v1/summary.json`, poses from the Gate 1 sweep, truth at 15.1 Hz.

### 1. Scale ratio per clip and the fitted interval

| extractor | forward_5 (fit) | forward_3 (fit) | forward_6 | 45_9 | outdoor_5 | 90 % interval (fit) |
| --- | --- | --- | --- | --- | --- | --- |
| Pi3 | 7.30 | 9.36 | 8.28 | 8.02 | 17.83 | 7.40 - 9.25 (median 8.33) |
| DA3 | 6.84 | 10.79 | 11.41 | 7.21 | 21.58 | 7.04 - 10.59 (median 8.82) |
| VGGT | 9.99 | 11.15 | 10.11 | 10.35 | 19.81 | 10.05 - 11.09 (median 10.57) |

Held-out coverage of the fitted interval (3 eval clips each): **Pi3 2/3, VGGT 2/3, DA3 1/3.**
The one that misses is always `outdoor_forward_5`, where the scale is about twice the indoor
value: an interval fitted on indoor flights does not transfer to the outdoor flight.

### 2. Apparent gravity (oracle lean, m/s², should be about 9.81)

| extractor | forward_5 | forward_3 | forward_6 | 45_9 | outdoor_5 |
| --- | --- | --- | --- | --- | --- |
| Pi3 | **10.84** (R² 0.86) | 13.34 (0.64) | **10.71** (0.66) | 86.8 (0.00) | 53.9 (0.12) |
| DA3 | 36.9 (0.24) | 20.9 (0.47) | 18.2 (0.48) | 3162 (~0) | 3021 (~0) |
| VGGT | 27.0 (0.28) | 15.3 (0.55) | 21.2 (0.34) | -233 (~0) | 1105 (~0) |

The statistic tracks the Gate 1 verdict exactly: where direction error is 12-22 deg the
apparent gravity is 11-13 m/s² (within 10-36 % of the true value); where direction error is
60-83 deg the slope collapses and `g_app` explodes or flips sign. **It is a usable
self-check**: a clip whose apparent gravity is far from g is a clip whose path cannot carry a
verdict, and no ground truth is needed to compute it once a scale is assumed.

### 3. fps sensitivity

Every result above with the time base wrong by 2x multiplies the recovered acceleration by 4
and so multiplies `g_app` by 4 (Pi3 forward_5: 10.84 -> 43.36; forward_6: 10.71 -> 42.84;
halving the fps divides by 4). Correlation and R² are unchanged, so **fit quality cannot
detect a wrong fps** — only the absolute magnitude can, and only if a scale is trusted.

### 4. Plausibility-interval rule

With intervals this narrow (Pi3: 7.4-9.3) the rule "inconsistent only if no scale in the
interval works" collapses to "the interval must be right". On held-out indoor clips it is
(2/3, and the miss is the outdoor flight). On the outdoor flight it is not: the scale doubles,
so a verdict that carries the indoor interval would call a correct flight inconsistent. **The
scale interval must be conditioned on the flight type, and it cannot be borrowed across
regimes.**

## What a verdict may claim, and may not

**May claim**, for a clip whose extractor passed Gate 1 and whose reliability signals are
inside the pre-registered bounds:

> Under the model's declared frame rate and a scale calibrated on flights of this kind, the
> depicted lean is inconsistent with the lean required by the recovered path, by more than the
> verifier's tolerance, on at least N of M testable samples.

**May not claim:**

- that the flight is physically impossible in the world (only inconsistent with the flatness
  model at the assumed scale and fps);
- anything at all when no scale inside the plausibility interval agrees with the pixels —
  that case is "untestable", not "violated";
- anything on clips where the recovered acceleration direction is inside its noise floor;
- anything about manoeuvres not in the test set, or about flights longer than the clip.

## What this does not show

- The depicted-lean version of the apparent-gravity statistic (needs GeoCalib on UZH).
- A calibrated scale for generated video: the interval comes from real flights of the same
  dataset, and generated imagery may sit outside it.

## Caveats

- The interval in the results has few clips per model; it is a pilot, not a deployment
  calibration. The pre-registration fixes the evaluation split, so more clips can be added
  without re-scoring these.
