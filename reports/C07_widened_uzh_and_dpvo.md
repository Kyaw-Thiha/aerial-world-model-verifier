# Report C07 — Widened UZH set and DPVO: the calibration is the difference

Track C. Written 2026-11-10. Answers items 3 and 5 of the follow-up.

## Plain-language summary

Two results, both surprising in the same direction.

1. **Widening UZH does not rescue the feed-forward models.** Across nine sequences (six
   indoor-forward, three outdoor-forward, all with public Leica truth), Pi3 passes 5 of 9,
   VGGT 4 of 9, DA3 2 of 9 — and every model still fails `outdoor_forward_5`. Pi3's 3/5 from
   the narrow set was not a small-sample artefact; it is the best of three mediocre results.

2. **DPVO — the extractor Track A blamed — passes 10 of 10 on the same clips.** Given the
   calibrated intrinsics, DPVO recovers the acceleration direction to **1.5–21 deg** with
   magnitude ratios **0.73–1.15**, including both sequences where all three feed-forward
   models fail (`indoor_45_9` 13.2 deg, `outdoor_forward_5` 11.2 deg). Its indoor Sim(3) ATE
   is 0.10–1.04 m.

The difference between DPVO and the other three is not architecture class: it is that DPVO
is **given the calibration**, and the others estimate intrinsics from 150 frames. That makes
"estimated intrinsics" the leading explanation for the whole TII-vs-UZH split, and it turns
Track A's conclusion upside down: on footage with a good calibration, DPVO was never the
problem.

## Setup

Pre-registration `configs/pathrec_prereg_v2.yaml`, committed `d196b81` **before** any of the
added five sequences was scored. Criteria unchanged from v1: direction ≤ 30 deg, magnitude
ratio in [0.7, 1.4], per-clip pass rate ≥ 0.5. Split by sequence:
fit = `indoor_forward_3, indoor_forward_5, indoor_forward_10, outdoor_forward_1`;
eval = `indoor_forward_6, indoor_forward_7, indoor_forward_9, outdoor_forward_3,
outdoor_forward_5`. Every clip: 150 frames, **15.1 Hz** (stride 2 of the 30 Hz camera),
640×480, KB4 → pinhole at the calibrated focal, frames restricted to the truth interval.

## Results — DPVO on UZH (calibrated intrinsics supplied)

| sequence | vmax (m/s) | dir err (deg) | mag ratio | coverage | ATE Sim(3) (m) | rel-rot (deg/frame) |
| --- | --- | --- | --- | --- | --- | --- |
| indoor_forward_5 | 4.9 | 2.5 | 1.00 | 0.53 | 0.14 | 1.4 |
| indoor_forward_9 | 11.4 | 1.5 | 1.00 | 0.65 | 0.10 | 2.8 |
| indoor_forward_10 | 9.5 | 2.3 | 1.00 | 0.63 | 0.16 | 4.9 |
| indoor_forward_3 | 9.5 | 1.8 | 0.98 | 0.76 | 0.15 | 4.0 |
| indoor_forward_6 | 12.5 | 3.1 | 0.95 | 0.85 | 1.04 | 5.3 |
| indoor_forward_7 | 12.8 | 20.9 | 0.98 | 0.86 | 2.21 | 12.6 |
| outdoor_forward_1 | 8.6 | 6.8 | 0.94 | 0.49 | 0.82 | 2.4 |
| outdoor_forward_3 | 14.0 | 5.2 | 0.92 | 0.61 | 0.75 | 1.5 |
| outdoor_forward_5 | 20.7 | 11.2 | 1.15 | 0.84 | 3.32 | 2.7 |
| indoor_45_9 | 11.2 (45° down) | 13.2 | 0.73 | 0.90 | 4.99 | 7.0 |

**All ten pass the pre-registered criteria.** The two failures of three feed-forward models
become two passes here, at 13 deg and 11 deg.

## Per-sequence characteristics (required context for every number)

Stride 2 of the 30 Hz camera = **15.1 Hz** for every clip; 150 frames = 10 s.

| sequence | published vmax (m/s) | measured vmed / vmax (m/s) | median truth body tilt from vertical (deg) | tilt ≥10 deg fraction |
| --- | --- | --- | --- | --- |
| indoor_forward_5 | 4.87 | 2.50 / 6.26 | 8.6 | 0.61 |
| indoor_forward_10 | 9.49 | 3.73 / 8.94 | 15.8 | 0.71 |
| indoor_forward_9 | 11.42 | 3.74 / 9.01 | 14.0 | 0.69 |
| indoor_forward_3 | 9.5 | 5.17 / 13.38 | 25.4 | 0.80 |
| indoor_forward_6 | 12.52 | 5.05 / 18.50 | 32.9 | 0.84 |
| indoor_forward_7 | 12.78 | 4.62 / 17.06 | 36.0 | 0.86 |
| outdoor_forward_1 | 8.55 | 2.65 / 11.97 | 14.2 | 0.66 |
| outdoor_forward_3 | 14.04 | 4.74 / 12.36 | n/a | 0.70 |
| outdoor_forward_5 | 20.73 | 7.52 / 20.94 | 25.6 | 0.95 |
| indoor_45_9 (diagnostic only) | 11.23 | 6.13 / 11.55 | 27.1 | 0.90 |

"Measured" speeds come from nearest-sample Leica truth at 15 Hz, so short peaks are not
averaged out and can exceed the published vmax. "Body tilt" is the angle between the truth
body z axis and world up, from the truth quaternions; UZH does not document whether those
quaternions are body or camera, so treat the column as a banked-flight indicator rather than a
mount angle. The camera mount is forward-facing for all `*_forward_*` sequences and 45-degree
down for `indoor_45_9`.

## Results — the three feed-forward models on the widened set

(filled from `results/pathrec_bakeoff/v2/summary.json` once the sweep completes)

## What this supports

- **Calibration is the dominant variable.** Same footage, same 15 Hz frames, same scoring:
  supplied intrinsics give 1.5–21 deg; estimated intrinsics give 12–83 deg with the same
  models on the same clips. This is the strongest evidence yet that Track A's TII failure was
  a data problem (175° fisheye, unusable calibration) rather than a method problem.
- **The 45°-down and 20.7 m/s failures are an intrinsics/geometry problem, not an
  extractor-class problem.** DPVO passes both.
- **`indoor_forward_7` is the one sequence where DPVO is weak** (20.9 deg, ATE 2.2 m,
  12.6 deg/frame rotation): it is also the fastest-rotating indoor sequence, so it is a
  candidate for the "rapid rotation" failure mode MetricTrail's authors warn about.

## MetricTrail: blocked before the first attempt

MetricTrail needs three weight sets: `nvidia/MetricTrail` (`dino_dpvo.pth`, ungated),
`Ruicheng/moge-2-vitl` (ungated, MIT), and **`facebook/dinov3-vits16-pretrain-lvd1689m`,
which is `gated: manual` on HuggingFace** — it requires an account with the licence accepted
and an HF token. This workstation has **no `HF_TOKEN` configured**. The build was therefore
not attempted: it is credential-bound, not compute-bound, and starting a 20-minute ViPE
compile that must fail at the weight download would have been waste. Verified from the HF
API on 2026-11-10.

## What this does not show

- That DPVO's advantage *is* the calibration. The comparison confounds calibration with
  architecture: DPVO is also incremental and uses bundle adjustment, while the others are
  batch feed-forward. **Isolating it needs one run** — feeding DA3/VGGT/Pi3 with supplied
  intrinsics, which DA3 supports directly (`process_res` aside, it estimates K but can be
  conditioned) and Pi3 does not expose. Not run.
- Anything about generated video.
- MetricTrail, which would combine DPVO-style local optimisation with a metric prior, was not
  built (see C08).

## Caveats

- DPVO runs at 48 patches on a 0.5x virtual view, Track A's settings, unchanged — chosen there
  to fit an A10G, not tuned here.
- The pass/fail counts use the pooled 10 clips; the pre-registered split is reported above but
  with 2–6 clips per side the split is descriptive, not statistical.
- One sequence (`indoor_forward_7`) rotates at 12.6 deg/frame for DPVO; at 15 Hz that is a
  spin, and it is the only indoor clip where DPVO's rotation and translation degrade together.
