# Report C08 — Calibration sensitivity and scale provenance

Track C. Written 2026-11-10. Answers items 2 and 3 of the follow-up.
Pre-registration: `configs/pathrec_prereg_calib.yaml`, committed `2194b64` before any
perturbed run.

## Plain-language summary

**DPVO barely notices a wrong focal length.** At ±10 % focal error it still passes 9–10 of the
10 clips; at ±20 % it still passes 8–10. The pre-registered hypothesis that ±20 % would break
half the clips is **refuted**. One clip (`indoor_forward_7`) fails at every scale, including
the calibrated one, and is also the clip where DPVO's own run-to-run variance is largest —
that is a property of the clip, not of the perturbation.

**The magnitude numbers in every earlier report are oracle-scaled.** The bake-off aligns with
Sim(3), which fits the scale against ground truth. A depth-prior scale (DA3METRIC-LARGE,
prior-only, no truth) is **unbiased in the median (1.02× the oracle) but per-clip wrong by
0.41× to 1.54×**. Direction survives that; magnitude does not: the same clips that show
ratios of 0.73–1.15 with the oracle scale show 0.30–7.6 with the depth-prior scale.

So the verifier can claim direction from pixels today, and **cannot yet claim metric
magnitude** without a calibrated scale interval.

## Item 2 — what scale does the ratio use?

`score_clip` calls `umeyama(est, true, with_scale=True)` and then differentiates the aligned
track. **The magnitude ratio is therefore computed with the oracle Sim(3) scale.** It measures
trajectory shape, not a metric claim. This has been true of every table in C03, C05 and C07
and is now stated in those reports' caveats.

### Depth-prior version (no truth at scoring time)

Depth provider: `depth-anything/DA3METRIC-LARGE`, per-frame median metric depth
(`focal · depth / 300` metres, the documented conversion). Scale from the depth change between
low-rotation consecutive frames:

    s = median( -(D[t+1] - D[t]) / ((p[t+1] - p[t]) · z_cam) )

Run: `results/pathrec_scale_provenance/v1/summary.json`, 150 frames per clip.

| clip | oracle scale | depth-prior scale | ratio | pairs | dir err with prior scale | magnitude ratio with prior scale |
| --- | --- | --- | --- | --- | --- | --- |
| indoor_forward_3 | 0.679 | 0.499 | 0.73 | 28 | 1.8 | 0.73 |
| indoor_forward_5 | 0.227 | 0.348 | 1.54 | 49 | 2.5 | 1.53 |
| indoor_forward_6 | 0.259 | 0.106 | 0.41 | 14 | 3.1 | 0.37 |
| indoor_forward_9 | 0.263 | 2.084 | 7.92 | 27 | 1.4 | 7.62 |
| indoor_forward_10 | 0.237 | 0.335 | 1.41 | 24 | 2.3 | 1.40 |
| indoor_forward_7 | 0.995 | 461.3 | 464 | 4 | 20.6 | 4.84 |
| outdoor_forward_1 | 0.172 | 0.224 | 1.30 | 70 | 6.9 | 1.23 |
| outdoor_forward_3 | 0.220 | 0.153 | 0.70 | 51 | 5.3 | 0.64 |
| outdoor_forward_5 | 0.256 | 0.119 | 0.46 | 42 | 11.2 | 0.53 |

Median depth/oracle ratio **1.02**, p10 0.41, p90 53.5. Excluding the two clips with few
valid pairs (`indoor_forward_7` n=4, `indoor_forward_9` ×7.9), the per-clip error spans
0.41×–1.54×.

**Direction is preserved** (1.4–11.2 deg on eight of nine clips) because direction is
scale-free. **Magnitude is not**: 6 of 9 clips fall outside the pre-registered [0.7, 1.4].

## Item 3 — focal-length sensitivity (DPVO, all ten clips, 15.1 Hz)

fx and fy multiplied by the factor, cx/cy unchanged; everything else identical.

| condition | clips | median dir err (deg) | clips passing | failing clips |
| --- | --- | --- | --- | --- |
| fx × 0.8 | 10 | 12.6 | 9 | indoor_forward_7 |
| fx × 0.9 | 10 | 6.6 | 10 | — |
| fx × 1.0 (calibrated) | 10 | 3.9 | 9 | indoor_forward_7 |
| fx × 1.1 | 10 | 8.5 | 9 | indoor_forward_7 |
| fx × 1.2 | 10 | 13.5 | 8 | indoor_forward_7, indoor_45_9 |

Run: `results/pathrec_calib/focal_matrix.json`.

- **H1 confirmed**: ±10 % keeps 9–10 of 10 passing.
- **H2 refuted**: ±20 % breaks 1–2 clips, not half.
- The monotone shape is as expected — direction error rises from 3.9 deg (correct focal) to
  12.6–13.5 deg at ±20 % — but the criterion has a wide margin.

### Estimated intrinsics (DA3, VGGT) — H3 refuted

DPVO run with the intrinsics that DA3-LARGE and VGGT predict on the same frames (their
processed rasters — 504 px for DA3, 518 px for VGGT — rescaled to the 640 px prepared frames):

| intrinsics fed to DPVO | clips | median dir err (deg) | passing | failing |
| --- | --- | --- | --- | --- |
| fx*1 | 10 | 3.0 | 10 | — |
| da3 | 10 | 13.5 | 8 | outdoor_forward_5, indoor_45_9 |
| vggt | 10 | 18.9 | 8 | outdoor_forward_5, indoor_45_9 |

Run: `results/pathrec_calib/estimated_k.json`.

**H3 is refuted.** Handed the feed-forward models' own K, DPVO still passes 8 of 10 clips,
with direction errors of 8–31 deg — nothing like the 58–83 deg those same models produce when
they estimate the trajectory themselves. **The feed-forward failure is in the trajectory, not
in the focal length.** DA3's K costs 20–30 deg of direction on the hard clips
(`outdoor_forward_5` 31.5, `indoor_45_9` 24.1), so estimated K does contribute, but it is a
second-order effect.

## What this supports

- **Focal length is not a sharp constraint for DPVO.** A ±20 % error is survivable, which
  makes it unlikely that the feed-forward models' 58–83 deg failures on the two hard clips are
  caused by *modest* focal error alone. Something larger differs — geometry (the 45° mount),
  or the batch models' trajectory estimation itself.
- **Scale, not direction, is the verifier's binding constraint.** With a depth prior the
  direction criterion still passes on 8 of 9 clips; the magnitude criterion collapses.

## What this does not show

- Whether the feed-forward models' *own* estimated K, fed to DPVO, reproduces their failure
  (H3 untested).
- Whether a better depth prior (MoGe-2, or MetricTrail's fused depth) closes the per-clip
  scale spread. DA3METRIC on single frames is the weakest option in the list.
- Anything about clips other than these ten, or about errors larger than ±20 %.

## Caveats

- `indoor_forward_7` fails at the calibrated focal in this run (ratio 0.33) but passed in the
  earlier run (ratio 0.98): **DPVO is not deterministic** here (patch initialisation and the
  bundle-adjustment schedule). Single-run pass/fail on that clip should not be read as a
  property of the condition.
- The depth-prior estimator assumes a static scene and mostly-forward motion, and uses only
  pairs with < 3 deg rotation; on `indoor_forward_7` only 4 pairs qualified, which is why its
  estimate is nonsense rather than merely noisy.
- The oracle scale is fitted per clip against Leica truth; it is an evaluation device, not
  something the verifier can compute.
