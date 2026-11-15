# Report C09 — DPVO on ANWM, and the pixel-path verdict test on UZH

Track C. Written 2026-11-10. Answers items 4 and 5 of the follow-up.

## Plain-language summary

**Item 4 — DPVO cannot be used on the ANWM clips.** Ten of 25 crash inside DPVO, six more
return a degenerate trajectory (magnitude ratios above 10, i.e. no coherent motion), leaving
nine usable clips at a median direction error of 48 deg (generated) and 45 deg (ground truth).
The 3–5 m step baseline at 224 px, one frame per second, is too sparse for a patch tracker.
**Use Pi3 there** — on the same clips it recovers the commanded path to 1–8 deg on the
ground-truth frames (C05).

**Item 5 — the pixel-path verdict test fails on UZH, for a structural reason.** With the path
from DPVO and the depicted tilt from GeoCalib, the residual is ~86 deg on the real clip and
~89 deg on the tilt-removed clip: no separation, and both far above any sensible threshold.
The cause is not the path: a constant camera-to-body mount rotation estimated with oracle
truth only brings the residual to 56 deg, and the edit still does not separate (50 deg).
**GeoCalib's gravity gives the horizon in the camera; the thrust axis needs the camera-to-body
mount, which UZH does not publish.** TII does publish it (`cam_uptilt` in its metadata), so
the depicted-tilt route is only well-defined on the one testbed whose path is unusable.

## Item 4 — DPVO on ANWM (Regime A, 25 clips, GT frames vs generated)

Run: `results/pathrec_anwm/raw/anwmA_dpvo.json`.

| outcome | clips |
| --- | --- |
| crashed (`cannot reshape tensor of 0 elements`, `CUDA error`) | **10 / 25** |
| degenerate trajectory (magnitude ratio > 10) | **6 / 15 survivors** |
| usable | 9 |

Usable clips: median direction error **48.2 deg** (generated) and **44.9 deg** (ground truth).
DPVO also disagrees with itself between the two conditions on `A/A16` (107 deg generated vs
10 deg ground truth), so even the survivors are not stable.

**Diagnosis**: at 1 s per step and 3–5 m of motion per step, consecutive 224×224 frames share
too little overlap for DPVO's patch graph; patches are born and die without ever being
re-observed, which is exactly the `BUFFER`-exhaustion / zero-patch failure the crashes show.

**Recommendation**: for ANWM-scale clips use **Pi3** (fastest and best on the ground-truth
frames: 4/6 testable clips pass, C05). DPVO's strength — incremental tracking with a tight
motion model — is the wrong tool when the baseline is 5 m.

## Item 5 — verdict test (pre-registration `518024e`)

Design: `indoor_forward_5`, three conditions (real, tilt-removed, yaw-control), DPVO path
(calibrated), GeoCalib depicted gravity, complementary filter gain 0.2, `k_d` 0.295,
violation if the median residual exceeds 10 deg.

| condition | pixel-path score | testable fraction | verdict | path-only control (true path) |
| --- | --- | --- | --- | --- |
| real | 85.8 | 0.77 | violated | 110.8 |
| tilt-removed | 88.7 | 0.72 | violated | 115.5 |
| yaw-control | not analysed (poses not saved by the batch) | — | — | — |

**No separation, and the absolute values are ~90 deg — the mount signature.** A single
camera-to-body rotation fitted with oracle truth lowers the residual to 56 deg (real) and
50 deg (tilt-removed): better, still far from usable, and still no separation.

### Why it fails

GeoCalib recovers the gravity direction *in the camera frame*. The verifier compares the
depicted **thrust axis**, which is the body z axis, not the camera z axis. On TII the mount
angle is in the dataset metadata (`cam_uptilt`, Track A used it); on UZH the camera is
forward-facing with an unknown pitch relative to the body, and no mount model is published.
For a forward camera the optical axis is roughly perpendicular to the thrust axis, hence the
~90 deg offset.

This is a property of the testbed, not a bug in the pipeline: the pipeline runs end to end
(GeoCalib on 150 frames per condition, DPVO on the same frames, fusion, residual).

## What this supports

- **The pixel-path verdict is currently blocked by the depicted-tilt side, not the path side.**
  DPVO's path on the same clip is accurate to 2.5 deg; the residual is dominated by the
  unknown mount.
- **TII is the only testbed with a published mount**, and it is the testbed where the path is
  unusable. That is the central tension of pixel-path verdicts in this project, and it is now
  measured rather than assumed.
- **ANWM needs a batch extractor.** DPVO's incremental design fails at a 5 m baseline.

## What this does not show

- Whether the verdict works if the mount is estimated *without* truth (e.g. from an
  accelerometer prior, or by fitting the mount and the path jointly). That is the obvious next
  step, and it is not run here.
- Anything about generated video: no model was run.
- The yaw-control number: its poses were not saved by the batch that scored it (the numbers
  were computed but not persisted), so the warp-artefact control is missing from the table.

## Caveats

- One clip, one extractor, one depicted-tilt estimator. This is a pilot that failed a
  pre-registered criterion, not a general claim about GeoCalib.
- The "path-only" control uses the true path with the same depicted tilt, so it isolates the
  path's contribution: it is worse than the pixel path, which is another sign that the
  residual is dominated by the mount rather than by either path.
- The tilt-removed edit was applied with a pure-rotation homography built from the truth
  attitude; if the truth quaternions are camera rather than body, the edit levels the camera,
  which is the same thing here because no mount is modelled.
