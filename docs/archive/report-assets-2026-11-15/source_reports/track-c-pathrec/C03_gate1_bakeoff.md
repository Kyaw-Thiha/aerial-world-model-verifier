# Report C03 — Gate 1: can the path be recovered from pixels?

Track C. Written 2026-11-10. Task 3 of handoff 06.

## Plain-language summary

**Gate 1 passes on exactly one extractor, on exactly part of one testbed: Pi3 on the three
forward-facing UZH sequences. Everything else fails the pre-registered criteria.**

- On UZH-FPV (calibrated pinhole, 15 Hz, Leica truth), the three forward-facing sequences
  give acceleration-direction errors of **12.5 deg (Pi3), 20-31 deg (DA3, VGGT)** with
  magnitude ratios of **1.12 (Pi3) to 1.6 (VGGT)**. Only Pi3 stays inside the 30 deg / [0.7,
  1.4] envelope on 3 of 5 clips, which meets the 50 % coverage criterion.
- The two hard sequences (a 45-deg downward camera and a 20.7 m/s outdoor flight) fail for
  **all three** extractors: 58 to 83 deg of direction error and magnitude ratios of 3.8 to
  6.4. That is not model-specific; it is the data.
- On TII (fisheye crop, no calibration, heavy blur at up to 20 m/s) **every extractor fails
  every clip**: 45 to 99 deg of direction error. Track A's conclusion generalises beyond
  DPVO: it is not the extractor, it is the footage.

The second finding is a warning about metrics. On TII the **Sim(3)-aligned trajectory error
is 2.4 to 4.5 m on 400 to 500 m of flight** — about 1 % — while the acceleration direction is
garbage. A trajectory can look excellent by the standard SLAM metric and still carry no
usable acceleration, so every number below sits beside its direction error.

## Question and setup

**Question.** On testbeds where the true path is known, which extractors recover horizontal
acceleration (direction, magnitude, coverage) well enough to feed the lean verifier, and on
which data?

**Pre-registration.** `configs/pathrec_prereg.yaml`, committed at `6a16297` *before* any
extractor was scored. Criteria: median acceleration direction error ≤ 30 deg, magnitude
ratio in [0.7, 1.4], per-clip pass rate ≥ 50 %, on clips whose true required tilt is ≥ 10 deg
for at least 20 % of samples.

**Testbeds.**
- **T1 UZH-FPV**, five sequences, 150 frames each, every second frame of 30 Hz (15.1 Hz
  measured), undistorted KB4 → pinhole at the calibrated focal, frames restricted to the
  Leica-truth interval. Speeds (published vmax): `indoor_forward_5` 4.9, `indoor_forward_3`
  9.5, `indoor_45_9` 11.2, `indoor_forward_6` 12.5, `outdoor_forward_5` 20.7 m/s.
- **T2 TII**, four flights (01a, 01p, 04a, 07p), 100 frames at stride 10 of 120 Hz (12 Hz),
  central crop at fov 100 deg (320x240, fx = 134.26 px), no released calibration used.
- **T3 ANWM**: prepared (loader + same-start level renders) but **not scored**; see
  "what this does not show".

**Extractors.** DA3-LARGE-1.1, VGGT-1B, Pi3, all run on Modal A10G with their documented
preprocessing. DPVO was not re-run on UZH (Track A's build); see caveats. MetricTrail was not
built. All three are batch models: 150 frames is inside their documented range.

**Differentiation.** Savitzky-Golay, window 0.5 s, polyorder 3, `k_d` = 0.295 fixed.

## Results

### T1 UZH-FPV (15 Hz, calibrated pinhole)

Run: `results/pathrec_bakeoff/v1/summary.json`, pre-registration `6a16297`, frames at 15.1 Hz.

| extractor | sequence | vmax (m/s) | accel dir err (deg) | mag ratio | lean R² | coverage | rel-rot (deg/frame) | pass |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Pi3 | indoor_forward_5 | 4.9 | 12.5 | 1.12 | 0.87 | 0.53 | 1.4 | yes |
| Pi3 | indoor_forward_3 | 9.5 | 22.3 | 1.23 | 0.26 | 0.76 | 3.9 | yes |
| Pi3 | indoor_forward_6 | 12.5 | 20.0 | 1.23 | 0.24 | 0.85 | 5.2 | yes |
| Pi3 | indoor_45_9 | 11.2 | 62.9 | 6.41 | 0.00 | 0.90 | 8.4 | no |
| Pi3 | outdoor_forward_5 | 20.7 | 62.0 | 3.76 | 0.06 | 0.84 | 2.8 | no |
| DA3 | indoor_forward_5 | 4.9 | 28.7 | 1.27 | 0.25 | 0.53 | 1.5 | yes |
| DA3 | indoor_forward_3 | 9.5 | 31.5 | 1.52 | 0.47 | 0.76 | 3.9 | no |
| DA3 | indoor_forward_6 | 12.5 | 22.6 | 1.51 | 0.37 | 0.85 | 5.0 | no |
| DA3 | indoor_45_9 | 11.2 | 82.6 | 5.93 | -0.01 | 0.90 | 9.8 | no |
| DA3 | outdoor_forward_5 | 20.7 | 58.5 | 4.72 | -0.03 | 0.84 | 3.0 | no |
| VGGT | indoor_forward_5 | 4.9 | 28.4 | 1.60 | 0.30 | 0.53 | 1.5 | no |
| VGGT | indoor_forward_3 | 9.5 | 23.3 | 1.28 | 0.38 | 0.76 | 3.8 | yes |
| VGGT | indoor_forward_6 | 12.5 | 24.4 | 1.26 | 0.19 | 0.85 | 4.9 | yes |
| VGGT | indoor_45_9 | 11.2 | 77.6 | 5.35 | 0.00 | 0.90 | 9.6 | no |
| VGGT | outdoor_forward_5 | 20.7 | 76.8 | 3.97 | 0.02 | 0.84 | 3.7 | no |

Per-extractor verdict against the pre-registered criteria (per-clip pass rate ≥ 0.5):

| extractor | clips passing | rate | Gate 1 |
| --- | --- | --- | --- |
| Pi3 | 3 / 5 | 0.60 | **PASS** on UZH |
| VGGT | 2 / 5 | 0.40 | fail (marginally) |
| DA3 | 1 / 5 | 0.20 | fail |

### T2 TII (12 Hz, fisheye cropped, no calibration)

| extractor | flight | accel dir err (deg) | mag ratio | ATE Sim(3) (m) |
| --- | --- | --- | --- | --- |
| DA3 | 01a-ellipse | 87.3 | 172.0 | 3.28 |
| DA3 | 01p-ellipse | 80.4 | 15.6 | 4.39 |
| DA3 | 04a-ellipse | 98.6 | 79.4 | 2.82 |
| DA3 | 07p-lemniscate | 83.9 | 8.1 | 4.06 |
| VGGT | 01a-ellipse | 74.4 | 9.1 | 3.27 |
| VGGT | 01p-ellipse | 95.9 | 2.4 | 4.52 |
| VGGT | 04a-ellipse | 44.7 | 6.8 | 2.88 |
| VGGT | 07p-lemniscate | 96.2 | 2.3 | 3.79 |
| Pi3 | 01a-ellipse | 71.6 | 49.4 | 2.43 |
| Pi3 | 01p-ellipse | 74.5 | 5.8 | 3.27 |

Every TII row fails both criteria. Magnitude ratios above 1 mean the recovered acceleration
is *too large* relative to the truth lean, the signature of a path that moves but does not
lean with its motion.

## What this supports

- **Gate 1 passes only for Pi3 on UZH**, and only on the forward-facing sequences. The
  pre-registered rule is applied as written: 3 of 5 clips pass, 0.60 >= 0.50.
- **The TII failure is not a DPVO artefact.** Three unrelated architectures, none sharing
  DPVO's patch graph, fail the same way on the same footage (45-99 deg). Track A's
  "the extractor is the problem" reading is too narrow: the *data* is.
- **The UZH failure on two sequences is also not model-specific.** At 45 deg downward and at
  20.7 m/s, DA3, VGGT and Pi3 all degrade together (58-83 deg), while rotation stays good
  (2.8-9.8 deg per frame). Whatever breaks is in the translation, not the attitude.
- **Magnitude is the weaker half of the verdict.** Even the passing clips sit at ratio
  1.12-1.28; the failing ones overshoot by 4-6x, i.e. the recovered path *moves* but its lean
  does not follow its motion. A magnitude criterion that is not scale-anchored will not
  survive contact with generated video.
- **ATE is the wrong gate**: about 1 % trajectory error coexists with 80 deg acceleration
  error on TII.

## What this does not show

- **ANWM (T3) is not scored.** The loader is built and unit-tested, and the clip convention is
  corrected (frame *i* corresponds to `positions[i+1]`, 1 s per step, level ground-truth
  renders). No extractor was run on those clips, so nothing is claimed about generated video.
- **DPVO was not re-run** on UZH (Track A's build would need a fresh image): its TII numbers
  stand, and UZH is the missing cell in this table.
- **The 45-deg and outdoor UZH failures are unexplained.** A timing offset was ruled out for
  the contaminated clip by lag search, but the clean re-prepared clips were not lag-searched.
  Candidates, untested: heavy inter-frame rotation, motion blur, and the 45-deg mounting
  making the "flattest axis is up" check unreliable.
- **MetricTrail was not built.** It is the only candidate with metric scale, and the only one
  that could rescue the magnitude criterion without a depth model bolted on afterwards.
- Nothing about clips longer than 150 frames, other resolutions, or night/low-texture footage.

## Caveats

- One preparation race produced a stride-8 clip whose speed did not correlate with truth at
  any lag; `uzh/` was wiped and all five sequences re-prepared in a single run before this
  table. The withdrawn numbers are recorded in `docs/decisions.md`.
- Measured peak speeds exceed published ones (`indoor_forward_6` 18.5 vs 12.52 m/s): the truth
  is sampled at 15 Hz with nearest-sample lookup, so short peaks are not averaged out. The
  published value is the reference; the measured one is what the extractor saw.
- The up axis is taken from the flattest positional axis and required to be at least 2x
  flatter than the next; when that fails the clip's tilt numbers are flagged unreliable.
- `rel-rot err` compares *relative* rotations, so a constant camera-convention offset cancels;
  a large value is a genuine disagreement about rotation rate.
