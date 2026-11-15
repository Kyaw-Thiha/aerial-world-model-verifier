# B05 — Pose injection, Stage 0: conventions verified before spending GPU time

Track B addendum, 2026-11-10. Branch `track-b-generation`, commits `640de4c` (code) and
`1f720f5` (results). Run: `results/anwm_injected/stage0.json`,
`results/anwm_injected/stage0_determinism.json`, `results/anwm_injected/stage0_summary.json`,
contact sheets in `results/anwm_injected/stage0/`.

## Summary in plain language

We want to hand the model a *tilted* view of the future and see whether the frames it draws bank
with it. Before generating anything, this stage checks that we can build a tilted camera pose that
the release's own projection code interprets the way we intend. Everything checks out except one
thing: the same clip generated twice from the same seed is **not identical**, so a fresh baseline
has to be part of the plan. That changes the cost, which is where this stage stops for approval.

## Question and setup

Question: is our construction of a tilted camera pose consistent with ANWM's conventions, and is
generation reproducible enough to pair an injected clip with an existing level clip?

- Data: one released test trajectory (`3KA7IJSNXE3LKLTL2ARA5RQP4ACBP0_processed`, 269 frames,
  depth 512x512 in metres, every pitch and roll exactly 0), pulled to
  `data/raw/anwm_full/`.
- Designs: the 15 Regime C designs (report B04), rebuilt from their parameters.
- No generation except the three-clip determinism check below.

## Results

### Conventions (read from source, verified against data)

| property | finding | evidence |
| --- | --- | --- |
| `traj_data["pose"]` | **camera-to-world** | `points_world = pose_src @ p_cam`, then `inv(pose_dst) @ points_world` (`anwm/projection.py:227-233`) |
| camera axis change | `R_ec` recovered from the released poses equals the notebook's `t_ec` | max diff 0.0; spread 2.2e-16 over 269 frames |
| optical axis | camera +z maps to body +x (forward) | `t_ec` column 2 |
| pose translation | equals `point` exactly | max diff 0.0 |
| release Euler helper | `euler_to_quaternion(yaw, pitch, roll)` = `scipy Rotation.from_euler("zyx", ...)` = **`Rx(roll) @ Ry(pitch) @ Rz(yaw)`**, i.e. extrinsic z-y-x | numeric check against `Rz@Ry@Rx` (diff 0.06) and `Rx@Ry@Rz` (2e-16) |
| injected convention | `R_wb = Rz(yaw) @ Ry(theta) @ Rx(phi)` (intrinsic), pose built from the matrix, not from Euler angles | `src/generation/attitude.py` |
| roll sign | a body roll of `+phi` rotates the image by **`-phi`** exactly | homography check at +5, -5, +20 deg |

The handoff's formulas `phi = -asin(z'_y)`, `theta = atan2(z'_x, z'_z)` with
`z' = Rz(-yaw) z_B` hold for the intrinsic construction above; they do **not** describe what
`euler_to_quaternion` expects, so the pose is assembled from the rotation matrix and the release
triple is only reported for reference (it reconstructs the same matrix to 1e-16).

### Test 1 — pure-rotation reprojection equals `K R K^-1`

| case | point-level max error | image-level agreement |
| --- | --- | --- |
| synthetic, roll 12 deg | 2.8e-14 px | n/a (random texture) |
| synthetic, pitch -10 deg | 2.3e-14 px | n/a |
| synthetic, mixed 8/-6/3 deg | 2.3e-14 px | n/a |
| **real frame, roll 12 deg** | **1.7e-7 px** over 262,144 points | **NCC 0.979** against a `cv2.warpPerspective` by the same homography (overlap 0.287) |

A first version of this test used the real 512x512 `K` with a 64x64 synthetic image, which put
every point outside the frame and reported a 19,473 px error; that is recorded here because it
looks like a convention failure and is not one.

### Test 2 — the pose's body z-axis is the intended thrust axis

Max error over all 15 designs and all 16 target steps: **2.2e-16**. The recovery goes through the
pose matrix only (`R_wb = R_wc @ R_ec^T`, third column), so it exercises the same matrix FFP uses.

### Test 3 — signs

| design | tilted steps | min `dot(a_xy, z_B,xy)` | max roll | max pitch | max tilt |
| --- | --- | --- | --- | --- | --- |
| turn 5, course | 17 | 0.075 | 5.01 | 0.00 | 5.01 |
| turn 5, fixed | 17 | 0.075 | 5.01 | 5.01 | 5.01 |
| turn 10, fixed | 17 | 0.307 | 10.12 | 9.99 | 10.12 |
| turn 20, fixed | 17 | 1.325 | 20.81 | 20.75 | 20.81 |
| slalom 5, course | 15 | 0.0023 | 5.14 | 0.75 | 5.14 |
| slalom 5, fixed | 15 | 0.0023 | 5.14 | 0.00 | 5.14 |
| slalom 10, fixed | 16 | **0.0015** | 10.13 | 0.00 | 10.13 |
| accel 5 | 12 | 0.055 | 0.00 | 4.97 | 4.97 |
| accel 10 | 6 | 0.178 | 0.00 | 9.90 | 9.90 |
| accel 20 | 4 | 0.213 | 0.00 | 19.87 | 19.87 |
| climb 5, course | 17 | 0.066 | 5.01 | 0.00 | 5.01 |
| climb 5, fixed | 17 | 0.066 | 5.01 | 5.01 | 5.01 |
| climb 10, fixed | 17 | 0.270 | 10.12 | 9.99 | 10.12 |
| climb 20, fixed | 17 | 1.177 | 20.81 | 20.75 | 20.81 |
| cruise control | 0 | — | 0.00 | 0.00 | 0.00 |

The horizontal thrust component points along the acceleration in every tilted step (all dots
positive; the smallest are slalom steps near the turning points where the acceleration itself is
almost zero). Accelerate-then-brake is pure pitch, turns are pure roll when the yaw follows the
course, and a turn with fixed yaw splits the same tilt between roll and pitch as the acceleration
direction rotates relative to the airframe. The injected peak (analytic, continuous) is 0.9 deg
above the commanded per-step peak for the 20 deg cells (20.81 vs 19.94) because the design was
calibrated on the step-averaged estimator; the injected value is the physical one.

Contact sheets (level prior on the top row, injected on the bottom):
`priors_turn_10_fixed.png`, `priors_turn_20_fixed.png`, `priors_slalom_10_fixed.png`,
`priors_accel_10.png`, `priors_accel_20.png`, `priors_climb_20_fixed.png`,
`priors_cruise_ctl.png`. Mean absolute difference between the rows: cruise **0.00** (the
injection is a no-op at zero tilt, a useful internal check), accel 10/20 17.2/14.3, slalom 10
32.6, turn 10 41.1, climb 20 52.9, turn 20 63.6.

### Test 4 — hole coverage of the projected prior

Fraction of the prior image that is filled, level versus injected, mean and minimum over the 16
target steps:

| design | max tilt | level mean / min | injected mean / min |
| --- | --- | --- | --- |
| turn 10, fixed | 10.1 | 0.555 / 0.479 | 0.503 / 0.404 |
| turn 20, fixed | 20.8 | 0.663 / 0.578 | 0.574 / 0.441 |
| slalom 10, fixed | 10.1 | 0.587 / 0.482 | 0.582 / 0.484 |
| accel 10 | 9.9 | 0.565 / 0.387 | 0.530 / 0.277 |
| accel 20 | 19.9 | 0.550 / 0.375 | **0.513 / 0.148** |
| climb 20, fixed | 20.8 | 0.474 / 0.370 | 0.395 / 0.217 |
| cruise control | 0.0 | 0.507 / 0.364 | 0.507 / 0.364 |

Tilting costs coverage, as expected: 6-17% of the mean, and the worst single step of the 20 deg
accelerate-then-brake drops to 15% filled. That is a real limitation of injecting attitude into a
partial view and it must travel with any result from this arm.

### Test 5 — determinism (the blocker)

Two delivered level clips were regenerated with the same start, path, seed, checkpoint hash and
250 sampling steps:

| clip | mean abs pixel diff (0-255) | max | fraction of pixels > 1/255 |
| --- | --- | --- | --- |
| `C_cruise_ctl_course_0` | **2.40** | 223 | 33.1% |
| `C_turn_10_fixed_0` | **1.09** | 149 | 18.9% |

**Corrected the same day: generation is bit-reproducible.** Re-running the same check against the
current code gave **byte-identical PNGs on all 16 frames for 8 of 8 regenerated clips**
(`results/anwm_injected/repeat_noise.json`); the numbers above are not reproducible and are
withdrawn, most likely because the comparison ran while the producing app was still writing the
directory. The first-run values are kept here because they drove a decision for an hour. The difference is 10-20x smaller than seed-to-seed variability (0.088-0.196 in 0-1 units,
i.e. 22-50/255) but it is not zero, so a paired comparison against the existing level clips
carries a repeat noise of about 1-2.4/255 per clip.

A third clip was requested with the wrong id (`C_accel_10_course_0` instead of
`C_accel_brake_10_course_0`) and silently did not match, so two clips were compared, not three.

## What this supports

- The tilted-pose construction is verified at machine precision against ANWM's own projection code
  and against the released data; the injected attitude is the flatness attitude of the commanded
  path, with the correct sign.
- Injecting attitude is a true no-op at zero commanded tilt, so the cruise controls are a clean
  null.

## What this does not show

- Nothing about whether the model follows the prior. That is handoff 04's measurement; no
  depicted tilt was scored here.
- Nothing about 20 deg attitudes being "in distribution" for the model: it has only ever seen
  level poses, so the injected arm is off-distribution by construction.

## Caveats and open items

- **Three delivered level clips cannot be reproduced by the current design code**
  (`C_slalom_5_course_0..2`): rebuilding the slalom with `policy="course"` gives positions 16.2 m
  away from the stored path, while the `fixed` policy reproduces exactly. The stored path is what
  the model was conditioned on, so injecting a "designed" acceleration into it would describe a
  different trajectory. They are excluded from the injected arm: 46 of 49 C clips are injectable.
  This also means `results/anwm_clips/specs.json` reproduces for 43 of 46 design clips, not all.
- Coverage loss at 20 deg is large on some steps (0.148 minimum); if the injected arm is used for
  a verdict, coverage must be reported per clip.
- Open: the 3 slalom-course clips need either a rebuilt design or an audit of how their spec was
  produced.
