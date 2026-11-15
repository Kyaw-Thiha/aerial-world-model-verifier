# Injected-attitude clip set (`clips_injected`)

Track B addendum, 2026-11-10. Companion to `docs/track_b_clips.md`; read that first.

## What is different from the level set

Every clip in the level set is generated with **level** target poses, so the only place the
commanded path can affect the picture other than through the action vector is the reprojected
future frame (FFP). In this set the FFP target poses carry the **attitude the commanded path
requires** (flatness: the body z axis is the thrust axis `(a + g e3)/||.||`). Everything else —
start state, positions, yaw, seed, sampling steps, checkpoint — is identical to the paired level
clip, which is the baseline.

Arms:

| arm | thrust | status |
| --- | --- | --- |
| `level` | not injected (existing set) | baseline, `data/processed/anwm_clips/` |
| `injected_true` | drag-free flatness thrust axis | generated |
| `injected_mirrored` | horizontal part of the thrust axis flipped (lean away) | built, not generated |

## Layout

```
data/processed/anwm_clips_regen/C/<clip_id>/     same-seed level twins (9 clips, kept beside the originals)
data/processed/anwm_clips_injected/<arm>/C/<clip_id>/
  frames/00.png .. 15.png     generated frames, 224x224
  priors/00.png .. 15.png     the FFP image the model was conditioned on for that step
  manifest.json
  strip.png                   contact sheet (frames)
  priors_strip.png            contact sheet (priors: level above, injected below, when built)
```

R2 mirrors the whole tree under `track-b/clips_injected/` (`injected_true/`, plus
`regen_level/` for the twins), with `clips_injected_index.csv` and the evaluation artifacts
(`eval_injected_injected_true.json`, `coverage_by_cell.csv`, `prior_difference.json`,
`repeat_noise.json`, `stage0_summary.json`) at its root.

`<clip_id>` equals the level clip id for `injected_true` and `<id>__<seed>` for
`injected_mirrored`, so the pairing is a string match.

## manifest.json fields added for this arm

| field | meaning |
| --- | --- |
| `arm` | `level` / `injected_true` / `injected_mirrored` |
| `injected_roll_deg`, `injected_pitch_deg` | per-step angles in the convention below (17 values, index 0 unused) |
| `convention` | free-text identifier of the convention version |
| `paired_level_id` | the level clip this one is paired with |
| `slim_checkpoint_sha256`, `sampling_steps`, `seed`, `code_commit` | as in the level set |
| `prior_coverage_per_step` | 16 values: fraction of the **512x512** FFP image that is filled at each target step (the definition used in Stage 0) |
| `prior_coverage_mean`, `prior_coverage_min` | summary of the above, for stratification |

The spec file `results/anwm_clips_injected/specs.json` also carries the full thrust vectors
`(17,3)` and the analytic acceleration used to build them.

## Convention (verified, see reports/b05)

* `traj_data["pose"]` is **camera-to-world**: FFP computes `world = pose_src @ p_cam` then
  `p_dst = inv(pose_dst) @ world`.
* `R_ec` recovered from the released poses equals the notebook's `t_ec` (rotation only) to
  2.2e-16; the optical axis (camera +z) maps to body +x.
* Tilted poses are built as `R_wc = Rz(yaw) @ Ry(theta) @ Rx(phi) @ R_ec` (instrinsic ZYX) and the
  roll/pitch follow from `z_B` as `phi = -asin(z'_y)`, `theta = atan2(z'_x, z'_z)` with
  `z' = Rz(-yaw) z_B`.
* **Do not** feed `(yaw, theta, phi)` to `anwm.projection.euler_to_quaternion`: that function is
  `scipy Rotation.from_euler("zyx")`, which is *extrinsic* z-y-x (`Rx @ Ry @ Rz`). The pose in
  this set is built from the rotation matrix, and the equivalent release triple is reported
  separately only for completeness.

## Pairing table

`clips_injected_index.csv` (next to the arm folders, mirrored to R2) has one row per clip with:
`clip_id, arm, maneuver, policy, designed_tilt_deg, commanded_tilt_1s_drag_free_deg,
injected_tilt_drag_free_deg, injected_roll_max_deg, injected_pitch_max_deg, paired_level_id,
level_source, repeat_noise_mae, prior_cov_level_mean, prior_cov_level_min, prior_cov_inj_mean,
prior_cov_inj_min, seed, checkpoint_sha256, code_commit, n_frames`.

`level_source` is `regenerated` for the nine clips that have a same-seed twin on disk
(`data/processed/anwm_clips_regen/`), whose mean absolute pixel difference from the original is
`repeat_noise_mae` — the null for an injected-vs-level comparison. `prior_cov_*` for the delivered
level clips comes from `coverage_level_original.json` (recomputed offline, since those manifests
predate the field).

## Reading a clip

```python
import json, numpy as np
m = json.load(open("data/processed/anwm_clips_injected/injected_true/C/<id>/manifest.json"))
roll = np.array(m["injected_roll_deg"][1:])     # step 1..16, radians? no: degrees
pitch = np.array(m["injected_pitch_deg"][1:])
path = m["commanded_path"]["positions"]         # 17 waypoints, world frame
```

## How to use this set (for the scoring step)

1. Join `clips_injected_index.csv` on `clip_id` with the level clips' own index; the `arm` column
   separates `injected_true` from `level`.
2. Compare each injected clip with its `paired_level_id`. Where `level_source` is `regenerated`,
   the same-seed twin difference (`repeat_noise_mae`, measured in 0-255 pixel units) is the null:
   any injected-vs-level difference below it is indistinguishable from reseeding the sampler.
3. Stratify by `prior_cov_inj_min`. Four of the 49 clips have blank priors for late steps
   (`C_turn_5_course_0/1`, `C_climb_turn_5_course_0/1`); their injected frames beyond the first
   empty step were conditioned on nothing and cannot support a prior-preservation claim.
4. The injected attitude is off-distribution by construction: the released model saw only level
   poses. A null result (frames stay level) is informative about the architecture; a positive one
   is evidence that the prior channel carries attitude.
