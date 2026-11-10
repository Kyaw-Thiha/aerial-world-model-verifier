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
data/processed/anwm_clips_injected/<arm>/C/<clip_id>/
  frames/00.png .. 15.png     generated frames, 224x224
  priors/00.png .. 15.png     the FFP image the model was conditioned on for that step
  manifest.json
  strip.png                   contact sheet (frames)
  priors_strip.png            contact sheet (priors: level above, injected below, when built)
```

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

## Reading a clip

```python
import json, numpy as np
m = json.load(open("data/processed/anwm_clips_injected/injected_true/C/<id>/manifest.json"))
roll = np.array(m["injected_roll_deg"][1:])     # step 1..16, radians? no: degrees
pitch = np.array(m["injected_pitch_deg"][1:])
path = m["commanded_path"]["positions"]         # 17 waypoints, world frame
```
