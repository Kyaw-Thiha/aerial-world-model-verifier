# ANWM clip set: format for scoring

Track B writes generated flights here so Track A can run the verifier on them without reading
any ANWM code. Everything is on R2 under `track-b/clips/` (mirrored from
`data/processed/anwm_clips/`) and indexed by `clips_index.csv`.

## Layout

```
data/processed/anwm_clips/
  clips_index.csv          one row per clip (see below)
  <regime>/<clip_id>/
    frames/00.png .. 15.png     generated frames, 224x224 RGB, one per commanded step
    manifest.json               commanded path, start state, provenance
    gt/                         Regime A only: the recorded frames the model was compared to
    contact_sheet.png           Regime A only: ground truth row over prediction row
```

`<regime>` is `A` (native commands), `B` (real-flight replay, rescaled) or `C` (designed
maneuvers). `clip_id` is unique across the set.

## manifest.json

```json
{
  "id": "C_turn_10_fixed_0",
  "regime": "C",
  "start_state": {"traj": "3KA7IJSN..._processed", "time": 79},
  "commanded_path": {
    "positions": [[x, y, z], ... 17 rows, world frame, metres, z-up],
    "yaw": [... 17 values, radians, heading about world z],
    "step_spacing_m": 5.0
  },
  "designed_required_tilt_deg": 10.0,
  "seed": 0,
  "sampling_steps": 250,
  "slim_checkpoint_sha256": "...",
  "code_commit": "...",
  "frames": 16,
  "visual_integrity": {"frame_std": [...], "temporal_delta": 0.01},
  "meta": {"maneuver": "turn", "policy": "fixed", "reachable": true, ...}
}
```

Notes that matter for scoring:

- **Context is 16 frames.** The released checkpoint was trained with `context_size = 16`
  (its `pos_embed` has 17 rows), not the 4 in the shipped config; every clip here uses 16.
- **One frame per commanded step.** `frames/i.png` is the model's depiction *after* applying
  the action that ends at `positions[i+1]`. There is no fps in the release; the working
  assumption is 1 s per step (unverified). The commanded path is in the manifest so the
  required tilt can be recomputed at any dt with `src/generation/paths.py` without
  regenerating anything.
- **Tilt the path demands, not the tilt the path depicts.** ANWM's released frames are exactly
  level (roll = pitch = 0) by construction, so for Regime A the recorded frames are a *level*
  reference; do not treat them as ground-truth attitude.
- **Regime C is designed, Regime A is native, Regime B is recorded-and-rescaled.** Regime B was
  rescaled into ANWM's trained envelope (5 m/s horizontal, 2 m/s vertical, 15 deg/s yaw), and
  `meta` records `length_scale_lam`, `time_scale_tau` and `accel_scale_lam_over_tau2`, so a
  scorer can undo the rescaling if it wants the original flight.
- `visual_integrity.frame_std` is the per-frame standard deviation of the generated pixels and
  `temporal_delta` the mean absolute frame-to-frame difference. They are there to flag collapse
  (near-zero temporal delta) and not a score.
- The `Clip` schema is unchanged; these manifests are a superset of the fields a clip needs
  (frames, fps, intrinsics, uptilt) and the intrinsics for ANWM are the recorded
  `K = [[256,0,256],[0,256,256],[0,0,1]]` at 512x512 before the 224x224 resize, with an
  exactly level camera.

## clips_index.csv

Columns: `clip_id, regime, maneuver, policy, designed_tilt_deg, reachable, n_frames,
peak_speed_ms, peak_yaw_rate_deg_s, step_spacing_m, required_tilt_1s_drag_free_deg,
required_tilt_1s_kd_deg, start_traj, start_time, seed, checkpoint_sha256`.

`variance.json` in the same directory holds pairwise frame differences for the clips that were
generated with three seeds.

`required_tilt_*` is the peak per-step required tilt of the commanded path at 1 s, from
`src/generation/paths.py`; the full dt sweep is in `results/anwm_path_audit/audit.json`.

## Recomputing the commanded tilt for a clip

```python
import json, numpy as np
from src.generation.paths import PathSpec, K_D

m = json.load(open("data/processed/anwm_clips/A/A00/manifest.json"))
cp = m["commanded_path"]
spec = PathSpec(name=m["id"], regime=m["regime"],
                pos=np.array(cp["positions"]), yaw=np.array(cp["yaw"]))
for dt in (0.25, 2 / 3, 1.0):
    print(dt, spec.peak_tilt(dt, 0.0), spec.peak_tilt(dt, K_D))   # peak per-step, deg
print(spec.audit())   # median / p90 / fraction above 10 deg per dt and estimator
```

`PathSpec.audit()` returns one pooled population per cell and asserts the
`median > 10 deg => fraction >= 0.5` invariant, which is the fixed version of the audit that
started this handoff.
