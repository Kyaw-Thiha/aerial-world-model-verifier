# verdict_clip schema v2 (Track D, 2026-11-14)

Schema v1 is unchanged and still loads.  v2 adds two arrays that move the **required**
side of the verdict onto the dense-rate truth, and one convenience array.  `pos` and
`yaw` are kept: they are the provenance, the fallback, and the only path truth the
statistic is allowed to read.

## Fields

| field | shape | units | meaning |
| --- | --- | --- | --- |
| `t` | (N,) | s | frame times, strictly increasing |
| `pos` | (N, 3) | m | commanded path, world z-up (mocap or ground truth) |
| `yaw` | (N,) | rad | body heading about world z |
| `depicted_lean` | (N, 2) | rad-equivalent | depicted lean in the yaw-aligned frame, `l = (cos φ sin θ, −sin φ)`; NaN where the estimator failed |
| `depicted_conf` | (N,) | estimator-specific | per-frame confidence; for GeoCalib `latitude_confidence_mean`, higher is better; NaN allowed |
| **`lean_req`** | (N, 2) | rad-equivalent | **drag-free required lean** `Rz(−yaw) [u0_x, u0_y]`, `u0 = normalize(a + g e3)` |
| **`vel_req`** | (N, 2) | s | **velocity regressor** `Rz(−yaw) v_h / g` (multiply by a drag coefficient in 1/s) |
| `tilt_req_deg` | (N,) | deg | optional required tilt `arccos(u0_z)`; recomputed from `lean_req` when absent |
| `meta.json` | — | — | dataset, sequence/flight, clip id, condition, fps, mount source and uptilt, estimator and variant, truth-frame notes, `schema_version: 2`, `required_lean_source`, `required_window_samples`, `required_dt_s` |

`lean_req` and `vel_req` must be supplied together.  They are **sampled at the frame times
`t`**, so all arrays share `N`.  `tilt_req_deg` is optional because
`tilt = arcsin(|lean_req|)` for a unit thrust axis.

## How to compute them (the definition, not an implementation detail)

1. Take the commanded path and heading at their **native** rate (TII mocap: 274.73 Hz).
2. Differentiate position twice with a Savitzky-Golay window fixed in **seconds**:
   **0.37 s**, polyorder 3.  This is the Phase 0 window (101 native samples); a window in
   samples would change meaning with the sample rate.
3. `u0 = (a + g e3) / ||a + g e3||`, `g = 9.81 m/s²`.
4. `lean_req = Rz(−yaw) [u0_x, u0_y]`, `vel_req = Rz(−yaw) [v_x, v_y] / g`.
5. Sample at the frame times by linear interpolation.  Frames outside the native SG
   interior (half a window at each end) are **NaN, never extrapolated**.
6. `tilt_req_deg = degrees(arccos(u0_z))` sampled the same way.

The statistic then fits `l_dep = G1 · lean_req + G2 · vel_req + b`: the acceleration gain,
a drag nuisance in 1/s, and a free 2-D mount intercept.  The scorer never differentiates
`pos` at the 12 Hz clip rate when `lean_req` is present (it records
`GainResult.method.required_source = "lean_req"`).

Reference implementation: `src/verdict/dense.py::dense_required_lean` and
`src/verdict/dense.py::truth_lean`.

## Why v2

The stored TII per-frame grid is 12 Hz (stride 10).  At 12 Hz a 0.37 s window is five
samples, and the second derivative is noisy enough to attenuate the gain.  On dense truth
the decomposition (report D02) gives a formula gain of **0.991** [0.935, 1.010] for truth
attitude on required lean, against a v1 real-clip gain of 0.745.  Most of the remaining
gap is estimator shrinkage, not differentiation; dense required lean raises the verdict
gain only from **0.745 to 0.764** [0.718, 0.818].  The arrays are in the schema because
the consumer must not have to guess which side was noisy.

## Backward compatibility

- A v1 clip (no `lean_req`) loads and scores exactly as before: the scorer differentiates
  `pos` at `t` with the configured `sg_window_s` (0.37 s), recording
  `required_source = "pos"`.
- A v2 clip must have both `lean_req` and `vel_req`; the loader rejects one without the
  other.
- `pos` and `yaw` are never optional and never removed.  No other mocap-derived quantity
  is allowed in a clip.

## What Track E must re-export

UZH-FPV clips should be re-exported in v2 with:

- `lean_req`/`vel_req` computed from the UZH ground-truth position and heading at the
  ground-truth rate, with the **same 0.37 s / poly-3** window in seconds, sampled at the
  camera frame times;
- `tilt_req_deg` (optional but preferred);
- `meta.json` with `schema_version: 2`, the estimator and variant, the mount source and
  rotation, and `required_lean_source`;
- `depicted_lean` and `depicted_conf` exactly as in v1.

Score with the frozen `configs/gain_frozen.yaml` **unchanged**; do not recalibrate the
band on UZH.  If the ground-truth rate is below about 30 Hz, record it and report the
window in samples; a 0.37 s window that resolves to fewer than five samples cannot carry
a second derivative and the clip must be marked not testable rather than scored.

## Example writer

```python
import numpy as np
np.savez_compressed(
    "verdict_clip.npz",
    t=t, pos=pos, yaw=yaw,
    depicted_lean=depicted_lean, depicted_conf=depicted_conf,
    lean_req=lean_req, vel_req=vel_req, tilt_req_deg=tilt_req_deg,
)
```

`meta.json` is a sibling file.  `src/verdict/schema.py` validates the result, and
`scripts/score_clip.py` is the only consumer entry point.
