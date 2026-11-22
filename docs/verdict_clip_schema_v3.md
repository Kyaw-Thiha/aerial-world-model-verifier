# verdict_clip schema v3 (Track K, 2026-11-22)

Schema v2 is unchanged and still loads.  v3 replaces the **required** side with the
required side of a named, fitted airframe model, stored **exactly** at the clip's frame
times, and stores the required **thrust axis** beside the lean because a lean cannot
represent an inverted axis.  Nothing in a v2 file is edited; v3 files are new files (a
parallel tree), and the v2 arrays are copied under `*_v2_*` names for provenance.

## Why v3

Two measured defects in v2's consumer contract, both in `reports/K01`:

1. **`lean_req` is the drag-free lean**, but the consumer form
   `lean_req + k_d·vel_req` is a first-order linearisation, not the flatness axis
   `normalise(a + g e3 + k_d v)`.  On 3,521 testable TII frames the linear and exact
   required sides differ by a median 2.17 deg and a p95 20.4 deg, and the linear vector
   can leave the unit disk (worst norm 1.06), where `up_body_from_lean` clips it.
2. **A lean cannot represent an inverted thrust axis.**  4,507 TII native frames have a
   required tilt above 85 deg, and some are inverted (`z < 0`): a path demanding downward
   thrust.  That is a physically meaningful (infeasible) requirement, and the angle between
   axes carries it while a lean round trip silently makes it upright.

v3 removes both: the required side is the model's own axis, and no `(k_d − 0.295)` or any
other rescaling is legal.

## Fields

| field | shape | units | meaning |
| --- | --- | --- | --- |
| `t`, `pos`, `yaw`, `depicted_lean`, `depicted_conf` | as v2 | — | unchanged |
| **`lean_req_v3`** | (N, 2) | rad-equivalent | model-required lean in the path-yaw frame, `Rz(−yaw)·[u_x, u_y]` for the model axis `u` |
| **`tilt_req_v3_deg`** | (N,) | deg | `degrees(arccos(u_z))`; values above 90 deg mean an inverted required axis |
| **`axis_req_v3`** | (N, 3) | unit vector | the model-required thrust axis in the world frame; **this is the field to score with** |
| `lean_req_v2_dragfree` | (N, 2) | rad-equivalent | the v2 drag-free lean, copied |
| `vel_req_v2` | (N, 2) | s | the v2 velocity regressor `v_h/g`, copied |
| `tilt_req_v2_deg` | (N,) | deg | the v2 drag-free tilt, copied |
| `meta.json` | — | — | all v2 metadata plus `schema_version: 3`, `model_name`, `model_parameters`, `fit_fold`, `required_lean_source`, and `lean_req_v3_semantics` |

`lean_req_v3`, `tilt_req_v3_deg` and `axis_req_v3` share `N` with `t` and are NaN on the
same frames.  NaN means outside the native Savitzky-Golay interior or a degenerate axis,
never extrapolated.

## How to compute them

1. Load the truth path and attitude at the native rate (TII 274.73 Hz, UZH 500 Hz).
2. Differentiate position with the project's window for that dataset (TII 0.37 s, UZH
   0.10 s, polyorder 3) to get `v`, `a`, `jerk`; mask the first and last half window.
3. Evaluate the fitted model axis `u(t)` (`src/airframe/models.py::axis_for`).  For M4 the
   implicit rotor-drag equation is solved by 12 fixed-point iterations.
4. Sample `u` at the clip's frame times by linear interpolation and renormalise; frames
   outside the valid interior are NaN.
5. `lean_req_v3 = Rz(−yaw)·[u_x, u_y]` in the **clip's** `yaw`; `tilt_req_v3_deg =
   degrees(arccos(u_z))`; `axis_req_v3 = u`.
6. Write a **new** directory; never write into a v2 tree.

Reference implementation: `src/airframe/schema_v3.py` (`sample_axis`,
`required_lean_at`, `write_v3_clip`, `export_tree`) and the generator
`scripts/export_schema_v3.py --fit results/k02_fit/<run>/fit.json`.  The generator writes
`results/k02_downstream/<run>/schema_v3_manifest.csv` with the sha256 of every file.

## Which model, and which fit

The generator uses `fit.json`'s `full_fit[<selected model>]`: parameters fitted on all
eight TII flights, i.e. a deployment copy.  The downstream comparison in `K02` uses the
**leave-one-flight-out** parameters from `fit.json`'s `lofo[<model>]['units']` so that no
clip's own flight contributes to its required side; those arrays are generated in memory
and are not part of the delivered tree.  Every file records which convention it used in
`fit_fold`.

## Backward compatibility

- A v2 clip loads and scores under the v2 contract exactly as before.  A v1 clip still
  works through `pos` differentiation.
- A v3 clip must not be fed to a v2 consumer: `lean_req` is absent and the semantics
  differ.  The v3 consumer scores the median tier as the angle between
  `depicted_lean`-reconstructed axis and `axis_req_v3`; the coherence tier uses the two
  lean arrays and masks frames whose tilt exceeds 85 deg (where a lean correlation is not
  defined).
- `pos` and `yaw` remain in both, so a v3 file can be audited against a v2 one without the
  model.
