# Track B report: ANWM inputs and first rollouts

Branch `track-b-generation` (worktree `../awmv-track-b`). Everything below is measured in this
repo unless tagged unverified. Compute: Modal only, app `awmv-b-*`, volumes `awmv-b-*`, R2
`track-b/`.

## Task 0: the three corrections

### 1. Audit table inconsistency (cause: aggregation, not data)

The table's "median" columns were the **mean of 33 per-trajectory medians** and its ">10 deg"
columns the **mean of per-trajectory fractions**. Per-trajectory medians are bimodal: 21 of the
33 trajectories have fewer than half their boundaries at transitions, so their median is exactly
0 deg, and the other 12 sit at 20+ deg. A mean of medians above 10 deg therefore does *not*
imply a majority of steps above 10 deg. Second, smaller cause: the SG ">10 deg" column used
`k_d = 0.297` while the SG median column was drag-free, so those two numbers were not even on the
same estimator.

Fix: `src/generation/audit.py::pooled_stats` computes median, p90 and fraction above the
threshold from **one pooled array**, and asserts the invariant `median > thresh => frac >= 0.5`;
`experiments/anwm_excitation_audit.py` now pools the raw per-trajectory arrays instead of
averaging scalars (the old scalar, kept as `per_trajectory_median_mean`, is a diagnostic only).
`experiments/anwm_audit_corrected.py` writes the corrected table.

Corrected numbers (33 test trajectories, one population per cell):

| dt | estimator | k_d | median | p90 | >10 deg |
| --- | --- | --- | --- | --- | --- |
| 1.0 s | per-step | 0 | 0.0 | 32.6 | 47.2% |
| 1.0 s | per-step | 0.295 | 8.6 | 36.1 | 47.2% |
| 1.0 s | SG-11 | 0 | 1.1 | 2.5 | 0.0% |
| 1.0 s | SG-11 | 0.295 | 4.9 | 8.4 | 1.8% |
| 0.667 s | per-step | 0 | 0.0 | 64.7 | 47.2% |
| 0.667 s | per-step | 0.295 | 12.7 | 62.3 | 76.6% |
| 0.667 s | SG-11 | 0 | 2.4 | 5.5 | 0.7% |
| 0.667 s | SG-11 | 0.295 | 7.4 | 12.7 | 28.0% |
| 0.25 s | per-step | 0 | 0.0 | 105.5 | 49.6% |
| 0.25 s | per-step | 0.295 | 31.0 | 106.1 | 79.0% |
| 0.25 s | SG-11 | 0 | 16.9 | 35.3 | 68.6% |
| 0.25 s | SG-11 | 0.295 | 23.9 | 42.7 | 82.0% |

Both numbers restated as asked: at dt = 1 s the **per-step median is 0.0 deg and 47.2% of steps
exceed 10 deg**; at dt = 0.25 s the **per-step median is 0.0 deg and 49.6% of steps exceed
10 deg**. The earlier "median 8.3 / 14.0 / 24.6 deg" figures are withdrawn: they are means of
per-trajectory medians, not medians of any step population.

### 2. The heading-jump statistic, defined

It is the change of **horizontal travel direction** between consecutive *translating* steps, not
the change of camera yaw. Recomputed on the same trajectories:

- |yaw change| per frame: median 0, p90 15, p99 15, max 15 deg; 0% above 30 deg.
- |travel-direction change| between successive translating steps: median 0, p90 90, p99 105,
  max 180 deg; 14.4% above 30 deg.

The quoted "p90 107 deg, p99 165 deg, 39% above 30 deg" is **not reproducible** under either
definition (the code that produced it was not kept) and is withdrawn. Only the travel-direction
change implies acceleration; camera yaw is unconstrained by flatness and legal at any rate. A
45 deg travel-direction change at zero yaw change is exactly AerialVLN's `MOVE_LEFT`, so this
quantity, not yaw, is what the verifier must watch.

### 3. Drag sentence and one `k_d`

The sentence "drag removes 6.6 deg on piloted flights and 2.2 on the 21 m/s ellipse" was a
misreading of a *residual* as a *reduction*. The table is right, and the corrected statement is:

- piloted: 6.57 -> 0.88 deg on 01p-ellipse (**5.68 deg removed**), 6.67 -> 0.94 on 07p-lemniscate
  (5.73), 7.17 -> 3.28 on 13p-trackRATM (3.89), 7.49 -> 3.89 on 16p-trackRATM (3.61);
- autonomous ellipses: 1.29 -> 1.26 and 1.19 -> 1.19 (**about 0.0-0.03 deg**), and the only
  autonomous flight drag helps is 13a-trackRATM (2.10 -> 0.96). There is no 2.2 deg reduction on
  any 21 m/s ellipse; the 20-22 m/s bin residual of about 11.5 deg is a *tail* statistic, not a
  reduction.

One `k_d` for both control modes: fitted separately on the Phase 0 derivation (w=101, pooled
median loss, bounds [0,3] 1/s) gives **piloted 0.296, autonomous 0.306, all flights 0.298**.
Cross-application moves the other group's median residual by 0.02 deg, so a single 0.295 is
valid; the drag *effect* differs by regime because the residuals differ, not the coefficient
(`results/anwm_excitation/kd_split.json`).

## Task 3 and Task 6: path library and path-only audit

`src/generation/paths.py` has one `PathSpec` (time, world position, yaw, metadata) with three
constructors. Actions are built by ANWM's own `_compute_actions`, `normalize_data` and
`get_delta_np` through `src/generation/anwm_env.py`, with one unit test that reproduces ANWM's
normalisation by hand and another that checks `get_delta_np`'s zero-padded first row.

Design caps (all from `third_party/anwm/config/data_config.yaml` / the AerialVLN action set):
5 m/s horizontal, 2 m/s vertical, step spacing inside the `action_stats` envelope, and 45 deg of
travel-direction change per step (above that the 1 s polyline aliases). **Regime C is checked
against the travel-direction cap rather than yaw rate**, because ANWM's own data contains 45 deg
strafes at zero yaw change; a `policy="course"` variant additionally holds yaw to 15 deg/s and is
reported where that makes the target unreachable.

Reachability at 5 m/s (drag-free peak required tilt from `src/physics`):

| maneuver | 5 deg | 10 deg | 20 deg |
| --- | --- | --- | --- |
| turn, yaw = course | yes | no (max 7.4) | no (max 7.4) |
| turn, yaw fixed | yes | yes | yes (19.9) |
| slalom, yaw fixed | yes | yes | no (max 11.0) |
| accelerate-then-brake | yes | yes | yes (19.9) |
| climb-and-turn, yaw fixed | yes | yes | yes |

The unreachable cells are a property of the caps, not of the generator: a coordinated turn at
5 m/s yaws at `g tan(tilt) / v`, so 20 deg of bank needs 41 deg/s of yaw, above ANWM's 15 deg/s
training step; a constant-speed slalom cannot exceed 11 deg because lateral speed has to come out
of the 5 m/s budget. Where a target is unreachable the design says so (`reachable: false`) and
reports the achieved value.

Path-only audit (`experiments/anwm_path_audit.py`, output `results/anwm_path_audit/audit.json`),
one pooled population per cell, all with `k_d = 0.295`:

| regime | dt | per-step median | per-step p90 | per-step >10 | SG median | SG >10 |
| --- | --- | --- | --- | --- | --- | --- |
| A native (33 traj) | 1.0 | 8.6 | 29.6 | 38.2% | 6.1 | 4.8% |
| A native | 0.667 | 12.7 | 62.4 | 77.4% | 9.2 | 40.7% |
| A native | 0.25 | 31.0 | 106.1 | 79.8% | 25.6 | 78.8% |
| B rescaled (25 windows) | 1.0 | 3.6 | 7.2 | 0.0% | 3.6 | 0.0% |
| B rescaled | 0.667 | 5.6 | 10.9 | 16.3% | 5.6 | 16.3% |
| B rescaled | 0.25 | 16.0 | 27.4 | 70.8% | 16.1 | 70.2% |
| C designed (22 designs) | 1.0 | 9.8 | 14.6 | 47.0% | 9.3 | 31.8% |
| C designed | 0.667 | 16.6 | 26.6 | 96.4% | 15.3 | 97.4% |
| C designed | 0.25 | 56.7 | 72.1 | 100% | 52.8 | 99.4% |

- **Smallest dt at which Regime A is consistent with level frames** (criterion: at most 5% of
  steps above 10 deg): **1.00 s** for the smoothed estimator (4.3% above 10 deg, median 6.1 deg).
  The per-step estimator never reaches the criterion on a 0.1-2.0 s grid, because the teleport
  transitions are permanent: 47% of boundaries are jumps at any dt. This is the sharpest version
  of the earlier finding and it confirms 1 s as the smallest defensible dt.
- **Regime B rescaling destroys most of the tilt signal.** A TII autonomous ellipse yaws at up to
  490 deg/s, so the 15 deg/s yaw cap forces a time stretch of about 33x (and the speed cap a
  length scale of 0.2-0.6). Median required tilt at 1 s falls from 66.8 deg (raw path at 1 s) to
  8.7 deg peak, median 3.6 deg; only 0% of steps exceed 10 deg at 1 s. A documented
  **off-envelope variant** (`yaw_rate_max = 60 deg/s`, clearly labelled off-expert) recovers a
  median of 4.6 deg and 1.6% of steps above 10 deg at 1 s. Consequence for the plan: Regime B as
  specified is essentially tilt-blind at 1 s and can carry depiction/integrity information but
  not a tilt verdict; the dose-response curve has to come from Regime C, with Regime B as the
  feasible-but-quiet control.

## Files

- `src/generation/audit.py`, `paths.py`, `anwm_env.py`, `experiments/anwm_{audit_corrected,kd_split,path_audit,build_clips}.py`
- `remote/b_anwm.py` (image, slim extraction, smoke, native validation, generation), `remote/b_native.py`
- `tests/test_anwm_audit.py`, `tests/test_generation_paths.py` (18 tests; full suite 53 passed, 1 skipped on the CPU env)
