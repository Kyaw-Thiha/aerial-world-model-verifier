# B02 — Corrected path audit: what ANWM's released paths and our designed paths demand

Track B. Written 2026-11-09. Branch `track-b-generation`.
Runs: `results/anwm_excitation/{audit,audit_corrected,kd_split}.json`
(commit **7107102**) and `results/anwm_path_audit/audit.json` (commit **ccf8d86**).
Reproduce: `python -m experiments.anwm_audit_corrected`, `python -m experiments.anwm_kd_split`,
`python -m experiments.anwm_path_audit` (all CPU; 2 s, 4 s, 15 s).

## Summary in plain language

If a drone flies a smooth curve, physics fixes how much it must lean: the lean is set by how hard
it is turning, how fast it is climbing and how much the air pushes back. ANWM's paths do not say
how long a step takes, so we sweep the candidate durations and ask how much leaning each one
demands. The released paths demand a lot: half of all between-frame steps are sharp changes, and
those changes require 27 to 83 degrees of lean depending on the assumed duration, while the
released frames show a perfectly level camera.

Our earlier table mixed two different averaging methods, which made two of its columns
contradict each other. This report fixes that, defines the "heading jump" statistic properly, and
adds the designed paths we will use for the controlled experiments. The headline for planning:
real racing flights squeezed into ANWM's gentle speed and turn limits demand almost no lean, so
they cannot carry a lean verdict at a one-second step; the designed maneuvers can.

## Question and setup

Questions:

1. How much required tilt does each input regime demand, at each candidate step duration?
2. How large are the heading jumps in the released data, and is that camera yaw or direction of
   travel?
3. Is a single drag coefficient valid for both piloted and autonomous real flights?

Setup:

- Released paths: 33 test-split trajectories, pose keys only, fetched by HTTP range
  (`data/raw/anwm/*.npz`, `scripts/fetch_anwm.py`).
- Real flights: 7 of the 8 Phase 0 TII flights, windows rescaled into ANWM's envelope
  (horizontal 5 m/s, vertical 2 m/s, yaw 15 deg/s).
- Designed paths: `src/generation/paths.py`, five maneuvers, two yaw policies, targets 5/10/20 deg
  at a nominal 1 s, action vectors built with ANWM's own `_compute_actions`, `normalize_data`,
  `get_delta_np`.
- Estimators: **per-step** (tilt demanded by each commanded velocity change) and **smoothed**
  (Savitzky-Golay over 11 frames, order 3). Drag model: `k_d = 0.295 /s`, world frame.

## Results

### 1. Released paths, one pooled population per cell

Per-step, drag-free / with `k_d`, n = 11,883 boundaries:

| dt | median | p90 | fraction > 10 deg | transition fraction | median among transitions |
| --- | --- | --- | --- | --- | --- |
| 1.0 s | 0.00 / 8.55 | 32.6 / 36.1 | 0.472 / 0.472 | 0.472 | 27.0 deg (drag-free) |
| 0.667 s | 0.00 / 12.70 | 64.7 / 62.3 | 0.472 / 0.766 | 0.472 | 48.9 deg |
| 0.25 s | 0.00 / 31.02 | 105.5 / 106.1 | 0.496 / 0.790 | 0.496 | 83.0 deg |

Smoothed (SG-11), drag-free / with `k_d`:

| dt | median | p90 | fraction > 10 deg |
| --- | --- | --- | --- |
| 1.0 s | 1.06 / 4.93 | 2.5 / 8.4 | 0.000 / 0.018 |
| 0.667 s | 2.39 / 7.44 | 5.5 / 12.7 | 0.007 / 0.280 |
| 0.25 s | 16.89 / 23.85 | 35.3 / 42.7 | 0.686 / 0.820 |

The per-step drag-free median is exactly 0 at every dt because 53% of boundaries are
constant-velocity cruise; the informative numbers are the transition fraction and the tilt among
transitions.

### 2. First run versus revised (the table that contradicted itself)

| quantity (dt = 1 s unless stated) | first run | revised | cause |
| --- | --- | --- | --- |
| per-step median | 8.34 / 14.02 / 24.58 deg | 0.00 / 0.00 / 0.00 deg | first run was the **mean of 33 per-trajectory medians** (bimodal: 0 for 21 trajectories, 20+ for 12), not the median of any population |
| per-step fraction > 10 deg | 0.459 / 0.459 / 0.483 | 0.472 / 0.472 / 0.496 | first run was the mean of per-trajectory fractions |
| smoothed ">10 deg" column | 0.022 / 0.296 / 0.802 | 0.000 / 0.007 / 0.686 (drag-free) | first run used `k_d` for this column and drag-free for the median column |
| "discrepancy vs depicted" | 4.96 / 7.57 / 24.26 | dropped | a mean of per-trajectory medians of \|tilt − 0\|; equals the smoothed tilt, listed above |

The old per-trajectory-mean diagnostics are kept in `audit_corrected.json` under
`per_step_median_spread` (mean-of-medians 8.34 / 13.87 / 24.58, median-of-medians 0.0 for all
three), so the old numbers remain recoverable in the run directory.

### 3. The heading-jump statistic, defined

Measured on the same 33 trajectories, absolute degrees:

| quantity | n | median | p90 | p99 | max | fraction > 30 deg |
| --- | --- | --- | --- | --- | --- | --- |
| camera yaw change per frame | 11,916 | 0 | 15 | 15 | 15 | 0.000 |
| travel-direction change between translating steps | 6,573 | 0 | 90 | 105 | 180 | 0.144 |

The old "p90 107, p99 165, 39% above 30 deg" matches neither and its code was not kept: withdrawn.
Only direction-of-travel changes require acceleration; camera yaw does not.

### 4. One drag coefficient for both control modes

Fitted on the Phase 0 derivation (Savitzky-Golay window 101, pooled-median loss, bounds [0, 3] /s):

| fit set | k_d (1/s) | median residual, piloted | median residual, autonomous |
| --- | --- | --- | --- |
| piloted only | 0.2962 | 1.163 | 1.322 |
| autonomous only | 0.3064 | 1.184 | 1.319 |
| all flights | 0.2984 | 1.164 | 1.320 |

Cross-application changes the other group's residual by 0.02 deg, so one coefficient is valid.
Per-flight drag-free → fitted (deg): 01a 1.29 → 1.26, 04a 1.19 → 1.19, 07a 2.94 → 2.93,
13a 2.10 → 0.96, 01p 6.57 → 0.88, 07p 6.67 → 0.94, 13p 7.17 → 3.28, 16p 7.49 → 3.89.
The earlier sentence "removes 6.6 deg on piloted flights and 2.2 on the 21 m/s ellipse" was a
misreading of a residual as a reduction; the table above is the corrected statement.

### 5. The three input regimes, all dt

Per-step, with `k_d`, one pooled population per cell:

| regime | n | dt | median | p90 | fraction > 10 deg |
| --- | --- | --- | --- | --- | --- |
| A native (33 trajectories) | 495 | 1.0 / 0.667 / 0.25 s | 8.55 / 12.71 / 31.02 | 29.6 / 62.4 / 106.1 | 0.382 / 0.774 / 0.798 |
| B rescaled (25 windows) | 20,077 | 1.0 / 0.667 / 0.25 s | 3.62 / 5.57 / 18.05 | 7.2 / 10.9 / 30.3 | 0.000 / 0.163 / 0.693 |
| B relaxed, 60 deg/s yaw cap (25 windows) | 4,992 | 1.0 / 0.667 / 0.25 s | 4.64 / 8.47 / 35.90 | 8.2 / 13.7 / 56.5 | 0.016 / 0.381 / 0.770 |
| C designed (22 designs) | 330 | 1.0 / 0.667 / 0.25 s | 9.83 / 16.62 / 56.69 | 14.6 / 26.6 / 72.1 | 0.470 / 0.964 / 1.000 |

- **Smallest dt at which the released paths are consistent with level frames** (criterion: at most
  5% of steps above 10 deg): **1.002 s** smoothed with drag, **0.585 s** smoothed drag-free; the
  per-step estimator never reaches the criterion on a 0.1–2.0 s grid because the teleports are
  permanent.
- **Effect of rescaling real flights**: median peak required tilt falls from 53.4 deg (raw path
  sampled at 1 s) to 8.7 deg; the per-window ratio has median 0.17. Rescaled speeds are 5 m/s by
  construction, but the time stretch (median factor 0.32 = λ/τ) is what removes the dynamics.

### 6. Designed-path reachability at 5 m/s, nominal 1 s (drag-free peak)

| maneuver | yaw policy | 5 deg | 10 deg | 20 deg |
| --- | --- | --- | --- | --- |
| turn | course-aligned | 5.0 | 7.4 (max) | 7.4 (max) |
| turn | yaw fixed | 5.0 | 10.0 | 19.9 |
| slalom | yaw fixed | 4.9 | 10.0 | 11.0 (max) |
| accelerate-then-brake | — | 5.0 | 9.9 | 19.9 |
| climb-and-turn | yaw fixed | 5.0 | 10.0 | 19.9 |

A coordinated turn at speed `v` yaws at `g·tan θ / v`, so 20 deg of bank at 5 m/s needs 41 deg/s,
above ANWM's 15 deg/s training step. Unreachable cells report `reachable: false`.

## What this supports

- The released paths are a teleport polyline; scoring them at transition events is the only
  defensible protocol, and 1 s is the smallest duration that does not contradict the level frames.
- A designed factorial can reach 5/10/20 deg of required tilt within ANWM's speed and step limits,
  but only with the yaw free (strafe geometry) or with longitudinal/vertical acceleration.
- Regime B (real flights rescaled) is quantitatively tilt-blind at 1 s: 0% of steps above 10 deg.

## What this does not show

- It does not establish the true per-step duration.
- It does not show what the model depicts; everything here is computed from commanded paths.
- It does not show that a 20 deg strafe is "in distribution" as a *picture*, only that its action
  vector is inside the trained action range.

## Caveats and open items

- The `>1 deg` "transition" mask is only meaningful drag-free: with drag, cruise already needs
  8.6 deg at 5 m/s, so the mask marks 76% of steps. Transitions are reported drag-free.
- Regime B windows were selected for motion (raw peak speed ≥ 5 m/s); stationary windows were
  dropped, which biases Regime B toward the faster parts of each flight.
- The relaxed Regime B variant is **off-distribution by construction** (60 deg/s yaw against a
  15 deg/s training step) and must be labelled as such wherever it is used.
- Open: whether the frame-level "level camera" convention survives outside scene 16.
