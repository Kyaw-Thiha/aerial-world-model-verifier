# Report 01 — Phase 0 follow-ups: the mocap physics floor and the oracle ladder

Track A. Written 2026-11-09. Every number below comes from the run directory named in
its table header; nothing here is a re-statement of a number from memory.

## Plain-language summary

If you know exactly where a drone went, second by second, you can work out how much it
*must* have tilted to get there: it has to lean to accelerate, and it has to lean a little
to fight air drag at a steady speed. We did that using only the motion-capture positions
of eight real races, then compared the answer with the tilt the drone actually had.

The answer is encouraging: **the "physics" part of our checker is accurate to about
1.2 degrees** once one drag coefficient is allowed. In plain terms, if a video's depicted
tilt is off by more than a few degrees, the motion is not the excuse — the picture is.
That is what makes the rest of the project possible: any large disagreement between a
video's tilt and its motion has to come from the measurement side or from the video.

The one important caveat is speed: at 20 m/s the same comparison is off by ~8.5 degrees on
average, because drag and motor dynamics at speed are not captured by a single number.

## The question and the setup

**Question.** Using only ground-truth motion, how well can we predict the drone's tilt?
This is the floor: no camera-based method can beat the error of the physics model itself.

**Setup.** 8 TII flights (4 autonomous, 4 piloted; ellipse, lemniscate and trackRATM
tracks; 6.7-19.1 m/s at p95). Mocap at 274.73 Hz. Positions differentiated twice with a
Savitzky-Golay filter (poly order 3) to get velocity and acceleration; the implied thrust
axis is `normalise(a + g·e_z + k_d·v)` with `g = 9.81`. One drag coefficient `k_d` per
fold, fitted by minimising the **median** residual (so dropouts cannot drag it). Split by
flight, leave-one-flight-out. The filter window (11-401 samples) and `k_d` are chosen on
the *fit* flights only.

**First run and revision.** The window and `k_d` selection has not been revised; the only
change after the first run was a *reporting* fix (medians and percentiles reported
alongside RMSE). No threshold or fit rule changed.

## Results

### The floor (run `results/phase0_floor/20261108T044153Z_f3b5d2bf`, commit `f3b5d2bf`)

114,059 held-out frames, pooled, leave-one-flight-out:

| estimator | median (deg) | p90 | p95 | mean |
| --- | --- | --- | --- | --- |
| flatness + fitted drag (the gate) | **1.244** | 4.387 | 6.494 | 2.129 |
| flatness, drag-free | 5.461 | 11.577 | 13.873 | 6.392 |
| body-frame x/y drag (diagnostic only) | 1.220 | 4.474 | 6.716 | 2.175 |

Drag is worth **4.22 degrees** of median residual. The body-frame diagnostic does not beat
the scalar model (1.220 vs 1.244), so the residual is not a body-vs-world drag-frame
mismatch.

Per-fold fitted values (each fold's `k_d` and window are recorded in `loo_folds`):

| flight held out | window (samples) | `k_d` (1/s) | held-out median (deg) | implied tilt above 10 deg |
| --- | --- | --- | --- | --- |
| 01a-ellipse | 101 | 0.297 | 1.26 | 0.40 |
| 04a-ellipse | 101 | 0.298 | 1.19 | 0.31 |
| 07a-lemniscate | 101 | 0.297 | 2.94 | 0.32 |
| 13a-trackRATM | 101 | 0.297 | 0.96 | 0.39 |
| 01p-ellipse | 101 | 0.304 | 0.94 | 0.95 |
| 07p-lemniscate | 101 | 0.290 | 0.95 | 0.93 |
| 13p-trackRATM | 101 | 0.295 | 3.28 | 0.95 |
| 16p-trackRATM | 101 | 0.296 | 3.88 | 0.97 |

All eight folds chose the 101-sample window (0.37 s) and `k_d` in 0.290-0.304. That
stability is what a real physical coefficient looks like.

### Residual against speed (same run, `residual_curves`, n-weighted across flights)

| speed (m/s) | 0-2 | 4-6 | 8-10 | 12-14 | 16-18 | 20-22 |
| --- | --- | --- | --- | --- | --- | --- |
| frames | 31,152 | 22,619 | 5,264 | 3,236 | 2,062 | 613 |
| median residual, fitted drag (deg) | 1.32 | 1.29 | 3.45 | 4.04 | 5.09 | 8.50 |

The drag term fixes the median at low speed and does not fix the tail at high speed
(per-flight high-speed p95 is 16.8 deg with the scalar drag model, 20.4 with body drag).

### How much of the footage carries a usable signal

| flight type | frames with implied tilt ≥ 10 deg |
| --- | --- |
| autonomous ellipse / lemniscate (01a, 04a, 07a, 13a) | 0.31-0.40 |
| piloted (01p, 07p, 13p, 16p) | 0.93-0.97 |

This is the origin of the abstention rule used later: in the autonomous flights, most of
the frame budget is near-level cruise, which cannot test a tilt violation.

### The oracle ladder, row 1 (the row missing from the Phase 1/2 summary)

| rung | number | what it is |
| --- | --- | --- |
| mocap position + mocap attitude | **1.24 deg** median residual | this run; the physics floor |

The rest of the ladder lives in report 03, which is where the position and attitude
sources are swapped for estimates.

## What this supports

- A physics-only floor of ~1.2 degrees, so the verifier's *model* is not the limiting
  error at moderate speed; a several-degree disagreement between video and motion can be
  attributed to the measurement or to the video.
- `k_d` = 0.295 /s is a stable, physically plausible coefficient, which is why later
  phases fix it rather than re-fitting it per clip.
- The speed curve and the testable fraction justify (a) speed-binned thresholds and
  (b) an abstention rule rather than scoring every clip.

## What this does not show

- Nothing about pixels. This is the best possible with perfect knowledge of the path.
- Not a validation of the drag model: 8.5 degrees at 20 m/s says the single-coefficient
  model is incomplete (blade flapping, thrust dynamics, mocap noise on the rigid body are
  the candidates, none tested here).
- The 0.37 s window low-passes the trajectory. A camera-rate estimator cannot reach this
  floor: on the fit flights the same sweep gives 4.16° at 11 samples (0.04 s), 2.09° at 21,
  1.59° at 31 (0.11 s) and 1.34° at 51 (`window_sweep_on_fit_flights` in every fold). The
  11-31-sample range is the fairer reference for later phases, and it is still shorter than
  a 12 Hz camera allows: later phases work at 0.42 s (5 samples) and see much larger error.

## Caveats and open items

- **Inferred, not measured:** that the remaining high-speed error is aerodynamic rather
  than numerical. No independent source of acceleration (IMU, motor thrust) was compared.
- A small constant attitude offset is visible on the two autonomous ellipse flights
  (best single rotation 3.86 and 3.94 deg, `frame_offset_deg`); it is reported, not
  subtracted.
- The per-frame arrays (`frames/<flight>.npz`) are gitignored; the run directory holds
  `metrics.json`, the config, the plots and the git hash.
