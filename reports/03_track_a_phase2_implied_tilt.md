# Report 03 — Phase 2: implied tilt from the recovered path, and the oracle ladder

Track A. Written 2026-11-09. Every number is attributed to a run directory and commit.
**This report contains a correction**: the first Phase 2 run's window sweep was inert (all
windows collapsed to the same 5-sample filter). Both sets of numbers are given.

## Plain-language summary

Phase 2 asks the harder half of the question: instead of *reading* the tilt from the
picture, *derive* it from where the video says the drone moved. To do that we ran a visual
odometry model (DPVO) on the video to recover the flight path, differentiated that path,
and applied the same physics as Phase 0.

**It does not work, and we know why.** The recovered path is not the real path: on one
flight DPVO's path is 41 m long where the drone actually flew 479 m, and the per-axis
correlation with the real path is essentially zero (0.01 / −0.09 / 0.13). Feeding a path
that does not track into the physics produces noise, not tilt: the implied tilt is off by
about **29-35°**, against a Phase 0 floor of 1.2°.

The important detail is that **the rotation half of the same model is excellent**
(0.15-1.5° per frame). So the failure is specifically in translation, not in the
odometry model as a whole, and not in the physics. That is why the verdict test in
report 04 gives the path away (commanded-path mode) instead of trying to recover it.

## The question and the setup

**Gate 2.** Does implied tilt from the recovered path agree with motion capture to
RMSE ≤ 8° with signal-to-noise > 3?

**Setup.** DPVO poses on the 100° virtual pinhole view at 0.5× (320×240), 48 patches per
frame, 12 Hz frames. The trajectory is gravity-aligned using the Phase 1 attitude, then
differentiated with Savitzky-Golay (poly 3) at a filter window swept in samples, with
`k_d = 0.295` fixed from Phase 0. Three scale options: oracle scale from mocap, a
per-clip fitted scale, and a depth-model scale (not run — no MoGe-2 pass).

## Results

### Gate 2

| run | window | implied-tilt RMSE | SNR | testable fraction (implied tilt > 10°) | verdict |
| --- | --- | --- | --- | --- | --- |
| first `results/phase2_implied/20261108T132738Z_c738aa8` (commit `c738aa8`) | "0.05 s" (inert sweep, see below) | **28.97°** | 0.84 | 0.006 | fail |
| corrected `results/phase2_implied/20261109T173302Z_d9de303` (commit `d9de303`) | 5 samples | **32.67°** | 0.58 | 0.000 | fail |

Target: RMSE < 8°, SNR > 3. Both runs fail both criteria; the correction makes it worse,
not better.

### The window sweep, and the correction

The first run swept windows as **seconds** — 0.05 to 0.37 s — on frames at ~12 Hz, where
those values round to 1-4 samples and the helper clamps to its 5-sample minimum. Every
entry in `windows` was therefore the *same* filter, to ten decimal places. The recorded
"chosen window 0.05 s" is not a selection; it is the clamp.

The corrected run sweeps **samples** (`20261109T173302Z_d9de303`):

| window (samples) | 5 | 7 | 9 | 11 | 15 | 21 | 31 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| implied-tilt RMSE (deg) | **32.67** | 33.18 | 33.68 | 34.20 | 35.30 | 37.15 | 41.06 |
| SNR | 0.58 | 0.58 | 0.57 | 0.56 | 0.55 | 0.52 | 0.48 |

Monotone in the wrong direction: shorter windows are less bad because they track the noise
less aggressively, and no window comes close to the target. The corrected run's
`gate2.chosen_window_s` field reports `5.0`, which is a **sample count written into a
field named for seconds** — a cosmetic bug in the corrected run, noted here rather than
silently fixed.

The two runs also disagree at nominally the same 5-sample window (28.97° vs 32.67°) and in
how many window entries are counted (`n_runs` 30 vs 56). That difference is **not
explained by the window** and is left as an open item rather than papered over.

### Why it fails: the trajectory

| quantity | value | source |
| --- | --- | --- |
| oracle scale (VO path against mocap), per flight | −0.003, −0.002, −0.009, +0.011, −0.041 (01a, 01p, 04a, 07a, 13a; the other three flights are in the same artifact) | `scale_oracle` in both runs |
| per-axis correlation, VO against mocap | 0.01 / −0.09 / 0.13 | Phase 2 handoff record, `docs/decisions.md` `[A] Gate 2`; **not re-derived in this pass** |
| path length, one flight (01p) | 41 m (VO) vs 479 m (mocap) | same record; not re-derived in this pass |
| DPVO rotation error, per frame | 0.15-1.5° | `vo_relative_rotation_check` in `results/phase1b_fusion/20261108T133135Z_c738aa8` |

An oracle scale near zero — negative on four of the five printed flights — means the
recovered trajectory is not even a scaled copy of the real one. Double-differentiating it
produces noise, and the implied tilt inherits it.

**A negative result worth keeping:** a per-clip fitted scale does *not* rescue this. The
fitted-scale ladder row is within 0.1° of the oracle row in the corrected run
(34.97 vs 34.89), i.e. the extra freedom buys nothing, because the problem is the shape of
the path, not its size. That also means this particular nuisance cannot be the mechanism
that hides a violation later (report 04 confirms the same thing from the other side).

### The oracle ladder (the rows missing from the Phase 1/2 summary)

Each rung swaps one source of truth for an estimate. Rows 1, 2, 6, 7 are **accuracy**
(numbers against mocap truth); rows 3-5 are marked for what they are.

| # | rung | number | what it is | source |
| --- | --- | --- | --- | --- |
| 1 | mocap position + mocap attitude | **1.24°** median residual | physics floor (accuracy) | `results/phase0_floor/20261108T044153Z_f3b5d2bf` |
| 2 | VO position (oracle scale) + mocap attitude | **28.97°** first run / **34.89°** corrected | accuracy, position estimated | the two Phase 2 runs above |
| 3 | mocap position + estimated attitude | **8.73°** tilt RMSE | accuracy, attitude estimated (the depicted side's error) | `results/phase1b_fusion/20261108T133135Z_c738aa8` |
| 4 | VO position (oracle scale) + estimated attitude | 2.75-8.30° residual | **self-consistency, not accuracy** — both sides share the drifting VO frame, so their azimuth error cancels | `results/phase2_implied/20261108T132738Z_c738aa8` (`residual_*`) |
| 5 | fully estimated, depth-model scale | not run | no MoGe-2 pass; `scale_depth.run = false` | both Phase 2 runs |
| 6 | fully estimated, fitted scale | **29.59°** first / **34.97°** corrected | accuracy | the two Phase 2 runs |
| 7 | commanded-path mode (verdict test) | see report 04 | mocap position + pixel attitude, heading from the path | `results/phase3a_verdict/20261109T090957Z_bc3a398` |

Reading: swapping the **position** source costs ~28-35°, swapping the **attitude** source
costs ~7°. The position is the binding error by a factor of four, and within DPVO it is
the translation half specifically — its rotation is the best-behaved component in the
whole pipeline. Row 4 must never be quoted as accuracy; it is in the table only to show
why a naive "the residual looks small" reading would be wrong.

## What this supports

- The decision in report 04 to run the verdict in **commanded-path mode**: the path is the
  part that cannot be measured, so it is given.
- The attribution "DPVO's translation is the problem, not its rotation", which points the
  next attempt at loop closure / dense BA / per-segment scale, not at a better gravity
  model.
- A reusable negative: a fitted per-clip scale does not have enough leverage to hide a
  violation, because it cannot change the path's shape.

## What this does not show

- That *any* visual odometry would fail on this footage. Only DPVO was tried.
- Nothing about depth-model scale; that rung is missing, not zero.
- The per-axis correlations and path lengths are quoted from the Phase 2 record; they were
  not re-derived from a saved artifact in this pass (**inferred from that record**).

## Caveats and open items

- **Correction:** the first run's window sweep was inert; treat "chosen window 0.05 s" as
  withdrawn. The corrected sweep is the one to cite.
- **Open:** why the corrected run differs from the first at the same 5-sample window
  (28.97 vs 32.67) and in `n_runs` (30 vs 56). To settle: diff the two configs and compare
  per-flight arrays, which needs a rerun of both — not done here.
- **Cosmetic bug:** `gate2.chosen_window_s` holds a sample count in the corrected run.
- The Phase 1 attitude used here is the fused filter (report 02), so rows 2 and 6 include
  the Gate 1 error; that is what the ladder is meant to expose, but it means row 2 is not a
  pure position-source comparison.
