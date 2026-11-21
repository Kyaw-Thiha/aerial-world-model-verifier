# I02 — The simulated labelled flight set: construction, splits and first batch

**Branch** `track-i-simset`, worktree `~/Documents/Projects/awmv-track-i`.
**Date** 2026-11-21. **Pre-registration** `configs/sim_prereg.yaml` (committed before any clip).
**Spend** see `docs/COST_LEDGER.md` (ceiling USD 40).

## Summary

The set is now defined, frozen and producing clips. Three things are committed and immutable:

* `configs/sim_prereg.yaml` — what a label means, which scenes are development and which are
  held out, the maneuver envelope, the defect families with their magnitudes and interval
  rules, the manifest schema, the evaluation protocol and the prohibitions.
* `configs/sim_splits_frozen.json` — the resolved 300-clip table (200 development over
  `urban_checker`, `forest_ridge`, `industrial`; 100 held-out over `field_settlement`,
  `coastal_cliff`), one row per clip with scene seed, maneuver and parameters, mount uptilt,
  defect family, magnitude and interval seed. sha256 in `docs/DATA.md`.
* The generator (`src/sim/{flight,scenes,labels,dataset}.py`) and its tests: every defect is
  checked to be exactly the stated change over exactly the stated interval, every clip builds
  to a 10-30 s flight inside the 2.0 thrust-to-weight limit, and the thrust-limit family is
  checked to be labelled on the path rather than on the camera.

Clips are rendered on an L40S with EEVEE (0.16 s/frame measured), encoded to H.264, published
to R2 under `track-i/` with a sha256 check on both ends, and listed row by row in
`results/i02/manifest.csv`.

## What is in the set

| | development | held-out |
| --- | --- | --- |
| scenes | `urban_checker`, `forest_ridge`, `industrial` (seeds 101-106) | `field_settlement`, `coastal_cliff` (seeds 201-206) |
| feasible | 80 | 40 |
| tilt removed (whole flight) | 27 | 13 |
| mirrored roll (whole flight) | 27 | 13 |
| tilt scaled x0.5 / x1.5 | 33 | 17 |
| partial defect over 10 / 25 / 50 % | 20 | 10 |
| thrust-limit path (peak > 2.0 g) | 13 | 7 |
| **total** | **200** | **100** |

Frames: **149,450** at 640x360, 30 fps, 10-30 s per clip, truth at 250 Hz. Mounts: 0, 15, 30
and 45 deg of uptilt, a quarter of the set each, so the mount is a factor the analysis can
condition on rather than a constant.

## The construction, in four sentences

A flight is closed-form (`src/sim/flight.py`): position, velocity and acceleration have exact
derivatives, and the attitude is the flatness attitude `z_B = unit(a + g e3 + k_d v)` with the
project's frozen `k_d = 0.295`, combined with the path's yaw. A defect replaces the attitude the
*camera* is rendered on — level, roll-flipped, or lean-scaled — over the whole flight or over
an exact interval, and the path is untouched, so the required lean is identical and only the
pixels change. The thrust-limit family is the opposite case: the path itself demands more than
2.0 g of thrust (over 60 deg of tilt), the camera is honest, and the label is on the flight.
Every clip is one row of the frozen table, so the manifest describes the data rather than a run.

## First batch

`X` clips are on R2 (`track-i/videos/…`, `track-i/gt/…`), listed in
`results/i02/manifest.csv`, covering every defect family and both splits. Their labels are in
the manifest with the exact intervals.

## The reader on the new families

`Y` — the number Step 3 asks for first, measured on development clips only.

## What this supports

* A labelled set of feasible and infeasible flights with exact ground truth, exact violation
  intervals and a frozen split, which is what Track F2 and G2 need for calibration and what
  this track needs to state a detection floor in simulation.

## What it does not show

* Nothing about detection rates yet: the full 300-clip render and the dev-pass scoring come
  after this batch is checked.
* Nothing about the held-out families: they are rendered but not scored until the development
  pass is closed and committed.
* Realism is not claimed. The scenes are procedural; the reader's error on them is the
  sim-to-real quantity to quote, not the visual plausibility of a frame.

## Caveats

* The set inherits the project's drag model, so agreement between simulated and real flights on
  the *required* side is by construction; the simulated set tests the depicted side.
* The clips are short (10-30 s) and the manoeuvres are analytic; nothing here exercises
  hand-flown irregularity, which is what TII supplies.
* Publishing runs on the workstation because the Modal secret cannot write to `track-i/`
  (`docs/decisions.md`); the videos are small (~90 kB per 12 s clip) so the transfer is cheap.
