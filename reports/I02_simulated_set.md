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

**23 clips are on R2** (`track-i/videos/<split>/<family>/<clip_id>.mp4`,
`track-i/gt/<clip_id>.npz`), listed row by row in `results/i02/manifest.csv`: 20 development
and 3 held-out, covering every defect family, all three development scene families, the two
held-out families and all four mounts. The manifest carries, per clip, the split, the scene and
its seed, the maneuver and parameters, the mount, the label, the violation, its magnitude in
degrees and the exact interval `[interval_start_s, interval_end_s]`. `docs/DATA.md` publishes
the split and the batch composition; `experiments/i02_build_manifest.py --verify` re-hashes
every object on R2 (**46 objects, 0 mismatches**).

The remaining 277 clips are rendering in 50-clip slices; the render is resumable (a clip whose
video is on the volume is skipped) and publishing is idempotent.

## The reader on the new families

The number Step 3 asks for first, measured on development clips only (six clips, 3,058 frames,
GeoCalib with the project's settings, no tuning):

| clip | family | maneuver | mount | truth (tilt med) | reader (tilt med) | up error, med (p90) | gain fwd / left |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `i02_dev_000` | industrial | hover | 0 deg | 0.00 | 1.19 | 1.19 (1.22) | — / — |
| `i02_dev_001` | urban_checker | ellipse | 15 deg | 22.21 | 22.33 | 0.28 (0.57) | 1.001 / 1.005 |
| `i02_dev_080` | forest_ridge | ellipse | 0 deg | 0.00 | 7.54 | 7.54 (11.44) | -0.29 / 0.25 |
| `i02_dev_107` | forest_ridge | ellipse | 45 deg | 7.87 | 5.41 | 6.38 (10.66) | 0.82 / -0.11 |
| `i02_dev_134` | forest_ridge | ellipse | 30 deg | 4.72 | 10.41 | 11.90 (15.29) | -0.51 / 0.65 |
| `i02_dev_135` | industrial | lemniscate | 45 deg | 12.90 | 8.21 | 7.58 (12.30) | 1.43 / 0.57 |

The six clips above confound family with mount (three of the four hard ones are forest), so a
3 x 4 feasible grid -- one clip per (family, mount), no defect, all development scenes -- was
rendered and scored before any detection number was claimed. 1,700 development clips are not
needed to see the structure; 12 are enough to see that it exists:

| family | mount | clip | up error, median (p90) | gain fwd / left |
| --- | --- | --- | --- | --- |
| urban_checker | 0 | `i02_dev_004` | 0.80 (1.41) | 0.88 / 0.99 |
| urban_checker | 15 | `i02_dev_001` | 0.28 (0.57) | 1.00 / 1.01 |
| urban_checker | 30 | `i02_dev_022` | 1.13 (1.98) | 1.05 / 0.99 |
| urban_checker | 45 | `i02_dev_007` | 1.95 (2.93) | 1.10 / 1.00 |
| industrial | 0 | `i02_dev_012` | 1.14 (1.61) | 0.70 / 0.97 |
| industrial | 15 | `i02_dev_009` | 1.45 (1.92) | 0.86 / 0.92 |
| industrial | 30 | `i02_dev_006` | 1.42 (7.87) | 0.89 / -- |
| industrial | 45 | `i02_dev_003` | 4.94 (28.06) | 0.88 / 0.70 |
| forest_ridge | 0 | `i02_dev_008` | **6.32** (9.26) | 0.27 / 1.07 |
| forest_ridge | 15 | `i02_dev_005` | **8.31** (11.33) | 0.53 / 1.29 |
| forest_ridge | 30 | `i02_dev_002` | **9.40** (13.56) | 0.61 / 0.56 |
| forest_ridge | 45 | `i02_dev_011` | **10.54** (14.18) | 1.22 / 0.80 |

Three conclusions, and they are findings about the *reader*, not about the defects:

* **the scene sets the floor, not the violation.** The six-clip sample's 6-12 deg errors on
  forest clips reappear on forest *feasible* clips (6.3-10.5 deg), so they are the family's
  bias (canopy occludes the horizon and the ground texture), not an effect of the defect;
* **mount angle matters where the scene is open**: urban and industrial go from 0.3-1.2 deg at
  0-30 deg of uptilt to 2-5 deg at 45 deg, with a heavy tail (p90 28 deg on one industrial
  clip) -- the 45 deg camera looks mostly at sky;
* **the gain is ~1 where the lean is measurable** (0.86-1.10 on urban and industrial) and
  noisy in the forest (0.27-1.29), which is the same pattern the real TII analysis found
  between the piloted and the two trackRATM flights.

The practical consequence for Tracks F2 and G2 is that a pooled detection rate over this set
would be an average over two very different readers. The report and the manifest therefore
carry the family and the mount on every row, and the evaluation protocol stratifies by
family (the pre-registration's `primary_outcomes` are per defect family; the scene family is
the stratification this evidence now requires).

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
