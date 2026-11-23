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

The rest of the set renders in detached 100-150-clip batches (one container at a time; two
concurrent writers to the same Modal volume are not safe here). The render is resumable -- a
clip whose video is on the volume is skipped -- and publishing is idempotent, so a batch that
dies part-way costs only the clip in flight.

## The reader on the new families

The number Step 3 asks for first, measured on development clips only (GeoCalib with the
project's settings, no tuning). The first six clips scored, before the grid existed, are what
showed that the sample confounds family with mount:

| clip | family | maneuver | mount | truth (tilt med) | reader (tilt med) | up error, med (p90) | gain fwd / left |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `i02_dev_000` | industrial | hover | 0 deg | 0.00 | 1.19 | 1.19 (1.22) | — / — |
| `i02_dev_001` | urban_checker | ellipse | 15 deg | 22.21 | 22.33 | 0.28 (0.57) | 1.001 / 1.005 |
| `i02_dev_080` | forest_ridge | ellipse | 0 deg | 0.00 | 7.54 | 7.54 (11.44) | -0.29 / 0.25 |
| `i02_dev_107` | forest_ridge | ellipse | 45 deg | 7.87 | 5.41 | 6.38 (10.66) | 0.82 / -0.11 |
| `i02_dev_134` | forest_ridge | ellipse | 30 deg | 4.72 | 10.41 | 11.90 (15.29) | -0.51 / 0.65 |
| `i02_dev_135` | industrial | lemniscate | 45 deg | 12.90 | 8.21 | 7.58 (12.30) | 1.43 / 0.57 |

All **80 feasible development clips**, 19,081 frames, one row per
(family, mount) cell -- the 3 x 4 grid the downstream tracks asked for, with every feasible
development clip in it rather than one example:

| family | mount | clips | frames | up error, median (range across clips) | gain, fwd |
| --- | --- | --- | --- | --- | --- |
| forest_ridge | 0 | 6 | 1415 | **7.02** (6.27-8.59) | 0.27 |
| forest_ridge | 15 | 7 | 1776 | **7.62** (6.95-9.07) | 0.78 |
| forest_ridge | 30 | 7 | 1798 | **9.41** (8.65-14.86) | 0.66 |
| forest_ridge | 45 | 6 | 1002 | **10.69** (9.02-12.60) | 1.10 |
| industrial | 0 | 7 | 1846 | **1.18** (1.03-1.29) | 0.78 |
| industrial | 15 | 6 | 1711 | **1.30** (0.67-2.14) | 0.89 |
| industrial | 30 | 7 | 1902 | **1.27** (0.70-1.42) | 0.91 |
| industrial | 45 | 7 | 1277 | **2.79** (1.45-4.94) | 1.04 |
| urban_checker | 0 | 7 | 1700 | **0.85** (0.50-1.15) | 0.86 |
| urban_checker | 15 | 7 | 2107 | **0.49** (0.28-0.57) | 1.00 |
| urban_checker | 30 | 6 | 1372 | **0.95** (0.92-1.13) | 1.06 |
| urban_checker | 45 | 7 | 1175 | **1.49** (1.29-1.95) | 1.05 |

Three conclusions, and they are findings about the *reader*, not about the defects:

* **the scene sets the floor, not the violation.** Dense forest reads 7.0-10.7 deg of up-vector
  error at every mount (six to seven clips per cell, ranges 6.3-14.9 deg), while the two open
  families stay at 0.5-2.8 deg. The same magnitudes appeared on forest *feasible* clips before
  any defect was scored, so this is the family's bias -- canopy occludes the horizon and the
  ground texture -- not an effect of the violation;
* **mount angle matters where the scene is open**: urban reads 0.85, 0.49, 0.95 and 1.49 deg at
  0/15/30/45 deg of uptilt and industrial 1.18, 1.30, 1.27 and 2.79 -- a factor of two to three
  between the 15 deg mount and the 45 deg one, because the 45 deg camera looks mostly at sky;
* **the lean gain is ~1 where the lean is measurable** (0.86-1.06 on urban, 0.78-1.04 on
  industrial) and noisy in the forest (0.27-1.10), which is the same pattern the real TII
  analysis found between the piloted flights and the two trackRATM flights.

The practical consequence for Tracks F2 and G2 is that a pooled detection rate over this set
would be an average over two very different readers. The report and the manifest therefore
carry the family and the mount on every row, and the evaluation protocol stratifies by
family (the pre-registration's `primary_outcomes` are per defect family; the scene family is
the stratification this evidence now requires).

## The labels checked against the rendered video

A label is worth what its verification is worth, so two partial-interval clips were decoded from
the published video and read frame by frame with the same reader:

| clip | violation | interval | frames in / out | reader tilt inside | reader tilt outside | truth inside / outside |
| --- | --- | --- | --- | --- | --- | --- |
| `i02_dev_180` | tilt removed | 11.40-13.34 s of 19.4 s (10 %) | 61 / 522 | 0.91 deg | 5.71 deg | 0.00 / 6.85 deg |
| `i02_dev_181` | mirrored roll | 8.60-10.60 s of 20.0 s (10 %) | 63 / 538 | 4.49 deg | 7.53 deg | 4.33 / 7.25 deg |

The tilt-removed clip is level exactly inside its interval and physical exactly outside it, to
within the reader's own error -- which is what the manifest's interval columns claim. The
mirrored-roll clip's *tilt* is unchanged inside and outside, as it must be (a roll sign flip
does not change how far the camera leans, only which way), which is why that family is a gain
test and not a residual test.

## What the reader's floor implies for the violations in the set

The violation magnitudes were drawn before the reader was measured (they come from the frozen
table's maneuver targets), so comparing them now is a statement about the set's difficulty, not
a post-hoc tuning:

| family | reader up error (median of the grid) | defect magnitude, median | min | max | magnitude / reader |
| --- | --- | --- | --- | --- | --- |
| urban_checker | 0.90 | 13.9 | 1.1 | 61.7 | 15.5 |
| industrial | 1.29 | 10.6 | 2.8 | 52.0 | 8.2 |
| forest_ridge | 8.52 | 15.3 | 2.9 | 97.7 | 1.8 |

So the open scenes carry violations 8-15 times the reader's own error, and the forest carries
them at about 1.8 times -- which is where a clip-level verdict will start to fail, and why the
forest clips are the ones the set most needs (they are what a reviewer would call the hard
cases, and they are also the ones whose results must not be pooled with the rest).

The thrust-limit family is measured differently on purpose: its `magnitude_deg` is the excess
of the peak required tilt over the 60 deg / 2.0 g envelope (1.8-8.3 deg across the dev set),
with the peak thrust ratio in its own column, because the violation is the path leaving the
envelope, not a tilt that is visible in a frame.

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

## Batch 2 — the full set on R2 (2026-11-22)

The first batch left 127 clips unrendered: 30 development clips (18 partial-interval at 10 / 25 /
50 % and 12 thrust-limit) and 97 held-out clips (`field_settlement` 49, `coastal_cliff` 48). They
were rendered with the frozen recipe — EEVEE Next, 16 spp, 640x360, 30 fps, exposure -3, one
container at a time, resumable — and published clip by clip to `track-i/` on R2.

**Publishing and verification.** The container leaves each clip's video and truth on the Modal
volume; `experiments/i02_publish.py` pulls them, uploads with the uploader's sha256 round-trip
check (the hash is computed on the bytes written and read back), and records the key, hash and
size in the manifest row. `experiments/i02_build_manifest.py --verify` then HEADs every object on
R2 and re-hashes a sample; `--full-hash` (added in this round) re-hashes **every** object instead.
The final check reports **600 objects** (300 clips x video + truth) present, 0 missing, 0 bad and
0 rows without an R2 key, every one fully re-hashed from R2.

**What changed in the repository.** `results/i02/manifest.csv` is 300 rows (200 development, 100
held-out), one per clip, each with its split, scene and seed, manoeuvre and parameters, mount,
label, violation, magnitude in degrees, exact interval, R2 keys and both sha256 values.
`docs/DATA.md` carries the regenerated dataset block. No frozen file changed:
`configs/sim_prereg.yaml` (`245b68bf…`) and `configs/sim_splits_frozen.json` (`cdf66e28…`) are
byte-identical to their committed values, and the render settings are the ones the manifest
records.

**Frames.** 149,450 in total (99,591 development, 49,859 held-out), matching the composition
recorded before rendering.

**Spend.** 7.55 USD for the whole render (Modal billing export, `awmv-i-dataset`), against the
40 USD ceiling.

**What is still not claimed.** Rendering and publishing are not scoring: no held-out clip has
been scored by any track. Track G3's development pass used the 170 clips that existed at its
freeze; the remaining 30 development clips enter its batch 2 under the same thresholds, and the
held-out clips are scored once in its frozen evaluation.
