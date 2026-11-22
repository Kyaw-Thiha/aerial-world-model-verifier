# L01 — The reader's error is set by the scene, and the frozen confidence gate does not remove it

**Decision.** On the 80 verified feasible development clips the frozen reader's error is 0.5-2.8 deg
in the open families and 7.0-10.7 deg in the dense forest at every mount, and the frozen S3 gate
abstains on half the forest clips but the half it keeps is **not** better (median 9.96 deg kept
against 7.59 dropped), so the gate cannot be the defence against scene bias. The error's
composition also changes with the scene: bias dominates in the forest (40.5 % of the squared
error, pooled), fast noise in the open families (51.2 % urban, 43.7 % industrial), which is the
same bias/slow/fast total as Track J measured on real TII (29.6 / 40.4 / 30.0) but distributed
differently.

## Question

Track I's simulated set gives hundreds of scenes with exact camera attitude. On the feasible
development clips, what is the frozen reader's error by scene family and mount, what is its
per-clip bias, does the frozen S3 confidence gate abstain on forest-like clips, and how does the
error split into a constant, a slow drift and fast noise (Track J's decomposition)?

## Setup

**Population.** All 80 feasible development clips of `configs/sim_splits_frozen.json` (sha256
`cdf66e28...`): 3 scene families (`urban_checker`, `forest_ridge`, `industrial`) x 4 mounts
(0/15/30/45 deg), 6-7 clips per cell, 2 scene seeds per family, 37,401 frames at the full 30 Hz.
Every clip's video and truth were re-downloaded from R2 and hashed against
`results/i02/manifest.csv` before scoring: **160 objects, 0 mismatches**. The 30 remaining
development clips are infeasible-defect clips and are not on R2; the 100 held-out clips are
3/100 on R2. No held-out clip was read.

**Instrument.** GeoCalib at the pinned commit, `weights="pinhole"`, `camera_model="pinhole"`,
scalar focal prior from the render's 90 deg horizontal field, one frame at a time — the same
route Track I scored, at stride 1 instead of its mixed stride 1/2 (`remote/l_score_sim.py` at
`674e7b8`, made resumable at `d048a97`; job `awmv-l-reader`, A10, four shards, 3.7 A10G-hours,
USD 4.1). Nothing was
tuned. The re-score reproduces I02 on the frames I02 scored: the worst per-clip median
per-frame difference is 0.255 deg (p90 1.148), the pooled clip-median difference is -0.0005 deg
(`configs/l01_amendment_c.yaml`). The per-frame residual is GeoCalib's own run-to-run variation
(Track A's identity-edit check: 0.13-0.16 deg), not a pipeline error.

**Statistics.** As pre-registered (`configs/l01_scene_stratified_prereg.yaml`, amendments A and
C): per-frame angle between the estimated and rendered up-vector; per-clip median, p90, fraction
> 10 and > 20 deg; per-clip signed bias in the verdict's lean basis (forward, left), reported in
degrees via `asin` of the mean lean error; the frozen S3 gate (`configs/gate_frozen.yaml`) applied
unchanged at clip level; Track J's exact DCT decomposition (imported, not re-implemented) at
0.5 Hz. This is a descriptive experiment with no success criterion. Three scene-family clusters
and 6-7 clips per cell are far below the ~10-cluster floor, so **no interval is computed and no
number here is a 95 % confidence interval**.

## Results

### The grid: the family sets the error, the mount moves it

| family | mount | clips | frames | median error, deg | clip range | bias fwd, deg | bias left, deg | >10 deg | gate kept |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| urban_checker | 0 | 7 | 3396 | 0.87 | 0.50-1.14 | -1.56 | -0.19 | 3.6 % | 7/7 |
| urban_checker | 15 | 7 | 3854 | 0.51 | 0.29-0.62 | +0.10 | +0.05 | 0.2 % | 7/7 |
| urban_checker | 30 | 6 | 2741 | 0.95 | 0.91-1.11 | +0.75 | +0.24 | 3.8 % | 4/6 |
| urban_checker | 45 | 7 | 2344 | 1.57 | 1.30-2.02 | +1.37 | +0.27 | 8.0 % | 6/7 |
| forest_ridge | 0 | 6 | 2827 | 6.95 | 6.20-8.55 | -3.55 | +3.26 | 10.3 % | 0/6 |
| forest_ridge | 15 | 7 | 3549 | 7.66 | 6.86-8.90 | -2.64 | +1.51 | 21.8 % | 2/7 |
| forest_ridge | 30 | 7 | 3592 | 9.38 | 8.48-14.92 | -4.56 | -2.88 | 51.8 % | 6/7 |
| forest_ridge | 45 | 6 | 2000 | 10.66 | 8.94-12.76 | +2.12 | -5.38 | 57.0 % | 4/6 |
| industrial | 0 | 7 | 3331 | 1.18 | 1.01-1.29 | -1.09 | -0.05 | 0.0 % | 7/7 |
| industrial | 15 | 6 | 3418 | 1.31 | 0.66-2.11 | -0.94 | -0.12 | 0.0 % | 6/6 |
| industrial | 30 | 7 | 3800 | 1.27 | 0.71-1.43 | -0.44 | -0.30 | 1.2 % | 7/7 |
| industrial | 45 | 7 | 2549 | 2.78 | 1.49-5.25 | +2.03 | -1.20 | 12.9 % | 7/7 |

The family gap is 6-9 deg at every mount and grows with uptilt: the max-minus-min of the three
family medians is **6.08, 7.15, 8.43 and 9.09 deg** at 0/15/30/45 deg. Within a family the mount
matters only where the scene is open: urban runs 0.87, 0.51, 0.95, 1.57 deg and industrial 1.18,
1.31, 1.27, 2.78 deg, the 45 deg camera looking mostly at sky. The forest runs 6.95, 7.66, 9.38
and 10.66 deg — its worst cell is a 14.9 deg clip.

| family | clips | frames | median error | bias fwd | bias left | >10 deg | >20 deg | gate kept |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| urban_checker | 27 | 12335 | 0.94 | +0.14 | +0.09 | 3.4 % | 2.1 % | 24/27 |
| forest_ridge | 26 | 11968 | 8.72 | -2.27 | -0.86 | 34.0 % | 2.8 % | 12/26 |
| industrial | 27 | 13098 | 1.29 | -0.08 | -0.43 | 2.9 % | 1.6 % | 27/27 |
| all | 80 | 37401 | 1.44 | -0.71 | -0.39 | 13.0 % | 2.2 % | 63/80 |

Family spread per mount (max - min of the three family medians):
- 0 deg mount: 6.08 deg (values 0.87, 6.95, 1.18)
- 15 deg mount: 7.15 deg (values 0.51, 7.66, 1.31)
- 30 deg mount: 8.43 deg (values 0.95, 9.38, 1.27)
- 45 deg mount: 9.09 deg (values 1.57, 10.66, 2.78)

### The bias is not a fixed forward offset, and it is not the same sign as the real one

On the forest, the mean lean bias is **-2.27 deg forward** (SE across clips 0.65 deg) and -0.86 deg
left; per mount it runs -3.55, -2.64, -4.56 and +2.12 deg forward with the left component
swinging from +3.26 to -5.38 deg. That pattern is a bias that **rotates with the mount**, not a
constant forward pitch. It is also the opposite sign to the real TII reader's +2.15 to +2.69 deg
forward pitch offset (J01): on the forest the reader under-reads the lean, on real TII it
over-reads it. In the open families the bias is under 0.2 deg forward (urban +0.14, industrial
-0.08) apart from the 45 deg industrial cell (+2.03 deg).

The per-scene seeds agree with the family, not with each other's luck: urban 0.96 and 0.94 deg,
forest 8.55 and 8.90, industrial 1.18 and 2.03. The industrial seed difference (0.85 deg) is the
largest within-family variation.

### The frozen gate abstains on the forest — and keeps the forest clips that are just as bad

| family | n | kept | keep rate | kept median | dropped median | kept >10 deg |
| --- | --- | --- | --- | --- | --- | --- |
| urban_checker | 27 | 24 | 0.889 | 0.92 | 0.95 | 3.7 % |
| forest_ridge | 26 | 12 | 0.462 | 9.96 | 7.59 | 48.3 % |
| industrial | 27 | 27 | 1.000 | 1.29 | nan | 2.9 % |
| all | 80 | 63 | 0.787 | 1.29 | 7.49 | 11.6 % |

The gate drops 14 of 26 forest clips (keep rate 0.462) against 3 of 54 open clips (0.944), so it
*does* react to the scene. But its decision does not separate good forest frames from bad: the
12 forest clips it keeps have a **median of 9.96 deg** against **7.59 deg** for the 14 it drops.
48.3 % of the kept forest frames are above 10 deg (11.6 % across all kept clips), and the
median of the kept clips is already near 10 deg — the gate is not a scene-bias detector. On the real TII population the same frozen gate keeps
59.7 % of 100 clips and removes 88 % of the >10 deg frames (J01); on this simulated forest it
keeps half and leaves the bias. The practical consequence: a downstream verdict that relies on
the gate to protect itself from scene bias will be wrong on forest-like footage.

### The error's composition follows the scene, and the total matches real TII

| family | pooled bias | pooled slow | pooled fast | per-clip bias (median) | per-clip slow | per-clip fast |
| --- | --- | --- | --- | --- | --- | --- |
| urban_checker | 4.8 % | 44.0 % | 51.2 % | 31.9 % | 27.5 % | 26.3 % |
| forest_ridge | 40.5 % | 37.4 % | 22.1 % | 49.2 % | 30.0 % | 16.3 % |
| industrial | 9.2 % | 47.0 % | 43.7 % | 59.9 % | 16.6 % | 17.8 % |
| all | 23.6 % | 41.1 % | 35.3 % | 50.0 % | 26.3 % | 20.2 % |

Pooled over both lean axes and all frames, the simulated reader splits **23.6 % constant /
41.1 % slow / 35.3 % fast**, against Track J's real-TII 29.6 / 40.4 / 30.0 — the totals are close,
which is a useful cross-check that the decomposition and the reader behave the same way on
simulated pixels. But the split by scene is not the same: the forest is **bias-dominated**
(40.5 / 37.4 / 22.1), almost exactly J01's *gate-kept* real anatomy (42.8 / 40.3 / 19.8), while
urban (4.8 / 44.0 / 51.2) and industrial (9.2 / 47.0 / 43.7) are noise-dominated. Averaging the
three would hide the mechanism: on the forest a per-clip constant removes the largest single
term; in the open scenes it removes almost nothing and fast noise dominates. (The per-clip median
shares — 50.0 / 26.3 / 20.2 — weight clips equally and are higher on bias than the frame-weighted
pool; both are reported because neither is "the" anatomy.)

### Reproduction of I02

The re-score's clip medians reproduce I02's grid with a pooled difference of -0.0005 deg; the
worst clip differs by 0.323 deg and 22 clips by more than 0.05 deg, entirely from run-to-run LM
variation amplified by the median's order statistic (on `i02_dev_039` the per-frame median
difference is 0.054 deg and the correlation 0.999, yet the clip median moves 0.32 deg because 151
frames sit in a dense band). The per-frame check in amendment C is the one used for the final
numbers; all 80 clips pass it.

## What this supports

* **A scene-stratified reader error on exact truth**: forest 7.0-10.7 deg at every mount, open
  scenes 0.5-2.8 deg, with the 3 x 4 grid and the family spread made explicit.
* **The gate is not a defence against scene bias**: it abstains on the forest at twice the open
  rate, but the forest clips it keeps are not better than the ones it drops.
* **A mechanism for the sim-to-real difference**: the simulated forest's bias is negative and
  rotates with the mount; the real TII bias is a positive forward pitch. A reader trained to
  correct one will not automatically correct the other, and the mixed arm in L02 has to be
  judged on real held-out flights, not on simulated family spread alone.
* **A quantitative target for L02**: any training arm that shrinks the scene-family spread has to
  move a 6-9 deg gap, while the open families' 0.5-2.8 deg leaves little room; the forest is
  where the measurable room is.

## What this does not show

* Nothing about the 30 infeasible development clips or the 100 held-out clips: they are not on
  R2 yet (I02 incomplete), so no detection rate and no held-out scene number is quoted.
* Nothing about real footage's scene dependence: the simulated families are procedural; the
  forest/open contrast is a statement about *these* renders and the reader, not about any real
  forest.
* No causal claim about which pixel cue causes the forest bias. It is consistent with canopy
  occluding the horizon, but this experiment did not ablate texture or horizon visibility.

## Caveats

* Descriptive only: 3 family clusters, 6-7 clips per cell; no intervals, no significance test.
* The reader route is pinhole with a focal prior on a 90 deg render; the real route is the
  distorted fisheye. The absolute errors are comparable only through the label construction,
  not as the same instrument setting.
* The lean-bias degrees are `asin` of a mean dimensionless lean component, a small-angle
  reading; the underlying statistic is the lean vector.
* The anatomy's "fast" band reaches 15 Hz here (30 Hz sampling) against 6 Hz on real TII (12 Hz
  sampling); a larger fast share in simulation is expected for that reason alone and the report
  does not read the fast difference as a physical difference.
* The gate was applied to the confidence GeoCalib reports in pinhole mode; the frozen tau was
  fitted on distorted-fisheye confidences. It is applied unchanged, as the pre-registration
  requires, but its scale is not guaranteed to transfer and this is the first test of that.

## Provenance

* **Run**: Modal `awmv-l-reader`, A10, four shards, 37,401 frames, 3.7 A10G-hours, USD 4.1
  (`docs/COST_LEDGER.md`). Code `remote/l_score_sim.py` at `674e7b8`, made resumable with
  batched commits at `d048a97`; analysis `experiments/l01_analysis.py` at `1177c6b`.
* **Inputs**: `track-i/videos/dev/...` and `track-i/gt/...` on R2, sha256-verified against
  `results/i02/manifest.csv` (160/160).
* **Results**: `results/l01/{per_clip.csv,cells.csv,families.csv,scenes.csv,anatomy.json,
  reader_rescore.json}` in git; the per-frame arrays (gravity, lean, truth, gate inputs) on R2 at
  `track-l/l01/per_frame/` with `results/l01/r2_manifest.json`.
* **Configs**: `configs/l01_scene_stratified_prereg.yaml` (674e7b8), amendments A and C;
  frozen `configs/gate_frozen.yaml` (sha256 `dc0cc4ff...`).
