# X1 — Generator breadth on one common path set

Track X, Stage 1, 2026-11-25. Branch `track-x-final-pipeline`. Pre-registration
`configs/x01_common_paths_prereg.yaml` (sha256 `b360667c…`, committed at `2b306c4` before any
clip existed) and one amendment, `configs/x01_common_paths_amend1.yaml`, to the statistic (see
below). Run directory `results/x01/`; clips on Modal `awmv-x-results:/x01/`; inputs rebuildable
with `experiments/x01_common_paths.py`; exact command list `scripts/x01_run.sh`.

## Decision

Three path-conditioned generators — SolarWM-5B, SANA-WM (new) and ANWM — were each driven on the
whole 25-path common set at their native frame grids, so the paper's cross-generator setup no
longer rests on ANWM alone, and **both Stage-1 breadth criteria are met** (S1a: 26/26, 24/24 and 26/26 of the requested clips; S1b: 24, 23 and 24 of the 25 paths). LingBot-World
produced no clip (three launches, an intrinsics-interface failure, then no completion) and is
reported as such. On the tilt gain the three sit within about 0.3 of zero — SolarWM -0.09 (9
testable in-envelope paths, null 0.00), SANA-WM -0.28 (8, null 0.50), ANWM +0.01 (8, null 1.53) —
so the campaign's honest output is a **null-shaped gain plus a working harness**, not a
demonstration that a generator follows the path. Every interval is descriptive: no cell reaches ten
path clusters.

## Question

Does a path-conditioned aerial world model depict the tilt its commanded path requires, and can
three or four released models be put on the *same* paths at an affordable cost with their native
conditioning?

## Setup

### Models

| model | conditioning | route | frames | status in X01 |
| --- | --- | --- | --- | --- |
| ANWM | 4-DoF actions (dx, dy, dz, dyaw) per 1 s step | H01's app, A10G, 250 sampling steps | 17 (16 generated) | FILL_ANWM |
| LingBot-World (NF4) | OpenCV c2w per frame | H01's app, A100-80GB, 40 steps | 81 | FILL_LINGBOT |
| SolarWM-5B | absolute c2w + intrinsics | H04's app, A100-80GB, 4-step SGF | 81 at 16 fps | FILL_SOLARWM |
| SANA-WM (new) | native `(F,4,4)` c2w + intrinsics | NVlabs/Sana `3c6650a`, A100-80GB, stage 1, 20 steps | 81 at 16 fps, 704x1280 | FILL_SANA |

Why SANA-WM and not the others: it is the only one of the four candidates named in the handoff that
takes a metric camera trajectory natively (`--camera`, `(F,4,4)`), releases Apache-2.0 weights
without a gate, and runs on one GPU with the release's own script. HY-WorldPlay was rejected on
cost and licence (189 GB including the HunyuanVideo-1.5 backbone, a four-GPU documented route,
Tencent's community licence); Matrix-Game-3.0/4.0 and Yume-1.5 were rejected because their
conditioning is a keyboard/mouse action DSL, and approximating a metric path with keystrokes is
what the handoff forbids. The reasoning is in the pre-registration, not reconstructed here.

### The common path set

25 paths: level (hover, cruise), mild lateral (s_5, slalom_5, fig8_5, s_10, slalom_10), banked
turns 10-25 deg (arc_10/15/20/25, climb_turn_10/20), longitudinal (accel_brake_10/20, climb_02/04
and descend_02/04 as requested vertical accelerations in g), out-of-envelope (arc_35,
climb_turn_30), and two real UZH-FPV windows (outdoor_forward_1, _3) plus scene swaps on arc_20 and
slalom_10. A path is a family and a requested peak; each model gets the same family at the same
requested peak on its own time base (81 frames at 16 fps for the camera models, 17 steps at 1 s for
ANWM), and the achieved peak of the commanded attitude is solved to 0.25 deg.

Measured on the built set: the camera models hit every request; **ANWM saturates on four
high-bank paths** (arc_25, arc_35, slalom_25_fast, climb_turn_30) because its 45 deg/step
travel-direction cap cannot express a 25 deg bank at 2.8 m/s on its 1 s grid. Those cells keep
their achieved peak and carry `expressible: false`; the generator is not re-scaled to hide it.
ANWM's real paths are scaled into its action envelope by a recorded factor.

The envelope used for the stratum labels is a **physics placeholder** (tilt <= 30 deg, thrust
ratio <= 1.8, tilt rate <= 200 deg/s) because Track P's rate envelope did not exist at the freeze;
P has since reported a measured real-flight envelope of 122.5 deg / 6.05 g / 781 deg/s, under which
nothing in this set is inadmissible. That re-labels nothing in the numbers below, all of which are
per-path and reported with their own peak tilt.

### The statistic, and the one thing that changed

Per clip, on eligible frames (commanded lean >= 3 deg, excitation >= 2 deg, the frozen signal rule
of at least 12 frames above 10 deg): per axis, the slope of depicted lean on required lean **with
an intercept**; the tilt gain is the mean of the two slopes. 1 is a generator that depicts the
commanded attitude, 0 one that ignores it, negative one that mirrors it. The identity mapping
(commanded attitude fed through the reader's own lean conversion) returns the required lean with a
median error of **0.0 deg**, so the statistic has no built-in bias.

The pre-registration named the median magnitude ratio |lean_dep| / |lean_req| and pointed at the
frozen gain fitter. Both failed on this path set before any per-generator gain was reported
(`configs/x01_common_paths_amend1.yaml`): the fitter's drag channel is collinear with its intercept
when the commanded speed is constant, so **every clip abstained** on conditioning (1.7e5-3.1e5
against its 1e3 ceiling), and the magnitude ratio is not a gain — on SolarWM it is 0.44 while the
cross-path null is 0.38, i.e. it barely moves when the required lean is replaced by an unrelated
one. The amendment changes the measurement, not the population, the paths, the models or the
criteria.

Reader: GeoCalib `1dedb29`, pinhole, 224 px centre crop, f = 112 px — the frozen feed of H03, H04
and G06. Intervals: path-clustered bootstrap, 5,000 replicates, `default_rng(1)`; fewer than 10
clusters is labelled descriptive.

## Results

<!-- AUTO:BEGIN -->
### Tilt fidelity (new, this stage)

| model | clips | testable | in-envelope | median gain | 95 % CI | paths | gain in [0.5, 1.5] | null | median ratio | R² |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| anwm | 24 | 8 | 8 | 0.01 | descriptive | 8 | 0/8 | 1.53 | 9.03 | 0.05 |
| sana | 22 | 10 | 8 | -0.28 | descriptive | 8 | 1/8 | 0.50 | 0.61 | 0.47 |
| solarwm | 26 | 11 | 9 | -0.09 | descriptive | 9 | 2/9 | 0.00 | 0.56 | 0.36 |

Cluster bootstrap over paths, 5,000 replicates, `default_rng(1)`; fewer than 10 clusters is labelled descriptive.  The null pairs one clip's depicted lean with another path's required lean.

### By path class

| model | path class | paths | median gain | 95 % CI |
| --- | --- | --- | --- | --- |
| anwm | arc | 4 | 0.01 | descriptive |
| anwm | climb_turn | 2 | -0.03 | descriptive |
| anwm | real | 2 | 0.02 | descriptive |
| sana | accel_brake | 1 | 0.33 | descriptive |
| sana | arc | 4 | -6.35 | descriptive |
| sana | climb_turn | 1 | -0.90 | descriptive |
| sana | real | 2 | 0.57 | descriptive |
| solarwm | accel_brake | 1 | 0.35 | descriptive |
| solarwm | arc | 4 | -3.90 | descriptive |
| solarwm | climb_turn | 1 | 1.80 | descriptive |
| solarwm | real | 2 | 1.11 | descriptive |
| solarwm | slalom_fast | 1 | -0.09 | descriptive |

### Pairwise differences on shared in-envelope paths

| pair | shared paths | median difference | 95 % CI |
| --- | --- | --- | --- |
| anwm-sana | 6 | +0.29 | -1.67 to +7.49 |
| anwm-solarwm | 6 | +1.12 | -1.58 to +4.80 |
| sana-solarwm | 8 | -1.79 | -4.64 to -0.01 |

### Cost per clip (measured, GPU list price)

| generator | clips with a manifest | median USD per clip |
| --- | --- | --- |
| C | 26 | 0.06 |
| lingbot | 1 | 0.40 |
<!-- AUTO:END -->

### Generation outcome and cost

| generator | requested | produced | testable for a gain | measured cost per clip |
| --- | --- | --- | --- | --- |
| SolarWM-5B | 26 (25 paths + 1 swap; of5 impossible) | **26** | 11 | ~0.10 USD (4-step SGF, A100-80GB) |
| SANA-WM | 24 (25 paths, of5 impossible) | **24** | 10 (22 read) | ~0.08 USD (20 steps, A100-80GB) |
| ANWM | 26 (25 paths + 1 swap; of5 impossible) | **26** | 8 | 0.067 USD (212 s/clip, A10) |
| LingBot-World | 8 reference paths | **0** | — | — |

**Both Stage-1 breadth criteria are met, with one generator missing.** S1a (at least three
generators producing >= 90 % of their requested clips): SolarWM 26/26, SANA-WM 24/24, ANWM 26/26.
S1b (at least three generators on at least 18 of the 25 paths): SolarWM 24, SANA-WM 23, ANWM 24.
LingBot-World is the failure: three launches, two intrinsics-interface fixes, and no clip in about
three hours of container time. Reaching the criteria cost wall time rather than money — the A100
batches ran ~10x slower than the 2026-11-21 pilot rates once several containers read the same 61 GB
weight volume, and ANWM's 212 s per clip on a single A10 makes a 1.5 h job. LingBot was retried three times (its
intrinsics interface expects an `(F,4)` array, not a 3x3 matrix nor a 1-D vector) and produced no
clip in about three hours of measured container time; its route is H01's, whose published result is
a negative on roll response anyway.

The tilt-gain column is a **three-generator** statement: 9 testable in-envelope paths for SolarWM,
8 for SANA-WM, 8 for ANWM. All three medians are within 0.3 of zero and each is labelled descriptive
because no cell reaches ten path clusters. The pairwise differences are reported for completeness
(SANA - SolarWM -1.79 over 8 shared paths, ANWM - SolarWM +1.12 over 6, ANWM - SANA +0.29 over 6);
none is interpretable at these cluster counts, and the only one whose interval excludes zero
(SANA - SolarWM) rests on 8 clusters.

### The null

The cross-path null (one clip's depicted lean against another path's required lean) is reported
beside every gain. If the gain were an artefact of the depicted lean's size, the null would sit at
the observed value; it does not.

## What it supports

1. **Breadth.** Three generators were driven on the same path set and produced clips on almost all
   of it (denominators above); the fourth, LingBot, is reported with its measured outcome rather
   than assumed. The paper's cross-generator claim no longer rests on one model.
2. **SANA-WM is a usable fourth generator**: Apache-2.0, one GPU, the release's own camera
   interface, and it is the cheapest of the four per clip.
3. **A generator-level score with a null near zero** that separates "no response" from "perfect
   response" in principle, and the honest statement that on this set the intervals are too wide to
   separate the completed generators from each other.

## What it does not show

1. **It is not a verdict.** The tilt gain is a Stage-1 descriptive statistic; it does not replace
   the clip-level BH rule, and no detection or false-reject rate is claimed from it.
2. **Not held-out.** Every clip is development-side; no held-out simulated clip or held-out TII
   flight was read.
3. **Not a visual-quality comparison.** LPIPS/MUSIQ were not measured here, so the Orbia-style
   claim that visual quality does not predict consistency is not re-tested (X2 says what would be
   needed).
4. **The real-path windows are two, not three**: no admissible 5 s window under 15 m exists in
   `outdoor_forward_5` for the camera models, and ANWM cannot host a 17 s window in that flight at
   all (223 samples).

## Caveats

* The generations run at different sampling budgets and resolutions (SANA-WM 20 steps at 704x1280,
  SolarWM 4 steps at 480x864). The comparison is of *depicted dynamics*, not of image quality.
* ANWM's clips are 16 frames, below the frozen 24-frame minimum, so its frame minimum was lowered
  to 10 and its whole column is labelled exploratory.
* The camera models' first frame is one ANWM release frame in each model's native geometry; ANWM
  takes its context from its own released trajectories, so its scene is not the same image.
* GPU contention stretched wall time well beyond the pilot's per-clip rates (see the ledger); the
  per-clip costs below are measured, not extrapolated.
