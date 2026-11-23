# N05 — C2 and coherence on the propagated lean over the 200 simulated dev clips

**Decision.** With a deployment-like **level-body start and no reader**, the propagated lean loses
the verdict: at a matched 7/80 feasible false-reject, C2 detects **0/27** tilt removals (second
DPVO draw 1/27) against the reader's **16/27**, and 7/27 mirrored rolls against 10/27. The cause
is the start, not the extractor: a fixed body-frame attitude error is **not** a constant lean
offset, so even with *true* rotations C2 leaves 8.54° on feasible clips, which raises the matched
threshold to 29.5° and destroys sensitivity; the coherence tier is additionally undefined on the
60 % of feasible clips whose required lean is constant.

Pre-registration `configs/n05_sim_verdict_prereg.yaml` (before any reader/DPVO run on the defect
clips). Code `experiments/n05_sim_verdicts.py`, diagnostics `experiments/n05_diagnose.py`.
Results `results/n05/`. All 200 development clips; **no held-out clip was read**.

## Plain-language summary

1. **What was tested.** All 200 dev clips, reader versus propagated lean, two tiers (C2 and
   coherence), thresholds fitted on feasible clips by leave-one-scene-seed-out and then matched
   to the same feasible false-reject count.
2. **The start.** The propagated arm uses `u_cam(0) = R_cb^T e3` — "the body was level at the
   first frame". No truth attitude and no reader reading is used anywhere on that arm. It is the
   only start that is simultaneously free of oracle and reader, and it is wrong by the clip's
   actual initial tilt whenever the clip starts banked (10-30° on the dynamic manoeuvres).
3. **The tiers do not absorb it.** At the matched point the reader detects 16/27 removals, 10/27
   mirrors and 12/33 scale edits; the propagated lean detects 0/27, 7/27 and 3/33 (draw 1) and
   1/27, 5/27 in draw 2.
4. **Why.** A start error is a fixed rotation in the *body* frame. C2 subtracts a 2-vector offset
   in the *yaw-frame lean*; those coincide only for small errors or a pure yaw offset. Propagating
   with the true rotations from a level start still leaves a 8.54° C2 residual on feasible clips,
   so the null distribution widens and the matched threshold rises to 29.5° — above almost every
   violation.
5. **The extractor is not the problem.** With an oracle start, C2 on the propagated lean is
   2.02° on feasible clips, and the matched-point detections become 6/27 removals, 7/27 mirrors,
   5/33 scale — still below the reader on removals, because the constant-bank removal clips are
   absorbed by the offset fit (see 6).
6. **Coherence is undefined on most clips.** 60 % of feasible clips hold a constant required lean
   (zero variance over the clip; coordinated constant-bank turns). A scale-free correlation cannot
   be computed there and saturates at 1.0. On the 87 clips with a varying lean, the natural
   threshold is the ceiling and the reader detects 10/14 removals and 10/14 mirrors against the
   propagated arm's 6/14 and 9/14.
7. **The reader's C2 is the only tier with real removal sensitivity**, and it needs no start:
   0.67° median residual on feasible clips against 10.85° for the propagated arm.

## Setup

* Population: 200 dev clips (80 feasible, 27 tilt-removed, 27 mirrored, 33 scaled, 20 partial,
  13 thrust-limit). Held-out never opened.
* Reader arm: the frozen GeoCalib-224 pinhole reader, L01 settings, stride 2 (15 Hz), freshly run
  on all 200 clips (`remote/n_readers.py`, copied from Track L).
* Propagated arm: DPVO (standing settings) on each clip at 640×360; level-body start; lean via
  `depicted_lean_from_gravity(up, R_cb)`. A second full draw was run as an exploratory robustness
  check (DPVO is not bit-reproducible; N06).
* Required side: each clip's exact `lean_req` from its own ground truth at the scored frames; the
  C2 velocity regressor from the ground-truth velocity and yaw.
* Thresholds: leave-one-scene-seed-out (six seeds) at the 95th percentile of feasible scores,
  then a matched global threshold at the (k+1)-th largest feasible score with k = reader C2's
  natural 7 false-rejects.

## Results — pre-registered tables

### Matched false-reject (7/80 feasible for every arm and tier)

| condition | C2 reader | C2 propagated | R reader | R propagated |
| --- | --- | --- | --- | --- |
| feasible | 7/80 | 7/80 | 0/80 | 0/80 |
| tilt_removed | **16/27** | **0/27** | 10/27 | 6/27 |
| mirrored_roll | **10/27** | **7/27** | 10/27 | 9/27 |
| tilt_scaled (x0.5 + x1.5) | **12/33** | **3/33** | 0/33 | 1/33 |
| partial (10/25/50 %) | 5/20 | 4/20 | 1/20 | 2/20 |
| thrust_limit (path-side) | 13/13 | 10/13 | 0/13 | 0/13 |

### Natural leave-one-seed-out thresholds

| condition | C2 reader | C2 propagated | R reader | R propagated |
| --- | --- | --- | --- | --- |
| feasible | 7/80 | 7/80 | 0/80 | 0/80 |
| tilt_removed | 11/27 | 0/27 | 10/27 | 6/27 |
| mirrored_roll | 8/27 | 5/27 | 10/27 | 9/27 |
| tilt_scaled | 8/33 | 3/33 | 0/33 | 1/33 |
| partial | 3/20 | 3/20 | 1/20 | 2/20 |
| thrust_limit | 13/13 | 6/13 | 0/13 | 0/13 |

### By scale factor and partial fraction (natural)

| condition | C2 reader | C2 propagated | R reader | R propagated |
| --- | --- | --- | --- | --- |
| scale x0.5 (17) | 6/17 | 0/17 | 0/17 | 0/17 |
| scale x1.5 (16) | 2/16 | 3/16 | 0/16 | 1/16 |
| partial 10 % (8) | 0/8 | 1/8 | 1/8 | 1/8 |
| partial 25 % (6) | 1/6 | 1/6 | 0/6 | 0/6 |
| partial 50 % (6) | 2/6 | 1/6 | 0/6 | 1/6 |

### Median residuals (degrees), all clips

| condition | C2 reader | C2 propagated |
| --- | --- | --- |
| feasible | **0.67** | **10.85** |
| tilt_removed | 6.14 | 3.93 |
| mirrored_roll | 3.85 | 15.30 |
| tilt_scaled | 4.33 | 9.80 |
| partial | 1.69 | 10.31 |
| thrust_limit | 18.39 | 33.95 |

### By family and mount

The full matrix is in `results/n05/summary.json` (`by_cell`); the pattern is that the propagated
arm's *feasible* false-flags concentrate at 45° (forest 3/6, industrial 4/7) while the reader's
concentrate at 15°/30° in forest (2/7, 3/7) — the mount-drift problem of N06, now visible as
false-rejects. Cells are 2-7 clips; every entry is a k/n count, not a rate with an interval.

### Second draw (exploratory)

| condition (matched) | C2 reader | C2 propagated draw 1 | C2 propagated draw 2 |
| --- | --- | --- | --- |
| feasible | 7/80 | 7/80 | 7/80 |
| tilt_removed | 16/27 | 0/27 | **1/27** |
| mirrored_roll | 10/27 | 7/27 | **5/27** |
| tilt_scaled | 12/33 | 3/33 | 3/33 |

The per-clip C2 difference between draws has median 1.02° and p90 9.71°; the conclusion does not
depend on the draw.

## Diagnostics (post-hoc, labelled)

| population | C2 reader | C2 DPVO + level start | C2 true rots + level start | C2 DPVO + oracle start |
| --- | --- | --- | --- | --- |
| feasible | 0.67 | 10.85 | **8.54** | 2.02 |
| tilt_removed | 6.14 | 3.93 | 0.00 | 3.93 |
| mirrored_roll | 3.85 | 15.30 | 11.26 | 5.32 |
| tilt_scaled | 4.33 | 9.80 | 8.85 | 4.43 |
| partial | 1.69 | 10.31 | 8.47 | 2.25 |

* **The start is the cause.** Replacing DPVO by the true rotations leaves 8.54° on feasible
  clips; the drift contributes the remaining ~2.3°. At matched false-reject, the oracle-start
  propagated C2 detects 6/27 removals, 7/27 mirrors, 5/33 scale — better than the level start but
  still behind the reader on removals.
* **Constant-bank removals are unobservable to C2.** With true rotations and a level start, the
  tilt-removed clips give a residual of exactly 0.00°: the video is level and the start is level,
  so the depicted lean is constant, and the offset fit explains the constant required lean. This
  is a property of the tier on constant-lean manoeuvres, not of the propagation.
* **Coherence needs varying signal.** 60 % of feasible clips (and 48 % of defect clips) have a
  required lean with zero variance, where the correlation is undefined and saturates. On the 87
  varying clips the natural threshold is at the ceiling; reader 10/14 removals and 10/14 mirrors,
  propagated 6/14 and 9/14.

## Yaw control

**Not measurable on the simulated set**: Track I's frozen split has no yaw-warp family, and a
world-yaw warp cannot be constructed without re-rendering. For an ideal propagated arm a world-yaw
warp is invisible to the lean (the up-vector is invariant), so its tier score would equal the
feasible score by construction — a statement, not a measurement. The measured yaw control is the
UZH one (N02b): reader F0 2/6, propagated F0 2/6, and the propagation did **not** remove the
reader's view dependence there.

## What this supports

* Without an absolute start, the propagated lean is **not** a drop-in replacement for the reader:
  the offset-free tiers do not absorb a level-start error of 10-30°, and the matched threshold is
  set by that error.
* The propagation itself is sound: the oracle-start residual is 2.02° on feasible clips, and N01
  showed 0.4-1.8° drift at 10 s in the gate strata. The missing ingredient is an absolute anchor.
* A deployment route needs the start from something: a flight-controller attitude, a single reader
  reading, or the generator's conditioning pose (N04). N05 deliberately excludes the first two.

## What this does not show

* How a **small** start error (1-3°, e.g. from a good flight controller or the commanded
  conditioning pose) behaves; Part 1's start-arm table suggests the propagated 10 s result then
  tracks the start error, but the tier comparison with such a start was not run.
* Anything about TII (excluded by instruction) or the held-out split (not read).

## Caveats

* The matched threshold is a global operating point, not a per-fold one; with six scene clusters
  it is descriptive, and the by-cell table is 2-7 clips per cell.
* The level start reads `R_cb` from the frozen split's mount (mechanical, deployment-available);
  it matches the ground-truth mount exactly (max difference 0).
* Coherence saturates on constant-lean clips; the pre-registration did not include a signal
  restriction, and the varying-subset table is a post-hoc diagnostic.
* DPVO draws differ: the per-clip C2 difference has p90 9.71°, so single-draw tier counts carry
  that noise. Both draws give the same verdict.

## Provenance

* Pre-registration `configs/n05_sim_verdict_prereg.yaml`; decisions `[N]`.
* Reader: `awmv-n-reader` on `awmv-n-data/sim200`, 200 clips at stride 2, 49,795 estimates
  (2,312 s wall over 4 shards; about 4.9 A10G-hours, USD 5.4).
* DPVO: two full draws of 200 clips at 640×360 (about 0.6 A10-hours each, USD 0.7 each).
* Inputs staged and hash-verified from the manifest: 200 videos and 200 gt files, 0 mismatches;
  no held-out file was opened.
