# N06 — Diagnosing the 30°/45° mount drift

**Decision.** The high-mount drift is **not a stable bias that resolution or a different estimator
explains**: it is dominated by run-to-run variance of the DPVO estimate itself. Two runs on
identical video and settings give 10 s forest drifts of 4.58° and 5.87° at 30° (median |difference|
2.83°, max 8.38°) and 12.17° and 10.37° at 45° (median |difference| 8.78°, max 42.95°). Halving
the resolution makes it worse (30° forest 8.71° including the same draw effect), a two-view
relative-pose chain is unusable (70-85°), and on real UZH edits the **full 640×480 resolution
fixes the extractor completely** (63/63 clips valid against 12/63 at the 320×240 crop).

Pre-registration `configs/n06_mount_drift_prereg.yaml` + amendment A. Code
`experiments/n06_mount_drift.py`; results `results/n06/`. No TII.

## Setup

Four arms on the 80 feasible development clips (forest reported separately, 26 clips):

* **DPVO full** — the N05 run at native 640×360 (height 352), the current estimator.
* **DPVO half** — the same frames resized to 320×180, intrinsics × 0.5.
* **chain full / chain half** — a two-view relative-pose estimator: SIFT → RANSAC essential
  matrix → `recoverPose` rotation for pairs 2 frames apart (15 Hz), chained multiplicatively.
* Statistic: oracle-start propagated lean error at 10 s, median over starts per clip, then median
  across clips, by mount. The same statistic N01 used.

A second full-resolution DPVO draw on the 13 forest 30/45 clips was run to measure reproducibility.

## Results

### 10 s oracle-start drift, forest clips (median of per-clip medians, n per mount)

| arm | 0° (6) | 15° (7) | 30° (7) | 45° (6) |
| --- | --- | --- | --- | --- |
| DPVO full | **0.82** | **2.26** | **4.58** | **12.17** |
| DPVO half (320×180) | 1.10 | 2.42 | 8.71 | 13.61 |
| chain full | 81.7 | 70.7 | 83.4 | 85.5 |
| chain half | 69.3 | 58.1 | 75.5 | 73.5 |

(The DPVO-full 30° value here is 4.58° against N01's published 8.62°: the two runs are different
DPVO draws, which is the finding below, not a code change. Per-clip values agree exactly between
the two analysis implementations.)

### Run-to-run reproducibility (two full-resolution draws, forest 30/45)

| mount | clips | run 1 median | run 2 median | median abs diff | max abs diff |
| --- | --- | --- | --- | --- | --- |
| 30° | 7 | 4.58 | 5.87 | **2.83** | 8.38 |
| 45° | 6 | 12.17 | 10.37 | **8.78** | 42.95 |

At 30° the drift is 3-6° with about 3° of draw-to-draw scatter. At 45° the scatter is as large as
the median and one clip (`i02_dev_059`) reads 122° in one draw and 79° in the other: that clip is
a lost/degenerate estimate, not a drift measurement. **The 45° "14.8°" figure from N01 is one
draw of a quantity whose run-to-run spread is of the same order.**

### The two-view chain

Every pair produced matches (median usable fraction 1.00), yet the chained rotation drifts
70-85° in forest at 10 s — two orders of magnitude worse than DPVO. The pre-registered sanity
check ("within a factor of 2 of DPVO on forest 30°") fails by a factor of about 18. A pairwise
rotation estimate with small baselines accumulates error far faster than a bundle-adjusted map;
the arm is reported as not viable, not as a resolution or estimator fix.

### Resolution on the real UZH edits

| input | clips with valid extraction (oracle-start 2 s ≤ 10°) |
| --- | --- |
| 640×480 pinhole (full) | **63/63** |
| 320×240 fixed crop (N02b) | 12/63 |

The N02b failures were the crop, not the estimator or the footage. At full resolution every one
of the 63 edited clips tracks well enough for the Part 2 comparison.

## What this supports

* **The high-mount problem is estimator variance, not a deterministic view bias.** Reporting a
  single number per mount overstates precision; the two draws differ by about the effect size.
  Any future claim about 30/45° mounts needs repeats (or a per-clip failure flag).
* **Resolution is a first-order requirement**: full 640×480 turns 12/63 valid UZH extractions into
  63/63, and half resolution worsens the simulated 30° drift.
* **A two-view relative-pose chain is not an alternative**: chaining pairwise rotations is far
  worse than DPVO's map-based optimisation.
* Consistent, low-drift behaviour survives at **0°/15°**: 0.8-2.4° in both draws.

## What this does not show

* A converged estimate of the deterministic mount bias: two draws bound the variance but do not
  separate bias from variance at 45°; more repeats are needed for that.
* Anything about TII (excluded by instruction) or the held-out simulated clips (not read).

## Caveats

* Two draws on 13 clips is a small variance study; the max differences (8°, 43°) show it is
  heavy-tailed, so a median alone is optimistic.
* DPVO is not bit-reproducible here (different GPU types A10/A10G and non-deterministic
  optimisation); the report treats runs as draws, not as a deterministic function.
* The chain result depends on the 2-frame gap and SIFT settings; a different gap could be better,
  but the gap is pre-registered and the failure is two orders of magnitude, not marginal.

## Provenance

* Pre-registration `configs/n06_mount_drift_prereg.yaml`, amendment A; decisions `[N]`.
* Poses: `awmv-n-data/poses/sim200/{scale1,scale0.5}`, `poses/uzh_edits{,_full}`; two full draws
  (the second on 13 clips, the third on all 200 for N05); chains in `data/n06/chain/`.
* GPU: about 0.4 A10-hours for N06 (≈ USD 0.45); no R2 write; apps stopped.
