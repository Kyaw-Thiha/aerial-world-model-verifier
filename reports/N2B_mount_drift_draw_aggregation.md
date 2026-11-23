# N2B — Part B(iii): two-draw robust aggregation against the high-mount drift

**Decision.** Averaging two independent DPVO draws does **not** meet the pre-registered reduction:
the 45° forest drift falls only to 8.65° from a mean-of-singles of 8.95° (needs 25 %), and at 30°
the aggregate (8.30°) is 81 % worse than the better single draw (4.58°). It does cut the worst
per-clip error on the high-mount forest clips (19.7° against 41.5° for the worse single draw), and
the two draws disagree by a median 15.6° at 45° in forest — large enough that the deployable remedy
is repeats plus a per-clip validity flag, not averaging.

Pre-registration `configs/n08_mount_draws_prereg.yaml`, committed before any score. Code
`experiments/n08_mount_draws.py`; results `results/n2/n08/`. CPU only, **zero spend**. No TII.

## Question

N06 found the 30°/45° mount drift dominated by run-to-run variance. If the drift is variance
around a small common bias, averaging independent draws should shrink it; if a systematic mount
bias dominates, it cannot. Which is it, per mount?

## Setup

**Change (the single one).** For every clip, form the 10 s oracle-start predicted world-up from
each draw separately (N06's construction), average the two unit vectors componentwise and
renormalise, then score the aggregate with N06's statistic: median error over the 10 s horizon
samples per clip, then median across clips, by mount; forest reported separately. A per-clip
validity flag drops a clip from every arm if either single draw reaches 85° at any horizon sample.

**Population.** The 80 feasible development clips (26 forest; 6–7 per mount). One clip
(`i02_dev_059`, 45° forest) is validity-flagged — its two draws reach 132° and 113° — leaving
**79 clips in every arm**. Held-out clips were never opened.

**Draws.** Draw 1 = `data/n05/poses/sim200/scale1`; draw 2 = `scale1_rep2`. Draw 2 reproduces N05's
published draw-2 C2 values exactly (median |difference| 0.0 over 200 clips), so the two draws are
genuinely independent runs.

## Results

10 s oracle-start drift (degrees), **all 80 feasible clips** (n = 20 per mount, 19 at 45°):

| arm | 0° | 15° | 30° | 45° |
| --- | --- | --- | --- | --- |
| draw 1 | 0.70 | 1.64 | 3.79 | 12.03 |
| draw 2 | 0.57 | 1.16 | 5.56 | 10.75 |
| **two-draw aggregate** | 0.69 | 1.60 | 4.72 | **8.65** |
| median of per-clip means | 0.83 | 1.86 | 5.52 | 12.18 |
| median disagreement | 0.69 | 1.47 | 6.64 | 13.69 |

**Forest only** (n = 6/7/7/5 — every cell descriptive, daggered):

| arm | 0° (6) | 15° (7) | 30° (7) | 45° (5) |
| --- | --- | --- | --- | --- |
| draw 1 | 0.82 | 2.26 | **4.58** | 12.06 |
| draw 2 | 0.69 | 1.21 | 8.15 | 5.85 |
| **aggregate** | 0.72 | 1.88 | 8.30 | 8.65 |
| max per-clip error, draw 1 | 1.60 | 4.11 | 8.95 | 12.06 |
| max per-clip error, draw 2 | 1.22 | 1.85 | 11.24 | 10.51 |
| max per-clip error, aggregate | 1.15 | 2.26 | 11.42 | **10.25** |

**Draw-to-draw disagreement** (angle between the two draws' predicted ups): median 6.6° at 30° and
13.7° at 45° over all clips; in forest 45°, the median is 15.6° and **every** clip exceeds 15°
disagreement on at least some samples. At 0°/15° the disagreement is 0.7–1.5°.

### Pre-registered criteria, scored

| criterion | result | numbers |
| --- | --- | --- |
| c1 reduction | **FAIL** | 45° forest aggregate 8.65° vs mean of singles 8.95° (−3.4 %, needs −25 %); 30° aggregate 8.30° vs better single 4.58° (+81 %, needs ≤ +10 %) |
| c2 no harm | **PASS** | 45° forest worst per-clip error: aggregate 19.7° vs worse single 41.5° |
| c3 reading | — | 45° forest median disagreement 15.6° is large → the estimator is unstable per clip; repeats plus a validity flag, not averaging |

## What this supports

* **The high-mount drift is not an independent zero-mean draw noise that averaging removes.** At
  45° the aggregate sits between the two draws and only 3 % below their mean; at 30° it lands near
  the worse draw. The errors are correlated in direction across draws (a common mount-dependent
  bias or a correlated failure mode), which is why combining does not help the median.
* **Averaging does help the tail.** The worst high-mount forest clip falls from 41.5° in the worse
  draw to 19.7° in the aggregate. A repeat-and-average is a robustness measure, not an accuracy
  measure.
* **The per-clip instability is confirmed and quantified**: median 15.6° draw disagreement at 45°,
  every forest 45° clip above 15° at some samples, and one clip (dropped) at 110–130°.
* **0°/15° are stable** (0.6–2.3° across both draws and the aggregate), so the low-mount result is
  not a draw artefact.

## What this does not show

* Whether a *different* single change would help: this is one change (aggregation). N06's chain and
  half-resolution arms are the other two and remain as reported.
* A converged per-mount bias: two draws cannot separate bias from variance. The 45° median of the
  aggregate (8.65°) is still far above the 0.7–1.9° of the low mounts, so a systematic component
  remains.
* Anything about TII (excluded) or the held-out sim clips (not read).

## Data-cache note (reported, not repaired)

N06's "second draw" was read from `data/n06/poses_repeat/scale1`. That directory **no longer holds
a third independent draw**: it now duplicates draw 1 on 188/200 clips (the 8 exceptions are the
clips where draws 1 and 2 happen to coincide). N06's published run-2 medians (5.87° at 30°,
10.37° at 45°) therefore cannot be reproduced from the current cache. N06's numbers stand as
reported and are **not** recomputed or reinterpreted here; this analysis uses `scale1_rep2`, which
does reproduce N05's draw-2 values. Recorded in `docs/decisions.md` and `docs/N2_STATE.md`.

## Caveats

* Forest cells are 5–7 clips; every forest number is descriptive, and the 45° forest maximum is one
  clip.
* The aggregate is a componentwise mean of unit vectors, not a geodesic mean; for disagreements up
  to ~16° the two agree to second order, so the choice does not drive the result.
* The validity flag (85° at any horizon sample) drops exactly one clip; a stricter flag would drop
  more and could change the tail reading.
* DPVO is not bit-reproducible, and the two draws here are two samples of a heavy-tailed quantity;
  a third draw could move the 45° median by the amount reported in N06.

## Provenance

* Pre-registration `configs/n08_mount_draws_prereg.yaml`; decisions `[N2]`.
* Poses `data/n05/poses/sim200/{scale1,scale1_rep2}` (read-only); truth from `data/n05/gt`.
* Code `experiments/n08_mount_draws.py`; results `results/n2/n08/{per_clip.csv,summary.json}`.
* Compute: local CPU, about 2 s, **USD 0.00**. No Modal app, no volume, no R2 write.
