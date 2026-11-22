# Report F05 — What path error the verifier needs (round 2, part 2)

Track F2. Written 2026-11-21. Pre-registration `configs/f04_round2_prereg.yaml` (commit
`63002b8`). Results: `results/f05_lean_error/lean_error.json`, `sequences.csv`.

**Decision: the metric changes the ranking, and the prediction is weak.** The thrust-axis error
a path implies in E05 rule-sized windows is 3.7° median for DPVO, 4.2° for Pi3X, 6.2° for
ViPE, 10.5° for MASt3R-SLAM and 12.4° for MapAnything — a different order from Track C's
bake-off direction error, which puts ViPE first. The windowed lean error correlates with the
bake-off error at 0.65 across 50 path–sequence pairs but does **not** predict per-clip verdict
quality well enough to use (Spearman 0.22–0.46 against part 1's statistic).

## Plain-language summary

The bake-off asks whether a recovered path's *median acceleration direction* matches truth.
The verifier asks something narrower and stricter: does the path imply the *thrust axis* the
airframe actually had, in the windows the verdict uses? This part measures the second quantity
for every stored path and compares the two.

Two things come out. First, the ordering differs. ViPE has the best bake-off direction error
(4.0° against DPVO's 4.2°) but a worse windowed lean error (6.2° against 3.7°), because its
trajectory error enters the thrust-axis computation through the velocity, the drag term and
the scale as well as through the acceleration direction. Pi3X, which the bake-off puts well
behind both (12.9° median direction), is second on the verifier's metric (4.2°), driven by
its six indoor sequences at 2.8–5.7°. Second, the two metrics are only moderately related
(Spearman 0.65, sequence-clustered interval 0.38–0.83 over 10 clusters).

The pre-registered prediction — that windowed lean error should predict per-clip verdict
quality — is not supported: Spearman 0.31 for DPVO, 0.22 for ViPE, 0.46 for MASt3R, 0.39 for both Pi3X
and MapAnything, with intervals that cross zero for four of the five. That is partly because part 1's statistic is a
poor verdict statistic (F04); the analysis can be repeated when a statistic with power exists.

## The metric

For one path on one sequence, with oracle attitude and the Sim(3) evaluation transform:

```
u_path = (s·a + g·e3 + k_d·s·v) / ||.||      spline derivatives; s from the 150-frame Sim(3)
u_true = R_bw · e3                            truth body z (thrust axis)
lean error = angle(u_path, u_true)             yaw-invariant
```

with `k_d = 0.295`, cubic B-spline at the E05 knot rule (`0.5/f95`), and non-overlapping
windows of `0.5/f95` s. The reported number is the median of the per-window medians; p75 and
the window count are in the CSV.

**One implementation correction, recorded because it changes the number by 5×.** The first
version of this script aligned the path to truth with the *attitude-derived* rotation
`mean(R_wc R_cbᵀ R_bwᵀ)` instead of the Sim(3) rotation. On `indoor_forward_3` that read
16.2° where the Sim(3) alignment reads 3.4°; the attitude product carries DPVO's absolute
orientation drift, while the position alignment does not. The pre-registration says "the
Sim(3) scale from the whole 150-frame alignment"; the rotation is taken from the same
alignment, and the diagnostic is in `docs/decisions.md`.

## Results

### Per path, over the ten sequences

| path | windowed lean error, median (range) | bake-off direction error, median (range) | windows per sequence |
| --- | --- | --- | --- |
| DPVO | **3.69°** (1.74–12.96) | 4.18° (1.45–20.88) | 6–28 |
| ViPE | 6.19° (2.66–9.55) | **3.97°** (2.13–11.97) | 6–28 |
| Pi3X | 4.19° (2.75–47.99) | 12.92° (6.5–48.0) | 6–28 |
| MASt3R-SLAM | 10.49° (2.57–38.25) | 39.05° (7.9–95.1) | 6–28 |
| MapAnything | 12.43° (3.26–60.97) | 40.61° (21.8–83.2) | 6–28 |

The number of windows follows the sequence's `f95`: 6 on `outdoor_forward_1` (window 1.42 s
over a 10 s path), 28 on `indoor_forward_6` (0.34 s). Six windows is a small sample; the p75
column in `sequences.csv` shows how much the per-sequence number moves.

### Per sequence, the three best paths

| sequence | DPVO lean | ViPE lean | Pi3X lean | DPVO bake-off | ViPE bake-off |
| --- | --- | --- | --- | --- | --- |
| indoor_forward_3 | 3.68 | 4.56 | 3.58 | 1.78 | 2.51 |
| indoor_forward_5 | 1.74 | 2.66 | 2.75 | 2.48 | 3.20 |
| indoor_forward_6 | 4.49 | 7.81 | 5.71 | 3.12 | 4.75 |
| indoor_forward_7 | 12.96 | 6.11 | 4.85 | 20.88 | 2.37 |
| indoor_forward_9 | 3.71 | 4.93 | 3.97 | 1.45 | 2.13 |
| indoor_forward_10 | 3.33 | 2.87 | 3.46 | 2.33 | 2.28 |
| outdoor_forward_1 | 2.66 | 6.28 | 3.24 | 6.88 | 5.42 |
| outdoor_forward_3 | 3.66 | 8.15 | 4.42 | 5.23 | 9.89 |
| outdoor_forward_5 | 5.21 | 9.55 | 10.81 | 11.20 | 11.97 |
| indoor_45_9 | 9.07 | 8.16 | 47.99 | 13.24 | 6.33 |

(ViPE is better on `indoor_forward_7` by both metrics, and worse on `indoor_forward_3`,
`_6`, `outdoor_forward_1`, `_3`, `_5`. Pi3X is the most consistent indoor path and fails only
on `outdoor_forward_5` (10.8°) and the 45-degree rig (48.0°). MASt3R and MapAnything are in
the CSV.)

### Correlations

Rank correlation between the windowed lean error and the bake-off direction error:

| set | Spearman | n | sequence-clustered interval |
| --- | --- | --- | --- |
| all path–sequence pairs | 0.65 | 50 | 0.38–0.83 |
| DPVO | 0.54 | 10 | — |
| ViPE | 0.79 | 10 | — |
| Pi3X | 0.43 | 10 | — |
| MASt3R | 0.72 | 10 | — |
| MapAnything | 0.33 | 10 | — |

The pooled value is descriptive: 10 clusters and 5 paths, so the interval is as much about
sequence sampling as about the metric.

### Does lean error predict per-clip verdict quality?

Per real clip (22 with a finite statistic per path), median windowed lean error over the
clip's window against part 1's real-condition statistic:

| path | Spearman | n | clustered interval | reading |
| --- | --- | --- | --- | --- |
| DPVO | 0.31 | 22 | −0.10–0.46 | no usable link |
| ViPE | 0.22 | 22 | −0.09–0.41 | no usable link |
| MASt3R | 0.46 | 20 | 0.11–0.70 | weak positive |
| Pi3X | 0.39 | 22 | −0.14–0.64 | no usable link |
| MapAnything | 0.39 | 22 | −0.07–0.71 | no usable link |

The pre-registered hypothesis ("lean error predicts verdict quality better than the bake-off
direction error") is **not supported** on this sample. The honest qualification is that the
verdict statistic in part 1 has little power (F04); the test would be more informative against
a statistic that separates, and round 1's shooting statistic is the natural candidate for a
re-run.

## What this supports

* **The bake-off metric is not the verifier's metric.** The clearest case is ViPE: best median
  acceleration direction, worse windowed thrust-axis error than DPVO on six of ten sequences.
  Any extractor selection for this project should use the windowed lean error.
* **DPVO is still the best path for the verifier by this measure**, with Pi3X close behind and
  ViPE third; MASt3R and MapAnything are far off.
* **The metric is sensitive to how the path is aligned to truth.** Using the attitude product
  instead of the Sim(3) rotation inflated `indoor_forward_3` from 3.4° to 16.2°, which would
  have changed the ranking. The choice is now recorded and fixed.
* **The bake-off's model-level pass rule is too coarse to rank paths for the verifier.** Four
  of the five models meet it — DPVO 10/10, ViPE 10/10, Pi3X 6/10, MASt3R 5/10 (exactly the
  ≥0.5 threshold) — while their windowed lean errors span 3.7–10.5°. Only MapAnything fails
  (0/10) and it sits at 12.4°. If path selection is to serve the verdict, the windowed lean
  error (or a similar thrust-axis metric) has to be the selection criterion, not the pass
  count.

## What this does not show

* Which part of the 3.7–12.4° is path shape, path scale, or the drag term: the metric mixes
  them by construction. Part 1's oracle-path rows isolate the statistic side.
* That the four-degree level is sufficient for a verdict: F04 shows that even DPVO's path
  leaves too much residual for the spline statistic, and round 1 shows 6/8 removal detection.
* Anything about simulated or generated video.

## Caveats

* Six windows on the shortest sequence; per-sequence numbers are medians of six to 28 values.
* The Sim(3) transform uses truth and is evaluation-only. A deployable system would have to
  estimate the scale (part 1's fit does, badly) and the gravity frame (from the attitude
  source).
* The truth world is gravity-aligned to 1.4° (E's audit), which sets a floor no path metric
  can go below without absorbing that error.
* `indoor_45_9` is included here even though it is out of scope for the depicted lean; part 2
  is path-only, and it is reported with the rest.

## Reproduce

```
python experiments/f05_lean_error.py --out results/f05_lean_error
```

Provenance: `results/f05_lean_error/lean_error.json` (code commit at run time, per-path
sequence rows, correlations with intervals), `sequences.csv` (one row per path × sequence).
Cost: zero GPU.
