# Report F06 — Feed-forward extractors and the simulated set (round 2, parts 3 and 4)

Track F2. Written 2026-11-21. Pre-registration `configs/f04_round2_prereg.yaml` (commit
`63002b8`). Code: `remote/f2_extractors.py`; results `results/f06_feedforward/` and the Pi3X block of
`results/f04_spline/summary.json`.

**Decision: no new extractor supplies the path the verifier needs, and the simulated set is
not ready.** Pi3X with the Kalibr intrinsics is the best of the new models (6/10 on the
bake-off, 4.2° windowed lean error) but feeding it into part 1's statistic gives the same
failure as every other path (removal 0/8, mirror 2/8). Track I has committed its I02 plan and
rendered 8 of its 300 clips, with no I02 report yet, so part 4 waits.

## Plain-language summary

Part 3 asked whether an extractor built for metric 3D or for long-horizon pose estimation does
better than the streaming trackers on drone footage, and whether either supplies the thrust
axis the verifier needs. Two models were runnable without an access request: MapAnything
(Apache-2.0 weights) and Pi3X (multi-modal, accepts a camera matrix). VGGT's released API has
no intrinsics input, so its calibrated run is not possible, and FFVO has no code release.

MapAnything is excellent at what it advertises and poor at what the verifier needs: its
trajectory scale is metric to within 0.83–1.47 of truth on all ten sequences, but its
acceleration direction error is 22–83° and it passes the bake-off on 0 of 10. Pi3X with the
Kalibr matrix passes 6/10 — all six indoor-forward sequences, at 10–14° direction error and
0.11–0.32 m ATE — and is the second-best path by windowed lean error (4.2° against DPVO's
3.7°). It does not rescue part 1's statistic.

Part 4 waits on Track I: the I02 commit specifies the 300-clip, 149,450-frame set and freezes
its splits, and 8 clips have been rendered; no I02 report or complete dataset exists, so the
pre-registered leave-scene-out run is not performed.

## Part 3 — the preflight, done before any GPU job

| model | code | weights | gating | intrinsics input | run? |
| --- | --- | --- | --- | --- | --- |
| MapAnything | `facebookresearch/map-anything` (Apache-2.0) | `facebook/map-anything-apache` (Apache-2.0), `facebook/map-anything` (CC BY-NC 4.0) | both ungated, no token needed | yes in the view dict — **but broken in the released code** (below) | yes, image-only |
| Pi3X | `yyfz/Pi3` (BSD-3 code, BSD-2 weights per C02) | `yyfz233/Pi3X` | ungated | yes, `intrinsics` conditioning | yes, with Kalibr K |
| VGGT | `facebookresearch/vggt` | `facebook/VGGT-1B` (CC BY-NC 4.0) | ungated | **no** — the released API estimates everything | no new run; C03's uncalibrated number quoted |
| FFVO (arXiv 2609.13733) | no repository or project page found | none | — | — | **no** |

The FFVO paper ("A Feedforward Pose Decoder for Long-Horizon Visual Odometry", Shih et al.,
12 Sep 2026) describes a Google-internal model evaluated on Waymo/KITTI/proprietary data; the
arXiv HTML has no code link. Nothing to run, so nothing was run.

**The MapAnything intrinsics path is broken in the released code.** Passing a per-view
`intrinsics` matrix (the documented multi-modal input) raises
`IndexError: The shape of the mask [1] at index 0 does not match the shape of the indexed
tensor [392, 518, 3]` inside `_encode_and_fuse_ray_dirs`, because `load_images` has already
resized the images to 392×518 and attached its own ray directions. The run therefore uses the
documented image-only path and MapAnything's own estimated intrinsics; recorded as a
reproduction finding, not worked around.

## Part 3 — results

### Bake-off and verifier metrics

Same clips, same metric and same pass rule as E7; the lean-error column is part 2's metric.

| model | bake-off pass | median direction error | median magnitude ratio | median windowed lean error | metric scale |
| --- | --- | --- | --- | --- | --- |
| DPVO (round 1) | 10/10 | 4.2° | 0.98 | **3.69°** | 0.17–0.99 (unitless by construction) |
| ViPE (round 1) | 10/10 | 4.0° | 0.99 | 6.19° | 0.17–0.99 |
| **Pi3X** | **6/10** | 12.9° | 1.12 | **4.19°** | Sim(3) 1.04–1.59 |
| MapAnything | **0/10** | 40.6° | 2.12 (range 1.43–15.07) | 12.43° | **Sim(3) 0.83–1.47** |
| MASt3R-SLAM (round 1) | 5/10 | 39.0° | 0.81 | 10.49° | 0.18–2.8 depending on collapse |

Per sequence:

| sequence | MapAnything dir / ratio | Pi3X dir / ratio | MapAnything ATE (m) | Pi3X ATE (m) |
| --- | --- | --- | --- | --- |
| indoor_forward_3 | 37.1 / 1.91 | 10.2 / 1.09 | 0.52 | 0.11 |
| indoor_forward_5 | 39.1 / 2.32 | 10.9 / 1.15 | 0.31 | 0.12 |
| indoor_forward_6 | 30.1 / 1.43 | 11.0 / 1.07 | 0.64 | 0.12 |
| indoor_forward_7 | 21.8 / 1.72 | 6.5 / 1.09 | 1.16 | 0.32 |
| indoor_forward_9 | 37.9 / 1.71 | 14.0 / 1.07 | 0.30 | 0.11 |
| indoor_forward_10 | 42.1 / 1.86 | 11.8 / 1.08 | 0.50 | 0.11 |
| outdoor_forward_1 | 83.2 / 13.73 | 33.6 / 1.37 | 3.77 | 0.54 |
| outdoor_forward_3 | 81.2 / 15.07 | 41.9 / 2.86 | 5.99 | 0.72 |
| outdoor_forward_5 | 66.7 / 8.67 | 41.9 / 2.00 | 9.19 | 1.85 |
| indoor_45_9 | 81.8 / 3.01 | 48.0 / 4.47 | 7.25 | 4.25 |

**MapAnything's metric scale is the best in the project** — Sim(3) 0.83–1.47, median 1.20,
against a truth scale of 1 by definition — and its trajectory error is small on the indoor
sequences (0.30–1.16 m). Its acceleration *direction* is 22–83°, so it fails the bake-off on
every sequence, and its windowed lean error (12.4° median) is the worst of the five. Metric
space and motion fidelity are separate properties, and the verifier needs the second.

**Pi3X with the Kalibr matrix is the best of the new models**: six indoor-forward passes at
6.5–14.0° direction, 1.07–1.15 magnitude ratio and 0.11–0.32 m ATE, then the same outdoor and
45-degree failures as the uncalibrated Pi3 in C03 (33.6–48.0°). Supplying the calibration
improved the ATE substantially (C03's Pi3 had 0.5–1.2 m indoors) but did not change which
sequences pass, so C07's "the calibration is the difference" remains true for the *feed-forward
batch models* in the sense C07 measured (DPVO versus these models), not as a claim that the
calibration is what each model lacks.

### Feeding the best into part 1

The pre-registered selection rule is "the extractor with the lowest median windowed lean
error". Among all five paths that is DPVO (3.69°), already part 1's primary, so no new path
replaces it. The best *new* extractor, Pi3X, was fed in as a labelled secondary (it is the
model part 3 actually ran). With estimated attitude and the Pi3X path, on the same 8 clips and
thresholds:

| statistic, Pi3X path | real | removal | mirror | AUC removal/mirror | gate |
| --- | --- | --- | --- | --- | --- |
| S_dyn (part 1) | 0.125 (1/8) | 0.000 (0/8) | 0.250 (2/8) | 0.58 / 0.67 | fail |

The gate therefore fails on the best new extractor as well. That is consistent with part 1's
oracle-path rows: the statistic detects 8/8 and 6/8 on the truth path for every extractor, so
what differs between paths is not enough to cross the bar.

## Part 4 — the simulated calibration set

**Not run: the set is not ready.** State on 2026-11-21:

* Branch `track-i-simset` at `2d815cc` ("I02: scene families, defect labels, dataset builder
  and the render/publish/scoring apps") with `configs/sim_splits_frozen.json` and
  `results/i02/composition.json`: **300 clips, 149,450 frames**, dev/held-out splits by scene
  family, defect labels including `tilt_removed`.
* `awmv-i-data:/i02/clips` holds **8 rendered clips** (`i02_dev_000`, `_001`, `_080`,
  `_107`, `_134`, `_135`, `_167`, `_180`), each with mp4, `gt.npz`, `render.npz` and
  `manifest_row.json`.
* There is **no `reports/I02_*.md`** on the branch yet.

The pre-registered part 4 runs the best pixel-path pipeline on the full set with
leave-scene-out calibration. Eight clips cannot carry it: with five scene families a leave-one-
scene-out calibration would have four families, and the defect labels are spread across the
300. A run on 8 clips would be anecdotal, would not be the registered analysis, and is not
reported as a result.

**What is waiting, and on what:** the remaining 292 renders and the I02 report. When they
exist, the run is: DPVO (the best path by part 2's metric) on all 300 clips, the verdict
statistic that has shown power at the pixel-path level — round 1's shooting statistic, not
part 1's spline statistic — calibrated leave-one-scene-out, reporting detection and
false-reject by defect family. It must be stated that simulated frames are cleaner than real
or generated video, so it is an upper bound on transfer.

## What this supports

* **Metric 3D does not imply motion fidelity.** MapAnything's scale is metric to 17 % while its
  acceleration direction is off by 40°; a metric scale is not what the verdict lacks (round 1's
  V5 showed the same from the other side).
* **The best new extractor is a batch model with the calibration supplied.** Pi3X at 6/10 and
  4.2° lean error is usable as a path for a position-level verdict; it is not better than DPVO
  by the verifier's metric.
* **No extractor tested in either round clears the verdict gate with the spline statistic.**
  The extractor question and the statistic question remain separable, and part 1 shows the
  statistic is the weaker link for this particular form.

## What this does not show

* What FFVO would do: it has no released code or weights.
* What VGGT would do with a calibration input: its API does not accept one.
* Anything about the simulated set beyond its readiness state.

## Caveats

* MapAnything ran with estimated intrinsics, so its failure is not evidence about a calibrated
  MapAnything; the calibrated input path is broken in the released code.
* Pi3X's outdoor and 45-degree failures match C03's uncalibrated Pi3, so the calibration does
  not rescue those rigs.
* The windowed lean error for MapAnything is computed with the Sim(3) scale, which is close to
  1 here; for the unitless paths it absorbs their arbitrary scale.
* Cost: MapAnything 10 sequences (431 s + 70 s), Pi3X 10 sequences (270 s + 108 s) on an
  A100-80GB, about **US$0.7**; the ledger records it.

## Reproduce

```
pixi run modal run remote/f2_extractors.py --method map --probe indoor_forward_5
pixi run modal run remote/f2_extractors.py --method map
pixi run modal run remote/f2_extractors.py --method pi3
python experiments/f05_lean_error.py --out results/f05_lean_error   # scores all five paths
python experiments/f04_spline_fit.py --path all --out results/f04_spline   # includes Pi3X
```

Provenance: `results/f06_feedforward/summary.json` (preflight and per-sequence scores),
`results/f05_lean_error/lean_error.json`, `results/f04_spline/summary.json` (Pi3X block); poses on
`awmv-f2-data:/poses/{mapanything,pi3x}`.
