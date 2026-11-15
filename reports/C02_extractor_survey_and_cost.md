# Report C02 — Extractor availability, cost, and the T1 substitution

Track C. Written 2026-11-10. Task 1 of handoff 06, plus the Task 2 blockers found while
preparing the testbeds.

## Plain-language summary

Five pose extractors are available, ungated, and cheap enough to run: **DPVO** (the baseline
this track is checking), **DA3**, **VGGT**, **Pi3**, and **MetricTrail** (DPVO plus a metric
depth prior, shipped inside NVIDIA's ViPE). Everything is public — no HuggingFace token, no
application form. Measured on an A10G, a 100-frame clip costs between 5 s (Pi3) and 22 s
(VGGT) and 7 to 14 GB of GPU memory, so the whole bake-off is a few dollars, not a budget
conversation.

Two blockers, both resolved by substitution rather than by waiting:

1. **Blackbird's download server is gone.** `blackbird-dataset.mit.edu` does not resolve, and
   the repo's own issue tracker has four open reports of the same ("download address dead",
   "unable to download dataset", "server unreachable from Colab"). The designated T1 cannot
   be fetched.
2. **Track A was wrong that UZH-FPV ships no calibration.** The calibration is a *separate*
   download at `rpg.ifi.uzh.ch/datasets/uzh-fpv/calib/*.zip` (4.1 MB per camera rig), not
   inside the sequence zip Track A inspected. It contains a proper Kalibr camchain for the
   Snapdragon: `pinhole` camera with `equidistant` (Kannala-Brandt) distortion,
   fx = 278.667, fy = 278.490, cx = 319.752, cy = 241.969 at 640x480.

So **UZH-FPV replaces Blackbird as T1**: real drone racing, 3.5 to 20.7 m/s, Leica
laser-tracker truth, and trustworthy intrinsics — the exact property T1 was chosen for.

## The availability survey

Every row was read from the project's repo or model card on 2026-11-10; "measured" means we
ran it here.

| extractor | code | weights | licence | intrinsics | metric scale | clips 16-150 |
| --- | --- | --- | --- | --- | --- | --- |
| DPVO | `princeton-vl/DPVO` | Dropbox `models.zip` | MIT | required | no | yes (streaming) |
| MetricTrail | `nv-tlabs/vipe` (`vipe/metrictrail/`) | `nvidia/MetricTrail` → `dino_dpvo.pth` | Apache-2.0 code; **NVIDIA non-commercial** weights | estimated (SharedK) or supplied | **yes** (MoGe-2 prior) | yes (streaming) |
| DA3 | `ByteDance-Seed/Depth-Anything-3` | `depth-anything/DA3-LARGE-1.1` | Apache-2.0 repo; LARGE weights **CC BY-NC 4.0** (HF card says apache-2.0 — conflict recorded) | estimated | no (LARGE); DA3METRIC-LARGE is Apache-2.0 | yes (batch) |
| VGGT | `facebookresearch/vggt` | `facebook/VGGT-1B` | custom, commercial-OK except military; weights **CC BY-NC 4.0** | estimated | no | yes (batch, memory-bound) |
| Pi3 | `yyfz/Pi3` | `yyfz233/Pi3` | BSD-3 code, **BSD-2** weights (academic) | estimated | no | yes (batch) |
| MegaSaM | `mega-sam/mega-sam` | upstream DROID-SLAM | Apache-2.0 | required | no | fallback only |
| MoGe-2 | `microsoft/moge` | `Ruicheng/moge-2-vitl` | **MIT** both | — | **yes** (depth only) | n/a |
| DASH-SLAM | — | — | **no code release found** | — | yes (DA3) | n/a |

DASH-SLAM contributes method, not code: it is the paper that says out loud that Sim(3) ATE
hides scale errors (64-76 m on 659-1427 m aerial paths while indoor scale was within 5 %).

## Measured cost (A10G, 24 GB)

| extractor | 16 frames, 224x224 | 100 frames, 320x240 | peak GPU (100 fr) | notes |
| --- | --- | --- | --- | --- |
| DPVO (Track A's numbers) | — | 56 frames/s | 1.16 GB | 48 patches, 320x240, `no_grad` |
| DA3-LARGE-1.1 | 15.1 s (model load dominates) | 16.9 s → 0.169 s/frame | 11.9 GB | depth returned, **not** metric |
| VGGT-1B | 2.7 s | 21.8 s → 0.218 s/frame | 14.0 GB | resizes to 518 px |
| Pi3 | 1.0 s | 5.5 s → 0.055 s/frame | 7.5 GB | fastest by 3-4x |

Cost estimate for the full bake-off (5 UZH sequences x 150 frames, 8 TII flights x 100,
82 ANWM clips x 16, three models): about **45-60 GPU-minutes, roughly US$1**, plus one
MetricTrail build and run (about 1 h, US$1.1). Total under US$3, so no approval gate is
triggered; recorded here because the handoff asks for an estimate before Task 3.

## What was built

| file | role |
| --- | --- |
| `src/pathrec/align.py` | Umeyama Sim(3)/SE(3), gravity-aware yaw+scale fit, path length, both-error reporting |
| `src/pathrec/lean.py` | differentiation with a window in seconds or steps, lean vector, direction/magnitude/correlation, required tilt with `k_d` |
| `src/pathrec/evaluate.py` | per-clip score: ATE (both), direction, acceleration, tilt, rotation, coverage, up-axis detection |
| `src/pathrec/reliability.py` | truth-free signals: parallax-to-depth, jitter ratio, rotation rate, cross-extractor agreement |
| `src/pathrec/priors.py` | scale ratios and intervals, the plausibility-interval rule, scale-free lean regression and apparent gravity |
| `src/pathrec/testbeds.py` | UZH-FPV (calibration + Leica truth), TII central crop, ANWM clips with their same-start truth frames |
| `remote/c_extractors.py` | DA3/VGGT/Pi3 inference, smoke test and UZH scoring, on Modal |
| `remote/c_prepare.py` | downloads UZH sequences, undistorts KB4 → pinhole, saves frames + truth, reports kinematics |
| `tests/pathrec/` | 26 unit tests over synthetic trajectories (identity, scale leakage, drag tilt, invariance) |

## Dataset facts recorded (do not guess these later)

- UZH-FPV timestamps: `left_images.txt` rows are `id timestamp_unix_seconds image_name`; the
  *filenames* carry an index, not a time. Ground truth is TUM-format
  `timestamp tx ty tz qx qy qz qw` at 500 Hz on the same Unix clock, and it covers only part
  of each recording (15100 rows for 1970 frames on `indoor_forward_6`), so frames outside the
  truth interval are dropped rather than extrapolated.
- UZH image rate is 30.1 Hz measured; we take every second frame (15 Hz) so a 0.5 s
  differentiation window spans 7 samples.
- The calibration zip's `camchain` is the authority for intrinsics; the sequence zips do not
  contain one.
- ANWM clips are nested (`A/A00`), 16 generated frames plus 16 same-start ground-truth frames,
  with a commanded path in the manifest. The commanded path is a command, not a depiction.

## What this supports

- The bake-off can start immediately with three extractors, and MetricTrail is available if
  its build succeeds; every licence permits non-commercial research use.
- UZH-FPV is a better T1 than Blackbird for the question at hand: it has the calibration
  quality T1 was supposed to isolate, plus 20.7 m/s flights.
- Cost is not a constraint: the whole study is a few dollars of A10G time.

## What this does not show

- Nothing about accuracy yet. No extractor has been scored against truth at the time of
  writing; the first UZH scores are in the Gate 1 report.
- MetricTrail's build was not validated in this pass. Its weight licence is
  non-commercial, and its DPVO extensions must compile inside ViPE's image.
- Blackbird remains unavailable; if its server returns, T1 could be re-run on rendered
  frames, which would add a synthetic-camera case UZH does not cover.

## Caveats

- DA3-LARGE returns *relative* depth; the metric path is `DA3METRIC-LARGE` (Apache-2.0) or
  MoGe-2, which is a separate choice in the scale study.
- VGGT's 16-frame number is at 518 px (its own preprocessing), so it is not comparable to
  Pi3's 224 px; the 100-frame column is the one to use for budgeting.
- The `evo`/`moviepy`/`open3d` dependency chain in DA3 is bypassed in our image by stubbing
  DA3's exporter package: we only need poses and depth, and the exporters pull gsplat.
