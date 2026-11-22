# Cost ledger

One row per job set: date, job, GPU, hours, USD.  Modal list prices: A10 1.10, L40S 1.95,
A100-80GB 2.50, H100 3.95 USD/h.  CPU jobs are recorded at $0.05/core-hour.  `estimate`
rows are posted before a job set expected to exceed US$20; `actual` rows are measured from
the run's own manifests.  Track H3/H4 ceiling: **US$60**, stop after the campaign.

| date | job | GPU | hours | USD | status | notes |
| --- | --- | --- | --- | --- | --- | --- |
| 2026-11-22 | H03 reader image build + rotation sign probe (12 reads, GeoCalib + Perspective Fields) | A10 | 0.09 | 0.10 | actual | image build dominates; 28 s of inference |
| 2026-11-22 | H03 readers on the 150-frame sample (GeoCalib pinhole + native fisheye, Perspective Fields) | A10 | 0.10 | 0.11 | actual | 75 s of inference, 150 frames |
| 2026-11-22 | H03 Perspective Fields recompute on full clips (LingBot, SolarWM grid/seeds/probes, ANWM, sample) | A10 | 0.70 | 0.77 | actual | about 9,000 frames, ~20 s per 81-frame clip |
| 2026-11-22 | H04 SolarWM convention probes (2 clips x 81 frames) | A100-80GB | 0.09 | 0.23 | actual | 330 s including model load |
| 2026-11-22 | H04 convention-probe reader | A10 | 0.03 | 0.03 | actual | 162 frames, 59 s |
| 2026-11-22 | H04 campaign, batch 1 (234 clips) | A100-80GB | 1.60 | 4.00 | wasted | **stopped before writing any output**: the paths were exactly rank-deficient for the frozen statistic (amendment 2) |
| 2026-11-22 | H04 campaign, batch 2 (234 clips, amendment-2 paths), several failed attempts | A100-80GB | 2.30 | 5.75 | actual | four 59-clip calls died after ~15 min (sub-batch of 40 too large for the release's infer step; no clip written), plus 3 diagnostic calls (3, 12, 9 clips) that succeeded |
| 2026-11-22 | H04 campaign, batch 3 (234 clips, sub-batch 12) | A100-80GB | 2.25 | 5.63 | actual | four parallel 59-clip calls, ~1930-2440 s each, all 234 clips written |
| 2026-11-22 | H04 reader (234 clips x 81 frames = 18,954 frames) | A10 | 2.10 | 2.31 | actual | four shards, 1613-1928 s each, 18,954 rows |
| 2026-11-22 | H04 yaw control (22 clips warped on CPU, then read) | CPU 8 + A10 | 0.16 + 0.23 | 0.08 + 0.25 | actual | warp 575 s, reader 825 s |
| 2026-11-22 | H04 publish (468 objects to R2 with sha256 read-back) | CPU 4 | 0.4 | 0.02 | actual | `track-h3/clips/solarwm-main/` + manifest |
| 2026-11-22 | H04 yaw-control warp (22 clips, CPU) | CPU 8 | 0.10 | 0.04 | actual | pure numpy remap, per-clip commits |
| 2026-11-22 | H04 yaw-control reader (22 clips) | A10 | 0.23 | 0.25 | actual | 1,782 frames |
| 2026-11-22 | H04 14B download (183.4 GB, 81 files) | CPU 16 | 0.22 | 0.18 | actual | 799 s; volume `awmv-h3-14b-weights` |
| 2026-11-22 | H04 14B config-validation attempts (4 failures) | A100-80GB:2 | ~0.05 | 0.13 | actual | each died in 11-20 s on a config contract, not on the model |
| 2026-11-22 | H04 14B timing test (3 clips, 50 steps) | A100-80GB:2 | 0.67 | 1.67 | actual | measured seconds/clip below |

**Track H3/H4 total (actual): about US$25 of the US$60 ceiling** (the 14B attempt adds the
183 GB download, the config-contract failures, the two-rank timing attempt and the probe), of
which about US$8 is
wasted work: the aborted first campaign (rank-deficient paths) and the failed 40-clip
sub-batches.  The pre-campaign estimate posted in the H04 pre-registration was
US$6.0 for the campaign and reader; the actual is higher by the aborted batch and by the
Perspective Fields recompute that the failed pre-registered rule triggered.  The 14B release
was downloaded and driven through its own CLI but produced no clip; LTX-22B and H3-33B were
never downloaded or run.
