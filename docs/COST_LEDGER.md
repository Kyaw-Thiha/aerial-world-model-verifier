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
| 2026-11-22 | H04 campaign, batch 2 (234 clips, amendment-2 paths) | A100-80GB | 2.00 | 5.00 | estimate | 4 parallel chunks x ~59 clips at ~28 s/clip including load |
| 2026-11-22 | H04 reader (234 clips x 81 frames = 18,954 frames) | A10 | 1.10 | 1.21 | estimate | ~5 frames/s, 4 parallel chunks |
| 2026-11-22 | H04 yaw-control warp (22 clips, CPU) | CPU 8 | 0.10 | 0.04 | estimate | pure numpy remap |

**Track H3/H4 total (estimate): about US$11.5 of the US$60 ceiling**, of which US$4.0 is the
aborted first campaign.  The pre-campaign estimate posted in the H04 pre-registration was
US$6.0 for the campaign and reader; the actual is higher by the aborted batch and by the
Perspective Fields recompute that the failed pre-registered rule triggered.  No 14B, LTX-22B
or H3-33B checkpoint was run.
