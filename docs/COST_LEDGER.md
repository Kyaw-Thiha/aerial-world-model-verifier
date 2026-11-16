# Cost ledger

One row per job set: date, job, GPU, hours, USD. GPU list prices are Modal's posted rates
(A10/A10G 1.10, L40S 1.95, A100-80GB 2.50, H100 3.95 USD/h); CPU jobs are recorded at
$0.05/core-hour and marked CPU. `estimate` rows are posted before a job set that is expected
to exceed US$20; `actual` rows are measured from the run's own manifest.

| date | job | GPU | hours | USD | status | notes |
| --- | --- | --- | --- | --- | --- | --- |
| 2026-11-16 | H01 ANWM seed generation (80 clips, seeds 1-4, 20 designed paths) | A10/A10G | 4.8 (216 s/clip measured) | 5.28 | running | two containers (`awmv-h-anwm-b` specs[40:], `awmv-h-anwm-c` specs[:40]); per-clip seconds in each manifest |
| 2026-11-16 | H01 LingBot download (33 GB NF4 weights) | CPU 16 cores | ~0.2 | ~0.16 | done | `awmv-h-lingbot` download function |
| 2026-11-16 | H01 LingBot probe (3 clips) + model loads | A100-80GB | 1.2 so far (loads dominate) | 3.00 | running | NF4 CPU load of 30 GB from the volume takes 30-40 min per container |
| 2026-11-16 | H01 LingBot grid (75 clips) | A100-80GB | TBD | TBD | not started | estimate posted before the job set |
| 2026-11-16 | H01 Wan 2.2 TI2V-5B: 5 clips | A10G | ~0.3 | 0.33 | running | 49 frames, 40 steps, CPU offload |
| 2026-11-16 | H01 depicted-tilt extraction (ANWM seeds, LingBot, Wan) | A10G | ~0.3 | 0.33 | not started | 224 px GeoCalib, ~3 min per 82 clips measured previously |
