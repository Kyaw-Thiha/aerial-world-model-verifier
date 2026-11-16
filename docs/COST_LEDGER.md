# Cost ledger

One row per job set: date, job, GPU, hours, USD. GPU list prices are Modal's posted rates
(A10/A10G 1.10, L40S 1.95, A100-80GB 2.50, H100 3.95 USD/h); CPU jobs are recorded at
$0.05/core-hour and marked CPU. `estimate` rows are posted before a job set that is expected
to exceed US$20; `actual` rows are measured from the run's own manifest.

| date | job | GPU | hours | USD | status | notes |
| --- | --- | --- | --- | --- | --- | --- |
| 2026-11-16 | H01 ANWM seed generation (80 clips, seeds 1-4, 20 designed paths) | A10 | 4.9 (218.8 s/clip measured, 80 clips) | 5.35 | done | `awmv-h-anwm-b` (specs[40:], 8872 s) and `awmv-h-anwm-c` (specs[:40], 8776 s) |
| 2026-11-16 | H01 LingBot download (33 GB NF4 weights) | CPU 16 cores | ~0.2 | ~0.16 | done | `awmv-h-lingbot` download function |
| 2026-11-16 | H01 LingBot probe (3 clips) + model load | A100-80GB | 0.9 (3 x 652 s + 0.33 load) | 2.25 | done | level, +20 ramp, -20 ramp; 21.1 GB peak |
| 2026-11-16 | H01 LingBot strong probes (2 clips: fast20, 60) | A100-80GB | 0.6 (2 x 648 s + 0.22 load) | 1.45 | done | diagnostic only; both show no usable roll response |
| 2026-11-16 | H01 LingBot grid (60 clips at 20 steps) | A100-80GB | 0 | 0 | **not run** | conditional on a roll response; the probe failed, so the spend was avoided |
| 2026-11-16 | H01 Wan 2.2 TI2V-5B: 5 clips | A10 | 0.2 (5 x 135 s + load) | 0.21 | done | 49 frames, 40 steps, CPU offload, 15.2 GB |
| 2026-11-16 | H01 depicted-tilt extraction (ANWM seeds, LingBot, Wan) | A10/A10G | ~0.3 | 0.33 | done | 224 px GeoCalib; ANWM 80 clips, LingBot 5, Wan 5, plus partial passes |

**Track H pilot total (estimate): about US$13 of the US$75 ceiling.** ANWM seed generation
about US$5.4 (80 clips plus the early discarded runs), LingBot-World about US$6.5 (five probe
clips plus four model loads, including the stopped/aborted attempts), Wan 2.2 about US$0.6,
reader extractions about US$0.2. No scale-up job was run, so the conditional 60-clip LingBot
grid (US$14) was not spent.
