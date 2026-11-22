# Cost ledger

One row per job set: date, job, GPU, hours, USD. Modal list prices: A10 1.10, L40S 1.95,
A100-80GB 2.50, H100 3.95 USD/h. CPU jobs are recorded at $0.05/core-hour. `estimate`
rows are posted before a job set expected to exceed US$20; `actual` rows are measured from
the run's own manifest.

| date | job | GPU | hours | USD | status | notes |
| --- | --- | --- | --- | --- | --- | --- |
| 2026-11-21 | H02 LingBot full-precision weights download (85 GB: fast 74 + T5/VAE) | CPU 16 | ~0.2 | ~0.16 | done | `awmv-h2-lingbot-full`, volume `awmv-h2-weights` |
| 2026-11-21 | H02 LingBot full-precision roll probe (5 clips, 49->45 frames) | A100-80GB | 0.15 (5 x ~60 s + 2 min load) | 0.38 | done | 55.2 GB peak, bf16, 51.7-62.9 s/clip |
| 2026-11-21 | H02 SolarWM 5B weights download (~61 GB) + image | CPU 16 | ~0.4 | ~0.32 | done | `awmv-h2-solarwm`, volume `awmv-h2-weights`; token account accepted 2026-11-21 |
| 2026-11-21 | H02 SolarWM probes (5 clips) + 3 failed setup calls | A100-80GB | 0.3 | 0.75 | done | 264 s for 5 clips including load; failures were publication/decode bugs, not model time |
| 2026-11-21 | H02 SolarWM grid (75 clips, 81 frames, 4 steps) | A100-80GB | 0.58 (2084 s incl. load/decode) | 1.45 | done | all 75 clips, 81 frames each |
| 2026-11-21 | H02 SolarWM seed batch (30 clips) | A100-80GB | 0.22 (782 s incl. load/decode) | 0.54 | done | 10 paths x 3 seeds |
| 2026-11-21 | H02 SolarWM seed reader (30 clips, every 2nd frame) | A100-80GB | 0.18 (649 s) | 0.45 | done | 1,230 frames |
| 2026-11-21 | H02 SolarWM grid reader (75 clips, every 2nd frame) | A100-80GB | 0.29 (1046 s) | 0.73 | done | 3,075 frames, GeoCalib |

**Track H2 pilot total (estimate): about US$5 of the US$90 ceiling.** SolarWM download CPU and
grid/seed generation/readers about US$4.0, LingBot download and full-precision probes about
US$0.6, R2/uploads negligible. No 14B or larger SolarWM route was run.
