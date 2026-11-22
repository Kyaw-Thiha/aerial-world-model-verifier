# Cost ledger

One row per job set: date, job, GPU, hours, USD. Modal list prices: A10 1.10, L40S 1.95,
A100-80GB 2.50, H100 3.95 USD/h. CPU jobs are recorded at $0.05/core-hour. `estimate`
rows are posted before a job set expected to exceed US$20; `actual` rows are measured from
the run's own manifest.

| date | job | GPU | hours | USD | status | notes |
| --- | --- | --- | --- | --- | --- | --- |
| 2026-11-21 | H02 LingBot full-precision weights download (85 GB: fast 74 + T5/VAE) | CPU 16 | ~0.2 | ~0.16 | done | `awmv-h2-lingbot-full`, volume `awmv-h2-weights` |
| 2026-11-21 | H02 LingBot full-precision roll probe (5 clips, 49->45 frames) | A100-80GB | 0.15 (5 x ~60 s + 2 min load) | 0.38 | done | 55.2 GB peak, bf16, 51.7-62.9 s/clip |
| 2026-11-21 | H02 SolarWM 5B weights download (~61 GB) + image | CPU 16 | ~0.4 | ~0.32 | running/committing | `awmv-h2-solarwm`, volume `awmv-h2-weights`; token account accepted 2026-11-21 |
| 2026-11-21 | H02 SolarWM 5B grid (75 clips, 81 frames, 4 steps) | A100-80GB | ~2.5 (est: 2 min/clip) | ~6.25 (est) | not started | estimate to be replaced by the first measured clip |
