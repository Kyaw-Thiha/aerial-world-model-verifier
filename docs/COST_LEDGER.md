# GPU cost ledger

One row per job, with the price basis used. Modal list prices: A10G 1.10, L40S 1.95,
A100-80GB 2.50, H100 3.95 USD/h. Times are the container's wall clock as reported by Modal;
USD is the price basis times the hours (the dashboard remains the authority).

| date | track | job | GPU | hours | USD | notes |
| --- | --- | --- | --- | --- | --- | --- |
| 2026-11-16 | G | `awmv-g-tilt-reader` smoke runs (2×, gradient flow + per-step cost) | A10G | 0.05 | 0.06 | before the real run |
| 2026-11-16 | G | `awmv-g-tilt-reader` folds, **run 1** (8 TII folds, batch 2) | A10G | 2.0 | 2.20 | **wasted**: a `NameError` on the volume handle killed every fold at write time after training. Recorded, not absorbed |
| 2026-11-16 | G | `awmv-g-tilt-reader` folds, run 2 (8 TII + 3 UZH folds) | A10G | 3.8 | 4.18 | the reported run |
| 2026-11-16 | G | no-finetune control fold (`epochs=0`, flight-01p) | A10G | 0.2 | 0.22 | harness control |
| | | **Track G total** | | **6.05** | **6.66** | pre-registered estimate 5 USD, ceiling 50 USD |

Notes
* The estimate in `configs/g03_tilt_reader_prereg.yaml` was 5 USD; the run came in at 6.66 USD
  because the LM-in-the-loop step cost (about 4 s per batch of two at 320 px) was higher than the
  smoke run's 1.25 s and because run 1 had to be repeated after a code bug. The overrun is
  recorded rather than absorbed, per the spend rule; nothing exceeded the 20 USD pre-post line.
* G01 and G02 are CPU-only and cost nothing.
