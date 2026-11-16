# GPU cost ledger

One row per job, with the price basis used. Modal list prices: A10G 1.10, L40S 1.95,
A100-80GB 2.50, H100 3.95 USD/h. Times are the container wall clock reported by Modal; USD is
the price basis times the hours (the dashboard remains the authority).

| date | track | job | GPU | hours | USD | notes |
| --- | --- | --- | --- | --- | --- | --- |
| 2026-11-16 | G | `awmv-g-tilt-reader` smoke runs (3×: gradient flow, step cost, fixed pass) | A10G | 0.1 | 0.11 | before and after the batch-size change |
| 2026-11-16 | G | TII folds, **run 1** (8 folds, batch 2) | A10G | 2.0 | 2.20 | **wasted**: `NameError` on the volume handle killed every fold at the write step, after training |
| 2026-11-16 | G | TII folds, run 2 (8 folds; train 740-780 s each, then evaluation of 270-1284 frames each) | A10G | 6.7 | 7.37 | the reported TII result |
| 2026-11-16 | G | UZH folds, run 2 (3 folds) | A10G | 0.5 | 0.55 | reported in the UZH block of G03 |
| | | **Track G total** | | **9.3** | **10.23** | pre-registered estimate 5 USD; ceiling 50 USD |

Notes
* The pre-registration estimated 5 USD. The run cost about twice that, for two reasons: the
  LM-in-the-loop step cost is ~4 s per batch of two at 320 px (the smoke run measured 1.25 s at
  batch 2 before the label fix, and the evaluation pass over 1,267-1,284 frames per fold is not
  cheap either), and run 1 had to be repeated after a code bug. Both are recorded rather than
  absorbed, per the spend rule. Nothing approached the 20 USD pre-post line or the 50 USD
  ceiling.
* G01 and G02 are CPU-only and cost nothing.
