# GPU cost ledger

One row per job, with the price basis used. Modal list prices: A10G 1.10, L40S 1.95,
A100-80GB 2.50, H100 3.95 USD/h. Times are the container's wall clock as reported by Modal;
USD is the price basis times the hours (the dashboard remains the authority).

| date | track | job | GPU | hours | USD | notes |
| --- | --- | --- | --- | --- | --- | --- |
| 2026-11-16 | G | `awmv-g-tilt-reader` smoke runs (2×) | A10G | 0.05 | 0.06 | gradient-flow and per-step-cost probe before the real run |
| 2026-11-16 | G | `awmv-g-tilt-reader` fold runs, estimated | A10G | 2.0 | 2.20 | estimate posted before the run: 11 folds × ~10 min; actual filled in below when the run completes |
