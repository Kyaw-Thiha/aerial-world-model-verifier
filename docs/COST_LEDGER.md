# GPU cost ledger

One row per job, with the price basis used. Modal list prices: A10G 1.10, L40S 1.95,
A100-80GB 2.50, H100 3.95 USD/h. Times are the container wall clock reported by Modal; USD is
the price basis times the hours (the dashboard remains the authority).

| date | track | job | GPU | hours | USD | notes |
| --- | --- | --- | --- | --- | --- | --- |
| 2026-11-21 | G2 | none | — | 0 | 0.00 | G04, G05 and the spike pilot are CPU-only and read stored artifacts |

Notes
* Track G2's ceiling is 10 USD (handoff 18). Nothing was spent: no Modal app was started, no
  volume created, and no R2 object written.
* Round 1 of this track (branches `track-g-segments` and `track-g-segments2`) spent 10.23 USD on
  the tilt-reader fine-tune; its ledger entry lives on that branch, not here, because this
  branch is off `main`.
