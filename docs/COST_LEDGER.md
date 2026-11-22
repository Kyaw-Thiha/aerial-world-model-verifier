# Cost ledger (Track L)

One row per paid job: date, job, GPU, hours, USD. Modal list prices used unless the dashboard
says otherwise: A10 USD 1.10/h, L40S 1.95/h, A100-80GB 2.50/h, H100 3.95/h. The GPU-hour is
taken from the container's own wall clock where the job reports it, otherwise from the billed
duration in `modal app list`. Track L's ceiling is **USD 50**.

Earlier tracks keep their ledgers on their own branches; this file starts at Track L because
`main` (the base of this branch) did not carry one.

## Track L — L01, scene-stratified reader error

| date | job | GPU | hours | USD | notes |
| --- | --- | --- | --- | --- | --- |
| 2026-11-22 | `awmv-l-reader` feasible-dev re-score with gate and confidence fields (80 clips, stride 2) | A10 | 1.2 (est.) | 1.32 (est.) | 19,081 frames at the 6.2 fps I02 measured; includes startup and image build |

Estimated before running: **USD 1.4** (1.2 GPU-h plus build). Below the USD 20 pre-post line.

## Track L — L02, reader fine-tune

| date | job | GPU | hours | USD | notes |
| --- | --- | --- | --- | --- | --- |
| — | arms (a) real-only, (b) sim-only, (c) mixed + domain randomisation | A10 | — | — | estimate fixed in `configs/l02_reader_prereg.yaml` after the timing smoke |
| — | final models and one held-out evaluation | A10 | — | — | |

## Running total

| item | USD |
| --- | --- |
| booked | 0.00 |
| committed (L01 scoring) | 1.32 |
| remaining against the 50.00 ceiling | 48.68 |
