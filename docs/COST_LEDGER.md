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
| 2026-11-22 | `awmv-l-reader` feasible-dev re-score with gate and confidence fields (80 clips, stride 1) | A10 | 3.7 (actual, 4 shards x ~55 min) | 4.1 | 37,401 frames; two attempts, the first stopped when per-clip volume commits serialised; the resumed pass skips stored clips |

Estimated before running: **USD 3.0** (2.7 GPU-h); the actual wall clock is longer because each
clip decodes and commits separately, and the ledger is updated with the billed hours when the
pass finishes. Below the USD 20 pre-post line. The estimate
first written here (1.2 h at stride 2) was revised to stride 1 before the run, on the grounds
that the anatomy split wants the full 30 Hz; recorded rather than overwritten.

## Track L — L02, reader fine-tune

| date | job | GPU | hours | USD | notes |
| --- | --- | --- | --- | --- | --- |
| 2026-11-22 | smoke: 32 frames, batch 1, mixed route | A10 | 0.01 | 0.01 | 0.67 s/frame measured |
| 2026-11-22 | first sim fold attempt (stopped: volume-read JPEGs dominated) | A10G | 1.0 | 1.1 | measured about 2 s/frame once the LM was in the loop on forest frames; no fold record |
| — | arms (a) 6 folds, (b) 3 folds, (c) 9 folds | A10G | 9.0 (est.) | 9.9 | 18 models x 600 frames x 2 epochs at about 2 s/frame plus eval |
| — | 3 final models + all-real evaluation (5,007 frames each) | A10G | 2.4 (est.) | 2.6 | final models also write the checkpoint the edit pass reads |
| — | downstream edit pass, 4 readers x 72 gate-kept clips x 5 conditions | A10G | 4.0 (est.) | 4.4 | about 17k frames per reader at 4-5 fps |

## Running total

| item | USD |
| --- | --- |
| booked | 4.11 |
| committed (L01 scoring, all 80 clips, actual) | 4.10 |
| committed (L02, estimate above) | 18.00 |
| remaining against the 50.00 ceiling | 28.99 |
