# GPU cost ledger

One row per job, with the price basis used. Modal list prices: A10G 1.10, L40S 1.95,
A100-80GB 2.50, H100 3.95 USD/h. Times are the container wall clock reported by Modal; USD is
the price basis times the hours (the dashboard remains the authority).

| date | track | job | GPU | hours | USD | notes |
| --- | --- | --- | --- | --- | --- | --- |
| 2026-11-21 | J | none (Part 0, J01 + J02) | — | 0 | 0.00 | anatomy and noise budget are CPU-only and read stored reader outputs, mocap and poses |

## Track J

* Ceiling for the track: 30 USD total (handoff 19), with a cost estimate required before any job
  set expected to exceed 20 USD.
* Spent so far: **0.00 USD**. No Modal app was started, no volume created, no R2 object written.
* The only GPU branch the rules could select is 1c (learned reader confidence). Rule 3 fires on
  its literal reading; the estimate and the decision are recorded in `reports/J01` and
  `docs/decisions.md` [J] before any job.

## Other tracks

Round 1 (F/G/H) and round 2 (G2, I) kept their ledgers on their own branches
(`track-g-segments2`, `track-g2-segments`, `track-i-simset`), because those branches are not
merged into `main`. This file starts on `main` with the J entry.
