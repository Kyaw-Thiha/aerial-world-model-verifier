# Cost ledger

One line per job set. Modal list prices at the time of writing: A10 1.10, L40S 1.95,
A100-80GB 2.50, H100 3.95 USD/h. Update this file in the same commit as the job that spends.

Track I's ledger entries live in its own branch (`track-i-simset`, `docs/COST_LEDGER.md`)
until it merges; this file records Track G3 only.

## Track G3 — segment detection on the simulated set (ceiling USD 8.00)

| date | job | GPU | hours | USD | running total | notes |
| --- | --- | --- | --- | --- | --- | --- |
| 2026-11-22 | setup, code port, pre-registration | — | 0 | 0.00 | 0.00 | CPU only |
| 2026-11-22 | reader engineering probes (6 single-clip runs, 3 GPUs) | A10/A100/L40S | ~0.4 | 0.40 | 0.40 | estimated from app runtimes |
| 2026-11-22 | dev reader, 4 concurrent L40S shards, 83/170 clips | L40S | ~4.8 | 9.36 | 9.76 | **overrun**: 4 concurrent containers ran 3-6x slower than one (a 343-frame probe took 5m31s against ~50s alone); stopped at the ceiling. Estimated from app runtimes (4 containers x ~72 min x 1.95); no CLI billing export was available |
| 2026-11-22 | yaw slice render, 10/12 clips (same 4-container window) | L40S | (in the 4.8 above) | 0.00 | 9.76 | included in the concurrent window |
| 2026-11-22 | ceiling raised to 14.00 by the user; remaining dev (87 clips, 45,683 frames) + 2 yaw renders + yaw reader, sequential single container | L40S | ~0.95 | ~1.85 | ~11.61 | one container at a time; no concurrency |
| 2026-11-22 | held-out pass (100 clips, 49,859 frames), later, once | L40S | ~0.9 | ~2.3 | ~13.9 | reserved |

All USD figures are **estimated from app runtimes at the Modal list prices** (L40S 1.95/h), not
read from a billing export. The 2026-11-22 overrun is recorded in `docs/decisions.md`.
