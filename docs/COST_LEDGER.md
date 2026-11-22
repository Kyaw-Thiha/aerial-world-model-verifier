# Cost ledger

One line per job set. Modal list prices at the time of writing: A10 1.10, L40S 1.95,
A100-80GB 2.50, H100 3.95 USD/h. Update this file in the same commit as the job that spends.

Track I's ledger entries live in its own branch (`track-i-simset`, `docs/COST_LEDGER.md`)
until it merges; this file records Track G3 only.

## Track G3 — segment detection on the simulated set (ceiling USD 8.00)

| date | job | GPU | hours | USD | running total | notes |
| --- | --- | --- | --- | --- | --- | --- |
| 2026-11-22 | setup, code port, pre-registration | — | 0 | 0.00 | 0.00 | CPU only |
| 2026-11-22 | reader engineering probes, dev reader (83 clips, incl. the 4-container window), yaw render (10 clips), diagnostics | L40S/A10/A100 | — | **2.71** | **2.71** | **billing export** (`modal billing report --for today`): score apps 2.013, yaw 0.698, diag 0.003. Not an estimate. |
| 2026-11-22 | remaining dev (87 clips, 45,683 frames) + 2 yaw renders + yaw reader, one container at a time | L40S | ~0.95 | ~1.5-1.9 | ~4.5 | projected; queueing delays wall time, not spend |
| 2026-11-22 | held-out pass (100 clips, 49,859 frames), later, once | L40S | ~0.9 | ~2.3 | ~6.8 | reserved |

**Correction (2026-11-22, after the billing export was found).** An earlier version of this file
estimated the first batch at 9.4 USD "from app runtimes" and called it an overrun of the 8.00
ceiling. That estimate was wrong: the containers were mostly **waiting for L40S capacity**, which
is not billed, and the measured per-clip compute stayed at 15-17 fps throughout. The actual spend
for everything up to and including the 4-container window is **2.71 USD**, so the ceiling was
never breached. The user raised it to 14.00 on the strength of the wrong estimate; that is
harmless and is recorded here rather than silently reverted. All figures above are from the
billing export for 2026-11-22.
