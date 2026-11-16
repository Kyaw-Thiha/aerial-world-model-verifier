# Cost ledger

One row per billed job set. Prices are Modal list prices unless a dashboard figure is
quoted: A10 USD 1.10/h, L40S USD 1.95/h, A100-80GB USD 2.50/h, H100 USD 3.95/h. Local
CPU/GPU work is free and not listed.

Track ceiling: **USD 40** (handoff 15); ask before exceeding.

| date (UTC) | job | GPU | hours | USD | notes |
| --- | --- | --- | --- | --- | --- |
| 2026-11-16 | `awmv-i-flightmare` probe (first attempt) | none (CPU image build) | 0.01 | 0.00 | image build failed: `libzmqpp-dev` absent on Ubuntu 22.04 |
| 2026-11-16 | `awmv-i-flightmare` probe (retry, zmqpp from source) | CPU container | 0.2 | 0.02 | build + Unity asset HEAD + xvfb GL probe |
| 2026-11-16 | `awmv-i-isaac` probe (image build + headless render attempt) | L40S | 0.3 | 0.60 | image build ~25 min (not GPU-billed) plus a short run; estimate |
| 2026-11-16 | `awmv-i-geocalib` scoring of spike clips (×2) | A10 | 0.1 | 0.11 | 91 frames per clip at ~5 fps |

Running total: **USD 0.73**. Ceiling remaining: USD 39.27.
