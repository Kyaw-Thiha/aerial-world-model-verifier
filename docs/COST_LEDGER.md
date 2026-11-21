# Cost ledger

One row per billed job set. Prices are Modal list prices unless a dashboard figure is quoted:
A10 USD 1.10/h, L40S USD 1.95/h, A100-80GB USD 2.50/h, H100 USD 3.95/h. Local CPU/GPU work is
free and not listed. The hours are wall-clock estimates of container time from the client logs
and the per-clip timings the jobs report; `modal app list` reports state, not dollars, and the
dashboard was not consulted. If the exact figure matters, read it there and replace these rows.

Track ceiling: **USD 40** (handoff 15); ask before exceeding.

## Track I — I01, the simulator spike (2026-11-16)

| job | GPU | hours | USD | notes |
| --- | --- | --- | --- | --- |
| `awmv-i-flightmare` probe, 1st attempt | CPU image build | 0.02 | 0.00 | `libzmqpp-dev` absent on 22.04 |
| `awmv-i-flightmare` probe, 2nd attempt | CPU container | 0.4 | 0.04 | zmqpp + flightlib build; stopped during the Unity download |
| `awmv-i-isaac` probe, image build + start | L40S | 0.05 | 0.10 | app reached "Startup Complete" in 24 s |
| `awmv-i-isaac` probe, API iterations | L40S | 0.6 | 1.17 | three driver-side failures, one run that did not return |
| `awmv-i-blender` throughput and EEVEE clip (7 runs) | L40S | 0.6 | 1.17 | 1 GB image; 91-frame EEVEE clip at 0.145 s/frame |
| `awmv-i-geocalib` spike scoring, 3 clips | A10 | 0.13 | 0.14 | 91 frames per clip |
| **subtotal** | | | **2.62** | |

## Track I — I02, the labelled set (2026-11-21)

| job | GPU | hours | USD | notes |
| --- | --- | --- | --- | --- |
| `awmv-i-blender` scene-family previews (3 runs) | L40S | 0.1 | 0.20 | five families, one static frame each |
| `awmv-i-dataset` smoke tests (4 failed calls) + 2 clips | L40S | 0.2 | 0.39 | image/tooling debugging |
| `awmv-i-dataset` first 12-clip batch | L40S | 0.4 | 0.78 | 24 min wall; 0.14-0.36 s/frame |
| `awmv-i-dataset` family x mount grid (11 clips, one cancelled call + resume) | L40S | 0.35 | 0.68 | forest clips 2.3x the cost per frame |
| `awmv-i-score` first dev pass, 6 clips | A10 | 0.25 | 0.28 | 3,058 frames at 2.5-7.7 fps |
| `awmv-i-score` grid pass, 11 clips at stride 2 | A10 | 0.2 | 0.22 | 2,700 frames in 468 s |
| **subtotal** | | | **2.55** | |

## Booked and committed

| item | GPU | hours | USD |
| --- | --- | --- | --- |
| booked so far | | | **5.17** |
| full-set render, 149,450 frames in slices | L40S | ~10 | ~20.0 |
| full-set reader scoring at stride 2 | A10 | ~5 | ~5.5 |
| **planned total** | | | **~30.7** |

Remaining headroom against the ceiling: about USD 9, which is the allowance for reruns.

The per-frame cost is what sets the plan and it is scene-dependent: open scenes render at
0.14-0.21 s/frame and the dense forest at 0.35 s/frame on the same L40S. The 149,450-frame
total at a 0.23 s/frame average is the 10 h above.
