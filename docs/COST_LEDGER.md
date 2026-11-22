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

## Track I — I02, the full-set render (2026-11-21, in progress)

| job | GPU | hours | USD | notes |
| --- | --- | --- | --- | --- |
| `awmv-i-dataset` render batch 1 (150 clips, detached) | L40S | 4.2 | 8.19 | 0.15-0.36 s/frame, ~2 min per clip with encode and truth |
| `awmv-i-dataset` render batch 2 (145 clips, detached, in progress) | L40S | ~4.0 | ~7.8 | |
| `awmv-i-score` dev feasible grid, 80 clips at stride 2 | A10 | 1.1 | 1.21 | 19,081 frames |
| **subtotal so far** | | | **~7.2** booked, ~7.8 committed | |

## Booked and committed

| item | GPU | hours | USD |
| --- | --- | --- | --- |
| booked so far | | | **12.4** |
| render batch 2 (in progress) | L40S | ~4 | ~7.8 |
| remaining clips after batch 2 (about 0) | L40S | ~0 | ~0 |
| dev-pass scoring of the defect clips, stride 2 | A10 | ~4.5 | ~5.0 |
| **planned total** | | | **~25.2** |

Remaining headroom against the ceiling: about USD 15, which is the allowance for reruns.

The per-frame cost is what sets the plan and it is scene-dependent: open scenes render at
0.14-0.21 s/frame and the dense forest at 0.35 s/frame on the same L40S. The 149,450-frame
total at a 0.23 s/frame average is the ~10 h above. Actual per-clip wall time averages 2 min
(encode, truth write and volume commit included), which is what the estimates use.
