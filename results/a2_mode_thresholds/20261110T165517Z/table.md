## TII thresholds by mode

| mode | k_d | pooled | per-fold min / median / max | held-out acceptance |
| --- | --- | --- | --- | --- |
| both kd0.0 | 12.95 | — | — |
| both kd0.15 | 8.40 | — | — |
| both kd0.295 | 6.46 | — | — |
| lag_only kd0.0 | 14.73 | 13.50 / 14.73 / 16.46 | 0.966 |
| lag_only kd0.15 | 10.60 | 10.60 / 11.26 / 14.39 | 0.966 |
| lag_only kd0.295 | 8.96 | 8.96 / 9.49 / 12.07 | 0.966 |

## Verdict vs threshold, on the primary-mode (drag-free lag-only) residuals

| threshold | Regime A infeasible/testable | C 10 deg | C 20 deg |
| --- | --- | --- | --- |
| 6.46 | 10/10 (median 33.8) | 12/12 (median 9.0) | 10/10 (median 19.1) |
| 8.40 | 10/10 (median 33.8) | 8/12 (median 9.0) | 10/10 (median 19.1) |
| 12.95 | 10/10 (median 33.8) | 0/12 (median 9.0) | 10/10 (median 19.1) |
| 14.73 (primary) | 10/10 (median 33.8) | 0/12 (median 9.0) | 10/10 (median 19.1) |

## Mode-consistent verdicts (each mode at its own threshold)

| mode | threshold | Regime A | C 10 deg | C 20 deg |
| --- | --- | --- | --- | --- |
| lag_only kd0.0 @ primary | 14.73 | 10/10 | 0/12 | 10/10 |
| lag_only kd0.295 @ own | 8.96 | 11/11 | 12/12 | 10/10 |
| parity kd0.295 @ own | 6.46 | 11/11 | 0/12 | 10/10 |
| parity kd0.0 @ own | 12.95 | 10/10 | 0/12 | 4/10 |

## Detection floor (required tilt, deg)

| mode | verdict threshold | required tilt at the floor |
| --- | --- | --- |
| lag_only kd0.0 (primary) | 14.73 | 15.52 |
| lag_only kd0.15 | 10.60 | 11.39 |
| lag_only kd0.295 | 8.96 | 9.75 |
| parity kd0.295 | 6.46 | 7.25 |
| parity kd0.15 | 8.40 | 9.19 |
| parity kd0.0 | 12.95 | 13.74 |

Null (cruise, lag-only, k_d 0.0): n=14, median 0.80, p95 2.24, p99 2.29; required tilt by cell {5: 5.02212340944601, 10: 8.601259763580813, 20: 21.704156080462305}.
