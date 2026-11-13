# Track D gain-test evaluation

run `d01` · git `613a102` · testable real clips 57

## Candidate rules (pooled, held-out at alpha 0.05)

| rule | FRR | removal | mirror | x0.5 | x1.5 | yaw | n real |
| --- | --- | --- | --- | --- | --- | --- | --- |
| R1_scalar_g1 | 8.8 | 100.0 | 96.5 | 98.2 | 96.5 | 22.8 | 57 |
| R2_axis_gains | 10.5 | 100.0 | 100.0 | 100.0 | 96.5 | 24.6 | 57 |
| R3_scalar_or_axis | 12.3 | 100.0 | 100.0 | 100.0 | 100.0 | 31.6 | 57 |

Pooled selection: **R1_scalar_g1**

## Selected rule vs baseline

| quantity | gain (pooled rule) | gain (nested selection) | baseline |
| --- | --- | --- | --- |
| frr | 8.8 | 10.5 | 3.4 |
| det_tilt_removal | 100.0 | 100.0 | 67.2 |
| det_mirror_roll | 96.5 | 100.0 | 94.8 |
| det_scale_0.5 | 98.2 | 100.0 | 17.2 |
| det_scale_1.5 | 96.5 | 96.5 | 24.1 |
| det_yaw_control | 22.8 | 24.6 | 8.6 |

## Matched false-reject

| family | gain @ own | baseline @ gain FRR | gain @ baseline FRR | baseline @ own |
| --- | --- | --- | --- | --- |
| tilt_removal | 100.0 | 82.8-94.8 | 100.0-100.0 | 67.2 |
| mirror_roll | 96.5 | 96.6-98.3 | 94.7-94.7 | 94.8 |
| scale_0.5 | 98.2 | 20.7-56.9 | 98.2-98.2 | 17.2 |
| scale_1.5 | 96.5 | 51.7-94.8 | 94.7-94.7 | 24.1 |

## Abstention

overall 43.0 % (baseline 42.0 %)

- flight-01a-ellipse: 2/5 testable
- flight-04a-ellipse: 1/7 testable
- flight-07a-lemniscate: 0/7 testable
- flight-13a-trackRATM: 0/11 testable
- flight-01p-ellipse: 26/26 testable
- flight-07p-lemniscate: 24/26 testable
- flight-13p-trackRATM: 3/11 testable
- flight-16p-trackRATM: 1/7 testable

Abstention reasons: {'signal_frames': 18, 'signal_fraction': 18, 'excitation': 14, 'condition_number': 11, 'gain_ci_half_width': 17, 'confidence_gate': 28}

## Fitted drag (1/s, primary, real)

- all: median 0.139 (IQR 0.106-0.178, n=57)
- trackratm: median 0.070 (IQR 0.007-0.144, n=4)
- non_trackratm: median 0.139 (IQR 0.106-0.178, n=53)
- piloted: median 0.139 (IQR 0.106-0.177, n=54)
- autonomous: median 0.174 (IQR 0.130-0.187, n=3)
