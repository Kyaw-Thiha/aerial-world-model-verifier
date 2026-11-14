# Track D gain-test evaluation

run `d02` · git `1df4084` · testable real clips 57

## Candidate rules (pooled, held-out at alpha 0.05)

| rule | FRR | removal | mirror | x0.5 | x1.5 | yaw | n real |
| --- | --- | --- | --- | --- | --- | --- | --- |
| R1_scalar_g1 | 10.5 | 100.0 | 98.2 | 100.0 | 98.2 | 19.3 | 57 |
| R2_axis_gains | 5.3 | 100.0 | 100.0 | 100.0 | 96.5 | 28.1 | 57 |
| R3_scalar_or_axis | 12.3 | 100.0 | 100.0 | 100.0 | 100.0 | 31.6 | 57 |

Pooled selection: **R2_axis_gains**

## Selected rule vs baseline

| quantity | gain (pooled rule) | gain (nested selection) | baseline |
| --- | --- | --- | --- |
| frr | 5.3 | 5.3 | 3.4 |
| det_tilt_removal | 100.0 | 100.0 | 67.2 |
| det_mirror_roll | 100.0 | 100.0 | 94.8 |
| det_scale_0.5 | 100.0 | 100.0 | 17.2 |
| det_scale_1.5 | 96.5 | 96.5 | 24.1 |
| det_yaw_control | 28.1 | 28.1 | 8.6 |

## Matched false-reject

| family | gain @ own | baseline @ gain FRR | gain @ baseline FRR | baseline @ own |
| --- | --- | --- | --- | --- |
| tilt_removal | 100.0 | 67.2-67.2 | 98.2-100.0 | 67.2 |
| mirror_roll | 100.0 | 94.8-94.8 | 98.2-100.0 | 94.8 |
| scale_0.5 | 100.0 | 17.2-17.2 | 7.0-100.0 | 17.2 |
| scale_1.5 | 96.5 | 24.1-24.1 | 7.0-96.5 | 24.1 |

## Abstention

overall 43.0 % (baseline 42.0 %)

- flight-01a-ellipse: 2/5 testable
- flight-04a-ellipse: 2/7 testable
- flight-07a-lemniscate: 0/7 testable
- flight-13a-trackRATM: 0/11 testable
- flight-01p-ellipse: 26/26 testable
- flight-07p-lemniscate: 24/26 testable
- flight-13p-trackRATM: 2/11 testable
- flight-16p-trackRATM: 1/7 testable

Abstention reasons: {'signal_frames': 18, 'signal_fraction': 17, 'excitation': 14, 'condition_number': 11, 'gain_ci_half_width': 17, 'confidence_gate': 28}

## Fitted drag (1/s, primary, real)

- all: median 0.158 (IQR 0.115-0.195, n=57)
- trackratm: median 0.058 (IQR 0.033-0.064, n=3)
- non_trackratm: median 0.159 (IQR 0.118-0.199, n=54)
- piloted: median 0.156 (IQR 0.115-0.191, n=53)
- autonomous: median 0.193 (IQR 0.140-0.252, n=4)
