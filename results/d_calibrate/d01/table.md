# Monotone calibration test

Track flights: 3 testable under v2; **not validated**.

| method / rule | FRR | removal | mirror | x0.5 | x1.5 | yaw |
| --- | --- | --- | --- | --- | --- | --- |
| uncalibrated R1_scalar_g1 | 0.105 | 1.000 | 0.982 | 1.000 | 0.982 | 0.193 |
| uncalibrated R2_axis_gains | 0.053 | 1.000 | 1.000 | 1.000 | 0.965 | 0.281 |
| single curve R1_scalar_g1 | 0.061 | 0.982 | 0.985 | 0.855 | 0.978 | 0.156 |
| single curve R2_axis_gains | 0.057 | 1.000 | 0.985 | 0.590 | 0.601 | 0.127 |
| per_axis curve R1_scalar_g1 | 0.068 | 1.000 | 0.978 | 0.603 | 0.947 | 0.191 |
| per_axis curve R2_axis_gains | 0.064 | 1.000 | 1.000 | 0.954 | 0.781 | 0.173 |
