# Track D OLS vs Deming

Track flights: 4 testable clips; **not validated**.

| method | band | FRR | removal | mirror | x0.5 | x1.5 | yaw control |
| --- | --- | --- | --- | --- | --- | --- | --- |
| v1_ols | 0.528-0.985 | 0.070 | 1.000 | 0.947 | 0.982 | 0.930 | 0.193 |
| v1_deming | 0.486-0.907 | 0.070 | 1.000 | 0.965 | 0.737 | 0.930 | 0.246 |
| v2_ols | 0.543-0.936 | 0.105 | 1.000 | 0.982 | 1.000 | 0.982 | 0.193 |
| v2_deming | 0.519-0.872 | 0.070 | 1.000 | 0.982 | 0.842 | 0.965 | 0.316 |

Real-clip gain medians (IQR):

- v1_ols: piloted 0.753 (0.713-0.814, n=50); autonomous 0.627 (0.591-0.764, n=4); track 0.518 (0.508-0.548, n=3)
- v1_deming: piloted 0.730 (0.704-0.785, n=50); autonomous 0.633 (0.430-0.699, n=4); track 0.524 (0.504-0.531, n=3)
- v2_ols: piloted 0.770 (0.739-0.823, n=50); autonomous 0.645 (0.627-0.659, n=4); track 0.541 (0.525-0.543, n=3)
- v2_deming: piloted 0.762 (0.725-0.802, n=50); autonomous 0.623 (0.434-0.671, n=4); track 0.534 (0.521-0.537, n=3)
