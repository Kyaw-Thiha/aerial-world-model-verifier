# H03 tables (generated from results/h03/analysis.json)

## Control errors (roll, degrees)

| control | n | median | 95% | IQR |
| --- | --- | --- | --- | --- |
| pooled | 60 | 0.92 | [0.55, 1.44] | 0.39-3.06 |
| simulated | 30 | 0.61 | [0.35, 1.17] | 0.19-1.70 |
| real | 30 | 1.30 | [0.70, 2.67] | 0.56-3.80 |
| real, native fisheye | 30 | 1.03 | [0.79, 2.04] | - |

## Independent estimators' own control errors

| estimator | n | median |
| --- | --- | --- |
| Perspective Fields (pooled) | 60 | 1.11 |
| Perspective Fields (simulated) | 30 | 0.92 |
| Perspective Fields (real) | 30 | 1.34 |
| classical (simulated) | 30 | 2.56 |
| classical (real) | 11 | 43.41 |

Bar = 1.37 deg; overall verdict **not_valid_on_generated_frames**.

## Reader vs independent estimates, per generated source

| source | n | reader vs PF | reader vs classical | classical vs PF | coverage PF/classical | status |
| --- | --- | --- | --- | --- | --- | --- |
| anwm | 30 | 1.23 | 7.24 | 6.19 | 1.00/0.90 | fail |
| lingbot_nf4 | 15 | 2.01 | 1.40 | 3.18 | 1.00/1.00 | fail |
| lingbot_bf16 | 15 | 0.74 | 9.92 | 10.56 | 1.00/1.00 | fail |
| solarwm | 30 | 2.23 | 2.33 | 4.45 | 1.00/1.00 | fail |
