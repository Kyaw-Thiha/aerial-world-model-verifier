# Track D coherence tier

run `d01` · n real 57 · x0.5/x1.5 expected at chance

| family | n | AUROC corr | corr det | AUROC sign | sign det | yaw det (corr) |
| --- | --- | --- | --- | --- | --- | --- |
| tilt_removal | 57 | 0.999 | 1.000 | 0.998 | 1.000 | 0.105 |
| scale_0.5 | 57 | 0.706 | 0.175 | 0.568 | 0.070 | 0.105 |
| scale_1.5 | 57 | 0.622 | 0.246 | 0.464 | 0.088 | 0.105 |
| mirror_roll | 57 | 1.000 | 1.000 | 0.999 | 1.000 | 0.105 |
| yaw_control | 57 | 0.600 | 0.105 | 0.522 | 0.070 | 0.105 |

Track flights: 4 testable clips; not validated.

