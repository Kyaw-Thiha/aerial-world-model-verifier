# Label-free bandwidth window

c = 0.5, window = c / f95, floor 8 samples

## TII

testable real: 57/100; abstention 43.0%

| flight | n | median window s | min | max | samples |
| --- | --- | --- | --- | --- | --- |
| flight-01a-ellipse | 5 | 0.103 | 0.056 | 0.490 | 28 |
| flight-01p-ellipse | 26 | 0.178 | 0.098 | 0.280 | 49 |
| flight-04a-ellipse | 7 | 0.063 | 0.038 | 0.392 | 17 |
| flight-07a-lemniscate | 7 | 0.107 | 0.029 | 0.329 | 30 |
| flight-07p-lemniscate | 26 | 0.218 | 0.065 | 0.327 | 60 |
| flight-13a-trackRATM | 11 | 0.050 | 0.012 | 0.218 | 14 |
| flight-13p-trackRATM | 11 | 0.218 | 0.115 | 0.281 | 60 |
| flight-16p-trackRATM | 7 | 0.219 | 0.123 | 0.247 | 60 |

R2: FRR 0.053 (n=57), removal 1.000, mirror 1.000, x0.5 1.000, x1.5 0.965, yaw 0.298

## UZH forward

real testable: 0/27; abstention 100.0%

| sequence | n | median window s | min | max | samples |
| --- | --- | --- | --- | --- | --- |
| indoor_forward_10 | 3 | 0.077 | 0.077 | 0.080 | 1 |
| indoor_forward_3 | 3 | 0.124 | 0.111 | 0.133 | 2 |
| indoor_forward_5 | 3 | 0.071 | 0.071 | 0.100 | 1 |
| indoor_forward_6 | 3 | 0.117 | 0.095 | 0.153 | 2 |
| indoor_forward_7 | 3 | 0.111 | 0.090 | 0.117 | 2 |
| indoor_forward_9 | 3 | 0.124 | 0.095 | 0.332 | 2 |
| outdoor_forward_1 | 3 | 0.077 | 0.074 | 0.105 | 1 |
| outdoor_forward_3 | 3 | 0.111 | 0.074 | 0.124 | 2 |
| outdoor_forward_5 | 3 | 0.091 | 0.087 | 0.095 | 1 |

UZH coherence at 0.10 s: FRR corr 0.667, sign 0.630, n=27
