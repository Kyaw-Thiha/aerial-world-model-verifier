| variant | arm / path | real FRR | null | yaw | removal | mirror | x0.5 | x1.5 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| V1 preintegration | estimated attitude, pixel path | 0.125 (1/8) | 0.125 (1/8) | 0.250 (2/8) | 0.125 (1/8) | 0.625 (5/8) | 0.250 (2/8) | 0.750 (6/8) |
| V1 preintegration | oracle attitude, pixel path | 0.125 (1/8) | 0.125 (1/8) | 0.125 (1/8) | 0.125 (1/8) | 0.125 (1/8) | 0.125 (1/8) | 0.125 (1/8) |
| V1 preintegration | estimated attitude, oracle path | 0.250 (2/8) | 0.250 (2/8) | 0.250 (2/8) | 1.000 (8/8) | 1.000 (8/8) | 1.000 (8/8) | 1.000 (8/8) |
| V1 preintegration | oracle attitude, oracle path | 0.125 (1/8) | 0.125 (1/8) | 0.125 (1/8) | 0.125 (1/8) | 0.125 (1/8) | 0.125 (1/8) | 0.125 (1/8) |
| V2 shooting (primary) | estimated attitude, pixel path | 0.125 (1/8) | 0.125 (1/8) | 0.250 (2/8) | 0.750 (6/8) | 0.500 (4/8) | 0.375 (3/8) | 0.750 (6/8) |
| V2 shooting (primary) | oracle attitude, pixel path | 0.125 (1/8) | 0.125 (1/8) | 0.125 (1/8) | 0.125 (1/8) | 0.125 (1/8) | 0.125 (1/8) | 0.125 (1/8) |
| V2 shooting (primary) | estimated attitude, oracle path | 0.125 (1/8) | 0.125 (1/8) | 0.125 (1/8) | 1.000 (8/8) | 0.875 (7/8) | 0.875 (7/8) | 1.000 (8/8) |
| V2 shooting (primary) | oracle attitude, oracle path | 0.125 (1/8) | 0.125 (1/8) | 0.125 (1/8) | 0.125 (1/8) | 0.125 (1/8) | 0.125 (1/8) | 0.125 (1/8) |
| V3 band coherence | estimated attitude, pixel path | 0.250 (2/8) | 0.250 (2/8) | 0.125 (1/8) | 0.125 (1/8) | 0.000 (0/8) | 0.250 (2/8) | 0.125 (1/8) |
| V3 band coherence | oracle attitude, pixel path | 0.250 (2/8) | 0.250 (2/8) | 0.250 (2/8) | 0.250 (2/8) | 0.250 (2/8) | 0.250 (2/8) | 0.250 (2/8) |
| V3 band coherence | estimated attitude, oracle path | 0.125 (1/8) | 0.125 (1/8) | 0.125 (1/8) | 0.500 (4/8) | 0.500 (4/8) | 0.250 (2/8) | 0.125 (1/8) |
| V3 band coherence | oracle attitude, oracle path | 0.250 (2/8) | 0.250 (2/8) | 0.250 (2/8) | 0.250 (2/8) | 0.250 (2/8) | 0.250 (2/8) | 0.250 (2/8) |
| V4 velocity-drag | estimated attitude, pixel path | -- | -- | -- | -- | -- | -- | -- |
| V4 velocity-drag | oracle attitude, pixel path | -- | -- | -- | -- | -- | -- | -- |
| V4 velocity-drag | estimated attitude, oracle path | -- | -- | -- | -- | -- | -- | -- |
| V4 velocity-drag | oracle attitude, oracle path | -- | -- | -- | -- | -- | -- | -- |
| V5 physics scale | estimated attitude, pixel path | 0.000 (0/8) | 0.000 (0/8) | 0.000 (0/8) | 0.250 (2/8) | 0.000 (0/8) | 0.000 (0/8) | 0.000 (0/8) |
| V5 physics scale | oracle attitude, pixel path | 0.000 (0/8) | 0.000 (0/8) | 0.000 (0/8) | 0.000 (0/8) | 0.000 (0/8) | 0.000 (0/8) | 0.000 (0/8) |
| V5 physics scale | estimated attitude, oracle path | 0.000 (0/8) | 0.000 (0/8) | 0.000 (0/8) | 0.000 (0/8) | 0.000 (0/8) | 0.000 (0/8) | 0.000 (0/8) |
| V5 physics scale | oracle attitude, oracle path | 0.000 (0/8) | 0.000 (0/8) | 0.000 (0/8) | 0.000 (0/8) | 0.000 (0/8) | 0.000 (0/8) | 0.000 (0/8) |

Sequence-clustered 95 % intervals over the primary sequences (descriptive, 4 clusters):

| variant | real | removal | mirror |
| --- | --- | --- | --- |
| V1 preintegration | 0.00-0.38 | 0.00-0.38 | 0.25-1.00 |
| V2 shooting (primary) | 0.00-0.38 | 0.50-1.00 | 0.00-1.00 |
| V3 band coherence | 0.00-0.50 | 0.00-0.38 | 0.00-0.00 |
| V4 velocity-drag | -- | -- | -- |
| V5 physics scale | 0.00-0.00 | 0.00-0.50 | 0.00-0.00 |
