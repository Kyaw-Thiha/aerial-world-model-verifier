# stride-1 versus stride-2 reader consistency (Track G3)

| old stride | clips | frames compared | max abs d lean | clips outside 1e-9 |
| --- | --- | --- | --- | --- |
| 1 | 3 | 1252 | 2.309e-01 | 3 |
| 2 | 53 | 12249 | 1.011e+00 | 53 |

Two independent runs of the same clips (the reader's own stochasticity):
components 1202, median 2.187e-03, p90 6.755e-03, p99 2.237e-02, max 5.555e-02.

Total: 56 clips, 13501 shared frames, max abs d lean 1.011e+00.
