# N01 — Rotation-propagated gravity: the Part 0 pre-test

**Decision.** Part 0 **passes** the pre-registered rule: at a 10 s horizon with an oracle start,
the propagated lean drifts a median **1.80°** in simulated forest (22 clips, reader 8.51°) and
**0.43°** on the three real outdoor UZH sequences (reader 8.36°), each far under the bar of half
the reader, with **no extractor failures**. Parts 1-4 therefore proceed — with the caveat stated
up front that the forest pass is carried by the 0° and 15° mounts (0.89° and 0.88°) while the
30° and 45° mounts drift 8.62° and 14.83°, and that the idea is confirmed dead on TII racing.

Pre-registration `configs/n01_pretest_prereg.yaml` (commit `ebacc7c`), written before any 1, 5,
10 or 20 s number existed and before the new DPVO runs. Code `experiments/n01_rotation_pretest.py`
(CPU), tests `tests/test_n_rotation.py`; GPU `remote/n_rotation.py`, app `awmv-n-rotation`
(A10), volume `awmv-n-data`. Results `results/n01_pretest/`, run summaries
`data/n01/runs/` (94 runs, 42,782 frames, 830 s of A10 = USD 0.25).

## Plain-language summary

1. **The quantity.** Read gravity once, at the true value (an oracle start), then carry it
   forward with the camera's estimated rotation. The error after a horizon is the angle between
   that propagated up-vector and the true up-vector. Yaw drift cannot enter: a rotation about
   world up leaves the up-vector unchanged.
2. **In simulated forest the rotation is better than the reader by a factor of five at 10 s**
   (1.80° against 8.51°), and the error grows slowly (0.32° at 1 s, 1.17° at 5 s). That is the
   scene where the reader fails, so this is the result the track exists for.
3. **But the forest pass is not uniform.** By mount, the 10 s median is 0.89° (0°, 5 clips),
   0.88° (15°, 7 clips), **8.62° (30°, 7 clips)** and **14.83° (45°, 3 clips)**. The 30° and 45°
   mounts are not better than the reader (8.51°); the stratum median passes because the low
   mounts are excellent. Anyone planning to use this must condition on mount, not quote the
   pooled number.
4. **On the real outdoor UZH forward rigs it is excellent:** 0.43° at 10 s against the reader's
   8.36°, and 0.45° at 20 s where a 20 s pair exists. The reader error there is a scene bias that
   a one-off read plus rotation avoids entirely.
5. **The weak cases reproduce J01 exactly and are not rescued.** TII racing: 42.31° at 10 s
   against the reader's 3.42°, and two of eight flights (13p, 16p) are detected as *frozen*
   rotation estimates (the extractor's median step is under 10 % of the truth's). The 45°-down
   UZH rigs: 12.67° at 10 s against a reader of 12.18°, with `indoor_45_9` at 106° (lost).
6. **The in-between scenes are honest losses.** Urban 3.69° against a 0.91° reader (ratio 4.07)
   and industrial 2.01° against 1.27° (1.58): where the reader is good, propagation is worse.
   That is the expected trade, and it is why the two sources should check each other (Part 3).
7. **No second extractor was run, by the pre-registered rule.** The rule orders a ViPE arm only
   if the primary ratio lies within 0.20 of the 0.50 boundary; 0.21 and 0.05 do not.

## Question

With an oracle initial gravity, how far does the propagated lean drift over 1, 5, 10 and 20 s in
the scenes where the reader fails — simulated forest and real outdoor UZH — and is the 10 s
drift at most half the reader's error there, with extraction failing on at most 20 % of clips?

## Setup

**Populations.** Simulated: the 26 feasible development clips of Track I's frozen split in
`forest_ridge` (mounts 0/15/30/45°; four clips of 10.0 s are `insufficient_duration` and are
excluded from the 10 s median and the failure denominator). Real outdoor: the three
`outdoor_forward` UZH sequences (450/450/223 pinhole frames, 30.0/30.0/14.8 s). Reported but not
decisive: urban and industrial simulated families, the six indoor-forward UZH sequences, the five
45°-down UZH sequences, and the eight TII flights.

**Extractor.** DPVO at commit `0ac95b6`, the project's standing settings (48 patches, removal 12,
optimisation 8, lifetime 5), never tuned here. UZH: a fresh run on the 450-frame prepared
pinhole sequences on `awmv-e-data` (Track C's stored poses stop at 150 frames and cannot reach a
10 s horizon). Simulated: a fresh run on the 80 feasible clips (640x360 pinhole, fx = fy = 320,
height cropped to 352). TII: the stored stride-8 poses, unchanged. Every UZH sequence and every
simulated clip returned a pose for every input frame (450/450, 450/450, 223/223; 80/80 clips).

**Statistic.** For each start frame i with a partner j at t_i + h (within 1.5 samples):
`u_prop = (R_est[j]^T R_est[i]) u_true[i]`, error `= angle(u_prop, u_true[j])` in degrees.
Per-unit median over starts, then the median across units (primary); the frame-pooled median is
reported beside it. Everything is yaw-free.

**Reader reference on the same units.** Simulated: the frozen GeoCalib-224 reader of L01, per-clip
median of `up_error_deg`. UZH: E01 variant a (pinhole, Kalibr focal prior) recomputed from the
stored per-frame JSONL and the same truth. The pipeline check reproduces E01's nine published
per-sequence medians to **0.069°** worst case (outdoor 8.36/7.83/9.59 against 8.36/7.90/9.59).

## Validation before the rule

| check | result |
| --- | --- |
| identity: propagate with the TRUE rotations | max 1.9e-6° (arccos rounding), 15 pairs |
| reproduce J01 TII 0.5 s and 1.0 s, 8 flights | exact to 0.01° on all eight |
| reproduce J01 UZH DPVO/ViPE 1 s on the 150-frame poses | 0.235/0.190/0.185 (DPVO), 0.289/0.214/0.240 (ViPE); J01 identical |
| simulated geometry: up from `R_bw_cam @ R_cb` vs the clip's `lean_dep` | 3.9e-16 on all 80 clips |
| UZH reader reproduction (E01 variant a) | worst 0.069° (outdoor_forward_3) |

The simulated convention needed care and is recorded in `docs/decisions.md` [N]: `R_bw_cam` in
Track I's ground-truth file is the **body** attitude the camera is rendered on; the camera-to-
world is `R_bw_cam @ R_cb`.

## Results

### The pre-registered rule

| stratum | units | evaluable at 10 s | reader | 10 s propagated | ratio | failures |
| --- | --- | --- | --- | --- | --- | --- |
| forest, simulated | 26 | 22 | 8.51° | **1.80°** | **0.21** | 0/26 |
| real outdoor UZH | 3 | 3 | 8.36° | **0.43°** | **0.05** | 0/3 |

Rule (verbatim): proceed if, in forest and real outdoor UZH, the 10 s median propagated-lean
error is at most half the reader's error and the failure rate is at most 20 %. **Both strata
pass both clauses.** Pooled-over-frames reading: 2.20° (forest) and 0.36° (outdoor) — the same
verdict under both readings.

### Horizons, by stratum

| stratum | reader | 1 s | 5 s | 10 s | 20 s |
| --- | --- | --- | --- | --- | --- |
| forest, sim (22) | 8.51 | 0.32 | 1.17 | 1.80 | 6.21 (7 units) |
| urban, sim (23) | 0.91 | 0.39 | 1.93 | 3.69 | 7.32 (6) |
| industrial, sim (23) | 1.27 | 0.28 | 0.73 | 2.01 | 1.85 (10) |
| outdoor UZH (3) | 8.36 | 0.18 | 0.52 | 0.43 | 0.45 (2) |
| indoor UZH (6) | 0.96 | 0.25 | 0.54 | 0.71 | 1.28 (5) |
| 45°-down UZH (5) | 12.18 | 0.84 | 2.98 | 12.67 | 15.41 (3) |
| TII (6) | 3.42 | 3.73 | 21.21 | 42.31 | 10.97 (6) |

All cells are the median of per-unit medians; per-unit and per-horizon numbers with start counts
are in `results/n01_pretest/horizons.csv` and `tables.md`.

### Forest by mount (the caveat in numbers)

| mount | 1 s | 5 s | 10 s | 20 s |
| --- | --- | --- | --- | --- |
| 0° (5) | 0.20 | 0.89 | **0.89** | 1.56 (2) |
| 15° (7) | 0.28 | 0.92 | **0.88** | 4.54 (2) |
| 30° (7) | 0.92 | 3.18 | **8.62** | 9.65 (3) |
| 45° (3) | 1.62 | 8.30 | **14.83** | — |

At 30° and 45° the propagated error at 10 s is at or above the forest reader's (8.51°), so a
per-mount reading of the rule would fail at high uptilt even though the stratum-level rule
passes. The drifts are concentrated in turning manoeuvres at high mount (`i02_dev_059`, 45°
slalom, 51.1°; `i02_dev_035`, 45° climb-turn, 10.4°; the 30° turns 5.5-9.5°).

### Real outdoor UZH, per sequence

| sequence | reader | 1 s | 5 s | 10 s | 20 s |
| --- | --- | --- | --- | --- | --- |
| outdoor_forward_1 | 8.36 | 0.18 | 0.52 | 0.43 | 0.44 |
| outdoor_forward_3 | 7.83 | 0.11 | 0.21 | 0.22 | 0.47 |
| outdoor_forward_5 | 9.59 | 0.23 | 0.53 | 0.44 | — (14.8 s) |

### Failure rates (crash, partial poses, frozen estimate, no pair)

| stratum | attempted | crash | partial | frozen | no 10 s pair | rate |
| --- | --- | --- | --- | --- | --- | --- |
| forest, sim | 26 | 0 | 0 | 0 | 0 | 0 % |
| outdoor UZH | 3 | 0 | 0 | 0 | 0 | 0 % |
| indoor UZH | 6 | 0 | 0 | 0 | 0 | 0 % |
| 45°-down UZH | 5 | 0 | 0 | 0 | 0 | 0 % |
| TII | 8 | 0 | 0 | **2** | 0 | 25 % |

The two TII frozen estimates are `flight-13p-trackRATM` and `flight-16p-trackRATM`: DPVO's median
per-step rotation is 0.21° and 0.14° against truth's 7.57° and 8.51° — the rotations are close to
frozen, exactly J01's finding (1.5 % of the rotation budget on 16p). Their per-flight 10 s errors
are 40.3° and 51.3°.

### The 45°-down UZH rigs (descriptive)

`indoor_45_9` diverges (10 s median 106.0°, reader 63.6°), `outdoor_45_1` reaches 36.8° (reader
69.7°), while `indoor_45_2/4/13` are 3.8/12.7/1.6°. These sequences are the reader's worst case
*and* the rotation's; neither source is usable there, which is worth saying before Part 3 tries to
combine them.

## What this supports

* On calibrated, feature-rich imagery, **one gravity reading plus video rotation replaces a
  per-frame absolute reader for at least 10 s**, with margin (0.05-0.21 of the reader's error) in
  the two strata where the reader fails.
* The idea is **mount-dependent**: at 0-15° uptilt in forest it is excellent, at 30-45° it is not
  better than the reader. Any deployment claim must be conditioned on view geometry.
* TII racing and the 45°-down UZH rigs are **out of scope**, confirmed rather than assumed.

## What this does not show

* **No intervals.** Three outdoor sequences are three clusters; forest has four mount clusters of
  3-7 clips. Every spread here is descriptive, not a 95 % confidence interval.
* **No verdict.** This measures the lean error only. Whether an offset-free tier still detects
  removal/mirror/gain when the lean is propagated is Part 2.
* **No second extractor.** ViPE was not run (the pre-registered trigger did not fire), so
  "video rotation" here means DPVO specifically; ViPE was only 0.05° worse at 1 s on the three
  UZH sequences J01 scored.
* **No start estimation.** The oracle start is not implementable; Part 1 prices the reader-based
  starts and the translation-direction anchor.
* **No generated video.** DPVO on generated clips is Part 4; the addendum's track-noise
  measurement on the generated clips is reported in N02.

## Caveats

* The forest stratum pass hides the mount split (30° 8.62°, 45° 14.83°); the pre-registered rule
  was stratum-level and both readings are stated above. It was not re-specified after the result.
* Four of 26 forest clips are exactly 10.0 s and have no usable 10 s pair besides the boundary
  one; they are excluded from the 10 s median and the failure denominator as pre-registered.
* The TII 20 s column draws on a different, smaller set of flights than the 10 s column (only
  flights longer than 20 s have pairs) and is not comparable across columns.
* The 45°-mount forest group has only three evaluable clips, and the outdoor UZH group three
  sequences; both are flagged in every table.

## Provenance

* Pre-registration `configs/n01_pretest_prereg.yaml` (`ebacc7c`); decisions entry `[N]`;
  population and input hashes `results/n01_pretest/provenance.json`.
* Poses: app `awmv-n-rotation` on volume `awmv-n-data`; per-run summaries `data/n01/runs/`
  (94 runs, 42,782 frames, 830 s A10 = USD 0.25 at 1.10 USD/h; image build is CPU).
* Reader inputs: L01 per-frame scores (branch `track-l-reader-simreal`, `2fd64d8`) and E's UZH
  JSONL on `awmv-e-data` (branch `track-e-depicted-uzh`, `dd209bf`), sha256 in the provenance file.
* No R2 object was written; no Modal app is left running.
