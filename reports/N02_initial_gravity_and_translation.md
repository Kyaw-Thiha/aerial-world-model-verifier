# N02 — Initial gravity, drift, and the translation-direction addendum

**Decision.** With one gravity reading as the start, the propagated lean carries the *start's* own
error and nothing more: the first-frame reader costs 8.85° in forest and 8.22° on real outdoor
UZH at 10 s (oracle 2.25° / 0.42°), and neither a 1 s average nor a Huber fit beats a single
frame in the open scenes. The translation-direction addendum **fails** its pre-registered rule —
the essential-matrix direction is accurate on real outdoor UZH (1.78° median at ≥ 8 m/s, inside
the 3° bar) but not in simulated forest (18.36°), so the two-stratum "both" condition fails, and
the pre-registered starts that would use it (a, b) were therefore not run.

Pre-registrations: `configs/n02_initial_gravity_prereg.yaml` (`8fbd8d6`, before any number here),
amendments `configs/n02_amendment_a.yaml` (turn test definition), `..._b.yaml` (detector
settings), `..._c.yaml` (the `recoverPose` reference convention, a real bug). Code
`experiments/n02_initial_gravity.py`, `experiments/n02_translation_arm.py`,
`experiments/n02_turn_test.py`. CPU only, zero GPU spend in this part. Results
`results/n02/`.

## Plain-language summary

1. **The start decides everything.** Part 0 showed rotation drift is small (0.4-1.8° at 10 s with
   the oracle). Reading the start from the reader throws that away: the propagated error becomes
   the reader's own error (8.85° forest, 8.22° outdoor), because a wrong start is a constant
   offset that the propagation faithfully carries.
2. **Averaging the reader over 1 s makes it worse, not better.** In the open simulated families
   the first-frame start is 0.73° (urban) and 1.27° (industrial) while the 1 s average is 2.99°
   and 2.09°, and the Huber fit 4.81° and 1.87°. Rotating each reading into the start frame with
   the estimated rotation injects the rotation drift into the average; a single reading does not
   pay that price.
3. **The addendum idea works exactly where the reader fails on real data, and fails where the
   renders are.** The translation direction from an essential matrix is 1.78° from truth on the
   real outdoor UZH sequences (and 1.1-3.0° per sequence) — well inside the 3° bar. On the
   simulated forest it is 18.36° at ≥ 8 m/s.
4. **The forest failure has a geometric cause, not a feature-count cause.** Every simulated forest
   clip that reaches 8 m/s has a 30° or 45° mount: at 45° the camera looks down and forward
   flight runs close to the optical axis, which is the degenerate direction for translation
   recovery. The 45° group is 25.92°; the 30° group is 10.43°. Pairs at 0°/15° never reach the
   speed cut.
5. **At the shortest gap the direction is worst** (forest 27.1° at 0.1 s falling to 10.0° at
   0.5 s; industrial 21.7° to 8.7°): it is a baseline/depth problem, exactly as the parallax
   argument predicts.
6. **Turns do not rescue it.** With the direction corrected by the second window the lean error
   at θ ≥ 30° is 6.13° on UZH against an 8.28° video-only baseline — a 26 % reduction, short of
   the pre-registered 5° / 30 % bar. A control with *perfect* directions reduces the same
   baseline to 4.03°, so the mechanism has headroom and the essential-matrix noise is the
   binding constraint, not the algebra.
7. **The evaluability floor was mis-specified and is reported, not repaired.** The pre-registered
   floor asked for ≥ 5 clips and ≥ 100 pairs per stratum; the real-outdoor stratum is three
   sequences by construction. Its accuracy criterion passes (1.78° / 2.67° per second, 209
   high-speed pairs), but the literal rule cannot mark it evaluable. The overall verdict is
   unaffected because forest fails by 6× the bar; the floor is left as written.

## Part 1 — the cost of the start

Arms (descriptive, as pre-registered): oracle; first-frame reader; first-T reader (T = 1 s,
readings carried into the start frame with the estimated rotations and averaged); Huber fit of a
constant up vector over the first 3 s and over all frames up to the 10 s horizon (non-causal; the
constant fitted in the two lean axes is the initial gravity *plus* the reader's constant offset).
Horizon 10 s, starts every 1 s. Median of per-unit medians.

| population | oracle | first frame | first 1 s | Huber 3 s | Huber all |
| --- | --- | --- | --- | --- | --- |
| forest, sim (26) | **2.25** | 9.23 | 9.02 | 8.49 | 7.34 |
| urban, sim (27) | 4.62 | **4.85** | 6.60 | 10.28 | 8.60 |
| industrial, sim (27) | 2.50 | **2.66** | 7.49 | 10.52 | 9.73 |
| outdoor UZH (3) | **0.42** | 7.65 | 8.83 | 12.91 | 9.82 |
| indoor UZH (6) | 0.66 | **0.96** | 9.92 | 8.74 | 7.84 |
| TII dev (6) | 44.22 | 41.83 | 38.23 | 30.17 | **26.22** |
| TII held out (2) | 42.12 | 35.29 | 32.01 | 31.23 | **26.80** |

Start errors (the same arms, at the start frame): forest 8.85 / 8.09 / 8.52 / 4.98; outdoor UZH
8.22 / 7.96 / 11.93 / 9.87; urban 0.73 / 2.99 / 5.25 / 4.81. The propagated 10 s error tracks the
start error closely: **rotation drift adds little once the start is wrong**, which is the point
of Part 0 read backwards.

On TII the direction reverses — the Huber start *improves* the propagated result (44.2° → 26.2°)
because the reader's start error partially cancels the rotation drift that dominates there. That
is a cancellation, not a fix: no arm is close to usable on TII.

**Not run, by rule:** the `reader_plus_translation` start arm, because the addendum rule failed
(the pre-registration gates it).

## The translation-direction addendum

**Measurement.** SIFT (4000 features, contrast 0.01, ratio 0.75) → RANSAC essential matrix
(1.0 px, 0.999) → `recoverPose`. Direction error `delta = angle(t_est, t_true)` with
`t_true = -R_cw(j)^T (pos_j - pos_i)` (OpenCV's convention, amendment C). Gaps 0.1/0.2/0.3/0.5 s,
starts every 3 frames. 49,144 simulated pairs and 558 UZH pairs; 20 inliers minimum.

### Direction error by family and speed

| stratum | pairs | median | ≥ 8 m/s: pairs | median | per 1 s block |
| --- | --- | --- | --- | --- | --- |
| forest, sim | 13,766 | 15.24 | 2,410 | **18.36** | 13.82 |
| industrial, sim | 15,408 | 12.96 | 4,002 | 10.77 | 6.82 |
| urban, sim | 13,404 | 13.28 | 1,319 | 16.44 | 8.31 |
| outdoor UZH | 537 | 1.84 | 209 | **1.78** | 2.67 |

UZH per sequence at ≥ 8 m/s: `outdoor_forward_1` 1.14° (39 pairs), `_3` 2.03° (133),
`_5` 2.98° (37). UZH has only the 0.2 s gap: at 15.07 Hz the 0.1/0.3/0.5 s gaps are not whole
frames, so the gap sweep is a simulated-set result.

### Direction error by pair gap (median, all speeds)

| stratum | 0.1 s | 0.2 s | 0.3 s | 0.5 s |
| --- | --- | --- | --- | --- |
| forest, sim | 27.08 | 15.22 | 12.11 | 10.02 |
| industrial, sim | 21.68 | 12.92 | 10.29 | 8.71 |
| urban, sim | 19.95 | 13.64 | 11.44 | 9.34 |
| outdoor UZH | — | 1.84 | — | — |

### Forest by mount (the geometric cause)

| mount | pairs | all speeds | ≥ 8 m/s | high-speed pairs |
| --- | --- | --- | --- | --- |
| 0° | 3,703 | 12.21 | — | 0 |
| 15° | 4,423 | 12.62 | — | 0 |
| 30° | 4,077 | 22.01 | 10.43 | 990 |
| 45° | 1,563 | 24.82 | 25.92 | 1,420 |

No 0° or 15° forest clip reaches 8 m/s, so the high-speed population is entirely 30°/45° — the
views where forward motion is closest to the optical axis and translation direction is worst
conditioned.

### The pre-registered rule

| stratum | evaluable | pair median ≥ 8 m/s | 1 s median ≥ 8 m/s | verdict |
| --- | --- | --- | --- | --- |
| forest | yes (8 clips, 2,410 pairs) | 18.36° | 13.82° | **fail** |
| real outdoor UZH | **no** (3 clips < the 5-clip floor; 209 pairs) | 1.78° | 2.67° | cannot pass |

Rule (verbatim): proceed if the median error is 3° or less in forest and real outdoor at speeds
of 8 m/s or more. **The addendum fails**: forest is six times the bar. The UZH accuracy numbers
meet the bar but the evaluability floor — copied from the simulated strata, where 5 clips is
sensible — cannot be met by a stratum the same pre-registration defines as three sequences. This
is recorded as a design flaw in the pre-registration, not repaired after the result; the overall
verdict does not depend on it.

### Robustness diagnostics

* Sign ambiguity: at ≥ 8 m/s, the sign-ambiguous error `min(delta, 180-delta)` is 16.92° in
  forest (so the failure is direction accuracy, not a flipped baseline) and 1.78° on UZH (no
  sign problem).
* The pre-amendment formula (`R_cw(i)^T d`) would have reported 159-171° medians on every
  stratum; it is kept as `delta_old_deg` in the pair tables and excluded from every rate.
* Pair failures: 10-12 % of simulated pairs (mostly `few_features` on the industrial and urban
  renders), 4 % of UZH pairs. Inlier counts are 82-293 median; Sampson residuals of the
  RANSAC inliers are 10-15 px (the OpenCV-5 threshold semantics are opaque; the estimate is
  validated on a synthetic pair to 2.15°).

## Turn observability

Operationalisation (amendment A): start at window 1 with the first-frame reader, propagate to
window 2 with DPVO, fit one constant body-frame lean correction to (i) the window-2 direction
constraint alone and (ii) both constraints. Residual lean error at window 2 by the angle θ
between the two velocity directions. Rotation about the velocity axis is not observable from
directions and is stated as a limit, not measured.

| stratum | θ | n | video only | one constraint | two constraints |
| --- | --- | --- | --- | --- | --- |
| UZH outdoor | 0-10° | 237 | 8.11 | 8.38 | 8.70 |
| UZH outdoor | 10-30° | 1,000 | 7.98 | 8.32 | 12.05 |
| UZH outdoor | **≥ 30°** | 8,652 | **8.28** | 7.21 | **6.13** |
| forest, sim | ≥ 30° | 68,340 | 9.10 | 12.53 | 12.76 |
| industrial, sim | ≥ 30° | 68,345 | 1.90 | 5.56 | 7.98 |
| urban, sim | ≥ 30° | 56,352 | 2.24 | 7.00 | 9.57 |

Reading (pre-registered): supported if at θ ≥ 30° the two-constraint median is ≤ 5° and at least
30 % below the baseline. UZH: 6.13° is 26 % below 8.28° — **not supported**, though the direction
of the effect is right. On the simulated clips the correction makes things worse, because the
direction error there (10-26°) is larger than the error being corrected.

**Ideal-direction control.** Replacing the measured directions by the truth directions in the same
solver reduces the UZH baseline from 8.22° to 4.03° median (1,760 window pairs). So the algebra
and the constant-correction model can halve the error; what stops it is the essential-matrix
noise and its tail (UZH `delta` p90 17.8° at speed), not the formulation.

**The limit, stated.** One translation direction observes only the tilt component that moves the
velocity direction. Two non-parallel directions span the two lean axes in principle, which is what
the UZH θ ≥ 30° row shows, but one pair of directions cannot observe rotation about the travel
axis and a noisy pair cannot deliver the accuracy a start needs. On the simulated forest the
direction is too inaccurate for either use.

## Generated-clip track noise

Measured on the local copies of the H04 SolarWM clips (234 videos, 81 frames at 16 fps) from
track-m's verified mirror; nothing was re-downloaded and no held-out simulated clip was read.
Calibration-free fundamental-matrix fit (RANSAC 1.0 px): the **Sampson residual of the RANSAC
inliers in pixels**, median and p90 per arm and draw, with the fraction of pairs below the
20-inlier floor. This is the number the budget says decides whether a geometry route is viable
on generated video; it is reported even though the addendum rule failed.

<!-- TRACK_NOISE -->

## What this supports

* **The start is the whole problem.** Parts 2-4 must be judged with the reader start's error in
  the lean, not the oracle's; the offset-free tiers are the only ones that can ignore it.
* **Translation direction is a usable absolute reference on real outdoor imagery** (1.8°) and is
  not on the simulated renders at high speed/high mount (18-26°). A hybrid that trusts it only on
  real-looking, low-mount, near-field footage is defensible; a general one is not.
* **Turns carry real information** (UZH 8.28° → 6.13° at θ ≥ 30°) but not enough at this noise
  level, and the ideal-direction control bounds what better tracking could buy (about 4°).

## What this does not show

* No verdict behaviour: whether the propagated lean keeps the tiers' detection is Part 2.
* No coverage: Part 3's gate is where the constant offset and scene bias interact.
* No generated-video attitude: Part 4 needs the rotation extractor on the generated clips; the
  track-noise number above only says whether the feature geometry exists.
* No intervals: 3 UZH sequences and 6-7 simulated clips per mount; every spread is descriptive.

## Caveats

* The addendum's evaluability floor (≥ 5 clips) is unmeetable for a 3-sequence stratum; the
  accuracy numbers pass but the literal rule marks it not evaluable. Left as written.
* The UZH gap sweep is one gap (0.2 s) because of the 15.07 Hz frame rate; the gap trend is a
  simulated-set result.
* The Huber "all frames" arm is non-causal (it uses frames up to the horizon); the 3 s arm is
  short-horizon but still uses future frames relative to the start.
* The turn test pools overlapping window pairs (tens of thousands); they are not independent and
  no interval is computed.
* SIFT settings were changed before measurement (amendment B) because the default threshold left
  some renders below 30 keypoints; the change is uniform across strata.

## Provenance

* Pre-registration `8fbd8d6`; amendments A/B/C; decisions `[N]`.
* Poses: `awmv-n-rotation` / `awmv-n-data` (Part 0); readers: L01 per-frame scores and E's UZH
  JSONL (sha256 in `results/n01_pretest/provenance.json`).
* Pair tables `results/n02/pairs_{sim,uzh}.csv` (49,702 rows), per-second table, turn table,
  generated track-noise table; all CPU, USD 0.00.
