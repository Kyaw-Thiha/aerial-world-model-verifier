# M04 — The sign sub-tier with a reader-noise-tied dead-band

Track M follow-up, new pre-registered variant. Written 2026-11-22. Branch `track-m-gates`.
Pre-registration `configs/m04_sign_deadband_prereg.yaml` (committed at `e945f19`, **before** the
variant was computed). Code `experiments/m04_sign_deadband.py`. Results `results/m04/`. CPU
only, **zero GPU spend**. No frozen config was touched: `configs/coherence_frozen.yaml`
(`tau_sign = 0.292`) stands as published and is reported beside this variant.

## Decision

**The published sign sub-tier is not usable as a detection tier at a 5 % budget: at its own
leave-one-flight-out cut it flags 7/69 real clips and 10/69 tilt removals, which is no
separation at all, because the statistic is discrete enough that the 95th percentile lands on
its ceiling (score 1.0) for most folds. Tying the dead-band to the measured reader noise
(`eps = sin(sigma_n)`, `sigma_n = 1.91°`) fixes it — 4/55 false-reject, 55/69 removal, 55/69
mirror, 5/55 yaw control — and meets the pre-registered success criterion, at the cost of 14
clips whose lean never rises above the noise and which therefore have no sign to read.**

## Plain-language summary

1. **Why the frozen rule fails is discreteness, not the dead-band.** A frame's sign is one bit;
   a window has 6-8 frames per axis, so `s_j` takes a handful of values and
   `1 - min(s_x, s_y)` takes fewer. The 95th percentile of that distribution is 1.0 on most
   folds, so the tier only flags clips that reach the ceiling and a removal that also reaches
   it is indistinguishable from a real clip that does. The published `EPS = 1e-4` is a
   numerical floor, so on a clip whose required lean is a few hundredths of the reader's noise
   both signs are coin flips and each frame adds an uninformative bit.
2. **The noise dead-band makes the statistic informative.** With `eps = sin(1.0 sigma_n)` the
   tier's rate on real clips falls to 4/55 while tilt removal is flagged in 55/69 and mirrored
   roll in 55/69 — the two families the sign tier exists to catch. The yaw control is 5/55
   against the frozen version's 8/69, so it does not get worse.
3. **It costs coverage, and honestly so.** 14 of the 69 gate-kept development clips have no
   window with two sign-bearing frames above the noise; they return no value at all rather than
   a coin-flip one. That is the same 20 % of the population Gate A calls `weak_signal`, reached
   from the video side.
4. **On transferred thresholds it does not transfer well.** Applied to the 80 feasible
   simulated clips the real-fitted cut flags 19/80 (24 %) and detects 55/90; the frozen cut
   (which is at the ceiling) flags nothing at all. The cliff at the threshold is a property of
   a discrete statistic and it is the reason this variant should be evaluated only where its
   own cut is fitted.
5. **On the H04 arm-A clips (post-hoc illustration) the tier is not a discriminator under
   either rule.** Arm A, B and C all sit at the ceiling (median 1.0) under the frozen rule, and
   the dead-band mainly reduces how many clips are scoreable (arm A 60 → 32 at 1 sigma) without
   separating the arms. No claim is made from these clips; they were not used to choose the
   dead-band.

## Question

The published sign sub-tier scores a frame's sign agreement unless the lean component is below
1e-4, a numerical floor. Does tying the dead-band to the measured reader noise make the
sub-tier's false-reject, yaw-control and detection rates better than the frozen version at its
own threshold?

## Setup

**Statistic.** Per axis `j`, `s_j` = the fraction of frames with matching signs, counted only
over frames with `|l_dep,j| > eps` and `|l_req,j| > eps`. Clip statistic `1 - min(s_x, s_y)`,
maximised over the published 8-frame windows (hop 4); a window abstains with fewer than 6
usable frames or fewer than 2 frames passing the dead-band.

**Dead-bands.** Frozen: `1e-4`. Variants: `sin(0.5 sigma_n)`, `sin(1.0 sigma_n)` (primary),
`sin(2.0 sigma_n)` with `sigma_n = 1.9106°` from M01, i.e. 0.01667 / 0.03335 / 0.06667 in lean
units.

**Thresholds.** Leave-one-flight-out 95th percentile on the real dev gate-kept clips, per
variant. The median LOFO cut is 1.0 (frozen), 0.487 (0.5 sigma), 0.446 (1.0 sigma), 0.348
(2.0 sigma) — the first is the ceiling.

**Populations.** Real dev (69 gate-kept clips, all edit families and both controls), real
held-out (descriptive), the 80 feasible + 90 defect simulated dev clips at the *transferred*
real-dev cut, and the 78 H04 arm-A clips as a post-hoc illustration only.

## Results — the real development clips (k/n)

| variant | false-reject | removal | mirror | x0.5 | x1.5 | yaw | null |
| --- | --- | --- | --- | --- | --- | --- | --- |
| frozen (1e-4) | 7/69 | 10/69 | 10/69 | 8/69 | 7/69 | 8/69 | 7/69 |
| 0.5 sigma_n | 4/56 | 56/69 | 57/69 | 5/69 | 5/69 | 6/69 | 4/69 |
| **1.0 sigma_n** | **4/55** | **55/69** | **55/69** | 5/69 | 5/69 | **5/55** | 4/55 |
| 2.0 sigma_n | 4/55 | 46/69 | 55/69 | 5/69 | 5/69 | 5/55 | 4/55 |

At the primary dead-band the pre-registered success criterion is met: the yaw-control rate
falls by 4-5 percentage points and tilt-removal and mirror detection gain 45 of 69 clips each.
The x0.5 and x1.5 families stay at chance, as they must for a sign statistic.

Note the denominators: 55 not 69 because 14 clips have no window with two sign-bearing frames
above the noise. Those clips are *not* silently counted as non-detections — the variant
returns no value and the denominator reflects it, which is the difference between this table
and the published one.

## Results — transferred to the simulated dev clips

| variant | transferred cut | false-reject | detected |
| --- | --- | --- | --- |
| frozen (1e-4) | 1.000 | 0/80 | 0/90 |
| 0.5 sigma_n | 0.487 | 20/80 | 62/90 |
| 1.0 sigma_n | 0.446 | 19/80 | 55/90 |
| 2.0 sigma_n | 0.348 | 15/80 | 47/90 |

The frozen cut at the ceiling flags nothing on the simulated set: the statistic saturates there
too. The dead-band variants transfer but flag a quarter of the feasible clips, which is the
discreteness cliff again — a fitted cut would sit elsewhere. By family at 1.0 sigma the median
score is 0.50 in forest against 0.0 in industrial and urban.

## Results — H04 arm A, post-hoc illustration only

| variant | arm A median (n) | arm B median (n) | arm C median (n) |
| --- | --- | --- | --- |
| frozen (1e-4) | 1.00 (60) | 1.00 (56) | 1.00 (60) |
| 1.0 sigma_n | 1.00 (32) | 1.00 (14) | 1.00 (24) |

All three arms sit at the ceiling under both rules: the sign sub-tier does not separate a
flatness arm from a level or mirrored one on these clips. These clips were not used to choose
the dead-band and no threshold is fitted on them; the table is reported because the
pre-registration asks for it, and it says the tier is not a generator discriminator.

## What this supports

* **A concrete defect in the published sign sub-tier**: at a 5 % budget it cannot distinguish a
  tilt removal from a real clip (10/69 against 7/69). The cause is the interaction of a
  one-bit-per-frame statistic with a 95th-percentile cut, not the value of `EPS`.
* **A one-parameter fix that uses an already-measured quantity**: the reader noise M01 estimated
  for Gate A is also the right dead-band for the sign tier, and no second free parameter is
  introduced.
* **A coverage cost stated as coverage**: 14 of 69 clips become unscoreable, which is the same
  population Gate A abstains on from the path side.

## What this does not show

* Whether the dead-band should also be applied to the *correlation* sub-tier. It should not be
  obvious either way and it was not tested.
* Any improvement in the combined verdict: this variant is a sub-tier, and it is not spliced
  into the M03 configurations. Doing so would require re-running the end-to-end comparison with
  a re-fitted threshold, which is a new pre-registration.
* Anything about the H04 arms beyond "the sign statistic does not separate them".

## Caveats

* The frozen comparator's threshold sits on the statistic's ceiling, so the comparison is
  partly a comparison of two *calibrations* rather than two statistics. Both are reported with
  their own cut, as the pre-registration requires, and the ceiling is named.
* 55 clips over six flights is the smallest population in this track; every interval would be
  descriptive and none is quoted.
* The discrete statistic's behaviour depends on the window length; the 8-frame published window
  is used unchanged.

## Provenance

* **Code**: `experiments/m04_sign_deadband.py`; frozen comparator is
  `src/verdict/coherence.py::_sign_agreement` (EPS = 1e-4).
* **Results**: `results/m04/{sign_deadband.json, real_sign.csv, sim_sign.csv, h04_sign.csv}`.
* **Inputs**: schema-v2 clips plus the schema-v3 exact required lean (as M01); the simulated
  GT npz; the H04 path file and Track M's reader scores.
