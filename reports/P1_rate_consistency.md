# P1 — Body-rate consistency on video rotation (Check B)

**Decision.** Check B **fails its pre-registered pooled criteria and works where the rotation
extractor is stable.** At the matched 4-of-80 false-reject the absolute-rate tier detects **2/19
tilt removals, 1/16 mirrors and 0/22 scale defects** — because the threshold is set to **40.2
deg/s** by feasible clips at the 30° and 45° mounts, whose DPVO rotation drifts by 10-16 deg/s;
at the 0°/15° mounts alone the same statistic, thresholded within mount, detects **6/6 removals,
4/10 mirrors and 16/16 scale defects at a 1/21 false-reject**. On real flight the three outdoor
forward UZH sequences pass (window false-reject **0/107, 0/109, 1/54** at leave-one-out
thresholds) and the 45°-down rig does not (54/58); TII is not evaluable (noise floor 60 deg/s);
and on the 234 generated SolarWM clips the check does **not** separate the physical arm from the
level and mirrored arms (median 14.2 vs 25.9 and 22.2 deg/s, all below the threshold).

Pre-registration `configs/p1_rate_consistency_prereg.yaml` (`667ae11`, committed before any
video-derived rate was scored). Code `experiments/p1_rate_consistency.py`, `experiments/p_common.py`,
`src/verdict/rate.py`. Results `results/p/p1/`. CPU only, zero spend. Development only: the 100
held-out sim clips and TII flights 13p/07a were never opened.

## Reading of the brief

The rate residual is computed with a per-clip **gauge fit** rather than the literal
`(R_mount omega_cam) x b3_req`, because the literal form is not invariant to the unconstrained
yaw about the thrust axis (P0 measured 8.7 deg/s of median error on perfectly feasible ground
truth; `docs/decisions.md` [P-a]). The literal form is measured here on real video as an
exploratory column and is much worse: median **26.6 deg/s** on the 14 UZH sequences against
**4.6 deg/s** for the gauge fit on the three outdoor-forward ones.

## Setup

| element | value |
| --- | --- |
| statistic | `median_k |rb_k − rd_k|` in deg/s over testable windows (B_abs, primary); `|e|/|rb|` (B_rel); `cos(rb, rd)` (B_coh, low is bad) |
| window | 0.5 s, hop 0.25 s; 0.25 s and 1.0 s computed, not reported as primary |
| testability | median `|rb|` ≥ 3 × the source's P0 null residual: 6.6 (sim), 3.2 (UZH), 75.5 (TII) deg/s |
| sim threshold | top 4 of 80 feasible clip scores = **40.18 deg/s** (B_abs), 1.22 (B_rel), 0.191 (B_coh) |
| real threshold | leave-one-sequence-out, 95th percentile of the other three sequences' testable windows |
| rotation | DPVO draw 1 for sim (30 fps), the stored 450-frame UZH poses (15.07 fps), TII stride-8 poses (15 fps) |

## Results

### The pooled operating point is set by the high mounts

| mount | feasible clips (testable) | median B_abs (deg/s) | median B_coh | gauge-fit residual (deg) |
| --- | --- | --- | --- | --- |
| 0° | 4 | 0.97 | 0.9996 | 0.43 |
| 15° | 17 | 1.25 | 0.9972 | 0.93 |
| 30° | 16 | 10.15 | 0.6947 | 4.12 |
| 45° | 20 | 16.50 | 0.9351 | 12.02 |

The extractor, not the flight, is the difference. With an oracle attitude the same residual is
0.0008 deg/s (P0). On 30°/45° mounts DPVO's orientation drifts by 10-60° over a clip while its
*increments* stay accurate (median 0.49 deg/step at 30°); a slowly varying orientation error
rotates the rate vector `omega x b3` without changing its length, so the rate magnitudes agree
(22.7 vs 22.1 deg/s on `i02_dev_018`) while the directions do not (median cosine 0.22). That is
the same 30°/45° drift N01 and N06 reported, seen in a new statistic.

### Detection at the pre-registered point (confirmatory)

| population | n | testable | B_abs | B_rel | B_coh |
| --- | --- | --- | --- | --- | --- |
| feasible | 80 | 57 | 4/57 | 4/57 | 4/57 |
| feasible, dynamic | 53 | 50 | 4/50 | 4/50 | 4/50 |
| tilt removed | 27 | 19 | **2/19** | 0/19 | 14/16 |
| mirrored roll | 27 | 16 | **1/16** | 3/16 | 0/16 |
| tilt scaled | 33 | 22 | **0/22** | 3/22 | 2/22 |
| partial interval | 20 | 12 | 0/12 | 2/12 | 3/12 |
| thrust limit (path side) | 13 | 13 | 3/13 | 0/13 | 3/13 |

Scene-clustered bootstrap (3 clusters — descriptive, not a 95 % interval): removal 10.5 %
[0, 33.3], mirror 6.2 % [0, 25.0], scale 0 %. The detectors differ in kind: B_coh catches
removals (a level camera's rate has no component along the required one) and cannot see mirrors;
B_abs sees neither at this threshold. B_rel is the same as B_abs because its threshold sits at
1.22 (> 1: a residual larger than the required rate itself).

**Budget curve (B_abs, feasible flags from 80):**

| flagged feasible | 2 | 4 | 6 | 8 | 10 |
| --- | --- | --- | --- | --- | --- |
| threshold (deg/s) | 50.1 | 40.2 | 28.1 | 24.6 | 23.3 |
| removal | 0/19 | 2/19 | 8/19 | 10/19 | 10/19 |
| mirror | 0/16 | 1/16 | 3/16 | 3/16 | 4/16 |
| scale | 0/22 | 0/22 | 0/22 | 1/22 | 3/22 |
| partial | 0/12 | 0/12 | 2/12 | 3/12 | 4/12 |

Removal detection only becomes real above a 7 % false-reject, and the threshold falls from 40 to
23 deg/s — i.e. the whole curve is being dragged by the same handful of high-mount clips.

### Partial intervals

At the pre-registered point: 2/8 hit at 10 % coverage, 1/6 at 25 %, 2/6 at 50 %; median IoU 0.0
in every cell (the flagged windows are elsewhere in the clip). This is worse than G06's residual
scan (0/5 at 25 %, 1/5 at 50 %, IoU 0.15-0.24) and is a consequence of the threshold, not of the
window rule.

### Controls

| control | result |
| --- | --- |
| yaw about the thrust axis (20° amplitude, 2 s period), feasible clips | 4/57 flagged — exactly the feasible rate |
| null edit (DPVO draw 2 on the same feasible clips) | 2/57 flagged |
| camera-roll yaw (about the optical axis), exploratory | 3/4 (0°), 17/17 (15°), 9/16 (30°), 5/20 (45°) |

The yaw control passes: a rotation about the thrust axis is invisible, as it must be. The
camera-roll variant is visible at every mount with an uptilt, which is geometrically correct (a
rigid camera turned about its optical axis moves the body) and is why the image-roll control
from Track E cannot be reused here.

### Real flights (leave-one-sequence-out)

| sequence | windows | threshold (deg/s) | window false-reject | clip score | rejected |
| --- | --- | --- | --- | --- | --- |
| outdoor_forward_1 | 107 | 43.5 | 0/107 | 3.63 | no |
| outdoor_forward_3 | 109 | 43.6 | 0/109 | 4.60 | no |
| outdoor_forward_5 | 54 | 31.6 | 1/54 (1.9 %) | 6.76 | no |
| outdoor_45_1 | 58 | 10.9 | **54/58 (93.1 %)** | 25.72 | **yes** |

Three of four sequences pass at 0-1.9 %. The 45°-down rig is rejected wholesale — the same
sequence whose reader is broken in E02 and whose rotation drifts in N01. Among the ten indoor
sequences (descriptive) the clip scores run 6.3-22.3 deg/s, i.e. the same spread driven by the
extractor. **The threshold itself is not portable**: the LOO window threshold is 31-44 deg/s
because the other sequences include the broken ones.

### Generated clips (SolarWM, exploratory, never used for a threshold)

| arm | n | testable | median clip score (deg/s) | flagged at the sim 4/80 (40.2 deg/s) |
| --- | --- | --- | --- | --- |
| A flatness | 78 | 78 | 14.2 | 2/78 |
| B level | 78 | 78 | 25.9 | 4/78 |
| C mirrored roll | 78 | 78 | 22.2 | 7/78 |

The check does not separate the arms: the physically consistent arm A already sits at 14 deg/s,
which is 5-14× the sim low-mount residual, because SolarWM's own rotation drifts (N04: 6.7° at
5 s). Whatever the generator does to the attitude, the video's rotation is not accurate enough
for this statistic to see it at 81 frames. **This is a negative result for the generator use
case**, and it was not tuned.

### Exploratory (post hoc, not in the pre-registration)

1. **Within-mount thresholds** (5 % of that mount's feasible clips): mount 0 (4 testable) threshold
   1.27 deg/s, mirror 0/3, scale 8/8, no testable removal; mount 15 threshold 5.95 deg/s,
   false-reject 1/17, removal 6/6, mirror 4/7, scale 8/8. Pooled 0+15: threshold 5.95, false-reject 1/21
   (4.8 %), removal 6/6, mirror 4/10, scale 16/16. Mount 30: threshold 28.1, removal 3/7, mirror
   2/6, scale 0/6. Mount 45: threshold 51.9, all 0. **This is the check's real operating
   characteristic**: it works wherever the extractor does.
2. **A per-clip rate-offset tier** (subtract the mean 3-DoF residual over the clip, the C2
   analogue) does not help: the offset is small compared with the drift contamination and the
   matched threshold barely moves (13.6 vs 13.6 deg/s on `i02_dev_003`).
3. **Lag** of the cross-correlation between `|rb|` and `|rd|`: sim −0.03 s (p10/p90 −0.50/+0.37),
   UZH +0.03 s (−0.02/+0.03), generated −0.06 s (−0.25/+0.25), TII +0.27 s (0.07/0.43, the
   rotation noise). No attitude-loop lag is resolvable on real flight; the UZH spread of 0.05 s is
   the only tight one.

### Criteria (pre-registered)

| criterion | result |
| --- | --- |
| C1 real false-reject ≤ 10 % per held-out sequence | **FAIL** — outdoor_45_1 at 93 %; the other three pass at 0-1.9 % |
| C2 ≥ 50 % of testable dynamic whole-clip camera defects | **FAIL** — 3/53 (6 %) |
| C3 ≥ 25 % of testable dynamic mirrors | **FAIL** — 1/16 (6 %) |
| C4 yaw control and null edit ≤ 4/80 | **PASS** — 4/57 and 2/57 |

## What this supports

* The rate statistic is a working check *conditional on a stable rotation estimate*: at the
  0°/15° mounts it detects every tilt removal and every scale defect at a 4.8 % false-reject, and
  the three outdoor forward UZH sequences are not rejected by a leave-one-out threshold.
* Removals and scale defects are the check's signal; mirrors are not, even with a stable
  extractor (a rigid rotation of a planar lean trajectory is a mirror's best disguise, exactly
  the mechanism that lets C2 absorb constant-bank removals).
* The high-mount DPVO drift is a first-order limitation of *any* rate check that needs the world
  attitude: it rotates the rate vector without changing its magnitude.

## What it does not show

* No claim about held-out sim clips or held-out TII flights.
* No claim that the check fails on real 45°-down footage for a physics reason: the residual there
  is dominated by the extractor.
* No generator-level verdict: SolarWM's own rotation noise buries the arm differences; ANWM and
  LingBot have no cached rotations.

## Caveats

* Thresholds come from the simulated feasible clips and are dominated by 4 of 80 clips. The
  real-calibrated column is not better: τ = 30.1 deg/s pooled over the four outdoor sequences
  gives 4/57 feasible flags and 6/19 removals — the same numbers, because the broken sequence is
  in the pool.
* The gauge fit is 3 DoF per clip; on a nearly planar lean trajectory it is ill-conditioned,
  which is measured (the fit residual is reported per clip) rather than removed.
* The camera-roll control shows the statistic is not invariant to image roll under a tilted
  mount; that is geometry, not a bug, but it means a generator that yaws the image will be
  flagged with a magnitude that depends on the mount.
