# P0 — Error budget for the body-rate check, and the stop-rule verdict

**Decision.** The budget **passes**: at 0.5 s windows the minimum detectable thrust-axis rate
error (95th percentile at a 5 % false-reject) is **3.0 deg/s** on the simulated development set
and **1.8 deg/s** on the four outdoor UZH sequences, against required rates of **9.1** and
**29 deg/s**; only **11 of the 53** dynamic whole-clip camera-defect clips (all of them mirrored
rolls, 11/16) have a defect rate error below that floor, i.e. 21 % against the 50 % stop rule, so
**Check B proceeds** (P1). The one design change forced by this part: the handoff-literal rate
form is not yaw-gauge invariant and leaves a median **8.7 deg/s** residual on ground truth that is
perfectly feasible, so the check uses a per-clip gauge fit, whose noiseless residual is **0.0008
deg/s**.

Pre-registration `configs/p0_error_budget_prereg.yaml` (committed before this run, `c61c102`).
Code `experiments/p0_error_budget.py`, `experiments/p_common.py`, `experiments/p_units.py`,
`src/verdict/rate.py`; tests `tests/test_rate.py` (10 pass). Results `results/p/p0/`. CPU only,
**zero spend**. Development data only; no held-out clip or flight was opened.

## Reading of the brief

Three readings, stated because the user is unavailable (all recorded in `docs/decisions.md`):

1. **The literal formula is replaced by a gauge fit.** `rd = (R_mount omega_cam) x b3_req` is
   under-determined: the yaw about `b3` is unconstrained, and an arbitrary frame introduced by
   `R_mount` changes the answer at first order in `|omega|` times the yaw offset. The test case in
   `tests/test_rate.py` shows a 0.63 rad/s error on the literal form and 1e-8 rad/s with the fit.
   P0 measures the literal form anyway, as the substitution error the brief asks for.
2. **Generated clips.** H04's 234 cached clips are all SolarWM; ANWM/LingBot rotations do not
   exist locally (N04: 20/80 ANWM clips usable, LingBot frames not local). Part 3 will score
   SolarWM and report the other two as not measurable.
3. **"UZH-FPV (4 sequences)"** is read as the four outdoor sequences, the only four that exist.

## Question

Before any video rotation is scored: how large is the required thrust-axis rate, how large is its
noise at the rotation quality the project has, and what is the smallest camera-attitude defect the
body-rate statistic can see at a matched 5 % false-reject? The answer decides whether Check B is
worth running at all.

## Setup

**Path side.** ``b3_req = (a + g e3 + k_d v)/|.|`` with `g = 9.80665`, `k_d = 0.295`, from the
truth position track resampled to 100 Hz and differentiated by Savitzky-Golay (0.31 s, poly 3).
The measured −3 dB point of that derivative filter is **4.05 Hz**. A 0.11 s variant changes the
median required rate by **0.0 %** (sim), **1.3 %** (UZH) and **5.9 %** (TII), so the choice of
0.31 s does not manufacture the signal.

**Both rates from one operator.** The residual is built from the *same* 3-point central
difference on the video frame grid applied to `b3_req` and to the depicted attitude; otherwise a
100 Hz Savitzky-Golay derivative and a 30 Hz finite difference differ by their bandwidth alone.
The noiseless residual on feasible ground truth is **0.0008 deg/s** median (p90 0.03) — that is
the pipeline floor, and it is three orders of magnitude below the noise floor.

**Depicted side.** Three pose sources, all measured against ground truth here:
sim 30 fps (values 0.134 deg/step median, p90 1.44), UZH 15.07 fps (0.118, p90 0.55), TII 15.0 fps
(2.56, p90 12.1 — the tracking failure N01 reported). Noise is simulated as an iid isotropic
Gaussian rotation vector per step, calibrated so the median step angle matches the measured one;
that puts all of the measured error at the highest frequency, which over-states the rate noise
(the real error is drift-dominated and cancels in a difference).

**Detectable level.** Per clip, `MDR` is the 95th percentile of the 0.5 s window statistic over
eight noise draws on the *same path* with the feasible (flatness) attitude. `delta` is the same
statistic with the labelled defect attitude and **no** noise — the rate error the defect actually
creates. The stop rule counts clips with `delta < MDR` among dynamic (peak required tilt ≥ 15 deg)
whole-clip camera defects.

## Results

### Noise, required rate, and the floor (all cells at 0.5 s)

| source | step error (deg) | required \|rb\| (deg/s) p10/50/90 | null window residual (deg/s) | MDR (deg/s) p10/50/90 |
| --- | --- | --- | --- | --- |
| sim, 200 clips | 0.134 | 0.0 / 9.1 / 42.6 | 2.20 | 2.93 / 3.00 / 3.22 |
| UZH outdoor, 4 | 0.118 | 14.9 / 29.0 / 60.4 | 1.07 | 1.42 / 1.76 / 5.19 |
| TII dev, 6 | 2.559 | 3.5 / 13.6 / 77.5 | 25.15 | 45.9 / 60.1 / 167.4 |

By sim family the median required rate is 10.5 deg/s (forest), 11.0 (industrial) and 4.5 (urban);
the floor does not depend on the family because it is set by the extractor, not the scene.
Window length barely moves the floor (sim 3.26 / 3.00 / 2.76 deg/s at 0.25 / 0.5 / 1.0 s), so the
0.5 s choice is not load-bearing. **Signal-to-noise at window level is 4.1 on sim and 27 on UZH;
on TII it is 0.5**, which is the quantitative form of "TII rotation tracking is dead".

### The substitution error the brief asks about

| form | sim noiseless residual (deg/s) | UZH noiseless residual (deg/s) |
| --- | --- | --- |
| gauge fit (used) | **0.0008** (p90 0.03) | 0.15 (p90 1.08) |
| handoff-literal, arbitrary frame | **8.74** (p90 60.4) | 0.57 (p90 3.9) |

On the simulated ground truth the literal form is wrong by a median 8.7 deg/s and up to 60 deg/s
on perfectly feasible flight — four times the noise floor at the median and 27 times at p90. The UZH cell is small only because the
reconstructed UZH null attitude uses a zero yaw, which happens to agree with the minimal frame;
it is not evidence that the literal form works on real video. The algebra in `docs/decisions.md`
[P-a] and `tests/test_rate.py` settle it.

### Defect rate error vs the floor

| defect (sim dev, 0.5 s) | n | delta median (deg/s) | MDR median | delta ≥ MDR |
| --- | --- | --- | --- | --- |
| tilt removed, dynamic | 16 | 26.4 | 3.01 | **16/16** |
| tilt removed, static | 11 | 9.0 | 2.95 | 2/11 |
| mirrored roll, dynamic | 16 | 0.65 | 3.02 | **5/16** |
| mirrored roll, static | 11 | — | 2.96 | 0/11 |
| tilt scaled, dynamic | 21 | 9.4 | 3.04 | **21/21** |
| tilt scaled, static | 12 | 3.9 | 2.97 | 1/12 |
| partial interval, dynamic | 12 | 23.9 | 3.04 | 11/12 |
| partial interval, static | 8 | — | 2.95 | 0/8 |
| thrust limit (path side) | 13 | 1.8 | 3.54 | 0/13 |

The mirrored-roll column is the one real loss: a **rigid** rotation of the lean trajectory can
undo a mirror whenever that trajectory is (close to) planar, which is exactly the case for a
steady bank or a straight acceleration. The five detectable dynamic mirrors are the manoeuvres
whose lean direction rotates on two axes. This is measured, not inferred, and P1 tests it on the
video.

### The smallest scale error

On the 53 dynamic feasible clips, the x-scale defect detects:

| \|1 − s\| | 0.1 | 0.2 | 0.3 | 0.4 | 0.5 |
| --- | --- | --- | --- | --- | --- |
| detected | 25/106 | 51/106 | 73/106 | 85/106 | 96/106 |

so a 20 % gain error is at the 50 % detection point of this check on dynamic simulated flight,
and the largest defect the set carries (x0.5 / x1.5) is seen on 91 % of clips.

### The stop rule

Population 53 dynamic whole-clip camera-defect clips. Not detectable: **11** (all mirrored roll).
Fraction 20.8 %, below the pre-registered 50 % → **do not stop**.

## What this supports

* Check B has a usable signal on simulated and on real outdoor UZH footage: the required rate is
  4-27 times the 5 % noise floor at 0.5 s windows.
* Whole-clip tilt removal and tilt scaling are comfortably above the floor on dynamic clips; the
  partial-interval defects are too, when the interval is inside a dynamic segment.
* The mirrored-roll family is a known blind spot of a rate check once the gauge is free, and the
  stop rule is not triggered because it is one family of three.
* TII cannot support this check at 15 Hz stride-8 rotations (SNR 0.5); it is reported as not
  evaluable rather than as a negative result.

## What it does not show

* Nothing about the detection rate on video: this part is ground truth plus synthetic noise.
* Nothing about static (peak required tilt < 15 deg) clips, where the required rate is under the
  floor by construction; they will be reported as untestable, not as passing.
* Nothing about the false-reject on real flights: that is P1 Part 2.

## Caveats

* The iid noise model is conservative for the high-frequency content that a difference sees, so
  the real floor may be lower than 3.0 deg/s. The measured floor is re-derived on real video in
  P1 and reported beside it.
* The gauge fit is 3 degrees of freedom per clip. It is a per-clip nuisance (the analogue of the
  C2 tier), not a start attitude, and it cannot absorb a time-localised defect.
* The static-clip `delta` cells are mostly `NaN`: the defect is uniform over the clip, the
  required rate is at the floor, and no window is testable. They are reported as untestable.
