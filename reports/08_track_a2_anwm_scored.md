# Report 08 — Scoring ANWM's generated flights (Track A2, Task 3)

Track A2. Written 2026-11-09. Pre-registration: amendment 3 (ANWM protocol, committed
`ea1dce4` before any generated clip was scored) and the frozen gate
`configs/gate_frozen.yaml` (commit `9e1656d`). Runs: `results/a2_anwm/<run>/` (required tilt
from the commanded paths), `results/a2_anwm_score/<run>/` (scoring, controls, dt sweep).
Extraction rows: `results/gate_a2/{anwm,floor,edits}/`. Track B's files were read, never
modified.

## Plain-language summary

**What ANWM depicts.** On all 82 generated clips the estimator reads a tilt of about
**0.8°** — essentially level — with a cruise-control band of 0.50–2.29° and a seed-repeat
band of 0.50–2.66°. That is 0.2–0.3° above the 0.57° it reads on released level frames, so
the generator adds almost no tilt of its own and does not remove what it was given.

**What the commanded paths demand.** At the paper-consistent 1 s per step, with a threshold
calibrated in the same mode as the residual (drag-free, lag-only, 14.73°; detection floor
15.6°), the noise-free part of the claim is narrow: the **designed-20° family is infeasible
in every mode** (5 of 5), the **designed-10° family is not robust** (feasible drag-free,
infeasible with `k_d` 0.295), and **every testable Regime A clip is infeasible** (8 of 8,
34.7°) but only indicatively. All previous counts at a 6.46° cut are withdrawn as
threshold/mode mismatches.

That is a much smaller population than an earlier draft of this report claimed. The
"smooth-flight lower bound" behind the earlier count (22 testable, 17 infeasible) was not a
bound: on plateau-dominated paths a spline's median tilt exceeds the per-step median on
345 of 492 rows, and its peak exceeds the per-step peak on 422 of 492. Amendment 5 records
the error and its consequence; the corrected numbers are the ones quoted here, and the old
statistic is kept below as a sensitivity.

**The designed-control check is the sharpest result.** Regime C was built with 5°, 10° and
20° of required tilt on purpose. Depicted tilt is **0.79°, 0.78°, 0.77°** — flat. The
generator does not bank more when the path demands more; if anything the 20° family is
marginally flatter. Here the population is large enough to be decisive: 11 of 49 Regime C
clips pass the testability rule and **all 11 are infeasible**, with median residual 9.6°
(spline) to 12.3° (per-step).

**The minimum consistent dt is 1.2–2.8 s, depending on the clip.** For each testable clip
there is a time step at which the commanded path stops demanding more tilt than the pixels
show: median **1.55 s** for the designed maneuvers (11 clips) and **2.7 s** for the testable
Regime A clips (9 clips) with drag. At the step spacing those paths were built with, that
implies a drone moving at ~3.2 m/s (C) and ~1.4 m/s (A) — against the ~5 m/s the release
was trained on. The generated video is the picture of a drone flying the commanded path
**more slowly than the path's own step spacing implies**.

## 0. Positive controls — run first, and they decide whether anything above counts

"Reads about 0°" means nothing if the estimator is blind at 224×224.

### a. Noise floor on generated frames

| clip | depicted median | p90 |
| --- | --- | --- |
| C_cruise_ctl_course_0 | 0.50° | 1.51° |
| C_cruise_ctl_course_0_s1 | 0.50° | 0.80° |
| C_cruise_ctl_course_0_s2 | 0.66° | 1.50° |
| C_cruise_ctl_course_1 | 2.29° | 5.57° |
| C_cruise_ctl_course_1_s1 | 2.24° | 4.64° |
| C_cruise_ctl_course_1_s2 | 2.18° | 4.18° |
| C_cruise_ctl_course_2 | 0.73° | 1.53° |

Pooled over the 112 cruise frames: **median 0.89°, p99 6.70°, max 6.75°**. Median of the
per-clip medians 0.73°; seed repeats 0.79°; the released-level frames give median 0.58° and
**p99 4.17°**. Two things follow. One path (`course_1` and its seeds) sits systematically at
2.2–2.3°, so the generator's own bias depends on the path — that is why the cruise band, not
a single number, is the comparison. And the generated-frame tail is heavier than the
ground-truth floor: its p99 (6.70°) is above the 6.46° verdict threshold, where the
ground-truth p99 (4.17°) is not. A single generated frame at the 99th percentile is therefore
*not* distinguishable from a 6-degree error by this estimator, which is why every clipping
claim below is a clip-level median and why 12 excitation frames are required.

### b. Edit recovery — can the estimator see tilt in these frames?

Twelve clips, pure camera roll and pitch of 5/10/20°, pinhole remap at f = 112 px, same
GeoCalib commit.

| edit | applied | recovered | median absolute error | p90 |
| --- | --- | --- | --- | --- |
| roll | 5° | 3.27° | 1.73° | 2.96° |
| roll | 10° | 8.70° | 1.30° | 2.24° |
| roll | 20° | 19.42° | 0.58° | 1.85° |
| pitch | 5° | 2.48° | 2.52° | 4.54° |
| pitch | 10° | 8.26° | 1.74° | 3.45° |
| pitch | 20° | 19.15° | 0.85° | 2.18° |

**Stop rule (pre-registered): median error ≤ 3° at 10°, ≤ 5° at 20°.** Measured 1.74° and
0.85° — **passes**. Generated frames at 224 px do carry visible tilt, with a systematic
~15 % underestimate at small angles (the estimator's own compression toward level).

*Correction recorded in `docs/decisions.md`:* the first version of this control rotated
about the camera y axis for "pitch", which is a yaw; it reported ~0° recovery. The axis
mapping is now pinned by a unit test.

### c. Gate transfer

Frozen gate (S3, tau = −0.00167) applied to 16-frame groups of the 186 released level
frames: **0 of 10 dropped**. Applied to the generated clips: **7 of 82 dropped** (8.5 %).
The gate is transferable to 224 px generated frames — the condition for interpreting
anything below.

### d. Cross-check against Track B's own audit

Track B's `clips_index.csv` columns `required_tilt_1s_*` are the **maximum over per-step
tilts**. Reproduced by this round's per-step estimator to 0.003° (drag-free) and 1.0° (with
drag); read as a median instead, 22–24 of 82 clips move by more than 10°. Track A's own
`anwm_specs/clips_v1/specs.json` contains the same mismatch (a −20 to −23° "delta") left
unexplained. Every table below names its statistic; the median is primary because the
verifier's clip score is a median.

## 1. The comparison, and the rule for calling a path infeasible

**Corrected rule (amendment 5, the primary).** Required tilt is evaluated frame by frame at
the same times as the depicted tilt — frame *i* is the depiction after the step ending at
waypoint *i+1*, so 16 frames from 17 waypoints — both through the verifier's own flatness
expression \(z_B \propto a + g e_3 + k_d v\). The residual is \(|\text{required} -
\text{depicted}|\) per frame and the clip statistic is its median, exactly as the verdict
scores a clip. A clip is **testable** when the gate keeps it, the rollout has not collapsed,
and at least 12 frames carry a required tilt ≥ 10° (the protocol's excitation rule). It is
**infeasible** when that median residual exceeds the frozen verdict threshold (6.46° at
`k_d` = 0.295).

Two interpolants are reported for every table: **spline** (natural cubic through the
waypoints) and **step** (the acceleration that produced each frame). Where they disagree the
disagreement is the interpolation ambiguity and is stated.

**Retracted rule (amendment 3).** The earlier draft compared the *median over an
interpolant's samples* against the *median depicted tilt*, and called the spline "the least
tilt any flight could need". That last claim is false and a unit test caught it:

- the spline's median exceeds the per-step median on **345 of 492** clip/dt/`k_d` rows,
  because repeated waypoints make most per-step accelerations exactly zero;
- its peak exceeds the per-step peak on **422 of 492** rows (natural cubic overshoot);
- a natural spline minimises ∫|a|² *within C² interpolants*; a flight may have
  discontinuous acceleration, so it is not an admissible lower bound.

The old tables are still produced and appear as §6. They are labelled sensitivity, not
bound.

Testability bookkeeping under the corrected rule: gate keeps 23 of 25 Regime A, 5 of 8
Regime B and 47 of 49 Regime C clips; nothing is collapsed; **A15 and all eight Regime B
clips are not testable** (no frame reaches a 10° requirement), as pre-registered.

## 2. Regime A — indicative, not rigorous

**Headline, restated (amendment 6): there is no rigorous violation at 1 s; the spline
results are indicative.** The only interpolation-free statement available is the peak bound
of §3b, and at dt = 1 s it is satisfied for every Regime A clip (thrust-to-weight lower
bound ≤ 0.51). The frame-level residuals here depend on how the waypoints are connected, and
the release does not specify that.

| `k_d` | interpolant | gate-kept | testable | untestable (excitation only) | infeasible | median residual (testable) |
| --- | --- | --- | --- | --- | --- | --- |
| 0.295 | spline | 23 | 9 | 14 | 9 | 39.4° (28.6° with parity fits) |
| 0.295 | step | 23 | 2 | 21 | 2 | 21.0° (13.5°) |
| 0 | spline | 23 | 8 | 15 | 8 | 38.3° |
| 0 | step | 23 | 2 | 21 | 2 | 24.5° |

Both interpolants put the testable clips 13–29° above the 6.46° threshold even after the
verdict's own nuisance fits. The magnitude is barely sensitive to the interpolator, but the
*existence* of a required tilt at a given frame is an interpolation choice, so the finding is
indicative, not rigorous.

**Spline end-condition sensitivity** (testable / infeasible at `k_d` 0.295): natural 9/9,
not-a-knot 10/10, clamped 10/10; drag-free 8/8, 8/8, 9/9. Median residual of the testable
clips is 38.3–39.4° in every variant.

**The testable clips** (spline, `k_d` 0.295): A03, A04, A05, A07, A10, A11, A13, A14, A17 —
step spacing 2.5–4.8 m, 12–15 excitation frames, required median 17.5–42.2° (max
55.8–74.0°), depicted 0.40–1.10°, residual 17.1–40.8°. The other 14 Regime A clips are
plateau-dominated and cannot be tested at 1 s.

## 2b. Nuisance modes: parity, lag-only, and what the ±7° grid absorbs

The TII verdict fits a mount uptilt offset (±7°, step 1°) and a lag (±150 ms, step 25 ms).
Applying the same grid to generated frames needs a decision, because ANWM's camera is
exactly level: the ±7° grid is *TII's* mount uncertainty, and on this render it can absorb a
real violation. Three modes are therefore reported (amendment 8):

- **none** — nothing fitted;
- **lag-only** — lag fitted, mount fixed at its known 0°;
- **full parity** — both fitted, as on TII.

**Straight-cruise null (7 clips, no designed violation):**

| `k_d` | none | lag-only | full parity |
| --- | --- | --- | --- |
| 0 | 0.73° | 0.73° | 0.65–0.68° |
| 0.295 | **7.8°** | 7.8° | 1.2–1.4° |

The lag absorbs **0.0°** on generated clips — at 1 s per step, 150 ms is a sixth of a frame.
The uptilt grid absorbs the rest. Under `k_d` = 0.295 the unfitted null is 7.8°, so the
"parity" improvement there is the fit repairing a disagreement between the racing-quad drag
model and the render, not repairing a violation. Drag-free the null needs no repair at all,
which is why amendment 8 makes **drag-free + lag-only** primary.

**What the ±7° grid can absorb, measured** (`k_d` 0.295, spline; median over clips):

| cell | none | lag-only | full parity | absorbed by uptilt | clips at the ±7° bound |
| --- | --- | --- | --- | --- | --- |
| Regime A | 39.4° | 34.7° | 28.6° | **6.4°** | 0.72 |
| Regime C | 12.6° | 12.6° | 6.0° | **6.4°** | **0.92** |

Drag-free the absorption is 3.4° (C) and 5.0° (A). So the grid costs about 6.4° of
sensitivity with drag and 3.4° drag-free, and on 92 % of Regime C clips it is saturating at
the bound — the violation is larger than the grid can hide.

**Consequence for the verdicts** (testable clips, drag-free, spline):

| cell | none / lag-only | full parity |
| --- | --- | --- |
| Regime A | 8 of 8 infeasible | 8 of 8 |
| Regime C, designed 10° | **6 of 6** | 0 of 6 |
| Regime C, designed 20° | **5 of 5** | 5 of 5 |
| Regime C, total | **11 of 11** | 5 of 11 |

The headline is the left column (drag-free, no uptilt absorption). The right column is the
conservative bound and is quoted whenever the finding is repeated.


## 3. Regime C — dose-response, with mode-consistent thresholds

**Drag-free + lag-only is primary** (amendment 8). Thresholds are now calibrated in the same
mode as the residual they judge (amendment 9). The 6.46° cut used earlier is TII's value at
`k_d` 0.295 with both nuisances fitted; applying it elsewhere mixed thresholds with
residuals and is withdrawn.

**Thresholds and detection floors, per mode** (TII conformal refit with the gate applied,
pooled over all eight flights, fold spread in brackets; ANWM nulls from the cruise clips and
the 186 released level frames):

| mode | TII conformal | folds | cruise p95 / p99 | GT p99 | primary | detection floor |
| --- | --- | --- | --- | --- | --- | --- |
| **drag-free, lag-only** | **14.73°** | 13.5–16.5 | 4.78 / 6.70 | 4.17 | **14.73°** | **15.6°** |
| drag-free, full parity | 12.95° | 10.6–13.8 | 4.78 / 6.70 | 4.17 | 12.95° | 13.8° |
| `k_d` 0.295, lag-only | 8.96° | 9.0–12.1 | 8.32 / 8.48 | 4.17 | 8.96° | 16.6° |
| `k_d` 0.295, full parity | 6.46° | 5.8–10.1 | 8.32 / 8.48 | 4.17 | **8.48°** (cruise p99 wins) | 16.1° |

The detection floor is the primary threshold plus the median cruise residual in that mode:
**14–17° of required tilt in every mode**, which is the same order as TII's own frontier.

**Verdicts by mode, per interpolant** (testable clips; cells are `infeasible / n`):

| cell | drag-free, lag-only (14.73°) | `k_d` 0.295, lag-only (8.96°) | drag-free, full parity (12.95°) | `k_d` 0.295, full parity (8.48°) |
| --- | --- | --- | --- | --- |
| Regime A | **8/8** spline, 2/2 step | 9/9 spline, 2/2 step | 8/8, 2/2 | 9/9, 2/2 |
| C, 5° | 0/0 (not testable) | 0/0 | 0/0 | 0/0 |
| C, 10° | **0/6**, 0/6 | **6/6**, 6/6 | 0/6, 0/6 | 0/6, 0/6 |
| C, 20° | **5/5**, 5/5 | **5/5**, 5/5 | **5/5**, 5/5 | **5/5**, 5/5 |

**Plainly: the 20° cell is robust — every mode, every interpolant, 5 of 5 infeasible. The 10°
cell is not: it is infeasible only at `k_d` 0.295 (11.6° residual against 8.96°) and feasible
at the drag-free primary (9.2° against 14.73°).** The earlier "6/6 at 6.46°" reading is
exactly the mismatched threshold.

Dose and slope are unchanged by mode: depicted **+0.040 °/°** against required **+0.49 to
+0.66 °/°**, seed spread ≤ 0.60°.


## 3b. Rigorous peak bounds (interpolator-free, attitude-free)

Everything above depends on an interpolation choice and on the depicted tilt. This check
depends on neither. For any **continuous** trajectory through the sampled positions, with
step dt and mean step velocities v_i,

    v_{i+1} - v_i = (1/dt) ∫ [v(s+dt) - v(s)] ds = (1/dt) ∫∫ a ,

so the peak acceleration over the two-step window obeys **|a|_peak ≥ |v_{i+1} − v_i| / dt**
(the form |Δv|/(2dt) is the same statement, a factor two weaker). Projecting onto the
horizontal plane gives the same bound for |a_h|, and since u = a + g e₃ + k_d v with
u_h = a_h + k_d v_h,

    |u|_peak / g  ≥  max( |Δv_h|/dt − k_d |v_h| ) / g

which is a **thrust-to-weight lower bound** that needs no assumption about attitude, no
cap on the airframe and no depicted tilt. `src/physics/peaks.py`, tested in
`tests/test_peaks.py`.

| dt | `k_d` | regime | peak \|a\| (median of per-clip maxima) | thrust/weight lower bound | clips above T/W 1 / 2 / 3 |
| --- | --- | --- | --- | --- | --- |
| 0.25 s | 0.295 | A | 86.2 m/s² | **7.85** | 0.96 / 0.96 / **0.96** |
| 0.25 s | 0.295 | B | 2.9 | 0.05 | 0 / 0 / 0 |
| 0.25 s | 0.295 | C | 18.2 | 0.80 | 0.49 / 0.43 / 0.18 |
| 0.667 s | 0.295 | A | 12.1 | 1.03 | 0.96 / 0 / 0 |
| 0.667 s | 0.295 | C | 2.6 | 0.00 | 0 / 0 / 0 |
| **1.0 s** | 0.295 | A | 5.4 | 0.43 | 0 / 0 / 0 |
| **1.0 s** | 0.295 | C | 1.1 | 0.00 | 0 / 0 / 0 |

Drag-free values are within 4 % of these (the drag term only lowers the bound).

Read as a physical statement: **at dt = 0.25 s, 24 of 25 Regime A commanded paths require a
thrust-to-weight ratio above 3 somewhere in a two-step window** — beyond any racing
quadrotor — without reference to a single pixel. At dt = 1 s the same bound is satisfied
everywhere (≤ 0.51 drag-free), so this check does not add a violation at the nominal time
base; it rules out the fine time bases outright and independently corroborates that the time
base is the axis that matters.

A tilt version is available but needs an explicit cap on vertical acceleration, because a
large acceleration pointing straight up produces little tilt:
`tilt_lower_bound_deg(|a_h|, a_z_max)`. With a_z ≤ 2 g (an assumption, not a measurement) the
dt = 0.25 s Regime A bound is a median peak tilt of 69.8°; that number inherits the cap and is
reported as a conditional, not as a headline.

## 4. Minimum consistent dt — per cell, per mode, conditional on testability

Conditional on being testable at 1 s; untestable clips are censored, not dropped. Spline
interpolant unless stated.

| cell | testable at 1 s | no fit / lag-only | full parity |
| --- | --- | --- | --- |
| Regime A, drag-free | 8 of 23 | 1.25 s | 1.00 s |
| Regime A, `k_d` 0.295 | 9 of 23 | 1.40 s | 0.90 s |
| Regime C, 5° | 0 of 21 | not interpretable | not interpretable |
| Regime C, 10°, drag-free | 6 of 11 | **1.10 s** | 0.85 s |
| Regime C, 20°, drag-free | 5 of 8 | **1.58 s** | 1.23 s |
| Regime C, 20°, `k_d` 0.295 | 5 of 8 | 1.67 s | 1.30 s |

The aggregate medians (0.6–0.85 s) are artifacts of the untestable 5° family. The defensible
statement is per cell: **the designed-20° family becomes consistent at 1.2–1.7 s**, above the
1 s the release's stated speeds imply.


## 5. What this supports

- **One robust cell, one not.** With a threshold calibrated in the same mode (amendment 9),
  the **designed-20° family is infeasible in every mode and both interpolants** (5 of 5, at
  12.9–21.2° against a 14.73° primary threshold). The **designed-10° family is not robust**:
  infeasible only at `k_d` 0.295 (11.6° against 8.96°), feasible at the drag-free primary
  (9.2° against 14.73°). The earlier "6/6" came from the mismatched 6.46° cut.
- **Regime A is indicative**, not rigorous: 8 of 8 testable clips exceed the threshold
  (34.7° lag-only), but no peak-bound violation exists at 1 s, and the residual depends on
  the interpolant.
- **A dose-response failure that cannot be explained by noise.** Regime C's designed 5/10/20°
  families are depicted at 0.79/0.78/0.77° with a seed band of at most ±1.3° (p10–p90 of the
  flat 5° family is 0.53–2.47°, still far below 5°).
- **A quantified time-base explanation** rather than a bare "breaks physics", but one that
  is conditional on testability at 1 s: the same clips stop contradicting their paths at
  dt ≈ 1.55 s (Regime C) to 2.7 s (Regime A), and 36 of 47 Regime C clips are untestable at
  1 s and censored in that accounting.

## 6. What this does not show

- **Not** that ANWM "cannot fly". It shows the depicted tilt is inconsistent with the
  commanded paths at the stated time base, under a specific gate, interpolator and drag.
- **Not** a claim about Regime B: 0 of 8 clips is testable at 1 s, so its 2.2–8.7° commanded
  tilt is below what the estimator can resolve. Reported as not testable, not as consistent.
- **Not** independent of the gate: 7 clips are dropped by it, and the gate was fitted on TII
  footage. The transfer check is the only evidence it belongs here.
- **Not** an endorsement of the max statistic: with the maximum over steps (Track B's
  column) the same clips look far worse (up to 36° at 1 s). The median is the conservative
  choice and is what the tables above report.
- **Not** a test of ANWM's own time convention: the release defines no seconds. dt = 1 s is
  the paper-consistent reading, and every table is an axis over dt rather than a claim at one
  value.
- **Not** a validated speed measurement: dt_min is a consistency bound, not an estimate of
  how fast the model "thinks" it is flying.

## 7. Caveats

- Regime A's commanded paths contain many repeated waypoints (stalls). The per-step
  estimator's peak on those clips reaches 55° at 0.25 s while its median is 0°, which is the
  reason the excitation rule exists and the reason the testable population is 2–9 clips
  rather than 25. Neither interpolant is a bound (amendment 5); the frame-level residual is
  the statistic used.
- The estimator's ~15 % compression toward level (control b) means depicted tilt is, if
  anything, slightly *under*-reported — which makes the infeasibility finding conservative.
- Regime C's "designed" tilt is a property of the Track B generator, not of the recorded
  path; the required-tilt column is recomputed here from the commanded positions, and the two
  differ (designed 5° → 9.4–9.5° required with drag) because the design targeted a peak and
  the recomputation starts from positions.
- No clip was "collapsed" by the pre-registered frame-statistic rule (minimum median
  frame-to-frame difference over the set is 0.035 against a 0.02 floor).

## 8. Cross-check across the three tasks

What the three reports together support, and what they do not.

**Supported.**

1. *A real flights' tilt can be read from pixels.* Depicted-vs-mocap error has median
   2.6–4.3° on the autonomous non-racing flights and 0.57° on released level frames, and the
   estimator recovers known 10–20° camera rotations in **generated** frames to 0.9–1.7°
   median error (report 08, control b).
2. *A pixel-only gate can tell when that reading cannot be trusted.* AUROC 0.946
   [0.884, 1.000] against mocap error, dropping all eight clips whose error exceeds 10°
   (report 06).
3. *With that gate, the verifier accepts well-behaved real footage at 0.949* and ranks
   tilt-removed footage above the real clip at AUROC 0.92 (report 07).
4. *ANWM's generated flights depict a level aircraft while their commanded paths demand
   banking.* The designed-20° family is depicted at 0.77° where its paths require 21.7°
   (drag-free), and it is infeasible in three of four modes at each mode's own threshold
   (5 of 5 at the primary; 0 of 5 for the step reading under drag-free full parity, where the
   ±7° uptilt grid absorbs the difference). The designed-10° family is not robust, the
   Regime A result is indicative (no peak-bound violation at 1 s), and the weakest 20° margin
   is **+0.25°** at the strictest fold, with **−2.75°** in the most conservative mode.
5. *The time base explains most of it.* Conditional on 1 s testability the same clips stop
   contradicting their paths at dt ≈ 1.55 s (Regime C) to 2.7 s (Regime A), longer than the
   1 s the release's stated speeds imply. At 0.25 s the rigorous peak check is violated
   outright.

**Not supported.**

1. *"A pass means the flight is consistent with its path."* At the drag-free primary the TII
   verdict rejects a full tilt removal only about half the time (0.50 on 8 testable pairs), so
   a pass is "no gross violation", not consistency.
2. *"The verifier reaches its gate-3 targets."* Acceptance 0.949 is one clip short of 0.95
   on 59 clips, and no violation magnitude reaches 80 % rejection. On the pre-registered
   per-condition gate, the small violations are rejected at 0.11 and 0.07.
3. *"The gate is free."* It abstains on 41 % of clips, 86 % on the racing flights, and on
   79 % of tilt-removal edits. Any acceptance number is an acceptance number on what it
   keeps.
4. *"ANWM breaks physics."* Without naming dt, `k_d`, the interpolator, the nuisances and
   the gate, the statement is not supported. With them named, the supported form is (4) —
   and it holds for the designed-20 family and the testable Regime A clips, a minority of
   the set, not for the whole of it.
5. *"Regime B is fine."* 0 of 8 Regime B clips are testable; nothing is known about them.
6. *"The path can be recovered from video."* Parked and untouched; nothing here depends on
   it, because the path is commanded in every number above.

**Internal consistency checks that passed.** The Gate 3 baseline reproduces through the
extended code (report 07 §1); the required-tilt estimator reproduces Track B's per-step
numbers to 0.003° once the statistic is matched; the depicted-tilt construction reproduces
the published 0.57° floor; the lower bound never exceeds the upper bound; A15 and Regime B
are not testable, as predicted before scoring; the gate drops 0 of 10 ground-truth clips.

## 9. Retained for comparison: the amendment-3 statistic (sensitivity, not a bound)

The pre-registered amendment-3 tables, computed as median-of-interpolant against median
depicted tilt, are in `results/a2_anwm_score/<run>/summary.json` (`regime_A_share`,
`regime_C_dose_response`) and are summarised here only so the change is auditable.

| statistic | Regime A testable | Regime A infeasible | Regime C testable | Regime C infeasible |
| --- | --- | --- | --- | --- |
| spline median vs depicted median (amendment 3) | 22 | 17 | 21–42 | 19–42 |
| frame-level median residual (amendment 5) | 2–9 | 2–9 | 11 | 11 |

The amendment-3 rule inflated Regime A testability from 2–9 clips to 22 and its infeasible
count from 2–9 to 14–17. It also made the 5° Regime C family look testable with drag (21
clips) when the excitation rule says those paths never demand 10° at 1 s. The direction of
the finding is unchanged; the prevalence in the earlier draft was overstated and is
withdrawn.


## 10. Proposed pose-injection arm (design only; nothing run)

The decisive test is not more tilt levels — it is whether the generator responds to the pose
channel at all. The arm below is the minimal experiment that answers that, and it is ordered
first because its outcome decides whether the larger design is worth buying.

**The arm.** Take the **existing 49 Regime C clips** (their paths, starts and seeds already
exist) and generate each one **twice**:

- **level arm** — the existing clips serve as the level-pose baseline, same start, same seed,
  same path, same sampling steps;
- **injected arm** — the same path generated with flatness-derived roll and pitch injected
  into the *target poses*: for the commanded acceleration at each step, set
  `z_B = (a + g e₃)/‖a + g e₃‖` and use the roll and pitch that implies. This is a different
  factor from tilt level: the path is unchanged and only the target attitude differs.

Pairs are `(clip, start, seed)`, so every injected clip has a level twin generated under
identical conditions. 49 new clips × 235 s = 3.2 GPU-h ≈ **US$3.5** at the unverified
US$1.10/h — under the US$5 line, no approval gate. A mirrored-sign control (inject −θ) on 15
clips adds ≈ US$1.1, total ≈ **US$4.6**, still under the line.

**Why not the 144-clip design first.** If injection moves the depicted slope, adding level
clips buys a sharper estimate of a slope that already exists; if it does not move, no number
of extra level clips will make the generator honour the pose channel. The 4-tilt-level design
is therefore deferred until this arm reports.

**Scoring plan for the injected arm** (written before generation; to be frozen as an
amendment when Track B confirms the format):

1. **Controls first, and they can void the run.**
   - *null edit*: identity remap of the injected frames vs the raw frames — must agree to
     ≤0.2° median, as it did on the level set (0.006°).
   - *edit recovery*: known camera rotations of 5/10/20° on ≥10 injected clips; if the
     median error at 10° exceeds 3° or at 20° exceeds 5° (the existing rule), the arm is
     reported as "estimator not validated on the injected distribution" and nothing else is
     claimed. Injected views are out of the level distribution and may be harder.
   - *gate transfer*: the frozen gate applied to the injected frames; if it drops >50 % of
     them, the gate is declared non-transferable to this distribution and the arm reports
     only the controls.
2. **Depicted vs required, paired.** For each pair compute `depicted_median` (drag-free,
   lag-only, per amendment 8) and the required tilt from the same path. Report, per designed
   tilt: the paired difference (injected − level) with a paired sign test and exact binomial
   interval, and the slope of depicted on required across the injected clips. A generator
   that honours the pose channel gives a positive slope near 1; the level set gives +0.04
   against a required slope of +0.5–0.66.
3. **Mirrored-sign control.** On the 15 mirrored clips the injected tilt has the opposite
   sign; the depicted change must flip with it. If the depicted change is unchanged under
   mirroring, the effect is a rendering artefact of the injection, not a response to the
   commanded attitude.
4. **Same testability rule**, with the secondary excitation rule below, and the peak bound as
   the interpolation-free cross-check (it is unchanged by injection: the path is the same).

**Secondary excitation rule (pre-registered for this arm; primary unchanged).** Primary stays
≥12 of 16 frames with required ≥10°. Secondary: **≥8 of 16 frames with required ≥10° and max
required ≥0.8 × designed tilt**, with the clip statistic taken as the median residual over the
excited frames only. Rationale: a maneuver that holds its tilt for 8–11 steps is a real test
the primary discards, while a two-step pulse is not. Both are reported on every table.

**Cost summary.**

| item | clips | time | cost |
| --- | --- | --- | --- |
| injected arm | 49 | 3.2 GPU-h | ~US$3.5 |
| mirrored-sign control | 15 | 1.0 GPU-h | ~US$1.1 |
| scoring + controls (Track A) | 64 × 16 frames | 0.15 h | ~US$0.2 |
| **total** | | **~4.4 GPU-h** | **≈US$4.8** |
