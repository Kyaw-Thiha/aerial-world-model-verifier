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

**What the commanded paths demand.** At the paper-consistent 1 s per step, the commanded
motion requires a tilt the frames do not show. Judged the way the verdict protocol judges a
clip — per-frame required vs depicted tilt, clip statistic = median, testable only with at
least 12 frames above the 10° excitation floor — **every Regime A clip that is testable at
all is infeasible** (8–9 of 25 via the spline, 2 of 25 via the per-step estimator), with a
median residual of 21–39°, against a 6.46° threshold.

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

## 2. Regime A — what the pixels cannot support

Frame-level rule, dt = 1 s, gate applied.

| `k_d` | interpolant | kept | testable | infeasible | median residual (testable) |
| --- | --- | --- | --- | --- | --- |
| 0.295 | spline | 23 | **9** | **9** | 39.4° |
| 0.295 | step | 23 | **2** | **2** | 21.0° |
| 0 | spline | 23 | 8 | 8 | 38.3° |
| 0 | step | 23 | 2 | 2 | 24.5° |

Every testable clip is infeasible, under both interpolants and both drag settings. The
population is the caveat: the per-step rule leaves only 2 testable clips of 25 because most
Regime A commanded paths are plateau-dominated (fewer than 12 frames demand ≥ 10°), and the
spline leaves 9. At dt = 0.25 s the same paths demand more and the testable population
grows; the dt axis in §3 covers this.

## 3. Regime C — dose-response against the designed tilt

Frame-level rule, dt = 1 s. Rows are (spline / step), `k_d` = 0.295.

| designed | n | depicted median | required median | residual median | testable | infeasible |
| --- | --- | --- | --- | --- | --- | --- |
| 5° | 21 | 0.79° | 9.5° / 9.4° | 8.6° / 8.5° | 0 / 0 | 0 / 0 |
| 10° | 12 | 0.78° | 11.0° / 11.2° | 9.8° / 10.0° | 6 / 6 | **6 / 6** |
| 20° | 9 | 0.77° | 22.1° / 20.4° | 19.7° / 18.0° | 5 / 5 | **5 / 5** |

The 5° family is below the excitation floor and is reported as not testable, not as
consistent. Depicted tilt is flat across a **4× change in designed tilt** — 0.79 / 0.78 /
0.77 — with a seed-repeat band of 0.50–2.66°, so the flatness is not seed noise. All 11
testable Regime C clips are infeasible, with a median residual of 9.6–12.6°.

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

## 4. Minimum consistent dt

The smallest dt at which the frame-level median residual comes within the 6.46° threshold;
sweep 0.10–5.00 s in 0.05 s steps. Reported **only for clips that are testable at 1 s** — a
plateau-dominated clip is "consistent" at every dt because most of its frames demand no
tilt, and including it reports an unconstrained minimum (it did, first run: a median of
0.10 s at 28 m/s, which is an artifact, not a result).

| `k_d` | interpolant | regime | n | median dt_min | p10–p90 | implied speed |
| --- | --- | --- | --- | --- | --- | --- |
| 0.295 | spline | A | 9 | **2.70 s** | 2.23–2.93 | 1.4 m/s |
| 0.295 | spline | C | 11 | **1.55 s** | 1.35–2.00 | 3.2 m/s |
| 0.295 | step | A | 2 | 1.75 s | — | 1.9 m/s |
| 0.295 | step | C | 11 | 1.55 s | 1.35–1.90 | 3.2 m/s |
| 0 | spline | A | 8 | 2.78 s | 2.28–3.02 | 1.4 m/s |
| 0 | spline | C | 11 | 1.25 s | 1.15–1.80 | 4.0 m/s |
| 0 | step | C | 11 | 1.20 s | 1.10–1.70 | 4.1 m/s |

Implied speed is step spacing divided by dt_min. Read literally: at 1 s the Regime C
designed maneuvers depict a flight consistent with ~3.2 m/s rather than the 5 m/s they were
built for (drag-free ~4.0 m/s), and the testable Regime A clips need ~2.7 s (spline) — a
~1.4 m/s flight. The time base is an axis, and the release defines none; what the pixels
rule out is the 1 s reading that the paper's stated speeds imply.

The earlier amendment-3 sweep (median-of-spline vs median depicted, all kept clips) gave
1.39 s (A) and 1.24 s (C) at `k_d` 0.295; it is retained in `summary.json` as sensitivity.
Restricting to testable clips and using the frame-level statistic lengthens the required dt.

## 5. What this supports

- **A specific, bounded claim.** At dt = 1 s and `k_d` = 0.295, every clip that passes the
  verdict protocol's own testability rule has a median per-frame residual above 6.46°:
  9 of 25 Regime A clips (spline; 2 of 25 per-step) and 11 of 49 Regime C clips, with
  median residuals of 39° / 21° and 9.6–12.6° respectively. The minority that is testable
  is not what the video shows; the plateau-dominated majority is untested.
- **A dose-response failure that cannot be explained by noise.** Regime C's designed 5/10/20°
  families are depicted at 0.79/0.78/0.77° with a seed band of at most ±1.3° (p10–p90 of the
  flat 5° family is 0.53–2.47°, still far below 5°).
- **A quantified time-base explanation** rather than a bare "breaks physics": the same
  clips stop contradicting them at dt ≈ 1.2–2.8 s, depending on the clip.

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
   banking.* At dt = 1 s and `k_d` = 0.295, every clip that passes the verdict's own
   testability rule has a median per-frame residual above 6.46°: 9 of 25 Regime A clips
   (spline; 2 of 25 per-step) and 11 of 49 Regime C clips. The designed 5/10/20° controls
   are all depicted at ≈0.8° (report 08).
5. *The time base explains most of it.* The same clips stop contradicting their paths at
   dt ≈ 1.2–2.8 s, longer than the 1 s the release's stated speeds imply.

**Not supported.**

1. *"The verifier reaches its gate-3 targets."* Acceptance 0.949 is one clip short of 0.95
   on 59 clips, and no violation magnitude reaches 80 % rejection. On the pre-registered
   per-condition gate, the small violations are rejected at 0.11 and 0.07.
2. *"The gate is free."* It abstains on 41 % of clips, 86 % on the racing flights, and on
   79 % of tilt-removal edits. Any acceptance number is an acceptance number on what it
   keeps.
3. *"ANWM breaks physics."* Without naming dt, `k_d`, the interpolator and the gate, the
   statement is not supported. With them named, the supported form is the sentence in (4)
   above.
4. *"Regime B is fine."* 0 of 8 Regime B clips are testable; nothing is known about them.
5. *"The path can be recovered from video."* Parked and untouched; nothing here depends on
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

