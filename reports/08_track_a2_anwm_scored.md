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

**What the commanded paths demand.** At the paper-consistent 1 s per step, the *smoothest
possible* flight through each commanded path (a minimum-acceleration spline, the least tilt
any interpolator can ask for) demands a median of 5.0° drag-free and 9.4° with racing-quad
drag. On the 22 Regime A clips that clear the detectability floor, **14 (drag-free) to 17
(with drag) are infeasible**: even the lower bound asks for more tilt than the frames show,
by more than the verifier's 6.46° threshold.

**The designed-control check is the sharpest result.** Regime C was built with 5°, 10° and
20° of required tilt on purpose. Depicted tilt is **0.790°, 0.785°, 0.771°** — flat. The
generator does not bank more when the path demands more; if anything the 20° family is
marginally flatter.

**The minimum consistent dt is about 1.2–1.4 s.** For each clip there is a time step at which
the commanded path stops demanding more tilt than the pixels show: median 1.39 s (Regime A)
and 1.24 s (Regime C) with drag, 1.03 s and 0.83 s drag-free. At the step spacing the paths
were built with, that implies the depicted flight is consistent with a drone moving at a
median 2.9 m/s (A) / 3.9 m/s (C) with drag, 3.9 / 6.2 m/s drag-free — against the ~5 m/s the
release was trained on. Read either way, the generated video is the picture of a drone
flying the commanded path **more slowly than the path's own step spacing implies**.

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

Median of medians 0.73°; seed repeats 0.79°; released-level reference 0.57°. One path
(`course_1` and its seeds) sits systematically at 2.2–2.3°, i.e. the generator's own bias
depends on the path — that is why the cruise band, not a single number, is the comparison.

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

For each clip, dt and `k_d`:

- **lower bound** — natural cubic spline through the commanded waypoints (20 samples per
  step): the smoothest flight that visits every commanded point;
- **upper bound** — per-step: the whole velocity change between consecutive waypoints;
- required tilt is the verifier's own flatness expression
  \(z_B \propto a + g e_3 + k_d v\), evaluated with `src/physics`.

A clip is declared **infeasible only when the lower bound** exceeds the depicted tilt by
more than the frozen verdict threshold (6.46° at `k_d` = 0.295). If even the least-tilt
flight through the commanded waypoints asks for tilt the pixels do not show, no
interpolator can rescue the path. The lower bound never exceeds the upper bound in any
clip/dt/`k_d` (checked in the run).

Testability: gate kept (7 clips dropped), rollout not collapsed (none), and the lower bound
reaches 10° somewhere at that dt (otherwise "below detectability"). Clip **A15** and all
eight Regime B clips come out **not testable**, as expected: A15 is a pure 2 m/s climb whose
lower bound never reaches 10° at 1 s, and Regime B's rescaled paths never do at any dt with
drag (0 detectable at `k_d` 0.295; 21 of 49 Regime C clips are detectable drag-free, 42 with
drag).

## 2. Regime A — the share of generated flights the pixels cannot support

| dt | `k_d` | testable | infeasible | share of testable | share of all 25 |
| --- | --- | --- | --- | --- | --- |
| 0.25 s | 0 | 22 | 16 | 0.73 | 0.64 |
| 0.25 s | 0.295 | 22 | 20 | **0.91** | 0.80 |
| 0.667 s | 0 | 22 | 14 | 0.64 | 0.56 |
| 0.667 s | 0.295 | 22 | 17 | 0.77 | 0.68 |
| **1.0 s** | **0** | 22 | **14** | **0.64** | **0.56** |
| **1.0 s** | **0.295** | 22 | **17** | **0.77** | **0.68** |

All infeasible clips have the depicted tilt within three times the measured noise floor.

## 3. Regime C — dose-response against the designed tilt

Nominal dt = 1 s.

| designed | n | depicted median (p10–p90) | `k_d` | required lower bound | testable | infeasible |
| --- | --- | --- | --- | --- | --- | --- |
| 5° | 21 | 0.790° (0.53–2.47) | 0.295 | 9.45° | 21 | **19** |
| 5° | 21 | 0.790° (0.53–2.47) | 0 | 5.01° | 0 | 0 |
| 10° | 12 | 0.785° (0.61–2.26) | 0.295 | 10.97° | 11 | **11** |
| 10° | 12 | 0.785° (0.61–2.26) | 0 | 8.49° | 11 | 6 |
| 20° | 9 | 0.771° (0.61–3.50) | 0.295 | 21.21° | 8 | **7** |
| 20° | 9 | 0.771° (0.61–3.50) | 0 | 20.68° | 8 | 5 |

Depicted tilt is **flat across a 4× change in designed tilt**. The 5° family drag-free is
not testable (required 5.0° < 10° floor); with drag it is, and 19 of 21 are infeasible.
Seed-repeat band on depicted tilt: 0.50–2.66°, i.e. the flatness is not seed noise.

## 4. Minimum consistent dt

The smallest dt at which the lower bound's median required tilt comes within 6.46° of the
depicted tilt; sweep 0.10–5.00 s, step 0.01 s. Every clip reaches one inside the sweep.

| `k_d` | regime | n | median | p10–p90 | implied speed (median) |
| --- | --- | --- | --- | --- | --- |
| 0.295 | A | 23 | **1.39 s** | 0.21–2.09 | 2.9 m/s |
| 0.295 | B | 5 | 0.65 s | 0.41–1.07 | 3.9 m/s |
| 0.295 | C | 47 | **1.24 s** | 0.99–1.56 | 3.9 m/s |
| 0 | A | 23 | 1.03 s | 0.15–1.95 | 3.9 m/s |
| 0 | B | 5 | 0.31 s | 0.29–0.36 | 5.2 m/s |
| 0 | C | 47 | 0.83 s | 0.10–1.31 | 6.2 m/s |

The implied speed is step spacing divided by dt_min. Under the paper-consistent 1 s, the
regime C designed maneuvers are consistent with flight at ~3.9 m/s instead of the 5 m/s they
were built for; drag-free, ~6.2 m/s. The parameter that makes ANWM consistent is the time
base, and it needs to be ~25 % longer than the release implies (or the speed ~25 % lower).

## 5. What this supports

- **A specific, bounded claim.** At dt = 1 s and `k_d` = 0.295, on the 22 testable Regime A
  clips, the **lower bound** on the tilt their commanded paths require exceeds the depicted
  tilt by more than 6.46° on 17 of them (14 of 22 drag-free). Even the smoothest flight
  through those waypoints is not what the video shows.
- **A dose-response failure that cannot be explained by noise.** Regime C's designed 5/10/20°
  families are depicted at 0.79/0.78/0.77° with a seed band of at most ±1.3° (p10–p90 of the
  flat 5° family is 0.53–2.47°, still far below 5°).
- **A quantified time-base explanation** rather than a bare "breaks physics": the same
  clips become consistent at dt ≈ 1.2–1.4 s.

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

- Regime A's commanded paths contain many repeated waypoints (stalls). The per-step upper
  bound is therefore huge on those clips (up to 55° at 0.25 s); the lower bound is what the
  infeasibility rule uses, and it is unaffected by the plateaus.
- The estimator's ~15 % compression toward level (control b) means depicted tilt is, if
  anything, slightly *under*-reported — which makes the infeasibility finding conservative.
- Regime C's "designed" tilt is a property of the Track B generator, not of the recorded
  path; the required-tilt column is recomputed here from the commanded positions, and the two
  differ (designed 5° → lower bound 9.45° with drag) because the design targeted a peak and
  the spline smooths the path.
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
   banking.* At dt = 1 s and `k_d` = 0.295, the **lower bound** on required tilt exceeds the
   depicted tilt by more than 6.46° on 17 of 22 testable Regime A clips (14 drag-free), and
   the designed 5/10/20° controls are all depicted at ≈0.8° (report 08).
5. *The time base explains most of it.* The same clips are consistent at dt ≈ 1.2–1.4 s.

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
