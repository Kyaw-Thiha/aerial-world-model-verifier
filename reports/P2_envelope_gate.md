# P2 — Thrust-envelope admissibility of the commanded path (Check C)

**Decision.** The path-only envelope **detects all 13 path-side thrust-limit defects with zero
false positives on the 80 feasible simulated paths, and the physics bound it uses rejects real
drone flight.** Under Track I's frozen design limits (60° tilt, 2.0 g thrust ratio, 720 deg/s)
**11 of 20 real development flights are rejected** — 8 on tilt and 9 on thrust, including two
autonomous TII racing flights at 6.3 g and two indoor UZH flights at 2.5-2.6 g — so the
pre-registered C1 ("no real flight rejected") **fails**; the real-flight envelope is instead
**122.5° / 6.05 g / 781 deg/s / 457 m/s³**, and nothing in the 200 simulated paths or the 22 H04
commanded paths exceeds it.

Pre-registration `configs/p2_envelope_prereg.yaml` (committed before any envelope number). Code
`experiments/p2_envelope_gate.py`. Results `results/p/p2/`. CPU only, zero spend. Six TII
development flights and 14 UZH sequences; TII 13p/07a never opened.

## Question

Given only the commanded path, is the flight admissible for a quadrotor — and does the envelope
separate the simulated set's path-side defects from feasible flight without rejecting real drone
flight?

## Setup

Five quantities per path, from the truth/mocap position track resampled to 100 Hz and
differentiated by the same 0.31 s Savitzky-Golay filter as P0: peak required tilt, peak thrust
ratio `|a + g e3 + k_d v|/g`, peak specific acceleration, peak required thrust-axis rate, peak
jerk. The envelope has two parts:

* **physics** (from Track I's frozen simulated-set pre-registration): tilt ≤ 60°, thrust ratio ≤
  2.0, rate ≤ 720 deg/s; no physics bound exists for jerk.
* **data** (leave-one-flight-out on real flight): per-flight 99.5th percentile of each quantity,
  threshold = the maximum over the *other* flights. 99.5 % of a 30 s flight is 11,000 frames, so
  a single bad frame cannot define the envelope; with 20 flights the maximum is the only
  threshold that can satisfy "must not reject real flight", and the leave-one-out false-reject is
  reported as a count.

## Results

### Part 1 — real flights (confirmatory)

| gate | rejected | detail |
| --- | --- | --- |
| physics (tilt ≤ 60°, TWR ≤ 2.0, rate ≤ 720) | **11/20** | tilt 8/20, thrust 9/20, rate 1/20 |
| data envelope, leave-one-flight-out | 1/20 per quantity | the extreme flight, by construction |
| conjunction | 9/20 pass | every failure is the physics gate |

Every TII racing flight exceeds 2.0 g (2.25-6.34); the two autonomous ellipse flights reach
91-92° of required tilt and 6.3 g. Two UZH indoor flights also exceed 2.0 g
(`indoor_forward_6` 2.46, `indoor_forward_7` 2.58) and three reach 77-88° of required tilt at
the peak (`indoor_45_9`, `indoor_forward_6`, `indoor_forward_7`). **The 60°/2.0 g limits are a
design envelope for the simulated set, not a physical feasibility bound**: a racing-class airframe at 6:1 thrust-to-weight is physically admissible at
80° of tilt.

Full-flight and airborne-interior peaks are identical on every real sequence (the N3 airborne
motion rule removes take-off and landing rests but not these peaks), so this is not an
end-of-recording artefact.

### Part 2 — commanded paths

| population | outside physics tilt | outside physics thrust | outside data envelope (any quantity) |
| --- | --- | --- | --- |
| 200 simulated dev paths | 13/200 | 13/200 | 0/200 |
| 80 simulated feasible paths | 0/80 | 0/80 | 0/80 |
| 22 H04 commanded paths | 0/22 | 0/22 | 0/22 |

The 13 outside the physics gate are exactly the 13 `thrust_limit` clips (median peak tilt 66.5°
against 19.5° for the rest). The feasible simulated paths reach 50.4° and 1.57 g, inside the
design limit; nothing in the simulated set or in the generator's commanded paths is anywhere near
what a real racing drone does.

### Part 3 — path-side defects (confirmatory)

| cell | result |
| --- | --- |
| thrust-limit dev clips detected from the path alone | **13/13** |
| feasible dev paths flagged | 0/80 |
| recomputed peak tilt vs the manifest's stored value | max abs difference 0.182° |

C2 passes. The detection is not a tautology about the label: the peak tilt is recomputed from the
position track with the project's own differentiator and reproduces the manifest to 0.18°.

## Criteria

| criterion | result |
| --- | --- |
| C1 no real development flight rejected by the physics gate | **FAIL** — 11/20 |
| C2 ≥ 12/13 thrust-limit paths detected, 0/80 feasible flagged | **PASS** — 13/13 and 0/80 |

## What this supports

* A path-only gate detects path-side infeasibility (over-thrust edits) perfectly on this set, at
  zero false positives, and the numbers reproduce the manifest.
* The real-flight envelope is much wider than the simulated set's design envelope; a generator
  cannot be held to 60°/2.0 g as a matter of physics, only as a matter of the dataset's design.
* Which paths a generator is accountable for is therefore a *stated* envelope, not a discovered
  one: the simulated set and the H04 paths sit entirely inside the real-flight envelope, so the
  path-side test never fires on generated flight.

## What it does not show

* Nothing about whether a depicted flight matches its path — that is Check B, and it needs video.
* Nothing about real held-out flights.
* No jerk or rate verdict from physics: both bounds are data-derived, and at 781 deg/s and
  457 m/s³ the data envelope is so wide that it will not fire on any plausible generated path.

## Caveats

* The physics limits are inherited verbatim from Track I's frozen pre-registration, which built
  the simulated set to them. They were frozen before this track and are not re-tuned here.
* With 20 real flights, the leave-one-out maximum rule has no statistical power to detect a
  too-tight envelope: exactly the globally extreme flight can be rejected, and nothing else. The
  11/20 physics count is the informative number.
* The 99.5th percentile was chosen to be robust to single-frame truth artefacts; at 500 Hz it is
  still sensitive to a 0.1 s event, which is why the physics gate (a max) is reported separately.
