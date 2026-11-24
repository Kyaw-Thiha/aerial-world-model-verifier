# N3A — Anchor availability and accuracy from ground truth (no video)

**Decision.** The pre-registered anchor gate **fails by a wide margin on the simulated set**: a
window with implied lean ≤ 2° held for 1 s exists in **9/200** development clips (4.5 %) and in
**0/131** dynamic clips (peak required tilt ≥ 15°), so the anchored start is unavailable exactly
where the reader's lean is biased. Where an anchor does exist it is **accurate** — the anchor error
is the clip's true tilt at the window, median **0.00°** at P1 and 2.24° at P2, with the flatness
model check at most 0.002° on sim — and on real footage the quiet windows that the physics can use are
**0/14** UZH sequences and **0/8** TII flights once pre-takeoff and post-landing rests are excluded:
both datasets' only quiet moments are with the drone on a support, where the flatness equation does
not hold.

Pre-registration `configs/n3a_anchor_prereg.yaml` + amendment A (derivative pipeline) + amendment B
(airborne requirement), all committed before the counts they govern. Code
`experiments/n3a_anchor.py`, helper module `src/verdict/anchor.py`, results `results/n3/n3a/`.
Truth trajectories only — **no reader score, no pose rotation, no verdict tier in this report**.
All 200 development clips; no held-out clip read. CPU, **USD 0.00**.

## Question

A quiet window in the path pins the start attitude by physics: if acceleration and velocity are near
zero, flatness forces a near-zero lean, so the thrust axis is near vertical and the world-up
direction in the camera is the known mount rotation applied to vertical,
`u_cam(t_a) = R_cb^T e3`. How often is such a window available inside a scored clip, where does it
sit, and how large is the anchor error?

## Setup

**Statistic.** With `q = a + g e3 + k_d v` (`g = 9.80665`, `k_d = 0.295 /s`, the values the frozen
required side uses), `theta_imp(t) = angle(q, e3)`. This is the ONLY quantity that decides whether a
window pins the start: no outcome, no reader, no video enters it. On the simulated truth
`theta_imp` reproduces the stored `tilt_req` to **0.005°** (spot check, `i02_dev_005`), which is the
`g` difference alone.

**Tiers.** P1 `theta_imp ≤ 2.0°` for ≥ 1.0 s (primary — set by N2A's finding that a 2° start error
costs only 0.8° on the propagated F0 of feasible clips), P2 ≤ 4.0°, P3 ≤ 6.0°, all with the same
1.0 s duration. Windows are searched inside the scored interval only.

**Selection.** Among qualifying windows, the one with the smallest mean `theta_imp` (ties →
earliest); the anchor frame is the minimum-`theta_imp` frame in it, snapped to the nearest posed
frame. `anchor_first` takes the earliest window instead (Part B).

**Populations.** 200 simulated development clips (truth at 250 Hz, `acc` and `vel` taken directly
and, as a sensitivity, differentiated from positions at 100 Hz with a Savitzky–Golay filter);
14 UZH-FPV sequences (500 Hz mocap, positions differentiated the same way); 8 TII dev flights
(274.73 Hz mocap). TII is excluded from every verdict number by instruction — here it is only a
ground-truth availability row.

**Airborne requirement (amendment B).** While the drone rests on a support a normal force replaces
thrust, so the flatness equation does not hold and a quiet window there does NOT pin the start. A
window qualifies on real footage only if it lies strictly between the unit's first and last
"sustained motion" (|v| ≥ 2 m/s for 0.5 s); a unit that never reaches that speed (a pure hover) is
treated as airborne throughout. The filter is **not** applied to the simulated set, which is
rendered in mid-air and never takes off or lands. Raw (unfiltered) coverage is printed beside every
filtered number.

## Results — availability

Simulated development set, `k/n` with the denominator printed:

| tier | all 200 | dynamic 131 | forest 66 | forest feasible 26 | urban 67 | industrial 67 |
| --- | --- | --- | --- | --- | --- | --- |
| P1 (2°, 1 s) | **9/200** | **0/131** | 3/66 | 3/26 | 3/67 | 3/67 |
| P2 (4°, 1 s) | 25/200 | 0/131 | 5/66 | 4/26 | 8/67 | 12/67 |
| P3 (6°, 1 s) | 37/200 | 2/131 | 5/66 | 4/26 | 17/67 | 15/67 |

By flight type (`maneuver` in the Track I manifest) at P1: **hover 8/8**, accel_brake 1/24, and
**0/28 climb_turn, 0/41 ellipse, 0/28 lemniscate, 0/35 slalom, 0/36 turn**. The P1 anchors are the
hover clips plus one accel_brake clip; every manoeuvring family is at zero. At P3 the lemniscate
family reaches 15/28 and slalom 7/35, but climbs, ellipses and turns stay at zero.

By mount at P1: 0° 5/50, 15° 0/50, 30° 4/50, 45° 0/50 — the availability that exists is not
mount-driven, it is manoeuvre-driven.

**How far from viable.** The availability surface (descriptive) over threshold × duration shows the
trade-off is not gentle:

| duration | 2° | 4° | 6° | 8° | 10° | 12° | 15° |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0.25 s | 14/200 | 38/200 | 46/200 | 54/200 | 55/200 | 48/200 | 52/200 |
| 0.5 s | 8/200 | 34/200 | 40/200 | 36/200 | 41/200 | 38/200 | 38/200 |
| 1.0 s | 9/200 | 25/200 | 37/200 | 49/200 | 64/200 | 83/200 | 97/200 |

and the per-clip quality of the quietest window is the same statement without a threshold: the
quietest 1 s window of a dev clip has a max-implied-tilt at the 10th/25th/50th/75th/90th percentile
of **3.55 / 8.64 / 15.38 / 30.92 / 45.54°**. Half the clips cannot produce a 1 s window below 15°
at all. For dynamic clips the same surface gives 0/131 at 2°, 4° and 6°, 9/131 at 8°, 28/131 at 15°.

**Real footage.**

| dataset | raw (no filter) | airborne | anchor position | anchor error |
| --- | --- | --- | --- | --- |
| UZH, 14 sequences, P1 | 12/14 | **0/14** | 1–4 % of the sequence | n/a |
| UZH, P3 | 12/14 | 1/14 | 57.7 % | 3.16° |
| TII, 8 flights, P1 | 6/8 | **0/8** | 81–99 % of the flight | n/a |
| TII, P3 | 6/8 | 0/8 | — | — |

The raw UZH windows are the drone on its support before takeoff: **all 12 sit before the first
sustained motion**, with a true tilt of 0.33–9.24° while the implied tilt is ≤ 0.12° — the flatness
equation is exactly what fails there. The raw TII windows are 4 pre-start rests and 2 windows after
the last sustained motion, at 0.02–0.62 m altitude against flight maxima of 0.98–1.89 m. The single
surviving UZH window is a genuine mid-flight pause at 57.7 % of the sequence, 2.79° implied against
3.16° true.

## Results — accuracy

Media of the anchor error (angle between the true world-up in the camera at the anchor frame and
`R_cb^T e3`) over the clips that have an anchor, and the flatness model check (angle between the
implied thrust axis `q/|q|` and the true body z-axis at the same frame):

| set | tier | n | error median | p90 | max | model check median |
| --- | --- | --- | --- | --- | --- | --- |
| sim | P1 | 9 | **0.00°** | 0.07° | 0.37° | 0.00° |
| sim | P2 | 25 | 2.24° | 2.81° | 3.52° | 0.00° |
| sim | P3 | 37 | 2.30° | 3.35° | 5.07° | 0.00° |
| UZH | P3 | 1 | 3.16° | 3.16° | 3.16° | 2.51° |
| sim (positions differentiated) | P1/P2/P3 | 9/25/38 | 0.00/2.24/2.30° | — | 3.52/5.07° | 0.00° |

Three readings, in order of importance.

1. **The error is the true tilt, and it never exceeds the tier threshold by much.** P2's worst
   anchor carries 3.52° of real tilt for a 4° allowance; P3's worst 5.07° for a 6° allowance. The
   physics-controlled part of the route works.
2. **The P1 accuracy number is nearly trivial, and the report says so.** Eight of the 9 P1 anchors
   are hover clips whose true tilt is exactly 0, so the median error is exactly 0 and the p90 is
   0.07°; only `i02_dev_056` (an accel-brake clip) carries a real 0.37°. The informative accuracy
   rows are P2/P3, where 16 and 28 of the anchors are on non-hover clips.
3. **The flatness model itself is exact on sim** (0.00°, because the simulated `acc`/`vel` are the
   arrays `tilt_req` was built from) and is 2.51° at the single real anchor — the drag model's own
   error at that moment, not an anchor error.

**Differentiating positions instead of using the stored derivatives changes nothing**: 9/200 vs
9/200 at P1, 25/200 vs 25/200 at P2, 37/200 vs 38/200 at P3. The availability conclusion is a
property of the trajectory, not of the derivative pipeline.

**Mount error.** The anchor error equals the true total tilt and therefore contains no mount term at
all: `angle(R_cb^T R_bw^T e3, R_cb^T e3) = angle(R_bw^T e3, e3)`. A mis-specified mount rotates the
anchor *direction* in the camera by exactly the perturbation (measured 2.000° for a 2° perturbation
and 5.000° for a 5° one at every mount), and what that does to the readout is measured in Part B: it
cancels at the anchor frame and grows with the rotation performed since, so it is not a lean offset.

## Pre-registered criteria, scored

| criterion | result |
| --- | --- |
| `availability_P1` — ≥ 70 % of 200, ≥ 60 % of 131 dynamic, ≥ 60 % of 26 forest feasible | **FAIL**: 9/200 (4.5 %), 0/131 (0 %), 3/26 (11.5 %) |
| `accuracy_P1` — median ≤ 2.0°, p90 ≤ 3.0°, model check ≤ 0.5° | **PASS**, but near-trivially: 8/9 anchors are hovers with true tilt 0, the ninth 0.37° |
| `real_footage` (reported, not gating) | UZH 0/14 and TII 0/8 at P1 after the airborne filter |

Per the pre-registered `downstream_contract`, Part B is therefore **descriptive**, Part D (fallbacks)
is **run**, and the anchored route cannot be a general replacement for the reader's start.

## What this supports

* **The anchor route fails on availability, not on accuracy.** A 2°-accurate start exists whenever
  the drone genuinely stops moving in the air, and it is exactly as accurate as the physics says;
  but the simulated manoeuvres (ellipse, turn, slalom, lemniscate, climb-turn, accel-brake) never
  stop, so **0 of 131 dynamic clips** contain one. The same holds on real footage once rests on a
  support are excluded: UZH 0/14 and TII 0/8.
* **Availability is manoeuvre-driven, not mount-driven or family-driven.** Hover 8/8 against 0/24
  to 0/41 for every other manoeuvre family; the three scene families and the four mounts differ only
  by which clips happen to contain a hover or a lemniscate pause.
* **A "quiet window" test cannot distinguish an airborne hover from a drone on a support** using
  the path alone unless takeoff and landing are identified. This is the failure that made 12/14 UZH
  and 6/8 TII raw anchors unusable, and it is the reason the pre-registered test needs the airborne
  requirement to mean anything.
* **Relaxing the threshold does not rescue the route.** Reaching even a third of the simulated clips
  needs a 10° allowance at 1 s, and a 10° start error is the size of the forest bias the reader
  already has (N2A: reader-start F0 forest feasible median 8.4°, reader 8.7°).

## What this does not show

* Anything about a start that is *not* anchored: this report is silent on reader, oracle and
  propagation quality (Part B), and on the drift tail (Part C).
* Anything about a drone that does hover. A deployment whose flights include real pauses (delivery,
  inspection) would see a different availability curve; the numbers here are for this
  manoeuvre-generating pipeline and this footage.
* Anything about using the *video* to confirm that a quiet window is airborne (a static scene, a
  visible ground plane, an optical-flow magnitude). That would be a different route and is not
  tested.
* The held-out simulated clips and the held-out real flights (never read).

## Caveats

* Nine P1 anchors, eight of them hover clips: the P1 accuracy numbers are degenerate and are
  labelled so.
* The airborne rule is a correction made after seeing the first count (amendment B). It changes no
  simulated number (the filter is not applied there) and makes the real-footage rows negative; the
  raw rows are printed beside the filtered ones.
* UZH and TII rows are 14 and 8 clusters; they are descriptive only and are not pooled with sim.
* The single UZH P3 anchor is one clip. It is reported as one clip.
* The simulated `acc`/`vel` are the arrays the required side was built from, so the model check is
  exact by construction there; the differentiating-position sensitivity is what shows the
  availability conclusion is not an artifact of that.

## Provenance

* Pre-registrations: `configs/n3a_anchor_prereg.yaml`, `configs/n3a_amendment_a.yaml`,
  `configs/n3a_amendment_b.yaml`; decisions `[N3-a]`, `[N3-b]`.
* Code `experiments/n3a_anchor.py`, `src/verdict/anchor.py`; results
  `results/n3/n3a/{windows.csv,surface.csv,summary.json}`. Code commit `4611b97`; results and
  amendments in `51c2f0e` and the checks commit; the summary JSON is regenerated by
  `PYTHONPATH=. pixi run python experiments/n3a_anchor.py` in about 20 s.
* Inputs: `data/n05/gt` (200 clips, 250 Hz), `data/n05/scores` (frame grids only),
  `data/n05/poses/sim200/scale1` (frame times only), `data/uzh/<seq>/{truth.npz,meta.json}`,
  `data/raw/tii/{autonomous,piloted}/mocap_*.csv`, `results/n2/inputs/manifest.csv`
  (sha256 `50aacbb2…ce6794`).
* Compute: local CPU, 20 s, **USD 0.00**. No Modal app, no volume, no R2 write.
