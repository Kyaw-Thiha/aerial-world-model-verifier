# N2C — Part C: provenance of the Part 4 comparison (start attitude, horizon, like-for-like)

**Decision.** The Part 4 comparison is **not like for like**: it sets the propagated arm's
**relative** attitude error over a 1 s horizon (3.54°, oracle start) against the reader's
**absolute** per-frame error (9.07°), so the two numbers measure different quantities and the
propagated arm is also given a free starting attitude. Put on the same footing, the propagation
genuinely tracks attitude *changes* better than the reader (relative 1 s error **3.62°** vs
**7.39°**), but with the reader's own frame-0 attitude as the start its **absolute** error is
**9.56°** over the clip, against the reader's **9.07°** — no better — and only the oracle start
(4.41°) beats the reader absolutely. Part 4's numbers stand as published; this note adds the
decomposition and does not rewrite them.

Pre-registration `configs/n09_part4_provenance_prereg.yaml`, committed before this arm was
computed. Code `experiments/n09_part4_provenance.py`; results `results/n2/n09/`. CPU only,
**zero spend**. 78 arm-A SolarWM clips.

## The three questions the brief asks

### 1. Exactly what start attitude did the SolarWM propagation use?

The **true commanded camera-to-world attitude at frame 0** — the pose the generator was
conditioned on: `R_cw(0) = body_rotation(yaw₀, pitch₀, roll₀) @ R_bc`, with `R_bc =
camera_to_body_pitched(0)`, taken from Track H4's path table. It is an **oracle start**: by
construction its error is zero, so N04's 1/2/5 s numbers measure relative-rotation fidelity only.
This is stated in N04 and reproduced here (oracle absolute error at frame 0 = 0.00° on all 78
clips).

### 2. What horizon does the reader's 9.07° refer to?

A **per-frame, zero-horizon** quantity: for each clip, the median over its 81 frames of the angle
between the reader's up and the commanded up, and then the median of those 78 per-clip medians. It
is not a 1, 2 or 5 s number. It is also an **absolute** error, against the commanded attitude, not
a change over time.

### 3. Is the comparison like for like?

**No, in two ways**, and the distinction matters:

*Quantity.* The propagated 3.54° is a relative error over a 1 s horizon (how much the estimated
rotation drifts); the reader's 9.07° is an absolute error at a single frame. A relative signal can
be excellent while its absolute level is wrong, and vice versa.

*Information.* The propagated arm is given the true start; the reader is given nothing. The
deployment-fair version gives the propagated arm the reader's own frame-0 reading.

Recomputing both arms under both readings on the same 78 clips:

| reading | 0 s | 1 s | 2 s | 5 s |
| --- | --- | --- | --- | --- |
| **relative** (N04's form), oracle start | 0.00 | 3.62 | 4.62 | 6.69 |
| **relative**, reader start | 0.00 | 3.70 | 4.56 | 7.30 |
| **relative**, reader | 0.00 | **7.39** | **7.46** | 6.80 |
| **absolute**, oracle start | 0.00 | 2.03 | 3.60 | 6.69 |
| **absolute**, reader start | 6.98 | 5.21 | 9.23 | 12.37 |
| **absolute**, reader | 6.98 | **4.71** | **8.90** | 9.23 |

Median over all frames: oracle **4.41°**, reader start **9.56°**, reader **9.07°**.

Reading:
* **Relative.** The propagation (oracle or reader start — the start cancels) is about half the
  reader's relative error at 1–2 s (3.6–3.7° vs 7.4°). This is a real propagation advantage and it
  is what N04's 3.54° actually measures.
* **Absolute at a matched horizon.** With a reader start the propagation is **worse** than the
  reader at 1 s (5.21° vs 4.71°) and 2 s (9.23° vs 8.90°), and worse at 5 s. It is not worse
  against the reader's all-frame median (9.56° vs 9.07°) but that compares a 1 s absolute error to
  a zero-horizon median. The pre-registered rule ("reader-start 1 s below the reader's all-frame
  median") therefore returns *matched*, on a comparison that is itself not horizon-matched; the
  horizon-matched columns are the substantive reading and they say no.
* **Absolute, oracle start.** Only the oracle arm beats the reader absolutely (4.41° vs 9.07°),
  and that is bought entirely by the free start.

## Would the start be available from the conditioning frame in deployment?

**Yes in the scenario N04 tests, conditionally.** The generator's conditioning pose *is* an input:
SolarWM/H04 is conditioned on a commanded camera-to-world pose, so a world model that exposes its
conditioning pose hands the verifier the exact start for free. What that depends on:

* **Whether the model exposes a pose at all.** If a generator is conditioned only on pixels (no
  pose), the start must be estimated from the first frame — and then it carries the reader's own
  error, which the reader-start arm shows is enough to erase the absolute advantage.
* **Whether the frame is the verifier's frame.** The start must be in the same world frame as the
  path's required lean. Yaw is free (the lean is yaw-invariant), but a generator that works in a
  relative or arbitrary frame needs an alignment; a constant roll/pitch convention error would not
  cancel.
* **Its accuracy.** A flight controller or navigation filter supplies 1–3°; N2A prices a 2° error
  at +0.8° of F0 residual on feasible clips and no change on the defect conditions. So a 1–3°
  start behaves close to the oracle **for the residual**, but the residual is not the same as the
  verdict operating point, which N2A shows is set by the 45° drift.

## Is a Part 4 redo needed?

**No.** N04's numbers are correct for what they measure (relative fidelity from an oracle start)
and are reproduced here (oracle relative 1 s 3.62° against the published 3.54°). What is wrong is
the *framing* of the comparison against the reader's 9.07°: the two are different quantities and
different information budgets. The correction is this note, not a new run. The actionable
statement is: **the propagated route needs a start better than the reader's first frame; a
conditioning pose or a flight-controller attitude provides one, a reader-based start does not.**

## What this supports

* The propagated lean is a good *relative* rotation signal on generated video (3.6–3.7° per 1 s
  against the reader's 7.4°), which is the piece Part 4 was really testing.
* The absolute advantage over the reader requires an absolute attitude source that is not the
  reader. This is the same conclusion N2A reaches independently on the simulated set (reader-start
  F0 forest feasible 8.4° against the reader's 8.7°; oracle 1.7°).

## What this does not show

* Anything about a deployed generator that does not expose a pose: the start's availability is a
  property of the model interface, not measured here.
* A converged decomposition of the reader's 7.4° relative 1 s error into bias change and noise.
* Anything about TII (excluded) or the simulated held-out clips (not read).

## Caveats

* The 78 clips are three draws of 26 paths, so the clips are not independent; every number is a
  median over clips and the cells are not intervals.
* The absolute curve is anchored at frame 0 of each clip; a different anchor would change the
  absolute numbers but not the oracle-versus-reader-start ordering.
* The relative statistic is N04's construction re-expressed on the up stream; the oracle arm
  reproduces N04 to within 0.08° at 1 s, which is the check.
* The pre-registered binary decision passed on a horizon-mismatched comparison (see above) and is
  reported as passed, but the substantive reading does not rest on it.

## Provenance

* Pre-registration `configs/n09_part4_provenance_prereg.yaml`; decisions `[N2]`.
* Inputs: `results/h04/depicted_main/depicted.jsonl` and `results/h04/paths/paths.npz` from
  `awmv-h3` (read-only); `data/n04/poses/gen/fov90` (read-only, Track N's Part 4 poses);
  `results/n04/clips.csv`.
* Code `experiments/n09_part4_provenance.py`; results `results/n2/n09/{per_clip.csv,summary.json}`.
* Compute: local CPU, about 1.5 s, **USD 0.00**. No Modal app, no volume, no R2 write.
