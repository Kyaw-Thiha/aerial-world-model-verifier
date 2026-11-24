# N3B — Verdict tiers on the lean propagated from a path anchor, and the no-anchor fallback (Part D)

**Decision.** The anchored start is **more accurate than the reader wherever it exists** — on the 16
feasible clips that have a 6° anchor the anchored F0 residual is a median **0.03°** against the
reader's 1.17° (max 9.08°) and a reader start's 1.08° (max 9.14°) — but it reaches **none** of the
clips that matter: at the matched pooled 8/80 it detects **6/27** tilt removals against the reader's
22/27, **6/27** mirrors against 15/27, and **0/9** forest removals, because **not one forest
tilt-removal clip and not one of the six drift-tail clips has an anchor at any tier**. All five
pre-registered behavioural criteria fail and, per the pre-registered gate, this part is
**descriptive**: the route is not a deployable replacement for the reader's start, and Part D
reports the fallback as abstention on 163–191 of 200 clips.

Pre-registration `configs/n3b_anchored_tiers_prereg.yaml` (committed before any anchored score).
Code `experiments/n3b_anchored_tiers.py`, fallbacks `experiments/n3d_fallbacks.py`, results
`results/n3/n3b/` and `results/n3/n3d/`. All 200 development clips; **no held-out clip read**.
CPU, **USD 0.00**.

## Question

At a matched feasible false-reject, does F0 on the lean propagated from a physics anchor beat the
reader's lean in forest and on the 30/45° mounts, now that the start comes from the path?

## Setup

**Arms.** `anchor1` (P1 anchor, 2°/1 s, primary), `anchor1_P2`, `anchor1_P3` (4° and 6°), and
`anchor_multi_P3` (per frame, propagate from the nearest qualifying window) are the anchored arms;
`oracle_window` uses the **true** world-up at the same P3 anchor frame, which separates the anchor's
angular error from the propagation drift; `reader`, `level`, `reader_start`, `oracle` and
`oracle_2deg` are the N05/N2A references, recomputed here and **verified against
`results/n2/n07/per_clip.csv` (maximum F0 difference 0.000 on all five)**.

**Tiers.** F0 (K03 thrust-axis angle, no nuisance) is confirmatory; C2 and R are re-checked with
N05's functions. S is not re-run (N2A found it unusable at a 5–10 % budget on sim).

**Matching.** Every arm cut at its own (k+1)-th largest pooled feasible score with k = 8. An arm
that cannot score a clip abstains; the headline counts an abstention as a **non-detection**, and the
number of scored feasible clips is printed beside every threshold. Within-subset operating points
(section 4) are labelled descriptive.

**Gate.** Part B is confirmatory only if N3A's availability criterion holds. It does not (9/200
clips, 0/131 dynamic, 3/26 forest feasible), so everything below is **descriptive**.

## Results

### 1. Coverage is the whole story

| arm | all 200 | feasible 80 | removed 27 | mirrored 27 | scaled 33 | partial 20 | thrust-limit 13 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `anchor1` (P1) | **9** | 9 | 0 | 0 | 0 | 0 | 0 |
| `anchor1_P2` (P2) | **25** | 14 | 2 | 3 | 3 | 3 | 0 |
| `anchor1_P3` (P3) | **37** | 16 | 6 | 6 | 6 | 3 | 0 |

The P1 arm is scoreable only on the nine hover/accel-brake clips, all feasible: **it has nothing to
detect by construction**. That single row is the decisive fact — the clips the reader gets wrong
(dynamic, high mount, forest) are exactly the clips with no anchor.

### 2. The matched pooled 8/80 point is degenerate for an anchored arm, and the report says so

| condition | reader | reader start | oracle | `anchor1_P3` | `anchor1_P2` | `anchor_multi_P3` | `oracle_window` |
| --- | --- | --- | --- | --- | --- | --- | --- |
| feasible | 8/80 | 8/80 | 8/80 | 8/80 | 6/80 | 8/80 | 8/80 |
| tilt_removed | **22/27** | 13/27 | 13/27 | 6/27 | 2/27 | 6/27 | 6/27 |
| mirrored_roll | 15/27 | 16/27 | **19/27** | 6/27 | 3/27 | 6/27 | 6/27 |
| tilt_scaled | **15/33** | 7/33 | 8/33 | 6/33 | 3/33 | 6/33 | 6/33 |
| partial | 4/20 | 6/20 | 6/20 | 3/20 | 3/20 | 3/20 | 3/20 |
| thrust_limit | 13/13 | 13/13 | 3/13 | 0/13 | 0/13 | 0/13 | 0/13 |
| matched cut | 9.33° | 17.98° | 16.44° | **0.00°** | 0.00° | 0.00° | 0.00° |
| feasible scored | 80 | 80 | 80 | 16 | 14 | 16 | 16 |

The anchored arms' cut collapses to **0.00°** because their feasible scores are mostly the exact
zeros of the simulated hovers; the "8 flagged" are simply the eight nonzero ones. The comparison at
this operating point is therefore unfavourable and **uninformative in both directions**: the arm is
strict, not badly calibrated. The informative reading is section 3.

Forest stratum at the same point: reader 9/9 removals with 8/26 feasible false-rejects; reader start
4/9 with 3/26; `anchor1_P3` **0/9** with 2/26; oracle 4/9 with 2/26. **Every one of the nine forest
tilt-removal clips has no anchor at P1, P2 or P3**, so the forest ceiling for the anchored arm is
0/9 regardless of threshold.

### 3. Where an anchor exists, the anchored lean is the best lean in the study

F0 by condition **over the clips that have a P3 anchor** (descriptive; the n column is the
denominator):

| set | n | median | max |
| --- | --- | --- | --- |
| `anchor1_P3` on feasible | 16 | **0.03°** | 2.54° |
| `oracle_window` (true up at the same frame) on feasible | 16 | 0.02° | 1.25° |
| `oracle` (true up at frame 0) on feasible | 16 | 0.02° | 1.25° |
| `reader` on feasible | 16 | 1.17° | **9.08°** |
| `reader_start` on feasible | 16 | 1.08° | 9.14° |
| `anchor1_P3` on removals / mirrors / scaled / partial | 6 / 6 / 6 / 3 | 7.62° / 7.26° / 9.11° / 3.16° | — |

Paired on the same 16 clips, `anchor1_P3` beats the reader by a median **1.03°** (14/16 clips) and a
reader start by **0.74°** (13/16); it is 0.00° worse than the oracle's 0.02° median — i.e. **the
anchor costs essentially nothing against a true start on these clips**. Note what the reader's 1.17°
median and 9.08° maximum on *hover* clips means: the reader is biased by the scene even when the
drone is level, which is the bias the anchor is immune to.

**Descriptive within-subset operating point** (cut = the worst anchored feasible residual, so 0
false-rejects by construction; every rate is over the clips with an anchor, never over the full
condition):

| arm | cut | feasible | removed | mirrored | scaled | partial |
| --- | --- | --- | --- | --- | --- | --- |
| `anchor1_P2` | 2.54° | 0/14 | 2/2 | 3/3 | 3/3 | 3/3 |
| `anchor1_P3` | 2.54° | 0/16 | 6/6 | 5/6 | 6/6 | 2/3 |
| `anchor_multi_P3` | 2.62° | 0/16 | 6/6 | 5/6 | 6/6 | 3/3 |
| `oracle_window` | 1.28° | 0/16 | 6/6 | 5/6 | 6/6 | 1/3 |

Near-perfect separation on 16 feasible against 21 defect clips — with two caveats that keep this
from being a result: n is tiny and daggered, and the feasible residuals are dominated by simulated
hovers whose true lean is exactly zero, so the null is cleaner than any real video's.

### 4. C2 and coherence (secondary comparison)

At the same matched pooled 8/80: C2 reader 17/27 removals and 10/27 mirrors against
`anchor1_P3` 6/27 and 6/27 (cut 0.00°); coherence reader 10/27 and 10/27 against `anchor1_P3` 2/27
and 4/27. Neither changes the reading and both reproduce N05's ordering (the reader's C2 remains the
only tier with real removal sensitivity at a working false-reject rate). `anchor1_P3` again flags
8 feasible clips out of its 16 scored, at a 0.00° cut.

### 5. The drift tail is untouched, and B4's "pass" is vacuous

Feasible clips with F0 above 20°: reader 0/80, reader start 6/80, oracle 6/80, level 28/80,
**`anchor1`/`anchor1_P2`/`anchor1_P3`/`anchor_multi_P3`/`oracle_window` 0 of their 9/14/16/16/16
scored clips**. Verified directly: **none of the six tail clips (`i02_dev_003/019/038/043/059/063`,
all 30–45° mounts) has an anchor at any tier.** So the anchored arms do not cure the drift tail;
they abstain on it. The pre-registered B4 criterion is formally met and is reported as **vacuous**.

### 6. Draw-2 robustness

The anchored arms are unchanged on the second DPVO draw: median feasible F0 0.00° at P2 and P3, zero
feasible clips above 20°, same coverage. The anchor frame is a trajectory quantity, so a different
pose draw cannot move it; only the propagation changes, and it does not change the reading.

### 7. UZH — not identifiable, reported as absent rather than skipped

The pre-registration promised a descriptive UZH table. It cannot be produced: after the airborne
filter (N3A) the anchored arm has **0 of 14 sequences at P1/P2 and 1 of 14 at P3**, so the anchored
UZH F0 exists for a single sequence and no condition-wise comparison is possible. Rather than print
a table of n = 1, this is recorded as **not identifiable** (also in `docs/decisions.md`). The reader
reference for UZH remains N02b's.

## Part D — what the verdict is for a clip with no anchor

Pre-registered rule: a clip with no anchor **abstains**; no reader start, no level start, no
substitute. Consequences, `k/n`:

| tier | no anchor, all 200 | no anchor, dynamic 131 | no anchor, forest feasible 26 | unreachable removals | unreachable mirrors |
| --- | --- | --- | --- | --- | --- |
| P1 | 191/200 | **131/131** | 23/26 | 27/27 | 27/27 |
| P2 | 175/200 | 131/131 | 22/26 | 25/27 | 24/27 |
| P3 | 163/200 | 129/131 | 22/26 | 21/27 | 21/27 |

(`results/n3/n3d/coverage.csv` has the same table by family, mount and condition.) The anchored
verdict is therefore **silent on every dynamic clip at P1 and P2** and on 129 of 131 at P3, and its
detection ceiling is 2/27 removals even at P3. A verifier that abstains on the whole dynamic set is
not a fallback for the reader: it is a different instrument that happens to be blind where the
reader is biased.

## Pre-registered criteria, scored

| criterion | result |
| --- | --- |
| B1 gate (N3A availability) | **FAIL** → Part B descriptive |
| B2 forest F0: removals ≥ 8/9 **and** forest false-rejects ≤ 4/26 | **FAIL**: 0/9 removals (2/26 false-rejects) |
| B3 overall F0: removals ≥ 18/27 **and** mirrors ≥ 15/27 | **FAIL**: 6/27 and 6/27 |
| B4 drift: ≤ 3 feasible above 20° **and** cut ≤ 12° | **PASS, vacuous** — 0 of the scored clips are above 20° because the tail clips have no anchor; cut 0.00° |
| B5 beats a reader start: +3 forest removals **or** −2 forest false-rejects | **FAIL**: −4 removals, +1 false-reject |
| `oracle_window` upper bound | meets neither B2 (0/9 forest removals) nor B3 |

## What this supports

* **The anchor is an excellent start and an unusable verdict input.** On the clips that have one,
  the anchored lean matches an oracle start to 0.01° and beats the reader by about a degree — and
  detects 6/27 removals because it can only see 6 removal clips.
* **Availability is the binding constraint, and it is structural, not a tuning issue.** No forest
  removal and none of the six drift-tail clips has an anchor at any tier up to 6°; the detection
  ceiling at P3 is 6/27 by coverage alone (Part D).
* **Anchoring does not address the 30/45° drift tail.** The tail lives in the clips that never stop,
  and those are the clips with no anchor; the anchored arms' tail-free record is abstention.
* **A reader start remains better than nothing and worse than a good external attitude**: reader
  start 13/27 removals and 4/9 forest removals at matched 8/80, the anchor 6/27 and 0/9 on the
  clips it can see, the reader 22/27 and 9/9.

## What this does not show

* Whether a *deployment* with genuine hovers (delivery, inspection, a generator that starts and ends
  at rest) would get the anchored lean's accuracy on real verdicts: all the accuracy evidence here
  is on simulated hovers with an exact-zero null.
* Whether combining the anchor with the reader (anchor where available, reader elsewhere) helps —
  that would be a mixture and it was not pre-registered. The honest expectation from these numbers
  is that it changes at most the hover clips, where all arms already agree.
* Anything about UZH verdicts (not identifiable) or TII (excluded by instruction).
* Anything about the held-out simulated clips (never read).

## Caveats

* The anchored arms' matched cut of 0.00° makes the pooled matched tables uninformative in both
  directions; the section 3 paired table and the section 3 subset operating point are the readable
  evidence, and both are descriptive and small.
* Section 3's near-perfect separation rests on 16 feasible clips whose null is an exact zero, and on
  cells of 2–6 clips. It is a mechanism check, not a performance claim.
* The `thrust_limit` condition (13 clips, a path-side label) is 0/13 for every anchored arm — those
  paths are dynamic by definition, so none has an anchor.
* Matching is pooled; per-family matching would change nothing, because the forest cells are empty
  by coverage.
* DPVO draw 2 confirms the reading, so it is not draw-dependent.

## Provenance

* Pre-registration `configs/n3b_anchored_tiers_prereg.yaml`; decisions `[N3-c]`.
* Inputs: `results/n3/n3a/windows.csv` (anchor frames, cross-checked to 1e-9 against a re-derivation
  in the run), `results/n2/n07/per_clip.csv` (F0 cross-check, max difference 0.000),
  `data/n05/{gt,scores,poses/sim200/{scale1,scale1_rep2}}`, `results/n05/clips.csv`,
  `results/n2/inputs/manifest.csv`.
* Methods reused unchanged: F0 = K03 thrust-axis angle via `up_body_from_lean` and
  `src.physics.flatness.angle_between_deg`; C2/R = `experiments/n07_sim_tiers.tier_scores`;
  propagation = `src/verdict/anchor.propagate_from`.
* Code `experiments/n3b_anchored_tiers.py` (about 12 s), `experiments/n3d_fallbacks.py` (2 s);
  every headline number is re-checked by `experiments/n3_checks.py`.
  Results `results/n3/n3b/{per_clip.csv,per_clip_draw2.csv,summary.json}`,
  `results/n3/n3d/{coverage.csv,summary.json}`.
* Compute: local CPU, **USD 0.00**. No Modal app, no volume, no R2 write.
