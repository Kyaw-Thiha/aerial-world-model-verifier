# Report 07 — The verdict test again, with the frozen gate and drag as an axis (Track A2, Task 2)

Track A2. Written 2026-11-09. Pre-registration: amendment 2 (commit `d8941e8`), amendment 4
(deviation record), frozen gate `configs/gate_frozen.yaml` (commit `9e1656d`, fitted by
report 06). Runs: `results/a2_verdict_gated/<run>/metrics.json` (pre-registered, gate per
condition, all three `k_d`), `results/a2_verdict_gated_baseline/<run>/` (gate off, Gate 3
protocol, both row sets), `results/a2_verdict_gated_pairgate/<run>/` (analysis only, see
§5). Gate 3 itself: `results/phase3a_verdict/20261109T090957Z_bc3a398`, report 04.

## Plain-language summary

**What the gate buys.** Acceptance rises from **0.71 to 0.95** at `k_d` = 0.295 — but the
number is taken over **59 testable clips instead of 83**, so 41 % of clips are now
abstentions (28 % gate, 13 % the pre-registered excitation rule). Almost all of the
surviving clips come from the two piloted flights: 01p and 07p contribute 51 of the 59.

**What it costs.** Rejection of the *half* tilt change falls from 0.37 to **0.11**, and of
×1.5 from 0.35 to **0.07**, because the gate drops 79 % of the tilt-removal edits while
dropping only 28 % of the real clips. The ranking is still good — AUROC 0.93 (×0.5), 0.89
(×1.5), 0.99 (mirrored roll) — but the operating point no longer rejects them. The
minimum-detectable-violation target (80 % rejection of anything) is **still not met** at any
`k_d`.

**Why it happens.** The gate is applied to each clip on its own pixels (amendment 2). An
edited frame is a frame the estimator's confidence was never calibrated on, so the gate
preferentially abstains on exactly the violations it should catch. §5 runs the variant where
the gate is applied to the *real* clip of a pair and inherited by its edits: acceptance is
the same 0.949, but tilt removal is rejected 0.78 on all 59 pairs instead of 0.75 on 8. That
variant is **not** pre-registered and is reported as analysis.

**Drag.** The conservative, drag-free claim (`k_d` = 0) gives slightly *lower* accept
separation than the racing-quad value: pooled AUROC 0.885 vs 0.918, unmatched 0.835 vs 0.867,
acceptance 0.966 (n = 58) vs 0.949 (n = 59). Which direction the number moves is a property
of the footage, not of the verifier: with drag, the implied axis tilts *with* the velocity
even in a straight line, which on these clips happens to sit closer to what the pixels show.

## 1. Reproduction of Gate 3 through the extension

Before anything was added, the extended code re-scored the original artifacts. This is the
baseline the gated numbers are measured against.

| run | rows | threshold policy | pooled AUROC | acceptance | records differing |
| --- | --- | --- | --- | --- | --- |
| Gate 3, original code | Gate 3 | `non_trackratm` | 0.8214 | 0.7108 (59/83) | — |
| extended code, Gate 3 rows | Gate 3 | `non_trackratm` | **0.8214** | **0.7108 (59/83)** | **0 of 600** |
| extended code, re-extracted rows | A2 | `non_trackratm` | 0.8210 | 0.7229 (60/83) | 516 of 600 |

On Gate 3's own rows the extended code reproduces the operating point **exactly**: identical
AUROC to ten decimal places, identical acceptance, zero of 600 clip-condition scores
differing. The 0.8210 / 0.7229 row is the same code on the **re-extracted** rows: 516 of 600
scores move (max 6.04°, a nuisance grid tie), which moves one clip across the threshold.
Amendment 4 records that with the same numbers. Run directories:
`results/a2_verdict_baseline_origrows/` and `results/a2_verdict_gated_baseline/`.

The Gate 3 report's "first run" and "after the pairing fix" numbers are **not in the
repository** — only the final run (`090957`) is. They are not reconstructed here or quoted
from memory.

## 2. The gated verdict

Pre-registered setting: gate per condition, conformal threshold refitted on all eight
flights after the gate, excitation rule unchanged, no flight excluded. Threshold in
parentheses.

| `k_d` | acceptance (n) | Wilson 95 % | threshold | pooled AUROC (paired) | pooled AUROC (unmatched) | yaw control AUROC |
| --- | --- | --- | --- | --- | --- | --- |
| 0.295 | **0.949** (59) | 0.861–0.983 | 6.46° | **0.918** | **0.867** | 0.667 |
| 0.15 | 0.966 (59) | 0.885–0.991 | 8.40° | 0.923 | 0.874 | 0.566 |
| 0 | 0.966 (58) | 0.883–0.990 | 12.95° | 0.885 | 0.835 | 0.607 |

Gate 3, for comparison, at `k_d` 0.295 with the old policy: acceptance 0.711 (83),
pooled 0.821, yaw 0.617.

"Paired" is Gate 3's convention (each edit against its own real clip, up to three times per
real clip); "unmatched" puts every testable real clip in the positive class once and every
edit in the negative class, which is the harder reading and is lower in every row.

### Rejection per edit family at the operating point

| family | `k_d` 0.295 | `k_d` 0.15 | `k_d` 0 | n (0.295) |
| --- | --- | --- | --- | --- |
| tilt removal (×0) | 0.750 (AUROC 1.000) | 0.750 (1.000) | 0.500 (1.000) | **8** |
| tilt ×0.5 | 0.111 (0.928) | 0.083 (0.899) | 0.057 (0.889) | 36 |
| tilt ×1.5 | 0.071 (0.892) | 0.071 (0.923) | 0.073 (0.860) | 56 |
| mirrored roll | 0.962 (0.990) | 0.962 (0.986) | 0.961 (0.999) | 52 |
| yaw control | 0.057 (0.667) | 0.038 (0.566) | 0.038 (0.607) | 53 |

**State next to every AUROC:** the yaw control — the same warp, no tilt change — still
scores 0.57–0.67, so that share of any AUROC is resampling artifact. Its rejection at the
operating point is 0.04–0.06, inside the 5 pp budget, which is the one Gate 3 criterion that
now passes.

### Minimum detectable violation

| `k_d` | violation median | rejection | n |
| --- | --- | --- | --- |
| 0.295 | 23.6° (×0) | 0.75 | 8 |
| 0.295 | 11.5° (×0.5) | 0.11 | 36 |
| 0.295 | 12.0° (×1.5) | 0.07 | 56 |
| 0 | 19.4° (×0) | 0.50 | 8 |
| 0 | 9.6° (×0.5) | 0.06 | 35 |
| 0 | 10.3° (×1.5) | 0.07 | 55 |

**No violation magnitude on this curve reaches 80 % rejection** at any `k_d`. Gate 3 reached
none either; the gate moves the largest-violation rejection from 0.37 to 0.75 while removing
90 % of that family's comparisons.

### Abstention

| group | abstained (all causes) | of which gate | of which excitation |
| --- | --- | --- | --- |
| overall | **0.41** | 0.28 | 0.13 |
| piloted | 0.214 | — | — |
| autonomous | 0.867 | — | — |
| trackRATM | 0.862 | — | — |

Per flight (identity condition, `k_d` 0.295): 01a 0/5 dropped by the gate (2 testable),
01p 0/26 (26), 04a 1/7 (2), 07a 7/7 (0), 07p 1/26 (25), 13a 5/11 (0), 13p 8/11 (3),
16p 6/7 (1).

### Gate drop rate by condition — the reason the rejection rates collapse

| condition | gate-dropped | testable | median score (testable) |
| --- | --- | --- | --- |
| real (identity) | 0.28 | 59 | 2.47° |
| tilt removal | **0.79** | 8 | 13.20° |
| tilt ×0.5 | 0.51 | 36 | 4.64° |
| tilt ×1.5 | 0.31 | 56 | 4.19° |
| mirrored roll | 0.34 | 53 | 17.66° |
| yaw control | 0.34 | 53 | 2.75° |

The gate is fitted on real clips and applied to edited ones; the edit changes the image in a
way the confidence was never calibrated on. That is a design consequence of amendment 2's
"gate each clip on its own pixels", and it is the single largest effect in this report.

## 3. Acceptance per flight (`k_d` 0.295)

| flight | accepted / testable | acceptance |
| --- | --- | --- |
| 01p-ellipse | 26/26 | 1.00 |
| 07p-lemniscate | 25/25 | 1.00 |
| 04a-ellipse | 2/2 | 1.00 |
| 01a-ellipse | 1/2 | 0.50 |
| 13p-trackRATM | 1/3 | 0.33 |
| 16p-trackRATM | 1/1 | 1.00 |
| 07a-lemniscate | 0/0 | — (all abstained) |
| 13a-trackRATM | 0/0 | — (all abstained) |

With 07a and 13a contributing no testable clips at all, "acceptance 0.949" is a statement
about the two piloted ellipse/lemniscate flights and a handful of autonomous clips.

## 4. What this supports

- The verifier **accepts well-behaved real footage at 0.95** once unreliably-estimated clips
  are declared not testable, on a gate that never sees mocap.
- The **ranking** survives the gate and the drag axis: 0.89–1.00 per family, 0.87 pooled
  unmatched at `k_d` = 0.295, and the yaw control's artifact bound is now inside budget.
- The **conservative drag-free claim** is quantified: same conclusions, slightly lower
  separation.

## 5. Analysis only: gate the pair, not each condition

Not pre-registered. The same frozen gate is evaluated on each clip's **real** frames and that
decision is inherited by its edits, so the comparison set is not filtered by the edit.

| quantity | per condition (pre-registered) | pair level (analysis) |
| --- | --- | --- |
| acceptance | 0.949 (59) | 0.949 (59) |
| pooled AUROC (paired / unmatched) | 0.918 / 0.867 | 0.911 / 0.911 |
| tilt removal rejected | 0.750 (n = 8) | **0.780 (n = 59)** |
| tilt ×0.5 rejected | 0.111 (36) | 0.220 (59) |
| tilt ×1.5 rejected | 0.071 (56) | 0.119 (59) |
| mirrored roll rejected | 0.962 (52) | 0.949 (59) |
| yaw control rejected | 0.057 (53) | 0.136 (59) |
| smallest violation at 80 % rejection | none | none |

The acceptance gain does not depend on gating the edit; only the power does. If the protocol
is revisited, this is the change to propose — and it would need its own pre-registration.

## 6. What this does not show

- **Not** that the verifier meets its own target. Acceptance 0.949 is one clip below 0.95 on
  59 clips, and the 80 %-rejection frontier is still not reached.
- **Not** a result on racing footage: trackRATM flights are 86 % abstained, so their
  behaviour is unmeasured, not measured and fine.
- **Not** comparable across `k_d` in the sense of a like-for-like threshold: each refits its
  own conformal quantile (6.46 / 8.40 / 12.95), so the acceptance rates are operating points,
  not a common cut.
- **Not** evidence that the yaw control's 0.57–0.67 AUROC is harmless in deployment; it is
  reported because any AUROC that ignores it is inflated.

## 7. Caveats

- The gate's own thresholds are leave-one-flight-out; the verdict's conformal thresholds are
  leave-one-flight-out as well, so no flight sets its own bar.
- Paired AUROCs with `n` = 8 (tilt removal) are close to meaningless as point estimates and
  are quoted only to show how few comparisons survive.
- The unmatched AUROC counts each real clip once; the paired one counts it once per family
  member. Both are reported because the difference is 0.05 at `k_d` = 0.295.
- Amendment 4: the gate rows differ from Gate 3's by the estimator's run-to-run noise; the
  measured effect on the baseline is one clip.

### Rejection: definitions, and what may be quoted

Two tables have been circulating and they are **not** the same quantity. Both are computed
here on the same records, so the difference is decomposed rather than argued (amendment 8):

- **Population** — *testable pairs*: both the real clip and its edit are testable, i.e. pass
  the gate **and** the excitation rule (this is the pre-registered set). *Gate-only pairs*:
  both pass the gate, excitation ignored.
- **Threshold** — *held-out*: each edit is compared with its own clip's leave-one-flight-out
  conformal threshold (5.76–12.95° depending on `k_d` and fold; this is the pre-registered
  operating point). *fixed*: a single 6.46° cut applied to every `k_d`.

| `k_d` | family | testable, held-out | testable, fixed | gate-only, held-out | gate-only, fixed |
| --- | --- | --- | --- | --- | --- |
| 0.295 | tilt ×0 | **0.75** (n=8) | 1.00 (8) | 0.70 (10) | 0.90 (10) |
| 0.295 | tilt ×0.5 | **0.11** (36) | 0.31 (36) | 0.13 (38) | 0.32 (38) |
| 0.295 | tilt ×1.5 | **0.07** (56) | 0.11 (56) | 0.09 (58) | 0.12 (58) |
| 0.295 | mirrored roll | **0.96** (52) | 0.98 (52) | 0.94 (54) | 0.96 (54) |
| 0.295 | yaw control | **0.06** (53) | 0.04 (53) | 0.07 (55) | 0.05 (55) |
| 0 | tilt ×0.5 | **0.06** (35) | 0.80 (35) | 0.08 (38) | **0.76** (38) |
| 0 | tilt ×1.5 | **0.07** (55) | 0.82 (55) | 0.07 (58) | **0.79** (58) |
| 0 | yaw control | **0.04** (52) | 0.42 (52) | 0.04 (55) | 0.42 (55) |
| 0 | mirror | 0.96 (51) | 1.00 (51) | 0.91 (54) | 0.98 (54) |

**The swing is the threshold, not the population.** At `k_d` = 0, tilt ×0.5 reads 0.06
(testable, held-out) → 0.80 (testable, fixed) → 0.08 (gate-only, held-out) → 0.76 (gate-only,
fixed): population moves it by 0.02, fixing the threshold at 6.46° while the correct
operating point is 12.95° moves it by 0.70.

**Headline: the bold column** — testable pairs at each clip's leave-one-flight-out conformal
threshold. The earlier "gate-only pairs at a fixed 6.46°" table is **withdrawn as a
cross-`k_d` comparison**; it mixed a threshold calibrated for one operating point with scores
from another, and the yaw-control row shows the cost (0.42 at a 6.46° cut against a ~0.04
budget). It survives only as the decomposition above. The gate-only/held-out column is a
legitimate sensitivity: it shows the gate itself, with excitation held out, does not drive
the rejection rates (differences ≤ 0.05 everywhere).

## 4. What this supports

- The verifier **accepts well-behaved real footage at 0.95** once unreliably-estimated clips
  are declared not testable, on a gate that never sees mocap.
- The **ranking** survives the gate and the drag axis: 0.89–1.00 per family, 0.87 pooled
  unmatched at `k_d` = 0.295, and the yaw control's artifact bound is now inside budget.
- The **conservative drag-free claim** is quantified: same conclusions, slightly lower
  separation.

## 5. Analysis only: gate the pair, not each condition

Not pre-registered. The same frozen gate is evaluated on each clip's **real** frames and that
decision is inherited by its edits, so the comparison set is not filtered by the edit.

| quantity | per condition (pre-registered) | pair level (analysis) |
| --- | --- | --- |
| acceptance | 0.949 (59) | 0.949 (59) |
| pooled AUROC (paired / unmatched) | 0.918 / 0.867 | 0.911 / 0.911 |
| tilt removal rejected | 0.750 (n = 8) | **0.780 (n = 59)** |
| tilt ×0.5 rejected | 0.111 (36) | 0.220 (59) |
| tilt ×1.5 rejected | 0.071 (56) | 0.119 (59) |
| mirrored roll rejected | 0.962 (52) | 0.949 (59) |
| yaw control rejected | 0.057 (53) | 0.136 (59) |
| smallest violation at 80 % rejection | none | none |

The acceptance gain does not depend on gating the edit; only the power does. If the protocol
is revisited, this is the change to propose — and it would need its own pre-registration.

## 6. What this does not show

- **Not** that the verifier meets its own target. Acceptance 0.949 is one clip below 0.95 on
  59 clips, and the 80 %-rejection frontier is still not reached.
- **Not** a result on racing footage: trackRATM flights are 86 % abstained, so their
  behaviour is unmeasured, not measured and fine.
- **Not** comparable across `k_d` in the sense of a like-for-like threshold: each refits its
  own conformal quantile (6.46 / 8.40 / 12.95), so the acceptance rates are operating points,
  not a common cut.
- **Not** evidence that the yaw control's 0.57–0.67 AUROC is harmless in deployment; it is
  reported because any AUROC that ignores it is inflated.

## 7. Caveats

- The gate's own thresholds are leave-one-flight-out; the verdict's conformal thresholds are
  leave-one-flight-out as well, so no flight sets its own bar.
- Paired AUROCs with `n` = 8 (tilt removal) are close to meaningless as point estimates and
  are quoted only to show how few comparisons survive.
- The unmatched AUROC counts each real clip once; the paired one counts it once per family
  member. Both are reported because the difference is 0.05 at `k_d` = 0.295.
- Amendment 4: the gate rows differ from Gate 3's by the estimator's run-to-run noise; the
  measured effect on the baseline is one clip.

### Rejection on pairs where both clips pass the gate

The per-family numbers above require both clips to be *testable*, which includes the
excitation rule. Isolating the gate, on pairs where the real and the edited clip both pass the
gate (threshold 6.46° applied to both):

| family | `k_d` 0.295 | `k_d` 0.15 | `k_d` 0 |
| --- | --- | --- | --- |
| tilt ×0 | 0.90 (n=10, AUROC 0.99) | 0.90 (10, 0.95) | 0.90 (10, 0.92) |
| tilt ×0.5 | 0.32 (38, 0.90) | 0.45 (38, 0.88) | 0.76 (38, 0.85) |
| tilt ×1.5 | 0.12 (58, 0.87) | 0.52 (58, 0.91) | 0.79 (58, 0.85) |
| mirrored roll | 0.96 (54, 0.99) | 0.96 (54, 0.99) | 0.98 (54, 0.99) |
| yaw control | 0.05 (55, 0.66) | 0.07 (55, 0.56) | 0.42 (55, 0.61) |

The comparison set is 3–7× larger than the testable-only one, and the ranking is unchanged
(0.85–0.99). Rejection at a fixed 6.46° cut rises with the drag-free setting because the
scores themselves rise, so the three columns are not operating points of a common threshold —
each `k_d` has its own conformal threshold (6.46 / 8.40 / 12.95).
