# N3C — A label-free reliability flag for the high-mount drift tail

**Decision.** A single-draw rotation-increment roughness (`jit`, the 90th percentile of
`|w_k − w_{k−1}|` on the per-frame rotation increments) flags **3 of the 6** drift-tail clips at a
pre-registered 10 % abstention budget, i.e. **half** the tail, and abstaining on those 20 clips lets
the oracle-start F0 cut fall from **16.44° to 10.92°**, which raises its removals from 13/27 to
**15/24** and its forest removals from 4/9 to **6/7** at an unchanged false-reject rate (8/74 against
8/80). The pre-registered recall criterion (≥ 2/3 of the tail) **fails**; the flag is a weak,
high-mount-driven predictor, not a reliable gate, and no single-draw statistic tested separates the
tail cleanly.

Pre-registration `configs/n3c_drift_flag_prereg.yaml` (committed before any flag statistic existed).
Code `experiments/n3c_drift_flag.py`, results `results/n3/n3c/`. CPU, **USD 0.00**. All 200
development clips; no held-out clip read.

## Question

The verdict's F0 operating point is set by a handful of feasible clips whose propagated lean drifts
past 20°. Can a statistic from the same tracking run — no ground truth, no second draw — identify
those clips so the verdict can abstain on them and use a tighter cut on the rest?

## Setup

**Target set.** By the pre-registered rule the target is the anchored arm's own tail if it has ≥ 3
clips, otherwise the oracle-start tail. The anchored arm has **0** feasible clips above 20° — not
because the tail is cured but because it abstains on 163 of 200 clips (N3B §5) — so the fallback
applies: the target is the **6 feasible clips with an oracle-start F0 above 20°**,
`i02_dev_003, 019, 038, 043, 059, 063`, all at 30° or 45° mount, all dynamic.

**Candidates** (all from DPVO draw 1's rotation sequence, no truth, no reader):
`jit` = p90 of `|w_k − w_{k−1}|` with `w_k = angle(R_k^T R_{k+1})` (primary); `spike` = max
`|w_k − median(w)|`; `rough` = RMS residual of a degree-3 local fit to the rotation-vector path;
`cover` = `1 − n_pose/n_input`; `draw_dis` = the pre-registered two-draw disagreement on the anchored
arm (unusable — only 37 clips have an anchored value); `draw_dis_oracle` = the same disagreement on
the oracle arm, added as a labelled amendment so the two-draw reference is measurable.

**Threshold.** Flag the worst 10 % of all 200 clips (20 clips), label-free. 5 % and 20 % are
reported as a sensitivity curve. Nothing is tuned. **Abstention accounting**: an abstained clip gets
no verdict, so it leaves both the numerator and the denominator; both counts are printed.

## Results

### Recall and precision at the pre-registered budget

| flag | 5 % (10 clips) | 10 % (20 clips) | 20 % (40 clips) |
| --- | --- | --- | --- |
| `jit` (primary) | 3/6, precision 3/10 | **3/6, precision 3/20** | 4/6, precision 4/40 |
| `spike` | 0/6 | 3/6, precision 3/20 | 3/6 |
| `rough` | 0/6 | 0/6 | 0/6 |
| `cover` | 0/6 | 0/6 | 0/6 |
| `draw_dis` (pre-registered form) | 0/6 | 0/6 | 0/6 (37 clips have a value) |
| `draw_dis_oracle` (2 draws, amendment) | 1/6 | 3/6, precision 3/20 | 5/6, precision 5/40 |

`jit` catches `003, 019, 059` and misses `038, 043, 063`. `cover` is exactly uninformative — every
run completes (`n_pose = n_input` on all 200 clips) — so "track length" is not a usable signal here.
`rough` is uninformative at every budget. Everything is 6 nominal clusters; these are descriptive
counts, not rates with intervals.

### What the flag actually fires on

Of the 20 flagged clips: **14 at 45° mount, 3 at 30°, 2 at 15°, 1 at 0°** (and 18 of the 20 are
dynamic clips). The flag is largely a proxy for "this clip rotates a lot", which is why its precision
on the tail is 3/20 while the tail's base rate is 6/80. The three non-target feasible clips it flags
(`047`, `055`, `074`) are all high-mount and one of them (`055`) is at 16.4° — just under the tail
cut. There is no evidence that it identifies *tracking failure* as opposed to *fast rotation*.

### The operating point after abstaining

Applied to the oracle-start arm (the arm that has the tail), cut at the matched 8 pooled feasible
false-rejects among the surviving clips:

| | before | after abstaining on the 20 `jit` flags |
| --- | --- | --- |
| matched cut | 16.44° | **10.92°** |
| feasible false-rejects | 8/80 | 8/74 (6 feasible abstained) |
| tilt_removed | 13/27 | **15/24** (3 abstained) |
| mirrored_roll | 19/27 | 18/25 (2 abstained) |
| forest removals | 4/9 | **6/7** (2 abstained) |
| forest feasible | 2/26 | 2/23 (3 abstained) |
| high-mount feasible | 7/40 | 7/34 (6 abstained) |

The same treatment with the two-draw reference (`draw_dis_oracle`, 20 flagged) gives a cut of 9.81°,
18/25 removals and **8/8** forest removals (1 abstained): the two-draw disagreement is the stronger
predictor, as N06/N08 would lead one to expect, and it costs a second pose run.

### Criteria, scored

| criterion | result |
| --- | --- |
| C1: `jit` at the 10 % budget flags ≥ 2/3 of the target | **FAIL**: 3/6 |
| C2: after abstaining, cut ≤ 12° and removals fall by ≤ 2 | **PASS**: 10.92°, removals 13 → 15 (of 24 judged) |
| C3: ≤ 8 of 26 forest feasible clips abstained | **PASS**: 3/26 |

## What this supports

* **The tail is partly predictable from one tracking run and mostly not.** Half of it is caught by
  a rotation-increment roughness at a 10 % abstention cost; the other half looks like smooth drift,
  which is exactly N06's diagnosis (run-to-run variance, not a visible local artefact).
* **Removing the tail is worth a lot to the verdict.** The oracle-start cut falls 5.5° and forest
  removals go from 4/9 to 6/7 at the same false-reject rate — the tail really is what sets the
  operating point, as N2A predicted.
* **A two-draw disagreement is a better flag than any single-draw statistic tested** (8/8 forest
  removals after abstaining), at double the pose cost.
* **No cheap single-draw gate is reliable enough to deploy as an abstention rule**: 3/20 precision
  means 17 wasted abstentions, half of them on defect clips whose detections are then lost.

## What this does not show

* Whether a *combined* statistic (jitter × mount × dynamic) would do better — deliberately out of
  scope (a second chance was pre-registered as excluded).
* Whether the flag transfers to real footage or to the anchored arm (which has no tail to protect).
* Anything about the held-out clips (never read).

## Caveats

* Six target clips, 20 abstentions: every number here is a count, and the difference between 3/6 and
  4/6 is one clip. No interval is claimed.
* The pre-registered `draw_dis` form is unusable because the arm it was defined on (the anchored arm)
  scores only 37 clips; the oracle-arm version is reported as a labelled amendment, and it is the one
  that should be read.
* "After abstaining" holds the *count* of false-rejects at 8 with a smaller denominator (8/74), i.e.
  the rate rises slightly (10.0 % → 10.8 %). Abstention is not free; it is reported both ways.
* The flag is computed on the rotation increments of the whole clip; a clip that is dynamic only in
  one interval gets one flag for the whole clip.

## Provenance

* Pre-registration `configs/n3c_drift_flag_prereg.yaml`; decisions `[N3-c]`.
* Inputs: `data/n05/poses/sim200/{scale1,scale1_rep2}` (rotation sequences and run metadata only),
  `results/n3/n3b/{per_clip.csv,per_clip_draw2.csv}` (F0 by arm), `results/n2/inputs/manifest.csv`.
* Code `experiments/n3c_drift_flag.py` (about 17 s); results `results/n3/n3c/{flags.csv,
  sensitivity.csv,summary.json}`.
* Compute: local CPU, **USD 0.00**. No Modal app, no volume, no R2 write.
