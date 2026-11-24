# N3D — The no-anchor fallback: abstention, and what it costs

**Decision.** On the pre-registered rule (a clip with no anchor **abstains** — no reader start, no
level start, no substitute) the anchored verdict gives **no answer on 191 of the 200** simulated
development clips at the primary tier, **on all 131** dynamic clips at both the 2° and 4° tiers, and
its detection ceiling is **0/27 tilt removals and 0/27 mirrors at P1** and 6/27 and 6/27 at P3 — so
the anchor cannot serve as a fallback instrument for the reader: it is a precise instrument that is
silent exactly where the verdict is needed.

Pre-registration: the `downstream_contract` of `configs/n3a_anchor_prereg.yaml` and the abstention
rule of `configs/n3b_anchored_tiers_prereg.yaml`. Part D was required because N3A's primary tier
covers 0 of the 131 dynamic clips (below the pre-registered 60 % gate). Code
`experiments/n3d_fallbacks.py`, results `results/n3/n3d/`. All 200 development clips; no held-out
clip read. CPU, **USD 0.00**.

## Question

Given that some clips have no anchor, what verdict do they get, and what does abstaining cost?

## Setup

Every clip of the 200-clip development set, the three pre-registered anchor tiers (2°, 4°, 6° held
for 1 s), and the pre-registered fallback: **abstain**. `unreachable` counts the defect clips that
have no anchor and therefore can never be flagged by the anchored arm — the complement is its
detection ceiling.

## Results

| tier | clips with no anchor | dynamic clips with no anchor | unreachable removals | unreachable mirrors | unreachable scale edits |
| --- | --- | --- | --- | --- | --- |
| P1 (2°, 1 s) | **191/200** | **131/131** | **27/27** | **27/27** | 33/33 |
| P2 (4°, 1 s) | 175/200 | 131/131 | 25/27 | 24/27 | 30/33 |
| P3 (6°, 1 s) | 163/200 | 129/131 | 21/27 | 21/27 | 27/33 |

By condition, the clips that *do* have an anchor are almost all feasible: at P1 the scoreable set is
9/80 feasible and **0/27 removals, 0/27 mirrors, 0/33 scaled, 0/20 partial, 0/13 thrust-limited**; at
P3 it is 16/80 feasible and 6/27, 6/27, 6/33, 3/20, 0/13. By family the scoreable fraction is
3/66 · 3/67 · 3/67 at P1 and 5/66 · 15/67 · 17/67 at P3 (forest · industrial · urban); by mount it is
5/50 at 0°, 0/50 at 15°, 4/50 at 30°, 0/50 at 45° at P1 and 19/50, 6/50, 6/50, 6/50 at P3.
`results/n3/n3d/coverage.csv` is the per-clip table.

## What this supports

* **Abstaining is the honest answer, and it is also a hard limit.** With no reader start substituted,
  the anchored arm's ceiling on removals is 0/27 at the primary tier: two thirds of the defect set
  simply cannot be judged. Reporting detection rates for this arm at P1 without the abstention count
  would be misleading; every table in N3B prints it.
* **The gap is structural.** The unreachable clips are the dynamic, high-mount, forest clips — the
  ones the reader is biased on and the ones the verifier exists to check (N3B §1, §5).
* **A mixture would inherit the limit.** An "anchor where available, reader elsewhere" scheme was not
  pre-registered and was not measured; Part D shows only that the anchored half contributes nothing
  outside the 9–37 clips it can see.

## What this does not show

* Any fallback other than abstention (that is the point: none was pre-registered and none is
  evaluated).
* What a deployment with genuine in-air hovers would look like — there the scoreable fraction is a
  property of the flight, not of this dataset.
* Anything about the held-out clips (never read).

## Caveats

* The rates here are coverage counts, not accuracies; the accuracy of the clips that *are* judged is
  in N3B §3.
* The tier label ("P1/P2/P3") is a threshold on the implied tilt, so a looser tier always covers more;
  the table is not a tuning surface but a statement of what each pre-registered tier can reach.

## Provenance

* Rules: `configs/n3a_anchor_prereg.yaml` (`downstream_contract`),
  `configs/n3b_anchored_tiers_prereg.yaml` (`abstentions`); decisions `[N3-a]`, `[N3-c]`.
* Inputs: `results/n3/n3a/windows.csv` (anchor flags), `results/n3/n3b/per_clip.csv` (conditions),
  `results/n2/inputs/manifest.csv` (family, mount, peak tilt).
* Code `experiments/n3d_fallbacks.py` (about 2 s); results `results/n3/n3d/{coverage.csv,
  summary.json}`. Headline numbers re-checked by `experiments/n3_checks.py`.
* Compute: local CPU, **USD 0.00**. No Modal app, no volume, no R2 write.
