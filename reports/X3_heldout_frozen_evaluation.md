# X3 — One-shot held-out evaluation: not run

Track X, Stage 4, 2026-11-25. Frozen design `configs/x_final_frozen.yaml`
(sha256 `0d10597bd8b7d052…`, committed at `9d604ae`). Development evidence:
`reports/X2_final_design_dev.md`, `reports/X1_generator_breadth.md`.

## Decision

**The one-shot held-out run did not happen, and no held-out number is reported.** The frozen
pipeline is a composition of components whose development numbers already exist in their own
tracks (G06/G07 for the check and its thresholds, N3B/N3D for the anchored start and the
abstention, N3C for the drift flag), but the composition itself — one script that emits
window-level coverage, clip-level verdicts and denominators per family and mount, on the frozen
populations — was not written and validated in this run's window. Running the held-out set through
an unvalidated composition would have spent the one-shot read on a pipeline whose development pass
had not been done in this track, so it was not started. Nothing was read from
`results/g07/scores/` or from the held-out TII flights.

## Question

Does the frozen pipeline meet its pre-registered criteria (D1 false-reject, D2 detection, D3
coverage) on the 100 held-out simulated clips and the two held-out TII flights, development beside
held-out, with the yaw control and the null beside every rate?

## Setup

Frozen as written: Check A on a lean propagated from a path anchor (abstain where none), drift-flag
abstention at a 10 % budget, `coh12_p1` windows (12 frames, hop 4), clip-level BH q = 0.10,
thresholds from `configs/coherence_frozen.yaml` with no recalibration, criteria D1-D4 and H1 from
the frozen file. Populations: `field_settlement` and `coastal_cliff` (100 clips), TII
`flight-13p-trackRATM` and `flight-07a-lemniscate`.

## Results

None. No held-out clip was opened, no held-out flight was scored, and no criterion is reported as
met or unmet.

## What it supports

That the freeze exists and is auditable: the design, its thresholds and its criteria were committed
before any held-out read, and their sha256 is in `docs/decisions.md`. A later run can therefore
execute the frozen pass once, with no researcher degree of freedom left in it.

## What it does not show

Anything about generalisation. In particular it does not show that the pipeline works on the two
held-out scene families or on the two held-out TII flights.

## Caveats

* The held-out read is still unspent. `docs/X_STATE.md` carries the exact resume command.
* The development composition is the blocking item, not the held-out data: the components'
  development numbers are published, but they were produced by different scripts with different
  conventions, and composing them without a single script that emits one table would risk silently
  mixing denominators.
* If a later run changes any threshold after seeing a held-out outcome, the change is exploratory
  by the project's rule and must be labelled so.
