# Track P state — rate consistency (Check B) and thrust envelope (Check C)

Branch `track-p-rate-envelope`, worktree `~/Documents/Projects/awmv-p`, branched from
`track-n3-anchored-start` (`e3ef1d6`). That branch exists locally and on `origin`; it is not an
ancestor of `main`, so the N3 tip is the base the handoff names first.

Frozen inputs verified before any work (read from `track-i-simset`, which is not an ancestor of
this branch, so the files are not in `configs/` here):
`configs/sim_splits_frozen.json` sha256
`cdf66e28387178be9767a2da1f5f24cb4a5c6df2e7a1c013ecdaa6166c120dbd` and
`configs/sim_prereg.yaml` sha256
`245b68bfb34991813636c224445e6d7ec683be0eedc8c2fe580ca6cee83e75fb`, both matching the handoff.
Neither was edited; no frozen config exists on this branch and none was added under those names.

Data note: this worktree's `data/` is a symlink to the shared gitignored tree
(`~/Documents/Projects/aerial-world-model-verifier/data`). Read-only use.

## Status

| item | state |
| --- | --- |
| P0 error budget + stop-rule verdict | not started |
| P1 rate consistency (Check B) | not started |
| P2 thrust envelope (Check C) | not started |
| docs / ledger / LaTeX / checks | not started |

## Readings taken (the user is unavailable)

1. **Generated clips.** The handoff says H04 holds 234 clips from ANWM, LingBot-World and
   SolarWM. The inventory in `results/n04/clips.csv` is 234 **SolarWM** clips (22 paths x 3 arms
   A/B/C x 3 draws + swap paths), and `data/n04/poses/gen/fov90` holds DPVO rotations for exactly
   those 234. N04 measured ANWM two-frame rotation as usable on 20/80 clips and reported LingBot
   frames as not local. Part 3 therefore scores SolarWM (234 clips) and reports ANWM and LingBot
   as not measurable from cached rotations, with the reason and the cost of computing them.
2. **UZH real false-positive population.** The handoff says "UZH-FPV (4 sequences)". The outdoor
   UZH sequences are exactly four (`outdoor_forward_1/3/5`, `outdoor_45_1`); that is the primary
   real population, and all ten indoor sequences are reported descriptively.
3. **TII held-out flights.** `13p` and `07a` are held out and are never opened here; the six
   development flights are `01a, 04a, 13a, 01p, 07p, 16p`.

## Spend

Ceiling 3 USD. Spend: 0.00 USD (CPU only; no Modal app, no volume, no R2 write).

## Resume

```
cd ~/Documents/Projects/awmv-p
PYTHONPATH=. pixi run python experiments/p0_error_budget.py
PYTHONPATH=. pixi run python experiments/p1_rate_consistency.py
PYTHONPATH=. pixi run python experiments/p2_envelope_gate.py
```
