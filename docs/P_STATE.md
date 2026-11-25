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
| P0 error budget + stop-rule verdict | **done** — verdict: proceed. `reports/P0_error_budget.md` |
| P1 rate consistency (Check B) | **done** — pre-registered criteria C1-C3 FAIL, C4 PASS; stratified exploratory result. `reports/P1_rate_consistency.md` |
| P2 thrust envelope (Check C) | **done** — C1 FAIL (11/20 real flights), C2 PASS (13/13, 0/80). `reports/P2_envelope_gate.md` |
| docs / ledger / LaTeX / checks | **done** — 56/56 number checks pass |

P0 in one line: the 5 % floor is 3.0 deg/s (sim) and 1.8 deg/s (UZH) against required rates of
9.1 and 29 deg/s; 11 of 53 dynamic camera-defect clips are below it (all mirrored roll, 11/16),
20.8 % < 50 %, so Check B proceeds. The handoff-literal rate form leaves 8.7 deg/s on feasible
ground truth and is replaced by a per-clip gauge fit (0.0008 deg/s noiseless).

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

P1 in one line: at the matched 4/80 the pooled threshold is 40.2 deg/s (set by the 30/45 deg
mount DPVO drift) and detection is 2/19 removals, 1/16 mirrors, 0/22 scale; at the 0/15 deg
mounts (exploratory) 6/6, 4/10, 16/16 at 1/21. Real outdoor forward UZH passes (0/107, 0/109,
1/54); the 45-down rig fails (54/58); TII not evaluable; SolarWM arms indistinguishable.

P2 in one line: the physics gate (60 deg / 2.0 g / 720 deg/s) detects 13/13 thrust-limit paths
with 0/80 false positives but rejects 11/20 real flights; the real envelope is 122.5 deg / 6.05 g
/ 781 deg/s / 457 m/s^3 and nothing simulated or generated exceeds it.

## Spend

Ceiling 3 USD. Spend: **0.00 USD** (CPU only; no Modal app, no volume, no R2 write).

## Checks

`experiments/p_checks.py` — 56/56 pass. `tests/test_rate.py` — 10/10 pass.

## Resume

```
cd ~/Documents/Projects/awmv-p
PYTHONPATH=. pixi run python experiments/p0_error_budget.py
PYTHONPATH=. pixi run python experiments/p1_rate_consistency.py
PYTHONPATH=. pixi run python experiments/p2_envelope_gate.py
PYTHONPATH=. pixi run python experiments/p_checks.py
```

Open items: (1) the rate check needs a stable rotation estimate — an alternative extractor on the
30/45 deg mounts would be the single measurement that changes the conclusion; (2) a per-window or
drift-robust gauge was not pre-registered and was not run; (3) ANWM/LingBot rotations do not exist
locally, so the generated-clip arm test is SolarWM only; (4) held-out sim clips and TII 13p/07a
were never opened.
