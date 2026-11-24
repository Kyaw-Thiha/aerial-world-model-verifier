# Track N3 state — path-anchored start for the propagated lean

Branch `track-n3-anchored-start`, worktree `~/Documents/Projects/awmv-n3`, branched from
`track-n2-sim-tiers` (`0baf445`). `git merge-base --is-ancestor track-n2-sim-tiers main` is
**false** at branch time, so the branch-from-N2 case of the handoff applies (not main).

Frozen inputs verified before any work: `configs/sim_splits_frozen.json` sha256
`cdf66e28387178be9767a2da1f5f24cb4a5c6df2e7a1c013ecdaa6166c120dbd` and
`configs/sim_prereg.yaml` sha256 `245b68bfb34991813636c224445e6d7ec683be0eedc8c2fe580ca6cee83e75fb`,
both matching the handoff. Neither was edited, and no frozen config was.

Data note: this worktree's `data/` is a symlink to the shared gitignored tree
(`~/Documents/Projects/aerial-world-model-verifier/data`), as `awmv-n` has it. Read-only use;
nothing under `data/` was written.

## Status — all parts done, nothing running

| part | state | report |
| --- | --- | --- |
| Pre-registrations N3A (+amendments A, B), N3B, N3C | done, committed before scoring | — |
| **Part A** — anchor availability and accuracy from truth (no video) | **done** | `reports/N3A_anchor_availability.md` |
| **Part B** — F0/C2/R on the anchor-propagated lean, 200 dev clips | **done**, descriptive (B1 gate fails) | `reports/N3B_anchored_tiers.md` |
| **Part C** — label-free drift flag | **done** | `reports/N3C_drift_flag.md` |
| **Part D** — fallbacks for clips with no anchor | **done** (was required: P1 dynamic coverage 0/131) | `reports/N3B_anchored_tiers.md` §Part D |
| Docs, ledger, tests, LaTeX candidate | done | `report_candidates/n3_anchored_start.tex` |

## Decisions in one line each

* **The anchor is accurate and unavailable.** 9/200 dev clips, 0/131 dynamic, 3/26 feasible forest
  at 2°/1 s; 0/14 UZH and 0/8 TII once rests on a support are excluded. The nine P1 anchors are the
  eight hover clips plus one accel-brake clip.
* **A quiet window cannot be told from a drone on a support** from the path alone without a takeoff
  test; all 12 raw UZH anchors are pre-takeoff rests with 0.33–9.24° of true tilt against ≤ 0.12°
  implied. Amendment B adds the airborne rule and applies it to real footage only.
* **Where an anchor exists the anchored lean is the best in the study** (F0 median 0.03° on the 16
  anchored feasible clips against the reader's 1.17° and a reader start's 1.08°, 0.01° from an
  oracle start) — and it detects 6/27 removals at the matched 8/80 because it scores only 16
  feasible and 21 defect clips. **No forest removal and none of the six drift-tail clips has an
  anchor at any tier ≤ 6°.**
* **The drift tail is half predictable from one draw** (`jit` 3/6 at a 10 % abstention budget),
  which drops the oracle arm's cut from 16.44° to 10.92° and raises removals 13/27 → 15/24; the
  pre-registered recall criterion fails, and the two-draw disagreement is better (8/8 forest).

## Spend

Ceiling 15 USD. Spend: **0.00 USD**. Everything is a CPU reanalysis of stored Track N inputs;
no Modal app, volume, R2 write, or detached job was created. Ledger: `docs/COST_LEDGER.md`.

## Resume / verify

```
cd ~/Documents/Projects/awmv-n3
PYTHONPATH=. pixi run python experiments/n3a_anchor.py        # ~20 s
PYTHONPATH=. pixi run python experiments/n3b_anchored_tiers.py  # ~12 s (asserts the N07 and N3A cross-checks)
PYTHONPATH=. pixi run python experiments/n3c_drift_flag.py    # ~17 s
PYTHONPATH=. pixi run python experiments/n3d_fallbacks.py     # ~2 s
PYTHONPATH=. pixi run pytest -q tests/test_anchor.py          # 15 tests
PYTHONPATH=. pixi run python experiments/n3_checks.py          # 107 report-number checks, expect 0 failed
```

`experiments/n3b_anchored_tiers.py` fails loudly if the reader/level/reader-start/oracle F0 values
stop matching `results/n2/n07/per_clip.csv`, or if its re-derived anchor frames stop matching
`results/n3/n3a/windows.csv`.

Full suite at this commit: `261 passed, 17 failed, 3 skipped`; all 17 failures are
`tests/test_generation_paths.py` and all are `FileNotFoundError:
third_party/anwm/anwm/config.py` — the gitignored ANWM clone, the documented pre-existing
dependency (README "Setup"). Nothing in N3 touches that module.

## Open items / what would be next

* The one measurement that would change the conclusion is **availability**: a generator or dataset
  with genuine in-air pauses. The accuracy side is settled (the anchored lean matches an oracle
  start on the clips that have one).
* The start remains a sensor/conditioning-input question. N2A priced a 2° external attitude and the
  present track shows the trajectory cannot supply it; nothing in N3 tests an IMU or a
  conditioning-pose readout.
* A mixture (anchor where available, reader elsewhere) was not pre-registered and was not run; the
  expectation from these numbers is that it changes only the hover clips.
* Held-out simulated clips and held-out real flights: never read by this branch.
