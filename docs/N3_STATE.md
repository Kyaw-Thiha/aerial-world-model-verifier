# Track N3 state — path-anchored start for the propagated lean

Branch `track-n3-anchored-start`, worktree `~/Documents/Projects/awmv-n3`, branched from
`track-n2-sim-tiers` (`0baf445`). `git merge-base --is-ancestor track-n2-sim-tiers main` is
**false** at branch time, so the branch-from-N2 case of the handoff applies (not main).

Frozen inputs verified before any work: `configs/sim_splits_frozen.json` sha256
`cdf66e28387178be9767a2da1f5f24cb4a5c6df2e7a1c013ecdaa6166c120dbd` and
`configs/sim_prereg.yaml` sha256 `245b68bfb34991813636c224445e6d7ec683be0eedc8c2fe580ca6cee83e75fb`,
both matching the handoff. Neither was edited.

Data note: this worktree's `data/` is a symlink to the shared gitignored tree
(`~/Documents/Projects/aerial-world-model-verifier/data`), as `awmv-n` has it. Read-only use.

## Status

| part | state | report |
| --- | --- | --- |
| Pre-registrations N3A, N3B, N3C | written and committed before scoring | — |
| Part A — anchor availability and accuracy | running | `reports/N3A_anchor_availability.md` |
| Part B — F0/C2/R on the anchor-propagated lean | pending | `reports/N3B_anchored_tiers.md` |
| Part C — drift reliability flag | pending | `reports/N3C_drift_flag.md` |
| Part D — fallbacks | conditional on Part A coverage | — |

## Spend

Ceiling 15 USD for this track. Spend so far: **0.00 USD**. Everything here is a CPU reanalysis of
stored Track N inputs (`data/n05/{gt,scores,poses}`, `data/n01/poses`, `data/uzh`, `data/clips_v3`);
no new reader run, no new DPVO run, no Modal app, no R2 write. Ledger: `docs/COST_LEDGER.md`.
