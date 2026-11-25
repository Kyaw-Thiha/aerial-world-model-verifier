# Track V state — vertical-motion fidelity

Updated: 2026-11-25 (handoff 31, unattended run).

## Reading taken where the handoff left a choice

1. **The check is the height-ratio check, with a scale-free secondary.** The commanded side is
   `ratio_cmd = z(t2)/z(t1)` (simulator, ground at z = 0) or `1 + (z(t2)-z(t1))/(s h(t1))`
   (real footage, one scale per chunk). The scale-free `rho = dz/L` version is pre-registered
   as *exploratory* because the primary needs a start height that generated clips do not have.
2. **The estimator is Pi3X** (Pi3's successor, ungated, already in this project), every 4th
   video frame at 7.5 Hz, reconstructed in overlapping 8 s chunks (amendment 2) so that the
   density does not fall with clip length. VGGT is a cross-check only if Pi3X fails its own
   controls.
3. **TII is not in the real pilot**: fisheye 640x480 with a distortion model the reader feed
   does not carry; recorded as a limitation, not worked around.
4. **Vertical defects are re-rendered, not video-warped**: the camera's altitude is edited on
   the frozen path and the camera attitude stays the commanded path's flatness attitude, so the
   tilt check sees a feasible clip and only a vertical check can react.

## Done

* Frozen hashes verified on `main` before the worktree was created: `configs/sim_splits_frozen.json`
  `cdf66e28…`, `configs/sim_prereg.yaml` `245b68bf…`.
* Branch `track-v-vertical`, worktree `~/Documents/Projects/awmv-v`, from `main` (`8f404af`).
* Part 0 counts (from `results/i02/manifest.csv` + local gt): **28 of 200 dev clips have
  commanded vertical motion**, all of them `climb_turn`; peak |dz| 9.4-48.4 m from a 12 m start
  (ratio 1.8-5.0); `forest_ridge` contributes 3, `industrial` 16, `urban_checker` 9; all four
  mounts (0/15/30/45 deg uptilt) appear for industrial and urban, forest only 15/45.
* Pre-registrations committed before any output: `configs/v00_feasibility_prereg.yaml`
  (`2e948fa`), amendments 1 and 2 (`cf8b0eb`).
* Frames staged at 7.5 Hz and uploaded: 33 units (30 sim + 3 UZH outdoor), volume
  `awmv-v-data:/v_frames`, 4,271 PNGs, 406 MB.
* Code: `src/vertical/{plane,metrics}.py`, `remote/v_recon.py` (Pi3X + plane fit),
  `experiments/v_stage_frames.py`, `experiments/v00_feasibility.py`,
  `src/sim/vertical_edits.py`, `remote/v_render.py` (Part 1 renders).

## Running

* `remote/v_recon.py --all-units` (33 units, L40S): Part 0 reconstructions.

## Next

1. Pull `recon/` from `awmv-v-data`, run `experiments/v00_feasibility.py`, apply the stop rule.
2. Write `reports/V0_feasibility.md` (or stop).
3. If it passes: `configs/v10_vertical_prereg.yaml`, run `remote/v_render.py --all-specs`
   (38 renders), reconstruct the edit clips, score Part 1; reconstruct the 14 UZH sequences for
   Part 2 false positives; identify and reconstruct the generated clips for Part 3.
4. `reports/V1_vertical_fidelity.md`, `report_candidates/v_vertical.tex`, ledger, decisions.

## Spend

Ceiling 15 USD. Committed so far: Pi3X image build (cached) + ~5 probe chunk passes, about
0.15 USD. Estimate for Part 0: 0.8-1.2 USD (33 units x ~3 chunks x 10-20 s on L40S at
1.95 USD/h). Part 1 renders: about 2.0 USD (38 clips x ~400-650 frames at 0.19 s/frame).
Part 1 + Part 2 reconstructions: about 1.8 USD. Total projected: about 5 USD.

## Resume commands

```
cd ~/Documents/Projects/awmv-v
MB=~/Documents/Projects/aerial-world-model-verifier/.pixi/envs/default/bin/modal
PYTHONPATH=. $MB run remote/v_recon.py --all-units          # Part 0 reconstructions
PYTHONPATH=. $PY scripts/pull_volume.py awmv-v-data recon data/v/recon
PYTHONPATH=. $PY experiments/v00_feasibility.py
```
