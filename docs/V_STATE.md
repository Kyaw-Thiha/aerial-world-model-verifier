# Track V state — vertical-motion fidelity

Updated: 2026-11-25 (handoff 31, unattended run).

## Reading taken where the handoff left a choice

1. **The check is EyeHeight's height ratio, extended to commanded vertical motion.** Primary
   statistic `e = log(h(t2)/h(t1)) - log(ratio_cmd)` over 3 s windows; a 5 s window and a
   scale-free `rho = dz/L` version are pre-registered secondaries/exploratory, because the
   commanded ratio needs a start height that generated clips do not have.
2. **Pi3X is the estimator** (Pi3's successor, ungated, already in this project), every 4th
   frame at 7.5 Hz, reconstructed in overlapping 8 s chunks so density does not fall with clip
   length. Ground plane per window from the reconstruction's own points.
3. **TII is not in the real population** (fisheye distortion not carried by the reader feeds);
   UZH-FPV is.
4. **Vertical defects are re-renders** that edit the camera's altitude on the frozen path and
   leave the commanded attitude, so the tilt check sees a feasible clip.

## Outcome in one line

**The pre-registered stop rule fired.** The estimator's error on real UZH-FPV outdoor footage is
median |log ratio error| 0.50, 4-8x the smallest labelled vertical defect (0.06-0.11) on 28/28
climb clips, so V1 (defect renders, real false-reject table, generated clips) was not run. On
the rendered clips the same estimator is accurate (median 0.05-0.1), so the failure is the
geometry of low, fast, outdoor footage: the ground normal is known to ~2-8 deg and the ratio
error scales as delta * L / h, with L/h ~ 0.4-1.3 on rendered 12-60 m flights and 8-12 on the
1-2 m real ones.

## Done

* Frozen hashes verified on `main` (`cdf66e28…`, `245b68bf…`); worktree `awmv-v`, branch
  `track-v-vertical` from `8f404af`.
* Part 0a counts: 28/200 development clips have commanded vertical motion (all `climb_turn`,
  9.3-48.4 m, eight feasible), spread over three families and up to four mounts; the three UZH
  outdoor sequences have 0.8-1.6 m (median) vertical change per 3 s.
* Pre-registrations: `configs/v00_feasibility_prereg.yaml` + amendments 1-4,
  `configs/v10_vertical_prereg.yaml` + amendment 1 (both committed before any score; V1 stays
  unrun).
* Estimator and analysis: `src/vertical/{plane,metrics,recon_io,local_windows}.py`,
  `remote/v_recon.py`, `remote/v_render.py`, `experiments/v_stage_frames.py`,
  `v_stage_generated.py`, `v_stage_defects.py`, `v00_feasibility.py`, `v00_figure.py`,
  `v10_score.py`; unit tests `tests/test_vertical_{plane,metrics}.py`.
* Inputs staged: 44 Part 0/Part 2 units (30 simulated + 14 UZH) at 7.5 Hz on the volume;
  16 ANWM and 18 SolarWM climb clips pulled and staged (unrun, Part 3).
* One null render tested end to end (`i02_dev_005`, 364 frames, 0.40 s/frame on A10G,
  rendered ratio exactly 1.0) — the Part 1 renderer works and was not scaled up.

## Finished

* The amended pilot ran to completion (44/44 units). `experiments/v00_feasibility.py` produced
  the final numbers; `experiments/v00_rigs.py` the exploratory 14-rig table;
  `experiments/v00_figure.py` the evidence figure.
* The stop rule fired (condition A, 28/28); V1 was not run. Report `reports/V0_feasibility.md`,
  candidate section `report_candidates/v_vertical.tex`.
* All Modal apps are stopped; billing checked (5.02 USD). The branch is pushed and **not
  merged**.

## Spend

Ceiling 15 USD. **Spent 5.02 USD** (`modal billing report`, by app name: `awmv-v-recon` 4.83,
`awmv-v-render` 0.19). About 1.0 USD of that was lost to two concurrent code versions and about
0.2 USD to the null-render test.

## Resume commands

```
cd ~/Documents/Projects/awmv-v
MB=~/Documents/Projects/aerial-world-model-verifier/.pixi/envs/default/bin/modal
PY=~/Documents/Projects/aerial-world-model-verifier/.pixi/envs/default/bin/python
PYTHONPATH=. $MB run remote/v_recon.py --all-units
PYTHONPATH=. $PY scripts/pull_volume.py awmv-v-data recon data/v/recon
PYTHONPATH=. $PY experiments/v00_feasibility.py
PYTHONPATH=. $PY experiments/v00_figure.py outdoor_forward_1
```
