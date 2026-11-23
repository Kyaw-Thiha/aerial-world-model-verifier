# Track N state — attitude from video rotation

Branch `track-n-rotation`, worktree `~/Documents/Projects/awmv-n`, off `main` `6bc7b5d`
(consolidation wave 1). **Not merged.** Read-only dependencies: `track-j-reader` (J01 rotation
numbers), `track-e-depicted-uzh` (UZH Pinhole frames, v3 clips, reader JSONL), `track-i-simset`
(the frozen simulated split and its 300 clips), `track-l-reader-simreal` (L01 reader and the
verified 80-clip mirror), `track-m-gates` (H04/ANWM mirrors) and `track-h3-reader-validity`
(H04 path table and reader JSONL). Nothing was merged or rewritten.

## Reports

| report | what it settles |
| --- | --- |
| `reports/N01_rotation_pretest.md` | Part 0 PASS: 10 s propagated lean 1.80° in forest, 0.43° outdoors, against reader errors of 8.5°/8.4°; mount-dependent (30°/45° mounts 8.62°/14.83°); TII dead. |
| `reports/N02_initial_gravity_and_translation.md` | Part 1 start arms (the start dominates; a single reader frame is the best implementable start) and the translation-direction addendum FAIL (forest 18.36° at ≥8 m/s; UZH 1.78° but the 3-sequence stratum cannot meet the 5-clip floor; turns not supported). |
| `reports/N02b_propagated_lean_verdicts.md` | Descriptive Part 2 on the UZH v3 edits: C2 absorbs the propagated start (3.63° vs 8.57° real) but no tier detects better; the 320×240 crop fails extraction on 3/9 clips. |
| `reports/N03_agreement_gate_and_coverage.md` | Descriptive Part 3: the reader-vs-propagation gate collapses to 5/35 dynamic clips and detects nothing; Gate B was never computed on UZH. |
| `reports/N04_generated_video_rotation.md` | Part 4: all 234 SolarWM clips track (81/81 poses); propagated tilt 3.54/4.40/6.69° at 1/2/5 s vs a per-frame reader of 9.07°; ANWM 20/80 usable. |

## Round 2 (2026-11-23)

* **N05** (pre-registered `configs/n05_sim_verdict_prereg.yaml`): C2 and coherence on the
  propagated lean over all 200 simulated dev clips, with a LEVEL-BODY start (`u_cam(0)=R_cb^T e3`,
  no oracle, no reader) versus the reader lean, at matched false-reject. Held-out not read.
* **N06** (pre-registered `configs/n06_mount_drift_prereg.yaml` + amendment A): the 30/45° mount
  drift diagnosed across resolution (640×360 vs 320×180) and a two-view relative-pose chain.
* **N04 clarification**: the report now states the oracle start exactly and that the reader's
  9.07° is a per-frame (zero-horizon) median, so the like-for-like column is the propagated 1 s.

## Spend

Ceiling 30 USD. Spent before round 2: 0.41 USD (N01 0.25, N04 0.13, N02b 0.03). Round 2 estimate
7.0 USD (reader 200 clips + DPVO 200 + N06). TII is excluded by instruction: no TII job was run
in round 2.

## Open items / honest limits

* No held-out simulated clip has been read; the 100 held-out clips remain untouched.
* The simulated set has no yaw-warp family, so the yaw control is not measurable on sim; the
  measured yaw control is the UZH one (N02b), where propagation did **not** remove it.
* Part 2/3 remain descriptive: three UZH sequence clusters, not the confirmatory TII population.
* The translation-direction addendum's evaluability floor was mis-specified for a 3-sequence
  stratum; recorded, not repaired after the result.
