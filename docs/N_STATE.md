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

* **N05 (done).** `reports/N05_sim_propagated_verdicts.md`. C2 and coherence on the propagated
  lean over all 200 dev clips with a **level-body start, no oracle, no reader**, versus the reader
  at matched false-reject (7/80). Result: the propagated arm loses the verdict — C2 0/27 removals
  (draw 2: 1/27) against the reader's 16/27, 7/27 mirrors against 10/27. Cause: a fixed body-frame
  start error is not a yaw-frame lean offset (true-rotation C2 residual 8.54° on feasible clips),
  so the matched threshold rises to 29.5°. With an oracle start the same tier detects 6/27
  removals, so the propagation is sound and the missing piece is an absolute anchor. Coherence is
  undefined on the 60 % of feasible clips with a constant required lean (reported on the
  varying-lean subset as a post-hoc diagnostic). Two DPVO draws agree.
* **N06 (done).** `reports/N06_mount_drift_diagnosis.md`. The 30/45° drift is dominated by DPVO
  run-to-run variance (two draws: 30° median 4.58/5.87°, |diff| 2.83°; 45° 12.17/10.37°, |diff|
  8.78°, max 42.95°). Half resolution worsens it; a two-view relative-pose chain is unusable
  (70-85°); full 640×480 fixes the UZH edited-clip extraction completely (63/63 vs 12/63 at the
  320×240 crop).
* **N04 clarification (done).** The report states the oracle start exactly (the true commanded
  c2w at frame 0, i.e. the generator's conditioning pose — available in deployment from the
  conditioning input or the flight controller, but not free of error) and that the reader's 9.07°
  is a per-frame (zero-horizon) median, so the like-for-like column is the propagated 1 s.

## Spend

Ceiling 30 USD. Round 1 spend 0.41 USD; round 2 spend **4.04 USD** (reader 2.83, DPVO 1.21);
total **4.45 USD**. TII is excluded by instruction: no TII job was run in round 2.

## Open items / honest limits

* No held-out simulated clip has been read; the 100 held-out clips remain untouched.
* The simulated set has no yaw-warp family, so the yaw control is not measurable on sim; the
  measured yaw control is the UZH one (N02b), where propagation did **not** remove it.
* Part 2/3 remain descriptive: three UZH sequence clusters, not the confirmatory TII population.
* The offset-free rationale holds only for small start errors; N05's level start (10-30° wrong)
  is not absorbed. A tier comparison with a good absolute start (1-3°) was not run and is the
  next question.
* The translation-direction addendum's evaluability floor was mis-specified for a 3-sequence
  stratum; recorded, not repaired after the result.
