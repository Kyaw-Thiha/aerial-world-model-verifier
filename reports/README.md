# Track B reports

One file per gate or handoff, numbered `bNN` (Track A uses its own numbering in the same folder).

| file | covers | status |
| --- | --- | --- |
| `b01_anwm_feasibility_redo.md` | what the ANWM release actually is: repo, weights, dataset, action format, undefined dt, measured inference cost | final |
| `b02_corrected_path_audit.md` | corrected required-tilt audit, heading statistic, drag coefficient split, three input regimes, design reachability | final |
| `b03_gate2_native_pipeline.md` | Gate 2: five native test rollouts, LPIPS, non-collapse, pose-convention check | final (pass) |
| `b04_clip_set.md` | the 82-clip set: composition, start filter, coverage, seed variance, first-vs-final delivery | final |
| `b05_pose_injection_stage0.md` | pose-injection conventions verified before generation (homography, thrust axis, signs, coverage, determinism) | final, with a same-day correction on determinism |
| `b06_pose_injection_stage1.md` | the injected-true arm and the Checkpoint 2 materials | pending |

Living specifications stay in `docs/`: `docs/track_b_clips.md` (level set),
`docs/track_b_clips_injected.md` (injected set), `docs/anwm_feasibility.md`, `docs/decisions.md`.
