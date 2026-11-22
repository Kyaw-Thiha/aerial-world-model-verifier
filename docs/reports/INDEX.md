# Index of track reports

Every numbered report from every track, one line each, with its owning track and status.
Report text is **not** edited by the consolidation; where a report is superseded or contains
stale sections, that is recorded here and in `report/CHECKS.md` / `report/MISSING.md`.

Files live in `reports/` at the repository root. Numbering restarts per track: Track A/A2 uses
`NN_`, Track B `bNN_`, Track C `CNN_`, Track D `DNN_`, Track E `ENN_`.

`reports/README.md` is Track B's own short index of its `bNN` files and is kept as written.

## Track A — perception (branch `track-a-perception`, HEAD `164c5e0`)

| report | title | status |
| --- | --- | --- |
| `01_track_a_phase0_floor.md` | Phase 0 follow-ups: mocap physics floor and the oracle ladder | current |
| `02_track_a_phase1_depicted_attitude.md` | Phase 1: the depicted attitude (Gates 1a and 1) | current |
| `03_track_a_phase2_implied_tilt.md` | Phase 2: implied tilt from the recovered path, oracle ladder | current — contains its own correction section |
| `04_track_a_gate3_verdict.md` | Gate 3: the verdict test (Phase 3a) | current; its pre-registration sha256 is a *historic* value, see note below |
| `05_track_a_task4_dpvo_calibration.md` | Task 4: DPVO on a well-calibrated dataset | parked by the user; no number exists; **superseded in practice by `C01`/`C07`** |

## Track A2 — confidence gate (branch `track-a2-gate`, HEAD `428dd3f`)

| report | title | status |
| --- | --- | --- |
| `06_track_a2_confidence_gate.md` | A label-free confidence gate for the depicted tilt | current; the only source for the permutation-null summary statistics |
| `07_track_a2_gated_verdict.md` | The verdict test again, with the frozen gate and drag as an axis | current, but **sections 4-7 are duplicated**; the first occurrence is the one the report assets use |
| `08_track_a2_anwm_scored.md` | Scoring ANWM's generated flights | current for the numbers it reports; **§9 (Regime A amendment-3 vs frame-level counts) is stale** — recorded as prose-only in `tab_e5_anwm-verdict_7` |

## Track B — generation and input regimes (branch `track-b-generation`, HEAD `9ca9e90`)

| report | title | status |
| --- | --- | --- |
| `b01_anwm_feasibility_redo.md` | ANWM feasibility redo: what the release actually is | current; itself supersedes the first feasibility pass and its retraction |
| `b02_corrected_path_audit.md` | Corrected path audit of released and designed paths | current |
| `b03_gate2_native_pipeline.md` | Gate 2: the released model reproduces its own test trajectories | current |
| `b04_clip_set.md` | The ANWM clip set: 82 generated flights in three input regimes | current |
| `b05_pose_injection_stage0.md` | Pose injection stage 0: conventions verified before GPU spend | current |
| `b06_pose_injection_stage1.md` | Pose injection stage 1: the injected-true arm | current; scoring of the injected clips is still pending (`report/MISSING.md` item 1) |

## Track C — path from pixels (branch `track-c-pathrec`, HEAD `ed76a08`)

| report | title | status |
| --- | --- | --- |
| `C01_track_a_dpvo_reproduction.md` | Reproducing Track A's DPVO translation number | current; corrects Track A's "41 m vs 479 m" reading |
| `C02_extractor_survey_and_cost.md` | Extractor availability, cost, and the T1 substitution | current |
| `C03_gate1_bakeoff.md` | Gate 1: can the path be recovered from pixels? | current |
| `C04_gate2_verdict_scope.md` | Gate 2: what a pixel-path verdict may claim | current |
| `C05_anwm_attribution.md` | T3 attribution: ANWM generated vs ground-truth frames | current |
| `C06_failure_diagnostics.md` | Diagnostics: why the two hard UZH sequences fail | current |
| `C07_widened_uzh_and_dpvo.md` | Widened UZH set and DPVO: the calibration is the difference | current |
| `C08_calibration_and_scale.md` | Calibration sensitivity and scale provenance | current |
| `C09_anwm_dpvo_and_verdict_pilot.md` | DPVO on ANWM, and the pixel-path verdict pilot on UZH | current |

## Track D — verdict statistic (branch `track-d-verdict-stat`, HEAD `2a91765`)

| report | title | status |
| --- | --- | --- |
| `D01_gain_test.md` | The gain test on TII | **superseded** by `D02` for the frozen statistic (schema v1, rule R1); kept as the first-run record |
| `D02_gain_test_v2.md` | Gain decomposition, schema v2, coherence tier, OLS vs Deming | current; the frozen statistic (`configs/gain_frozen.yaml`) |
| `D03_shrinkage_window.md` | Shrinkage, per-type margins, flight-level FRR, window sensitivity | current; its window-sensitivity statement is qualified by `D04`/`E05` |
| `D04_bandwidth_window.md` | Label-free acceleration-bandwidth window on TII and UZH | current; **still contains the superseded UZH window statement and 35-62 samples at 500 Hz**, corrected by `E05` |

## Track E — depicted tilt on UZH-FPV (branch `track-e-depicted-uzh`, HEAD `dd209bf`)

| report | title | status |
| --- | --- | --- |
| `E01_uzh_audit_and_perframe.md` | UZH-FPV audit and whether the estimator transfers | current |
| `E02_mount_and_verdict.md` | Mount check, verdict clips, frozen-threshold transfer | current |
| `E03_dense_schema_calibration.md` | Dense-lean clip set, estimator calibration, the 45-rig question | current; CPU only |
| `E04_v3_r2_coherence.md` | v3: fixed-crop edits, official schema v2, Track D's R2 + coherence on UZH | current |
| `E05_window_rule.md` | Independent test of Track D's label-free window rule (`0.5 / f95`) | current; **supersedes** D04's UZH window numbers (0.07-0.12 s → 0.34-1.42 s at 500 Hz) |

## Track F — pixel-path verdict (branch `track-f-pixel-path`)

| report | title | status |
| --- | --- | --- |
| `F01_pixel_path_stage0.md` | Stage 0: statistics that avoid the second derivative | current; **decision: fail** (shooting 6/8 removal, 4/8 mirror; truth path 8/8 and 7/8) |
| `F02_stage1_extractors.md` | Stage 1: ViPE and MASt3R-SLAM on UZH-FPV, and the stage-0 tests on the best | current; ViPE 10/10 on the bake-off and no improvement to the gate; MASt3R 5/10; MegaSaM not run (Drive-only weights) |
| `F03_stage2_anwm.md` | Stage 2: ANWM clips against their commanded path | current; ViPE paths unusable on generated clips |

## Track F2 — pixel-path round 2 (branch `track-f2-pixel-path`)

| report | title | status |
| --- | --- | --- |
| `F04_spline_fit.md` | Spline-smoothed joint fit | current; **decision: fail** (pixel path 1/8 removal, 2/8 mirror; truth path 8/8, 6/8); the joint fit's scale collapse is a finding |
| `F05_lean_error_scoring.md` | What path error the verifier needs | current; windowed thrust-axis error ranks paths differently from the bake-off metric (correlation 0.65) and does not predict per-clip verdict quality |
| `F06_feedforward_and_sim.md` | MapAnything, Pi3X, and the simulated set | current; MapAnything 0/10 with metric scale, Pi3X 6/10 with Kalibr K; no extractor moves the gate; part 4 waiting on Track I (8/300 clips, no I02 report) |

## Track H — generator pilot (branch `track-h-generator-pilot`)

| report | title | status |
| --- | --- | --- |
| `H01_generator_pilot.md` | Generator pilot: ANWM seed variance, LingBot-World roll injection, Wan 2.2 cost | current; pilot only, no scale-up until the user decides |

## Notes

* Report 04 quotes the pre-registration sha256 `b8fd7013…` for `configs/verdict_preregistration.yaml`
  at "commits `d6b2003`, amendment `addd9ec`". That hash is the file **after amendment 1**
  (`addd9ec`); the file at the original commit `d6b2003` hashes `203e9518…`. It was amended again
  (amendment 2, `d8941e8`, `4e8661bf…`) and the current file, at `0834204`, hashes `c8ffac93…` —
  the value `report/tables/tab_frozen-configs_1.tex` records. The sequence is chronology, not
  drift; every version is frozen at its own commit. Recorded, not resolved.
* Track D's `D01`/`D02` split, the duplicated A2 report 07 sections, report 08's stale §9 and
  report 03's transcribed DPVO correlations are all restated in `report/CHECKS.md`
  ("Known open discrepancies"). Nothing was re-analysed during consolidation.
* Superseded material is listed in `docs/archive/README.md`; it stays where the reports and
  scripts expect it.
