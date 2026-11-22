# Experiments and where their pieces live

One row per report experiment, E1 to E10 as named in `report/`. "Owning track" is where the
work was done; several experiments read results produced by an earlier track, and those are
listed as contributors.

Every number in `report/tables` and `report/figures` is traceable to `report/data/<asset>.src.json`,
which records the exact file, its branch, its commit and its sha256 at build time. This file is
the human-readable map; the `.src.json` files are the machine-readable one.

Regenerating scripts: `report/scripts/e<N>_*.py` (plus `methodology.py` and `corrections.py`),
run through `report/scripts/make_all.py`. Every one of them is CPU-only and reads stored
results; none of them runs an experiment.

## Summary

| E | experiment | owning track | branch (HEAD at consolidation) | stored results | report(s) |
| --- | --- | --- | --- | --- | --- |
| E1 | physics floor (mocap, no pixels) | A | `track-a-perception` (`164c5e0`) | `results/phase0_floor/` | 01 |
| E2 | depicted tilt on TII | A (+ A2 for the gate) | `track-a-perception`, `track-a2-gate` | `results/phase1_depicted/`, `phase1b_fusion/`, `phase3a_verdict/*_lean`, `a2_gate/` | 02, 03, 06 |
| E3 | verdict on real flights | A then A2 | `track-a-perception`, `track-a2-gate` (`428dd3f`) | `results/phase3a_verdict/`, `a2_verdict_gated*/`, `a2_modes/` | 04, 06, 07 |
| E4 | ANWM data audit | B (+ A noise floor) | `track-b-generation` (`9ca9e90`) | `results/anwm_path_audit/`, `anwm_clips/`, `anwm_scoring/` | b01, b02, b04 |
| E5 | ANWM verdict | A2 (+ B clips) | `track-a2-gate` (`428dd3f`) | `results/a2_anwm*/`, `a2_parity/`, `a2_modes/`, `a2_checks/` | 08, b03 |
| E6 | pose injection into ANWM frames | B | `track-b-generation` (`9ca9e90`) | `results/anwm_injected/` (stage 0 + stage 1), `anwm_injected/sheets/` | b05, b06 |
| E7 | path from pixels | C (+ A ladder, E audit) | `track-c-pathrec` (`ed76a08`) | `results/pathrec_*/`, `results/phase2_implied/` | C01-C09, 03 |
| E8 | gain and coherence tests | D | `track-d-verdict-stat` (`2a91765`) | `results/d_eval/`, `d_decomposition/`, `d_coherence/`, `d_deming/`, `d_window/`, `d_shrinkage/`, `d_flights/`, `d_calibrate/` | D01, D02, D03 |
| E9 | UZH transfer | E (+ A estimator) | `track-e-depicted-uzh` (`dd209bf`) | `results/e_audit/`, `e_mount/`, `e_perframe/`, `e_verdict_v3/`, `e_calib/` | E01-E04 |
| E10 | window rule and prototype | D (+ E independent test) | `track-d-verdict-stat`, `track-e-depicted-uzh` | `results/d_window_rates/`, `d_bandwidth/`, `d_uzh_v2/`, `e_verdict_v3/rule/` | D03, D04, E05 |

R2 prefixes are `track-a/`, `track-a/a2/`, `track-b/`, `track-c/`, `track-d/`, `track-e/`
(counts and bytes: `report/data/r2_counts.json`, measured 2026-11-15; verified read-only again
during consolidation — see `docs/DATA.md` for the current status of listing access).

---

## E1 — physics floor

* **Owning track / branch**: A, `track-a-perception` (`164c5e0`).
* **What it is**: the mocap-only floor. TII mocap at 275 Hz, Savitzky-Golay derivatives, one
  world-frame drag coefficient per flight-fold. No pixels, no GPU.
* **Config**: `configs/phase1.yaml` (frozen, `e4fd5d9`, sha256 `c5d3b55e…`). Each run also stores
  its resolved `config.yaml`.
* **Result**: `results/phase0_floor/20261108T044153Z_f3b5d2bf/` (`metrics.json`, per-flight
  arrays, plots). Gate: pooled held-out median 1.24 deg with drag, 5.46 drag-free.
* **Report**: `reports/01_track_a_phase0_floor.md`.
* **Regenerate the assets**: `report/scripts/e1_physics_floor.py`.
* **Assets**: `tab_e1_physics-floor_1`, `fig_e1_physics-floor_1`, `report/data/e1_physics-floor.csv`.

## E2 — depicted tilt (TII)

* **Owning track / branch**: A, `track-a-perception`; the confidence gate is A2,
  `track-a2-gate` (`428dd3f`, gate frozen at `9e1656d`).
* **What it is**: GeoCalib body attitude from the raw fisheye, nuisance fit, per-frame error
  against mocap; fusion of odometry rotation with gravity; six gate candidates, one chosen.
* **Configs**: `configs/phase1.yaml`; gate frozen in `configs/gate_frozen.yaml`
  (sha256 `dc0cc4ff…`).
* **Results**: `results/phase1_depicted/` (per-frame `.npz` + `metrics.json`),
  `results/phase1b_fusion/`, `results/phase3a_verdict/20261109T074134Z_492033f_lean/leans.json`,
  `results/a2_gate/20261109T201449Z_298ff20/gate_validation.json`.
* **Reports**: 02, 03, 06.
* **Regenerate**: `report/scripts/e2_depicted_tilt.py`, `e2_extra.py`.
* **Note**: the permutation-null distribution is not stored; only its median (0.822) and 95th
  percentile (0.874) survive, transcribed from report 06 (`report/MISSING.md` item 6).

## E3 — verdict on real flights

* **Owning track / branch**: A (`164c5e0`, report 04, run `bc3a398`), then A2 (`428dd3f`,
  reports 06-07, gate `9e1656d`, modes run `71fb651`).
* **What it is**: the pre-registered verdict on real TII clips with pixel edits (identity, tilt
  removal, mirrored roll, yaw control, tilt scaling), conformal threshold, acceptance/AUROC.
* **Configs**: `configs/verdict_preregistration.yaml` (frozen at `0834204`, sha256 `c8ffac93…`;
  amendments 1 `addd9ec`, 2 `d8941e8`, 3 `ea1dce4`), `configs/gate_frozen.yaml`.
* **Results**: `results/phase3a_verdict/20261109T090957Z_bc3a398/` (main),
  `..._ad3952a_ablations/`, `results/a2_verdict_gated{,_baseline,_pairgate}/`, `results/a2_modes/`,
  `results/a2_checks/`.
* **Reports**: 04 (first verdict), 06, 07 (gated verdict, drag as an axis).
* **Regenerate**: `report/scripts/e3_verdict_real.py`, `e3_extra.py`.
* **Caveat**: some thresholds and per-clip scores exist only in report 04 prose; the script
  parses/transcribes them and the `.src.json` says so.

## E4 — ANWM data audit

* **Owning track / branch**: B, `track-b-generation` (`9ca9e90`); Track A supplied the level-frame
  noise floor (`results/anwm_scoring/…/noise_floor.json`).
* **What it is**: what the released ANWM paths and our designed maneuvers actually demand in
  tilt; the level-frame noise floor at 224 and 512 px.
* **Configs**: none frozen for the audit; clip specs are stored in
  `results/anwm_clips/specs.json` (committed at `bd032be`).
* **Results**: `results/anwm_path_audit/audit.json`, `results/anwm_excitation/`,
  `results/anwm_clips/clips_index.csv`, `results/anwm_native/`, `results/anwm_scoring/`.
* **Reports**: `b01`, `b02`, `b03`, `b04`.
* **Regenerate**: `report/scripts/e4_anwm_audit.py`.

## E5 — ANWM verdict

* **Owning track / branch**: A2, `track-a2-gate` (`428dd3f`), scoring the Track B clip set.
* **What it is**: depicted vs designed tilt on generated ANWM flights, margins per calibration
  mode, infeasible/testable counts, controls, nuisance absorption, minimum consistent step.
* **Configs**: `configs/gate_frozen.yaml`; protocol amendment 3 (`ea1dce4`).
* **Results**: `results/a2_anwm/`, `results/a2_anwm_score/`, `results/a2_parity/`,
  `results/a2_modes/`, `results/a2_checks/`.
* **Reports**: 08 (and `b03` for the native reproduction gate).
* **Regenerate**: `report/scripts/e5_anwm_verdict.py`, `e5_extra.py`.
* **Caveat**: report 08 §9 (Regime A statistic comparison) is stale and exists only as prose;
  `tab_e5_anwm-verdict_7` transcribes it and says so.

## E6 — pose injection

* **Owning track / branch**: B, `track-b-generation` (`9ca9e90`).
* **What it is**: re-render ANWM frames with a commanded pose injected, stage 0 (conventions,
  no generation) then stage 1 (the injected-true arm); LPIPS and prior coverage per design.
* **Results**: `results/anwm_injected/` (`stage0.json`, `stage0/`, `sheets/`,
  `eval_injected_injected_true.json`, `coverage_by_cell.csv`).
* **Reports**: `b05`, `b06`.
* **Regenerate**: `report/scripts/e6_pose_injection.py`.
* **Pending**: depicted-tilt scoring of the injected clips (no verdict asset exists;
  `report/MISSING.md` item 1).

## E7 — path from pixels

* **Owning track / branch**: C, `track-c-pathrec` (`ed76a08`). Track A's oracle ladder
  (`results/phase2_implied/`) and Track E's UZH audit feed the same tables.
* **What it is**: can the drone's path be recovered from pixels — DPVO vs feed-forward
  extractors, intrinsic sensitivity, ANWM frames, and the pixel-path verdict pilot.
* **Configs**: `configs/pathrec_prereg.yaml` (`6a16297`), `pathrec_prereg_v2.yaml`,
  `pathrec_prereg_calib.yaml` (`2194b64`), `pathrec_prereg_verdict.yaml`.
* **Results**: `results/pathrec_bakeoff/{v1,v2}/`, `pathrec_repro/`, `pathrec_calib/`,
  `pathrec_anwm/`, `pathrec_priors/`, `pathrec_scale_provenance/`, `pathrec_verdict/`,
  `pathrec_diag/`, plus `results/phase2_implied/`.
* **Reports**: `C01`-`C09`.
* **Regenerate**: `report/scripts/e7_path_from_pixels.py`, `e7_extra.py`.
* **R2**: the `track-c/` prefix is **empty** (0 objects); Track C's record is the git branch.

## E8 — gain and coherence tests

* **Owning track / branch**: D, `track-d-verdict-stat` (`2a91765`).
* **What it is**: the frozen gain statistic (rule R2_axis_gains) and the coherence statistic, on
  the schema-v2 dense-lean TII population; decompositions, Deming fits, window and shrinkage
  sensitivity, flight-clustered intervals.
* **Configs**: `configs/gain_frozen.yaml` (`7e8c42e`, `28f91f7f…`),
  `configs/coherence_frozen.yaml` (`7e8c42e`, `40a40bea…`), pre-registrations
  `gain_prereg.yaml`, `coherence_prereg.yaml`, acceptance spec `prototype_acceptance_spec.yaml`.
* **Results**: `results/d_eval/{d01,d02}/`, `d_decomposition/`, `d_coherence/`, `d_deming/`,
  `d_window/`, `d_shrinkage/`, `d_flights/`, `d_calibrate/`, `d_repro_baseline/`.
* **Reports**: `D01` (superseded), `D02` (frozen statistic), `D03`.
* **Regenerate**: `report/scripts/e8_gain_coherence.py`.
* **Caveat**: D03 says 20,000 bootstrap draws, D04's appendix says 5,000 with seed 1; the asset
  uses the stored `d_flights` interval (see `report/CHECKS.md`).

## E9 — UZH transfer

* **Owning track / branch**: E, `track-e-depicted-uzh` (`dd209bf`); the estimator itself is
  Track A's.
* **What it is**: does the depicted-tilt estimator and the verdict transfer to UZH-FPV — audit,
  mount check, dense schema calibration, v3 fixed-crop edits, R2 + coherence on forward rigs.
* **Configs**: `configs/e_prereg.yaml`, `e_verdict_prereg.yaml`, `e_acceptance_ref.yaml`,
  `e_prereg_v3.yaml` (`1f021c2`), `e_prereg_v4_window_rule.yaml` (`1befcd1`).
* **Results**: `results/e_audit/`, `e_mount/`, `e_perframe/`, `e_clips/`, `e_calib/`,
  `e_verdict/`, `e_verdict_v3/`.
* **Reports**: `E01`-`E04`.
* **Regenerate**: `report/scripts/e9_uzh_transfer.py`.
* **Caveat**: at the 0.37 s headline the stored run has an 11-clip median-residual population
  where the handoff expects 18; the table prints the stored denominators.

## E10 — window rule and prototype

* **Owning track / branch**: D, `track-d-verdict-stat` (`2a91765`, report D04 and
  `configs/window_bandwidth_prereg.yaml` at `c7af793`), independently tested by E,
  `track-e-depicted-uzh` (`dd209bf`, report E05).
* **What it is**: a label-free acceleration-bandwidth window rule (`0.5 / f95`) with its
  truth-rate window per flight and its samples at clip rate; the scope matrix for what is
  validated; the v0.1 prototype spec.
* **Configs**: `configs/window_bandwidth_prereg.yaml` (`c7af793`, `75b208da…`),
  proposals `window_amendment_proposal.yaml`, `window_bandwidth_amendment_proposal.yaml`,
  `coarse_path_rule_proposal.yaml` (proposal only — nothing is built from it).
* **Results**: `results/d_window_rates/`, `d_bandwidth/`, `d_uzh_v2/`, `d_window/`,
  `results/e_window_rule.json`, `results/e_verdict_v3/rule/`.
* **Reports**: `D03`, `D04`, `E05`; prototype: `docs/prototype_v0.1.md`,
  `docs/verdict_prototype.md`.
* **Regenerate**: `report/scripts/e10_window_rule.py`.
* **Open discrepancy (recorded, not resolved)**: D's 0.07-0.12 s window "at the truth rate" is
  superseded by E05 (0.34-1.42 s at 500 Hz; 0.07-0.11 s only at the 15 Hz clip rate).

---

## Track L — reader on simulated scenes (2026-11-22)

| ID | what | branch / commit | config | stored results | report |
| --- | --- | --- | --- | --- | --- |
| L01 | frozen reader's error on the 80 feasible simulated development clips, by scene family and mount, with the frozen S3 gate and J01's anatomy split | `track-l-reader-simreal` (`674e7b8`, `2bf2cb2`) | `configs/l01_scene_stratified_prereg.yaml` (+ amendments A, C) | `results/l01/`; per-frame arrays on R2 `track-l/l01/per_frame/` | `reports/L01_scene_stratified_reader_error.md` |
| L02 | fine-tune arms (real-only, sim-only, mixed+DR) with matched budget and held-out discipline; downstream coherence tier on regenerated pixel edits | `track-l-reader-simreal` | `configs/l02_reader_prereg.yaml` (+ amendment B) | `results/l02/`; fold records `awmv-l-data /l02/folds`, checkpoints `/l02/models` | `reports/L02_reader_finetune_sim_real.md` |

L01's reader job is `remote/l_score_sim.py` (Modal `awmv-l-reader`); L02's training is
`remote/l_tilt_reader.py` (`awmv-l-tilt-reader`) and the edit reader is
`remote/l_edit_reader.py` (`awmv-l-edit-reader`). Inputs are copied byte-identically from
`track-i-simset`, `track-g-segments2` and `track-j-reader` (provenance in
`results/l00/provenance.json`, `docs/decisions.md` [L]).
