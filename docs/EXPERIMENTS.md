# Experiments and where their pieces live

One row per report experiment, E1 to E13 as named in `report/`, plus the G and J series.
"Owning track" is where the work was done; several experiments read results produced by an
earlier track, and those are listed as contributors.

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
| E11 | pixel-path verdict, stage 0 (Track F) | F | `track-f-pixel-path` | `results/f01_stage0/` | F01 |
| E12 | pixel-path round 2: spline joint fit, lean-error scoring, feed-forward extractors | F2 | `track-f2-pixel-path` | `results/f04_spline/`, `f05_lean_error/`, `f06_feedforward/` | F04, F05, F06 (round 1: E11 on `track-f-pixel-path`) |
| E13a | generator pilot: ANWM seed variance, LingBot-World, Wan 2.2 | H | `track-h-generator-pilot` | `results/h01_seed_specs/`, `h01_seed_variance/`, `h01_lingbot/`, `h01_lingbot_b/`, `h01_wan/`, `h01_tables/`, `h01_r2/` | H01 |
| E13b | generator coverage with path control (H2) | H2 | `track-h2-generator-coverage` | `results/h02_inventory.json`, `h02_solarwm*`, `h02_lingbot_full*` | H02 |
| G01 | segment-level detection: 8/12-frame windows, residual/coherence/sign tiers, max-over-windows and persistence scan, conformal LOFO calibration | G | `track-g-segments2` (`c189727`) | `results/g01_segments/` | G01 |
| G02 | combined score: max of standardised tiers, LOFO conformal threshold, ROC, minimum-detectable-violation curve | G | `track-g-segments2` (`c189727`) | `results/g02_combined/` | G02 |
| G03 | depicted-tilt reader fine-tune on 5,007 TII frames (8 folds) and 1,097 UZH frames (3 folds), plus the downstream verdict | G | `track-g-segments2` (`c189727`) | `results/g03_tilt_reader/` | G03 |
| G04 | path-demand windows (item 1), preliminary against the track | G2 | `track-g2-segments` (`61401ae`) | `results/g04_path_demand/` | G04 |
| G05 | segments on the simulated set (blocked; design frozen, one exploratory pilot) | G2 | `track-g2-segments` (`61401ae`) | `results/g05_spike_pilot/` | G05 |
| J01 | reader error anatomy | J | `track-j-reader` | `results/j01_anatomy/` | J01 |
| J02 | synthetic noise budget | J | `track-j-reader` | `results/j02_noise_budget/` | J02 |
| J03 | per-clip calibration (1a) | J | `track-j-reader` | `results/j03_perclip/` | J03 |
| J04 | learned reader confidence (1c) | J | `track-j-reader` | `results/j04_confidence/` | J04 |

R2 prefixes are `track-a/`, `track-a/a2/`, `track-b/`, `track-c/`, `track-d/`, `track-e/`, `track-h/`, `track-h2/`
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

## E11 — pixel-path verdict, stage 0 (Track F)

* **Owning track / branch**: F, `track-f-pixel-path`. Report `reports/F01_pixel_path_stage0.md`;
  this is the experiment indexed as E11 in the report candidate
  `report_candidates/e11_pixel_path_verdict.tex`.
* **What it is**: five verdict statistics that avoid double-differentiating the DPVO path
  (shooting primary, preintegration, band-limited coherence, velocity-drag, physics scale),
  each under oracle and estimated attitude and with the pixel path or the truth path, on
  UZH-FPV forward rigs. The edit family, population rules and thresholds follow E8/E9.
* **Configs**: `configs/f01_pixel_path_prereg.yaml` (`6b88d5a`), amendment 1
  `configs/f01_pixel_path_prereg_amend1.yaml` (`17d21c6`, paired scale/yaw ruler).
* **Inputs**: Track C's DPVO poses on the Modal volume `awmv-c-data` at `poses/dpvo/*.npz`
  (150 frames per sequence, 15.07 Hz; local copy `data/pathrec/poses/dpvo/`, sha256 in the
  summary), Track E's schema-v2 clips `data/clips_v2_w370ms/`, UZH truth `data/uzh/`, the E05
  window rule `results/e_window_rule.json`, and the DA3 depth prior
  `results/pathrec_calib/depth.json`.
* **Results**: `results/f01_stage0/summary.json`, `rows.csv`, `table_stage0.md`,
  `edit_path_check.json`; stage 1 (`F02`) in `results/f02_stage1/` (extractor poses scored
  with the same metric, plus a full stage-0 rerun on ViPE's path in
  `results/f02_stage1/stage0_vipe/`); stage 2 (`F03`) in `results/f03_stage2/`.
* **Regenerate**: `experiments/f01_stage0.py` (CPU, no spend), then
  `experiments/f01_tables.py`; the one-sequence edit-invariance check is
  `experiments/f01_edit_path_check.py`; stage 1 is `remote/f_vipe.py` and
  `remote/f_mast3r.py` (GPU) scored by `experiments/f02_stage1.py`; stage 2 is
  `remote/f_anwm.py` scored by `experiments/f03_anwm.py`.
* **Caveats**: the stored paths cover prepared frames 0-149, so only clips ending by frame
  150 are in the primary population; the real path is reused for the edits (one-sequence
  check only); the primary population has 8 clips in 4 sequences, so intervals are
  descriptive.

## E12 — pixel-path round 2 (Track F2)

* **Owning track / branch**: F2, `track-f2-pixel-path`, branched from `main` at
  `41d1dad`. Round 1 (`reports/F01`-`F03`, branch `track-f-pixel-path`, head `a9dc5cf`) is a
  dependency: the population, edits, controls and the amendment-1 pairing are carried over
  unchanged.
* **What it is**: (i) a cubic B-spline path with analytic acceleration and a joint flatness
  fit in positive scale and initial velocity, tested against the round-1 gate; (ii) a windowed
  thrust-axis ("lean") error metric for stored paths, and whether it predicts verdict quality;
  (iii) MapAnything and Pi3X run on UZH-FPV with the Kalibr intrinsics, scored with (ii);
  (iv) the simulated calibration set, waiting on Track I.
* **Configs**: `configs/f04_round2_prereg.yaml` (`63002b8`).
* **Inputs**: the stored poses `data/pathrec/poses/{dpvo,vipe,mast3r}` (round 1) plus
  `{mapanything,pi3x}` pulled from the `awmv-f2-data` volume; Track E clips
  `data/clips_v2_w370ms`; UZH truth `data/uzh`; E05 window rule `results/e_window_rule.json`.
* **Results**: `results/f04_spline/` (combined run; `summary.json`, `rows.csv`,
  `diagnostic.json`, `pi3x/`), `results/f05_lean_error/` (`lean_error.json`,
  `sequences.csv`), `results/f06_feedforward/summary.json`.
* **Regenerate**: `experiments/f04_spline_fit.py` (CPU), `experiments/f05_lean_error.py`
  (CPU), `experiments/f06_score_extractors.py` (CPU); the GPU runs are `remote/f2_extractors.py`
  (`--method map|pi3`).
* **Outcome**: the registered spline statistic fails the gate on every stored path while
  detecting 8/8 and 6/8 on the truth path; its joint fit is degenerate (scale collapse); Pi3X
  is the best new extractor (6/10 bake-off, 4.2 deg lean error) and does not change the gate;
  the simulated set is 8/300 rendered.

## E13a — generator pilot (Track H)

* **Owning track / branch**: H, `track-h-generator-pilot`.
* **What it is**: three questions asked before any scale-up. (1) Does ANWM's depicted tilt
  vary across sampling seeds, and does an oracle best-of-N rescue a designed path? (2) Does
  LingBot-World (camera-pose-conditioned, open weights) depict the attitude its pose input
  commands? (3) What do Wan 2.2 and the other open models cost per clip?
* **Configs**: `configs/h01_pilot_prereg.yaml` (commit `245f7d9`, sha256 `3529b920…`); all
  thresholds and populations are fixed there before any new clip is scored. The frozen gate
  (`configs/gate_frozen.yaml`) and the frozen verdict threshold (6.46 deg) are used unchanged.
* **Seed population**: 20 Regime C designed paths (`results/h01_seed_specs/paths.json`),
  seeds 0-4; seed 0 reuses the existing Track B clips, seeds 1-4 are 80 new ANWM rollouts
  (`awmv-h-anwm`, A10G, 250 sampling steps, same checkout `59def562…`).
* **Reader**: unchanged — GeoCalib 224 px, `camera_to_body_pitched`, the same row schema as
  `results/gate_a2/anwm/depicted.jsonl`; the new rows are `results/h01_seed_depicted/`.
* **Results**: `results/h01_seed_variance/` (per-clip, per-path, best-of-N, summary),
  `results/h01_seed_variance_existing/` (the five pre-existing 3-seed paths, reported
  separately), `results/h01_lingbot/`, `results/h01_wan/`, `results/h01_tables/`,
  `results/h01_r2/*.json` (R2 manifest).
* **Raw clips**: R2 `track-h/clips/anwm-seeds/`, `track-h/clips/lingbot/`,
  `track-h/clips/wan/`, with sha256 manifests in `results/h01_r2/`.
* **Outcome**: ANWM seed sampling does not rescue the designed paths (oracle best-of-N
  reduction about 0.2 deg, no feasibility flips); LingBot-World under the community NF4 build
  translates but does not follow a roll command (20 and 60 deg ramps both stay within about
  1 deg of the level baseline), so its 60-clip grid was **not run**; Wan 2.2 TI2V-5B is a
  cost-only arm.
* **Report**: `reports/H01_generator_pilot.md`.
* **Cost**: `docs/COST_LEDGER.md`.
* **Stop rule**: the pilot ends with the report; no scale-up until the user decides.

## E13b — generator coverage with path control (Track H2)

* **Owning track / branch**: H2, `track-h2-generator-coverage`, based on `main` (`41d1dad`);
  read-only dependency on `track-h-generator-pilot` (H01) for the reader, first frames and
  designed paths.
* **What it is**: inventory of path/pose-controlled world models; SolarWM-5B roll-injection,
  seed variance and commanded-path verdict; full-precision LingBot roll probes beside H01's
  NF4 result.
* **Configs**: `configs/h02_pilot_prereg.yaml` (`5c6e5c6`) and amendment 1 (bank-in, 16 m
  translation scale).
* **Results**: `results/h02_inventory.json`, `results/h02_solarwm/`,
  `results/h02_lingbot_full/`, `results/h02_*_depicted/`, `results/h02_r2/*.json`.
* **R2**: `track-h2/clips/...` with sha256 manifests.
* **Report**: `reports/H02_generator_coverage.md`; candidate section
  `report_candidates/h02_generator_coverage.tex`.
* **Outcome**: SolarWM-5B follows a continuously ramping roll (level 1.4 deg; +20 ramp ends
  22.4, -20 18.2, +60 45.8) but not a held path attitude: on the 75-clip grid the roll criterion
  fails (1/5 paths by tilt, 1/5 by mirrored sign) and the verdict leaves 15/25 testable and 5
  feasible (all slalom-10) under H02's absolute 12-frame rule, rejecting every testable
  turn-10/20 clip; under H01's 12/16 fraction only the ten turn clips are testable, all
  infeasible (0/10 feasible). Seed sampling moves the
  depicted tilt more than ANWM's (median range 2.6 deg), but best-of-3 cuts the residual by
  1.0 deg median and flips 1/10 paths, below the pre-registered 2 deg / 20 % bar.
  Full-precision LingBot still shows no coherent roll in the reader; quantisation is not
  established as the cause.
* **Stop rule**: pilot only; no scale-up until the user decides.

## Track G — verifier statistics (branch `track-g-segments2`, off `main` `41d1dad`)

Track G does not add a dataset; it adds statistics over the schema-v2 clips every other track
already produced. Nothing GPU-side except G03.

| ID | experiment | stored results | report | configs |
| --- | --- | --- | --- | --- |
| G01 | segment-level detection: 8/12-frame windows, residual/coherence/sign tiers, max-over-windows and persistence scan statistics, conformal LOFO calibration, partial-edit injections at 10/25/50 % | `results/g01_segments/{dev,heldout}_w{8,12}_*/` | `reports/G01_segment_detection.md` | `g01_segment_prereg.yaml`, `g01_segment_amendment_a.yaml` |
| G02 | combined score: max of standardised tiers, LOFO conformal threshold, ROC, minimum-detectable-violation curve | `results/g02_combined/{dev,heldout_devfit}/` | `reports/G02_combined_score.md` | `g02_combined_prereg.yaml` |
| G03 | depicted-tilt reader fine-tune on 5,007 mocap-labelled TII frames (8 LOFO folds) and 1,097 UZH forward-rig frames (3 folds), plus the downstream verdict | `results/g03_tilt_reader/`, Modal volume `awmv-g-data` | `reports/G03_tilt_reader.md` | `g03_tilt_reader_prereg.yaml` |

Regenerate G01/G02 with `experiments/g01_segment_detection.py` and
`experiments/g02_combined_score.py` (both CPU, both read only stored clips and Track D's frozen
tables). G03 is `experiments/g03_prepare_frames.py` + `experiments/g03_prepare_uzh.py` (CPU,
builds the labelled frame sets) then `modal run remote/g_tilt_reader.py` (A10G) and
`experiments/g03_evaluate.py` (CPU, after pulling the fold files).

New code: `src/verdict/segments.py`. New data: `data/g03_frames/` (372 MB),
`data/g03_uzh/` (144 MB), `data/raw/tii_video_full/` (3.6 GB) — all gitignored and rebuildable
from the release and from R2 (`track-a/tii_video/`).

## Track G2 — segment detection, round 2 (branch `track-g2-segments`, off `main` `41d1dad`)

Round 2 continues Track G (branch `track-g-segments2`, unmerged when this started). It reads
Track D's schema-v2 clips, round 1's segment code, and — read-only — Track I's I01 spike.
Everything here is CPU; nothing was spent.

| ID | experiment | stored results | report | configs |
| --- | --- | --- | --- | --- |
| G04 | path-demand windows (item 1): demand windows (≥6/8 frames at ≥15 deg) and quiet windows (all 8 ≤3 deg) selected from the path alone, residual and depicted-tilt statistics, LOFO conformal calibration on the gate-kept population, partial-edit injections, synthetic unrequited-tilt injection | `results/g04_path_demand/` | `reports/G04_path_demand_segments.md` | `g04_path_demand_prereg.yaml` |
| G05 | segments on the simulated set (items 1-6): **blocked**, no I02 labelled set; the design is frozen unexecuted and one exploratory 3-clip pilot is run | `results/g05_spike_pilot/` | `reports/G05_segments_on_simulated_set.md` | `g05_round2_design_prereg.yaml` |

Regenerate: `experiments/g04_path_demand.py` (CPU, ~50 s) and `experiments/g05_spike_pilot.py`
(CPU, reads `../awmv-track-i/results/i01_spike`). The every-window arm of G04 reproduces round
1's published detection counts exactly, which is the correctness check for the ablation.

Dependency to record: G05 needs Track I report I02 (≥8 dev + ≥4 holdout scenes with partial
violation intervals and controls). Until it exists, no item 2-6 ablation can be pre-registered
honestly and no confirmatory evaluation can run.

## J01 — reader error anatomy

* **Owning track / branch**: J, `track-j-reader`.
* **What it is**: what the depicted-tilt error is made of, per flight and per axis: per-flight
  bias, drift below 0.5 Hz and error above 0.5 Hz (orthogonal DCT-II split); autocorrelation and
  integral time; whether the bias is a rigid mount rotation; the >10 deg tail's run structure and
  covariates; and the per-step plus horizon error of DPVO/ViPE rotations.
* **Configs**: `configs/j01_anatomy_prereg.yaml`, `j01_anatomy_amendment_a.yaml` (input facts),
  `j01_anatomy_amendment_b.yaml` (the trimmed decomposition is not identified).
* **Inputs**: `results/gate_a2/tii/<flight>__identity.jsonl` (frozen reader, `track-a2-gate`
  `bc278bf`), `results/j01_anatomy/../j_inputs/g03_folds/` (G03 out-of-fold reader,
  `track-g-segments2` `cfea5c8`, staged), `data/processed/p3a/p3a_v1` (mocap geometry),
  `data/g03_frames` (frames, for the blur proxy), `data/raw/dpvo/*_stride8_*` and
  `data/pathrec/poses/{dpvo,vipe}` (rotations).
* **Results**: `results/j01_anatomy/` — `decomposition.csv`, `decomposition_pooled.json`,
  `acf.csv`, `psd.csv`, `bias.csv`, `bias_structure.json`, `tail.json`, `blur.csv`,
  `rotation.json`, `rules.json`, `input_check.json`.
* **Report**: `reports/J01_reader_error_anatomy.md`.
* **Regenerate**: `PYTHONPATH=. python experiments/j01_anatomy.py` (~25 s).

## J02 — synthetic noise budget

* **Owning track / branch**: J, `track-j-reader`.
* **What it is**: keep every real clip's path, replace the depicted side by a perfect depiction
  plus a controlled error with the measured bias, slow/fast split and correlation time, and
  measure the minimum detectable uniform tilt reduction at a 5 % leave-one-flight-out
  false-reject — against window length, sigma, bias, correlation time and slow share, with the
  analytic formula as a cross-check and a replay of the measured error as a sanity anchor.
* **Configs**: `configs/j02_noise_budget_prereg.yaml`.
* **Results**: `results/j02_noise_budget/summary.json`.
* **Report**: `reports/J02_noise_budget.md`.
* **Regenerate**: `PYTHONPATH=. python experiments/j02_noise_budget.py`.

## J03 — per-clip calibration and its absorption (Stage 1a)

* **Owning track / branch**: J, `track-j-reader`.
* **What it is**: a per-clip Huber offset (and k_d on a fixed grid in [0, 0.40]) fitted on the clip
  under test, on the gate-kept population; the same two tiers refitted; the absorption of every edit
  type against the frozen fixed-calibration statistic; the fit-free coherence tier as the reference;
  the minimum detectable local tilt step per statistic.
* **Configs**: `configs/j03_perclip_prereg.yaml`.
* **Results**: `results/j03_perclip/{table.json, absorption.json, mdv.json, diagnostics.json,
  rows.csv, mdv_rows.csv}`.
* **Report**: `reports/J03_per_clip_calibration.md`.
* **Regenerate**: `PYTHONPATH=. python experiments/j03_per_clip_calibration.py` (~30 s, CPU).
* **Headline**: the clip-median tier loses 31 of 42 full-clip removal detections and 44 of 48 x1.5
  detections to the fit; the coherence tier detects 33/45/54 at 25/50/100 % coverage at a 3/69
  false-reject and has a 14-17 deg minimum detectable local step against > 25 deg for every fitted
  statistic.

## J04 — learned reader confidence (Stage 1c)

* **Owning track / branch**: J, `track-j-reader`.
* **What it is**: a logistic model on the reader's own stored per-frame outputs predicting
  `theta > 10 deg`, leave-one-flight-out; its AUROC and calibration against the frozen S3 signal on
  the same frames; and the G-round window statistic re-run with the predicted `1-p` as a weight
  and as a mask on the dev block.
* **Configs**: `configs/j04_learned_confidence_prereg.yaml`.
* **Results**: `results/j04_confidence/{per_frame.json, downstream.json, rows.csv}`.
* **Report**: `reports/J04_fusion_and_learned_confidence.md`.
* **Regenerate**: `PYTHONPATH=. python experiments/j04_learned_confidence.py` (~25 s, CPU).
* **Status**: selected by rule 3's literal reading; the premise is weak (88 % of the tail is in
  clips the frozen gate drops). Stage 1b was not selected by any rule and was not run.

---

## Track M — abstention gates (branch `track-m-gates`, 2026-11-22)

Track M designs two abstention gates and evaluates them end to end. Gate A is path-side (the
required lean's per-axis spread, schema v3 exact); Gate B is video-side (agreement between two
independent readers on the same 224 px feed: Perspective Fields and GeoCalib-pinhole). Both are
treated as selective prediction with conformal risk control on the gate-kept real clips.

| M | experiment | code | configs | stored results | report |
| --- | --- | --- | --- | --- | --- |
| M01 | Gate A: path-side signal strength, coverage/risk curves, yaw-control mechanism | `experiments/m01_gate_a.py`, `src/verdict/{gate_a,selective}.py` | `configs/m01_gate_a_prereg.yaml`, `configs/m01_gate_a_frozen.yaml` | `results/m01/` | M01 |
| M02 | Gate B: reader agreement, AUROC against reader error, forest check, ensemble reader | `experiments/m02_gate_b.py`, `remote/m_readers.py`, `src/verdict/gate_b.py` | `configs/m02_gate_b_prereg.yaml`, `configs/m02_gate_b_frozen.yaml` | `results/m02/` | M02 |
| M03 | end-to-end: G0 / GA / GB / GAB x R / F0 / C2 on real, simulated and generated clips | `experiments/m03_end_to_end.py`, `experiments/m03_generators.py` | `configs/m03_endtoend_prereg.yaml` | `results/m03/` | M03 |

R2: Track M writes nothing outside `track-m/`. It reads `track-i/` (the simulated clips),
`track-l/l01/per_frame/` (the simulated reader table), `track-h3/` (SolarWM clips) and
`track-h/` (ANWM frames) without writing to them; the TII camera zips and the A2 edit specs are
read from Track A2's `awmv-a-data` volume, and the reader scores live on `awmv-m-data`.

Dependencies, read-only: `track-k-calibration` (schema v3, `src/airframe/*` and the K03 tier
code, copied byte-identically), `track-h3-reader-validity` (H04/H03), `track-l-reader-simreal`
(L01), `track-i-simset` (the frozen simulated split).
| M04 | sign sub-tier with a reader-noise-tied dead-band (new pre-registered variant, frozen rule reported beside it) | `experiments/m04_sign_deadband.py` | `configs/m04_sign_deadband_prereg.yaml` | `results/m04/` | M04 |
