# Superseded material — index, not a move

Nothing was moved into this folder. The consolidation rule was *add indexes, do not reshuffle*:
the report scripts, the numbered reports and `report/data/*.src.json` all reference these paths, so
moving them would break provenance for no gain. This file is the map of what is superseded, where
it still lives, and what replaced it.

Directory: `docs/archive/` — pointer only. Actual superseded files stay at their original paths.

## Reports

| superseded | by | why it stays |
| --- | --- | --- |
| `reports/D01_gain_test.md` | `reports/D02_gain_test_v2.md` | first-run record (schema v1, rule R1); D01's own header already says so; report D02 cites it |
| `reports/05_track_a_task4_dpvo_calibration.md` | `reports/C01_track_a_dpvo_reproduction.md`, `reports/C07_widened_uzh_and_dpvo.md` | the task was parked; C01/C07 did the work and corrected the Δ-path reading |
| Track A report 03's Phase 2 numbers | its own correction section inside the same report | both runs are printed side by side in the report |
| Track A2 report 07, sections 4-7 | the *first* occurrence in the same report | duplicated sections; the report-assets build uses the first occurrence |
| Track A2 report 08 §9 (Regime A statistic) | the frame-level statistic in `reports/D02` | exists only as prose; `tab_e5_anwm-verdict_7` transcribes it and flags it |
| `reports/D04`'s UZH window statement (0.07-0.12 s, 35-62 samples at 500 Hz) | `reports/E05_window_rule.md` (0.34-1.42 s at 500 Hz) | D04 is the record of the pre-registration; E05 is the independent test |

## Configs

| superseded | by | note |
| --- | --- | --- |
| `configs/window_amendment_proposal.yaml`, `configs/window_bandwidth_amendment_proposal.yaml`, `configs/coarse_path_rule_proposal.yaml` | the frozen `configs/window_bandwidth_prereg.yaml` | these are **proposals** and were never frozen; they stay as the amendment record |

Frozen configs are immutable and are not listed here at all: `configs/phase1.yaml`,
`configs/verdict_preregistration.yaml`, `configs/gate_frozen.yaml`, `configs/pathrec_prereg.yaml`,
`configs/gain_frozen.yaml`, `configs/coherence_frozen.yaml`,
`configs/window_bandwidth_prereg.yaml`.

## Results

| superseded | by | note |
| --- | --- | --- |
| `results/a2_verdict_baseline_origrows/` | `results/a2_verdict_gated_baseline/` | kept: report 07 explains which rows were replaced |
| `results/phase2_implied/20261108T*` (first run, inert window sweep) | `results/phase2_implied/20261109T173302Z_d9de303/` | kept: report 03 documents both |
| `results/d_eval/d01/` (schema v1) | `results/d_eval/d02/` (schema v2) | kept: `tab_e8_...` uses d02, D01 is the record |
| `results/e_verdict/` (v1) | `results/e_verdict_v3/` | kept: E01/E02 cite v1, E04/E05 cite v3 |

## Documents

| superseded | by | note |
| --- | --- | --- |
| `docs/track_a_report.md` (early running notes) | `reports/01_track_a_phase0_floor.md` … `reports/05_track_a_task4_dpvo_calibration.md` | kept; nothing in `report/` reads it |
| the handoff briefs 02-10 | `reports/` | never committed in the first place; see `docs/handoffs/README.md` |

## What was actually moved

* The pre-consolidation copies live **outside** the repository, in
  `~/Documents/Projects/awmv-archive/` (bundles, tags, untracked files, the data manifest).
* `docs/archive/report-assets-2026-11-15/` — the unique part of `awmv-report-assets-2026-11-15.zip`
  (`source_reports/` and `MANIFEST.txt`, 73 files) is committed here, with a member-by-member audit
  in its `README.md`. All 203 members of the zip are therefore on `main`; the zip is kept until the
  user releases it (its deletion is conditioned on the LaTeX sources being integrated first).
* `track-d-verdict-stat.bundle` stays at its original path; a copy is also on R2 at
  `track-d/track-d-verdict-stat.bundle`, and the branch it contains is reachable from `main` and
  from `archive/track-d-pre-consolidation`.
