# Track A artifact manifest

| artifact | where | size | notes |
| --- | --- | --- | --- |
| TII camera zips, 8 flights | Modal Volume `awmv-a-data:/tii_video`, R2 `track-a/tii_video/` | 3.77 GB | sha256 in `tii_video/manifest.json`; local copy is only `data/raw/tii_video` CSVs (0.17 GB) |
| Sampled frames | `awmv-a-data:/tii_frames` (48/flight), `:/tii_windows` (12/flight) | 30 MB | pulled to `data/raw/tii_frames`, `data/raw/tii_windows` (gitignored) |
| GeoCalib runs | `awmv-a-data:/geocalib/{probe,div_frames_20,free_frames_20,div_zips_6,div10}` | < 10 MB | `div10` is the primary run (5,007 frames); per-frame rows in JSONL, local copies in `data/raw/geocalib` |
| DPVO poses + weights | `awmv-a-data:/dpvo/*/poses.npz`, `:/dpvo_models/dpvo.pth` | 272 KB + 100 MB | local copies in `data/raw/dpvo` |
| Phase 1a metrics | `results/phase1_depicted/<run>/` | - | `metrics.json`, `config.yaml`, plots, `git.txt`; per-frame `frames_*.npz` gitignored |
| Camera check + solve | `results/phase1_camera/`, `results/phase1_camera_solve/` | - | overlays PNG, `metrics.json` |
| Phase 1b metrics | `results/phase1b_fusion/<run>/` | - | `metrics.json`, `gain_sweep.png` |
| Phase 2 metrics | `results/phase2_implied/<run>/` | - | `metrics.json`, `window_sweep.png` |

## Phase 3a (verdict test, handoff 04)

| artifact | where | size | notes |
| --- | --- | --- | --- |
| Edit/clip specs `p3a_v1` | `data/processed/p3a/p3a_v1/*.npz` (gitignored), R2 `track-a/phase3a/specs/`, Modal Volume `:/edits/p3a_v1` | ~1 MB | per-frame rotation matrices for 6 conditions, frame names, clip table; built by `experiments/phase3a_make_specs.py` |
| GeoCalib on edited frames | Modal Volume `:/geocalib/geocalib_p3a_v1/<flight>__<condition>.jsonl`, R2 `track-a/phase3a/geocalib/` | ~1 MB/clip | 8 flights x 6 conditions; local copy in `results/phase3a_verdict/geocalib_p3a_v1` |
| ANWM noise floor | `results/anwm_scoring/anwm_floor_v1/floor_{512,224}.jsonl`, R2 `track-a/phase3a/anwm_floor/` | 1 MB | 186 released level frames, two sizes |
| ANWM sampled frames | `data/raw/anwm_frames` (gitignored), Modal Volume `:/anwm_frames` | 68 MB | 12 test trajectories x 13-16 frames from the shard index via HTTP Range |
| Verdict runs | `results/phase3a_verdict/<run_id>/` | - | `metrics.json`, `config.yaml`, plots, git hash, prereg hash |
| Edit preview | `results/phase3a_verdict/<run_id>_edit_preview/` | - | contact sheets proving the warps are geometric |
| Verdict run (final) | `results/phase3a_verdict/20261109T090638Z_c0845a5/`, R2 `track-a/phase3a/verdict/<run>/` | < 1 MB | `metrics.json` (580 clip-condition records), `roc.png`, `mdv_curve.png`, `score_cdf.png`, config, git hash, prereg sha256 |
| Anomaly/ablation run | `results/phase3a_verdict/<run>_ablations/ablations.json` | < 1 MB | four nuisance fits, absorption table |
| ANWM clip scoring | `results/anwm_scoring/<run>/not_available.json` | - | Track B had not delivered clips; recorded rather than substituted |

## Track A2 (handoff 05)

Full per-object list with sha256: `results/a2_manifest.md` (71 objects, all uploaded and
hash-verified; upload report `results/a2_manifest_upload.json`).

| artifact | where | size | notes |
| --- | --- | --- | --- |
| GeoCalib rows with uncertainty/confidence, TII | `results/gate_a2/tii/`, R2 `track-a/a2/geocalib/tii/`, Volume `:/geocalib/gate_a2_tii` | 16 MB | 8 flights x 6 conditions, re-run at the Gate 3 commit |
| GeoCalib rows, generated clips | `results/gate_a2/anwm/depicted.jsonl`, R2 `track-a/a2/geocalib/anwm_depicted.jsonl` | 0.7 MB | 82 clips x 16 frames, 224 px, f = 112 |
| GeoCalib rows, level frame set | `results/gate_a2/floor/floor_224.jsonl` | 0.1 MB | 186 released level frames (gate transfer + noise floor) |
| Edit-recovery rows | `results/gate_a2/edits/edits.jsonl` | 0.8 MB | 12 clips x 7 conditions (identity, roll/pitch 5/10/20) |
| Gate validation + frozen gate | `results/a2_gate/`, `configs/gate_frozen.yaml`, R2 `track-a/a2/gate/` | 0.6 MB | per-signal AUROC, LOFO coverage, per-clip table |
| Gated verdict | `results/a2_verdict_gated[-baseline|_pairgate]/`, R2 `track-a/a2/verdict_gated/` | 3.2 MB | `k_d` in {0, 0.15, 0.295}; baseline and analysis-only pair-gate runs |
| Required tilt from commanded paths | `results/a2_anwm/`, R2 `track-a/a2/anwm_required/` | 6 MB | both interpolators, three dt, dt sweep 0.1-5 s |
| ANWM scoring | `results/a2_anwm_score/`, R2 `track-a/a2/anwm_scored/` | 0.2 MB | controls, regime tables, dt_min distributions |
