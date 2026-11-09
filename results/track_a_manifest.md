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
