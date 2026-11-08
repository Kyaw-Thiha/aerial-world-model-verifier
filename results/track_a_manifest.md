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
