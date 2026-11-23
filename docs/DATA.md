# Data: what lives where, and how to get it back

Rule of the project: **the repository holds code, configs, reports and small result summaries;
everything large goes to Cloudflare R2 or a Modal Volume.** `.gitignore` enforces it
(`data/`, `third_party/`, `.pixi/`, `results/**/*.npy|npz|mp4|pt|ckpt`).

## 1. In git (this repository)

| path | size | what |
| --- | --- | --- |
| `results/` | 217 MB, 827 tracked files | per-run summaries: `metrics.json`, CSV, JSON, small PNG overlays. **The report scripts read these** — do not gitignore them |
| `report/` | 1.1 MB | generated tables, figures, provenance (`data/*.csv`, `data/*.src.json`) |
| `reports/`, `docs/` | 436 KB, 416 KB | the numbered track reports and the documents of record |
| `configs/` | 180 KB | frozen configs and pre-registrations (immutable) |
| `src/`, `tests/`, `experiments/`, `scripts/`, `remote/` | ~2.5 MB | code |

Large binaries are excluded by suffix: only PNG summaries and JSON/CSV/YAML/MD are tracked under
`results/`. Where a run produced `.npz` per-frame arrays, the arrays are **not** in git.

## 2. On Cloudflare R2 (read-only for this consolidation)

Bucket `aerial-world-modal-verifier`, reached with the `R2_*` keys in `.env` (S3 API). Counts and
bytes measured 2026-11-15T10:03Z by `rclone size --json`, recorded in `report/data/r2_counts.json`
and `report/cutoff.json`:

| prefix | objects | bytes | contents |
| --- | --- | --- | --- |
| `track-a/` (includes `track-a/a2/`) | 288 | 3,833,554,623 | Track A edit specs, per-frame arrays, GeoCalib batches, DPVO input |
| `track-a/a2/` (nested, counted separately) | 122 | 42,490,106 | Track A2 gate frames and gate artefacts |
| `track-b/` | 3,641 | 4,936,179,889 | ANWM generated clips, injected frames, native reproduction, sheets |
| `track-c/` | 0 | 0 | **empty** — Track C has no R2 objects; its record is the branch |
| `track-d/` | 67 | 36,959,802 | Track D schema-v2 clips and statistic artefacts (includes the D branch bundle) |
| `track-e/` | 595 | 31,612,527 | UZH-FPV clip exports, per-frame arrays, backups |
| `track-h/` | 1,381 | 124,595,362 | Track H pilot clips: 80 ANWM seed clips (1,361 objects, 121.4 MB), five LingBot-World probe clips, five Wan 2.2 clips, each with a sha256 manifest in `results/h01_r2/` |
| `track-h2/` | 230 | 152,932,119 | Track H2 pilot clips: SolarWM probes/grid/seeds and full-precision LingBot probes (mp4 + meta), each with a sha256 manifest in `results/h02_r2/` |
| `track-i/` | see the Track I block below | — | Simulated labelled flights: `videos/<split>/<family>/<clip_id>.mp4`, `gt/<clip_id>.npz` (250 Hz truth) |

Access status during consolidation: an `rclone size` against `r2remote:aerial-world-modal-verifier/track-d/`
returned **403 AccessDenied**, so the counts above are the 2026-11-15T10:03Z measurement and were
**not** re-verified. Nothing was written or deleted (the `track-h/` prefix is new and was populated by Tracks H and H2).
Re-check with:

```
set -a; . ./.env; set +a
rclone size --json "r2remote:$R2_BUCKET/track-d/"
```

The project's own helpers are `scripts/sync_r2.py` (one file, sha256-verified round trip) and
`scripts/sync_r2_tree.py` (a tree, skipping large binaries), both via `src/utils/r2.py`.

## 3. On Modal (read-only for this consolidation)

No apps were running at consolidation time (`modal app list --limit 0 --json` → `[]`).
Volumes present:

| volume | created | belongs to |
| --- | --- | --- |
| `awmv-a-data` | 2026-11-08 | Track A (TII raw video, GeoCalib weights cache); the Track A2 gate extraction ran as the Modal app `awmv-a-gate` against this volume |
| `awmv-b-weights`, `awmv-b-hf-cache`, `awmv-b-native`, `awmv-b-results` | 2026-11-09 | Track B (ANWM weights, HF cache, native run, generated clips) |
| `awmv-c-models`, `awmv-c-data` | 2026-11-10 | Track C (extractors, UZH inputs) |
| `awmv-e-data` | 2026-11-13 | Track E (UZH exports, backups) |
| `awmv-smoke-vol` | 2026-11-07 | compute-plumbing smoke test |
| `awmv-i-data` | 2026-11-16 | Track I (simulated labelled set): `spike/` job specs and pose files, `frames/<clip>/` rendered clips, `geocalib/` per-clip scoring |

Pull something off a volume with `scripts/pull_volume.py <volume> <path> <dest>` (the `modal volume
get` CLI trips on an existing local directory in this version). Nothing was stopped, moved or
deleted.

## 4. Local, gitignored

| path | size | recoverable? |
| --- | --- | --- |
| `data/` | 3.3 GB, 42,535 files | From R2, or by re-running the fetch scripts. A full manifest (path, size, mtime) is in `awmv-archive/untracked/data-manifest.tsv`. Layout: `data/raw/` (1.4 GB TII video + ANWM trajectory pickles + GeoCalib tarballs), `data/clips*/` (1.2 GB extracted clip sets: `clips`, `clips_v2*`, `clips_v3`), `data/uzh/` + `uzh_members/` (370 MB), `data/processed/` (411 MB) |
| `third_party/anwm` | 1.7 GB | A separate clone of <https://github.com/EmbodiedCity/ANWM.code> at commit `657a802`. Not a submodule. 17 tests need it |
| `.pixi/` | 9.6 GB across worktrees | Rebuild with `pixi install` |
| `.env` | 400 B | **Not recoverable** — it holds the live credentials. Names only in `.env.example` |

Fetch paths (the scripts are the source of truth):

```
pixi run fetch-tii                 # scripts/fetch_tii.py   — TII mocap + video chunks (Range reads)
python scripts/fetch_tii_video.py  # full video zips
python scripts/fetch_anwm.py       # ANWM release trajectories
python scripts/fetch_anwm_frames.py
python scripts/fetch_track_e_clips.py  # UZH-FPV clip exports
python scripts/uzh_prepare.py
git clone https://github.com/EmbodiedCity/ANWM.code third_party/anwm && git -C third_party/anwm checkout 657a802
pixi run python scripts/pull_volume.py awmv-c-data poses/dpvo data/pathrec/poses/dpvo
```

`data/pathrec/poses/dpvo/` holds Track C's DPVO camera-to-world poses for the ten UZH
sequences, pulled from the Modal volume `awmv-c-data` on 2026-11-16 (12 files, 140 KB; the
sha256 of each sequence file is recorded in `results/f01_stage0/summary.json`). The volume
copies do not carry timestamps, so the frame grid is the prepared-frame order (150 frames at
15.07 Hz), which is how Track C ran them.

Stored pose sets (gitignored; each 150 frames per sequence at 15.07 Hz) are pulled with
`scripts/pull_volume.py`: DPVO from `awmv-c-data:poses/dpvo` (Track C), ViPE and MASt3R-SLAM
from `awmv-f-data:poses/{vipe,mast3r}` (Track F), and MapAnything and Pi3X from
`awmv-f2-data:poses/{mapanything,pi3x}` (Track F2). The round-2 scores record the code commit
and the volume paths; the per-file sha256 is in `results/f01_stage0/summary.json` (round 1) and
the round-2 summaries.

## 5. What is not recoverable from anywhere

* **`.env`** (R2, Modal, RunPod, W&B keys). If it is lost, re-mint the keys.
* **`.npz` per-frame arrays that were never uploaded.** Most were; the ones referenced by
  `report/data/*.src.json` are listed there by path, so a missing file can be identified exactly.
  `report/MISSING.md` lists the assets that cannot be rebuilt (E6 scoring, E8 per-axis lean
  arrays, the E2 permutation-null draws).
* **The `track-c/` R2 prefix was never populated**, so anything Track C only uploaded locally is
  not on R2. Its results are in git under `results/pathrec_*`.
* **The pre-consolidation worktrees' untracked state** is preserved in
  `awmv-archive/untracked/` (Track C's `data/` tree and its one untracked result file,
  `results/pathrec_calib/intrinsics.json`, copied as `awmv-archive/untracked/track-c/intrinsics.json`).

## 5a. Track J (branch `track-j-reader`)

| path | size | what |
| --- | --- | --- |
| `results/j_inputs/g03_folds/` | 1.4 MB | **staged copy** of G03's out-of-fold per-frame reader predictions (11 JSONs) from `track-g-segments2` `cfea5c8`, plus `_summary.json`; `results/j_inputs/provenance.json` records each sha256 |
| `results/j01_anatomy/` | < 1 MB | decomposition, ACF/PSD, bias structure, tail, blur proxy, rotation, rules |
| `results/j02_noise_budget/` | < 1 MB | synthetic noise-budget table |

Track J stages the G03 folds so the branch reads nothing from another branch at run time and
its provenance is in git. Everything else it reads (`data/processed/p3a`, `data/processed/track_d`,
`data/g03_frames`, `data/raw/dpvo`, `data/pathrec/poses`, `data/uzh`) is the shared gitignored
`data/` tree, which is identical across worktrees. It wrote nothing to R2 and created no volume.

## 6. Restore recipes

Full history restore from the consolidation bundles (any branch):

```
mkdir /tmp/awmv-restore && cd /tmp/awmv-restore && git init -q
git bundle verify /path/to/awmv-archive/backups/track-d-pre-consolidation.bundle
git fetch /path/to/awmv-archive/backups/track-d-pre-consolidation.bundle \
    'refs/heads/*:refs/heads/*'
git checkout track-d-verdict-stat
```

Or, inside the repository, `git checkout archive/track-d-pre-consolidation`.

Results needed by the report scripts are already in git under `results/`; nothing else is required
to rebuild `report/` (see `docs/REPRODUCE.md`).

## 7. Track G artefacts (2026-11-16)

Nothing large is committed. Track G reads existing clips and writes small summaries.

| path | size | what | rebuild |
| --- | --- | --- | --- |
| `data/g03_frames/<flight>/` | 372 MB | the 5,007 TII camera frames the verdict scores, extracted from the release zips, plus `labels.json` with the mocap attitude label per frame | `experiments/g03_prepare_frames.py` |
| `data/g03_uzh/<sequence>/` | 144 MB | 1,097 UZH forward-rig pinhole frames with Kalibr+Leica labels | `experiments/g03_prepare_uzh.py` |
| `data/raw/tii_video_full/` | 3.6 GB | the eight `camera_<flight>.zip` members pulled from R2 `track-a/tii_video/{autonomous,piloted}/` (until 2026-11-16 only `flight-01a` was local) | `scripts`-style ranged or plain GET against the bucket keys listed there |
| Modal volume `awmv-g-data` | small | per-fold fine-tune outputs, `<dataset>__<holdout>.json`, written by `remote/g_tilt_reader.py` | re-run the app |
| `results/g01_segments/`, `results/g02_combined/`, `results/g03_tilt_reader/` | < 2 MB each | every number in reports G01-G03, in git | the three experiment scripts + `experiments/g03_evaluate.py` |

R2 listing worked again on 2026-11-16 (the 403 recorded in section 2 was not reproduced), so the
TII video zips can be fetched by key without a Modal volume. Nothing was written or deleted on
R2 or on the other tracks' volumes; Track G created exactly one volume, `awmv-g-data`.

## 8. Track G2 artefacts (2026-11-21)

| path | size | what | rebuild |
| --- | --- | --- | --- |
| `results/g04_path_demand/` | < 1 MB | every number in report G04 (`summary.json`, `variants.csv`) | `experiments/g04_path_demand.py` |
| `results/g05_spike_pilot/` | < 10 KB | the exploratory 3-clip pilot on Track I's I01 renders | `experiments/g05_spike_pilot.py` |

No new bulk data: G2 reads `data/processed/track_d/clips` (already documented), and the pilot
reads Track I's worktree `../awmv-track-i/results/i01_spike` read-only. No Modal volume and no
R2 object was created or written by this track.

## 9. Round-2 consolidation backups (2026-11-22)

No data was moved. The pre-round-2 tips of all nine refs (the seven merged branches, `main`, and
the stale `track-g-segments`) are tagged `archive/<branch>-pre-round2` in the repository and
bundled outside it:

* `~/Documents/Projects/awmv-archive/backups/<branch>-pre-round2.bundle` — one per ref,
  `git bundle verify`-clean; sha256 in `SHA256SUMS-round2.txt` beside them. Each is about 139 MB.
* Restore one into an empty repository: `git fetch <bundle> 'refs/heads/*:refs/heads/*'`, then
  check out the branch.

No R2 object and no Modal volume was created, written or deleted in round 2.

## 10. Track I — simulated labelled set (2026-11-22)

<!-- track-i-dataset:begin (generated by experiments/i02_build_manifest.py) -->
### Track I — simulated labelled flight set (`track-i/` on R2)

Pre-registration `configs/sim_prereg.yaml` sha256 `245b68bfb3499181…`; frozen clip table `configs/sim_splits_frozen.json` sha256 `cdf66e28387178be…` (300 clips: 200 development, 100 held-out).

Scene families are split before any clip existed and never mixed: **development** `urban_checker`, `forest_ridge`, `industrial` (seeds 101-106); **held-out** `field_settlement`, `coastal_cliff` (seeds 201-206). A held-out clip is rendered with the batch but is not scored until the development pass is committed.

Objects on R2: `track-i/videos/<split>/<family>/<clip_id>.mp4` (H.264 CRF 18, 30 fps), `track-i/gt/<clip_id>.npz` (250 Hz truth: position, velocity, acceleration, yaw, commanded and rendered attitude, required and depicted lean/tilt, defect mask). Manifest: `results/i02/manifest.csv` (one row per clip, sha256 of both objects, the exact violation interval and the development/held-out label).

| split | clips on R2 | families | defect families | frames |
| --- | --- | --- | --- | --- |
| dev | 200 | forest_ridge, industrial, urban_checker | feasible, mirrored_roll, partial, thrust_limit, tilt_removed, tilt_scaled | 99591 |
| heldout | 100 | coastal_cliff, field_settlement | feasible, mirrored_roll, partial, thrust_limit, tilt_removed, tilt_scaled | 49859 |

First usable batch: the clips listed in `results/i02/manifest.csv`; each row carries the interval `[interval_start_s, interval_end_s]` and the magnitude in degrees, so a detector (Track G2) and a clip-level verdict (Tracks F2, G2) can both read the same labels.
<!-- track-i-dataset:end -->

## 11. Track H3/H4 (2026-11-22)

| location | contents |
| --- | --- |
| git `results/h03/` | 150-frame manifest, per-frame scored CSV, classical rows, reader/PF rows (JSONL), rotation sign probe, analysis JSON, overlay PNGs |
| git `results/h03_recompute/` | Perspective Fields rows on the full H01/H02 clips and the reader-vs-PF comparison |
| git `results/h04/paths/` | per-path verification numbers (JSON) and the required side; `paths.npz` is gitignored |
| git `results/h04/depicted_main/`, `scored_main*.json` | campaign reader rows and scores |
| R2 `track-h3/clips/solarwm-main/`, `track-h3/reader/`, `track-h3/manifests/` | the 234 campaign mp4s with per-object sha256, the reader rows, and the manifests |
| Modal `awmv-h3-data` | `/h03_frames` (the 150-frame sample), `/h03_readers` (reader + PF rows), `/h03_recompute` (full-clip PF rows) |
| Modal `awmv-h3-results` | `/solarwm/main/<clip>/{clip.mp4,frames/*.png,manifest.json}`, `/solarwm/probes`, `/solarwm/main_yaw+20` (yaw control), `/h04/depicted_*` (reader rows), `/h04/yaw_poses.json` |
| local `data/h03_frames` (42 MB), `data/h04_solarwm_inputs` (61 MB), `data/h03_label` | sample frames, SolarWM inputs, the optional labelling tool; all gitignored |

R2 access status: with the key in `.env`, **reads are forbidden** (403 on HEAD/GET/list for
every prefix tried) and only `PUT` under `track-h3/` was used; every object is verified by
re-reading its sha256 from the upload response.  Track I clips and ANWM clips were therefore
pulled from their Modal volumes, not from R2.

## 12. Track K (2026-11-22)

Track K used only data already on disk: TII mocap (`data/raw/tii/`) and UZH-FPV Leica truth
(`data/uzh/*/truth.npz`), plus the schema-v2 TII clip tree (`data/processed/track_d/clips/`)
and the frozen gate rows. No R2 object was written and no Modal app or volume was used.

New generated tree (gitignored, reproducible from the committed generator):

| path | what |
| --- | --- |
| `data/processed/track_k/clips_v3/<flight>/<clip>__<condition>/` | schema v3: the same clip arrays as v2 plus `lean_req_v3` (the selected airframe model's required lean at the frame times) and `tilt_req_v3_deg`; the v2 arrays are kept under `*_v2_*` names for provenance. Built by `scripts/export_schema_v3.py` from `results/k02_fit/<run>/fit.json`. The v2 tree is read only. |

Small summaries under `results/k01/` and `results/k02_*` are in git; the v3 tree is not
(it is ~20 MB of `.npz`, regenerable in under a minute from the committed fit file).

## 13. Track L (2026-11-22)

| location | objects | contents |
| --- | --- | --- |
| R2 `track-l/l01/per_frame/` | 80 | per-frame reader arrays for the feasible simulated development clips (gravity, lean, truth, gate inputs), manifest `results/l01/r2_manifest.json` |
| Modal `awmv-l-data` | volume | `/l01/inputs` (videos, truth), `/l02/frames` (3,200 training frames), `/l02/real_frames` (5,007 TII frames), `/l02/folds` (per-fold records), `/l02/models` (final checkpoints), `/l02/downstream` (edit-reader outputs) |
| local `data/l01`, `data/l02` | gitignored | downloaded inputs and score arrays; reconstructable from R2 and the volume |

The L01/L02 jobs read Track I's published videos and truth from R2 after verifying their sha256
against `results/i02/manifest.csv`; nothing under `track-i/` was written.

## 14. Track G3 additions (2026-11-22)

| path | where | what |
| --- | --- | --- |
| `results/g06/scores/<clip_id>_score.json` | git | stride-1 GeoCalib reader output for all 170 simulated development clips (per-frame `lean_est`, `lean_req`, `tilt_est`, `tilt_gt`, `up_error`). ~20 MB. Published so a later track can reuse them without a GPU run. |
| `results/g06/consistency/` | git | stride-1 versus stride-2 comparison and the run-to-run envelope |
| `results/g06/ablation/` | git | A1-A5 tables, MDV curves, `tables.md` |
| `results/g06/yaw_slice/` | git | the auxiliary yaw slice's analysis |
| `yaw_slice/<clip_id>/` | Modal volume `awmv-g3-data` | the 12 re-rendered yaw-control clips (`<clip_id>.mp4`, `render.npz`, `yaw_meta.json`); not in git |
| `track-g3/yaw_slice/` | R2 | publication target for the same 12 clips (pending) |

The yaw slice is Track G3's own auxiliary set: 12 feasible development clips re-rendered with the
camera yawed about world up. It is **not** part of Track I's frozen 300 and is never pooled with
it (`configs/g06_yaw_slice.json`).

## 14a. Track G3 held-out artefacts (2026-11-23)

| path | size | what | rebuild |
| --- | --- | --- | --- |
| `results/g07/scores/` | ~3.5 MB, 100 files | held-out stride-1 reader output (`<clip>_score.json`), pulled from `awmv-g3-data:scores/stride1/heldout_c*` | `scripts/g07_score_heldout.sh` then `g07_pull_and_run.sh` |
| `results/g07/*.csv`, `sim_summary.json` | < 1 MB | every number in report G07 | `experiments/g07_heldout.py` |
| `awmv-g3-data` volume | — | the authoritative score copies plus the dev scores and yaw slice | — |

The 100 held-out score files are committed because they are small and every held-out number
depends on them; the mp4/GT inputs remain on Track I's volume (`awmv-i-data:i02/clips/`), read
only. Nothing was written to Track I's volume or R2.

## 15. Track M — Gate B reader inputs (2026-11-22)

Track M's Gate B reads two independent estimators on the same 224 px centre crop. Nothing is
staged in git; the inputs live on the Modal volume `awmv-m-data` under `/m02/inputs` and the
scores under `/m02/scores/<tag>`. The plan (`plan.json`, sha256 `1e3b12e9…`) fixes the frame
list, the stride and the focal prior per source.

| source | frames | stride | focal prior (224 crop) | staged from |
| --- | --- | --- | --- | --- |
| TII real clips, 6 conditions | 100 clips x 48 | 2 | 112 px | camera zips on `awmv-a-data/tii_video`, edit rotations on `awmv-a-data/edits/p3a_v1` |
| simulated dev clips | 170 clips | 4 | 199.1 px (320 x 224/360) | R2 `track-i/videos/dev` + `track-i/gt` |
| H04 SolarWM | 234 clips | 2 | 112 px | R2 `track-h3/clips/solarwm-main` |
| H01 ANWM | 80 clips x 16 | 1 | 112 px | R2 `track-h/clips/anwm-seeds/C` |

The simulated set's truth is the render's own attitude (`lean_req`, `lean_dep`, `R_cb` in the gt
npz); the frozen reader on the simulated clips is Track L's stored per-frame table
(`track-l/l01/per_frame/`, 80 objects, 3.6 MB), not re-run here.

## 16. Track N artefacts (2026-11-22)

| path | size | what | rebuild |
| --- | --- | --- | --- |
| `results/n01_pretest/` | < 150 KB | every number in N01: population table, UZH reader medians, per-unit and per-horizon errors, summary and verdict | `experiments/n01_rotation_pretest.py` |
| `data/n01/poses/{uzh,sim}/` | 3.6 MB | DPVO poses for the 14 UZH sequences (450 frames each) and the 80 simulated clips; gitignored | `modal run remote/n_rotation.py`, then `scripts/pull_volume.py awmv-n-data poses data/n01/poses` |
| `data/n01/uzh_frames/` | 149 MB | the 450/450/223 pinhole frames of the three outdoor UZH sequences, pulled read-only from `awmv-e-data`; gitignored | `scripts/pull_volume.py awmv-e-data uzh/<seq>/frames_pinhole data/n01/uzh_frames/<seq>` |
| Modal volume `awmv-n-data` | small | `poses/` and `runs/` for the 94 DPVO runs | re-run the app |
| `results/n02/` | 12 MB (gzipped pair tables) | every number in N02: start arms, translation-direction pairs, turn test, generated track noise | `experiments/n02_*.py` (CPU) |
| `results/n04/` | < 1 MB | every number in N04: clip hashes, tracking, horizons, ANWM fit | `experiments/n04_generated_rotation.py` |
| `results/n02b/`, `results/n03/` | < 1 MB | every number in N02b and N03: per-clip tier scores, gate curve | `experiments/n02b_part2_verdicts.py`, `experiments/n03_agreement_gate.py` |
| `results/n05/`, `results/n06/` | 1 MB | N05 per-clip tier scores, diagnostics, two draws; N06 drift and reproducibility | `experiments/n05_*.py`, `experiments/n06_mount_drift.py` |
| `data/n05/{videos,gt,scores,poses}` | 300 MB | the 200 dev clips (held-out absent), their gt, reader scores, DPVO poses (2 draws + half res) | manifest hash-verified; `remote/n_readers.py`, `remote/n_rotation.py` |
| Modal volume `awmv-n-data/sim200` | ~1 GB | 200 videos + gt + reader scores + poses | re-run the apps |
| `data/n01/poses/uzh_edits/` | 1 MB | DPVO poses for the 63 Track E v3 edited clips | `modal run remote/n_rotation.py --which uzh_edit` |
| `data/n04/poses/gen/fov*` | 4 MB | DPVO poses for the 234 H04 clips and 24 focal-sensitivity runs | `modal run remote/n_rotation.py --which gen` |

No R2 object was created or written. The volume is the only new Modal object; the app was stopped.

## 17. Wave-2 consolidation backups (2026-11-23)

The pre-wave-2 tips of the seven merged branches and `main` are tagged
`archive/<branch>-pre-round2-wave2` and bundled in
`~/Documents/Projects/awmv-archive/backups/<branch>-pre-round2-wave2.bundle`, each
`git bundle verify`-clean, sha256 in `SHA256SUMS-round2-wave2.txt` (142-205 MB each). Restore as in
section 9. No R2 object and no Modal volume was created, written or deleted in wave 2.
