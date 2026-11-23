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

## 10. Track N artefacts (2026-11-22)

| path | size | what | rebuild |
| --- | --- | --- | --- |
| `results/n01_pretest/` | < 150 KB | every number in N01: population table, UZH reader medians, per-unit and per-horizon errors, summary and verdict | `experiments/n01_rotation_pretest.py` |
| `data/n01/poses/{uzh,sim}/` | 3.6 MB | DPVO poses for the 14 UZH sequences (450 frames each) and the 80 simulated clips; gitignored | `modal run remote/n_rotation.py`, then `scripts/pull_volume.py awmv-n-data poses data/n01/poses` |
| `data/n01/uzh_frames/` | 149 MB | the 450/450/223 pinhole frames of the three outdoor UZH sequences, pulled read-only from `awmv-e-data`; gitignored | `scripts/pull_volume.py awmv-e-data uzh/<seq>/frames_pinhole data/n01/uzh_frames/<seq>` |
| Modal volume `awmv-n-data` | small | `poses/` and `runs/` for the 94 DPVO runs | re-run the app |
| `results/n02/` | 12 MB (gzipped pair tables) | every number in N02: start arms, translation-direction pairs, turn test, generated track noise | `experiments/n02_*.py` (CPU) |

No R2 object was created or written. The volume is the only new Modal object; the app was stopped.
