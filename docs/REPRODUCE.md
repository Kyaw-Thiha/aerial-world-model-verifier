# Reproducing the checks, the report assets and the report

Everything here is CPU-only. No GPU job, no Modal app and no network access is needed except the
first `pixi install`.

## 1. Environment

```
pixi install                     # default env, CPU only (python 3.12)
pixi run -e gpu ...              # only if you need torch/opencv locally
pixi run test                    # pytest -q
```

OR use the checked-in lockfile exactly as committed — `pixi.lock` is identical on every track
branch and was not regenerated during consolidation. To verify the lock resolves on a clean
checkout:

```
git clone <this repo> /tmp/awmv-clean && cd /tmp/awmv-clean
pixi install --locked
pixi run pytest -q
```

Known dependency of the suite: `tests/test_generation_paths.py` (17 tests) imports
`third_party/anwm`, which is a **gitignored clone** of
<https://github.com/EmbodiedCity/ANWM.code> (commit `657a802`):

```
git clone https://github.com/EmbodiedCity/ANWM.code third_party/anwm
git -C third_party/anwm checkout 657a802
```

Without it those 17 tests fail and the rest pass. That is a property of the project, not of the
consolidation (the same 17 failed in the pre-consolidation worktrees that lacked the clone).

## 2. Test suite

On the consolidated `main` after round 1 (2026-11-15):

```
$ pixi run pytest -q
221 passed, 3 skipped
```

After round 2, wave 1 (2026-11-22, Tracks F, F2, G, G2, H, H2 and J merged):

```
$ pixi run pytest -q
258 passed, 3 skipped
```

After round 2, wave 2 (2026-11-23, Tracks I, H3/H4, K, L, G3, M and N added):

```
$ pixi run pytest -q
328 passed, 4 skipped
```

Without the `third_party/anwm` clone the wave-2 suite reports 17 failed (only
`tests/test_generation_paths.py`), 311 passed, 4 skipped.

Reference points from before consolidation: Track D `142 passed`; `report-assets`
`136 passed, 6 skipped` (it predates D's last commit); `main` alone `25 passed, 6 skipped`.

## 3. Rebuild the report assets

```
cd report
python3 scripts/make_all.py      # 16/16 asset groups built
python3 scripts/check_numbers.py # cross-check, rewrites CHECKS.md
```

`make_all.py` reads only stored results (`results/`, `reports/`, `docs/`, `configs/`), never an
experiment. Expected result: **16/16 asset groups built**, 48 table fragments, 11 figures and 44
`data/` files regenerated. Round 2 reproduces the same result and the same `CHECKS.md` baseline;
the new tracks' LaTeX candidates in `report_candidates/` are **not** part of `make_all.py` and do
not change any asset.

Determinism, measured during consolidation:

* every `.tex`, `.csv` and `.json` asset came back **byte-identical**;
* the 11 vector PDFs differ only in the `/CreationDate` metadata that matplotlib writes
  (identical size, identical content after normalising the date);
* `CHECKS.md` came back byte-identical.

Where the scripts read from. `report/scripts/common.py` resolves each track's sources in this
order:

1. `$AWMV_SOURCES/awmv-track-<track>` (default `AWMV_SOURCES=/home/kevin/Documents/Projects`) —
   the pre-consolidation per-branch worktrees, if they still exist;
2. this repository's own root — the consolidated layout, where every track's `results/` and
   `reports/` live on one branch.

Both layouts hold the same bytes at the same relative paths. Under layout 2 the provenance files
(`report/data/*.src.json`) keep the original per-branch `branch`/`commit` values from
`report/cutoff.json` and record the path actually read; `tab_repro_1` and `tab_frozen-configs_1`
come out identical either way. To force one layout:

```
AWMV_SOURCES=/nonexistent python3 scripts/make_all.py   # consolidated repo only
AWMV_SOURCES=/home/kevin/Documents/Projects python3 scripts/make_all.py  # worktrees if present
```

## 4. The numeric cross-check baseline

`report/CHECKS.md` records every value the assets print against the value the source report
quotes, with tolerances (angles 0.02 deg, rates/slopes 0.005, counts exact). The baseline is:

```
Checked 103 values: 100 match, 2 mismatch, 1 not checkable.
```

The two mismatches are shown, not resolved:

* **E1** drag reduction on `flight-01p-ellipse`: report 5.68 deg, stored result 5.629 deg;
* **E7** DPVO calibrated median direction error: report 3.9 deg, stored result 4.18 deg.

The one "not checkable" is E5's per-design depicted tilt medians (0.77-0.79 deg), which exist only
as table values. Any *change* in these numbers is a finding to report, not to fix.

`report/MISSING.md` lists the assets that cannot be built at all (E6 scoring arm, E8 per-axis lean
arrays, E2 permutation-null draws, and others), each with what would make it buildable.

## 5. Frozen configs

Their sha256 must match `report/tables/tab_frozen-configs_1.tex`:

```
cd /path/to/repo
sha256sum configs/phase1.yaml configs/verdict_preregistration.yaml configs/gate_frozen.yaml \
          configs/pathrec_prereg.yaml configs/gain_frozen.yaml configs/coherence_frozen.yaml \
          configs/window_bandwidth_prereg.yaml
```

Expected prefixes (full value in the table's source and in the CSV `report/data/repro.csv`):

| config | sha256 (first 16) |
| --- | --- |
| `configs/phase1.yaml` | `c5d3b55e3a2877af` |
| `configs/verdict_preregistration.yaml` | `c8ffac93521cbaf2` |
| `configs/gate_frozen.yaml` | `dc0cc4ff24e0a35d` |
| `configs/pathrec_prereg.yaml` | `845f4c698f7a63cc` |
| `configs/gain_frozen.yaml` | `28f91f7ffd94c106` |
| `configs/coherence_frozen.yaml` | `40a40beadf372552` |
| `configs/window_bandwidth_prereg.yaml` | `75b208da5889f133` |

These files are immutable: do not edit, move or rename them. `configs/*_proposal.yaml` and
`configs/*_amendment_proposal.yaml` are proposals, not frozen.

## 6. Building the LaTeX

The assets-only test document (tables + figures laid out with no prose) builds in place:

```
cd report
latexmk -pdf assets_test.tex      # or: pdflatex assets_test && pdflatex assets_test
```

It produces `report/assets_test.pdf` (gitignored, ~400 KB) with no undefined references and no
missing `\input`/`\includegraphics`.

The full report sources (`main.tex`, `sections/`) are expected to live in `report/` and to
`\input{tables/...}` / `\includegraphics{figures/...}` relative to `report/`, so that
`latexmk -pdf main.tex` from `report/` works with the same asset paths as `assets_test.tex`.

> **Status:** the LaTeX report sources were **not present** in either consolidation
> (`report_tex` / `report_tex.zip` was never provided, and round 2 searched every branch,
> worktree and the archive). Only `assets_test.tex` is in the repository; it builds to a 9-page
> PDF with 0 undefined references. If the sources are added later, put `main.tex` and
> `sections/` under `report/` and build from there; if their macros point at `../report/tables`,
> change the include root to `report/` rather than moving the assets. `report_candidates/*.tex`
> are section drafts and are not included in either build.

## 7. Verifying provenance after the fact

```
python3 - <<'PY'
import json, glob
for f in sorted(glob.glob("report/data/*.src.json")):
    d = json.load(open(f))
    for s in d["sources"]:
        print(d["asset"], s.get("branch"), s.get("commit", "")[:7], s.get("path"))
PY
```

Every listed `commit` is reachable from `main` (checked with `git merge-base --is-ancestor`
during consolidation; see `docs/CONSOLIDATION_REPORT.md`).

## 8. Track I: the simulated labelled set

Everything below is deterministic from `configs/sim_prereg.yaml` +
`configs/sim_splits_frozen.json`; the table's sha256 is published in `docs/DATA.md` and the
prereg's in the same block.

```
# 1. the frozen table (writes or checks configs/sim_splits_frozen.json)
python experiments/i02_freeze_splits.py            # write
python experiments/i02_freeze_splits.py --check    # verify it still reproduces

# 2. ground truth at the video rate and the composition summary (no GPU)
python experiments/i02_export_gt.py

# 3. render a slice on Modal (EEVEE, L40S); --clip-ids, or --split dev --start N --limit M
pixi run modal run remote/i_dataset.py --split dev --start 0 --limit 50 --commit <sha>

# 4. move the slice to R2 and update the manifest + docs/DATA.md
python experiments/i02_build_manifest.py --pull --verify --docs

# 5. the reader on rendered clips (development clips only until the dev pass is closed)
pixi run modal run remote/i_score.py --clip-ids i02_dev_000,i02_dev_001 --stride 2
```

Rendering is resumable: a clip whose `i02/clips/<id>/<id>.mp4` is already on the volume is
skipped (`--force` re-renders). Publishing is idempotent: rows with an `r2_video` key are not
uploaded again. The render app writes video and truth to the volume and nothing to R2 because
the Modal secret cannot write to `track-i/` (`docs/decisions.md`); the workstation's key can.
