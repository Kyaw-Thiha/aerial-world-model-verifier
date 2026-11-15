# aerial-world-model-verifier (AWMV)

A physics-based verifier for aerial video world models. A quadrotor can only thrust along its
body axis, so the path a flight takes *requires* a particular tilt (`a + g e3` gives the thrust
axis; drag modifies it). The verifier measures the tilt the video actually **depicts** from the
pixels and compares it with the tilt the depicted motion **requires**. A generated flight that
accelerates without tilting is infeasible, and the verifier is built to say so with a calibrated
threshold and a stated abstention rule.

First target: **ANWM**, an action-conditioned aerial world model whose actions are 4-DoF
(dx, dy, dz, dyaw) and whose authors state that pitch and roll are ignored — so its flights
accelerate without tilting by construction. The dataset under test is **TII Race Against the
Machine** (real fisheye video with mocap truth) and, for transfer, **UZH-FPV**.

## What the evidence says (as of 2026-11-14)

* The **measurement floor is good enough**: the mocap-only physics floor is 1.24 deg (fitted drag)
  and the pixel-based depicted tilt is ~4 deg per frame on TII.
* The **verdict works on real flights with a stated scope**: it catches tilt removal and mirrored
  roll on piloted TII flights; autonomous racing flights mostly abstain.
* ANWM's generated flights **depict ~0.04 deg of tilt per degree designed** against the ~0.5-1.0
  the physics requires, so they are infeasible on the designed manoeuvres — with the caveats in
  `reports/08_track_a2_anwm_scored.md`.
* The **window rule** (a label-free acceleration-bandwidth choice, `0.5 / f95`) and the
  **gain/coherence statistics** are validated on piloted TII and indoor UZH forward rigs, and
  **not** validated elsewhere (`docs/prototype_v0.1.md`, `docs/verdict_prototype.md`).

Every number in the report comes from a stored result file; `report/CHECKS.md` records 103
cross-checks against the source reports (100 match, 2 mismatch, 1 not checkable) and never
resolves a disagreement silently.

## Layout

```
configs/        frozen configs and pre-registrations (immutable), proposals, experiment yaml
src/            flat package: datasets, physics, perception, fusion, depicted, score,
                pathrec, generation, negatives, verdict, utils
experiments/    one script per experiment/phase (phase0_floor.py ... e_window_rule.py)
remote/         Modal apps (GPU batch jobs); nothing here runs locally
scripts/        fetch_*.py, sync_r2*.py, export/pull helpers
tests/          unit tests (221 passing on main; 17 need third_party/anwm)
docs/           decisions log, project structure, track reports index, experiments index,
                data map, reproduction guide, handoffs, archive pointers
reports/        the numbered track reports (A 01-05, A2 06-08, B b01-b06, C C01-C09,
                D D01-D04, E E01-E05) — see docs/reports/INDEX.md
results/        stored run outputs the report scripts read (small JSON/CSV/PNG only)
report/         the built report assets: tables/, figures/, data/, scripts/, ASSETS.md,
                CHECKS.md, MISSING.md, cutoff.json, assets_test.tex
data/           raw and processed data (gitignored, 3.3 GB) — see docs/DATA.md
third_party/    the ANWM clone (gitignored)
```

## Setup

```
pixi install          # conda+pypi env from the committed pixi.lock (python 3.12, CPU only)
pixi run test         # pytest -q
```

The suite depends on the gitignored ANWM clone for 17 of its tests:

```
git clone https://github.com/EmbodiedCity/ANWM.code third_party/anwm
git -C third_party/anwm checkout 657a802
```

Secrets live in `.env` (gitignored; names only in `.env.example`). Never commit them.

## Where things are

| I want… | go to |
| --- | --- |
| the report's tables and figures | `report/tables`, `report/figures`; index in `report/ASSETS.md` |
| the cross-checks and what is missing | `report/CHECKS.md`, `report/MISSING.md` |
| which experiment is which, and what regenerates it | `docs/EXPERIMENTS.md` |
| every report, one line each, with status | `docs/reports/INDEX.md` |
| how to rebuild the assets and check them | `docs/REPRODUCE.md` |
| data: git vs R2 vs Modal vs local | `docs/DATA.md` |
| why a structural choice was made | `docs/decisions.md` (per-entry track tags `[A]`…`[E]`) |
| the planned repository growth | `docs/project_structure.md` |
| the literature and research plan | `docs/project_context.md`, `docs/paper_tracker.csv` |
| what superseded what | `docs/archive/README.md` |

## Compute and storage

Local CPU for tests, the report build and analysis; **Modal** for GPU batches (weights cached on
`awmv-*` volumes); **Cloudflare R2** (bucket `aerial-world-modal-verifier`) for large artefacts
under the prefixes `track-a/`, `track-a/a2/`, `track-b/`, `track-c/` (empty), `track-d/`,
`track-e/`. Nothing large is committed: `.gitignore` drops `data/`, `third_party/`, `.pixi/` and
the large binary suffixes under `results/`.

## Repository state

This repository is the **consolidation of seven track branches** (A, A2, B, C, D, E and the
report-assets branch) into `main`, done on 2026-11-15 with merge commits and no history rewriting.
The pre-consolidation tips are tagged `archive/<track>-pre-consolidation`, and the result is
tagged `consolidated-2026-11-15`. What was merged, every conflict and how it was resolved, and
how to roll back: `docs/CONSOLIDATION_REPORT.md`.
