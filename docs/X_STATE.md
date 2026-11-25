# Track X state — final pipeline, generator breadth and one-shot held-out

Branch `track-x-final-pipeline` (worktree `~/Documents/Projects/awmv-x`), from `main` `8f404af`.
Handoff 32. Ceiling **35 USD**; Stage 1 at most 30, Stage 4 at most 5.

## Gating status (the rule of this track)

Stages 2-4 require Track P and Track V to have written their **decision** reports. On
2026-11-25T01:00-05:00:

* `track-p-rate-envelope` at `667ae11`: **P0 report written** (`reports/P0_error_budget.md`,
  stop-rule verdict "proceed") and **P1 pre-registered**; no P1/P2 decision report yet.
* `track-v-vertical` at `bcbdd40`: **V1 pre-registered** and staging done (ANWM 16, SolarWM 18
  clips scored); no V0/V1 decision report yet.

So Stage 1 is proceeding and Stages 2-4 are **not** started. Re-check with the resume command
below before any Stage-2 work.

## Status at the cut-off (2026-11-25 ~02:45 EST)

Stage 1 ran; X1 is written with the numbers that exist. Stages 2-4: the design is frozen
(`configs/x_final_frozen.yaml`, `0d10597b…`) and X2 records the development evidence that exists;
**the held-out pass was not run** (X3 says so). The blocking item is one script that composes the
frozen components into a single development table with coverage and denominators.

## Done

* Frozen hashes verified at start: `configs/sim_splits_frozen.json` `cdf66e28…`,
  `configs/sim_prereg.yaml` `245b68bf…` — both match the handoff.
* `configs/x01_common_paths_prereg.yaml` — Stage-1 pre-registration (models, 25 paths, envelope
  rule, sampling, costs, criteria), committed **before** any clip was generated (`2b306c4`).
* `experiments/x01_common_paths.py` → `results/x01/paths.npz`, `results/x01/paths_summary.json`
  (1,248 arrays, 104 model-path cells) and the per-model inputs under `data/x01/` (gitignored).
* Apps: `remote/x01_solarwm.py`, `remote/x01_anwm.py`, `remote/x01_lingbot.py`,
  `remote/x01_sana.py`, `remote/x01_readers.py`. Volumes `awmv-x-weights`, `awmv-x-results`.
* Runner: `scripts/x01_run.sh` (exact commands).

## Generation, as it stands

* SolarWM draw 42 (26 clips) and draw 43 (8 clips): **done**.
* SANA-WM: **23 of 24** clips (outdoor_forward_5 has no admissible window).
* ANWM: **17 of 25** clips at the cut-off, still running (218 s/clip on one A10).
* LingBot-World: **0 clips** after three launches (the intrinsics file must be an `(F,4)` array;
  fixed twice, then the model load + first clip never completed in the window). Reported as not
  produced, with the spend recorded.
* Readers: SolarWM 26+8, SANA 5 (shard still running), ANWM 17.

Every Modal app is stopped or finishes on its own; no detached orphan was left running. If ANWM is
still running when this is read, it will write the remaining clips to `awmv-x-results:/x01/anwm/C`
and the next step is to re-read and re-score it.

## Next (in order)

1. When the four generation apps finish: pull the reader runs and score.
   `pixi run modal run remote/x01_readers.py::main --model <model> --tag <tag> --shards 4`
   then `scripts/x01_pull.sh` and `pixi run python experiments/x01_score.py`.
2. Write `reports/X1_generator_breadth.md` (decision first), update `docs/EXPERIMENTS.md`,
   `docs/DATA.md`, `docs/decisions.md`, `docs/COST_LEDGER.md`.
3. Re-check P and V. If both have decision reports, continue with Stage 2 (final design,
   `configs/x_final_frozen.yaml`, sha256 in `docs/decisions.md`), Stage 3 (development evidence,
   cross-generator table, human labelling kit) and — **only after the freeze commit** — the
   single Stage-4 held-out run.
4. If P or V is still missing: stop here, do not guess their outcomes.

## Exact resume command

```
cd ~/Documents/Projects/awmv-x && git fetch origin && \
  for b in track-p-rate-envelope track-v-vertical; do
    echo "== $b"; git log --oneline origin/$b -3 2>/dev/null || git log --oneline $b -3;
    git ls-tree --name-only $b reports/ | grep -E '/(P|V)[0-9]';
  done
```

If (and only if) `reports/P1_*.md`/`reports/P2_*.md` and `reports/V0_*.md`/`reports/V1_*.md`
exist with decision paragraphs, run Stages 2-4 in the order of the handoff.

## Spend

Estimate posted before any job (in the pre-registration): **20.5 USD** cap, ceiling 35.
Actuals are in `docs/COST_LEDGER.md` (filled from the run JSONs and `modal billing report`).
