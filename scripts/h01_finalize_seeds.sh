#!/usr/bin/env bash
# H01: final ANWM seed pipeline (reader rows -> local clips -> analysis -> tables).
# Run after both shards have finished (40 clips each).
#
#   set -a; . ./.env; set +a
#   bash scripts/h01_finalize_seeds.sh
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

for s in anwm_b anwm_c; do
  tag="${s/_/-}"
  pixi run modal run remote/h01_gate.py::main --source "$s" --out-tag "$tag"
  pixi run python scripts/pull_volume.py awmv-h-gate-results "/h01_depicted/$tag" \
      "results/h01_depicted_$tag"
done

mkdir -p results/h01_seed_depicted
cat results/h01_depicted_anwm-b/depicted.jsonl \
    results/h01_depicted_anwm-c/depicted.jsonl > results/h01_seed_depicted/depicted.jsonl

for v in awmv-h-anwm-results-b awmv-h-anwm-results-c; do
  pixi run python scripts/pull_volume.py "$v" /clips data/processed/h01_seed_clips
done

PYTHONPATH=. pixi run python experiments/h01_seed_variance.py
PYTHONPATH=. pixi run python experiments/h01_report_tables.py
echo "done: results/h01_seed_variance/summary.json and results/h01_tables/tables.md"
