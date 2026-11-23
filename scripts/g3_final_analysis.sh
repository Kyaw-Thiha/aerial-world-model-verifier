#!/usr/bin/env bash
# Download every committed stride-1 score and rebuild the G06 tables.
# CPU only; idempotent.
set -euo pipefail
cd "$(dirname "$0")/.."

rm -rf results/g06/_dlfinal
mkdir -p results/g06/_dlfinal results/g06/scores
pixi run modal volume get --force awmv-g3-data scores/stride1 results/g06/_dlfinal/
cp results/g06/_dlfinal/stride1/c*/*.json results/g06/scores/
echo "score files: $(ls results/g06/scores | wc -l)"

pixi run python experiments/g3_consistency.py \
    --new results/g06/scores \
    --old "$HOME/Documents/Projects/awmv-track-i/results/i02/scores" \
    --out results/g06/consistency \
    --repeat-a results/g06/_smoke/detA --repeat-b results/g06/_smoke/detB

pixi run python experiments/g3_ablation.py --scores results/g06/scores --out results/g06/ablation
pixi run python experiments/g3_extra.py --scores results/g06/scores --out results/g06/ablation
pixi run python experiments/g3_tables.py --dir results/g06/ablation \
    --yaw results/g06/yaw_slice/yaw_summary.json \
    --consistency results/g06/consistency/consistency.json
