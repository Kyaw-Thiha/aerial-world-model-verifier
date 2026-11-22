#!/usr/bin/env bash
# Track G3: score batch 2 (the 30 development clips Track I published after the G06 freeze),
# then rebuild the dev tables with them. Frozen thresholds are unchanged: batch 2 enters the
# same leave-one-scene-out calibration, it does not reset it.
set -euo pipefail
cd "$(dirname "$0")/.."

ids=$(cat results/g06/dev_ids_batch2.txt)
pixi run modal run remote/g3_score.py::score_l40s \
    --clip-ids "$ids" --out scores/stride1/c5 --batch 8

rm -rf results/g06/_dlb2
mkdir -p results/g06/_dlb2
pixi run modal volume get --force awmv-g3-data scores/stride1 results/g06/_dlb2/
cp results/g06/_dlb2/stride1/c5/*.json results/g06/scores/
rm -rf results/g06/_dlb2
echo "scores now: $(ls results/g06/scores | wc -l)"

pixi run python experiments/g3_ablation.py --scores results/g06/scores --out results/g06/ablation
pixi run python experiments/g3_extra.py --scores results/g06/scores --out results/g06/ablation
pixi run python experiments/g3_tables.py --dir results/g06/ablation \
    --yaw results/g06/yaw_slice/yaw_summary.json \
    --consistency results/g06/consistency/consistency.json
