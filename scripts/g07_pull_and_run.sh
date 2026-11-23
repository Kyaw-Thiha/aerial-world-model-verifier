#!/usr/bin/env bash
# Pull the held-out stride-1 scores and run the frozen evaluation once.
set -euo pipefail
cd "$(dirname "$0")/.."
rm -rf results/g07/_dl
mkdir -p results/g07/_dl results/g07/scores results/g07/dev_scores
pixi run modal volume get --force awmv-g3-data scores/stride1 results/g07/_dl/
cp -f results/g07/_dl/stride1/heldout_c*/*.json results/g07/scores/ 2>/dev/null || true
echo "held-out score files: $(ls results/g07/scores | wc -l)"
pixi run python experiments/g07_heldout.py \
    --scores-dev results/g06/scores \
    --scores-heldout results/g07/scores \
    --out results/g07
