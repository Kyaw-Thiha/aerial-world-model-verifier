#!/usr/bin/env bash
# X01: the exact commands of the generator-breadth campaign, in the order they were run.
# Everything runs on Modal; nothing here writes outside the awmv-x-* volumes or results/x01.
set -euo pipefail
cd "$(dirname "$0")/.."
set -a; . ./.env; set +a

# 0. the path set (local, CPU)
pixi run python experiments/x01_common_paths.py

# 1. SANA-WM image + weights (CPU; the image build is the expensive part)
pixi run modal run remote/x01_sana.py::download

# 2. SANA-WM probe: two clips, to price the campaign before committing to 25
pixi run modal run remote/x01_sana.py::main --mode probe --tag probe --step 20

# 3. generation
pixi run modal run remote/x01_solarwm.py::main --tag d42 --chunks 4 --sub 7 --seed 42 --wait
pixi run modal run remote/x01_solarwm.py::main --tag d43 --chunks 2 --sub 4 --seed 43 --wait \
  --ids arc_10__S1,arc_20__S1,s_10__S1,slalom_10__S1,climb_02__S1,climb_turn_10__S1,accel_brake_10__S1,real_uzh_of1__S1
pixi run modal run remote/x01_anwm.py::main
pixi run modal run remote/x01_lingbot.py::main --frame-num 81 --steps 40
pixi run modal run remote/x01_sana.py::main --tag full --step 20 --shard 0 --shards 2

# 4. the frozen reader on every produced clip (one call per model, 2-4 shards)
pixi run modal run remote/x01_readers.py::main --model solarwm --tag d42 --shards 4
pixi run modal run remote/x01_readers.py::main --model solarwm --tag d43 --shards 2
pixi run modal run remote/x01_readers.py::main --model anwm --tag x01 --shards 2
pixi run modal run remote/x01_readers.py::main --model lingbot --tag x01 --shards 2
pixi run modal run remote/x01_readers.py::main --model sana --tag full --shards 2

# 5. pull the small artefacts and score
bash scripts/x01_pull.sh
pixi run python experiments/x01_score.py
