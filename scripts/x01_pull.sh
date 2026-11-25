#!/usr/bin/env bash
# Pull every small X01 artefact (JSON/JSONL/log) off awmv-x-results into results/x01/.
set -euo pipefail
cd "$(dirname "$0")/.."
set -a; . ./.env; set +a
PYTHONPATH=. pixi run python - <<'PY'
import json
from pathlib import Path
import modal

fn = modal.Function.from_name("awmv-x-readers", "collect")
files = fn.remote()
root = Path("results/x01")
n = 0
for rel, text in files.items():
    if rel.startswith("depicted/"):
        dst = root / "depicted" / Path(rel).name if Path(rel).parent.name == "depicted" \
            else root / rel
    else:
        dst = root / "runs" / rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(text)
    n += 1
print(f"wrote {n} files under {root}")
PY
