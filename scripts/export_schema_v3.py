"""Export the TII clip set in schema v3 (model-required lean) without touching v2.

    PYTHONPATH=. python scripts/export_schema_v3.py --fit results/k02_fit/<run>/fit.json

Writes ``data/processed/track_k/clips_v3/<flight>/<clip>__<condition>/`` and a manifest
with the sha256 of every file.  The model is the one the pre-registered parsimony rule
selected; its parameters come from the full eight-flight fit in ``fit.json``.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.airframe.schema_v3 import export_tree

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit", default=None,
                    help="fit.json (default: the latest results/k02_fit/*/fit.json)")
    ap.add_argument("--model", default=None, help="default: the selected model")
    ap.add_argument("--out", default=str(ROOT / "data/processed/track_k/clips_v3"))
    args = ap.parse_args()
    fit_path = (Path(args.fit) if args.fit
                else sorted((ROOT / "results/k02_fit").glob("*/fit.json"))[-1])
    fit = json.loads(fit_path.read_text())
    model = args.model or fit["selection"]["selected"]
    rows = export_tree(fit, model, args.out)
    man = ROOT / "results/k02_downstream" / "schema_v3_manifest.csv"
    man.parent.mkdir(parents=True, exist_ok=True)
    man.write_text("flight,clip_id,condition,sha256\n"
                   + "\n".join(",".join(r) for r in rows) + "\n")
    print(f"{len(rows)} clips written to {args.out} with model {model}")
    print(f"manifest {man}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
