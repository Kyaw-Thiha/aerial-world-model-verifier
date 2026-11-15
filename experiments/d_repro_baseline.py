"""Task 0: reproduce the current median-residual statistic's headline numbers.

Primary mode per amendment 9: drag-free, lag-only (mount uptilt offset fixed at 0, lag
fitted within +/-150 ms), frozen confidence gate, all-flight threshold policy.  Expected:
threshold 14.733 deg, held-out acceptance 0.966 (n=58), tilt removal rejected 0.500 on 8
per-condition-testable pairs, mirrored roll 0.961, yaw control 0.038.  If these do not
reproduce, the gain test is measured against a different baseline and the run is void.

    PYTHONPATH=. pixi run python experiments/d_repro_baseline.py
"""
from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

from experiments.a2_modes import tii_records
from experiments.phase3a_verdict import evaluation

ROOT = Path(__file__).resolve().parents[1]


def git_hash() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                                       text=True).strip()
    except Exception:
        return "unknown"


def main() -> int:
    prereg = yaml.safe_load((ROOT / "configs/verdict_preregistration.yaml").read_text())
    gate = yaml.safe_load((ROOT / "configs/gate_frozen.yaml").read_text())
    records = tii_records(ROOT / "results/gate_a2/tii", 0.0, "lag_only", prereg, gate)
    res = evaluation(records, prereg, policy="all_flights")
    folds = [f["thresholds"]["pooled"]["threshold"] for f in res["fold_thresholds"]]
    out = {
        "run": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "git": git_hash(),
        "expected": {"threshold": 14.733, "fold_min": 13.50, "fold_max": 16.46,
                     "acceptance": 0.966, "n": 58, "tilt_removal": 0.500,
                     "mirror_roll": 0.961, "yaw_control": 0.038},
        "measured": {
            "threshold_pooled": res["pooled_thresholds"]["pooled"]["threshold"],
            "threshold_per_fold": folds,
            "threshold_fold_min": float(np.min(folds)),
            "threshold_fold_max": float(np.max(folds)),
            "acceptance": res["heldout_acceptance"],
            "acceptance_wilson95": res["heldout_acceptance_wilson95"],
            "n": res["heldout_n"],
            "families": {c: {"n": f.get("n"), "rejection_rate": f.get("rejection_rate"),
                             "auroc": f.get("auroc")}
                         for c, f in res["families"].items()},
        },
    }
    m = out["measured"]
    checks = {
        "threshold": abs(m["threshold_pooled"] - 14.733) < 0.01,
        "fold_min": abs(m["threshold_fold_min"] - 13.50) < 0.01,
        "fold_max": abs(m["threshold_fold_max"] - 16.46) < 0.01,
        "acceptance": abs(m["acceptance"] - 0.9655) < 0.01,
        "n": m["n"] == 58,
        "tilt_removal": abs(m["families"]["tilt_k0"]["rejection_rate"] - 0.5) < 0.01,
        "mirror_roll": abs(m["families"]["mirror_roll"]["rejection_rate"] - 0.961) < 0.01,
        "yaw_control": abs(m["families"]["yaw_control"]["rejection_rate"] - 0.038) < 0.01,
    }
    out["reproduced"] = checks
    out["all_reproduced"] = bool(all(checks.values()))
    run_dir = ROOT / "results/d_repro_baseline" / out["run"]
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "metrics.json").write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps({"reproduced": out["all_reproduced"], "checks": checks,
                      "threshold": m["threshold_pooled"],
                      "acceptance": m["acceptance"], "n": m["n"],
                      "families": {k: v["rejection_rate"] for k, v in m["families"].items()}},
                     indent=1))
    return 0 if out["all_reproduced"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
