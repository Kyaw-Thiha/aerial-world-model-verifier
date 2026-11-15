"""The committed path audit must satisfy the population invariant in every cell."""
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "results/anwm_path_audit/audit.json"


@pytest.mark.skipif(not AUDIT.exists(), reason="path audit artifact not generated")
def test_every_audit_cell_is_one_consistent_population():
    d = json.loads(AUDIT.read_text())
    n = 0
    for regime in ("A_native", "B_rescaled", "C_designed"):
        for dt, cells in d[regime]["cells"].items():
            for key, cell in cells.items():
                n += 1
                assert cell["n"] > 0, (regime, dt, key)
                if cell["median"] is not None and cell["median"] > 10.0:
                    assert cell["frac_gt"] >= 0.5, (regime, dt, key, cell)
    assert n >= 36
