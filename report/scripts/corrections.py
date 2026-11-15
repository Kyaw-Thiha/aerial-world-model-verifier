"""Write report/data/corrections.csv from docs/decisions.md.

One row per withdrawn or revised claim: date, source tag, claim, what replaced it,
source entry (file and line). Text is transcribed from the decisions log; no numeric
result is taken from the prose when a stored result exists elsewhere.
"""
from __future__ import annotations

import common as C

DECISIONS = "docs/decisions.md"
ASSET = "corrections"

ROWS = [
    ("2026-11-08", "handoff/A",
     "ANWM has no public repository, weights or data.",
     "Repository is ANWM.code; weights and data exist.",
     "docs/decisions.md L67"),
    ("2026-11-08", "A",
     "Piloted flights do not reconcile with the released mount.",
     "They reconcile using elevation + body tilt; analytic mount used, 6-7 deg residual reported.",
     "docs/decisions.md L123"),
    ("2026-11-09", "A2",
     "Amendment 1: thresholds fitted on the five non-trackRATM flights.",
     "Withdrawn by amendment 2: pixel-only confidence gate, no flight excluded by identity.",
     "docs/decisions.md L219"),
    ("2026-11-09", "A2",
     "Report 07 first version attributed 0.8210/0.7229 to near-tied nuisance fits on the same rows.",
     "Those are the re-extracted rows; on Gate 3 rows the reproduction is exact (59/83, 0/600 differ).",
     "docs/decisions.md L234"),
    ("2026-11-10", "A2",
     "Natural cubic spline through waypoints is the least tilt any flight could need.",
     "False on plateau paths (spline exceeds per-step on 345/492 rows); amendment 5 replaces the statistic.",
     "docs/decisions.md L241"),
    ("2026-11-10", "A2",
     "ANWM residuals computed without the verdict's nuisances.",
     "Amendment 6 makes full parity primary; drag at k_d 0.295 needed parity (7.8 deg unfitted).",
     "docs/decisions.md L244"),
    ("2026-11-10", "A2",
     "Primary ANWM mode was k_d 0.295 with both nuisances.",
     "Amendment 8: primary is drag-free + lag-only; full parity is the conservative bound.",
     "docs/decisions.md L250"),
    ("2026-11-10", "A2",
     "ANWM infeasibility counts used the 6.46 deg threshold while the primary mode was drag-free lag-only.",
     "Amendment 9 recalibrates per mode: 14.73 / 12.95 / 8.96 / 6.46 deg; designed-10 flips to 0/6 at the primary.",
     "docs/decisions.md L259"),
    ("2026-11-10", "A2",
     "DPVO path '41 m against 479 m'.",
     "Withdrawn: different units; replay gives 960 camera-unit path, ATE 5.97 m. Direction, not magnitude, fails.",
     "docs/decisions.md L256"),
    ("2026-11-10", "A2",
     "'UZH-FPV ships no calibration'.",
     "Withdrawn: release ships a Kalibr camchain.",
     "docs/decisions.md L257"),
    ("2026-11-14", "D",
     "Label-free rule picks about 0.10 s on UZH and the 500 Hz windows are 35-62 samples.",
     "That was the 15 Hz clip-rate application; at 500 Hz the rule picks 0.336-1.420 s (169-711 samples).",
     "docs/decisions.md L289"),
    ("2026-11-14", "D/E",
     "Prototype v0.1 states the UZH rule windows are 0.07-0.12 s at the truth rate.",
     "Superseded by E05: 0.34-1.42 s at 500 Hz; 0.07-0.11 s only at the 15 Hz clip rate, which fails the 8-sample floor.",
     "docs/prototype_v0.1.md; reports/E05_window_rule.md §2"),
]


def build():
    C.write_csv(ASSET, [list(r) for r in ROWS],
                ["date", "source_tag", "claim", "replacement", "source_entry"])
    C.provenance(ASSET, [("d", DECISIONS)], notes=(
        "Curated from the decisions log; each row names the log entry. The prototype "
        "row is also sourced to E05 because the correction lives in that report."),
        derived=True)
    return dict(n=len(ROWS))


if __name__ == "__main__":
    print(build())
