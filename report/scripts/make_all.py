"""Rebuild every report asset from the stored sources.

Run: python3 report/scripts/make_all.py
(no GPU, no network, no experiments)
"""
from __future__ import annotations

import importlib
import sys
import traceback

SCRIPTS = [
    "e1_physics_floor",
    "e2_depicted_tilt",
    "e2_extra",
    "e3_verdict_real",
    "e3_extra",
    "e4_anwm_audit",
    "e5_anwm_verdict",
    "e5_extra",
    "e6_pose_injection",
    "e7_path_from_pixels",
    "e7_extra",
    "e8_gain_coherence",
    "e9_uzh_transfer",
    "e10_window_rule",
    "methodology",
    "corrections",
]


def main():
    failures = []
    for name in SCRIPTS:
        try:
            mod = importlib.import_module(name)
            out = mod.build()
            print(f"OK   {name}: {out}")
        except Exception as exc:  # noqa: BLE001
            failures.append((name, exc))
            print(f"FAIL {name}: {exc}")
            traceback.print_exc()
    print(f"\n{len(SCRIPTS) - len(failures)}/{len(SCRIPTS)} asset groups built")
    if failures:
        sys.exit(1)


if __name__ == "__main__":
    main()
