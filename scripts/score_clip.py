#!/usr/bin/env python
"""Score one clip with the gain test.

Usage
-----
    PYTHONPATH=. pixi run python scripts/score_clip.py PATH [--config FILE] [--json]

``PATH`` is a clip directory (containing ``verdict_clip.npz`` and ``meta.json``) or the
``.npz`` file itself.  With no ``--config`` the script uses the pre-registered defaults
and the frozen confidence gate, and reports no decision (bands are not calibrated).

The graded output (acceptance spec T5) always contains: required tilt, depicted tilt,
the scalar gain with a 95 % interval, both axis gains with intervals, the fitted drag,
the intercept, R-squared, the selected lag, whether the clip is testable, the abstention
reasons, and the flag.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.verdict import (  # noqa: E402
    Bands, FitOptions, JointBand, ScalarBand, assess, coherence, fit_gain, flag,
    gate_kept_from_conf, load_clip,
)


def load_config(path: Path | None) -> dict:
    if path is None or not path.exists():
        return {}
    return yaml.safe_load(path.read_text())


def options_from_config(cfg: dict) -> FitOptions:
    opts = cfg.get("options", cfg.get("statistic", {}))
    allowed = {f for f in FitOptions.__dataclass_fields__}
    kwargs = {k: v for k, v in opts.items() if k in allowed}
    if "lag_bounds_ms" in kwargs:
        kwargs["lag_bounds_ms"] = tuple(kwargs["lag_bounds_ms"])
    return FitOptions(**kwargs)


def bands_from_config(cfg: dict) -> Bands:
    b = cfg.get("bands") or {}
    out = Bands()
    if "scalar" in b:
        s = b["scalar"]
        out.scalar = ScalarBand(float(s["lo"]), float(s["hi"]), float(s.get("alpha", 0.05)),
                                int(s.get("n", 0)))
    if "axis" in b:
        a = b["axis"]
        keys = tuple(a.get("keys", ("gx", "gy")))
        out.axis = JointBand(keys, {k: float(v) for k, v in a["center"].items()},
                             {k: float(v) for k, v in a["scale"].items()},
                             float(a["threshold"]), float(a.get("alpha", 0.05)),
                             int(a.get("n", 0)))
    if "joint" in b:
        j = b["joint"]
        keys = tuple(j.get("keys", ("G1", "gx", "gy")))
        out.joint = JointBand(keys, {k: float(v) for k, v in j["center"].items()},
                              {k: float(v) for k, v in j["scale"].items()},
                              float(j["threshold"]), float(j.get("alpha", 0.05)),
                              int(j.get("n", 0)))
    return out


def gate_config_from(cfg: dict) -> dict | None:
    gate = cfg.get("gate", {})
    if gate is False or gate.get("enabled") is False:
        return None
    if isinstance(gate, dict) and "tau" in gate:
        return gate
    path = Path(gate.get("config", "configs/gate_frozen.yaml"))
    if not path.is_absolute():
        path = ROOT / path
    return yaml.safe_load(path.read_text()) if path.exists() else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("clip")
    ap.add_argument("--config", default=None)
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--no-gate", action="store_true")
    ap.add_argument("--coherence-config", default="configs/coherence_frozen.yaml")
    args = ap.parse_args()

    cfg = load_config(Path(args.config) if args.config else None)
    clip = load_clip(args.clip)
    opts = options_from_config(cfg)
    result = fit_gain(clip, opts)

    gate_cfg = None if args.no_gate else gate_config_from(cfg)
    gate_kept = gate_kept_from_conf(clip.depicted_conf, gate_cfg) if gate_cfg else None
    ab = assess(result, cfg.get("abstention"), gate_kept=gate_kept)

    bands = bands_from_config(cfg)
    rule = cfg.get("rule", "R3_scalar_or_axis")
    flagged = None
    if ab.testable and (bands.scalar or bands.axis or bands.joint):
        flagged = flag(result, rule, bands)

    coh_out = None
    if clip.lean_req is not None:
        coh = coherence(clip, lag_ms=result.lag_ms)
        coh_cfg = load_config(Path(args.coherence_config)) if args.coherence_config else {}
        tau_c = (coh_cfg.get("thresholds") or {}).get("tau_correlation")
        tau_s = (coh_cfg.get("thresholds") or {}).get("tau_sign")
        coh_out = {
            "r_x": coh.r_x, "r_y": coh.r_y, "s_x": coh.s_x, "s_y": coh.s_y,
            "r_min": coh.r_min, "s_min": coh.s_min,
            "correlation_score": coh.correlation_score,
            "sign_score": coh.sign_score,
            "tau_correlation": tau_c, "tau_sign": tau_s,
            "flagged_correlation": (None if tau_c is None or not ab.testable
                                    else bool(coh.correlation_score > tau_c)),
            "flagged_sign": (None if tau_s is None or not ab.testable
                             else bool(coh.sign_score > tau_s)),
        }

    out = {
        "clip": str(args.clip),
        "clip_id": clip.clip_id,
        "flight": clip.flight,
        "condition": clip.condition,
        "required_tilt_deg": result.required_tilt_deg,
        "depicted_tilt_deg": result.depicted_tilt_deg,
        "gain_g1": result.gain,
        "gain_g1_ci95": list(result.gain_ci),
        "gain_g1_half_width": result.gain_half_width,
        "axis_gains": {"gx": result.axis_gains[0], "gy": result.axis_gains[1]},
        "axis_ci95": {"gx": list(result.axis_ci[0]), "gy": list(result.axis_ci[1])},
        "drag_coefficient": result.drag,
        "drag_ci95": list(result.drag_ci),
        "intercept": {"x": result.intercept[0], "y": result.intercept[1]},
        "r_squared": result.r2,
        "lag_ms": result.lag_ms,
        "n_frames": result.n_frames,
        "n_used": result.n_used,
        "excitation_deg": result.excitation_deg,
        "condition_number": result.cond_number,
        "testable": ab.testable,
        "abstention_reasons": ab.reasons,
        "gate_kept": gate_kept,
        "rule": rule if flagged is not None else None,
        "flagged": flagged,
        "coherence": coh_out,
        "note": ("A pass means no violation of the tested size on a clip with enough "
                 "excitation; it is not proof of physical consistency."),
    }
    if args.json:
        print(json.dumps(out, indent=1, default=float))
    else:
        for k, v in out.items():
            print(f"{k}: {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
