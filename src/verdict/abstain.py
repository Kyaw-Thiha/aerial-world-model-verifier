"""Abstention rules for the gain statistic (pre-registered in ``configs/gain_prereg.yaml``).

A clip is *not testable* when the regression cannot answer the question, never
"infeasible".  Every rule returns a short reason string so the graded output can say why.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .fit import GainResult


@dataclass
class Abstention:
    testable: bool
    reasons: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {"testable": self.testable, "abstention_reasons": list(self.reasons)}


DEFAULT_CONFIG = {
    "min_frames": 24,
    "usable_fraction_min": 0.50,
    "signal_frames": {"required_tilt_deg_min": 10.0, "min_frames": 12,
                      "fraction_min": 0.20},
    "excitation": {"floor_deg": 2.0},
    "conditioning": {"max": 1000.0},
    "ci_half_width": {"max": 0.35},
}


def assess(result: GainResult, config: dict | None = None,
           gate_kept: bool | None = None) -> Abstention:
    """Apply the pre-registered abstention rules, in the registered order."""
    cfg = {**DEFAULT_CONFIG, **(config or {})}
    reasons: list[str] = []

    if result.n_frames < cfg["min_frames"]:
        reasons.append(f"min_frames {result.n_frames} < {cfg['min_frames']}")

    if result.finite_fraction < cfg["usable_fraction_min"]:
        reasons.append(f"usable_fraction {result.finite_fraction:.2f} < "
                       f"{cfg['usable_fraction_min']:.2f}")

    sig = cfg["signal_frames"]
    frac_signal = result.n_signal_frames / max(result.n_frames, 1)
    if result.n_signal_frames < sig["min_frames"]:
        reasons.append(f"signal_frames {result.n_signal_frames} < {sig['min_frames']}")
    if frac_signal < sig["fraction_min"]:
        reasons.append(f"signal_fraction {frac_signal:.2f} < {sig['fraction_min']:.2f}")

    floor = cfg["excitation"]["floor_deg"]
    if not np.isfinite(result.excitation_deg) or result.excitation_deg < floor:
        reasons.append(f"excitation {result.excitation_deg:.2f} deg < {floor:g}")

    if not np.isfinite(result.cond_number) or result.cond_number > cfg["conditioning"]["max"]:
        reasons.append(f"condition_number {result.cond_number:.0f} > "
                       f"{cfg['conditioning']['max']:.0f}")

    if result.bootstrap_ok:
        hw = result.gain_half_width
        if not np.isfinite(hw):
            reasons.append("gain_ci unavailable")
        elif hw > cfg["ci_half_width"]["max"]:
            reasons.append(f"gain_ci_half_width {hw:.2f} > {cfg['ci_half_width']['max']:g}")
    else:
        reasons.append("bootstrap unavailable")

    if gate_kept is False:
        reasons.append("confidence_gate")

    return Abstention(testable=not reasons, reasons=reasons)


def gate_kept_from_conf(depicted_conf: np.ndarray, gate_config: dict) -> bool:
    """Apply the frozen S3 signal to one clip's per-frame confidence.

    The frozen gate drops a clip when ``signal > tau`` where the S3 signal is the negative
    median ``latitude_confidence``.  ``depicted_conf`` in the schema is that
    ``latitude_confidence`` itself, so this is a sign change and a comparison, with the
    same ``min_keep`` rule (too few usable frames means the signal is undefined).
    """
    if gate_config is None:
        return True
    conf = np.asarray(depicted_conf, dtype=np.float64)
    ok = np.isfinite(conf)
    if ok.mean() < float(gate_config.get("min_keep", 0.6)) or ok.sum() == 0:
        return False
    signal = -float(np.median(conf[ok]))
    if gate_config.get("direction", "drop when signal > tau") != "drop when signal > tau":
        raise ValueError("unsupported gate direction")
    return bool(signal <= float(gate_config["tau"]))


def validation_status(meta: dict) -> tuple[str, bool]:
    """Whether the statistic is validated for this flight type.

    The rule is governance, not a threshold: piloted, non-track flights are validated;
    autonomous and track flights are **not validated** because the calibration evidence
    is a handful of clips (4 autonomous, 3 track on TII).  A not-validated clip still gets
    a graded result; it must not be used as evidence for or against a flight.
    """
    ft = str(meta.get("flight_type", "")).strip().lower()
    if not ft:
        track = bool(meta.get("trackratm")) or "trackratm" in str(
            meta.get("track", "")).lower()
        control = str(meta.get("control", "")).strip().lower()
        if track:
            ft = "track"
        elif control in ("piloted", "autonomous"):
            ft = control
    if ft == "piloted":
        return "validated", True
    return "not_validated", False
