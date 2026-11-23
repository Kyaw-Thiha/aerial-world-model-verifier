"""H04 scoring sanity: a perfect generator is accepted, a level one is flagged.

Synthetic clips built from the commanded paths only — no GPU, no model, no stored rows.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from src.perception.frames import camera_to_body_pitched
from src.verdict import Clip, FitOptions, assess, coherence, fit_gain, gate_kept_from_conf
from src.verdict.lean import roll_pitch_to_lean

ROOT = Path(__file__).resolve().parents[1]
GAIN = yaml.safe_load((ROOT / "configs/gain_frozen.yaml").read_text())
COH = yaml.safe_load((ROOT / "configs/coherence_frozen.yaml").read_text())
GATE = yaml.safe_load((ROOT / "configs/gate_frozen.yaml").read_text())


def _paths():
    from experiments.h04_paths import CLIP_S, DT, positions, derivatives, attitude
    t = np.arange(0.0, CLIP_S, DT)
    out = {}
    for name, peak in (("s_10", 10.0), ("arc_20", 20.0)):
        pos = positions(name.split("_")[0], peak, t)
        d = derivatives(pos)
        a = attitude(pos, d)
        idx = np.round(np.linspace(0, len(t) - 1, 81)).astype(int)
        from experiments.h04_paths import f95_and_window, required_side
        win = f95_and_window(pos, DT)
        rs = required_side(pos, win["clamped_s"], DT)
        out[name] = {"t": t[idx], "pos": pos[idx], "yaw": a["yaw"][idx],
                     "lean": rs["lean_req"][idx], "vel": rs["vel_req"][idx],
                     "tilt": rs["tilt_req_deg"][idx]}
    return out


def _score(lean, name, paths):
    p = paths[name]
    clip = Clip(t=p["t"], pos=p["pos"], yaw=p["yaw"], depicted_lean=lean,
                depicted_conf=np.ones(len(p["t"])), meta={"clip_id": name},
                lean_req=p["lean"], vel_req=p["vel"], tilt_req_deg=p["tilt"])
    opts = FitOptions(drag=GAIN["options"]["drag"], lag_search=True,
                      lag_bounds_ms=tuple(GAIN["options"]["lag_bounds_ms"]),
                      lag_step_ms=GAIN["options"]["lag_step_ms"],
                      bootstrap=True, n_boot=100, seed=1)
    res = fit_gain(clip, opts)
    ab = assess(res, GAIN.get("abstention"),
                gate_kept=gate_kept_from_conf(clip.depicted_conf, GATE))
    coh = coherence(clip, lag_ms=res.lag_ms)
    return res, ab, coh


def test_perfect_generator_is_testable_and_coherent():
    paths = _paths()
    rng = np.random.default_rng(0)
    for name in paths:
        lean = paths[name]["lean"] + 0.02 * rng.standard_normal(paths[name]["lean"].shape)
        res, ab, coh = _score(lean, name, paths)
        assert ab.testable, (name, ab.reasons)
        assert res.cond_number < 1000, (name, res.cond_number)
        assert coh.correlation_score <= COH["thresholds"]["tau_correlation"], (name, coh)
        assert coh.sign_score <= COH["thresholds"]["tau_sign"], (name, coh)


def test_level_camera_is_flagged():
    paths = _paths()
    for name in paths:
        lean = np.zeros_like(paths[name]["lean"])
        res, ab, coh = _score(lean, name, paths)
        if ab.testable:
            assert coh.correlation_score > COH["thresholds"]["tau_correlation"] or \
                coh.sign_score > COH["thresholds"]["tau_sign"], (name, coh)
