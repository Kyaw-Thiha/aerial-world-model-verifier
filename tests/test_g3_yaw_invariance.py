"""The invariance that makes a depicted-side yaw control impossible (Track G3, G06).

The schema's lean vector is the world-up vector expressed in the body frame:
``u_b = R_bw^T e3``.  A yaw control rotates the *world* frame about its vertical,
``R_bw -> Rz(alpha) R_bw``, and ``Rz(alpha) e3 = e3``, so ``u_b`` is unchanged.  The image
changes (the camera looks at a different part of the scene) but the depicted lean does not,
which is why Track G3 tests the reader's yaw behaviour with a rendered auxiliary slice and
not by composing the edit on the reader's output.  See ``configs/g06_segments_prereg.yaml``.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]


def _lean(up_b: np.ndarray) -> np.ndarray:
    """The schema lean from world up in the body frame (src.depicted.clips.lean_from_up_body)."""
    up_b = np.asarray(up_b, dtype=np.float64).reshape(-1, 3)
    n = np.linalg.norm(up_b, axis=1)
    u = up_b / n[:, None]
    theta = -np.arcsin(np.clip(u[:, 0], -1.0, 1.0))
    phi = np.arctan2(u[:, 1], u[:, 2])
    return np.column_stack([np.cos(phi) * np.sin(theta), -np.sin(phi)])


def _rz(alpha: float) -> np.ndarray:
    c, s = np.cos(alpha), np.sin(alpha)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _random_attitude(rng: np.random.Generator) -> np.ndarray:
    z = rng.normal(size=3)
    z[2] = abs(z[2]) + 0.2
    z /= np.linalg.norm(z)
    a = np.array([0.0, 0.0, 1.0]) if abs(z[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    x = np.cross(a, z)
    x /= np.linalg.norm(x)
    return np.column_stack([x, np.cross(z, x), z])


def test_world_yaw_leaves_the_lean_unchanged() -> None:
    rng = np.random.default_rng(1)
    e3 = np.array([0.0, 0.0, 1.0])
    worst = 0.0
    for _ in range(500):
        R = _random_attitude(rng)
        alpha = rng.uniform(-np.pi, np.pi)
        l0 = _lean(R.T @ e3)
        l1 = _lean((_rz(alpha) @ R).T @ e3)
        worst = max(worst, float(np.max(np.abs(l0 - l1))))
    assert worst < 1e-12, worst


def test_yaw_slice_tilts_are_nonzero() -> None:
    cfg = json.loads((REPO / "configs" / "g06_yaw_slice.json").read_text())
    assert len(cfg["clips"]) == 12
    cells = {(c["family"], c["uptilt_deg"]) for c in cfg["clips"]}
    assert len(cells) == 12
    for c in cfg["clips"]:
        # alpha = min(tilt_req, 60); a hover clip would give alpha = 0 and no yaw at all
        assert c["peak_tilt_req_deg"] > 1.0, c
        assert 0.0 <= c["uptilt_deg"] <= 45.0
