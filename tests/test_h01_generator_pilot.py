"""Track H unit checks: pre-registered seed population and the LingBot pose construction."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from experiments.h01_seed_specs import EXTRA_START1, SEEDS, select_paths
from src.generation.attitude import body_z_from_poses, roll_pitch_from_thrust
from src.perception.frames import camera_to_body_pitched

ROOT = Path(__file__).resolve().parents[1]


def test_seed_population_matches_preregistration():
    specs = json.loads((ROOT / "results/anwm_clips/specs.json").read_text())
    paths = select_paths(specs)
    ids = [p["id"] for p in paths]
    assert len(ids) == 20 and len(set(ids)) == 20
    assert all(cid in ids for cid in EXTRA_START1)
    # one start-0 clip per unique design cell
    cells = set()
    for p in paths[:15]:
        assert p["id"].endswith("_0") or p["id"].endswith("course_0")
        cells.add((p["meta"]["maneuver"], p["designed_tilt_deg"], p["meta"]["policy"]))
    assert len(cells) == 15
    assert SEEDS == (1, 2, 3, 4)


def test_pose_construction_recovers_the_injected_thrust_axis():
    rng = np.random.default_rng(1)
    n = 24
    pos = np.cumsum(rng.normal(scale=3.0, size=(n, 3)), axis=0)
    pos = pos - pos[0]
    yaw = np.linspace(-0.4, 0.9, n)
    zB = np.tile(np.array([0.0, 0.0, 1.0]), (n, 1))
    zB[:, 0] = np.sin(np.deg2rad(12.0)) * np.cos(np.linspace(0, 2 * np.pi, n))
    zB[:, 1] = np.sin(np.deg2rad(12.0)) * np.sin(np.linspace(0, 2 * np.pi, n))
    zB = zB / np.linalg.norm(zB, axis=1, keepdims=True)
    phi, theta = roll_pitch_from_thrust(zB, yaw)
    R_bc = camera_to_body_pitched(0.0)
    from src.generation.attitude import body_rotation

    R_wb = body_rotation(yaw, theta, phi)
    poses = np.tile(np.eye(4), (n, 1, 1))
    poses[:, :3, :3] = R_wb @ R_bc[None]
    poses[:, :3, 3] = pos
    recovered = body_z_from_poses(poses, R_bc)
    assert np.abs(recovered - zB).max() < 1e-12


def test_lingbot_inputs_are_consistent_when_present():
    man = ROOT / "data/h01_lingbot_inputs/manifest.json"
    if not man.exists():
        return
    doc = json.loads(man.read_text())
    assert len(doc["grid"]) == 75
    g = doc["grid"][0]
    poses = np.load(ROOT / g["dir"] / "poses.npy")
    thrust = np.load(ROOT / g["dir"] / "thrust.npy")
    K = np.load(ROOT / g["dir"] / "intrinsics.npy")
    assert poses.shape == (49, 4, 4) and thrust.shape == (49, 3) and K.shape == (49, 4)
    assert abs(np.linalg.det(poses[:, :3, :3]) - 1).max() < 1e-6
    assert np.abs(thrust).max() <= 1.0 + 1e-9
