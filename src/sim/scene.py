"""One procedural scene layout, consumed by every renderer adapter.

The point of sharing this is that the candidates are then compared on *rendering*, not on
scene content: same ground, same boxes, same trees, same sun. A renderer that cannot express
this layout is a fact about the renderer.

Coordinates: world metres, z up, origin at the ground under the flight's start. Extents are
chosen so a 640x360 camera at 8-20 m with a 90 deg field always has texture in frame:
buildings 4-30 m tall between 25 and 140 m out, a hill ring at 250-400 m for a horizon
silhouette, checker ground of 20 m squares.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

CHECKER_M = 20.0
GROUND_M = 800.0


@dataclass
class Box:
    x: float
    y: float
    w: float
    d: float
    h: float
    kind: str = "block"          # "block" | "tree"


def layout(seed: int = 7, n_blocks: int = 48, n_trees: int = 60,
           r_min: float = 25.0, r_max: float = 140.0) -> dict:
    """Deterministic scene description. Same seed -> same scene in every renderer."""
    rng = np.random.default_rng(seed)
    blocks = []
    for _ in range(n_blocks):
        ang = rng.uniform(0, 2 * np.pi)
        rad = rng.uniform(r_min, r_max)
        blocks.append(Box(float(rad * np.cos(ang)), float(rad * np.sin(ang)),
                          float(rng.uniform(4.0, 12.0)), float(rng.uniform(4.0, 12.0)),
                          float(rng.uniform(4.0, 30.0)), "block"))
    trees = []
    for _ in range(n_trees):
        ang = rng.uniform(0, 2 * np.pi)
        rad = rng.uniform(r_min * 0.7, r_max * 1.15)
        trees.append(Box(float(rad * np.cos(ang)), float(rad * np.sin(ang)),
                         float(rng.uniform(0.8, 1.6)), float(rng.uniform(0.8, 1.6)),
                         float(rng.uniform(4.0, 12.0)), "tree"))
    hills = []
    for i in range(18):
        ang = 2 * np.pi * i / 18.0 + rng.uniform(-0.06, 0.06)
        rad = rng.uniform(250.0, 400.0)
        hills.append(Box(float(rad * np.cos(ang)), float(rad * np.sin(ang)),
                         float(rng.uniform(120.0, 260.0)), float(rng.uniform(120.0, 260.0)),
                         float(rng.uniform(25.0, 70.0)), "hill"))
    return {"seed": int(seed), "checker_m": CHECKER_M, "ground_m": GROUND_M,
            "sun_energy": 1.6, "sun_elevation_deg": 45.0, "sun_azimuth_deg": 35.0,
            "sky_strength": 0.5,
            "objects": [b.__dict__ for b in blocks + trees + hills]}
