"""Track G3: assemble per-clip arrays for the window statistic.

Every clip is (t, depicted lean, required lean, truth mask).  The depicted lean is the reader's
GeoCalib output at stride 1 (``results/g06/scores``); the required lean and the truth-tilt are
the simulator's exact values stored beside the frames.  The path is never estimated.

The truth mask comes from the manifest's exact interval (``interval_start_s`` /
``interval_end_s``); a full-clip defect has the whole clip inside, a feasible clip has nothing
inside.  No mask is inferred from a statistic.
"""
from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

FPS = 30.0


@dataclass
class G3Clip:
    clip_id: str
    t: np.ndarray
    lean_dep: np.ndarray
    lean_req: np.ndarray
    tilt_req_deg: np.ndarray
    truth: np.ndarray
    meta: dict

    def __len__(self) -> int:
        return len(self.t)

    @property
    def yaw(self) -> np.ndarray:
        """The frozen residual route takes a yaw; with the same yaw on both axes the angle is
        invariant to its value, so zeros are exact rather than an approximation."""
        return np.zeros(len(self.t))


def read_manifest(path: str | Path) -> dict[str, dict]:
    rows = list(csv.DictReader(open(path)))
    return {r["clip_id"]: r for r in rows}


def truth_mask(row: dict, t: np.ndarray) -> np.ndarray:
    frac = float(row.get("interval_fraction") or 0.0)
    if not frac:
        return np.zeros(len(t), dtype=bool)
    a = float(row["interval_start_s"])
    b = float(row["interval_end_s"])
    return (t >= a) & (t < b)


def load_clip(clip_id: str, scores_dir: str | Path, manifest: dict[str, dict],
              video_dir: str | Path | None = None) -> G3Clip:
    """One clip.  ``scores_dir`` holds ``<clip_id>_score.json`` from remote/g3_score.py."""
    d = json.loads((Path(scores_dir) / f"{clip_id}_score.json").read_text())
    rows = d["rows"]
    frame = np.array([r["frame"] for r in rows], dtype=int)
    t = frame / FPS
    lean_dep = np.column_stack([[r["lean_est_x"] for r in rows],
                                [r["lean_est_y"] for r in rows]]).astype(float)
    lean_req = np.column_stack([[r["lean_req_x"] for r in rows],
                                [r["lean_req_y"] for r in rows]]).astype(float)
    tilt_req = np.array([np.degrees(np.arcsin(np.clip(np.linalg.norm(v), 0, 1)))
                         for v in lean_req])
    row = manifest.get(clip_id, {"clip_id": clip_id, "label": "feasible",
                                 "defect_family": "feasible", "interval_fraction": "0.0"})
    return G3Clip(clip_id=clip_id, t=t, lean_dep=lean_dep, lean_req=lean_req,
                  tilt_req_deg=tilt_req, truth=truth_mask(row, t), meta=row)


def load_split(scores_dir: str | Path, manifest: dict[str, dict],
               only: list[str] | None = None) -> dict[str, G3Clip]:
    ids = sorted(p.name.replace("_score.json", "")
                 for p in Path(scores_dir).glob("*_score.json"))
    if only is not None:
        ids = [c for c in ids if c in set(only)]
    return {c: load_clip(c, scores_dir, manifest) for c in ids}
