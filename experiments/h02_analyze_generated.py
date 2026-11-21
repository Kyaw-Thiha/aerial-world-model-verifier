"""H02: depicted vs injected attitude for SolarWM and full-precision LingBot clips.

Reads the reader rows written by ``remote/h02_gate.py`` and the per-sample poses built by
``experiments/h02_solarwm_inputs.py`` (or H01's LingBot inputs). The statistic is the same as
H01: frame-level depicted tilt, clip median; depicted and injected lean vectors in the world
frame using the commanded yaw.

    pixi run python experiments/h02_analyze_generated.py --depicted results/h02_solarwm_depicted/depicted.jsonl \
        --poses data/h02_solarwm_inputs --out results/h02_solarwm
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.a2_anwm_score import frame_arrays
from src.generation.attitude import body_z_from_poses
from src.perception.frames import camera_to_body_pitched
from src.physics.flatness import tilt_deg
from src.score.verdict import depicted_axis_world

ROOT = Path(__file__).resolve().parents[1]


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(ln) for ln in path.read_text().splitlines() if ln.strip()]


def yaw_from_poses(poses: np.ndarray, R_bc: np.ndarray) -> np.ndarray:
    R_wb = poses[:, :3, :3] @ R_bc.T
    return np.arctan2(R_wb[:, 1, 0], R_wb[:, 0, 0])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--depicted", required=True)
    ap.add_argument("--poses", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--label", default="")
    args = ap.parse_args()

    rows = load_jsonl(ROOT / args.depicted)
    by_clip: dict[str, list[dict]] = {}
    for r in rows:
        by_clip.setdefault(r["clip_id"], []).append(r)
    R_cb = camera_to_body_pitched(0.0)
    per_clip, per_frame = [], []
    for cid, rs in sorted(by_clip.items()):
        rs = sorted(rs, key=lambda r: int(Path(r["frame"]).stem))
        up = frame_arrays(rs)["up"]
        pdir = ROOT / args.poses / cid
        if not pdir.exists():
            per_clip.append({"clip_id": cid, "n_frames": len(up),
                             "depicted_tilt_median_deg": float(np.nanmedian(
                                 tilt_deg(depicted_axis_world(up, R_cb, np.zeros(len(up)))))),
                             "note": "no pose directory"})
            continue
        poses = np.load(pdir / "poses.npy")
        n = min(len(up), len(poses))
        yaw = yaw_from_poses(poses[:n], R_cb)
        dep = tilt_deg(depicted_axis_world(up[:n], R_cb, yaw))
        inj = tilt_deg(body_z_from_poses(poses[:n], R_cb))
        z_dep = depicted_axis_world(up[:n], R_cb, yaw)
        z_inj = body_z_from_poses(poses[:n], R_cb)
        lean, lean_i = z_dep[:, :2], z_inj[:, :2]
        cos = (lean * lean_i).sum(1) / np.maximum(
            np.linalg.norm(lean, axis=1) * np.linalg.norm(lean_i, axis=1), 1e-9)
        meta = {}
        mp = pdir / "meta.json"
        if mp.exists():
            meta = json.loads(mp.read_text())
        per_clip.append({
            "clip_id": cid, **{k: meta.get(k) for k in
                               ("kind", "path", "arm", "traj", "frame", "source_clip",
                                "seed", "peak_deg", "profile")},
            "n_frames": int(n),
            "depicted_tilt_median_deg": float(np.nanmedian(dep)),
            "depicted_tilt_p90_deg": float(np.nanpercentile(dep, 90)),
            "injected_tilt_median_deg": float(np.nanmedian(inj)),
            "injected_tilt_max_deg": float(np.nanmax(inj)),
            "depicted_lean_x": float(lean[:, 0].mean()),
            "depicted_lean_y": float(lean[:, 1].mean()),
            "injected_lean_x": float(lean_i[:, 0].mean()),
            "injected_lean_y": float(lean_i[:, 1].mean()),
            "lean_cos_median": float(np.nanmedian(cos)),
            "depicted_first10_deg": float(np.nanmedian(dep[:10])),
            "depicted_last10_deg": float(np.nanmedian(dep[-10:])),
            "injected_first10_deg": float(np.nanmedian(inj[:10])),
            "injected_last10_deg": float(np.nanmedian(inj[-10:])),
            "pearson_tilt": (float(np.corrcoef(dep, inj)[0, 1])
                             if np.std(dep) > 0 and np.std(inj) > 0 else None),
        })
        for i in range(n):
            per_frame.append({"clip_id": cid, "i": i,
                              "depicted_tilt_deg": float(dep[i]),
                              "injected_tilt_deg": float(inj[i]),
                              "depicted_zx": float(z_dep[i, 0]),
                              "depicted_zy": float(z_dep[i, 1]),
                              "injected_zx": float(z_inj[i, 0]),
                              "injected_zy": float(z_inj[i, 1])})
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(per_clip)
    df.to_csv(out / "clips.csv", index=False)
    pd.DataFrame(per_frame).to_csv(out / "frames.csv", index=False)
    summary = {"label": args.label, "depicted": args.depicted, "n_clips": len(df)}
    if "arm" in df and df["arm"].notna().any():
        agg = df.groupby(["path", "arm"], dropna=False).agg(
            n=("clip_id", "size"),
            depicted_median_deg=("depicted_tilt_median_deg", "median"),
            injected_median_deg=("injected_tilt_median_deg", "median"),
            first10=("depicted_first10_deg", "median"),
            last10=("depicted_last10_deg", "median"),
            lean_cos=("lean_cos_median", "median")).reset_index()
        agg.to_csv(out / "by_arm.csv", index=False)
        summary["by_arm"] = agg.to_dict(orient="records")
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1)[:4000])
    print(df.to_string(index=False, max_colwidth=40)[:6000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
