"""H03: recompute the H01/H02/E5 headline numbers with the independent estimator.

Runs after ``remote/h03_recompute.py``.  Two jobs:

1. Validate the Perspective Fields 3-D up reconstruction on the sample's controls (truth
   known), and compare it with the reader's own rows on the same frames.
2. Recompute the affected headline numbers on the full clips: the LingBot roll probes (H01
   NF4, H02 bf16), the SolarWM probes and the 75-clip designed grid (H02), and the ANWM seed
   clips (E5).  The reader rows are read read-only from the H01/H02 worktrees.

    PYTHONPATH=. pixi run -e gpu python experiments/h03_recompute_compare.py
"""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
H1 = Path("/home/kevin/Documents/Projects/awmv-track-h")
H2 = Path("/home/kevin/Documents/Projects/awmv-h2")


def wrap90(x):
    return (np.asarray(x, float) + 90.0) % 180.0 - 90.0


def roll_of(up):
    up = np.asarray(up, float)
    return np.degrees(np.arctan2(up[..., 0], -up[..., 1]))


def tilt_of(up, mount_deg: float = 0.0):
    """Depicted tilt: angle between the body z axis and world up.

    ``up`` is the world-up direction in *camera* coordinates, so it is rotated into the body
    frame through the recorded mount before the z-component is taken (the reader's rows are
    already in that frame because ``reader_tilts`` applies ``R_cb``)."""
    from src.perception.frames import camera_to_body_pitched
    R_cb = camera_to_body_pitched(np.deg2rad(mount_deg))
    up = np.asarray(up, float) @ R_cb.T
    return np.degrees(np.arccos(np.clip(up[..., 2], -1.0, 1.0)))


def load_pf() -> dict:
    """(group, batch, clip_id) -> {frame_name: row}"""
    out = defaultdict(dict)
    p = ROOT / "results/h03_recompute/pf_rows.jsonl"
    for line in p.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            out[(r["group"], r["batch"], r["clip_id"])][r["frame"]] = r
    return out


def load_reader(path: Path) -> dict:
    """clip_id -> {frame_name: row}"""
    out = defaultdict(dict)
    if not path.exists():
        return out
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        out[r.get("clip_id") or r.get("clip")][Path(str(r.get("frame", ""))).name] = r
    return out


def reader_tilts(rows: dict, mount_deg: float = 0.0) -> dict:
    from src.perception.frames import camera_to_body_pitched
    R_cb = camera_to_body_pitched(np.deg2rad(mount_deg))
    out = {}
    for clip, frames in rows.items():
        if not frames:
            continue
        names = sorted(frames)
        up = np.array([frames[n]["gravity_up_cam"] for n in names], float)
        out[clip] = {"frames": names,
                     "tilt": np.degrees(np.arccos(np.clip((up @ R_cb.T)[:, 2], -1, 1))),
                     "roll": roll_of(up)}
    return out


def pf_up_from_head(roll_deg, pitch_deg):
    """World-up in camera coordinates from Perspective Fields' parameter head.

    Their extrinsics are ``rotz(roll) rotx(elevation)`` in a y-up camera frame; the flip to
    the project's y-down camera frame is (x, -y, z).  Validated on the level probe: the
    recovered roll equals ``-pf_roll`` exactly and the level up vector is (0, -1, 0).
    """
    r, p = np.deg2rad(roll_deg), np.deg2rad(pitch_deg)
    R = np.array([[np.cos(r), -np.sin(r), 0], [np.sin(r), np.cos(r), 0], [0, 0, 1]]) @ \
        np.array([[1, 0, 0], [0, np.cos(p), -np.sin(p)], [0, np.sin(p), np.cos(p)]])
    up = R @ np.array([0.0, 1.0, 0.0])
    return up * np.array([1.0, -1.0, 1.0])


def pf_tilts(rows: dict) -> dict:
    out = {}
    for clip, frames in rows.items():
        names = sorted(frames)
        heads = [(n, frames[n].get("pf_roll_deg"), frames[n].get("pf_pitch_deg"))
                 for n in names]
        heads = [h for h in heads if h[1] is not None and h[2] is not None]
        if not heads:
            continue
        up = np.array([pf_up_from_head(r, p) for _, r, p in heads], float)
        out[clip] = {"frames": [h[0] for h in heads], "tilt": tilt_of(up),
                     "roll": roll_of(up)}
    return out


def compare(reader: dict, pf: dict, label: str) -> list[dict]:
    """Per-clip reader-vs-PF depicted tilt on the frames both cover."""
    out = []
    for clip in sorted(set(reader) & set(pf)):
        rn = {n: i for i, n in enumerate(reader[clip]["frames"])}
        pn = {n: i for i, n in enumerate(pf[clip]["frames"])}
        common = sorted(set(rn) & set(pn))
        if not common:
            continue
        rt = np.array([reader[clip]["tilt"][rn[n]] for n in common])
        pt = np.array([pf[clip]["tilt"][pn[n]] for n in common])
        out.append({"label": label, "clip": clip, "n": len(common),
                    "reader_tilt_median_deg": float(np.median(rt)),
                    "pf_tilt_median_deg": float(np.median(pt)),
                    "delta_deg": float(np.median(pt) - np.median(rt)),
                    "abs_delta_median_deg": float(np.median(np.abs(pt - rt)))})
    return out


def main() -> int:
    pf = load_pf()
    man = list(csv.DictReader((ROOT / "results/h03/frames_manifest.csv").open()))
    out = {}

    # ---- 1. reconstruction validation on the controls and the whole sample
    val = []
    for m in man:
        key = ("h03sample", "sample", f"{m['source']}/{m['clip_id']}")
        r = pf.get(key, {}).get(f"{int(m['frame_index']):03d}.png")
        if r is None or r.get("up_cam") is None or not m["up_cam_x"]:
            continue
        truth = np.array([float(m["up_cam_x"]), float(m["up_cam_y"]), float(m["up_cam_z"])])
        est = np.array(r["up_cam"])
        val.append({"source": m["source"], "clip": m["clip_id"], "i": int(m["frame_index"]),
                    "roll_err": float(abs(wrap90(roll_of(est)[()] - roll_of(truth)[()]))),
                    "tilt_err": float(abs(tilt_of(est)[()] - tilt_of(truth)[()])),
                    "field_roll": float(roll_of(est)[()]),
                    "pf_roll": float(r["pf_roll_deg"]),
                    "up_roll": float(r.get("up_roll_deg", np.nan))})
    by = defaultdict(list)
    for v in val:
        by[v["source"]].append(v)
    out["reconstruction_validation"] = {
        s: {"n": len(v),
            "roll_err_median": float(np.median([x["roll_err"] for x in v])),
            "tilt_err_median": float(np.median([x["tilt_err"] for x in v])),
            "tilt_err_p90": float(np.percentile([x["tilt_err"] for x in v], 90)),
            "field_vs_head_roll_median": float(np.median(
                [abs(wrap90(x["field_roll"] + x["pf_roll"])) for x in v]))}
        for s, v in by.items()}

    # ---- 2. headline recomputations
    r_ling_nf4 = reader_tilts(load_reader(H1 / "results/h01_lingbot_depicted/depicted.jsonl"))
    r_ling_bf16 = reader_tilts(load_reader(H2 / "results/h02_lingbot_full_depicted/depicted.jsonl"))
    r_sw_probes = reader_tilts(load_reader(H2 / "results/h02_solarwm_probes_depicted/depicted.jsonl"))
    r_sw_grid = reader_tilts(load_reader(H2 / "results/h02_solarwm_grid_depicted/depicted.jsonl"))
    r_anwm = reader_tilts(load_reader(H1 / "results/h01_seed_depicted/depicted.jsonl"))

    p_ling_nf4 = pf_tilts({c: v for (g, b, c), v in pf.items() if g == "lingbot_nf4"})
    p_ling_bf16 = pf_tilts({c: v for (g, b, c), v in pf.items() if g == "lingbot_bf16"})
    p_sw_probes = pf_tilts({c: v for (g, b, c), v in pf.items()
                            if g == "solarwm" and b == "probes"})
    p_sw_grid = pf_tilts({c: v for (g, b, c), v in pf.items()
                          if g == "solarwm" and b == "grid_flatness"})
    p_sw_seeds = pf_tilts({c: v for (g, b, c), v in pf.items()
                           if g == "solarwm" and b == "seeds"})
    p_anwm = pf_tilts({c: v for (g, b, c), v in pf.items() if g == "anwm_seeds"})

    out["lingbot"] = (compare(r_ling_nf4, p_ling_nf4, "lingbot_nf4")
                      + compare(r_ling_bf16, p_ling_bf16, "lingbot_bf16"))
    out["solarwm_probes"] = compare(r_sw_probes, p_sw_probes, "solarwm_probes")
    out["solarwm_grid"] = compare(r_sw_grid, p_sw_grid, "solarwm_grid")
    out["solarwm_seeds"] = compare(r_sw_grid, p_sw_seeds, "solarwm_seeds")
    out["anwm"] = compare(r_anwm, p_anwm, "anwm_seeds")

    def agg(items, key_family=None):
        if not items:
            return {}
        return {"n_clips": len(items),
                "reader_tilt_median_deg": float(np.median([i["reader_tilt_median_deg"] for i in items])),
                "pf_tilt_median_deg": float(np.median([i["pf_tilt_median_deg"] for i in items])),
                "delta_deg_median": float(np.median([i["delta_deg"] for i in items])),
                "abs_delta_median_deg": float(np.median([i["abs_delta_median_deg"] for i in items]))}

    out["summary"] = {k: agg(v) for k, v in out.items() if isinstance(v, list)}
    # SolarWM grid by arm
    arms = defaultdict(list)
    for i in out["solarwm_grid"]:
        arm = i["clip"].split("_")[-1]
        arms[arm].append(i)
    out["solarwm_grid_by_arm"] = {a: agg(v) for a, v in arms.items()}
    Path(ROOT / "results/h03_recompute").mkdir(parents=True, exist_ok=True)
    (ROOT / "results/h03_recompute/compare.json").write_text(json.dumps(out, indent=1,
                                                                       default=float))
    print(json.dumps(out["reconstruction_validation"], indent=1))
    print(json.dumps(out["summary"], indent=1))
    print(json.dumps(out["solarwm_grid_by_arm"], indent=1))
    for k in ("lingbot",):
        for i in out[k]:
            print(f"{i['label']:10s} {i['clip'][:44]:44s} reader {i['reader_tilt_median_deg']:5.2f} "
                  f"pf {i['pf_tilt_median_deg']:5.2f} d {i['delta_deg']:+5.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
