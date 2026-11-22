"""Freeze the clip table for the simulated labelled set (Track I, Step 2).

The pre-registration (`configs/sim_prereg.yaml`) fixes the *rules*; this script resolves them
into the explicit 300-row table that ships as `configs/sim_splits_frozen.json`. Freezing the
resolved table rather than "the seed" is deliberate: a seed plus code is only reproducible
until the code changes, while a table of (scene, maneuver, defect, magnitude, interval seed)
is a statement about the data that can be checked line by line.

Everything here is deterministic: one RNG, `default_rng(20261121)`, draws the parameters once,
and the output file's sha256 is what `docs/DATA.md` publishes. Re-running the script must
produce a byte-identical file; `--check` verifies that.

    python experiments/i02_freeze_splits.py            # write the table
    python experiments/i02_freeze_splits.py --check     # verify the committed table
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
PREREG = ROOT / "configs/sim_prereg.yaml"
OUT = ROOT / "configs/sim_splits_frozen.json"
SEED = 20261121
G = 9.81

# Clip counts per split. The totals are the pre-registration's; the split keeps every defect
# family and every magnitude band represented on both sides, because a held-out set that lacks
# a family cannot test it.
COUNTS = {
    "dev":     {"feasible": 80, "tilt_removed": 27, "mirrored_roll": 27,
                "tilt_scaled": 33, "thrust_limit": 13, "partial": 20},
    "heldout": {"feasible": 40, "tilt_removed": 13, "mirrored_roll": 13,
                "tilt_scaled": 17, "thrust_limit": 7, "partial": 10},
}
MANEUVERS = ["ellipse", "lemniscate", "slalom", "turn", "climb_turn", "accel_brake"]
BANDS = [(5, 10), (10, 20), (20, 35), (35, 50)]


def resolve_scenes(prereg: dict) -> dict:
    """(family, seed) -> split, from the rule in the pre-registration."""
    sc = prereg["scenes"]
    out = {}
    for split in ("dev", "heldout"):
        fams = sc["families"][split]
        for seed in sc["seeds"][split]:
            out[int(seed)] = {"split": split, "family": fams[int(seed) % len(fams)]}
    return out


def maneuver_params(maneuver: str, tilt_deg: float, rng: np.random.Generator,
                    radius_range: tuple[float, float] = (6.0, 18.0)) -> dict:
    """Parameters whose peak required tilt lands on `tilt_deg` (before drag).

    The closed forms are the ones the flight library uses, so the requested tilt is a design
    input rather than an outcome that has to be measured first:
      ellipse/turn    a = w^2 r = g tan(tilt)  -> v = sqrt(g tan(tilt) r)
      lemniscate      a = 4 w^2 a_scale at the waist -> w = sqrt(g tan(tilt) / (4 a_scale))
      slalom          a = amp k^2 v^2,  k = 2 pi / lambda
      climb_turn      same as turn, with a steady vertical rate added
      accel_brake     a = g tan(tilt) along the heading during the ramp
    """
    th = math.radians(tilt_deg)
    if maneuver in ("ellipse", "turn"):
        radius = float(rng.uniform(*radius_range))
        speed = float(math.sqrt(G * math.tan(th) * radius))
        arc = float(rng.uniform(120.0, 300.0))
        if maneuver == "ellipse":
            duration = 2 * math.pi * radius / max(speed, 1e-6)
            return {"radius": radius, "speed": min(speed, 18.0),
                    "duration": float(min(max(duration, 10.0), 30.0))}
        speed = min(speed, 18.0)
        # the pre-registration's duration window is met by the arc length, not by truncation
        duration = float(rng.uniform(12.0, 25.0))
        return {"speed": speed, "radius": radius,
                "arc_deg": float(math.degrees(speed / radius * duration))}
    if maneuver == "lemniscate":
        scale = float(rng.uniform(6.0, 12.0))
        w = math.sqrt(G * math.tan(th) / (4.0 * scale))
        speed = float(w * scale)
        return {"scale": scale, "speed": speed, "duration": 20.0}
    if maneuver == "slalom":
        amp = float(rng.uniform(1.5, 5.0))
        wavelength = float(rng.uniform(16.0, 40.0))
        k = 2 * math.pi / wavelength
        speed = float(min(math.sqrt(G * math.tan(th) / (amp * k * k)), 14.0))
        duration = float(min(max(3 * wavelength / speed, 10.0), 30.0))
        return {"speed": speed, "amplitude": amp, "wavelength": wavelength,
                "duration": duration, "height": 12.0}
    if maneuver == "climb_turn":
        radius = float(rng.uniform(*radius_range))
        speed = float(min(math.sqrt(G * math.tan(th) * radius), 14.0))
        duration = float(rng.uniform(12.0, 22.0))
        return {"speed": speed, "radius": radius, "vz": float(rng.uniform(0.5, 2.5)),
                "arc_deg": float(math.degrees(speed / radius * duration))}
    if maneuver == "accel_brake":
        # the smoothstep profile peaks at 1.5 (v_peak - v0) / ramp
        a = G * math.tan(th)
        ramp = float(rng.uniform(2.0, 3.0))
        v0 = float(rng.uniform(0.0, 3.0))
        v_peak = float(v0 + a * ramp / 1.5)
        return {"v0": v0, "v_peak": v_peak, "ramp_s": ramp,
                "hold_s": float(rng.uniform(6.0, 12.0))}
    if maneuver == "hover":
        return {"duration": float(rng.uniform(10.0, 12.0)), "height": 12.0}
    raise KeyError(maneuver)


def _maneuver_for(defect: str, k: int) -> str:
    """Round-robin maneuver choice per defect family, so every family is balanced.

    The pre-registration says every scene family carries every maneuver; this is what makes
    that true of the table rather than of the intention. Hover appears only among the feasible
    clips (10 % of them), and a thrust-limit path is built from the maneuvers that can express
    a large sustained acceleration.
    """
    if defect == "feasible":
        cycle = ["hover", "ellipse", "lemniscate", "slalom", "turn", "climb_turn",
                 "accel_brake", "ellipse", "turn", "slalom"]
        return cycle[k % len(cycle)]
    if defect == "thrust_limit":
        return ["ellipse", "turn", "climb_turn"][k % 3]
    return MANEUVERS[k % len(MANEUVERS)]


def build_table() -> dict:
    prereg = yaml.safe_load(PREREG.read_text())
    scenes = resolve_scenes(prereg)
    rng = np.random.default_rng(SEED)
    rows = []
    for split, counts in COUNTS.items():
        seeds = sorted(s for s, v in scenes.items() if v["split"] == split)
        i = 0
        for defect, n in counts.items():
            for j in range(n):
                seed = seeds[(i + j) % len(seeds)]
                band = BANDS[j % len(BANDS)]
                tilt = float(rng.uniform(band[0], band[1]))
                maneuver = _maneuver_for(defect, j)
                params = maneuver_params(maneuver, 0.0 if maneuver == "hover" else tilt, rng)
                row = {
                    "clip_id": f"i02_{split}_{i + j:03d}",
                    "split": split,
                    "family": scenes[seed]["family"],
                    "scene_seed": int(seed),
                    "maneuver": maneuver,
                    "params": params,
                    "peak_tilt_target_deg": 0.0 if maneuver == "hover" else round(tilt, 2),
                    "label": "feasible" if defect == "feasible" else "infeasible",
                    "defect_family": defect,
                    "violation": None if defect == "feasible" else defect,
                    "magnitude": None,
                    "interval_seed": int(rng.integers(0, 2**31 - 1)),
                    # The mount varies across the set (0/15/30/45 deg of uptilt, as TII flies
                    # 30-50) and is part of the frozen row, not a generator-side draw: the
                    # manifest has to say what camera made the clip.
                    "uptilt_deg": [0.0, 15.0, 30.0, 45.0][(i + j) % 4],
                    "mount_yaw_deg": 0.0,
                }
                if defect == "tilt_removed":
                    row["magnitude"] = 1.0            # fraction of the tilt removed
                elif defect == "mirrored_roll":
                    row["magnitude"] = -1.0          # roll gain (sign flip)
                elif defect == "tilt_scaled":
                    row["magnitude"] = 0.5 if (j % 2 == 0) else 1.5
                elif defect == "thrust_limit":
                    # a path whose peak required thrust exceeds thrust_to_weight_max = 2.0
                    # (60 deg). The radius is kept small so the required speed stays inside the
                    # library's own speed cap and the tilt is carried by the geometry.
                    row["params"] = maneuver_params(maneuver,
                                                    62.0 + float(rng.uniform(0.0, 6.0)), rng,
                                                    radius_range=(5.0, 11.0))
                    row["magnitude"] = 2.2
                elif defect == "partial":
                    base = ["tilt_removed", "mirrored_roll", "tilt_scaled"][j % 3]
                    row["violation"] = base
                    row["magnitude"] = (1.0 if base == "tilt_removed"
                                        else -1.0 if base == "mirrored_roll"
                                        else (0.5 if (j % 2 == 0) else 1.5))
                    row["fraction"] = [0.10, 0.25, 0.50][(j // 3) % 3]
                rows.append(row)
            i += n
    return {"seed": SEED, "date": "2026-11-21", "prereg_sha256": hashlib.sha256(
                PREREG.read_bytes()).hexdigest(),
            "scenes": {str(k): v for k, v in sorted(scenes.items())},
            "counts": COUNTS, "clips": rows}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    table = build_table()
    text = json.dumps(table, indent=1, sort_keys=False) + "\n"
    digest = hashlib.sha256(text.encode()).hexdigest()
    if args.check:
        if not OUT.is_file():
            print(f"MISSING {OUT}")
            return 1
        have = hashlib.sha256(OUT.read_text().encode()).hexdigest()
        print(f"{OUT.relative_to(ROOT)} sha256 {have}")
        print("MATCH" if have == digest else "DIFFERS from what the script produces now")
        return 0 if have == digest else 2
    OUT.write_text(text)
    n = len(table["clips"])
    print(f"wrote {OUT.relative_to(ROOT)}: {n} clips, sha256 {digest}")
    for split in COUNTS:
        sub = [c for c in table["clips"] if c["split"] == split]
        fams = sorted({c["family"] for c in sub})
        print(f"  {split}: {len(sub)} clips over families {fams}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
