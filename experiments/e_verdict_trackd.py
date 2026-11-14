"""Track E Task 5(b): Track D's frozen gain statistic, run on the UZH verdict clips.

Track D published `configs/gain_frozen.yaml` and `scripts/score_clip.py` (rule R1_scalar_g1, a
per-clip regression of depicted lean on required lean, with a per-clip 95 % band on the scalar
gain; TII-fitted band 0.528-0.948 from 57 real testable clips). Its branch is read-only here: a
detached worktree supplies `src/verdict`, and this script imports it rather than reimplementing
the statistic, so the number that runs is Track D's.

The output mirrors the (a) tables so the two statistics can be compared at *matched* false
reject: per condition, the fraction of testable clips the frozen band flags; the paired view;
and the abstention breakdown.

    python experiments/e_verdict_trackd.py --clips data/clips --d-worktree /tmp/awmv-d-ro
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

CONDITIONS = ("real", "null_edit", "tilt_removal", "scale_0.5", "scale_1.5", "mirror_roll",
              "yaw_control")


def add_worktree(branch: str, path: Path) -> Path:
    """Read-only detached worktree of Track D's branch (never edited)."""
    if not path.exists():
        r = subprocess.run(["git", "worktree", "add", "--detach", str(path), branch],
                           capture_output=True, text=True)
        if r.returncode != 0:
            raise SystemExit(f"cannot check out {branch}: {r.stderr.strip()}")
    return path


def clip_dirs(clips: Path, into: Path) -> list[Path]:
    """One directory per clip, in the layout Track D's loader expects."""
    into.mkdir(parents=True, exist_ok=True)
    out = []
    for npz in sorted(clips.glob("*/*.npz")):
        meta = npz.with_suffix(".meta.json")
        if not meta.exists():
            continue
        d = into / npz.stem
        d.mkdir(exist_ok=True)
        dst_npz, dst_meta = d / "verdict_clip.npz", d / "meta.json"
        if not dst_npz.exists():
            dst_npz.write_bytes(npz.read_bytes())
            dst_meta.write_text(meta.read_text())
        out.append(d)
    return out


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    s = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return float((c - s) / d), float((c + s) / d)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", default="data/clips")
    ap.add_argument("--d-worktree", default="/tmp/awmv-d-ro")
    ap.add_argument("--branch", default="track-d-verdict-stat")
    ap.add_argument("--out", default="results/e_verdict/trackd")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    d_wt = add_worktree(args.branch, Path(args.d_worktree))
    sys.path.insert(0, str(d_wt))
    import yaml

    from src.verdict import (Bands, FitOptions, JointBand, ScalarBand, assess, fit_gain,
                             flag, gate_kept_from_conf, load_clip)

    cfg = yaml.safe_load((d_wt / "configs" / "gain_frozen.yaml").read_text())
    opts = {k: v for k, v in cfg["options"].items() if k in FitOptions.__dataclass_fields__}
    if "lag_bounds_ms" in opts:
        opts["lag_bounds_ms"] = tuple(opts["lag_bounds_ms"])
    options = FitOptions(**opts)
    b = cfg["bands"]["scalar"]
    bands = Bands(scalar=ScalarBand(float(b["lo"]), float(b["hi"]), float(b["alpha"]),
                                    int(b["n"])))
    gate_cfg = None
    g = cfg.get("gate") or {}
    if isinstance(g, dict) and g.get("config"):
        p = d_wt / g["config"]
        gate_cfg = yaml.safe_load(p.read_text()) if p.exists() else None
    rule = cfg.get("rule", "R1_scalar_g1")

    dirs = clip_dirs(Path(args.clips), Path("/tmp/e_clips_schema"))
    if args.limit:
        dirs = dirs[: args.limit]
    rows = []
    for d in dirs:
        clip = load_clip(d)
        res = fit_gain(clip, options)
        kept = gate_kept_from_conf(clip.depicted_conf, gate_cfg) if gate_cfg else None
        ab = assess(res, cfg.get("abstention"), gate_kept=kept)
        fl = flag(res, rule, bands) if (ab.testable and bands.scalar) else None
        rows.append({"seq": clip.meta.get("sequence", ""), "clip": clip.clip_id,
                     "cond": clip.condition,
                     "rig": clip.meta.get("rig", ""),
                     "gain": res.gain, "gain_ci": list(res.gain_ci),
                     "required_tilt_deg": res.required_tilt_deg,
                     "depicted_tilt_deg": res.depicted_tilt_deg,
                     "r2": res.r2, "lag_ms": res.lag_ms, "drag": res.drag,
                     "excitation_deg": res.excitation_deg,
                     "testable": bool(ab.testable), "reasons": ab.reasons,
                     "gate_kept": kept, "flagged": fl})

    def rate(cond, field="flagged"):
        sel = [r for r in rows if r["cond"] == cond and r["testable"]]
        n = len(sel)
        k = sum(1 for r in sel if r[field] is True)
        lo, hi = wilson(k, n)
        return {"n": n, "k": k, "rate": (k / n) if n else None, "wilson_lo": lo,
                "wilson_hi": hi,
                "median_gain": (float(np.median([r["gain"] for r in sel
                                                 if r["gain"] is not None]))
                                if n else None)}

    # paired: the edit is detected when the real clip of the same pair was *not* flagged
    def stem(clip_id, cond):
        return clip_id[: -(len(cond) + 1)]

    real = {stem(r["clip"], r["cond"]): r for r in rows if r["cond"] == "real"}
    paired = {}
    for cond in CONDITIONS[1:]:
        n = k = 0
        for r in rows:
            if r["cond"] != cond or not r["testable"]:
                continue
            base = real.get(stem(r["clip"], r["cond"]))
            if base is None or not base["testable"] or base["flagged"] is True:
                continue
            n += 1
            k += int(r["flagged"] is True)
        lo, hi = wilson(k, n)
        paired[cond] = {"n": n, "k": k, "rate": (k / n) if n else None,
                        "wilson_lo": lo, "wilson_hi": hi,
                        "population": "real clip not flagged by the same frozen band"}

    summary = {
        "source": {"branch": args.branch, "config": "configs/gain_frozen.yaml",
                   "rule": rule,
                   "band_scalar": {"lo": b["lo"], "hi": b["hi"], "n": b["n"]},
                   "fitted_on": "TII, 57 real testable clips (Track D)"},
        "n_clips": len(rows),
        "by_condition": {c: rate(c) for c in CONDITIONS},
        "paired": paired,
        "abstention": {c: (sum(1 for r in rows if r["cond"] == c and not r["testable"])
                           / max(sum(1 for r in rows if r["cond"] == c), 1))
                       for c in CONDITIONS},
        "points_a": "statistic (a) used a median-axis residual at 14.73 deg; compare at "
                    "matched false reject on the same clips",
        "rows": rows,
    }
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps({k: summary[k] for k in ("by_condition", "paired", "abstention")},
                     indent=1)[:3000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
