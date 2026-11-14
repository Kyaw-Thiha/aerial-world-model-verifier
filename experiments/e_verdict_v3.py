"""Track E v3 scoring: Track D's R2 band, Track D's coherence tier, and the median residual.

Everything is imported from a read-only checkout of `track-d-verdict-stat` at the commit named in
`configs/e_prereg_v3.yaml`; the thresholds are frozen and are never refit here.

Reported, for every condition and every rig:

* (b1) flag rate under the frozen R2 axis-gain band (Mahalanobis distance > 2.1922);
* (b2) flag rate under the frozen coherence tier (correlation or sign);
* (a)  rejection rate under the frozen 14.73 deg median-axis-residual threshold;
* abstention, broken down by Track D's own reason order;
* the v3 edit-validity gate: null edit vs real, per-axis gain within 5 %, depicted lean within
  0.5 deg median.

Intervals: Wilson at frame level, plus a **sequence-clustered** bootstrap over the forward
sequences (the 9 of them), which is what the pre-registration makes the governing interval.

    python experiments/e_verdict_v3.py --clips data/clips_v2_w0.37 --tag w037
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from src.depicted.clips import up_body_from_lean
from src.depicted.metrics import angle_deg
from src.depicted.stat_a2 import FROZEN_THRESHOLD_DEG, score_clip_upb, depicted_axis_from_up_body

CONDITIONS = ("real", "null_edit", "yaw_control", "tilt_removal", "scale_0.5", "scale_1.5",
              "mirror_roll")
FAMILIES = CONDITIONS[1:]
FORWARD_RIGS = ("indoor_forward", "outdoor_forward")
REASONS = ("min_frames", "usable_fraction", "signal_frames", "excitation", "conditioning",
           "ci_half_width", "confidence_gate")


def yaml_load(path: Path) -> dict:
    import yaml

    return yaml.safe_load(Path(path).read_text())


def add_worktree(branch: str, path: Path, commit: str | None = None) -> Path:
    """Detached, read-only checkout of Track D's branch at the pinned commit."""
    if not path.exists():
        ref = commit or branch
        r = subprocess.run(["git", "worktree", "add", "--detach", str(path), ref],
                           capture_output=True, text=True)
        if r.returncode != 0:
            raise SystemExit(r.stderr)
    head = subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"],
                          capture_output=True, text=True).stdout.strip()
    if commit and not head.startswith(commit[:7]):
        subprocess.run(["git", "-C", str(path), "checkout", "--detach", commit],
                       check=True, capture_output=True)
    return Path(path)


def stage_clips(clips: Path, into: Path) -> Path:
    """One directory per clip in the layout Track D's loader expects."""
    if into.exists():
        import shutil

        shutil.rmtree(into)
    into.mkdir(parents=True)
    for npz in sorted(clips.glob("*/*.npz")):
        meta = npz.with_suffix(".meta.json")
        if not meta.exists():
            continue
        d = into / npz.stem
        d.mkdir()
        (d / "verdict_clip.npz").write_bytes(npz.read_bytes())
        (d / "meta.json").write_text(meta.read_text())
    return into


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    s = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return float((c - s) / d), float((c + s) / d)


def rig_of(seq: str) -> str:
    for p in ("indoor_forward", "outdoor_forward", "indoor_45", "outdoor_45"):
        if seq.startswith(p):
            return p
    return "unknown"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", default="data/clips_v2_w0.37")
    ap.add_argument("--tag", default="w037")
    ap.add_argument("--d-worktree", default="/tmp/awmv-d-ro")
    ap.add_argument("--branch", default="track-d-verdict-stat")
    ap.add_argument("--d-commit", default="851096c",
                    help="commit pinned by configs/e_prereg_v3.yaml")
    ap.add_argument("--out", default="results/e_verdict_v3")
    ap.add_argument("--n-boot", type=int, default=2000)
    args = ap.parse_args()

    d_wt = add_worktree(args.branch, Path(args.d_worktree), args.d_commit)
    gain_cfg = d_wt / "configs/gain_frozen.yaml"
    coh_cfg = d_wt / "configs/coherence_frozen.yaml"
    scorer = d_wt / "scripts/score_clip.py"
    meta_cfg = yaml_load(gain_cfg)
    coh_meta = yaml_load(coh_cfg)
    tau_c = coh_meta["thresholds"]["tau_correlation"]
    tau_s = coh_meta["thresholds"]["tau_sign"]
    tau_axis = meta_cfg["bands"]["axis"]["threshold"]
    rule = meta_cfg.get("rule", "R2_axis_gains")

    staged = stage_clips(Path(args.clips), Path("/tmp/e_clips_v3"))
    rows = []
    for d in sorted(staged.iterdir()):
        # their CLI, unmodified: the same command a user would type
        r = subprocess.run([sys.executable, str(scorer), str(d), "--config", str(gain_cfg),
                            "--coherence-config", str(coh_cfg), "--json"],
                           capture_output=True, text=True, cwd=str(d_wt),
                           env={"PYTHONPATH": str(d_wt), "PATH": "/usr/bin:/bin"})
        if r.returncode != 0:
            raise SystemExit(f"score_clip failed for {d.name}: {r.stderr[-800:]}")
        out = json.loads(r.stdout)
        meta = json.loads((d / "meta.json").read_text())
        seq = meta.get("sequence", "")
        clip_id = out["clip_id"]
        coh = out.get("coherence") or {}
        flagged_coh = None
        if out.get("testable") and coh:
            flagged_coh = bool(coh.get("flagged_correlation")) or bool(coh.get("flagged_sign"))
        # (a) median axis residual on the same clip, frozen threshold
        npz = np.load(d / "verdict_clip.npz")
        lean_imp = np.asarray(npz["lean_req"], dtype=np.float64)
        imp_axis = depicted_axis_from_up_body(up_body_from_lean(lean_imp), npz["yaw"])
        tilt_req = np.degrees(np.arccos(np.clip(imp_axis[:, 2], -1.0, 1.0)))
        a = score_clip_upb(up_body_from_lean(npz["depicted_lean"]), imp_axis, npz["yaw"],
                           npz["t"], np.eye(3), tilt_req, np.ones(len(npz["t"]), bool),
                           lag_grid_ms=np.array([0.0]))
        rows.append({
            "seq": seq, "clip": clip_id, "cond": out["condition"], "rig": rig_of(seq),
            "testable": bool(out["testable"]), "reasons": list(out["abstention_reasons"]),
            "flagged_r2": bool(out["flagged"]) if out["flagged"] is not None else None,
            "flagged_coh": flagged_coh,
            "g1": out["gain_g1"], "gx": out["axis_gains"]["gx"], "gy": out["axis_gains"]["gy"],
            "r_x": coh.get("r_x"), "r_y": coh.get("r_y"),
            "s_x": coh.get("s_x"), "s_y": coh.get("s_y"),
            "score_a": a.get("score"), "a_testable": bool(a.get("testable")),
            "gate_kept": out.get("gate_kept"),
            "tilt_req_deg": float(np.nanmedian(tilt_req)),
        })

    # ---- v3 edit-validity gate: null edit vs real
    # key on the real condition only: keying every row overwrote it with whichever condition
    # came last in file order, which made the validity gate compare the null edit to an edit
    by_key = {(r["seq"], r["clip"][: -(len(r["cond"]) + 1)]): r
              for r in rows if r["cond"] == "real"}
    gate_rows = []
    for r in rows:
        if r["cond"] != "null_edit":
            continue
        key = (r["seq"], r["clip"][: -(len(r["cond"]) + 1)])
        base = by_key.get(key)
        if base is None:
            continue
        dgx = abs(r["gx"] - base["gx"]) / max(abs(base["gx"]), 1e-9)
        dgy = abs(r["gy"] - base["gy"]) / max(abs(base["gy"]), 1e-9)
        gate_rows.append({"seq": r["seq"], "clip": key[1], "dgx_rel": float(dgx),
                          "dgy_rel": float(dgy), "pass_gain": bool(dgx <= 0.05 and dgy <= 0.05)})
    # depicted-lean angle needs the arrays, not the fit
    lean_angles = []
    for d in sorted(staged.iterdir()):
        meta = json.loads((d / "meta.json").read_text())
        if meta["condition"] != "null_edit":
            continue
        stem = meta["clip_id"][: -(len(meta["condition"]) + 1)]
        other = d.parent / f"{stem}_real"
        if not (other / "verdict_clip.npz").exists():
            continue
        a1 = np.load(d / "verdict_clip.npz")["depicted_lean"]
        a2 = np.load(other / "verdict_clip.npz")["depicted_lean"]
        ang = angle_deg(_lean_to_unit(a1), _lean_to_unit(a2))
        lean_angles.append({"seq": meta["sequence"], "clip": stem,
                            "lean_angle_median_deg": float(np.nanmedian(ang)),
                            "pass_lean": bool(np.nanmedian(ang) <= 0.5)})
    gate_rows = _merge_gate(gate_rows, lean_angles)

    # ---- rates
    def rate(sub, field):
        sel = [r for r in sub if r["testable"]]
        n = len(sel)
        k = sum(1 for r in sel if r[field])
        lo, hi = wilson(k, n)
        return {"n": n, "k": k, "rate": (k / n) if n else None, "wilson_lo": lo,
                "wilson_hi": hi}

    def block(sub):
        return {"n_clips": len(sub),
                "n_sequences": len({r["seq"] for r in sub}),
                "r2": {c: rate([r for r in sub if r["cond"] == c], "flagged_r2")
                       for c in CONDITIONS},
                "coherence": {c: rate([r for r in sub if r["cond"] == c], "flagged_coh")
                              for c in CONDITIONS},
                "median_residual": {
                    c: _a_rate([r for r in sub if r["cond"] == c]) for c in CONDITIONS},
                "abstention": _abstention([r for r in sub if r["cond"] == "real"]),
                "gate_dropped_real": sum(1 for r in sub
                                         if r["cond"] == "real" and r["gate_kept"] is False)}

    fwd = [r for r in rows if r["rig"] in FORWARD_RIGS]
    per_rig = {rig: block([r for r in rows if r["rig"] == rig]) for rig in FORWARD_RIGS}
    summary = {
        "tag": args.tag, "clips": str(args.clips), "n_rows": len(rows),
        "forward": block(fwd), "per_rig": per_rig,
        "clustered": _clustered(fwd, args.n_boot),
        "validity_gate": _gate_summary(gate_rows),
        "validity_rows": gate_rows,
        "rows": rows,
        "provenance": {"branch": args.branch, "gain_rule": rule,
                       "thresholds": {"tau_axis": tau_axis,
                                      "tau_corr": tau_c, "tau_sign": tau_s,
                                      "a_threshold_deg": FROZEN_THRESHOLD_DEG}},
    }
    out = Path(args.out) / args.tag
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps({"forward": {k: summary["forward"][k] for k in ("n_clips", "n_sequences")},
                      "validity": summary["validity_gate"]}, indent=1))
    for kind in ("r2", "coherence", "median_residual"):
        line = "  ".join(f"{c}={_fmt(summary['forward'][kind][c])}" for c in CONDITIONS)
        print(f"forward {kind:16s} {line}")
    return 0


def _lean_to_unit(lean: np.ndarray) -> np.ndarray:
    up = up_body_from_lean(np.asarray(lean))
    return up / np.maximum(np.linalg.norm(up, axis=1, keepdims=True), 1e-12)


def _fmt(r: dict) -> str:
    if not r or r.get("rate") is None:
        return "n/a"
    return f"{r['rate']:.3f}({r['n']})"


def _a_rate(sub):
    sel = [r for r in sub if r["a_testable"] and r["score_a"] is not None]
    n = len(sel)
    k = sum(1 for r in sel if r["score_a"] > FROZEN_THRESHOLD_DEG)
    lo, hi = wilson(k, n)
    return {"n": n, "k": k, "rate": (k / n) if n else None, "wilson_lo": lo, "wilson_hi": hi}


def _abstention(sub):
    n = len(sub)
    out = {"n": n, "abstained": sum(1 for r in sub if not r["testable"]), "by_reason": {}}
    for r in sub:
        for reason in r["reasons"]:
            key = next((k for k in REASONS if reason.startswith(k)), reason.split()[0])
            out["by_reason"][key] = out["by_reason"].get(key, 0) + 1
    out["rate"] = out["abstained"] / n if n else None
    return out


def _merge_gate(gain_rows, lean_rows):
    lean_by = {(r["seq"], r["clip"]): r for r in lean_rows}
    out = []
    for r in gain_rows:
        extra = lean_by.get((r["seq"], r["clip"]), {})
        out.append({**r, **{k: v for k, v in extra.items() if k not in ("seq", "clip")}})
    return out


def _gate_summary(rows: list[dict]) -> dict:
    if not rows:
        return {"n_pairs": 0}
    seqs: dict[str, list] = {}
    for r in rows:
        seqs.setdefault(r["seq"], []).append(r)
    per_seq = {}
    for seq, v in sorted(seqs.items()):
        passes = [r.get("pass_gain", False) and r.get("pass_lean", False) for r in v]
        per_seq[seq] = {
            "n_pairs": len(v),
            "pass_rate": float(np.mean(passes)),
            "max_dgx_rel": float(np.max([r["dgx_rel"] for r in v])),
            "max_dgy_rel": float(np.max([r["dgy_rel"] for r in v])),
            "lean_median_deg": float(np.median([r.get("lean_angle_median_deg", np.nan)
                                                for r in v])),
        }
    return {"n_pairs": len(rows),
            "pass_all": bool(all(v["pass_rate"] == 1.0 for v in per_seq.values())),
            "per_sequence": per_seq}


def _clustered(rows: list[dict], n_boot: int) -> dict:
    """Sequence-clustered bootstrap over the forward rigs; thresholds are frozen, no refit."""
    seqs = sorted({r["seq"] for r in rows})
    rng = np.random.default_rng(0)
    index = {s: [r for r in rows if r["seq"] == s] for s in seqs}
    out = {}
    for cond in CONDITIONS:
        vals = {"r2": [], "coherence": [], "a": []}
        for _ in range(n_boot):
            pick = rng.choice(len(seqs), size=len(seqs), replace=True)
            sub = [r for i in pick for r in index[seqs[i]] if r["cond"] == cond
                   and r["testable"]]
            if not sub:
                continue
            vals["r2"].append(np.mean([bool(r["flagged_r2"]) for r in sub]))
            vals["coherence"].append(np.mean([bool(r["flagged_coh"]) for r in sub]))
            a = [r for r in sub if r["a_testable"] and r["score_a"] is not None]
            if a:
                vals["a"].append(np.mean([r["score_a"] > FROZEN_THRESHOLD_DEG for r in a]))
        out[cond] = {k: ({"lo": float(np.percentile(v, 2.5)),
                          "hi": float(np.percentile(v, 97.5)),
                          "n_boot": len(v)} if len(v) >= 20 else None)
                     for k, v in vals.items()}
    out["_note"] = ("bootstrap resamples the forward sequences with replacement; thresholds are "
                    "the frozen ones, so this interval covers sampling of flights, not threshold "
                    "estimation")
    return out


if __name__ == "__main__":
    raise SystemExit(main())
